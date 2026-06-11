from __future__ import annotations

import argparse
import gc
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cli.logit_feature_score_suite.run_csqa_logit_feature_score_pipeline import (  # noqa: E402
    DEFAULT_BACKTRACK_SCALES,
    DEFAULT_FEATURE_NAMES,
    DEFAULT_MAX_DELTA_OVER_HIDDEN,
    LETTERS,
    all_layer_numbers,
    augment_region_models_with_derivative,
    build_full_cap_delta,
    compute_score_ascent_unit_delta,
    evaluate_candidate_delta,
    interpolate_score_state,
    now_id,
    parse_feature_names,
    parse_positive_scale_list,
    select_score_ascent_delta,
    slugify_model_id,
)
from src.cli.run_csqa_adaptive_contrastive_pipeline import (  # noqa: E402
    EXTRACT_BATCH_SIZE,
    choose_model_dtype_and_device_map,
    extract_split_cache,
    get_input_device,
    prepare_readout_context,
)
from src.cli.run_csqa_logit_feature_response_curve_pipeline import (  # noqa: E402
    GOOD_REGION_LOG_RATIO_THRESHOLD,
    GRID_POINTS,
    KDE_BANDWIDTH_MULTIPLIER,
    LOG_RATIO_SMOOTHING_SIGMA_BANDWIDTHS,
    SUPPORT_LOWER_QUANTILE,
    SUPPORT_UPPER_QUANTILE,
    build_distribution_models,
    build_feature_table,
    build_separation_summary,
    run_single_intervention_forward,
    select_top_k_layers_by_feature,
)
from src.cli.run_csqa_logit_feature_steering_pipeline import resolve_hf_token  # noqa: E402
from src.csqa.common import get_decoder_layers  # noqa: E402
from src.data.load_csqa import load_csqa  # noqa: E402


INTERVENTION_BATCH_SIZE = 2
CONTROL_INTERVENTION_TYPES = ("ascent", "descent", "random_same_norm")
RANDOM_ORTHO_EPS = 1e-8


def resolve_out_dir(out_dir: str | None, model_id: str) -> Path:
    if out_dir is None:
        run_name = f"{now_id()}_{slugify_model_id(model_id)}_csqa_logit_feature_score_control_pipeline"
        return REPO_ROOT / "data" / "generated" / "logit_feature_score_control_pipeline" / run_name
    path = Path(out_dir)
    return path if path.is_absolute() else (REPO_ROOT / path)


def build_random_same_norm_delta(
    reference_delta: torch.Tensor,
    *,
    intervention_mask: np.ndarray,
) -> torch.Tensor:
    reference_delta = reference_delta.detach()
    ref_float = reference_delta.float()
    ref_norm = ref_float.norm(dim=-1, keepdim=True)

    rand = torch.randn_like(reference_delta)
    rand_float = rand.float()
    ref_norm_sq = (ref_float * ref_float).sum(dim=-1, keepdim=True).clamp_min(1e-12)
    projection_coeff = (rand_float * ref_float).sum(dim=-1, keepdim=True) / ref_norm_sq
    ortho = rand - projection_coeff.to(rand.dtype) * reference_delta
    ortho_norm = ortho.float().norm(dim=-1, keepdim=True)

    fallback = rand / rand_float.norm(dim=-1, keepdim=True).clamp_min(1e-12).to(rand.dtype)
    ortho_unit = ortho / ortho_norm.clamp_min(1e-12).to(ortho.dtype)
    use_fallback = ortho_norm <= RANDOM_ORTHO_EPS
    random_unit = torch.where(use_fallback.expand_as(ortho_unit), fallback, ortho_unit)

    random_delta = random_unit * ref_norm.to(random_unit.dtype)
    mask = torch.as_tensor(intervention_mask, device=random_delta.device, dtype=random_delta.dtype).unsqueeze(-1)
    return (random_delta * mask).detach()


def compute_delta_norm_stats(
    token_hidden: torch.Tensor,
    *,
    delta: torch.Tensor,
) -> dict[str, np.ndarray]:
    delta_l2 = delta.detach().float().norm(dim=-1)
    token_hidden_l2 = token_hidden.detach().float().norm(dim=-1)
    delta_over_hidden = delta_l2 / token_hidden_l2.clamp_min(1e-12)
    return {
        "delta_l2_norm": delta_l2.detach().cpu().numpy().astype(np.float32),
        "delta_over_token_hidden_l2": delta_over_hidden.detach().cpu().numpy().astype(np.float32),
        "token_hidden_l2_norm": token_hidden_l2.detach().cpu().numpy().astype(np.float32),
    }


def run_score_control_policy(
    *,
    eval_rows: pd.DataFrame,
    eval_cache: dict[str, object],
    eval_feature_df: pd.DataFrame,
    selected_layers_df: pd.DataFrame,
    region_models: dict[tuple[str, int], dict[str, object]],
    backtrack_scales: list[float],
    max_delta_over_hidden: float,
    tok: AutoTokenizer,
    model: AutoModelForCausalLM,
    input_device: torch.device,
    decoder_layers,
    answer_id_tensor_cpu: torch.Tensor,
    answer_id_tensor_lm_head: torch.Tensor,
    lm_head_weight: torch.Tensor,
    maybe_apply_final_norm,
    vocab_size: int,
    max_seq_len: int,
    eval_split_name: str,
) -> pd.DataFrame:
    eval_lookup = eval_feature_df.set_index(["example_id", "feature_name", "layer_number"])
    clean_choice_logits = eval_cache["clean_choice_logits"]
    clean_best_non_choice_logit = eval_cache["clean_best_non_choice_logit"]
    clean_best_non_choice_token_id = eval_cache["clean_best_non_choice_token_id"]
    example_ids = [row["example_id"] for row in eval_cache["example_rows"]]
    example_id_to_index = {example_id: idx for idx, example_id in enumerate(example_ids)}

    selected_pairs = [
        (str(row.feature_name), int(row.layer_number))
        for row in selected_layers_df.itertuples(index=False)
    ]
    num_eval_batches = math.ceil(len(eval_rows) / INTERVENTION_BATCH_SIZE)
    total_steps = len(selected_pairs) * num_eval_batches
    rows: list[dict[str, object]] = []

    with tqdm(total=total_steps, desc=f"{eval_split_name} score-control sweep") as pbar:
        for feature_name, layer_number in selected_pairs:
            steering_module = decoder_layers[layer_number - 1]
            region_model = region_models[(feature_name, layer_number)]

            current_feature_all = np.array(
                [
                    float(eval_lookup.loc[(example_id, feature_name, layer_number), "feature_value"])
                    for example_id in example_ids
                ],
                dtype=np.float32,
            )
            current_state_all = interpolate_score_state(current_feature_all, region_model=region_model)

            for start in range(0, len(eval_rows), INTERVENTION_BATCH_SIZE):
                batch_df = eval_rows.iloc[start:start + INTERVENTION_BATCH_SIZE].reset_index(drop=True)
                batch_indices = [example_id_to_index[example_id] for example_id in batch_df["example_id"].tolist()]

                batch_current_feature = current_feature_all[batch_indices].astype(np.float32)
                batch_current_score = current_state_all["score_value"][batch_indices].astype(np.float32)
                batch_current_score_derivative = current_state_all["score_derivative"][batch_indices].astype(np.float32)
                batch_current_supported = current_state_all["supported"][batch_indices].astype(bool)
                batch_current_region = current_state_all["region_label"][batch_indices]
                batch_eligible = batch_current_supported & np.isfinite(batch_current_score_derivative) & (
                    np.abs(batch_current_score_derivative) > 1e-8
                )

                batch_cpu = prepare_batch(tok, batch_df["text"].tolist(), max_seq_len)
                decision_pos = batch_cpu.pop("decision_pos")
                batch = {k: v.to(input_device) for k, v in batch_cpu.items()}
                decision_pos = decision_pos.to(input_device)
                true_choice_idx = torch.tensor(
                    [LETTERS.index(str(x)) for x in batch_df["answerKey"].tolist()],
                    dtype=torch.long,
                    device=input_device,
                )
                token_hidden = eval_cache["hidden"][batch_indices, layer_number - 1, :].to(input_device)

                ascent_basis = compute_score_ascent_unit_delta(
                    token_hidden,
                    feature_name=feature_name,
                    layer_index_0based=layer_number - 1,
                    maybe_apply_final_norm=maybe_apply_final_norm,
                    lm_head_weight=lm_head_weight,
                    answer_id_tensor_lm_head=answer_id_tensor_lm_head.to(input_device),
                    vocab_size=vocab_size,
                    current_score_derivative=batch_current_score_derivative,
                    intervention_mask=batch_eligible,
                )
                effective_eligible = batch_eligible & (ascent_basis["score_grad_l2_norm"] > 0.0)
                raw_delta_stats = build_full_cap_delta(
                    token_hidden,
                    unit_delta=ascent_basis["unit_delta"],
                    intervention_mask=effective_eligible,
                    max_delta_over_hidden=float(max_delta_over_hidden),
                )
                selected_delta = select_score_ascent_delta(
                    token_hidden,
                    delta_raw=raw_delta_stats["delta_raw"],
                    current_score_value=batch_current_score,
                    intervention_mask=effective_eligible,
                    feature_name=feature_name,
                    layer_index_0based=layer_number - 1,
                    maybe_apply_final_norm=maybe_apply_final_norm,
                    lm_head_weight=lm_head_weight,
                    answer_id_tensor_lm_head=answer_id_tensor_lm_head.to(input_device),
                    vocab_size=vocab_size,
                    region_model=region_model,
                    backtrack_scales=backtrack_scales,
                )

                accepted_mask = selected_delta["accepted_intervention"].astype(bool)
                if bool(accepted_mask.any()):
                    ascent_delta = selected_delta["delta"]
                    descent_delta = (-ascent_delta).detach()
                    random_delta = build_random_same_norm_delta(
                        ascent_delta,
                        intervention_mask=accepted_mask,
                    )

                    branch_to_delta = {
                        "ascent": ascent_delta,
                        "descent": descent_delta,
                        "random_same_norm": random_delta,
                    }

                    branch_to_stats: dict[str, dict[str, object]] = {
                        "ascent": {
                            "steered_feature_value_local": selected_delta["steered_feature_value_local"],
                            "steered_score_value_local": selected_delta["steered_score_value_local"],
                            "steered_supported": selected_delta["steered_supported"],
                            "steered_region_label": selected_delta["steered_region_label"],
                            **compute_delta_norm_stats(token_hidden, delta=ascent_delta),
                        }
                    }
                    for intervention_type in ("descent", "random_same_norm"):
                        delta = branch_to_delta[intervention_type]
                        candidate_stats = evaluate_candidate_delta(
                            token_hidden,
                            delta=delta,
                            feature_name=feature_name,
                            layer_index_0based=layer_number - 1,
                            maybe_apply_final_norm=maybe_apply_final_norm,
                            lm_head_weight=lm_head_weight,
                            answer_id_tensor_lm_head=answer_id_tensor_lm_head.to(input_device),
                            vocab_size=vocab_size,
                            region_model=region_model,
                        )
                        branch_to_stats[intervention_type] = {
                            "steered_feature_value_local": candidate_stats["feature_value"],
                            "steered_score_value_local": candidate_stats["score_value"],
                            "steered_supported": candidate_stats["supported"],
                            "steered_region_label": candidate_stats["region_label"],
                            **compute_delta_norm_stats(token_hidden, delta=delta),
                        }

                    branch_to_outputs = {}
                    for intervention_type, delta in branch_to_delta.items():
                        branch_to_outputs[intervention_type] = run_single_intervention_forward(
                            batch=batch,
                            decision_pos=decision_pos,
                            model=model,
                            steering_module=steering_module,
                            delta=delta,
                            true_choice_idx=true_choice_idx,
                            answer_id_tensor_cpu=answer_id_tensor_cpu,
                        )

                    accepted_idx = np.where(accepted_mask)[0]
                    for batch_index in accepted_idx:
                        global_index = batch_indices[batch_index]
                        for intervention_type in CONTROL_INTERVENTION_TYPES:
                            branch_stats = branch_to_stats[intervention_type]
                            outputs = branch_to_outputs[intervention_type]
                            rows.append(
                                {
                                    "example_id": batch_df.loc[batch_index, "example_id"],
                                    "feature_name": feature_name,
                                    "layer_number": layer_number,
                                    "intervention_type": intervention_type,
                                    "max_delta_over_hidden": float(max_delta_over_hidden),
                                    "eligible_for_intervention": bool(effective_eligible[batch_index]),
                                    "accepted_intervention": True,
                                    "current_supported": bool(batch_current_supported[batch_index]),
                                    "current_region_label": str(batch_current_region[batch_index]),
                                    "steered_supported": bool(branch_stats["steered_supported"][batch_index]),
                                    "steered_region_label": str(branch_stats["steered_region_label"][batch_index]),
                                    "current_feature_value": float(ascent_basis["current_feature_value"][batch_index]),
                                    "current_score_value": float(batch_current_score[batch_index])
                                    if math.isfinite(float(batch_current_score[batch_index]))
                                    else np.nan,
                                    "current_score_derivative": float(batch_current_score_derivative[batch_index])
                                    if math.isfinite(float(batch_current_score_derivative[batch_index]))
                                    else np.nan,
                                    "accepted_step_scale": float(selected_delta["accepted_step_scale"][batch_index]),
                                    "steered_feature_value_local": float(
                                        branch_stats["steered_feature_value_local"][batch_index]
                                    )
                                    if math.isfinite(float(branch_stats["steered_feature_value_local"][batch_index]))
                                    else np.nan,
                                    "steered_score_value_local": float(
                                        branch_stats["steered_score_value_local"][batch_index]
                                    )
                                    if math.isfinite(float(branch_stats["steered_score_value_local"][batch_index]))
                                    else np.nan,
                                    "feature_grad_l2_norm": float(ascent_basis["feature_grad_l2_norm"][batch_index]),
                                    "score_grad_l2_norm": float(ascent_basis["score_grad_l2_norm"][batch_index]),
                                    "raw_delta_l2_norm": float(raw_delta_stats["raw_delta_l2_norm"][batch_index]),
                                    "raw_delta_over_token_hidden_l2": float(
                                        raw_delta_stats["raw_delta_over_token_hidden_l2"][batch_index]
                                    ),
                                    "delta_l2_norm": float(branch_stats["delta_l2_norm"][batch_index]),
                                    "delta_over_token_hidden_l2": float(
                                        branch_stats["delta_over_token_hidden_l2"][batch_index]
                                    ),
                                    "token_hidden_l2_norm": float(branch_stats["token_hidden_l2_norm"][batch_index]),
                                    "steered_best_non_choice_token_id": int(outputs["best_non_choice_token_id"][batch_index]),
                                    "steered_best_non_choice_logit": float(outputs["best_non_choice_logit"][batch_index]),
                                    "steered_logit_A": float(outputs["choice_logits"][batch_index, 0]),
                                    "steered_logit_B": float(outputs["choice_logits"][batch_index, 1]),
                                    "steered_logit_C": float(outputs["choice_logits"][batch_index, 2]),
                                    "steered_logit_D": float(outputs["choice_logits"][batch_index, 3]),
                                    "steered_logit_E": float(outputs["choice_logits"][batch_index, 4]),
                                    "clean_best_non_choice_token_id": int(clean_best_non_choice_token_id[global_index]),
                                    "clean_best_non_choice_logit": float(clean_best_non_choice_logit[global_index]),
                                    "clean_logit_A": float(clean_choice_logits[global_index, 0]),
                                    "clean_logit_B": float(clean_choice_logits[global_index, 1]),
                                    "clean_logit_C": float(clean_choice_logits[global_index, 2]),
                                    "clean_logit_D": float(clean_choice_logits[global_index, 3]),
                                    "clean_logit_E": float(clean_choice_logits[global_index, 4]),
                                }
                            )

                pbar.update(1)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    return pd.DataFrame(rows)


def prepare_batch(tok: AutoTokenizer, texts: list[str], max_seq_len: int) -> dict[str, torch.Tensor]:
    from src.csqa.common import encode_prompts  # noqa: WPS433

    return encode_prompts(texts, tok, max_seq_len)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--max-seq-len", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fit-split", type=str, default="validation")
    parser.add_argument("--eval-split", type=str, default="train")
    parser.add_argument("--fit-limit", type=int, default=None)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--validation-limit", type=int, default=None)
    parser.add_argument("--top-k-layers-per-feature", type=int, default=3)
    parser.add_argument(
        "--feature-names",
        type=str,
        default=",".join(DEFAULT_FEATURE_NAMES),
    )
    parser.add_argument("--max-delta-over-hidden", type=float, default=DEFAULT_MAX_DELTA_OVER_HIDDEN)
    parser.add_argument(
        "--backtrack-scales",
        type=str,
        default=",".join(str(x) for x in DEFAULT_BACKTRACK_SCALES),
    )
    parser.add_argument("--good-threshold-log-ratio", type=float, default=GOOD_REGION_LOG_RATIO_THRESHOLD)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    fit_limit = args.fit_limit if args.fit_limit is not None else args.train_limit
    eval_limit = args.eval_limit if args.eval_limit is not None else args.validation_limit
    if args.fit_split == args.eval_split:
        raise ValueError("--fit-split and --eval-split must be different")
    if args.max_delta_over_hidden <= 0:
        raise ValueError("--max-delta-over-hidden must be positive")
    backtrack_scales = parse_positive_scale_list(args.backtrack_scales)
    feature_names = parse_feature_names(args.feature_names)

    print(
        "[config]",
        json.dumps(
            {
                "dataset": "csqa",
                "model_id": args.model_id,
                "fit_split": args.fit_split,
                "eval_split": args.eval_split,
                "fit_limit": fit_limit,
                "eval_limit": eval_limit,
                "max_seq_len": args.max_seq_len,
                "seed": args.seed,
                "feature_names": feature_names,
                "top_k_layers_per_feature": int(args.top_k_layers_per_feature),
                "max_delta_over_hidden": float(args.max_delta_over_hidden),
                "backtrack_scales": backtrack_scales,
                "intervention_types": list(CONTROL_INTERVENTION_TYPES),
                "good_threshold_log_ratio": float(args.good_threshold_log_ratio),
                "support_lower_quantile": float(SUPPORT_LOWER_QUANTILE),
                "support_upper_quantile": float(SUPPORT_UPPER_QUANTILE),
                "kde_bandwidth_multiplier": float(KDE_BANDWIDTH_MULTIPLIER),
                "log_ratio_smoothing_sigma_bandwidths": float(LOG_RATIO_SMOOTHING_SIGMA_BANDWIDTHS),
            },
            indent=2,
        ),
    )

    fit_rows = load_csqa(split=args.fit_split, limit=fit_limit).copy()
    eval_rows = load_csqa(split=args.eval_split, limit=eval_limit).copy()
    for frame in [fit_rows, eval_rows]:
        frame["prompt_len_chars"] = frame["text"].str.len()

    model_dtype, device_map = choose_model_dtype_and_device_map()
    hf_token = resolve_hf_token()

    tok = AutoTokenizer.from_pretrained(args.model_id, token=hf_token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        token=hf_token,
        dtype=model_dtype,
        device_map=device_map,
        attn_implementation="eager",
    )
    model.eval()

    input_device = get_input_device(model)
    decoder_layers = get_decoder_layers(model)
    num_layers = len(decoder_layers)
    candidate_layer_numbers = all_layer_numbers(num_layers)
    vocab_size = int(model.config.vocab_size)

    readout_ctx = prepare_readout_context(
        model=model,
        tok=tok,
        input_device=input_device,
        num_layers=num_layers,
        max_seq_len=args.max_seq_len,
        probe_rows=eval_rows,
    )
    final_norm = readout_ctx["final_norm"]
    answer_id_tensor_cpu = readout_ctx["answer_id_tensor_cpu"]
    answer_id_tensor_lm_head = readout_ctx["answer_id_tensor_lm_head"]
    answer_choice_weight = readout_ctx["answer_choice_weight"]
    lm_head_weight = readout_ctx["lm_head_weight"]
    maybe_apply_final_norm = readout_ctx["maybe_apply_final_norm"]
    last_layer_needs_final_norm = readout_ctx["last_layer_needs_final_norm"]

    fit_cache = extract_split_cache(
        fit_rows,
        split_name=args.fit_split,
        tok=tok,
        model=model,
        input_device=input_device,
        answer_id_tensor_cpu=answer_id_tensor_cpu,
        answer_id_tensor_lm_head=answer_id_tensor_lm_head,
        answer_choice_weight=answer_choice_weight,
        final_norm=final_norm,
        num_layers=num_layers,
        max_seq_len=args.max_seq_len,
        batch_size=EXTRACT_BATCH_SIZE,
        last_layer_needs_final_norm=last_layer_needs_final_norm,
    )
    eval_cache = extract_split_cache(
        eval_rows,
        split_name=args.eval_split,
        tok=tok,
        model=model,
        input_device=input_device,
        answer_id_tensor_cpu=answer_id_tensor_cpu,
        answer_id_tensor_lm_head=answer_id_tensor_lm_head,
        answer_choice_weight=answer_choice_weight,
        final_norm=final_norm,
        num_layers=num_layers,
        max_seq_len=args.max_seq_len,
        batch_size=EXTRACT_BATCH_SIZE,
        last_layer_needs_final_norm=last_layer_needs_final_norm,
    )

    fit_feature_df = build_feature_table(
        split_name=args.fit_split,
        cache=fit_cache,
        feature_names=feature_names,
        maybe_apply_final_norm=maybe_apply_final_norm,
        lm_head_weight=lm_head_weight,
        answer_id_tensor_lm_head=answer_id_tensor_lm_head,
        input_device=input_device,
        vocab_size=vocab_size,
        active_layer_numbers=candidate_layer_numbers,
    )
    eval_feature_df = build_feature_table(
        split_name=args.eval_split,
        cache=eval_cache,
        feature_names=feature_names,
        maybe_apply_final_norm=maybe_apply_final_norm,
        lm_head_weight=lm_head_weight,
        answer_id_tensor_lm_head=answer_id_tensor_lm_head,
        input_device=input_device,
        vocab_size=vocab_size,
        active_layer_numbers=candidate_layer_numbers,
    )

    separation_df = build_separation_summary(fit_feature_df)
    selected_layers_df = select_top_k_layers_by_feature(
        separation_df,
        feature_names=feature_names,
        top_k=int(args.top_k_layers_per_feature),
    )
    region_models, distribution_grid_df = build_distribution_models(
        fit_feature_df=fit_feature_df,
        selected_layers_df=selected_layers_df,
        log_ratio_threshold=float(args.good_threshold_log_ratio),
        grid_points=GRID_POINTS,
    )
    distribution_grid_df = augment_region_models_with_derivative(region_models, distribution_grid_df)

    policy_outputs_df = run_score_control_policy(
        eval_rows=eval_rows,
        eval_cache=eval_cache,
        eval_feature_df=eval_feature_df,
        selected_layers_df=selected_layers_df,
        region_models=region_models,
        backtrack_scales=backtrack_scales,
        max_delta_over_hidden=float(args.max_delta_over_hidden),
        tok=tok,
        model=model,
        input_device=input_device,
        decoder_layers=decoder_layers,
        answer_id_tensor_cpu=answer_id_tensor_cpu,
        answer_id_tensor_lm_head=answer_id_tensor_lm_head,
        lm_head_weight=lm_head_weight,
        maybe_apply_final_norm=maybe_apply_final_norm,
        vocab_size=vocab_size,
        max_seq_len=args.max_seq_len,
        eval_split_name=args.eval_split,
    )

    out_dir = resolve_out_dir(args.out_dir, args.model_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(fit_cache["example_rows"]).to_parquet(out_dir / f"{args.fit_split}_examples.parquet", index=False)
    pd.DataFrame(eval_cache["example_rows"]).to_parquet(out_dir / f"{args.eval_split}_examples.parquet", index=False)
    pd.DataFrame(fit_cache["clean_output_rows"]).to_parquet(
        out_dir / f"{args.fit_split}_clean_final_outputs.parquet", index=False
    )
    pd.DataFrame(eval_cache["clean_output_rows"]).to_parquet(
        out_dir / f"{args.eval_split}_clean_final_outputs.parquet", index=False
    )
    fit_feature_df.to_parquet(out_dir / f"{args.fit_split}_univariate_feature_values.parquet", index=False)
    eval_feature_df.to_parquet(out_dir / f"{args.eval_split}_univariate_feature_values.parquet", index=False)
    separation_df.to_parquet(out_dir / "feature_layer_separation_summary.parquet", index=False)
    selected_layers_df.to_parquet(out_dir / "selected_layers_by_feature.parquet", index=False)
    distribution_grid_df.to_parquet(out_dir / "fit_distribution_grid.parquet", index=False)
    policy_outputs_df.to_parquet(out_dir / f"{args.eval_split}_score_control_policy_outputs_raw.parquet", index=False)

    run_config = {
        "dataset": "csqa",
        "model_id": args.model_id,
        "seed": int(args.seed),
        "max_seq_len": int(args.max_seq_len),
        "fit_split": args.fit_split,
        "eval_split": args.eval_split,
        "fit_limit": fit_limit,
        "eval_limit": eval_limit,
        "method": "logit_feature_score_control",
        "intervention_types": list(CONTROL_INTERVENTION_TYPES),
        "feature_names": feature_names,
        "candidate_layer_numbers": candidate_layer_numbers,
        "layer_selection_rule": "top_k_by_ks_statistic",
        "top_k_layers_per_feature": int(args.top_k_layers_per_feature),
        "max_delta_over_hidden": float(args.max_delta_over_hidden),
        "backtrack_scales": backtrack_scales,
        "good_threshold_log_ratio": float(args.good_threshold_log_ratio),
        "support_lower_quantile": float(SUPPORT_LOWER_QUANTILE),
        "support_upper_quantile": float(SUPPORT_UPPER_QUANTILE),
        "kde_bandwidth_multiplier": float(KDE_BANDWIDTH_MULTIPLIER),
        "log_ratio_smoothing_sigma_bandwidths": float(LOG_RATIO_SMOOTHING_SIGMA_BANDWIDTHS),
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    print(f"[done] Wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
