from __future__ import annotations

import argparse
import gc
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cli.run_csqa_adaptive_contrastive_pipeline import (  # noqa: E402
    EXTRACT_BATCH_SIZE,
    READOUT_BATCH_SIZE,
    choose_model_dtype_and_device_map,
    extract_split_cache,
    get_input_device,
    prepare_readout_context,
)
from src.cli.logit_feature_response_curve_suite.run_csqa_logit_feature_response_curve_pipeline import (  # noqa: E402
    GOOD_REGION_LOG_RATIO_THRESHOLD,
    GRID_POINTS,
    KDE_BANDWIDTH_MULTIPLIER,
    LOG_RATIO_SMOOTHING_SIGMA_BANDWIDTHS,
    SUPPORT_LOWER_QUANTILE,
    SUPPORT_UPPER_QUANTILE,
    build_distribution_models,
    build_feature_table,
    build_separation_summary,
    select_top_k_layers_by_feature,
)
from src.cli.logit_feature_score_suite.run_csqa_logit_feature_score_pipeline import (  # noqa: E402
    DEFAULT_FEATURE_NAMES,
    all_layer_numbers,
    augment_region_models_with_derivative,
    parse_feature_names,
)
from src.cli.run_csqa_logit_feature_steering_pipeline import resolve_hf_token  # noqa: E402
from src.csqa.common import get_decoder_layers  # noqa: E402
from src.data.load_csqa import load_csqa  # noqa: E402


LETTERS = ["A", "B", "C", "D", "E"]


def now_id() -> str:
    import time

    return time.strftime("%Y%m%d-%H%M%S")


def slugify_model_id(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", model_id).strip("-")


def resolve_out_dir(out_dir: str | None, model_id: str) -> Path:
    if out_dir is None:
        run_name = f"{now_id()}_{slugify_model_id(model_id)}_csqa_logit_feature_diagnostics_pipeline"
        return REPO_ROOT / "data" / "generated" / "logit_feature_diagnostics_pipeline" / run_name
    path = Path(out_dir)
    return path if path.is_absolute() else (REPO_ROOT / path)


def default_limit_for_split(split_name: str) -> int | None:
    return None


def build_layerwise_choice_readout_table(
    *,
    cache: dict[str, object],
    split_name: str,
    maybe_apply_final_norm,
    lm_head_weight: torch.Tensor,
    answer_id_tensor_lm_head: torch.Tensor,
    input_device: torch.device,
) -> pd.DataFrame:
    hidden = cache["hidden"]
    example_rows = cache["example_rows"]
    row_meta_records = pd.DataFrame(example_rows)[["example_id", "split"]].to_dict("records")
    rows: list[dict[str, object]] = []

    for layer_index in tqdm(range(hidden.shape[1]), desc=f"{split_name} layerwise readouts"):
        layer_hidden = hidden[:, layer_index, :]
        for start in range(0, layer_hidden.shape[0], READOUT_BATCH_SIZE):
            end = start + READOUT_BATCH_SIZE
            hidden_batch = layer_hidden[start:end].to(input_device).float()
            readout = maybe_apply_final_norm(hidden_batch, layer_index)
            full_logits = torch.matmul(
                readout.to(lm_head_weight.dtype),
                lm_head_weight.T,
            ).float()
            masked_logits = full_logits.clone()
            masked_logits[:, answer_id_tensor_lm_head] = -torch.inf
            best_non_choice_logit, best_non_choice_token_id = torch.max(masked_logits, dim=-1)
            choice_logits = full_logits.index_select(1, answer_id_tensor_lm_head)

            choice_logits_cpu = choice_logits.detach().cpu().numpy().astype(np.float32)
            best_non_choice_logit_cpu = best_non_choice_logit.detach().cpu().numpy().astype(np.float32)
            best_non_choice_token_id_cpu = best_non_choice_token_id.detach().cpu().numpy().astype(np.int64)

            for local_index, global_index in enumerate(range(start, min(end, layer_hidden.shape[0]))):
                meta = row_meta_records[global_index]
                row = {
                    "example_id": meta["example_id"],
                    "split": meta["split"],
                    "layer_number": layer_index + 1,
                    "best_non_choice_token_id": int(best_non_choice_token_id_cpu[local_index]),
                    "best_non_choice_logit": float(best_non_choice_logit_cpu[local_index]),
                }
                for choice_index, letter in enumerate(LETTERS):
                    row[f"logit_{letter}"] = float(choice_logits_cpu[local_index, choice_index])
                rows.append(row)

            del hidden_batch
            del readout
            del full_logits
            del masked_logits
            del best_non_choice_logit
            del best_non_choice_token_id
            del choice_logits

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the CSQA logit-feature diagnostics pipeline: all-layer feature traces, KS separation, and s(x) models without interventions."
    )
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--max-seq-len", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fit-split", type=str, default="train")
    parser.add_argument("--eval-split", type=str, default="validation")
    parser.add_argument("--fit-limit", type=int, default=None)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--validation-limit", type=int, default=None)
    parser.add_argument(
        "--feature-names",
        type=str,
        default=",".join(DEFAULT_FEATURE_NAMES),
    )
    parser.add_argument("--good-threshold-log-ratio", type=float, default=GOOD_REGION_LOG_RATIO_THRESHOLD)
    parser.add_argument("--grid-points", type=int, default=GRID_POINTS)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    fit_limit = (
        args.fit_limit
        if args.fit_limit is not None
        else (
            args.train_limit
            if args.fit_split == "train" and args.train_limit is not None
            else (
                args.validation_limit
                if args.fit_split == "validation" and args.validation_limit is not None
                else default_limit_for_split(args.fit_split)
            )
        )
    )
    eval_limit = (
        args.eval_limit
        if args.eval_limit is not None
        else (
            args.train_limit
            if args.eval_split == "train" and args.train_limit is not None
            else (
                args.validation_limit
                if args.eval_split == "validation" and args.validation_limit is not None
                else default_limit_for_split(args.eval_split)
            )
        )
    )
    if args.fit_split == args.eval_split:
        raise ValueError("--fit-split and --eval-split must be different")
    if args.grid_points <= 8:
        raise ValueError("--grid-points must be greater than 8")
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
                "distribution_layers": "all",
                "good_threshold_log_ratio": float(args.good_threshold_log_ratio),
                "grid_points": int(args.grid_points),
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
    fit_layerwise_readouts_df = build_layerwise_choice_readout_table(
        cache=fit_cache,
        split_name=args.fit_split,
        maybe_apply_final_norm=maybe_apply_final_norm,
        lm_head_weight=lm_head_weight,
        answer_id_tensor_lm_head=answer_id_tensor_lm_head,
        input_device=input_device,
    )
    eval_layerwise_readouts_df = build_layerwise_choice_readout_table(
        cache=eval_cache,
        split_name=args.eval_split,
        maybe_apply_final_norm=maybe_apply_final_norm,
        lm_head_weight=lm_head_weight,
        answer_id_tensor_lm_head=answer_id_tensor_lm_head,
        input_device=input_device,
    )

    separation_df = build_separation_summary(fit_feature_df)
    ranked_layers_df = select_top_k_layers_by_feature(
        separation_df,
        feature_names=feature_names,
        top_k=len(candidate_layer_numbers),
    )
    region_models, distribution_grid_df = build_distribution_models(
        fit_feature_df=fit_feature_df,
        selected_layers_df=ranked_layers_df,
        log_ratio_threshold=float(args.good_threshold_log_ratio),
        grid_points=int(args.grid_points),
    )
    distribution_grid_df = augment_region_models_with_derivative(region_models, distribution_grid_df)

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
    fit_layerwise_readouts_df.to_parquet(out_dir / f"{args.fit_split}_layerwise_choice_readouts.parquet", index=False)
    eval_layerwise_readouts_df.to_parquet(out_dir / f"{args.eval_split}_layerwise_choice_readouts.parquet", index=False)
    separation_df.to_parquet(out_dir / "feature_layer_separation_summary.parquet", index=False)
    distribution_grid_df.to_parquet(out_dir / "fit_distribution_grid.parquet", index=False)

    run_config = {
        "dataset": "csqa",
        "model_id": args.model_id,
        "seed": int(args.seed),
        "max_seq_len": int(args.max_seq_len),
        "fit_split": args.fit_split,
        "eval_split": args.eval_split,
        "fit_limit": fit_limit,
        "eval_limit": eval_limit,
        "method": "logit_feature_diagnostics",
        "feature_names": feature_names,
        "candidate_layer_numbers": candidate_layer_numbers,
        "distribution_build_rule": "all_layers_from_feature_layer_separation_summary",
        "good_threshold_log_ratio": float(args.good_threshold_log_ratio),
        "support_lower_quantile": float(SUPPORT_LOWER_QUANTILE),
        "support_upper_quantile": float(SUPPORT_UPPER_QUANTILE),
        "kde_bandwidth_multiplier": float(KDE_BANDWIDTH_MULTIPLIER),
        "log_ratio_smoothing_sigma_bandwidths": float(LOG_RATIO_SMOOTHING_SIGMA_BANDWIDTHS),
        "distribution_grid_points": int(args.grid_points),
        "extract_batch_size": EXTRACT_BATCH_SIZE,
        "readout_batch_size": READOUT_BATCH_SIZE,
        "last_layer_needs_final_norm": bool(last_layer_needs_final_norm),
        "saved_tables": [
            f"{args.fit_split}_examples.parquet",
            f"{args.eval_split}_examples.parquet",
            f"{args.fit_split}_clean_final_outputs.parquet",
            f"{args.eval_split}_clean_final_outputs.parquet",
            f"{args.fit_split}_layerwise_choice_readouts.parquet",
            f"{args.eval_split}_layerwise_choice_readouts.parquet",
            "feature_layer_separation_summary.parquet",
            "fit_distribution_grid.parquet",
        ],
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    print(f"[done] Wrote outputs to {out_dir}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
