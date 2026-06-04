from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cli.run_csqa_adaptive_contrastive_pipeline import (  # noqa: E402
    EXTRACT_BATCH_SIZE,
    choose_model_dtype_and_device_map,
    extract_split_cache,
    get_input_device,
    prepare_readout_context,
)
from src.cli.run_csqa_logit_feature_steering_pipeline import (  # noqa: E402
    FEATURE_NAMES,
    INTERVENTION_BATCH_SIZE,
    TARGET_LOWER_QUANTILE,
    TARGET_UPPER_QUANTILE,
    build_feature_table,
    build_feature_target_stats,
    compute_feature_steering_delta,
    resolve_hf_token,
)
from src.csqa.common import (  # noqa: E402
    encode_prompts,
    get_decoder_layers,
    repack_output_hidden,
    summarize_decision_logits,
    unpack_output_hidden,
)
from src.data.load_csqa import load_csqa  # noqa: E402


LETTERS = ["A", "B", "C", "D", "E"]


def now_id() -> str:
    import time

    return time.strftime("%Y%m%d-%H%M%S")


def repo_root() -> Path:
    return REPO_ROOT


def slugify_model_id(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", model_id).strip("-")


def resolve_out_dir(out_dir: str | None, model_id: str) -> Path:
    root = repo_root()
    if out_dir is None:
        run_name = f"{now_id()}_{slugify_model_id(model_id)}_csqa_feature_band_steering_pipeline"
        return root / "data" / "generated" / "csqa_feature_band_steering_pipeline" / run_name
    path = Path(out_dir)
    return path if path.is_absolute() else (root / path)


def run_eval_policy(
    *,
    eval_rows: pd.DataFrame,
    eval_cache: dict[str, object],
    eval_feature_df: pd.DataFrame,
    target_stats_df: pd.DataFrame,
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
    feature_lookup = (
        eval_feature_df.set_index(["example_id", "feature_name", "layer_number"])["feature_value"].sort_index()
    )
    target_lookup = target_stats_df.set_index(["feature_name", "layer_number"])

    clean_choice_logits = eval_cache["clean_choice_logits"]
    clean_best_non_choice_logit = eval_cache["clean_best_non_choice_logit"]
    clean_best_non_choice_token_id = eval_cache["clean_best_non_choice_token_id"]
    example_ids = [row["example_id"] for row in eval_cache["example_rows"]]
    example_id_to_index = {example_id: idx for idx, example_id in enumerate(example_ids)}

    rows: list[dict[str, object]] = []
    total_steps = len(decoder_layers) * len(FEATURE_NAMES)

    with tqdm(total=total_steps, desc=f"{eval_split_name} policy sweep") as pbar:
        for feature_name in FEATURE_NAMES:
            for layer_number in range(1, len(decoder_layers) + 1):
                steering_module = decoder_layers[layer_number - 1]
                target_row = target_lookup.loc[(feature_name, layer_number)]
                target_lower = float(target_row["target_lower"])
                target_upper = float(target_row["target_upper"])
                band_width = max(target_upper - target_lower, 1e-12)

                current_feature_all = np.array(
                    [
                        float(feature_lookup.loc[(example_id, feature_name, layer_number)])
                        for example_id in example_ids
                    ],
                    dtype=np.float32,
                )
                outside_low = current_feature_all < target_lower
                outside_high = current_feature_all > target_upper
                outside_band_all = outside_low | outside_high
                raw_distance_all = np.where(
                    outside_low,
                    target_lower - current_feature_all,
                    np.where(outside_high, current_feature_all - target_upper, 0.0),
                ).astype(np.float32)
                normalized_distance_all = (raw_distance_all / band_width).astype(np.float32)

                for start in range(0, len(eval_rows), INTERVENTION_BATCH_SIZE):
                    batch_df = eval_rows.iloc[start:start + INTERVENTION_BATCH_SIZE].reset_index(drop=True)
                    batch_indices = [example_id_to_index[example_id] for example_id in batch_df["example_id"].tolist()]
                    batch_current_feature = current_feature_all[batch_indices]
                    batch_outside_band = outside_band_all[batch_indices]
                    batch_distance = normalized_distance_all[batch_indices]

                    if not bool(batch_outside_band.any()):
                        for batch_index, row in batch_df.iterrows():
                            global_index = batch_indices[batch_index]
                            rows.append(
                                {
                                    "example_id": row["example_id"],
                                    "feature_name": feature_name,
                                    "layer_number": layer_number,
                                    "current_feature_value": float(batch_current_feature[batch_index]),
                                    "feature_outside_target_band": False,
                                    "normalized_distance_to_target_band": float(batch_distance[batch_index]),
                                    "target_lower": target_lower,
                                    "target_upper": target_upper,
                                    "steered_feature_value_local": float(batch_current_feature[batch_index]),
                                    "grad_l2_norm": 0.0,
                                    "delta_l2_norm": 0.0,
                                    "delta_over_token_hidden_l2": 0.0,
                                    "token_hidden_l2_norm": np.nan,
                                    "steered_best_non_choice_token_id": int(clean_best_non_choice_token_id[global_index]),
                                    "steered_best_non_choice_logit": float(clean_best_non_choice_logit[global_index]),
                                    "steered_logit_A": float(clean_choice_logits[global_index, 0]),
                                    "steered_logit_B": float(clean_choice_logits[global_index, 1]),
                                    "steered_logit_C": float(clean_choice_logits[global_index, 2]),
                                    "steered_logit_D": float(clean_choice_logits[global_index, 3]),
                                    "steered_logit_E": float(clean_choice_logits[global_index, 4]),
                                }
                            )
                        continue

                    batch_cpu = encode_prompts(batch_df["text"].tolist(), tok, max_seq_len)
                    decision_pos = batch_cpu.pop("decision_pos")
                    batch_cpu.pop("prompt_token_count")
                    batch = {k: v.to(input_device) for k, v in batch_cpu.items()}
                    decision_pos = decision_pos.to(input_device)
                    true_choice_idx = torch.tensor(
                        [LETTERS.index(str(x)) for x in batch_df["answerKey"].tolist()],
                        dtype=torch.long,
                        device=input_device,
                    )
                    steering_stats: dict[str, np.ndarray] = {}

                    def steering_hook(module, inputs, output):
                        hidden = unpack_output_hidden(output)
                        row_idx = torch.arange(hidden.shape[0], device=hidden.device)
                        token_hidden = hidden[row_idx, decision_pos]
                        stats = compute_feature_steering_delta(
                            token_hidden,
                            feature_name=feature_name,
                            layer_index_0based=layer_number - 1,
                            maybe_apply_final_norm=maybe_apply_final_norm,
                            lm_head_weight=lm_head_weight,
                            answer_id_tensor_lm_head=answer_id_tensor_lm_head.to(hidden.device),
                            vocab_size=vocab_size,
                            target_lower=target_lower,
                            target_upper=target_upper,
                            intervention_mask=batch_outside_band,
                        )
                        steering_stats.update({key: value for key, value in stats.items() if key != "delta"})
                        hidden_out = hidden.clone()
                        hidden_out[row_idx, decision_pos] = token_hidden + stats["delta"].to(hidden.device, dtype=hidden.dtype)
                        return repack_output_hidden(output, hidden_out)

                    handle = steering_module.register_forward_hook(steering_hook)
                    try:
                        with torch.no_grad():
                            out = model(**batch, return_dict=True, use_cache=False)
                    finally:
                        handle.remove()

                    full_logits = out.logits[torch.arange(len(batch_df), device=out.logits.device), decision_pos].float()
                    metrics = summarize_decision_logits(
                        full_logits,
                        true_choice_idx,
                        answer_id_tensor_cpu.to(full_logits.device),
                    )
                    choice_logits_cpu = metrics["choice_logits"].detach().cpu().numpy().astype(np.float32)
                    best_non_choice_logit_cpu = metrics["best_non_choice_logit"].detach().cpu().numpy().astype(np.float32)
                    masked_logits = full_logits.clone()
                    masked_logits[:, answer_id_tensor_cpu.to(full_logits.device)] = -torch.inf
                    best_non_choice_token_id_cpu = torch.argmax(masked_logits, dim=-1).detach().cpu().numpy().astype(np.int64)

                    for batch_index, row in batch_df.iterrows():
                        rows.append(
                            {
                                "example_id": row["example_id"],
                                "feature_name": feature_name,
                                "layer_number": layer_number,
                                "current_feature_value": float(batch_current_feature[batch_index]),
                                "feature_outside_target_band": bool(batch_outside_band[batch_index]),
                                "normalized_distance_to_target_band": float(batch_distance[batch_index]),
                                "target_lower": target_lower,
                                "target_upper": target_upper,
                                "steered_feature_value_local": float(steering_stats["steered_feature_value_local"][batch_index]),
                                "grad_l2_norm": float(steering_stats["grad_l2_norm"][batch_index]),
                                "delta_l2_norm": float(steering_stats["delta_l2_norm"][batch_index]),
                                "delta_over_token_hidden_l2": float(steering_stats["delta_over_token_hidden_l2"][batch_index]),
                                "token_hidden_l2_norm": float(steering_stats["token_hidden_l2_norm"][batch_index]),
                                "steered_best_non_choice_token_id": int(best_non_choice_token_id_cpu[batch_index]),
                                "steered_best_non_choice_logit": float(best_non_choice_logit_cpu[batch_index]),
                                "steered_logit_A": float(choice_logits_cpu[batch_index, 0]),
                                "steered_logit_B": float(choice_logits_cpu[batch_index, 1]),
                                "steered_logit_C": float(choice_logits_cpu[batch_index, 2]),
                                "steered_logit_D": float(choice_logits_cpu[batch_index, 3]),
                                "steered_logit_E": float(choice_logits_cpu[batch_index, 4]),
                            }
                        )

                pbar.update(1)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    return pd.DataFrame(rows)


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
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    fit_limit = args.fit_limit if args.fit_limit is not None else args.validation_limit
    eval_limit = args.eval_limit if args.eval_limit is not None else args.train_limit
    if args.fit_split == args.eval_split:
        raise ValueError("--fit-split and --eval-split must be different")

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
        maybe_apply_final_norm=maybe_apply_final_norm,
        lm_head_weight=lm_head_weight,
        answer_id_tensor_lm_head=answer_id_tensor_lm_head,
        input_device=input_device,
        vocab_size=vocab_size,
    )
    eval_feature_df = build_feature_table(
        split_name=args.eval_split,
        cache=eval_cache,
        maybe_apply_final_norm=maybe_apply_final_norm,
        lm_head_weight=lm_head_weight,
        answer_id_tensor_lm_head=answer_id_tensor_lm_head,
        input_device=input_device,
        vocab_size=vocab_size,
    )

    target_stats_df = build_feature_target_stats(fit_feature_df)

    eval_policy_outputs_raw_df = run_eval_policy(
        eval_rows=eval_rows,
        eval_cache=eval_cache,
        eval_feature_df=eval_feature_df,
        target_stats_df=target_stats_df,
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
    pd.DataFrame(fit_cache["clean_output_rows"]).to_parquet(out_dir / f"{args.fit_split}_clean_final_outputs.parquet", index=False)
    pd.DataFrame(eval_cache["clean_output_rows"]).to_parquet(out_dir / f"{args.eval_split}_clean_final_outputs.parquet", index=False)
    fit_feature_df.to_parquet(out_dir / f"{args.fit_split}_univariate_feature_values.parquet", index=False)
    eval_feature_df.to_parquet(out_dir / f"{args.eval_split}_univariate_feature_values.parquet", index=False)
    target_stats_df.to_parquet(out_dir / "feature_target_stats.parquet", index=False)
    eval_policy_outputs_raw_df.to_parquet(out_dir / f"{args.eval_split}_policy_outputs_raw.parquet", index=False)

    run_config = {
        "dataset": "csqa",
        "model_id": args.model_id,
        "seed": int(args.seed),
        "max_seq_len": int(args.max_seq_len),
        "fit_split": args.fit_split,
        "eval_split": args.eval_split,
        "fit_limit": fit_limit,
        "eval_limit": eval_limit,
        "method": "feature_band_steering",
        "feature_names": FEATURE_NAMES,
        "target_lower_quantile": TARGET_LOWER_QUANTILE,
        "target_upper_quantile": TARGET_UPPER_QUANTILE,
        "gate_rule": "intervene_if_feature_outside_correct_trace_iqr",
        "extract_batch_size": EXTRACT_BATCH_SIZE,
        "intervention_batch_size": INTERVENTION_BATCH_SIZE,
        "last_layer_needs_final_norm": bool(last_layer_needs_final_norm),
    }
    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    print(f"[done] wrote {out_dir}")


if __name__ == "__main__":
    main()
