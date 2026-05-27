from __future__ import annotations

import argparse
import gc
import json
import math
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.load_csqa import load_csqa
from src.cli.extract_csqa_steering_all_traces import (
    LETTERS,
    apply_token_steering,
    build_answer_token_ids,
    build_contrastive_mean_direction,
    encode_prompts,
    extract_clean_baseline,
    extract_train_hidden_cache,
    get_decoder_layers,
    get_input_device,
    repack_output_hidden,
    select_full_logits_at_decision,
    slugify_model_id,
    summarize_decision_logits,
    unpack_output_hidden,
)


def now_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def repo_root() -> Path:
    return REPO_ROOT


def resolve_out_dir(out_dir: str | None, model_id: str) -> Path:
    root = repo_root()
    if out_dir is None:
        run_name = f"{now_id()}_{slugify_model_id(model_id)}_csqa_contrastive_scale_sweep"
        return root / "data" / "generated" / "csqa_contrastive_scale_sweep" / run_name
    path = Path(out_dir)
    return path if path.is_absolute() else (root / path)


def parse_scales(raw: str) -> list[float]:
    values = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError("At least one steering scale is required.")
    return values


def run_contrastive_scale_sweep(
    frame: pd.DataFrame,
    *,
    tok: AutoTokenizer,
    model: AutoModelForCausalLM,
    input_device: torch.device,
    decoder_layers,
    answer_id_tensor: torch.Tensor,
    max_seq_len: int,
    batch_size: int,
    steering_scales: list[float],
    contrastive_directions: dict[int, torch.Tensor],
) -> pd.DataFrame:
    result_rows: list[dict[str, object]] = []
    layer_numbers = list(range(1, len(decoder_layers) + 1))

    for layer_number in layer_numbers:
        steering_module = decoder_layers[layer_number - 1]
        direction = contrastive_directions[layer_number]

        for steering_scale in steering_scales:
            for start in range(0, len(frame), batch_size):
                batch_df = frame.iloc[start:start + batch_size].reset_index(drop=True)
                batch_cpu = encode_prompts(batch_df["text"].tolist(), tok, max_seq_len)
                decision_pos = batch_cpu.pop("decision_pos")
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
                    token_hidden_float = token_hidden.float()
                    token_hidden_rms = token_hidden_float.pow(2).mean(dim=-1).sqrt()
                    token_hidden_l2_norm = token_hidden_float.norm(dim=-1)
                    direction_device = direction.to(hidden.device, dtype=hidden.dtype)
                    direction_l2_norm = direction_device.float().norm()
                    delta_l2_norm = float(steering_scale) * token_hidden_rms * direction_l2_norm
                    delta_over_token_hidden_l2 = delta_l2_norm / token_hidden_l2_norm.clamp_min(1e-12)

                    steering_stats["token_hidden_rms"] = token_hidden_rms.detach().cpu().numpy().astype(np.float32)
                    steering_stats["token_hidden_l2_norm"] = token_hidden_l2_norm.detach().cpu().numpy().astype(np.float32)
                    steering_stats["direction_l2_norm"] = np.full(
                        hidden.shape[0],
                        float(direction_l2_norm.item()),
                        dtype=np.float32,
                    )
                    steering_stats["delta_l2_norm"] = delta_l2_norm.detach().cpu().numpy().astype(np.float32)
                    steering_stats["delta_over_token_hidden_l2"] = (
                        delta_over_token_hidden_l2.detach().cpu().numpy().astype(np.float32)
                    )

                    hidden = apply_token_steering(hidden, decision_pos, direction, float(steering_scale))
                    return repack_output_hidden(output, hidden)

                handle = steering_module.register_forward_hook(steering_hook)
                try:
                    with torch.inference_mode():
                        out = model(**batch, return_dict=True, use_cache=False)
                finally:
                    handle.remove()

                full_logits = select_full_logits_at_decision(out.logits, decision_pos)
                metrics = summarize_decision_logits(full_logits, true_choice_idx, answer_id_tensor.to(full_logits.device))

                choice_logits_cpu = metrics["choice_logits"].detach().cpu().numpy().astype(np.float32)
                best_non_choice_logit_cpu = metrics["best_non_choice_logit"].detach().cpu().numpy().astype(np.float32)
                masked_logits = full_logits.clone()
                masked_logits[:, answer_id_tensor.to(full_logits.device)] = -torch.inf
                best_non_choice_token_id_cpu = (
                    torch.argmax(masked_logits, dim=-1).detach().cpu().numpy().astype(np.int64)
                )

                for batch_index, row in batch_df.iterrows():
                    result_rows.append(
                        {
                            "example_id": row["example_id"],
                            "method": "contrastive_mean_direction",
                            "layer_number": layer_number,
                            "scale": float(steering_scale),
                            "token_hidden_rms": float(steering_stats["token_hidden_rms"][batch_index]),
                            "token_hidden_l2_norm": float(steering_stats["token_hidden_l2_norm"][batch_index]),
                            "direction_l2_norm": float(steering_stats["direction_l2_norm"][batch_index]),
                            "delta_l2_norm": float(steering_stats["delta_l2_norm"][batch_index]),
                            "delta_over_token_hidden_l2": float(steering_stats["delta_over_token_hidden_l2"][batch_index]),
                            "steered_best_non_choice_token_id": int(best_non_choice_token_id_cpu[batch_index]),
                            "steered_best_non_choice_logit": float(best_non_choice_logit_cpu[batch_index]),
                            "steered_logit_A": float(choice_logits_cpu[batch_index, 0]),
                            "steered_logit_B": float(choice_logits_cpu[batch_index, 1]),
                            "steered_logit_C": float(choice_logits_cpu[batch_index, 2]),
                            "steered_logit_D": float(choice_logits_cpu[batch_index, 3]),
                            "steered_logit_E": float(choice_logits_cpu[batch_index, 4]),
                        }
                    )

                del out
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    return pd.DataFrame(result_rows)


def summarize_run(clean_baseline_df: pd.DataFrame, steering_results_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = clean_baseline_df.copy()
    clean_logits = baseline[[f"clean_logit_{letter}" for letter in LETTERS]].to_numpy(dtype=np.float64)
    baseline["clean_predicted_choice_idx"] = clean_logits.argmax(axis=1).astype(np.int64)
    baseline["clean_is_correct"] = baseline["clean_predicted_choice_idx"] == baseline["true_choice_idx"]

    results = steering_results_df.copy()
    steered_logits = results[[f"steered_logit_{letter}" for letter in LETTERS]].to_numpy(dtype=np.float64)
    results["steered_predicted_choice_idx"] = steered_logits.argmax(axis=1).astype(np.int64)

    analysis = results.merge(
        baseline[["example_id", "true_choice_idx", "clean_predicted_choice_idx", "clean_is_correct"]],
        on="example_id",
        how="left",
        validate="many_to_one",
    )
    analysis["steered_is_correct"] = analysis["steered_predicted_choice_idx"] == analysis["true_choice_idx"]
    analysis["prediction_changed"] = analysis["steered_predicted_choice_idx"] != analysis["clean_predicted_choice_idx"]
    analysis["rescued_error"] = (~analysis["clean_is_correct"]) & analysis["steered_is_correct"]
    analysis["harmed_correct"] = analysis["clean_is_correct"] & (~analysis["steered_is_correct"])
    analysis["orig_incorrect"] = (~analysis["clean_is_correct"]).astype(int)
    analysis["orig_correct"] = analysis["clean_is_correct"].astype(int)

    by_config = analysis.groupby(["layer_number", "scale"], as_index=False).agg(
        n=("example_id", "size"),
        n_rescued=("rescued_error", "sum"),
        n_harmed=("harmed_correct", "sum"),
        n_changed=("prediction_changed", "sum"),
        n_orig_incorrect=("orig_incorrect", "sum"),
        n_orig_correct=("orig_correct", "sum"),
        mean_delta_over_token_hidden_l2=("delta_over_token_hidden_l2", "mean"),
        mean_delta_l2_norm=("delta_l2_norm", "mean"),
    )
    by_config["rescue_rate_bad"] = by_config["n_rescued"] / by_config["n_orig_incorrect"]
    by_config["harm_rate_good"] = by_config["n_harmed"] / by_config["n_orig_correct"]
    by_config["net_gain_count"] = by_config["n_rescued"] - by_config["n_harmed"]
    by_config["net_gain_rate"] = by_config["net_gain_count"] / by_config["n"]

    by_scale = by_config.groupby("scale", as_index=False).agg(
        mean_net_gain_count=("net_gain_count", "mean"),
        max_net_gain_count=("net_gain_count", "max"),
        mean_rescue_rate_bad=("rescue_rate_bad", "mean"),
        mean_harm_rate_good=("harm_rate_good", "mean"),
        positive_layers=("net_gain_count", lambda s: int((s > 0).sum())),
        zero_layers=("net_gain_count", lambda s: int((s == 0).sum())),
        negative_layers=("net_gain_count", lambda s: int((s < 0).sum())),
        mean_delta_over_token_hidden_l2=("mean_delta_over_token_hidden_l2", "mean"),
    )
    return by_config, by_scale


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--max-seq-len", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scales", type=str, default="0.5,1,2,3,4,5,6,8")
    args = parser.parse_args()

    train_layer_extraction_batch_size = 4
    clean_baseline_batch_size = 4
    steering_batch_size = 2
    train_split = "train"
    target_split = "validation"
    steering_scales = parse_scales(args.scales)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_rows = load_csqa(split=train_split, limit=None).copy()
    target_rows = load_csqa(split=target_split, limit=None).copy()

    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            model_dtype = torch.bfloat16
        else:
            model_dtype = torch.float16
        device_map = "auto"
    else:
        model_dtype = torch.float32
        device_map = None

    tok = AutoTokenizer.from_pretrained(args.model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=model_dtype,
        device_map=device_map,
        attn_implementation="eager",
    )
    model.eval()

    input_device = get_input_device(model)
    decoder_layers = get_decoder_layers(model)
    num_layers = len(decoder_layers)
    hidden_size = int(getattr(model.config, "hidden_size", model.lm_head.weight.shape[1]))
    answer_token_ids = build_answer_token_ids(tok)
    answer_id_tensor = torch.tensor([answer_token_ids[letter] for letter in LETTERS], dtype=torch.long)

    clean_baseline_df = extract_clean_baseline(
        target_rows,
        split_name=target_split,
        tok=tok,
        model=model,
        input_device=input_device,
        answer_id_tensor=answer_id_tensor,
        max_seq_len=args.max_seq_len,
        batch_size=clean_baseline_batch_size,
    )

    train_clean_df, train_hidden_by_layer = extract_train_hidden_cache(
        train_rows,
        tok=tok,
        model=model,
        input_device=input_device,
        answer_id_tensor=answer_id_tensor,
        max_seq_len=args.max_seq_len,
        num_layers=num_layers,
        batch_size=train_layer_extraction_batch_size,
    )

    train_is_correct_mask = train_clean_df["clean_is_correct"].to_numpy(dtype=bool)
    contrastive_directions: dict[int, torch.Tensor] = {}
    direction_metadata_rows: list[dict[str, object]] = []
    n_train_correct = int(train_is_correct_mask.sum())
    n_train_incorrect = int((~train_is_correct_mask).sum())

    for layer_number in sorted(train_hidden_by_layer.keys()):
        direction, info = build_contrastive_mean_direction(train_hidden_by_layer[layer_number].to(torch.float32), train_is_correct_mask)
        contrastive_directions[layer_number] = direction
        direction_metadata_rows.append(
            {
                "method": str(info["method"]),
                "raw_direction_norm": float(info["raw_direction_norm"]),
                "probe_train_accuracy": np.nan,
                "layer_number": int(layer_number),
                "n_train_correct_used": n_train_correct,
                "n_train_incorrect_used": n_train_incorrect,
            }
        )
        del train_hidden_by_layer[layer_number]
        gc.collect()

    direction_metadata_df = pd.DataFrame(direction_metadata_rows)

    steering_results_df = run_contrastive_scale_sweep(
        target_rows,
        tok=tok,
        model=model,
        input_device=input_device,
        decoder_layers=decoder_layers,
        answer_id_tensor=answer_id_tensor,
        max_seq_len=args.max_seq_len,
        batch_size=steering_batch_size,
        steering_scales=steering_scales,
        contrastive_directions=contrastive_directions,
    )

    by_config_df, by_scale_df = summarize_run(clean_baseline_df, steering_results_df)

    output_dir = resolve_out_dir(args.out_dir, args.model_id)
    tmp_dir = output_dir.parent / f"_tmp_{output_dir.name}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=False)

    clean_baseline_df.to_parquet(tmp_dir / "steering_baseline.parquet", index=False)
    steering_results_df.to_parquet(tmp_dir / "steering_results.parquet", index=False)
    direction_metadata_df.to_parquet(tmp_dir / "steering_direction_metadata.parquet", index=False)

    run_config = {
        "model_id": args.model_id,
        "train_split": train_split,
        "target_split": target_split,
        "max_seq_len": int(args.max_seq_len),
        "seed": int(args.seed),
        "train_layer_extraction_batch_size": train_layer_extraction_batch_size,
        "clean_baseline_batch_size": clean_baseline_batch_size,
        "steering_batch_size": steering_batch_size,
        "steering_scales": steering_scales,
        "methods": ["contrastive_mean_direction"],
        "answer_token_ids": answer_token_ids,
        "num_train_examples": int(len(train_rows)),
        "num_target_examples": int(len(target_rows)),
        "num_layers": int(num_layers),
        "hidden_size": int(hidden_size),
        "model_dtype": str(model_dtype),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary_note": "Printed summaries are derived from raw outputs and are not saved as official analysis artifacts.",
    }
    with (tmp_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    shutil.move(tmp_dir.as_posix(), output_dir.as_posix())

    print("\nPER-SCALE SUMMARY")
    print(by_scale_df.round(4).to_string(index=False))
    print("\nTOP CONFIGS")
    print(
        by_config_df.sort_values(["net_gain_count", "n_rescued", "n_harmed"], ascending=[False, False, True])
        .head(20)
        .round(4)
        .to_string(index=False)
    )
    print(f"\n[done] wrote {output_dir}")


if __name__ == "__main__":
    main()
