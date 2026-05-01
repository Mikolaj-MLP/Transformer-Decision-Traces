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
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.load_csqa import load_csqa


LETTERS = ["A", "B", "C", "D", "E"]


def now_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def repo_root() -> Path:
    return REPO_ROOT


def slugify_model_id(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", model_id).strip("-")


def resolve_out_dir(out_dir: str | None, model_id: str) -> Path:
    root = repo_root()
    if out_dir is None:
        run_name = f"{now_id()}_{slugify_model_id(model_id)}_csqa_steering_all_traces"
        return root / "data" / "generated" / "csqa_steering_all_traces" / run_name
    path = Path(out_dir)
    return path if path.is_absolute() else (root / path)


def get_decoder_layers(model: AutoModelForCausalLM):
    candidates = [
        "model.layers",
        "transformer.h",
        "gpt_neox.layers",
    ]
    for path in candidates:
        current = model
        ok = True
        for part in path.split("."):
            if not hasattr(current, part):
                ok = False
                break
            current = getattr(current, part)
        if ok:
            return current
    raise ValueError("Could not locate decoder layers on this model.")


def build_answer_token_ids(tok: AutoTokenizer) -> dict[str, int]:
    answer_token_ids: dict[str, int] = {}
    for letter in LETTERS:
        ids = tok(" " + letter, add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            raise ValueError(f"Answer token '{letter}' is not single-token: {ids}")
        answer_token_ids[letter] = int(ids[0])
    return answer_token_ids


def encode_prompts(texts: list[str], tok: AutoTokenizer, max_seq_len: int) -> dict[str, torch.Tensor]:
    batch = tok(
        list(texts),
        add_special_tokens=False,
        truncation=True,
        max_length=max_seq_len,
        padding=True,
        return_tensors="pt",
    )
    decision_pos = []
    for mask in batch["attention_mask"]:
        nz = torch.nonzero(mask, as_tuple=False).view(-1)
        decision_pos.append(int(nz[-1].item()))
    batch["decision_pos"] = torch.tensor(decision_pos, dtype=torch.long)
    return batch


def get_input_device(model: AutoModelForCausalLM) -> torch.device:
    for param in model.parameters():
        if param.device.type != "meta":
            return param.device
    return torch.device("cpu")


def unpack_output_hidden(output):
    if isinstance(output, tuple):
        return output[0]
    return output


def repack_output_hidden(output, new_hidden):
    if isinstance(output, tuple):
        return (new_hidden,) + tuple(output[1:])
    return new_hidden


def select_full_logits_at_decision(logits: torch.Tensor, decision_pos: torch.Tensor) -> torch.Tensor:
    row_idx = torch.arange(logits.shape[0], device=logits.device)
    return logits[row_idx, decision_pos].float()


def summarize_decision_logits(
    full_logits: torch.Tensor,
    true_choice_idx: torch.Tensor,
    answer_id_tensor: torch.Tensor,
) -> dict[str, torch.Tensor]:
    masked_logits = full_logits.clone()
    masked_logits[:, answer_id_tensor] = -torch.inf
    best_non_choice_logit = torch.max(masked_logits, dim=-1).values

    choice_logits = full_logits.index_select(1, answer_id_tensor)
    predicted_choice_idx = torch.argmax(choice_logits, dim=-1)
    is_correct = predicted_choice_idx.eq(true_choice_idx)

    return {
        "predicted_choice_idx": predicted_choice_idx,
        "is_correct": is_correct,
        "best_non_choice_logit": best_non_choice_logit,
        "choice_logits": choice_logits,
    }


def apply_token_steering(
    hidden: torch.Tensor,
    decision_pos: torch.Tensor,
    direction: torch.Tensor,
    steering_scale: float,
) -> torch.Tensor:
    row_idx = torch.arange(hidden.shape[0], device=hidden.device)
    token_hidden = hidden[row_idx, decision_pos]
    rms = token_hidden.float().pow(2).mean(dim=-1, keepdim=True).sqrt().to(token_hidden.dtype)
    direction = direction.to(hidden.device, dtype=hidden.dtype)
    hidden_out = hidden.clone()
    hidden_out[row_idx, decision_pos] = token_hidden + (steering_scale * rms) * direction.unsqueeze(0)
    return hidden_out


def build_contrastive_mean_direction(
    hidden_cache: torch.Tensor,
    is_correct_mask: np.ndarray,
) -> tuple[torch.Tensor, dict[str, object]]:
    mask = torch.as_tensor(is_correct_mask, dtype=torch.bool)
    if bool(mask.all()) or bool((~mask).all()):
        raise ValueError("Train split must contain both correct and incorrect examples.")

    correct_mean = hidden_cache[mask].mean(dim=0)
    incorrect_mean = hidden_cache[~mask].mean(dim=0)
    raw_direction = (correct_mean - incorrect_mean).to(torch.float32)
    raw_norm = float(raw_direction.norm().item())
    direction = raw_direction / raw_direction.norm().clamp_min(1e-12)

    return direction.cpu(), {
        "method": "contrastive_mean_direction",
        "raw_direction_norm": raw_norm,
        "probe_train_accuracy": np.nan,
    }


def build_probe_normal_direction(
    hidden_cache: torch.Tensor,
    is_correct_mask: np.ndarray,
    train_epochs: int,
    learning_rate: float,
    weight_decay: float,
) -> tuple[torch.Tensor, dict[str, object]]:
    mask = torch.as_tensor(is_correct_mask, dtype=torch.bool)
    if bool(mask.all()) or bool((~mask).all()):
        raise ValueError("Train split must contain both correct and incorrect examples.")

    x = hidden_cache.to(torch.float32)
    y = mask.to(torch.float32).unsqueeze(-1)
    mu = x.mean(dim=0)
    sigma = x.std(dim=0).clamp_min(1e-6)
    xz = (x - mu) / sigma

    probe = nn.Linear(xz.shape[1], 1)
    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    for _ in tqdm(range(train_epochs), desc="probe training", leave=False):
        optimizer.zero_grad(set_to_none=True)
        logits = probe(xz)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        optimizer.step()

    with torch.inference_mode():
        logits = probe(xz)
        preds = logits.squeeze(-1).ge(0.0)
        train_accuracy = float(preds.eq(mask).float().mean().item())
        raw_weight = probe.weight.detach().cpu().squeeze(0).to(torch.float32)

    raw_direction = raw_weight / sigma.cpu()
    projection = x @ raw_direction
    if projection[mask].mean() < projection[~mask].mean():
        raw_direction = -raw_direction

    raw_norm = float(raw_direction.norm().item())
    direction = raw_direction / raw_direction.norm().clamp_min(1e-12)

    return direction.cpu(), {
        "method": "probe_normal_direction",
        "raw_direction_norm": raw_norm,
        "probe_train_accuracy": train_accuracy,
    }


def extract_clean_baseline(
    frame: pd.DataFrame,
    *,
    split_name: str,
    tok: AutoTokenizer,
    model: AutoModelForCausalLM,
    input_device: torch.device,
    answer_id_tensor: torch.Tensor,
    max_seq_len: int,
    batch_size: int,
) -> pd.DataFrame:
    baseline_rows: list[dict[str, object]] = []

    for start in tqdm(
        range(0, len(frame), batch_size),
        total=int(math.ceil(len(frame) / batch_size)),
        desc=f"{split_name} clean baseline",
    ):
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

        with torch.inference_mode():
            out = model(**batch, return_dict=True, use_cache=False)

        full_logits = select_full_logits_at_decision(out.logits, decision_pos)
        metrics = summarize_decision_logits(full_logits, true_choice_idx, answer_id_tensor.to(full_logits.device))

        choice_logits_cpu = metrics["choice_logits"].detach().cpu().numpy().astype(np.float32)
        predicted_choice_idx_cpu = metrics["predicted_choice_idx"].detach().cpu().numpy().astype(np.int64)
        is_correct_cpu = metrics["is_correct"].detach().cpu().numpy()
        best_non_choice_logit_cpu = metrics["best_non_choice_logit"].detach().cpu().numpy().astype(np.float32)

        for batch_index, row in batch_df.iterrows():
            pred_idx = int(predicted_choice_idx_cpu[batch_index])
            true_idx = int(true_choice_idx[batch_index].item())
            baseline_rows.append(
                {
                    "example_id": row["example_id"],
                    "split": split_name,
                    "true_choice_idx": true_idx,
                    "true_choice_letter": LETTERS[true_idx],
                    "clean_predicted_choice_idx": pred_idx,
                    "clean_predicted_choice_letter": LETTERS[pred_idx],
                    "clean_is_correct": bool(is_correct_cpu[batch_index]),
                    "clean_best_non_choice_logit": float(best_non_choice_logit_cpu[batch_index]),
                    "clean_logit_A": float(choice_logits_cpu[batch_index, 0]),
                    "clean_logit_B": float(choice_logits_cpu[batch_index, 1]),
                    "clean_logit_C": float(choice_logits_cpu[batch_index, 2]),
                    "clean_logit_D": float(choice_logits_cpu[batch_index, 3]),
                    "clean_logit_E": float(choice_logits_cpu[batch_index, 4]),
                }
            )

    return pd.DataFrame(baseline_rows)


def extract_train_hidden_cache(
    frame: pd.DataFrame,
    *,
    tok: AutoTokenizer,
    model: AutoModelForCausalLM,
    input_device: torch.device,
    answer_id_tensor: torch.Tensor,
    max_seq_len: int,
    num_layers: int,
    batch_size: int,
) -> tuple[pd.DataFrame, dict[int, torch.Tensor]]:
    clean_rows: list[dict[str, object]] = []
    hidden_blocks_by_layer: dict[int, list[torch.Tensor]] = {layer_number: [] for layer_number in range(1, num_layers + 1)}

    for start in tqdm(
        range(0, len(frame), batch_size),
        total=int(math.ceil(len(frame) / batch_size)),
        desc="train layer extraction",
    ):
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

        with torch.inference_mode():
            out = model(**batch, return_dict=True, use_cache=False, output_hidden_states=True)

        full_logits = select_full_logits_at_decision(out.logits, decision_pos)
        metrics = summarize_decision_logits(full_logits, true_choice_idx, answer_id_tensor.to(full_logits.device))
        row_idx = torch.arange(len(batch_df), device=decision_pos.device)

        for layer_number in range(1, num_layers + 1):
            hidden = out.hidden_states[layer_number][row_idx, decision_pos].detach().cpu().to(torch.float16)
            hidden_blocks_by_layer[layer_number].append(hidden)

        is_correct_cpu = metrics["is_correct"].detach().cpu().numpy()
        for batch_index, row in batch_df.iterrows():
            clean_rows.append(
                {
                    "example_id": row["example_id"],
                    "clean_is_correct": bool(is_correct_cpu[batch_index]),
                }
            )

    hidden_by_layer = {
        layer_number: torch.cat(blocks, dim=0)
        for layer_number, blocks in hidden_blocks_by_layer.items()
    }
    return pd.DataFrame(clean_rows), hidden_by_layer


def build_steering_directions(
    hidden_by_layer: dict[int, torch.Tensor],
    train_is_correct_mask: np.ndarray,
    *,
    probe_train_epochs: int,
    probe_train_learning_rate: float,
    probe_train_weight_decay: float,
) -> tuple[dict[tuple[int, str], torch.Tensor], pd.DataFrame]:
    steering_directions: dict[tuple[int, str], torch.Tensor] = {}
    metadata_rows: list[dict[str, object]] = []

    n_train_correct = int(train_is_correct_mask.sum())
    n_train_incorrect = int((~train_is_correct_mask).sum())

    for layer_number in tqdm(sorted(hidden_by_layer.keys()), desc="direction construction"):
        hidden_cache = hidden_by_layer[layer_number].to(torch.float32)

        contrastive_direction, contrastive_info = build_contrastive_mean_direction(
            hidden_cache,
            train_is_correct_mask,
        )
        contrastive_info["layer_number"] = layer_number
        contrastive_info["n_train_correct_used"] = n_train_correct
        contrastive_info["n_train_incorrect_used"] = n_train_incorrect
        steering_directions[(layer_number, str(contrastive_info["method"]))] = contrastive_direction
        metadata_rows.append(contrastive_info)

        probe_direction, probe_info = build_probe_normal_direction(
            hidden_cache,
            train_is_correct_mask,
            train_epochs=probe_train_epochs,
            learning_rate=probe_train_learning_rate,
            weight_decay=probe_train_weight_decay,
        )
        probe_info["layer_number"] = layer_number
        probe_info["n_train_correct_used"] = n_train_correct
        probe_info["n_train_incorrect_used"] = n_train_incorrect
        steering_directions[(layer_number, str(probe_info["method"]))] = probe_direction
        metadata_rows.append(probe_info)

        del hidden_by_layer[layer_number]
        gc.collect()

    return steering_directions, pd.DataFrame(metadata_rows)


def run_steering_scan(
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
    steering_directions: dict[tuple[int, str], torch.Tensor],
    baseline_lookup: dict[str, dict[str, object]],
) -> pd.DataFrame:
    result_rows: list[dict[str, object]] = []
    method_names = ["contrastive_mean_direction", "probe_normal_direction"]
    layer_numbers = list(range(1, len(decoder_layers) + 1))

    for layer_number in tqdm(layer_numbers, desc="steering scan"):
        steering_module = decoder_layers[layer_number - 1]

        for method_name in method_names:
            direction = steering_directions[(layer_number, method_name)]

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

                    def steering_hook(module, inputs, output):
                        hidden = unpack_output_hidden(output)
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
                    predicted_choice_idx_cpu = metrics["predicted_choice_idx"].detach().cpu().numpy().astype(np.int64)
                    is_correct_cpu = metrics["is_correct"].detach().cpu().numpy()
                    best_non_choice_logit_cpu = metrics["best_non_choice_logit"].detach().cpu().numpy().astype(np.float32)

                    for batch_index, row in batch_df.iterrows():
                        example_id = row["example_id"]
                        clean = baseline_lookup[example_id]
                        pred_idx = int(predicted_choice_idx_cpu[batch_index])
                        steered_is_correct = bool(is_correct_cpu[batch_index])
                        clean_is_correct = bool(clean["clean_is_correct"])
                        clean_pred_idx = int(clean["clean_predicted_choice_idx"])

                        result_rows.append(
                            {
                                "example_id": example_id,
                                "method": method_name,
                                "layer_number": layer_number,
                                "scale": float(steering_scale),
                                "steered_predicted_choice_idx": pred_idx,
                                "steered_predicted_choice_letter": LETTERS[pred_idx],
                                "steered_is_correct": steered_is_correct,
                                "prediction_changed": bool(pred_idx != clean_pred_idx),
                                "rescued_error": bool((not clean_is_correct) and steered_is_correct),
                                "harmed_correct": bool(clean_is_correct and (not steered_is_correct)),
                                "steered_best_non_choice_logit": float(best_non_choice_logit_cpu[batch_index]),
                                "steered_logit_A": float(choice_logits_cpu[batch_index, 0]),
                                "steered_logit_B": float(choice_logits_cpu[batch_index, 1]),
                                "steered_logit_C": float(choice_logits_cpu[batch_index, 2]),
                                "steered_logit_D": float(choice_logits_cpu[batch_index, 3]),
                                "steered_logit_E": float(choice_logits_cpu[batch_index, 4]),
                            }
                        )

    return pd.DataFrame(result_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--max-seq-len", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_layer_extraction_batch_size = 4
    clean_baseline_batch_size = 4
    steering_batch_size = 2
    probe_train_epochs = 100
    probe_train_learning_rate = 5e-2
    probe_train_weight_decay = 1e-4
    steering_scales = [1.5, 3.0]
    train_split = "train"
    target_split = "validation"

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
    baseline_lookup = clean_baseline_df.set_index("example_id").to_dict("index")

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
    steering_directions, steering_direction_metadata_df = build_steering_directions(
        train_hidden_by_layer,
        train_is_correct_mask,
        probe_train_epochs=probe_train_epochs,
        probe_train_learning_rate=probe_train_learning_rate,
        probe_train_weight_decay=probe_train_weight_decay,
    )

    steering_results_df = run_steering_scan(
        target_rows,
        tok=tok,
        model=model,
        input_device=input_device,
        decoder_layers=decoder_layers,
        answer_id_tensor=answer_id_tensor,
        max_seq_len=args.max_seq_len,
        batch_size=steering_batch_size,
        steering_scales=steering_scales,
        steering_directions=steering_directions,
        baseline_lookup=baseline_lookup,
    )

    output_dir = resolve_out_dir(args.out_dir, args.model_id)
    tmp_dir = output_dir.parent / f"_tmp_{output_dir.name}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=False)

    clean_baseline_df.to_parquet(tmp_dir / "steering_baseline.parquet", index=False)
    steering_results_df.to_parquet(tmp_dir / "steering_results.parquet", index=False)
    steering_direction_metadata_df.to_parquet(tmp_dir / "steering_direction_metadata.parquet", index=False)

    run_config = {
        "model_id": args.model_id,
        "train_split": train_split,
        "target_split": target_split,
        "max_seq_len": int(args.max_seq_len),
        "seed": int(args.seed),
        "train_layer_extraction_batch_size": train_layer_extraction_batch_size,
        "clean_baseline_batch_size": clean_baseline_batch_size,
        "steering_batch_size": steering_batch_size,
        "probe_train_epochs": probe_train_epochs,
        "probe_train_learning_rate": probe_train_learning_rate,
        "probe_train_weight_decay": probe_train_weight_decay,
        "steering_scales": steering_scales,
        "methods": ["contrastive_mean_direction", "probe_normal_direction"],
        "answer_token_ids": answer_token_ids,
        "num_train_examples": int(len(train_rows)),
        "num_target_examples": int(len(target_rows)),
        "num_layers": int(num_layers),
        "hidden_size": int(hidden_size),
        "model_dtype": str(model_dtype),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with (tmp_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    shutil.move(tmp_dir.as_posix(), output_dir.as_posix())
    print(f"[done] wrote {output_dir}")


if __name__ == "__main__":
    main()
