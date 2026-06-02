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
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.csqa.common import (  # noqa: E402
    AffineTranslator,
    build_answer_token_ids,
    build_choice_token_spans,
    build_contrastive_mean_direction,
    encode_prompts,
    get_decoder_layers,
    get_final_norm_module,
    repack_output_hidden,
    select_full_logits_at_decision,
    summarize_decision_logits,
    unpack_output_hidden,
)
from src.data.load_csqa import load_csqa  # noqa: E402


LETTERS = ["A", "B", "C", "D", "E"]
EXTRACT_BATCH_SIZE = 4
ATTENTION_BATCH_SIZE = 1
TUNED_LENS_BATCH_SIZE = 64
TUNED_LENS_MAX_EPOCHS = 10
TUNED_LENS_PATIENCE = 2
TUNED_LENS_LR = 1e-3
TUNED_LENS_WEIGHT_DECAY = 1e-5
STEERING_BATCH_SIZE = 2
DETECTOR_C_GRID = np.logspace(-3, 2, 12)
TARGET_RULES = ["midpoint_mean", "median_correct", "p75_correct", "mean_correct"]


def now_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def repo_root() -> Path:
    return REPO_ROOT


def slugify_model_id(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", model_id).strip("-")


def resolve_out_dir(out_dir: str | None, model_id: str) -> Path:
    root = repo_root()
    if out_dir is None:
        run_name = f"{now_id()}_{slugify_model_id(model_id)}_csqa_adaptive_contrastive_pipeline"
        return root / "data" / "generated" / "csqa_adaptive_contrastive_pipeline" / run_name
    path = Path(out_dir)
    return path if path.is_absolute() else (root / path)


def get_input_device(model: AutoModelForCausalLM) -> torch.device:
    for param in model.parameters():
        if param.device.type != "meta":
            return param.device
    return torch.device("cpu")


def choose_model_dtype_and_device_map() -> tuple[torch.dtype, str | None]:
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16, "auto"
        return torch.float16, "auto"
    return torch.float32, None


def maybe_clone_to_float_cpu(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy().astype(np.float32)


def build_policy_target(target_rule: str, correct_projection: np.ndarray, incorrect_projection: np.ndarray) -> float:
    if target_rule == "midpoint_mean":
        return float(0.5 * (correct_projection.mean() + incorrect_projection.mean()))
    if target_rule == "median_correct":
        return float(np.median(correct_projection))
    if target_rule == "p75_correct":
        return float(np.quantile(correct_projection, 0.75))
    if target_rule == "mean_correct":
        return float(correct_projection.mean())
    raise ValueError(f"Unknown target_rule: {target_rule}")


def choice_logits_to_pred_idx(choice_logits: np.ndarray) -> np.ndarray:
    return np.argmax(choice_logits, axis=1).astype(np.int64)


def choice_logits_to_entropy(choice_logits: np.ndarray) -> np.ndarray:
    probs = torch.softmax(torch.from_numpy(choice_logits), dim=1).numpy()
    log_probs = np.log(np.clip(probs, 1e-12, None))
    return -(probs * log_probs).sum(axis=1)


def choice_logits_to_gap(choice_logits: np.ndarray) -> np.ndarray:
    sorted_logits = np.sort(choice_logits, axis=1)[:, ::-1]
    return sorted_logits[:, 0] - sorted_logits[:, 1]


def choice_logits_and_best_non_choice_to_margin(choice_logits: np.ndarray, best_non_choice_logit: np.ndarray) -> np.ndarray:
    best_choice = np.max(choice_logits, axis=1)
    return best_choice - best_non_choice_logit


def extract_split_cache(
    frame: pd.DataFrame,
    *,
    split_name: str,
    tok: AutoTokenizer,
    model: AutoModelForCausalLM,
    input_device: torch.device,
    answer_id_tensor_cpu: torch.Tensor,
    answer_id_tensor_lm_head: torch.Tensor,
    answer_choice_weight: torch.Tensor,
    final_norm,
    num_layers: int,
    max_seq_len: int,
    batch_size: int,
    last_layer_needs_final_norm: bool,
) -> dict[str, object]:
    hidden_blocks: list[torch.Tensor] = []
    final_choice_prob_blocks: list[torch.Tensor] = []
    true_choice_idx_blocks: list[torch.Tensor] = []
    example_rows: list[dict[str, object]] = []
    clean_output_rows: list[dict[str, object]] = []
    clean_choice_logits_blocks: list[np.ndarray] = []
    clean_best_non_choice_logit_blocks: list[np.ndarray] = []
    clean_best_non_choice_token_id_blocks: list[np.ndarray] = []
    clean_is_correct_blocks: list[np.ndarray] = []

    def maybe_apply_final_norm(hidden: torch.Tensor, layer_index_0based: int) -> torch.Tensor:
        if final_norm is None:
            return hidden
        if layer_index_0based < (num_layers - 1):
            return final_norm(hidden)
        if last_layer_needs_final_norm:
            return final_norm(hidden)
        return hidden

    for start in tqdm(
        range(0, len(frame), batch_size),
        total=int(math.ceil(len(frame) / batch_size)),
        desc=f"{split_name} hidden extraction",
    ):
        batch_df = frame.iloc[start:start + batch_size].reset_index(drop=True)
        batch_cpu = encode_prompts(batch_df["text"].tolist(), tok, max_seq_len)
        decision_pos = batch_cpu.pop("decision_pos")
        prompt_token_count = batch_cpu.pop("prompt_token_count")
        batch = {k: v.to(input_device) for k, v in batch_cpu.items()}
        decision_pos = decision_pos.to(input_device)
        true_choice_idx = torch.tensor(batch_df["correct_idx"].tolist(), dtype=torch.long)

        with torch.inference_mode():
            out = model(**batch, output_hidden_states=True, return_dict=True, use_cache=False)

        row_idx = torch.arange(len(batch_df), device=input_device)
        per_layer_hidden = []
        for layer_index in range(num_layers):
            hidden = out.hidden_states[layer_index + 1][row_idx, decision_pos].detach().cpu().to(torch.float16)
            per_layer_hidden.append(hidden)

        final_raw = out.hidden_states[-1][row_idx, decision_pos]
        final_readout = maybe_apply_final_norm(final_raw, num_layers - 1)
        final_choice_logits = torch.matmul(
            final_readout.to(answer_choice_weight.dtype),
            answer_choice_weight.T,
        ).float()
        final_choice_probs = torch.softmax(final_choice_logits, dim=-1).detach().cpu()

        hidden_blocks.append(torch.stack(per_layer_hidden, dim=1))
        final_choice_prob_blocks.append(final_choice_probs)
        true_choice_idx_blocks.append(true_choice_idx)

        final_logits = out.logits[row_idx, decision_pos].float().detach().cpu()
        masked_logits = final_logits.clone()
        masked_logits[:, answer_id_tensor_cpu] = -torch.inf
        best_non_choice_logit, best_non_choice_token_id = torch.max(masked_logits, dim=-1)
        choice_logits = final_logits.index_select(1, answer_id_tensor_cpu)
        predicted_choice_idx = torch.argmax(choice_logits, dim=-1)
        is_correct = predicted_choice_idx.eq(true_choice_idx)

        clean_choice_logits_blocks.append(choice_logits.numpy().astype(np.float32))
        clean_best_non_choice_logit_blocks.append(best_non_choice_logit.numpy().astype(np.float32))
        clean_best_non_choice_token_id_blocks.append(best_non_choice_token_id.numpy().astype(np.int64))
        clean_is_correct_blocks.append(is_correct.numpy())

        for batch_index, row in batch_df.iterrows():
            example_rows.append(
                {
                    "example_id": row["example_id"],
                    "split": split_name,
                    "text": row["text"],
                    "answerKey": row["answerKey"],
                    "correct_idx": int(row["correct_idx"]),
                    "prompt_len_chars": int(row["prompt_len_chars"]),
                    "prompt_token_count": int(prompt_token_count[batch_index].item()),
                    "decision_pos": int(decision_pos[batch_index].item()),
                }
            )
            clean_output_rows.append(
                {
                    "example_id": row["example_id"],
                    "true_choice_idx": int(row["correct_idx"]),
                    "clean_logit_A": float(choice_logits[batch_index, 0].item()),
                    "clean_logit_B": float(choice_logits[batch_index, 1].item()),
                    "clean_logit_C": float(choice_logits[batch_index, 2].item()),
                    "clean_logit_D": float(choice_logits[batch_index, 3].item()),
                    "clean_logit_E": float(choice_logits[batch_index, 4].item()),
                    "best_non_choice_token_id": int(best_non_choice_token_id[batch_index].item()),
                    "best_non_choice_logit": float(best_non_choice_logit[batch_index].item()),
                }
            )

    return {
        "hidden": torch.cat(hidden_blocks, dim=0),
        "final_choice_probs": torch.cat(final_choice_prob_blocks, dim=0),
        "true_choice_idx": torch.cat(true_choice_idx_blocks, dim=0),
        "clean_choice_logits": np.concatenate(clean_choice_logits_blocks, axis=0),
        "clean_best_non_choice_logit": np.concatenate(clean_best_non_choice_logit_blocks, axis=0),
        "clean_best_non_choice_token_id": np.concatenate(clean_best_non_choice_token_id_blocks, axis=0),
        "clean_is_correct": np.concatenate(clean_is_correct_blocks, axis=0).astype(bool),
        "example_rows": example_rows,
        "clean_output_rows": clean_output_rows,
    }


def train_tuned_lenses(
    *,
    train_hidden: torch.Tensor,
    teacher_choice_probs_train: torch.Tensor,
    hidden_size: int,
    num_layers: int,
    answer_choice_weight: torch.Tensor,
    train_device: torch.device,
) -> tuple[list[AffineTranslator | None], pd.DataFrame]:
    lenses: list[AffineTranslator | None] = []
    train_history_rows: list[dict[str, object]] = []

    for layer_index in tqdm(range(num_layers), desc="tuned lens training"):
        if layer_index == (num_layers - 1):
            lenses.append(None)
            continue

        lens = AffineTranslator(hidden_size).to(train_device)
        optimizer = torch.optim.AdamW(
            lens.parameters(),
            lr=TUNED_LENS_LR,
            weight_decay=TUNED_LENS_WEIGHT_DECAY,
        )
        dataset = TensorDataset(train_hidden[:, layer_index, :], teacher_choice_probs_train)
        dataloader = DataLoader(dataset, batch_size=TUNED_LENS_BATCH_SIZE, shuffle=True)

        best_loss = float("inf")
        epochs_without_improvement = 0

        for epoch in range(1, TUNED_LENS_MAX_EPOCHS + 1):
            epoch_losses: list[float] = []
            lens.train()
            for xb, teacher_probs_b in dataloader:
                xb = xb.to(train_device).float()
                teacher_probs_b = teacher_probs_b.to(train_device)
                pred_readout = lens(xb)
                pred_choice_logits = torch.matmul(
                    pred_readout.to(answer_choice_weight.dtype),
                    answer_choice_weight.T,
                ).float()
                pred_log_probs = torch.log_softmax(pred_choice_logits, dim=-1)
                loss = F.kl_div(pred_log_probs, teacher_probs_b, reduction="batchmean")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.item()))

            mean_epoch_loss = float(np.mean(epoch_losses))
            train_history_rows.append(
                {
                    "layer_number": layer_index + 1,
                    "epoch": epoch,
                    "mean_kl_loss": mean_epoch_loss,
                }
            )

            if mean_epoch_loss < (best_loss - 1e-6):
                best_loss = mean_epoch_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= TUNED_LENS_PATIENCE:
                break

        lens.eval()
        lenses.append(lens.cpu())
        del lens
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return lenses, pd.DataFrame(train_history_rows)


def build_attention_feature_table(
    frame: pd.DataFrame,
    *,
    split_name: str,
    tok: AutoTokenizer,
    model: AutoModelForCausalLM,
    input_device: torch.device,
    max_seq_len: int,
    num_layers: int,
) -> pd.DataFrame:
    choice_span_map = {
        row["example_id"]: build_choice_token_spans(
            text=row["text"],
            tok=tok,
            max_seq_len=max_seq_len,
        )
        for _, row in frame[["example_id", "text"]].iterrows()
    }

    attention_rows: list[dict[str, object]] = []
    for start in tqdm(
        range(0, len(frame), ATTENTION_BATCH_SIZE),
        total=int(math.ceil(len(frame) / ATTENTION_BATCH_SIZE)),
        desc=f"{split_name} attention features",
    ):
        batch_df = frame.iloc[start:start + ATTENTION_BATCH_SIZE].reset_index(drop=True)
        batch_cpu = encode_prompts(batch_df["text"].tolist(), tok, max_seq_len)
        decision_pos = batch_cpu.pop("decision_pos")
        prompt_token_count = batch_cpu.pop("prompt_token_count")
        padded_seq_len = int(batch_cpu["input_ids"].shape[1])
        batch = {k: v.to(input_device) for k, v in batch_cpu.items()}
        decision_pos = decision_pos.to(input_device)

        with torch.inference_mode():
            out = model(**batch, output_attentions=True, return_dict=True, use_cache=False)

        for batch_index, row in batch_df.iterrows():
            example_id = row["example_id"]
            valid_len = int(prompt_token_count[batch_index].item())
            pad_offset = padded_seq_len - valid_len
            pos = int(decision_pos[batch_index].item())
            choice_spans = choice_span_map[example_id]

            for layer_index in range(num_layers):
                attn_row_full = out.attentions[layer_index][batch_index, :, pos, :].float().detach().cpu().numpy()
                attn_row_valid = attn_row_full[:, pad_offset:pad_offset + valid_len]
                squared_mass = np.sum(attn_row_valid ** 2, axis=1)
                if valid_len > 1:
                    head_renyi2_entropy = -np.log(np.clip(squared_mass, 1e-12, None)) / math.log(valid_len)
                else:
                    head_renyi2_entropy = np.zeros(attn_row_full.shape[0], dtype=np.float32)

                choice_mass_by_head = np.zeros((attn_row_full.shape[0], len(LETTERS)), dtype=np.float32)
                for choice_index, (span_start, span_end) in enumerate(choice_spans):
                    abs_start = pad_offset + span_start
                    abs_end = pad_offset + span_end
                    if abs_end > abs_start:
                        choice_mass_by_head[:, choice_index] = attn_row_full[:, abs_start:abs_end].sum(axis=1)

                aggregated_choice_mass = choice_mass_by_head.sum(axis=0)
                choice_mass_sum = float(aggregated_choice_mass.sum())
                if choice_mass_sum > 0.0:
                    choice_probs = aggregated_choice_mass / choice_mass_sum
                else:
                    choice_probs = np.full(len(LETTERS), 1.0 / len(LETTERS), dtype=np.float32)
                sorted_choice_probs = np.sort(choice_probs)[::-1]
                choice_entropy = -np.sum(choice_probs * np.log(np.clip(choice_probs, 1e-12, None)))

                attention_rows.append(
                    {
                        "example_id": example_id,
                        "layer_number": layer_index + 1,
                        "mean_head_renyi2_entropy_normalized": float(np.mean(head_renyi2_entropy)),
                        "aggregated_choice_attention_entropy_normalized": float(choice_entropy / math.log(len(LETTERS))),
                        "aggregated_choice_attention_top1_top2_probability_gap": float(
                            sorted_choice_probs[0] - sorted_choice_probs[1]
                        ),
                    }
                )

    return pd.DataFrame(attention_rows)


def prepare_readout_context(
    *,
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    input_device: torch.device,
    num_layers: int,
    max_seq_len: int,
) -> dict[str, object]:
    final_norm = get_final_norm_module(model)
    lm_head_weight = model.lm_head.weight.detach()
    lm_head_device = lm_head_weight.device
    answer_token_ids = build_answer_token_ids(tok)
    answer_ids = [answer_token_ids[letter] for letter in LETTERS]
    answer_id_tensor_cpu = torch.tensor(answer_ids, dtype=torch.long)
    answer_id_tensor_lm_head = answer_id_tensor_cpu.to(lm_head_device)
    answer_choice_weight = lm_head_weight.index_select(0, answer_id_tensor_lm_head)

    probe_rows = load_csqa(split="validation", limit=1).copy()
    probe_cpu = encode_prompts(probe_rows["text"].tolist(), tok, max_seq_len)
    probe_pos = int(probe_cpu["decision_pos"][0].item())
    probe_batch = {
        k: v.to(input_device)
        for k, v in probe_cpu.items()
        if k not in ["decision_pos", "prompt_token_count"]
    }
    with torch.inference_mode():
        probe_out = model(**probe_batch, output_hidden_states=True, return_dict=True, use_cache=False)

    raw_last = probe_out.hidden_states[-1][0, probe_pos].float()
    target_choice_logits = probe_out.logits[0, probe_pos, answer_id_tensor_lm_head].float().detach().cpu()
    raw_choice_logits = torch.mv(answer_choice_weight.detach().float().cpu(), raw_last.detach().cpu())

    if final_norm is not None:
        normed_last = final_norm(raw_last.unsqueeze(0)).squeeze(0)
        normed_choice_logits = torch.mv(answer_choice_weight.detach().float().cpu(), normed_last.detach().cpu())
        raw_err = torch.mean(torch.abs(raw_choice_logits - target_choice_logits)).item()
        normed_err = torch.mean(torch.abs(normed_choice_logits - target_choice_logits)).item()
        last_layer_needs_final_norm = bool(normed_err < raw_err)
    else:
        last_layer_needs_final_norm = False

    def maybe_apply_final_norm(hidden: torch.Tensor, layer_index_0based: int) -> torch.Tensor:
        if final_norm is None:
            return hidden
        if layer_index_0based < (num_layers - 1):
            return final_norm(hidden)
        if last_layer_needs_final_norm:
            return final_norm(hidden)
        return hidden

    return {
        "final_norm": final_norm,
        "answer_id_tensor_cpu": answer_id_tensor_cpu,
        "answer_id_tensor_lm_head": answer_id_tensor_lm_head,
        "answer_choice_weight": answer_choice_weight,
        "lm_head_weight": lm_head_weight,
        "lm_head_device": lm_head_device,
        "last_layer_needs_final_norm": last_layer_needs_final_norm,
        "maybe_apply_final_norm": maybe_apply_final_norm,
    }


def summarize_hidden_readout(
    hidden_batch: torch.Tensor,
    *,
    layer_index_0based: int,
    true_choice_idx_batch: torch.Tensor,
    maybe_apply_final_norm,
    final_norm,
    lm_head_weight: torch.Tensor,
    answer_id_tensor_lm_head: torch.Tensor,
    vocab_size: int,
    always_apply_final_norm_if_available: bool,
) -> dict[str, np.ndarray]:
    readout = hidden_batch.float()
    if always_apply_final_norm_if_available:
        if final_norm is not None:
            readout = final_norm(readout)
    else:
        readout = maybe_apply_final_norm(readout, layer_index_0based)

    full_logits = torch.matmul(
        readout.to(lm_head_weight.dtype),
        lm_head_weight.T,
    ).float()
    full_log_probs = torch.log_softmax(full_logits, dim=-1)
    full_probs = torch.exp(full_log_probs)
    full_entropy = -(full_probs * full_log_probs).sum(dim=-1)

    masked_logits = full_logits.clone()
    masked_logits[:, answer_id_tensor_lm_head] = -torch.inf
    best_non_choice_logit = torch.max(masked_logits, dim=-1).values

    choice_logits = full_logits.index_select(1, answer_id_tensor_lm_head)
    choice_log_probs = torch.log_softmax(choice_logits, dim=-1)
    choice_probs = torch.exp(choice_log_probs)
    choice_entropy = -(choice_probs * choice_log_probs).sum(dim=-1)
    sorted_choice_logits = torch.sort(choice_logits, dim=-1, descending=True).values
    best_choice_logit = torch.max(choice_logits, dim=-1).values

    return {
        "best_non_choice_logit": maybe_clone_to_float_cpu(best_non_choice_logit),
        "best_choice_minus_best_non_choice_logit": maybe_clone_to_float_cpu(best_choice_logit - best_non_choice_logit),
        "full_vocab_entropy_normalized": maybe_clone_to_float_cpu(full_entropy / math.log(vocab_size)),
        "answer_choice_entropy_normalized": maybe_clone_to_float_cpu(choice_entropy / math.log(len(LETTERS))),
        "answer_choice_top1_top2_logit_gap": maybe_clone_to_float_cpu(sorted_choice_logits[:, 0] - sorted_choice_logits[:, 1]),
    }


def build_detector_feature_table(
    *,
    split_name: str,
    cache: dict[str, object],
    attention_df: pd.DataFrame,
    lenses: list[AffineTranslator | None],
    final_norm,
    maybe_apply_final_norm,
    lm_head_weight: torch.Tensor,
    answer_id_tensor_lm_head: torch.Tensor,
    input_device: torch.device,
    vocab_size: int,
) -> pd.DataFrame:
    hidden = cache["hidden"]
    true_choice_idx = cache["true_choice_idx"]
    clean_is_correct = cache["clean_is_correct"]
    example_rows = pd.DataFrame(cache["example_rows"])[["example_id", "split"]]
    layer_rows: list[dict[str, object]] = []

    attention_lookup = attention_df.set_index(["example_id", "layer_number"])

    for layer_index in tqdm(range(hidden.shape[1]), desc=f"{split_name} detector features"):
        layer_hidden = hidden[:, layer_index, :]
        true_choice_idx_layer = true_choice_idx.clone()

        direct_metrics = summarize_hidden_readout(
            layer_hidden.to(input_device),
            layer_index_0based=layer_index,
            true_choice_idx_batch=true_choice_idx_layer.to(input_device),
            maybe_apply_final_norm=maybe_apply_final_norm,
            final_norm=final_norm,
            lm_head_weight=lm_head_weight,
            answer_id_tensor_lm_head=answer_id_tensor_lm_head,
            vocab_size=vocab_size,
            always_apply_final_norm_if_available=False,
        )

        lens = lenses[layer_index]
        if lens is None:
            tuned_hidden = layer_hidden.to(input_device).float()
        else:
            lens = lens.to(input_device)
            with torch.inference_mode():
                tuned_hidden = lens(layer_hidden.to(input_device).float())
            lens = lens.cpu()

        tuned_metrics = summarize_hidden_readout(
            tuned_hidden,
            layer_index_0based=layer_index,
            true_choice_idx_batch=true_choice_idx_layer.to(input_device),
            maybe_apply_final_norm=maybe_apply_final_norm,
            final_norm=final_norm,
            lm_head_weight=lm_head_weight,
            answer_id_tensor_lm_head=answer_id_tensor_lm_head,
            vocab_size=vocab_size,
            always_apply_final_norm_if_available=True,
        )

        for example_index, example_id in enumerate(example_rows["example_id"].tolist()):
            attn = attention_lookup.loc[(example_id, layer_index + 1)]
            layer_rows.append(
                {
                    "example_id": example_id,
                    "split": split_name,
                    "layer_number": layer_index + 1,
                    "final_error": int(not bool(clean_is_correct[example_index])),
                    "clean_is_correct": bool(clean_is_correct[example_index]),
                    "direct__best_non_choice_logit": float(direct_metrics["best_non_choice_logit"][example_index]),
                    "direct__best_choice_minus_best_non_choice_logit": float(
                        direct_metrics["best_choice_minus_best_non_choice_logit"][example_index]
                    ),
                    "direct__full_vocab_entropy_normalized": float(
                        direct_metrics["full_vocab_entropy_normalized"][example_index]
                    ),
                    "direct__answer_choice_entropy_normalized": float(
                        direct_metrics["answer_choice_entropy_normalized"][example_index]
                    ),
                    "direct__answer_choice_top1_top2_logit_gap": float(
                        direct_metrics["answer_choice_top1_top2_logit_gap"][example_index]
                    ),
                    "tuned__best_non_choice_logit": float(tuned_metrics["best_non_choice_logit"][example_index]),
                    "tuned__best_choice_minus_best_non_choice_logit": float(
                        tuned_metrics["best_choice_minus_best_non_choice_logit"][example_index]
                    ),
                    "tuned__full_vocab_entropy_normalized": float(
                        tuned_metrics["full_vocab_entropy_normalized"][example_index]
                    ),
                    "tuned__answer_choice_entropy_normalized": float(
                        tuned_metrics["answer_choice_entropy_normalized"][example_index]
                    ),
                    "tuned__answer_choice_top1_top2_logit_gap": float(
                        tuned_metrics["answer_choice_top1_top2_logit_gap"][example_index]
                    ),
                    "attention__mean_head_renyi2_entropy_normalized": float(
                        attn["mean_head_renyi2_entropy_normalized"]
                    ),
                    "attention__aggregated_choice_attention_entropy_normalized": float(
                        attn["aggregated_choice_attention_entropy_normalized"]
                    ),
                    "attention__aggregated_choice_attention_top1_top2_probability_gap": float(
                        attn["aggregated_choice_attention_top1_top2_probability_gap"]
                    ),
                }
            )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return pd.DataFrame(layer_rows)


def fit_layer_detectors(
    *,
    train_feature_df: pd.DataFrame,
    validation_feature_df: pd.DataFrame,
) -> tuple[dict[int, object], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_cols = [
        "direct__best_non_choice_logit",
        "direct__best_choice_minus_best_non_choice_logit",
        "direct__full_vocab_entropy_normalized",
        "direct__answer_choice_entropy_normalized",
        "direct__answer_choice_top1_top2_logit_gap",
        "tuned__best_non_choice_logit",
        "tuned__best_choice_minus_best_non_choice_logit",
        "tuned__full_vocab_entropy_normalized",
        "tuned__answer_choice_entropy_normalized",
        "tuned__answer_choice_top1_top2_logit_gap",
        "attention__mean_head_renyi2_entropy_normalized",
        "attention__aggregated_choice_attention_entropy_normalized",
        "attention__aggregated_choice_attention_top1_top2_probability_gap",
    ]

    detector_models: dict[int, object] = {}
    coefficient_rows: list[dict[str, object]] = []
    layer_summary_rows: list[dict[str, object]] = []
    detector_output_rows: list[dict[str, object]] = []

    for layer_number in tqdm(sorted(train_feature_df["layer_number"].unique()), desc="detector training"):
        train_part = train_feature_df.loc[train_feature_df["layer_number"].eq(layer_number)].copy()
        validation_part = validation_feature_df.loc[validation_feature_df["layer_number"].eq(layer_number)].copy()

        X_train = train_part[feature_cols].to_numpy(dtype=float)
        y_train = train_part["final_error"].to_numpy(dtype=int)
        X_validation = validation_part[feature_cols].to_numpy(dtype=float)
        y_validation = validation_part["final_error"].to_numpy(dtype=int)

        pipe = make_pipeline(
            StandardScaler(),
            LogisticRegressionCV(
                Cs=DETECTOR_C_GRID,
                cv=5,
                penalty="l1",
                solver="saga",
                scoring="average_precision",
                max_iter=5000,
                n_jobs=-1,
                random_state=42,
                refit=True,
            ),
        )
        pipe.fit(X_train, y_train)
        detector_models[int(layer_number)] = pipe

        scaler = pipe.named_steps["standardscaler"]
        model = pipe.named_steps["logisticregressioncv"]
        selected_c = float(np.ravel(model.C_)[0])

        train_prob = pipe.predict_proba(X_train)[:, 1]
        validation_prob = pipe.predict_proba(X_validation)[:, 1]
        train_pred = (train_prob >= 0.5).astype(int)
        validation_pred = (validation_prob >= 0.5).astype(int)

        train_flag_rate = float(train_pred.mean())
        validation_flag_rate = float(validation_pred.mean())
        validation_tp = int(np.sum((validation_pred == 1) & (y_validation == 1)))
        validation_fp = int(np.sum((validation_pred == 1) & (y_validation == 0)))
        validation_tn = int(np.sum((validation_pred == 0) & (y_validation == 0)))
        validation_fn = int(np.sum((validation_pred == 0) & (y_validation == 1)))

        validation_precision = float(
            validation_tp / max(validation_tp + validation_fp, 1)
        )
        validation_recall = float(
            validation_tp / max(validation_tp + validation_fn, 1)
        )
        validation_fpr = float(
            validation_fp / max(validation_fp + validation_tn, 1)
        )

        layer_summary_rows.append(
            {
                "layer_number": int(layer_number),
                "selected_c": selected_c,
                "train_error_rate": float(y_train.mean()),
                "validation_error_rate": float(y_validation.mean()),
                "train_roc_auc_error": float(roc_auc_score(y_train, train_prob)),
                "train_pr_auc_error": float(average_precision_score(y_train, train_prob)),
                "validation_roc_auc_error": float(roc_auc_score(y_validation, validation_prob)),
                "validation_pr_auc_error": float(average_precision_score(y_validation, validation_prob)),
                "train_flag_rate": train_flag_rate,
                "validation_flag_rate": validation_flag_rate,
                "validation_precision": validation_precision,
                "validation_recall": validation_recall,
                "validation_false_positive_rate": validation_fpr,
                "validation_tp": validation_tp,
                "validation_fp": validation_fp,
                "validation_tn": validation_tn,
                "validation_fn": validation_fn,
                "nonzero_feature_count": int(np.count_nonzero(model.coef_)),
            }
        )

        for feature_name, coefficient in zip(feature_cols, model.coef_.ravel()):
            coefficient_rows.append(
                {
                    "layer_number": int(layer_number),
                    "feature": feature_name,
                    "coefficient": float(coefficient),
                    "abs_coefficient": float(abs(coefficient)),
                    "is_nonzero": bool(coefficient != 0.0),
                    "selected_c": selected_c,
                    "intercept": float(model.intercept_[0]),
                    "feature_mean": float(scaler.mean_[feature_cols.index(feature_name)]),
                    "feature_scale": float(scaler.scale_[feature_cols.index(feature_name)]),
                }
            )

        for part_name, part, prob, pred in [
            ("train", train_part, train_prob, train_pred),
            ("validation", validation_part, validation_prob, validation_pred),
        ]:
            for row_index, example_id in enumerate(part["example_id"].tolist()):
                detector_output_rows.append(
                    {
                        "split": part_name,
                        "example_id": example_id,
                        "layer_number": int(layer_number),
                        "final_error": int(part["final_error"].iloc[row_index]),
                        "clean_is_correct": bool(part["clean_is_correct"].iloc[row_index]),
                        "detector_probability_error": float(prob[row_index]),
                        "detector_predicted_error": bool(pred[row_index]),
                    }
                )

    return (
        detector_models,
        pd.DataFrame(coefficient_rows),
        pd.DataFrame(layer_summary_rows),
        pd.DataFrame(detector_output_rows),
    )


def build_scale_rule_stats(
    *,
    train_hidden: torch.Tensor,
    validation_hidden: torch.Tensor,
    train_clean_is_correct: np.ndarray,
    cap_quantile: float,
) -> tuple[dict[int, torch.Tensor], pd.DataFrame, dict[int, np.ndarray], dict[int, np.ndarray]]:
    directions: dict[int, torch.Tensor] = {}
    stats_rows: list[dict[str, object]] = []
    train_projection_by_layer: dict[int, np.ndarray] = {}
    validation_projection_by_layer: dict[int, np.ndarray] = {}

    for layer_number in tqdm(range(1, train_hidden.shape[1] + 1), desc="scale-rule construction"):
        layer_hidden_train = train_hidden[:, layer_number - 1, :].to(torch.float32)
        direction, direction_info = build_contrastive_mean_direction(
            layer_hidden_train,
            train_clean_is_correct,
        )
        directions[layer_number] = direction

        train_projection = torch.matmul(layer_hidden_train, direction.to(layer_hidden_train.device)).detach().cpu().numpy()
        validation_projection = torch.matmul(
            validation_hidden[:, layer_number - 1, :].to(torch.float32),
            direction.to(validation_hidden.device if validation_hidden.is_cuda else torch.device("cpu")).to(torch.float32),
        ).detach().cpu().numpy()
        train_rms = (
            layer_hidden_train.pow(2).mean(dim=-1).sqrt().detach().cpu().numpy().astype(np.float32)
        )

        train_projection_by_layer[layer_number] = train_projection.astype(np.float32)
        validation_projection_by_layer[layer_number] = validation_projection.astype(np.float32)

        correct_projection = train_projection[train_clean_is_correct]
        incorrect_projection = train_projection[~train_clean_is_correct]
        incorrect_rms = train_rms[~train_clean_is_correct]

        for target_rule in TARGET_RULES:
            target_projection = build_policy_target(target_rule, correct_projection, incorrect_projection)
            required_scale_train_raw = np.maximum(
                0.0,
                (target_projection - incorrect_projection) / np.clip(incorrect_rms, 1e-12, None),
            )
            positive_required = required_scale_train_raw[required_scale_train_raw > 0.0]
            if positive_required.size == 0:
                scale_cap = 0.0
            else:
                scale_cap = float(np.quantile(positive_required, cap_quantile))

            stats_rows.append(
                {
                    "layer_number": layer_number,
                    "target_rule": target_rule,
                    "target_projection": float(target_projection),
                    "scale_cap_quantile": float(cap_quantile),
                    "scale_cap": float(scale_cap),
                    "direction_l2_norm": float(direction_info["raw_direction_norm"]),
                    "correct_projection_mean": float(correct_projection.mean()),
                    "incorrect_projection_mean": float(incorrect_projection.mean()),
                    "correct_projection_median": float(np.median(correct_projection)),
                    "correct_projection_p75": float(np.quantile(correct_projection, 0.75)),
                    "required_scale_train_mean": float(required_scale_train_raw.mean()),
                    "required_scale_train_median": float(np.median(required_scale_train_raw)),
                    "required_scale_train_p95": float(np.quantile(required_scale_train_raw, 0.95)),
                }
            )

        del layer_hidden_train
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return directions, pd.DataFrame(stats_rows), train_projection_by_layer, validation_projection_by_layer


def apply_token_steering_per_example(
    hidden: torch.Tensor,
    decision_pos: torch.Tensor,
    direction: torch.Tensor,
    steering_scales: torch.Tensor,
) -> torch.Tensor:
    row_idx = torch.arange(hidden.shape[0], device=hidden.device)
    token_hidden = hidden[row_idx, decision_pos]
    rms = token_hidden.float().pow(2).mean(dim=-1, keepdim=True).sqrt().to(token_hidden.dtype)
    direction = direction.to(hidden.device, dtype=hidden.dtype)
    scale_term = steering_scales.to(hidden.device, dtype=hidden.dtype).unsqueeze(-1)
    hidden_out = hidden.clone()
    hidden_out[row_idx, decision_pos] = token_hidden + (scale_term * rms) * direction.unsqueeze(0)
    return hidden_out


def run_validation_policy(
    *,
    validation_rows: pd.DataFrame,
    validation_cache: dict[str, object],
    validation_detector_outputs_df: pd.DataFrame,
    scale_rule_stats_df: pd.DataFrame,
    directions: dict[int, torch.Tensor],
    tok: AutoTokenizer,
    model: AutoModelForCausalLM,
    input_device: torch.device,
    decoder_layers,
    answer_id_tensor_cpu: torch.Tensor,
    max_seq_len: int,
) -> pd.DataFrame:
    detector_lookup = validation_detector_outputs_df.set_index(["example_id", "layer_number"])
    rule_lookup = scale_rule_stats_df.set_index(["layer_number", "target_rule"])

    validation_choice_logits = validation_cache["clean_choice_logits"]
    validation_best_non_choice_logit = validation_cache["clean_best_non_choice_logit"]
    validation_best_non_choice_token_id = validation_cache["clean_best_non_choice_token_id"]
    validation_hidden = validation_cache["hidden"]
    example_ids = [row["example_id"] for row in validation_cache["example_rows"]]
    example_id_to_index = {example_id: idx for idx, example_id in enumerate(example_ids)}

    policy_rows: list[dict[str, object]] = []
    total_steps = len(decoder_layers) * len(TARGET_RULES)

    with tqdm(total=total_steps, desc="validation policy sweep") as pbar:
        for layer_number in range(1, len(decoder_layers) + 1):
            steering_module = decoder_layers[layer_number - 1]
            direction = directions[layer_number]

            layer_hidden = validation_hidden[:, layer_number - 1, :].to(torch.float32)
            projection_all = torch.matmul(layer_hidden, direction.to(layer_hidden.device)).detach().cpu().numpy().astype(np.float32)
            rms_all = layer_hidden.pow(2).mean(dim=-1).sqrt().detach().cpu().numpy().astype(np.float32)
            hidden_l2_all = layer_hidden.norm(dim=-1).detach().cpu().numpy().astype(np.float32)

            for target_rule in TARGET_RULES:
                stats_row = rule_lookup.loc[(layer_number, target_rule)]
                target_projection = float(stats_row["target_projection"])
                scale_cap = float(stats_row["scale_cap"])

                detector_prob_all = np.array(
                    [
                        float(detector_lookup.loc[(example_id, layer_number), "detector_probability_error"])
                        for example_id in example_ids
                    ],
                    dtype=np.float32,
                )
                detector_pred_all = np.array(
                    [
                        bool(detector_lookup.loc[(example_id, layer_number), "detector_predicted_error"])
                        for example_id in example_ids
                    ],
                    dtype=bool,
                )

                required_scale_raw_all = np.maximum(
                    0.0,
                    (target_projection - projection_all) / np.clip(rms_all, 1e-12, None),
                ).astype(np.float32)
                applied_scale_all = np.where(
                    detector_pred_all,
                    np.minimum(required_scale_raw_all, scale_cap),
                    0.0,
                ).astype(np.float32)

                for start in range(0, len(validation_rows), STEERING_BATCH_SIZE):
                    batch_df = validation_rows.iloc[start:start + STEERING_BATCH_SIZE].reset_index(drop=True)
                    batch_indices = [example_id_to_index[example_id] for example_id in batch_df["example_id"].tolist()]
                    batch_scales = applied_scale_all[batch_indices]
                    batch_projection = projection_all[batch_indices]
                    batch_required_raw = required_scale_raw_all[batch_indices]
                    batch_detector_prob = detector_prob_all[batch_indices]
                    batch_detector_pred = detector_pred_all[batch_indices]
                    batch_rms = rms_all[batch_indices]
                    batch_hidden_l2 = hidden_l2_all[batch_indices]
                    direction_l2_norm = float(direction.float().norm().item())
                    batch_delta_l2 = batch_scales * batch_rms * direction_l2_norm
                    batch_delta_over_hidden = batch_delta_l2 / np.clip(batch_hidden_l2, 1e-12, None)

                    if float(batch_scales.max(initial=0.0)) <= 0.0:
                        for batch_index, row in batch_df.iterrows():
                            global_index = batch_indices[batch_index]
                            policy_rows.append(
                                {
                                    "example_id": row["example_id"],
                                    "layer_number": layer_number,
                                    "target_rule": target_rule,
                                    "detector_probability_error": float(batch_detector_prob[batch_index]),
                                    "detector_predicted_error": bool(batch_detector_pred[batch_index]),
                                    "projection": float(batch_projection[batch_index]),
                                    "projection_target": float(target_projection),
                                    "required_scale_raw": float(batch_required_raw[batch_index]),
                                    "scale_cap": float(scale_cap),
                                    "applied_scale": 0.0,
                                    "token_hidden_rms": float(batch_rms[batch_index]),
                                    "token_hidden_l2_norm": float(batch_hidden_l2[batch_index]),
                                    "direction_l2_norm": direction_l2_norm,
                                    "delta_l2_norm": 0.0,
                                    "delta_over_token_hidden_l2": 0.0,
                                    "steered_best_non_choice_token_id": int(validation_best_non_choice_token_id[global_index]),
                                    "steered_best_non_choice_logit": float(validation_best_non_choice_logit[global_index]),
                                    "steered_logit_A": float(validation_choice_logits[global_index, 0]),
                                    "steered_logit_B": float(validation_choice_logits[global_index, 1]),
                                    "steered_logit_C": float(validation_choice_logits[global_index, 2]),
                                    "steered_logit_D": float(validation_choice_logits[global_index, 3]),
                                    "steered_logit_E": float(validation_choice_logits[global_index, 4]),
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
                        token_hidden_float = token_hidden.float()
                        token_hidden_rms = token_hidden_float.pow(2).mean(dim=-1).sqrt()
                        token_hidden_l2_norm = token_hidden_float.norm(dim=-1)
                        direction_device = direction.to(hidden.device, dtype=hidden.dtype)
                        direction_norm = direction_device.float().norm()
                        steering_scale_tensor = torch.tensor(
                            batch_scales,
                            dtype=torch.float32,
                            device=hidden.device,
                        )
                        delta_l2_norm = steering_scale_tensor * token_hidden_rms * direction_norm
                        delta_over_token_hidden_l2 = delta_l2_norm / token_hidden_l2_norm.clamp_min(1e-12)

                        steering_stats["token_hidden_rms"] = token_hidden_rms.detach().cpu().numpy().astype(np.float32)
                        steering_stats["token_hidden_l2_norm"] = token_hidden_l2_norm.detach().cpu().numpy().astype(np.float32)
                        steering_stats["direction_l2_norm"] = np.full(
                            hidden.shape[0],
                            float(direction_norm.item()),
                            dtype=np.float32,
                        )
                        steering_stats["delta_l2_norm"] = delta_l2_norm.detach().cpu().numpy().astype(np.float32)
                        steering_stats["delta_over_token_hidden_l2"] = (
                            delta_over_token_hidden_l2.detach().cpu().numpy().astype(np.float32)
                        )

                        hidden = apply_token_steering_per_example(
                            hidden,
                            decision_pos,
                            direction,
                            steering_scale_tensor,
                        )
                        return repack_output_hidden(output, hidden)

                    handle = steering_module.register_forward_hook(steering_hook)
                    try:
                        with torch.inference_mode():
                            out = model(**batch, return_dict=True, use_cache=False)
                    finally:
                        handle.remove()

                    full_logits = select_full_logits_at_decision(out.logits, decision_pos)
                    metrics = summarize_decision_logits(
                        full_logits,
                        true_choice_idx,
                        answer_id_tensor_cpu.to(full_logits.device),
                    )

                    choice_logits_cpu = metrics["choice_logits"].detach().cpu().numpy().astype(np.float32)
                    best_non_choice_logit_cpu = metrics["best_non_choice_logit"].detach().cpu().numpy().astype(np.float32)
                    masked_logits = full_logits.clone()
                    masked_logits[:, answer_id_tensor_cpu.to(full_logits.device)] = -torch.inf
                    best_non_choice_token_id_cpu = (
                        torch.argmax(masked_logits, dim=-1).detach().cpu().numpy().astype(np.int64)
                    )

                    for batch_index, row in batch_df.iterrows():
                        policy_rows.append(
                            {
                                "example_id": row["example_id"],
                                "layer_number": layer_number,
                                "target_rule": target_rule,
                                "detector_probability_error": float(batch_detector_prob[batch_index]),
                                "detector_predicted_error": bool(batch_detector_pred[batch_index]),
                                "projection": float(batch_projection[batch_index]),
                                "projection_target": float(target_projection),
                                "required_scale_raw": float(batch_required_raw[batch_index]),
                                "scale_cap": float(scale_cap),
                                "applied_scale": float(batch_scales[batch_index]),
                                "token_hidden_rms": float(steering_stats["token_hidden_rms"][batch_index]),
                                "token_hidden_l2_norm": float(steering_stats["token_hidden_l2_norm"][batch_index]),
                                "direction_l2_norm": float(steering_stats["direction_l2_norm"][batch_index]),
                                "delta_l2_norm": float(steering_stats["delta_l2_norm"][batch_index]),
                                "delta_over_token_hidden_l2": float(
                                    steering_stats["delta_over_token_hidden_l2"][batch_index]
                                ),
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

            del layer_hidden
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return pd.DataFrame(policy_rows)


def summarize_policy_results(
    *,
    validation_clean_outputs_df: pd.DataFrame,
    policy_outputs_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = policy_outputs_df.merge(
        validation_clean_outputs_df.rename(
            columns={
                "clean_logit_A": "baseline_logit_A",
                "clean_logit_B": "baseline_logit_B",
                "clean_logit_C": "baseline_logit_C",
                "clean_logit_D": "baseline_logit_D",
                "clean_logit_E": "baseline_logit_E",
                "best_non_choice_token_id": "baseline_best_non_choice_token_id",
                "best_non_choice_logit": "baseline_best_non_choice_logit",
            }
        ),
        on="example_id",
        how="left",
        validate="many_to_one",
    )

    baseline_choice_logits = merged[
        ["baseline_logit_A", "baseline_logit_B", "baseline_logit_C", "baseline_logit_D", "baseline_logit_E"]
    ].to_numpy(dtype=np.float32)
    steered_choice_logits = merged[
        ["steered_logit_A", "steered_logit_B", "steered_logit_C", "steered_logit_D", "steered_logit_E"]
    ].to_numpy(dtype=np.float32)
    true_choice_idx = merged["true_choice_idx"].to_numpy(dtype=np.int64)

    clean_pred_idx = choice_logits_to_pred_idx(baseline_choice_logits)
    steered_pred_idx = choice_logits_to_pred_idx(steered_choice_logits)
    clean_is_correct = clean_pred_idx == true_choice_idx
    steered_is_correct = steered_pred_idx == true_choice_idx

    merged["clean_predicted_choice_idx"] = clean_pred_idx
    merged["steered_predicted_choice_idx"] = steered_pred_idx
    merged["clean_is_correct"] = clean_is_correct
    merged["steered_is_correct"] = steered_is_correct
    merged["prediction_changed"] = clean_pred_idx != steered_pred_idx
    merged["rescued_error"] = (~clean_is_correct) & steered_is_correct
    merged["harmed_correct"] = clean_is_correct & (~steered_is_correct)
    merged["clean_answer_choice_entropy"] = choice_logits_to_entropy(baseline_choice_logits)
    merged["steered_answer_choice_entropy"] = choice_logits_to_entropy(steered_choice_logits)
    merged["clean_answer_choice_top1_top2_logit_gap"] = choice_logits_to_gap(baseline_choice_logits)
    merged["steered_answer_choice_top1_top2_logit_gap"] = choice_logits_to_gap(steered_choice_logits)
    merged["clean_best_choice_minus_best_non_choice_logit"] = choice_logits_and_best_non_choice_to_margin(
        baseline_choice_logits,
        merged["baseline_best_non_choice_logit"].to_numpy(dtype=np.float32),
    )
    merged["steered_best_choice_minus_best_non_choice_logit"] = choice_logits_and_best_non_choice_to_margin(
        steered_choice_logits,
        merged["steered_best_non_choice_logit"].to_numpy(dtype=np.float32),
    )
    merged["delta_answer_choice_entropy"] = (
        merged["steered_answer_choice_entropy"] - merged["clean_answer_choice_entropy"]
    )
    merged["delta_answer_choice_top1_top2_logit_gap"] = (
        merged["steered_answer_choice_top1_top2_logit_gap"] - merged["clean_answer_choice_top1_top2_logit_gap"]
    )
    merged["delta_best_choice_minus_best_non_choice_logit"] = (
        merged["steered_best_choice_minus_best_non_choice_logit"]
        - merged["clean_best_choice_minus_best_non_choice_logit"]
    )

    summary_rows: list[dict[str, object]] = []
    clean_accuracy = float(clean_is_correct.mean())
    n_total = int(len(validation_clean_outputs_df))
    n_clean_correct = int(clean_is_correct.sum())
    n_clean_incorrect = int((~clean_is_correct).sum())

    for (layer_number, target_rule), part in merged.groupby(["layer_number", "target_rule"], sort=True):
        steered_correct = part["steered_is_correct"].to_numpy(dtype=bool)
        rescued = part["rescued_error"].to_numpy(dtype=bool)
        harmed = part["harmed_correct"].to_numpy(dtype=bool)
        flagged = part["detector_predicted_error"].to_numpy(dtype=bool)

        steered_accuracy = float(steered_correct.mean())
        rescued_count = int(rescued.sum())
        harmed_count = int(harmed.sum())
        net_gain_count = rescued_count - harmed_count
        summary_rows.append(
            {
                "layer_number": int(layer_number),
                "target_rule": target_rule,
                "n_total": n_total,
                "n_clean_correct": n_clean_correct,
                "n_clean_incorrect": n_clean_incorrect,
                "clean_accuracy": clean_accuracy,
                "steered_accuracy": steered_accuracy,
                "accuracy_delta": steered_accuracy - clean_accuracy,
                "flagged_count": int(flagged.sum()),
                "flag_rate": float(flagged.mean()),
                "rescued_count": rescued_count,
                "harmed_count": harmed_count,
                "net_gain_count": int(net_gain_count),
                "rescued_rate_among_incorrect": float(rescued_count / max(n_clean_incorrect, 1)),
                "harmed_rate_among_correct": float(harmed_count / max(n_clean_correct, 1)),
                "prediction_changed_count": int(part["prediction_changed"].sum()),
                "mean_applied_scale": float(part["applied_scale"].mean()),
                "median_applied_scale": float(part["applied_scale"].median()),
                "max_applied_scale": float(part["applied_scale"].max()),
                "mean_delta_over_token_hidden_l2": float(part["delta_over_token_hidden_l2"].mean()),
                "mean_delta_l2_norm": float(part["delta_l2_norm"].mean()),
                "mean_detector_probability_error": float(part["detector_probability_error"].mean()),
                "flagged_precision": float(
                    part.loc[part["detector_predicted_error"], "clean_is_correct"].rsub(1).mean()
                    if int(flagged.sum()) > 0
                    else np.nan
                ),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["net_gain_count", "accuracy_delta", "rescued_count", "harmed_count", "mean_applied_scale"],
        ascending=[False, False, False, True, True],
    ).reset_index(drop=True)
    return merged, summary_df


def select_best_policy(policy_summary_df: pd.DataFrame) -> dict[str, object]:
    best_row = policy_summary_df.iloc[0]
    return {
        "layer_number": int(best_row["layer_number"]),
        "target_rule": str(best_row["target_rule"]),
        "net_gain_count": int(best_row["net_gain_count"]),
        "accuracy_delta": float(best_row["accuracy_delta"]),
        "rescued_count": int(best_row["rescued_count"]),
        "harmed_count": int(best_row["harmed_count"]),
        "flagged_count": int(best_row["flagged_count"]),
        "mean_applied_scale": float(best_row["mean_applied_scale"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--max-seq-len", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scale-cap-quantile", type=float, default=0.95)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--validation-limit", type=int, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_rows = load_csqa(split="train", limit=args.train_limit).copy()
    validation_rows = load_csqa(split="validation", limit=args.validation_limit).copy()
    for frame in [train_rows, validation_rows]:
        frame["prompt_len_chars"] = frame["text"].str.len()

    model_dtype, device_map = choose_model_dtype_and_device_map()

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
    hidden_size = int(model.lm_head.weight.shape[1])
    vocab_size = int(model.config.vocab_size)

    readout_ctx = prepare_readout_context(
        model=model,
        tok=tok,
        input_device=input_device,
        num_layers=num_layers,
        max_seq_len=args.max_seq_len,
    )
    final_norm = readout_ctx["final_norm"]
    answer_id_tensor_cpu = readout_ctx["answer_id_tensor_cpu"]
    answer_id_tensor_lm_head = readout_ctx["answer_id_tensor_lm_head"]
    answer_choice_weight = readout_ctx["answer_choice_weight"]
    lm_head_weight = readout_ctx["lm_head_weight"]
    maybe_apply_final_norm = readout_ctx["maybe_apply_final_norm"]
    last_layer_needs_final_norm = readout_ctx["last_layer_needs_final_norm"]

    train_cache = extract_split_cache(
        train_rows,
        split_name="train",
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
    validation_cache = extract_split_cache(
        validation_rows,
        split_name="validation",
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

    lenses, tuned_lens_history_df = train_tuned_lenses(
        train_hidden=train_cache["hidden"],
        teacher_choice_probs_train=train_cache["final_choice_probs"].float(),
        hidden_size=hidden_size,
        num_layers=num_layers,
        answer_choice_weight=answer_choice_weight,
        train_device=input_device,
    )

    train_attention_df = build_attention_feature_table(
        train_rows,
        split_name="train",
        tok=tok,
        model=model,
        input_device=input_device,
        max_seq_len=args.max_seq_len,
        num_layers=num_layers,
    )
    validation_attention_df = build_attention_feature_table(
        validation_rows,
        split_name="validation",
        tok=tok,
        model=model,
        input_device=input_device,
        max_seq_len=args.max_seq_len,
        num_layers=num_layers,
    )

    train_feature_df = build_detector_feature_table(
        split_name="train",
        cache=train_cache,
        attention_df=train_attention_df,
        lenses=lenses,
        final_norm=final_norm,
        maybe_apply_final_norm=maybe_apply_final_norm,
        lm_head_weight=lm_head_weight,
        answer_id_tensor_lm_head=answer_id_tensor_lm_head,
        input_device=input_device,
        vocab_size=vocab_size,
    )
    validation_feature_df = build_detector_feature_table(
        split_name="validation",
        cache=validation_cache,
        attention_df=validation_attention_df,
        lenses=lenses,
        final_norm=final_norm,
        maybe_apply_final_norm=maybe_apply_final_norm,
        lm_head_weight=lm_head_weight,
        answer_id_tensor_lm_head=answer_id_tensor_lm_head,
        input_device=input_device,
        vocab_size=vocab_size,
    )

    detector_models, detector_coefficients_df, detector_summary_df, detector_outputs_df = fit_layer_detectors(
        train_feature_df=train_feature_df,
        validation_feature_df=validation_feature_df,
    )

    directions, scale_rule_stats_df, _, _ = build_scale_rule_stats(
        train_hidden=train_cache["hidden"],
        validation_hidden=validation_cache["hidden"],
        train_clean_is_correct=train_cache["clean_is_correct"],
        cap_quantile=args.scale_cap_quantile,
    )

    validation_detector_outputs_df = detector_outputs_df.loc[detector_outputs_df["split"].eq("validation")].copy()
    validation_policy_outputs_df = run_validation_policy(
        validation_rows=validation_rows,
        validation_cache=validation_cache,
        validation_detector_outputs_df=validation_detector_outputs_df,
        scale_rule_stats_df=scale_rule_stats_df,
        directions=directions,
        tok=tok,
        model=model,
        input_device=input_device,
        decoder_layers=decoder_layers,
        answer_id_tensor_cpu=answer_id_tensor_cpu,
        max_seq_len=args.max_seq_len,
    )

    train_examples_df = pd.DataFrame(train_cache["example_rows"])
    validation_examples_df = pd.DataFrame(validation_cache["example_rows"])
    train_clean_outputs_df = pd.DataFrame(train_cache["clean_output_rows"])
    validation_clean_outputs_df = pd.DataFrame(validation_cache["clean_output_rows"])

    validation_policy_outputs_derived_df, validation_policy_summary_df = summarize_policy_results(
        validation_clean_outputs_df=validation_clean_outputs_df,
        policy_outputs_df=validation_policy_outputs_df,
    )
    best_policy = select_best_policy(validation_policy_summary_df)

    out_dir = resolve_out_dir(args.out_dir, args.model_id)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_examples_df.to_parquet(out_dir / "train_examples.parquet", index=False)
    validation_examples_df.to_parquet(out_dir / "validation_examples.parquet", index=False)
    train_clean_outputs_df.to_parquet(out_dir / "train_clean_final_outputs.parquet", index=False)
    validation_clean_outputs_df.to_parquet(out_dir / "validation_clean_final_outputs.parquet", index=False)
    train_attention_df.to_parquet(out_dir / "train_attention_detector_features.parquet", index=False)
    validation_attention_df.to_parquet(out_dir / "validation_attention_detector_features.parquet", index=False)
    train_feature_df.to_parquet(out_dir / "train_detector_features.parquet", index=False)
    validation_feature_df.to_parquet(out_dir / "validation_detector_features.parquet", index=False)
    tuned_lens_history_df.to_parquet(out_dir / "tuned_lens_training_history.parquet", index=False)
    detector_coefficients_df.to_parquet(out_dir / "detector_coefficients.parquet", index=False)
    detector_summary_df.to_parquet(out_dir / "detector_summary.parquet", index=False)
    detector_outputs_df.to_parquet(out_dir / "detector_outputs.parquet", index=False)
    scale_rule_stats_df.to_parquet(out_dir / "scale_rule_stats.parquet", index=False)
    validation_policy_outputs_df.to_parquet(out_dir / "validation_policy_outputs_raw.parquet", index=False)
    validation_policy_outputs_derived_df.to_parquet(out_dir / "validation_policy_outputs_derived.parquet", index=False)
    validation_policy_summary_df.to_parquet(out_dir / "validation_policy_summary.parquet", index=False)

    run_config = {
        "model_id": args.model_id,
        "max_seq_len": int(args.max_seq_len),
        "seed": int(args.seed),
        "train_limit": None if args.train_limit is None else int(args.train_limit),
        "validation_limit": None if args.validation_limit is None else int(args.validation_limit),
        "num_layers": int(num_layers),
        "hidden_size": int(hidden_size),
        "detector_type": "l1_logistic_regression_cv",
        "detector_threshold_rule": "predict_proba>=0.5",
        "steering_method": "contrastive_mean_direction",
        "scale_policy": "projection_target_with_train_quantile_cap",
        "scale_cap_quantile": float(args.scale_cap_quantile),
        "target_rules": TARGET_RULES,
        "extract_batch_size": EXTRACT_BATCH_SIZE,
        "attention_batch_size": ATTENTION_BATCH_SIZE,
        "steering_batch_size": STEERING_BATCH_SIZE,
        "tuned_lens_batch_size": TUNED_LENS_BATCH_SIZE,
        "tuned_lens_max_epochs": TUNED_LENS_MAX_EPOCHS,
        "tuned_lens_patience": TUNED_LENS_PATIENCE,
        "tuned_lens_lr": TUNED_LENS_LR,
        "tuned_lens_weight_decay": TUNED_LENS_WEIGHT_DECAY,
        "best_policy": best_policy,
    }
    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    print("\nBEST POLICY")
    print(json.dumps(best_policy, indent=2))
    print(f"[done] wrote {out_dir}")


if __name__ == "__main__":
    main()
