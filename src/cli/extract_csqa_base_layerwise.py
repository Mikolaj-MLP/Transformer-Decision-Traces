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
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.load_csqa import load_csqa


LETTERS = ["A", "B", "C", "D", "E"]
EXTRACT_BATCH_SIZE = 4
READOUT_BATCH_SIZE = 32
ATTENTION_BATCH_SIZE = 1
SUBSTEP_BATCH_SIZE = 1
TUNED_LENS_BATCH_SIZE = 64
TUNED_LENS_EPOCHS = 2
TUNED_LENS_LR = 1e-3
TUNED_LENS_WEIGHT_DECAY = 1e-5


class AffineTranslator(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def now_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def repo_root() -> Path:
    return REPO_ROOT


def slugify_model_id(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", model_id).strip("-")


def resolve_out_dir(out_dir: str | None, model_id: str) -> Path:
    root = repo_root()
    if out_dir is None:
        run_name = f"{now_id()}_{slugify_model_id(model_id)}_csqa_base_layerwise"
        return root / "data" / "generated" / "csqa_base_layerwise" / run_name
    path = Path(out_dir)
    return path if path.is_absolute() else (root / path)


def get_final_norm_module(model: AutoModelForCausalLM):
    candidates = [
        "model.norm",
        "model.final_layernorm",
        "transformer.ln_f",
        "gpt_neox.final_layer_norm",
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
    return None


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


def get_post_attention_input_module(layer: nn.Module) -> nn.Module:
    candidates = [
        "post_attention_layernorm",
        "ln_2",
    ]
    for name in candidates:
        if hasattr(layer, name):
            return getattr(layer, name)
    raise ValueError("Could not locate post-attention input module on decoder layer.")


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
    prompt_token_count = []
    for mask in batch["attention_mask"]:
        nz = torch.nonzero(mask, as_tuple=False).view(-1)
        prompt_token_count.append(int(nz.numel()))
        decision_pos.append(int(nz[-1].item()))
    batch["decision_pos"] = torch.tensor(decision_pos, dtype=torch.long)
    batch["prompt_token_count"] = torch.tensor(prompt_token_count, dtype=torch.long)
    return batch


def build_choice_token_spans(
    text: str,
    tok: AutoTokenizer,
    max_seq_len: int,
) -> list[tuple[int, int]]:
    lines = str(text).split("\n")
    choice_lines = lines[2:7]
    if len(choice_lines) != 5:
        raise ValueError("Expected exactly 5 answer-choice lines.")

    full_ids = tok(str(text), add_special_tokens=False)["input_ids"]
    full_len = len(full_ids)
    kept_len = min(full_len, max_seq_len)
    left_trunc = max(0, full_len - max_seq_len)

    spans: list[tuple[int, int]] = []
    for choice_index in range(5):
        prefix_text = "\n".join(lines[: 2 + choice_index]) + "\n"
        end_text = "\n".join(lines[: 3 + choice_index])
        start_full = len(tok(prefix_text, add_special_tokens=False)["input_ids"])
        end_full = len(tok(end_text, add_special_tokens=False)["input_ids"])

        start_kept = max(0, min(kept_len, start_full - left_trunc))
        end_kept = max(0, min(kept_len, end_full - left_trunc))
        spans.append((int(start_kept), int(end_kept)))

    return spans


def true_choice_logit_minus_best_other_choice_logit(
    choice_logits: torch.Tensor,
    true_choice_idx: torch.Tensor,
) -> torch.Tensor:
    row_idx = torch.arange(choice_logits.shape[0], device=choice_logits.device)
    true_choice_logits = choice_logits[row_idx, true_choice_idx]
    masked = choice_logits.clone()
    masked[row_idx, true_choice_idx] = -torch.inf
    best_other_choice_logits = torch.max(masked, dim=-1).values
    return true_choice_logits - best_other_choice_logits


def unpack_output_hidden(output: torch.Tensor | tuple[torch.Tensor, ...]) -> torch.Tensor:
    if isinstance(output, tuple):
        return output[0]
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--max-seq-len", type=int, default=384)
    parser.add_argument("--top-k-final", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_rows = load_csqa(split="train", limit=None).copy()
    eval_rows = load_csqa(split="validation", limit=None).copy()

    for frame in [train_rows, eval_rows]:
        frame["prompt_len_chars"] = frame["text"].str.len()

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
        torch_dtype=model_dtype,
        device_map=device_map,
        attn_implementation="eager",
    )
    model.eval()

    input_device = model.device if hasattr(model, "device") else next(model.parameters()).device
    decoder_layers = get_decoder_layers(model)
    final_norm = get_final_norm_module(model)
    num_layers = len(decoder_layers)
    hidden_size = int(model.lm_head.weight.shape[1])

    answer_token_ids = build_answer_token_ids(tok)
    answer_ids = [answer_token_ids[letter] for letter in LETTERS]
    answer_id_tensor_cpu = torch.tensor(answer_ids, dtype=torch.long)

    lm_head_weight = model.lm_head.weight.detach()
    lm_head_device = lm_head_weight.device
    answer_id_tensor_lm_head = answer_id_tensor_cpu.to(lm_head_device)
    answer_choice_weight = lm_head_weight.index_select(0, answer_id_tensor_lm_head)

    probe_cpu = encode_prompts(eval_rows["text"].head(1).tolist(), tok, args.max_seq_len)
    probe_pos = int(probe_cpu["decision_pos"][0].item())
    probe_batch = {k: v.to(input_device) for k, v in probe_cpu.items() if k not in ["decision_pos", "prompt_token_count"]}
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

    def summarize_hidden_readout(
        hidden_batch: torch.Tensor,
        layer_index_0based: int,
        true_choice_idx_batch: torch.Tensor,
        final_choice_prob_batch: torch.Tensor,
        *,
        always_apply_final_norm_if_available: bool,
    ) -> dict[str, torch.Tensor]:
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
        predicted_vocab_token_id = torch.argmax(full_logits, dim=-1)

        masked_logits = full_logits.clone()
        masked_logits[:, answer_id_tensor_lm_head] = -torch.inf
        best_non_choice_logit, best_non_choice_token_id = torch.max(masked_logits, dim=-1)

        choice_logits = full_logits.index_select(1, answer_id_tensor_lm_head)
        choice_log_probs = torch.log_softmax(choice_logits, dim=-1)
        choice_probs = torch.exp(choice_log_probs)
        predicted_choice_index = torch.argmax(choice_logits, dim=-1)
        sorted_choice_logits = torch.sort(choice_logits, dim=-1, descending=True).values
        choice_gap = sorted_choice_logits[:, 0] - sorted_choice_logits[:, 1]
        row_idx = torch.arange(choice_logits.shape[0], device=choice_logits.device)
        true_choice_prob = choice_probs[row_idx, true_choice_idx_batch]
        true_choice_rank = 1 + (
            choice_logits > choice_logits[row_idx, true_choice_idx_batch][:, None]
        ).sum(dim=-1)
        true_choice_gap = true_choice_logit_minus_best_other_choice_logit(
            choice_logits,
            true_choice_idx_batch,
        )
        choice_kl_to_final = torch.sum(
            final_choice_prob_batch
            * (
                torch.log(torch.clamp(final_choice_prob_batch, min=1e-12))
                - choice_log_probs
            ),
            dim=-1,
        )
        choice_entropy = -(choice_probs * choice_log_probs).sum(dim=-1)
        best_choice_logit = torch.max(choice_logits, dim=-1).values

        return {
            "predicted_choice_idx": predicted_choice_index,
            "predicted_choice_is_correct": predicted_choice_index.eq(true_choice_idx_batch),
            "predicted_vocab_token_id": predicted_vocab_token_id,
            "best_non_choice_token_id": best_non_choice_token_id,
            "best_non_choice_logit": best_non_choice_logit,
            "best_choice_minus_best_non_choice_logit": best_choice_logit - best_non_choice_logit,
            "full_vocab_entropy": full_entropy,
            "answer_choice_entropy": choice_entropy,
            "answer_choice_top1_top2_logit_gap": choice_gap,
            "true_answer_logit_minus_best_other_choice_logit": true_choice_gap,
            "true_answer_probability_within_choices": true_choice_prob,
            "true_answer_rank_within_choices": true_choice_rank,
            "choice_kl_to_final": choice_kl_to_final,
            "logit_A": choice_logits[:, 0],
            "logit_B": choice_logits[:, 1],
            "logit_C": choice_logits[:, 2],
            "logit_D": choice_logits[:, 3],
            "logit_E": choice_logits[:, 4],
        }

    def extract_split_cache(frame: pd.DataFrame, capture_final_outputs: bool):
        hidden_blocks: list[torch.Tensor] = []
        final_choice_prob_blocks: list[torch.Tensor] = []
        true_choice_idx_blocks: list[torch.Tensor] = []
        example_rows: list[dict[str, object]] = []
        final_output_rows: list[dict[str, object]] = []
        final_topk_rows: list[dict[str, object]] = []

        for start in range(0, len(frame), EXTRACT_BATCH_SIZE):
            batch_df = frame.iloc[start:start + EXTRACT_BATCH_SIZE].reset_index(drop=True)
            batch_cpu = encode_prompts(batch_df["text"].tolist(), tok, args.max_seq_len)
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

            for batch_index, row in batch_df.iterrows():
                example_rows.append(
                    {
                        "example_id": row["example_id"],
                        "text": row["text"],
                        "answerKey": row["answerKey"],
                        "correct_idx": int(row["correct_idx"]),
                        "prompt_len_chars": int(row["prompt_len_chars"]),
                        "prompt_token_count": int(prompt_token_count[batch_index].item()),
                        "decision_pos": int(decision_pos[batch_index].item()),
                    }
                )

            if capture_final_outputs:
                final_logits = out.logits[row_idx, decision_pos].float().detach().cpu()
                final_log_probs = torch.log_softmax(final_logits, dim=-1)
                final_probs = torch.exp(final_log_probs)
                topk_values, topk_ids = torch.topk(final_logits, k=int(args.top_k_final), dim=-1)

                masked_logits = final_logits.clone()
                masked_logits[:, answer_id_tensor_cpu] = -torch.inf
                best_non_choice_logit, best_non_choice_token_id = torch.max(masked_logits, dim=-1)

                choice_logits = final_logits.index_select(1, answer_id_tensor_cpu)
                true_choice_idx_batch_cpu = torch.tensor(batch_df["correct_idx"].tolist(), dtype=torch.long)
                true_choice_logit_gap = true_choice_logit_minus_best_other_choice_logit(
                    choice_logits,
                    true_choice_idx_batch_cpu,
                )
                choice_probs = torch.softmax(choice_logits, dim=-1)
                choice_pred_idx = choice_logits.argmax(dim=-1)

                for batch_index, row in batch_df.iterrows():
                    final_output_rows.append(
                        {
                            "example_id": row["example_id"],
                            "true_choice_idx": int(row["correct_idx"]),
                            "true_choice_letter": LETTERS[int(row["correct_idx"])],
                            "clean_predicted_choice_idx": int(choice_pred_idx[batch_index].item()),
                            "clean_predicted_choice_letter": LETTERS[int(choice_pred_idx[batch_index].item())],
                            "clean_is_correct": bool(int(choice_pred_idx[batch_index].item()) == int(row["correct_idx"])),
                            "clean_true_answer_probability_within_choices": float(
                                choice_probs[batch_index, int(row["correct_idx"])].item()
                            ),
                            "clean_true_answer_rank_within_choices": int(
                                1
                                + (
                                    choice_logits[batch_index]
                                    > choice_logits[batch_index, int(row["correct_idx"])]
                                ).sum().item()
                            ),
                            "clean_true_answer_logit_minus_best_other_choice_logit": float(
                                true_choice_logit_gap[batch_index].item()
                            ),
                            "clean_logit_A": float(choice_logits[batch_index, 0].item()),
                            "clean_logit_B": float(choice_logits[batch_index, 1].item()),
                            "clean_logit_C": float(choice_logits[batch_index, 2].item()),
                            "clean_logit_D": float(choice_logits[batch_index, 3].item()),
                            "clean_logit_E": float(choice_logits[batch_index, 4].item()),
                            "best_non_choice_token_id": int(best_non_choice_token_id[batch_index].item()),
                            "best_non_choice_token_str": tok.convert_ids_to_tokens([int(best_non_choice_token_id[batch_index].item())])[0],
                            "best_non_choice_logit": float(best_non_choice_logit[batch_index].item()),
                        }
                    )

                    for rank_index in range(int(args.top_k_final)):
                        token_id = int(topk_ids[batch_index, rank_index].item())
                        final_topk_rows.append(
                            {
                                "example_id": row["example_id"],
                                "rank": rank_index + 1,
                                "token_id": token_id,
                                "token_str": tok.convert_ids_to_tokens([token_id])[0],
                                "logit": float(topk_values[batch_index, rank_index].item()),
                                "probability": float(final_probs[batch_index, token_id].item()),
                            }
                        )

        return {
            "hidden": torch.cat(hidden_blocks, dim=0),
            "final_choice_probs": torch.cat(final_choice_prob_blocks, dim=0),
            "true_choice_idx": torch.cat(true_choice_idx_blocks, dim=0),
            "example_id": frame["example_id"].tolist(),
            "answerKey": frame["answerKey"].astype(str).tolist(),
            "example_rows": example_rows,
            "final_output_rows": final_output_rows,
            "final_topk_rows": final_topk_rows,
        }

    train_cache = extract_split_cache(train_rows, capture_final_outputs=False)
    teacher_choice_probs_train = train_cache["final_choice_probs"].float()

    train_device = input_device
    lenses: list[AffineTranslator | None] = []
    lens_state_dicts: dict[int, dict[str, torch.Tensor]] = {}
    train_history_rows: list[dict[str, object]] = []

    for layer_index in range(num_layers):
        if layer_index == (num_layers - 1):
            lenses.append(None)
            continue

        lens = AffineTranslator(hidden_size).to(train_device)
        optimizer = torch.optim.AdamW(
            lens.parameters(),
            lr=TUNED_LENS_LR,
            weight_decay=TUNED_LENS_WEIGHT_DECAY,
        )

        dataset = TensorDataset(train_cache["hidden"][:, layer_index, :], teacher_choice_probs_train)
        dataloader = DataLoader(dataset, batch_size=TUNED_LENS_BATCH_SIZE, shuffle=True)

        for epoch in range(1, TUNED_LENS_EPOCHS + 1):
            epoch_losses = []
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

            train_history_rows.append(
                {
                    "layer_number": layer_index + 1,
                    "epoch": epoch,
                    "mean_kl_loss": float(np.mean(epoch_losses)),
                }
            )

        lens.eval()
        lens_cpu = lens.cpu()
        lenses.append(lens_cpu)
        lens_state_dicts[layer_index + 1] = {k: v.detach().cpu() for k, v in lens_cpu.state_dict().items()}
        del lens
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    train_history_df = pd.DataFrame(train_history_rows)

    del train_cache
    del teacher_choice_probs_train
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    eval_cache = extract_split_cache(eval_rows, capture_final_outputs=True)
    examples_df = pd.DataFrame(eval_cache["example_rows"]).drop_duplicates(subset=["example_id"]).reset_index(drop=True)
    clean_final_df = pd.DataFrame(eval_cache["final_output_rows"])
    clean_final_topk_df = pd.DataFrame(eval_cache["final_topk_rows"])

    choice_span_map = {
        row["example_id"]: build_choice_token_spans(
            text=row["text"],
            tok=tok,
            max_seq_len=args.max_seq_len,
        )
        for _, row in eval_rows[["example_id", "text"]].iterrows()
    }

    attention_rows: list[dict[str, object]] = []
    for start in range(0, len(eval_rows), ATTENTION_BATCH_SIZE):
        batch_df = eval_rows.iloc[start:start + ATTENTION_BATCH_SIZE].reset_index(drop=True)
        batch_cpu = encode_prompts(batch_df["text"].tolist(), tok, args.max_seq_len)
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

                aggregated_choice_mass = choice_mass_by_head.mean(axis=0)
                if aggregated_choice_mass.sum() <= 0:
                    aggregated_choice_probs = np.full(len(LETTERS), 1.0 / len(LETTERS), dtype=np.float32)
                else:
                    aggregated_choice_probs = aggregated_choice_mass / aggregated_choice_mass.sum()

                sorted_choice_probs = np.sort(aggregated_choice_probs)[::-1]
                attention_choice_predicted_choice_idx = int(np.argmax(aggregated_choice_probs))

                attention_rows.append(
                    {
                        "example_id": example_id,
                        "layer_number": layer_index + 1,
                        "mean_head_renyi2_entropy_normalized": float(np.mean(head_renyi2_entropy)),
                        "aggregated_choice_attention_entropy_normalized": float(
                            -(
                                aggregated_choice_probs
                                * np.log(np.clip(aggregated_choice_probs, 1e-12, None))
                            ).sum()
                            / math.log(len(LETTERS))
                        ),
                        "aggregated_choice_attention_top1_top2_probability_gap": float(
                            sorted_choice_probs[0] - sorted_choice_probs[1]
                        ),
                        "attention_choice_predicted_choice_idx": attention_choice_predicted_choice_idx,
                        "attention_choice_predicted_choice_letter": LETTERS[attention_choice_predicted_choice_idx],
                        "attention_choice_probability_A": float(aggregated_choice_probs[0]),
                        "attention_choice_probability_B": float(aggregated_choice_probs[1]),
                        "attention_choice_probability_C": float(aggregated_choice_probs[2]),
                        "attention_choice_probability_D": float(aggregated_choice_probs[3]),
                        "attention_choice_probability_E": float(aggregated_choice_probs[4]),
                    }
                )

    attention_outputs_df = pd.DataFrame(attention_rows)

    lenses_eval = [None if lens is None else lens.to(train_device) for lens in lenses]
    true_choice_idx_eval = eval_cache["true_choice_idx"].numpy()
    final_choice_probs_eval = eval_cache["final_choice_probs"].numpy().astype(np.float32)

    substep_rows: list[dict[str, object]] = []
    substep_handles: list[torch.utils.hooks.RemovableHandle] = []
    substep_cache: dict[str, dict[int, torch.Tensor]] = {
        "pre_attn": {},
        "post_attn": {},
        "post_mlp": {},
    }

    def clear_substep_cache() -> None:
        for key in substep_cache:
            substep_cache[key].clear()

    for layer_index, layer in enumerate(decoder_layers):
        post_attn_module = get_post_attention_input_module(layer)

        def make_pre_attn_hook(idx: int):
            def hook(module, args):
                substep_cache["pre_attn"][idx] = args[0].detach()
            return hook

        def make_post_attn_hook(idx: int):
            def hook(module, args):
                substep_cache["post_attn"][idx] = args[0].detach()
            return hook

        def make_post_mlp_hook(idx: int):
            def hook(module, args, output):
                substep_cache["post_mlp"][idx] = unpack_output_hidden(output).detach()
            return hook

        substep_handles.append(layer.register_forward_pre_hook(make_pre_attn_hook(layer_index)))
        substep_handles.append(post_attn_module.register_forward_pre_hook(make_post_attn_hook(layer_index)))
        substep_handles.append(layer.register_forward_hook(make_post_mlp_hook(layer_index)))

    try:
        for start in range(0, len(eval_rows), SUBSTEP_BATCH_SIZE):
            stop = min(start + SUBSTEP_BATCH_SIZE, len(eval_rows))
            batch_df = eval_rows.iloc[start:stop].reset_index(drop=True)
            batch_cpu = encode_prompts(batch_df["text"].tolist(), tok, args.max_seq_len)
            decision_pos = batch_cpu.pop("decision_pos")
            batch_cpu.pop("prompt_token_count")
            batch = {k: v.to(input_device) for k, v in batch_cpu.items()}
            decision_pos = decision_pos.to(input_device)
            true_choice_idx_batch = torch.tensor(
                batch_df["correct_idx"].tolist(),
                dtype=torch.long,
                device=train_device,
            )
            final_choice_prob_batch = eval_cache["final_choice_probs"][start:stop].to(train_device).float()

            clear_substep_cache()
            with torch.inference_mode():
                _ = model(**batch, return_dict=True, use_cache=False)

            for layer_index in range(num_layers):
                layer_substeps = []
                for substep_name in ["pre_attn", "post_attn", "post_mlp"]:
                    hidden_full = substep_cache[substep_name][layer_index]
                    layer_row_idx = torch.arange(len(batch_df), device=hidden_full.device)
                    layer_decision_pos = decision_pos.to(hidden_full.device)
                    layer_substeps.append(
                        (substep_name, hidden_full[layer_row_idx, layer_decision_pos])
                    )

                for substep_name, hidden_batch in layer_substeps:
                    metrics = summarize_hidden_readout(
                        hidden_batch=hidden_batch.to(train_device),
                        layer_index_0based=layer_index,
                        true_choice_idx_batch=true_choice_idx_batch,
                        final_choice_prob_batch=final_choice_prob_batch,
                        always_apply_final_norm_if_available=True,
                    )

                    for batch_index, row in batch_df.iterrows():
                        substep_rows.append(
                            {
                                "example_id": row["example_id"],
                                "layer_number": layer_index + 1,
                                "substep_name": substep_name,
                                "true_choice_idx": int(row["correct_idx"]),
                                "true_choice_letter": LETTERS[int(row["correct_idx"])],
                                "predicted_choice_idx": int(metrics["predicted_choice_idx"][batch_index].item()),
                                "predicted_choice_letter": LETTERS[int(metrics["predicted_choice_idx"][batch_index].item())],
                                "predicted_choice_is_correct": bool(metrics["predicted_choice_is_correct"][batch_index].item()),
                                "predicted_vocab_token_id": int(metrics["predicted_vocab_token_id"][batch_index].item()),
                                "best_non_choice_token_id": int(metrics["best_non_choice_token_id"][batch_index].item()),
                                "best_non_choice_logit": float(metrics["best_non_choice_logit"][batch_index].item()),
                                "best_choice_minus_best_non_choice_logit": float(
                                    metrics["best_choice_minus_best_non_choice_logit"][batch_index].item()
                                ),
                                "full_vocab_entropy": float(metrics["full_vocab_entropy"][batch_index].item()),
                                "full_vocab_entropy_normalized": float(
                                    metrics["full_vocab_entropy"][batch_index].item()
                                    / math.log(int(lm_head_weight.shape[0]))
                                ),
                                "answer_choice_entropy": float(metrics["answer_choice_entropy"][batch_index].item()),
                                "answer_choice_entropy_normalized": float(
                                    metrics["answer_choice_entropy"][batch_index].item()
                                    / math.log(len(LETTERS))
                                ),
                                "answer_choice_top1_top2_logit_gap": float(
                                    metrics["answer_choice_top1_top2_logit_gap"][batch_index].item()
                                ),
                                "true_answer_logit_minus_best_other_choice_logit": float(
                                    metrics["true_answer_logit_minus_best_other_choice_logit"][batch_index].item()
                                ),
                                "true_answer_probability_within_choices": float(
                                    metrics["true_answer_probability_within_choices"][batch_index].item()
                                ),
                                "true_answer_rank_within_choices": int(
                                    metrics["true_answer_rank_within_choices"][batch_index].item()
                                ),
                                "choice_kl_to_final": float(metrics["choice_kl_to_final"][batch_index].item()),
                                "logit_A": float(metrics["logit_A"][batch_index].item()),
                                "logit_B": float(metrics["logit_B"][batch_index].item()),
                                "logit_C": float(metrics["logit_C"][batch_index].item()),
                                "logit_D": float(metrics["logit_D"][batch_index].item()),
                                "logit_E": float(metrics["logit_E"][batch_index].item()),
                            }
                        )
    finally:
        for handle in substep_handles:
            handle.remove()

    substep_outputs_df = pd.DataFrame(substep_rows)

    layerwise_rows: list[dict[str, object]] = []
    for method_name in ["direct_readout", "tuned_lens"]:
        for layer_index in range(num_layers):
            for start in range(0, len(eval_rows), READOUT_BATCH_SIZE):
                stop = min(start + READOUT_BATCH_SIZE, len(eval_rows))
                hidden_batch = eval_cache["hidden"][start:stop, layer_index, :].to(train_device).float()

                if method_name == "direct_readout":
                    readout = maybe_apply_final_norm(hidden_batch, layer_index)
                else:
                    if layer_index == (num_layers - 1):
                        readout = maybe_apply_final_norm(hidden_batch, layer_index)
                    else:
                        readout = lenses_eval[layer_index](hidden_batch)

                full_logits = torch.matmul(
                    readout.to(lm_head_weight.dtype),
                    lm_head_weight.T,
                ).float()
                full_log_probs = torch.log_softmax(full_logits, dim=-1)
                full_probs = torch.exp(full_log_probs)
                full_entropy = -(full_probs * full_log_probs).sum(dim=-1)
                predicted_vocab_token_id = torch.argmax(full_logits, dim=-1)

                masked_logits = full_logits.clone()
                masked_logits[:, answer_id_tensor_lm_head] = -torch.inf
                best_non_choice_logit, best_non_choice_token_id = torch.max(masked_logits, dim=-1)

                choice_logits = full_logits.index_select(1, answer_id_tensor_lm_head)
                choice_log_probs = torch.log_softmax(choice_logits, dim=-1)
                choice_probs = torch.exp(choice_log_probs)
                predicted_choice_index = torch.argmax(choice_logits, dim=-1)
                sorted_choice_logits = torch.sort(choice_logits, dim=-1, descending=True).values
                choice_gap = sorted_choice_logits[:, 0] - sorted_choice_logits[:, 1]
                true_choice_idx_batch = torch.tensor(true_choice_idx_eval[start:stop], dtype=torch.long, device=train_device)
                true_choice_prob = choice_probs[torch.arange(stop - start, device=train_device), true_choice_idx_batch]
                true_choice_rank = 1 + (choice_logits > choice_logits[torch.arange(stop - start, device=train_device), true_choice_idx_batch][:, None]).sum(dim=-1)
                true_choice_gap = true_choice_logit_minus_best_other_choice_logit(
                    choice_logits,
                    true_choice_idx_batch,
                )
                choice_kl_to_final = torch.sum(
                    torch.as_tensor(final_choice_probs_eval[start:stop], device=train_device)
                    * (
                        torch.log(torch.clamp(torch.as_tensor(final_choice_probs_eval[start:stop], device=train_device), min=1e-12))
                        - choice_log_probs
                    ),
                    dim=-1,
                )
                choice_entropy = -(choice_probs * choice_log_probs).sum(dim=-1)

                choice_logits_cpu = choice_logits.detach().cpu().numpy().astype(np.float32)
                full_entropy_cpu = full_entropy.detach().cpu().numpy().astype(np.float32)
                choice_entropy_cpu = choice_entropy.detach().cpu().numpy().astype(np.float32)
                choice_gap_cpu = choice_gap.detach().cpu().numpy().astype(np.float32)
                true_choice_gap_cpu = true_choice_gap.detach().cpu().numpy().astype(np.float32)
                true_choice_prob_cpu = true_choice_prob.detach().cpu().numpy().astype(np.float32)
                true_choice_rank_cpu = true_choice_rank.detach().cpu().numpy().astype(np.int64)
                predicted_choice_cpu = predicted_choice_index.detach().cpu().numpy().astype(np.int64)
                predicted_vocab_token_cpu = predicted_vocab_token_id.detach().cpu().numpy().astype(np.int64)
                best_non_choice_id_cpu = best_non_choice_token_id.detach().cpu().numpy().astype(np.int64)
                best_non_choice_logit_cpu = best_non_choice_logit.detach().cpu().numpy().astype(np.float32)
                choice_kl_to_final_cpu = choice_kl_to_final.detach().cpu().numpy().astype(np.float32)

                for batch_index, example_index in enumerate(range(start, stop)):
                    layerwise_rows.append(
                        {
                            "example_id": eval_cache["example_id"][example_index],
                            "layer_number": layer_index + 1,
                            "readout_method": method_name,
                            "true_choice_idx": int(true_choice_idx_eval[example_index]),
                            "true_choice_letter": LETTERS[int(true_choice_idx_eval[example_index])],
                            "predicted_choice_idx": int(predicted_choice_cpu[batch_index]),
                            "predicted_choice_letter": LETTERS[int(predicted_choice_cpu[batch_index])],
                            "predicted_choice_is_correct": bool(
                                int(predicted_choice_cpu[batch_index]) == int(true_choice_idx_eval[example_index])
                            ),
                            "predicted_vocab_token_id": int(predicted_vocab_token_cpu[batch_index]),
                            "best_non_choice_token_id": int(best_non_choice_id_cpu[batch_index]),
                            "best_non_choice_logit": float(best_non_choice_logit_cpu[batch_index]),
                            "full_vocab_entropy": float(full_entropy_cpu[batch_index]),
                            "full_vocab_entropy_normalized": float(
                                full_entropy_cpu[batch_index] / math.log(int(lm_head_weight.shape[0]))
                            ),
                            "answer_choice_entropy": float(choice_entropy_cpu[batch_index]),
                            "answer_choice_entropy_normalized": float(
                                choice_entropy_cpu[batch_index] / math.log(len(LETTERS))
                            ),
                            "answer_choice_top1_top2_logit_gap": float(choice_gap_cpu[batch_index]),
                            "true_answer_logit_minus_best_other_choice_logit": float(
                                true_choice_gap_cpu[batch_index]
                            ),
                            "true_answer_probability_within_choices": float(true_choice_prob_cpu[batch_index]),
                            "true_answer_rank_within_choices": int(true_choice_rank_cpu[batch_index]),
                            "choice_kl_to_final": float(choice_kl_to_final_cpu[batch_index]),
                            "logit_A": float(choice_logits_cpu[batch_index, 0]),
                            "logit_B": float(choice_logits_cpu[batch_index, 1]),
                            "logit_C": float(choice_logits_cpu[batch_index, 2]),
                            "logit_D": float(choice_logits_cpu[batch_index, 3]),
                            "logit_E": float(choice_logits_cpu[batch_index, 4]),
                        }
                    )

    layerwise_outputs_df = pd.DataFrame(layerwise_rows)

    for lens in lenses_eval:
        if lens is not None:
            lens.cpu()

    output_dir = resolve_out_dir(args.out_dir, args.model_id)
    tmp_dir = output_dir.parent / f"_tmp_{output_dir.name}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=False)

    examples_df.to_parquet(tmp_dir / "examples.parquet", index=False)
    clean_final_df.to_parquet(tmp_dir / "clean_final_outputs.parquet", index=False)
    clean_final_topk_df.to_parquet(tmp_dir / "clean_final_topk.parquet", index=False)
    layerwise_outputs_df.to_parquet(tmp_dir / "layerwise_outputs.parquet", index=False)
    attention_outputs_df.to_parquet(tmp_dir / "attention_outputs.parquet", index=False)
    substep_outputs_df.to_parquet(tmp_dir / "substep_outputs.parquet", index=False)
    train_history_df.to_parquet(tmp_dir / "tuned_lens_training_history.parquet", index=False)
    torch.save(
        {
            "model_id": args.model_id,
            "num_layers": num_layers,
            "hidden_size": hidden_size,
            "state_dict_by_layer_number": lens_state_dicts,
        },
        tmp_dir / "tuned_lens_state.pt",
    )

    run_config = {
        "model_id": args.model_id,
        "train_split": "train",
        "eval_split": "validation",
        "max_seq_len": int(args.max_seq_len),
        "top_k_final": int(args.top_k_final),
        "seed": int(args.seed),
        "extract_batch_size": EXTRACT_BATCH_SIZE,
        "readout_batch_size": READOUT_BATCH_SIZE,
        "attention_batch_size": ATTENTION_BATCH_SIZE,
        "substep_batch_size": SUBSTEP_BATCH_SIZE,
        "tuned_lens_batch_size": TUNED_LENS_BATCH_SIZE,
        "tuned_lens_epochs": TUNED_LENS_EPOCHS,
        "tuned_lens_lr": TUNED_LENS_LR,
        "tuned_lens_weight_decay": TUNED_LENS_WEIGHT_DECAY,
        "answer_token_ids": answer_token_ids,
        "num_train_examples": int(len(train_rows)),
        "num_eval_examples": int(len(eval_rows)),
        "num_layers": int(num_layers),
        "hidden_size": int(hidden_size),
        "model_dtype": str(model_dtype),
        "last_layer_needs_final_norm": bool(last_layer_needs_final_norm),
        "tuned_lens_target": "final_answer_choice_distribution",
        "attention_target": "decision_position_compact_choice_attention_summaries",
        "substep_target": "decision_position_compact_substep_readouts",
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
