from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


LETTERS = ["A", "B", "C", "D", "E"]


class AffineTranslator(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


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
    }


def build_decoder_layer_output_metrics(
    choice_logits: np.ndarray,
    best_non_choice_logit: np.ndarray,
    true_choice_idx: np.ndarray,
) -> dict[str, np.ndarray]:
    shifted = choice_logits - choice_logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)
    log_probs = np.log(np.clip(probs, 1e-12, None))
    sorted_logits = np.sort(choice_logits, axis=1)[:, ::-1]
    sorted_probs = np.sort(probs, axis=1)[:, ::-1]
    pred_idx = np.argmax(choice_logits, axis=1)
    rows = np.arange(choice_logits.shape[0])
    entropy = -(probs * log_probs).sum(axis=1)
    surprisal = -log_probs
    varentropy = (probs * (surprisal - entropy[:, None]) ** 2).sum(axis=1)

    return {
        "predicted_choice_idx": pred_idx.astype(np.int64),
        "is_correct": (pred_idx == true_choice_idx).astype(bool),
        "answer_choice_entropy": entropy.astype(np.float32),
        "answer_choice_entropy_normalized": (entropy / math.log(5)).astype(np.float32),
        "choice_varentropy": varentropy.astype(np.float32),
        "choice_top1_probability": sorted_probs[:, 0].astype(np.float32),
        "answer_choice_top1_top2_logit_gap": (sorted_logits[:, 0] - sorted_logits[:, 1]).astype(np.float32),
        "best_choice_minus_best_non_choice_logit": (
            choice_logits.max(axis=1) - best_non_choice_logit
        ).astype(np.float32),
        "true_answer_probability_within_choices": probs[rows, true_choice_idx].astype(np.float32),
    }
