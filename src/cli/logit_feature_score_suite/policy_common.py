"""Małe funkcje techniczne współdzielone przez pętle interwencyjne."""

from __future__ import annotations

import gc
import math

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from src.csqa.common import encode_prompts
from src.score.constants import LETTERS


def prepare_batch(
    tok: AutoTokenizer,
    texts: list[str],
    max_seq_len: int,
    input_device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    encoded = encode_prompts(texts, tok, max_seq_len)
    decision_pos = encoded.pop("decision_pos").to(input_device)
    encoded.pop("prompt_token_count")
    return {key: value.to(input_device) for key, value in encoded.items()}, decision_pos


def true_choice_tensor(batch_df: pd.DataFrame, input_device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [LETTERS.index(str(value)) for value in batch_df["answerKey"].tolist()],
        dtype=torch.long,
        device=input_device,
    )


def finite_or_nan(value: float) -> float:
    return float(value) if math.isfinite(float(value)) else np.nan


def build_output_row(
    *,
    example_id: str,
    feature_name: str,
    layer_number: int,
    max_delta_over_hidden: float,
    eligible: bool,
    accepted: bool,
    batch_index: int,
    global_index: int,
    current_supported: np.ndarray,
    current_region: np.ndarray,
    current_score: np.ndarray,
    current_score_derivative: np.ndarray,
    ascent_basis: dict[str, object],
    raw_delta_stats: dict[str, object],
    branch_stats: dict[str, object],
    accepted_step_scale: np.ndarray,
    outputs: dict[str, np.ndarray],
    clean_choice_logits: np.ndarray,
    clean_best_non_choice_logit: np.ndarray,
    clean_best_non_choice_token_id: np.ndarray,
    intervention_type: str | None = None,
) -> dict[str, object]:
    row = {
        "example_id": example_id,
        "feature_name": feature_name,
        "layer_number": layer_number,
        "max_delta_over_hidden": float(max_delta_over_hidden),
        "eligible_for_intervention": bool(eligible),
        "accepted_intervention": bool(accepted),
        "current_supported": bool(current_supported[batch_index]),
        "current_region_label": str(current_region[batch_index]),
        "steered_supported": bool(branch_stats["steered_supported"][batch_index]),
        "steered_region_label": str(branch_stats["steered_region_label"][batch_index]),
        "current_feature_value": float(ascent_basis["current_feature_value"][batch_index]),
        "current_score_value": finite_or_nan(current_score[batch_index]),
        "current_score_derivative": finite_or_nan(current_score_derivative[batch_index]),
        "accepted_step_scale": float(accepted_step_scale[batch_index]),
        "steered_feature_value_local": finite_or_nan(
            branch_stats["steered_feature_value_local"][batch_index]
        ),
        "steered_score_value_local": finite_or_nan(
            branch_stats["steered_score_value_local"][batch_index]
        ),
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
        **{
            f"steered_logit_{letter}": float(outputs["choice_logits"][batch_index, idx])
            for idx, letter in enumerate(LETTERS)
        },
        "clean_best_non_choice_token_id": int(clean_best_non_choice_token_id[global_index]),
        "clean_best_non_choice_logit": float(clean_best_non_choice_logit[global_index]),
        **{
            f"clean_logit_{letter}": float(clean_choice_logits[global_index, idx])
            for idx, letter in enumerate(LETTERS)
        },
    }
    if intervention_type is not None:
        row["intervention_type"] = intervention_type
    return row


def release_batch_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
