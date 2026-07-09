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
    compute_feature_from_token_hidden,
    parse_float_list,
    run_single_intervention_forward,
    select_top_k_layers_by_feature,
)
from src.cli.run_csqa_logit_feature_steering_pipeline import resolve_hf_token  # noqa: E402
from src.csqa.common import encode_prompts, get_decoder_layers  # noqa: E402
from src.data.load_csqa import load_csqa  # noqa: E402


LETTERS = ["A", "B", "C", "D", "E"]
DEFAULT_FEATURE_NAMES = [
    "answer_choice_entropy_normalized",
    "answer_choice_top1_top2_logit_gap",
    "answer_choice_varentropy",
]
INTERVENTION_BATCH_SIZE = 2
DEFAULT_MAX_DELTA_OVER_HIDDEN = 0.005
DEFAULT_BACKTRACK_SCALES = [1.0, 0.5, 0.25, 0.125]
GRAD_NORM_EPS = 1e-12
SCORE_IMPROVEMENT_EPS = 1e-6
DEFAULT_TRAIN_LIMIT = 2000


def now_id() -> str:
    import time

    return time.strftime("%Y%m%d-%H%M%S")


def slugify_model_id(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", model_id).strip("-")


def resolve_out_dir(out_dir: str | None, model_id: str) -> Path:
    if out_dir is None:
        run_name = f"{now_id()}_{slugify_model_id(model_id)}_csqa_logit_feature_score_pipeline"
        return REPO_ROOT / "data" / "generated" / "logit_feature_score_pipeline" / run_name
    path = Path(out_dir)
    return path if path.is_absolute() else (REPO_ROOT / path)


def all_layer_numbers(num_layers: int) -> list[int]:
    return list(range(1, num_layers + 1))


def default_limit_for_split(split_name: str) -> int | None:
    return DEFAULT_TRAIN_LIMIT if split_name == "train" else None


def parse_positive_scale_list(raw: str) -> list[float]:
    values = parse_float_list(raw)
    if any(value > 1.0 + 1e-12 for value in values):
        raise ValueError("Backtrack scales must be in (0, 1]")
    return values


def parse_feature_names(raw: str) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("No feature names provided")
    unknown = [value for value in values if value not in DEFAULT_FEATURE_NAMES]
    if unknown:
        raise ValueError(f"Unknown feature names: {unknown}. Allowed: {DEFAULT_FEATURE_NAMES}")
    return values


def augment_region_models_with_derivative(
    region_models: dict[tuple[str, int], dict[str, object]],
    distribution_grid_df: pd.DataFrame,
) -> pd.DataFrame:
    distribution_grid_df = distribution_grid_df.copy()
    distribution_grid_df["score_derivative"] = np.nan
    for (feature_name, layer_number), model in region_models.items():
        grid = np.asarray(model["grid"], dtype=np.float64)
        score = np.asarray(model["log_ratio"], dtype=np.float64)
        score_derivative = np.gradient(score, grid)
        model["score_derivative"] = score_derivative
        mask = distribution_grid_df["feature_name"].eq(feature_name) & distribution_grid_df["layer_number"].eq(
            int(layer_number)
        )
        distribution_grid_df.loc[mask, "score_derivative"] = score_derivative.astype(np.float32)
    return distribution_grid_df


def interpolate_score_state(
    feature_values: np.ndarray,
    *,
    region_model: dict[str, object],
) -> dict[str, np.ndarray]:
    values = np.asarray(feature_values, dtype=np.float64)
    grid = np.asarray(region_model["grid"], dtype=np.float64)
    score = np.asarray(region_model["log_ratio"], dtype=np.float64)
    score_derivative = np.asarray(region_model["score_derivative"], dtype=np.float64)
    region_label_grid = np.asarray(region_model["region_label"], dtype=object)
    support_low = float(region_model["support_low"])
    support_high = float(region_model["support_high"])

    supported = (values >= support_low) & (values <= support_high)
    nearest_idx = np.searchsorted(grid, values, side="left")
    nearest_idx = np.clip(nearest_idx, 0, grid.shape[0] - 1)
    left_idx = np.clip(nearest_idx - 1, 0, grid.shape[0] - 1)
    use_left = np.abs(values - grid[left_idx]) <= np.abs(values - grid[nearest_idx])
    nearest_idx = np.where(use_left, left_idx, nearest_idx)
    region_label = region_label_grid[nearest_idx].astype(object)

    interp_score = np.interp(values, grid, score, left=score[0], right=score[-1]).astype(np.float32)
    interp_score_derivative = np.interp(
        values,
        grid,
        score_derivative,
        left=score_derivative[0],
        right=score_derivative[-1],
    ).astype(np.float32)

    interp_score = np.where(supported, interp_score, np.nan).astype(np.float32)
    interp_score_derivative = np.where(supported, interp_score_derivative, np.nan).astype(np.float32)
    return {
        "supported": supported.astype(bool),
        "score_value": interp_score,
        "score_derivative": interp_score_derivative,
        "region_label": np.asarray(region_label, dtype=object),
    }


def compute_score_ascent_unit_delta(
    token_hidden: torch.Tensor,
    *,
    feature_name: str,
    layer_index_0based: int,
    maybe_apply_final_norm,
    lm_head_weight: torch.Tensor,
    answer_id_tensor_lm_head: torch.Tensor,
    vocab_size: int,
    current_score_derivative: np.ndarray,
    intervention_mask: np.ndarray,
) -> dict[str, object]:
    score_derivative_tensor = torch.as_tensor(current_score_derivative, device=token_hidden.device, dtype=torch.float32)
    with torch.enable_grad():
        base = token_hidden.detach().clone().requires_grad_(True)
        current_feature = compute_feature_from_token_hidden(
            base,
            feature_name=feature_name,
            layer_index_0based=layer_index_0based,
            maybe_apply_final_norm=maybe_apply_final_norm,
            lm_head_weight=lm_head_weight,
            answer_id_tensor_lm_head=answer_id_tensor_lm_head,
            vocab_size=vocab_size,
        )

        unit_deltas: list[torch.Tensor] = []
        feature_grad_norms: list[float] = []
        score_grad_norms: list[float] = []
        for batch_index in range(base.shape[0]):
            score_scalar = float(score_derivative_tensor[batch_index].item())
            if (not bool(intervention_mask[batch_index])) or (not math.isfinite(score_scalar)) or abs(score_scalar) <= 1e-8:
                unit_deltas.append(torch.zeros_like(base[batch_index]))
                feature_grad_norms.append(0.0)
                score_grad_norms.append(0.0)
                continue

            grad_full = torch.autograd.grad(
                current_feature[batch_index],
                base,
                retain_graph=(batch_index < (base.shape[0] - 1)),
                create_graph=False,
                allow_unused=False,
            )[0]
            feature_grad = grad_full[batch_index]
            feature_grad_norm = float(feature_grad.detach().float().norm().item())
            score_grad = score_derivative_tensor[batch_index] * feature_grad
            score_grad_norm = float(score_grad.detach().float().norm().item())
            if (not math.isfinite(score_grad_norm)) or score_grad_norm <= GRAD_NORM_EPS:
                unit_deltas.append(torch.zeros_like(base[batch_index]))
                feature_grad_norms.append(feature_grad_norm if math.isfinite(feature_grad_norm) else 0.0)
                score_grad_norms.append(0.0)
                continue

            unit_delta = score_grad / score_grad.detach().float().norm().clamp_min(1e-12)
            unit_deltas.append(unit_delta.detach())
            feature_grad_norms.append(feature_grad_norm if math.isfinite(feature_grad_norm) else 0.0)
            score_grad_norms.append(score_grad_norm)

        unit_delta = torch.stack(unit_deltas, dim=0)

    token_hidden_l2 = token_hidden.detach().float().norm(dim=-1)
    return {
        "unit_delta": unit_delta.detach(),
        "current_feature_value": current_feature.detach().cpu().numpy().astype(np.float32),
        "feature_grad_l2_norm": np.asarray(feature_grad_norms, dtype=np.float32),
        "score_grad_l2_norm": np.asarray(score_grad_norms, dtype=np.float32),
        "token_hidden_l2_norm": token_hidden_l2.detach().cpu().numpy().astype(np.float32),
    }


def build_full_cap_delta(
    token_hidden: torch.Tensor,
    *,
    unit_delta: torch.Tensor,
    intervention_mask: np.ndarray,
    max_delta_over_hidden: float,
) -> dict[str, object]:
    token_hidden_l2 = token_hidden.detach().float().norm(dim=-1)
    scale = token_hidden_l2 * float(max_delta_over_hidden)
    mask_tensor = torch.as_tensor(intervention_mask, device=token_hidden.device, dtype=torch.float32)
    delta_raw = unit_delta * (scale * mask_tensor).unsqueeze(-1).to(unit_delta.dtype)
    raw_delta_l2 = delta_raw.detach().float().norm(dim=-1)
    raw_delta_over_hidden = raw_delta_l2 / token_hidden_l2.clamp_min(1e-12)
    return {
        "delta_raw": delta_raw.detach(),
        "raw_delta_l2_norm": raw_delta_l2.detach().cpu().numpy().astype(np.float32),
        "raw_delta_over_token_hidden_l2": raw_delta_over_hidden.detach().cpu().numpy().astype(np.float32),
        "token_hidden_l2_norm": token_hidden_l2.detach().cpu().numpy().astype(np.float32),
    }


def evaluate_candidate_delta(
    token_hidden: torch.Tensor,
    *,
    delta: torch.Tensor,
    feature_name: str,
    layer_index_0based: int,
    maybe_apply_final_norm,
    lm_head_weight: torch.Tensor,
    answer_id_tensor_lm_head: torch.Tensor,
    vocab_size: int,
    region_model: dict[str, object],
) -> dict[str, np.ndarray]:
    steered_feature = compute_feature_from_token_hidden(
        token_hidden + delta,
        feature_name=feature_name,
        layer_index_0based=layer_index_0based,
        maybe_apply_final_norm=maybe_apply_final_norm,
        lm_head_weight=lm_head_weight,
        answer_id_tensor_lm_head=answer_id_tensor_lm_head,
        vocab_size=vocab_size,
    ).detach()
    feature_values = steered_feature.cpu().numpy().astype(np.float32)
    score_state = interpolate_score_state(feature_values, region_model=region_model)
    return {
        "feature_value": feature_values,
        "score_value": score_state["score_value"].astype(np.float32),
        "supported": score_state["supported"].astype(bool),
        "region_label": score_state["region_label"],
    }


def select_score_ascent_delta(
    token_hidden: torch.Tensor,
    *,
    delta_raw: torch.Tensor,
    current_score_value: np.ndarray,
    intervention_mask: np.ndarray,
    feature_name: str,
    layer_index_0based: int,
    maybe_apply_final_norm,
    lm_head_weight: torch.Tensor,
    answer_id_tensor_lm_head: torch.Tensor,
    vocab_size: int,
    region_model: dict[str, object],
    backtrack_scales: list[float],
) -> dict[str, object]:
    batch_size = token_hidden.shape[0]
    accepted_delta = torch.zeros_like(delta_raw)
    accepted_feature_value = np.full(batch_size, np.nan, dtype=np.float32)
    accepted_score_value = np.full(batch_size, np.nan, dtype=np.float32)
    accepted_supported = np.zeros(batch_size, dtype=bool)
    accepted_step_scale = np.zeros(batch_size, dtype=np.float32)
    accepted_region_label = np.asarray(["unsupported"] * batch_size, dtype=object)
    accepted_mask = np.zeros(batch_size, dtype=bool)

    current_score_value = np.asarray(current_score_value, dtype=np.float32)
    unresolved = np.asarray(intervention_mask, dtype=bool).copy()
    for scale in backtrack_scales:
        if not bool(unresolved.any()):
            break

        candidate_delta = delta_raw * float(scale)
        candidate_stats = evaluate_candidate_delta(
            token_hidden,
            delta=candidate_delta,
            feature_name=feature_name,
            layer_index_0based=layer_index_0based,
            maybe_apply_final_norm=maybe_apply_final_norm,
            lm_head_weight=lm_head_weight,
            answer_id_tensor_lm_head=answer_id_tensor_lm_head,
            vocab_size=vocab_size,
            region_model=region_model,
        )
        improved = (
            unresolved
            & candidate_stats["supported"]
            & np.isfinite(candidate_stats["score_value"])
            & (candidate_stats["score_value"] > (current_score_value + SCORE_IMPROVEMENT_EPS))
        )
        if not bool(improved.any()):
            continue

        improved_idx = np.where(improved)[0]
        accepted_delta[improved_idx] = candidate_delta[improved_idx]
        accepted_feature_value[improved_idx] = candidate_stats["feature_value"][improved_idx]
        accepted_score_value[improved_idx] = candidate_stats["score_value"][improved_idx]
        accepted_supported[improved_idx] = candidate_stats["supported"][improved_idx]
        accepted_step_scale[improved_idx] = float(scale)
        accepted_region_label[improved_idx] = candidate_stats["region_label"][improved_idx]
        accepted_mask[improved_idx] = True
        unresolved[improved_idx] = False

    delta_l2 = accepted_delta.detach().float().norm(dim=-1)
    token_hidden_l2 = token_hidden.detach().float().norm(dim=-1)
    delta_over_hidden = delta_l2 / token_hidden_l2.clamp_min(1e-12)
    return {
        "delta": accepted_delta.detach(),
        "accepted_intervention": accepted_mask.astype(bool),
        "accepted_step_scale": accepted_step_scale.astype(np.float32),
        "steered_feature_value_local": accepted_feature_value.astype(np.float32),
        "steered_score_value_local": accepted_score_value.astype(np.float32),
        "steered_supported": accepted_supported.astype(bool),
        "steered_region_label": accepted_region_label.astype(object),
        "delta_l2_norm": delta_l2.detach().cpu().numpy().astype(np.float32),
        "delta_over_token_hidden_l2": delta_over_hidden.detach().cpu().numpy().astype(np.float32),
        "token_hidden_l2_norm": token_hidden_l2.detach().cpu().numpy().astype(np.float32),
    }


def run_score_policy(
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

    with tqdm(total=total_steps, desc=f"{eval_split_name} score-ascent sweep") as pbar:
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
                outputs = run_single_intervention_forward(
                    batch=batch,
                    decision_pos=decision_pos,
                    model=model,
                    steering_module=steering_module,
                    delta=selected_delta["delta"],
                    true_choice_idx=true_choice_idx,
                    answer_id_tensor_cpu=answer_id_tensor_cpu,
                )

                for batch_index, row in batch_df.iterrows():
                    global_index = batch_indices[batch_index]
                    rows.append(
                        {
                            "example_id": row["example_id"],
                            "feature_name": feature_name,
                            "layer_number": layer_number,
                            "max_delta_over_hidden": float(max_delta_over_hidden),
                            "eligible_for_intervention": bool(effective_eligible[batch_index]),
                            "accepted_intervention": bool(selected_delta["accepted_intervention"][batch_index]),
                            "current_supported": bool(batch_current_supported[batch_index]),
                            "current_region_label": str(batch_current_region[batch_index]),
                            "steered_supported": bool(selected_delta["steered_supported"][batch_index]),
                            "steered_region_label": str(selected_delta["steered_region_label"][batch_index]),
                            "current_feature_value": float(ascent_basis["current_feature_value"][batch_index]),
                            "current_score_value": float(batch_current_score[batch_index])
                            if math.isfinite(float(batch_current_score[batch_index]))
                            else np.nan,
                            "current_score_derivative": float(batch_current_score_derivative[batch_index])
                            if math.isfinite(float(batch_current_score_derivative[batch_index]))
                            else np.nan,
                            "accepted_step_scale": float(selected_delta["accepted_step_scale"][batch_index]),
                            "steered_feature_value_local": float(selected_delta["steered_feature_value_local"][batch_index])
                            if math.isfinite(float(selected_delta["steered_feature_value_local"][batch_index]))
                            else np.nan,
                            "steered_score_value_local": float(selected_delta["steered_score_value_local"][batch_index])
                            if math.isfinite(float(selected_delta["steered_score_value_local"][batch_index]))
                            else np.nan,
                            "feature_grad_l2_norm": float(ascent_basis["feature_grad_l2_norm"][batch_index]),
                            "score_grad_l2_norm": float(ascent_basis["score_grad_l2_norm"][batch_index]),
                            "raw_delta_l2_norm": float(raw_delta_stats["raw_delta_l2_norm"][batch_index]),
                            "raw_delta_over_token_hidden_l2": float(
                                raw_delta_stats["raw_delta_over_token_hidden_l2"][batch_index]
                            ),
                            "delta_l2_norm": float(selected_delta["delta_l2_norm"][batch_index]),
                            "delta_over_token_hidden_l2": float(
                                selected_delta["delta_over_token_hidden_l2"][batch_index]
                            ),
                            "token_hidden_l2_norm": float(selected_delta["token_hidden_l2_norm"][batch_index]),
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
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

    policy_outputs_df = run_score_policy(
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
    policy_outputs_df.to_parquet(out_dir / f"{args.eval_split}_score_ascent_policy_outputs_raw.parquet", index=False)

    run_config = {
        "dataset": "csqa",
        "model_id": args.model_id,
        "seed": int(args.seed),
        "max_seq_len": int(args.max_seq_len),
        "fit_split": args.fit_split,
        "eval_split": args.eval_split,
        "fit_limit": fit_limit,
        "eval_limit": eval_limit,
        "method": "logit_feature_score_ascent",
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
