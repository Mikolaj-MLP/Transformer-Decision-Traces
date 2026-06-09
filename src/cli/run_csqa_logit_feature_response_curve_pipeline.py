from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.neighbors import KernelDensity
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
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
from src.cli.run_csqa_logit_feature_steering_pipeline import (  # noqa: E402
    compute_feature_tensor,
    maybe_clone_to_float_cpu,
    resolve_hf_token,
)
from src.csqa.common import (  # noqa: E402
    encode_prompts,
    get_decoder_layers,
    repack_output_hidden,
    select_full_logits_at_decision,
    summarize_decision_logits,
    unpack_output_hidden,
)
from src.data.load_csqa import load_csqa  # noqa: E402


LETTERS = ["A", "B", "C", "D", "E"]
FEATURE_NAMES = [
    "answer_choice_entropy_normalized",
    "answer_choice_top1_top2_logit_gap",
    "answer_choice_varentropy",
]
DEFAULT_STEP_FRACTIONS = [0.25, 0.5, 1.0, 1.5]
DEFAULT_MAX_DELTA_OVER_HIDDEN_CAPS = [0.005, 0.01]
INTERVENTION_BATCH_SIZE = 2
GRID_POINTS = 512
GOOD_REGION_LOG_RATIO_THRESHOLD = math.log(1.5)
GRAD_NORM_EPS = 1e-12
KDE_JITTER_SCALE = 1e-6
SUPPORT_LOWER_QUANTILE = 0.01
SUPPORT_UPPER_QUANTILE = 0.99
KDE_BANDWIDTH_MULTIPLIER = 1.5
LOG_RATIO_SMOOTHING_SIGMA_BANDWIDTHS = 1.0


def now_id() -> str:
    import time

    return time.strftime("%Y%m%d-%H%M%S")


def slugify_model_id(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", model_id).strip("-")


def resolve_out_dir(out_dir: str | None, model_id: str) -> Path:
    if out_dir is None:
        run_name = f"{now_id()}_{slugify_model_id(model_id)}_csqa_logit_feature_response_curve_pipeline"
        return REPO_ROOT / "data" / "generated" / "logit_feature_response_curve_pipeline" / run_name
    path = Path(out_dir)
    return path if path.is_absolute() else (REPO_ROOT / path)


def all_layer_numbers(num_layers: int) -> list[int]:
    return list(range(1, num_layers + 1))


def summarize_logit_features(
    *,
    feature_names: list[str],
    full_logits: torch.Tensor,
    choice_logits: torch.Tensor,
    vocab_size: int,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for feature_name in feature_names:
        out[feature_name] = maybe_clone_to_float_cpu(
            compute_feature_tensor(
                feature_name=feature_name,
                full_logits=full_logits,
                choice_logits=choice_logits,
                vocab_size=vocab_size,
            )
        )
    return out


def build_feature_table(
    *,
    split_name: str,
    cache: dict[str, object],
    feature_names: list[str],
    maybe_apply_final_norm,
    lm_head_weight: torch.Tensor,
    answer_id_tensor_lm_head: torch.Tensor,
    input_device: torch.device,
    vocab_size: int,
    active_layer_numbers: list[int],
) -> pd.DataFrame:
    hidden = cache["hidden"]
    clean_is_correct = cache["clean_is_correct"]
    example_rows = pd.DataFrame(cache["example_rows"])[["example_id", "split"]]
    rows: list[dict[str, object]] = []

    layer_indices = [layer_number - 1 for layer_number in active_layer_numbers]
    for layer_index in tqdm(layer_indices, desc=f"{split_name} feature extraction"):
        layer_hidden = hidden[:, layer_index, :]
        feature_blocks: dict[str, list[np.ndarray]] = {feature_name: [] for feature_name in feature_names}

        for start in range(0, layer_hidden.shape[0], READOUT_BATCH_SIZE):
            end = start + READOUT_BATCH_SIZE
            hidden_batch = layer_hidden[start:end].to(input_device)
            readout = maybe_apply_final_norm(hidden_batch.float(), layer_index)
            full_logits = torch.matmul(readout.to(lm_head_weight.dtype), lm_head_weight.T).float()
            choice_logits = full_logits.index_select(1, answer_id_tensor_lm_head)
            feature_batch = summarize_logit_features(
                feature_names=feature_names,
                full_logits=full_logits,
                choice_logits=choice_logits,
                vocab_size=vocab_size,
            )
            for feature_name, values in feature_batch.items():
                feature_blocks[feature_name].append(values)

            del hidden_batch, readout, full_logits, choice_logits

        feature_values_by_name = {
            feature_name: np.concatenate(blocks, axis=0)
            for feature_name, blocks in feature_blocks.items()
        }

        for example_index, example_id in enumerate(example_rows["example_id"].tolist()):
            final_error = int(not bool(clean_is_correct[example_index]))
            for feature_name in feature_names:
                rows.append(
                    {
                        "example_id": example_id,
                        "split": split_name,
                        "layer_number": layer_index + 1,
                        "feature_name": feature_name,
                        "feature_value": float(feature_values_by_name[feature_name][example_index]),
                        "final_error": final_error,
                        "clean_is_correct": bool(clean_is_correct[example_index]),
                    }
                )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return pd.DataFrame(rows)


def empirical_cdf(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    sorted_values = np.sort(values)
    return np.searchsorted(sorted_values, grid, side="right") / float(sorted_values.shape[0])


def compute_ks_statistic(correct_values: np.ndarray, incorrect_values: np.ndarray) -> float:
    pooled_grid = np.sort(np.unique(np.concatenate([correct_values, incorrect_values], axis=0)))
    if pooled_grid.size == 0:
        return 0.0
    cdf_good = empirical_cdf(correct_values, pooled_grid)
    cdf_bad = empirical_cdf(incorrect_values, pooled_grid)
    return float(np.max(np.abs(cdf_good - cdf_bad)))


def build_separation_summary(feature_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped_keys = sorted(
        feature_df[["feature_name", "layer_number"]].drop_duplicates().itertuples(index=False, name=None)
    )
    for feature_name, layer_number in grouped_keys:
        part = feature_df.loc[
            feature_df["feature_name"].eq(feature_name)
            & feature_df["layer_number"].eq(layer_number)
        ].copy()
        correct_values = part.loc[part["clean_is_correct"], "feature_value"].to_numpy(dtype=float)
        incorrect_values = part.loc[~part["clean_is_correct"], "feature_value"].to_numpy(dtype=float)
        rows.append(
            {
                "feature_name": feature_name,
                "layer_number": int(layer_number),
                "ks_statistic": compute_ks_statistic(correct_values, incorrect_values),
                "correct_mean": float(correct_values.mean()),
                "incorrect_mean": float(incorrect_values.mean()),
                "correct_median": float(np.median(correct_values)),
                "incorrect_median": float(np.median(incorrect_values)),
                "correct_std": float(correct_values.std(ddof=0)),
                "incorrect_std": float(incorrect_values.std(ddof=0)),
                "n_correct": int(correct_values.shape[0]),
                "n_incorrect": int(incorrect_values.shape[0]),
            }
        )
    return pd.DataFrame(rows)


def select_top_k_layers_by_feature(
    separation_df: pd.DataFrame,
    *,
    feature_names: list[str],
    top_k: int,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for feature_name in feature_names:
        part = separation_df.loc[separation_df["feature_name"].eq(feature_name)].copy()
        part = part.sort_values(["ks_statistic", "layer_number"], ascending=[False, True]).head(top_k).copy()
        part["selection_rank"] = np.arange(1, len(part) + 1)
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


def silverman_bandwidth(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    n = values.shape[0]
    if n <= 1:
        return 0.1
    std = float(values.std(ddof=1)) if n > 1 else 0.0
    iqr = float(np.subtract(*np.percentile(values, [75, 25])))
    robust = iqr / 1.34 if iqr > 0 else std
    scale = min(x for x in [std, robust] if x > 0) if any(x > 0 for x in [std, robust]) else 1.0
    bw = 0.9 * scale * (n ** (-1 / 5))
    return float(max(bw, 1e-3))


def fit_kde(values: np.ndarray, bandwidth: float) -> KernelDensity:
    model = KernelDensity(kernel="gaussian", bandwidth=float(bandwidth))
    model.fit(values.reshape(-1, 1))
    return model


def gaussian_kernel_1d(sigma_grid: float) -> np.ndarray:
    if sigma_grid <= 1e-8:
        return np.array([1.0], dtype=np.float64)
    radius = max(1, int(math.ceil(3.0 * sigma_grid)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma_grid) ** 2)
    kernel /= np.sum(kernel)
    return kernel


def smooth_1d(values: np.ndarray, sigma_grid: float) -> np.ndarray:
    kernel = gaussian_kernel_1d(float(sigma_grid))
    if kernel.shape[0] == 1:
        return values.copy()
    pad = kernel.shape[0] // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def parse_float_list(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("No float values provided")
    if any(value <= 0 for value in values):
        raise ValueError("All values must be positive")
    return values


def build_distribution_models(
    *,
    fit_feature_df: pd.DataFrame,
    selected_layers_df: pd.DataFrame,
    log_ratio_threshold: float,
    grid_points: int,
) -> tuple[dict[tuple[str, int], dict[str, object]], pd.DataFrame]:
    models: dict[tuple[str, int], dict[str, object]] = {}
    rows: list[dict[str, object]] = []

    for feature_name, layer_number in selected_layers_df[["feature_name", "layer_number"]].itertuples(index=False):
        part = fit_feature_df.loc[
            fit_feature_df["feature_name"].eq(feature_name)
            & fit_feature_df["layer_number"].eq(layer_number)
        ].copy()
        correct_values = part.loc[part["clean_is_correct"], "feature_value"].to_numpy(dtype=float)
        incorrect_values = part.loc[~part["clean_is_correct"], "feature_value"].to_numpy(dtype=float)
        pooled = np.concatenate([correct_values, incorrect_values], axis=0)

        if np.unique(pooled).shape[0] <= 1:
            pooled = pooled + np.random.default_rng(42).normal(0.0, KDE_JITTER_SCALE, size=pooled.shape[0])
            correct_values = pooled[: correct_values.shape[0]]
            incorrect_values = pooled[correct_values.shape[0] :]

        bandwidth = KDE_BANDWIDTH_MULTIPLIER * silverman_bandwidth(pooled)
        kde_good = fit_kde(correct_values, bandwidth)
        kde_bad = fit_kde(incorrect_values, bandwidth)

        data_min = float(np.min(pooled))
        data_max = float(np.max(pooled))
        span = max(data_max - data_min, bandwidth * 6.0, 1e-3)
        pad = max(span * 0.15, bandwidth * 3.0)
        grid = np.linspace(data_min - pad, data_max + pad, grid_points, dtype=np.float64)

        log_p_good = kde_good.score_samples(grid.reshape(-1, 1))
        log_p_bad = kde_bad.score_samples(grid.reshape(-1, 1))
        log_ratio_raw = log_p_good - log_p_bad
        grid_step = float(grid[1] - grid[0]) if grid.shape[0] > 1 else 1.0
        sigma_x = max(bandwidth * LOG_RATIO_SMOOTHING_SIGMA_BANDWIDTHS, grid_step)
        sigma_grid = sigma_x / max(grid_step, 1e-12)
        log_ratio = smooth_1d(log_ratio_raw, sigma_grid)

        support_low = float(np.quantile(pooled, SUPPORT_LOWER_QUANTILE))
        support_high = float(np.quantile(pooled, SUPPORT_UPPER_QUANTILE))
        supported_mask = (grid >= support_low) & (grid <= support_high)

        good_mask = supported_mask & (log_ratio >= log_ratio_threshold)
        bad_mask = supported_mask & (log_ratio <= -log_ratio_threshold)
        if not bool(good_mask.any()):
            supported_indices = np.where(supported_mask)[0]
            candidate_indices = supported_indices if supported_indices.size > 0 else np.arange(grid.shape[0])
            max_idx = int(candidate_indices[np.argmax(log_ratio[candidate_indices])])
            good_mask[max_idx] = True
        region_label = np.full(grid.shape[0], "unsupported", dtype=object)
        region_label[supported_mask] = "neutral"
        region_label[bad_mask] = "bad"
        region_label[good_mask] = "good"

        models[(feature_name, int(layer_number))] = {
            "grid": grid,
            "log_p_good": log_p_good,
            "log_p_bad": log_p_bad,
            "log_ratio": log_ratio,
            "log_ratio_raw": log_ratio_raw,
            "region_label": region_label,
            "good_mask": good_mask,
            "supported_mask": supported_mask,
            "bandwidth": bandwidth,
            "support_low": support_low,
            "support_high": support_high,
            "smoothing_sigma_grid": float(sigma_grid),
        }

        for idx in range(grid.shape[0]):
            rows.append(
                {
                    "feature_name": feature_name,
                    "layer_number": int(layer_number),
                    "grid_x": float(grid[idx]),
                    "log_p_good": float(log_p_good[idx]),
                    "log_p_bad": float(log_p_bad[idx]),
                    "p_good": float(math.exp(log_p_good[idx])),
                    "p_bad": float(math.exp(log_p_bad[idx])),
                    "log_density_ratio_raw": float(log_ratio_raw[idx]),
                    "log_density_ratio": float(log_ratio[idx]),
                    "region_label": str(region_label[idx]),
                    "is_supported": bool(supported_mask[idx]),
                    "bandwidth": float(bandwidth),
                    "support_low": support_low,
                    "support_high": support_high,
                    "smoothing_sigma_grid": float(sigma_grid),
                }
            )

    return models, pd.DataFrame(rows)


def current_region_and_target(
    *,
    current_value: float,
    region_model: dict[str, object],
) -> tuple[str, float | None, float | None]:
    grid = region_model["grid"]
    region_label = region_model["region_label"]
    good_mask = region_model["good_mask"]

    nearest_idx = int(np.argmin(np.abs(grid - current_value)))
    current_region = str(region_label[nearest_idx])
    good_indices = np.where(good_mask)[0]
    if good_indices.size == 0:
        return current_region, None, None

    target_idx = int(good_indices[np.argmin(np.abs(grid[good_indices] - current_value))])
    target_value = float(grid[target_idx])
    distance = float(target_value - current_value)
    return current_region, target_value, distance


def compute_feature_from_token_hidden(
    token_hidden: torch.Tensor,
    *,
    feature_name: str,
    layer_index_0based: int,
    maybe_apply_final_norm,
    lm_head_weight: torch.Tensor,
    answer_id_tensor_lm_head: torch.Tensor,
    vocab_size: int,
) -> torch.Tensor:
    readout = maybe_apply_final_norm(token_hidden, layer_index_0based)
    full_logits = torch.matmul(readout.to(lm_head_weight.dtype), lm_head_weight.T).float()
    choice_logits = full_logits.index_select(1, answer_id_tensor_lm_head)
    return compute_feature_tensor(
        feature_name=feature_name,
        full_logits=full_logits,
        choice_logits=choice_logits,
        vocab_size=vocab_size,
    )


def compute_directional_delta_basis(
    token_hidden: torch.Tensor,
    *,
    feature_name: str,
    layer_index_0based: int,
    maybe_apply_final_norm,
    lm_head_weight: torch.Tensor,
    answer_id_tensor_lm_head: torch.Tensor,
    vocab_size: int,
    target_feature_value: np.ndarray,
    intervention_mask: np.ndarray,
) -> dict[str, object]:
    target_tensor = torch.as_tensor(target_feature_value, device=token_hidden.device, dtype=torch.float32)
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
        desired_shift = target_tensor - current_feature.detach()

        deltas: list[torch.Tensor] = []
        grad_norms: list[float] = []
        for batch_index in range(base.shape[0]):
            if (not bool(intervention_mask[batch_index])) or (abs(float(desired_shift[batch_index].item())) <= 1e-8):
                deltas.append(torch.zeros_like(base[batch_index]))
                grad_norms.append(0.0)
                continue

            grad_full = torch.autograd.grad(
                current_feature[batch_index],
                base,
                retain_graph=(batch_index < (base.shape[0] - 1)),
                create_graph=False,
                allow_unused=False,
            )[0]
            grad_i = grad_full[batch_index]
            grad_norm_sq = torch.dot(grad_i.float(), grad_i.float())
            grad_norm = float(torch.sqrt(torch.clamp(grad_norm_sq, min=0.0)).item())
            if (not math.isfinite(grad_norm)) or grad_norm_sq.item() <= GRAD_NORM_EPS:
                deltas.append(torch.zeros_like(base[batch_index]))
                grad_norms.append(0.0)
                continue

            delta_i = (desired_shift[batch_index] / grad_norm_sq) * grad_i
            deltas.append(delta_i.detach())
            grad_norms.append(grad_norm)

        basis_delta = torch.stack(deltas, dim=0)

    token_hidden_l2 = token_hidden.detach().float().norm(dim=-1)
    basis_l2 = basis_delta.detach().float().norm(dim=-1)
    basis_over_hidden = basis_l2 / token_hidden_l2.clamp_min(1e-12)

    return {
        "basis_delta": basis_delta.detach(),
        "current_feature_value": current_feature.detach().cpu().numpy().astype(np.float32),
        "target_feature_value": target_tensor.detach().cpu().numpy().astype(np.float32),
        "full_distance_to_good": desired_shift.detach().cpu().numpy().astype(np.float32),
        "grad_l2_norm": np.asarray(grad_norms, dtype=np.float32),
        "basis_delta_l2_norm": basis_l2.detach().cpu().numpy().astype(np.float32),
        "basis_delta_over_token_hidden_l2": basis_over_hidden.detach().cpu().numpy().astype(np.float32),
        "token_hidden_l2_norm": token_hidden_l2.detach().cpu().numpy().astype(np.float32),
    }


def summarize_delta_application(
    token_hidden: torch.Tensor,
    *,
    delta: torch.Tensor,
    feature_name: str,
    layer_index_0based: int,
    maybe_apply_final_norm,
    lm_head_weight: torch.Tensor,
    answer_id_tensor_lm_head: torch.Tensor,
    vocab_size: int,
    grad_l2_norm: np.ndarray,
) -> dict[str, object]:
    steered_feature = compute_feature_from_token_hidden(
        token_hidden + delta,
        feature_name=feature_name,
        layer_index_0based=layer_index_0based,
        maybe_apply_final_norm=maybe_apply_final_norm,
        lm_head_weight=lm_head_weight,
        answer_id_tensor_lm_head=answer_id_tensor_lm_head,
        vocab_size=vocab_size,
    ).detach()

    token_hidden_l2 = token_hidden.detach().float().norm(dim=-1)
    delta_l2 = delta.detach().float().norm(dim=-1)
    delta_over_hidden = delta_l2 / token_hidden_l2.clamp_min(1e-12)

    return {
        "steered_feature_value_local": steered_feature.cpu().numpy().astype(np.float32),
        "grad_l2_norm": np.asarray(grad_l2_norm, dtype=np.float32),
        "delta_l2_norm": delta_l2.detach().cpu().numpy().astype(np.float32),
        "delta_over_token_hidden_l2": delta_over_hidden.detach().cpu().numpy().astype(np.float32),
        "token_hidden_l2_norm": token_hidden_l2.detach().cpu().numpy().astype(np.float32),
    }


def apply_delta_over_hidden_cap(
    token_hidden: torch.Tensor,
    *,
    delta_raw: torch.Tensor,
    max_delta_over_hidden_cap: float,
) -> dict[str, object]:
    token_hidden_l2 = token_hidden.detach().float().norm(dim=-1)
    raw_delta_l2 = delta_raw.detach().float().norm(dim=-1)
    raw_delta_over_hidden = raw_delta_l2 / token_hidden_l2.clamp_min(1e-12)

    cap_tensor = torch.full_like(raw_delta_over_hidden, float(max_delta_over_hidden_cap))
    safe_ratio = cap_tensor / raw_delta_over_hidden.clamp_min(1e-12)
    scale = torch.minimum(torch.ones_like(raw_delta_over_hidden), safe_ratio)
    scale = torch.where(torch.isfinite(scale), scale, torch.ones_like(scale))
    delta_capped = delta_raw * scale.unsqueeze(-1).to(delta_raw.dtype)

    capped_delta_l2 = delta_capped.detach().float().norm(dim=-1)
    capped_delta_over_hidden = capped_delta_l2 / token_hidden_l2.clamp_min(1e-12)
    was_capped = raw_delta_over_hidden > float(max_delta_over_hidden_cap) + 1e-12

    return {
        "delta_capped": delta_capped,
        "token_hidden_l2_norm": token_hidden_l2.detach().cpu().numpy().astype(np.float32),
        "raw_delta_l2_norm": raw_delta_l2.detach().cpu().numpy().astype(np.float32),
        "raw_delta_over_token_hidden_l2": raw_delta_over_hidden.detach().cpu().numpy().astype(np.float32),
        "capped_delta_l2_norm": capped_delta_l2.detach().cpu().numpy().astype(np.float32),
        "capped_delta_over_token_hidden_l2": capped_delta_over_hidden.detach().cpu().numpy().astype(np.float32),
        "cap_scale": scale.detach().cpu().numpy().astype(np.float32),
        "was_capped": was_capped.detach().cpu().numpy().astype(bool),
    }


def run_single_intervention_forward(
    *,
    batch: dict[str, torch.Tensor],
    decision_pos: torch.Tensor,
    model: AutoModelForCausalLM,
    steering_module,
    delta: torch.Tensor,
    true_choice_idx: torch.Tensor,
    answer_id_tensor_cpu: torch.Tensor,
) -> dict[str, np.ndarray]:
    def steering_hook(module, inputs, output):
        hidden = unpack_output_hidden(output)
        row_idx = torch.arange(hidden.shape[0], device=hidden.device)
        hidden_out = hidden.clone()
        hidden_out[row_idx, decision_pos] = hidden[row_idx, decision_pos] + delta.to(
            hidden.device,
            dtype=hidden.dtype,
        )
        return repack_output_hidden(output, hidden_out)

    handle = steering_module.register_forward_hook(steering_hook)
    try:
        with torch.no_grad():
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
    best_non_choice_token_id_cpu = torch.argmax(masked_logits, dim=-1).detach().cpu().numpy().astype(np.int64)

    return {
        "choice_logits": choice_logits_cpu,
        "best_non_choice_logit": best_non_choice_logit_cpu,
        "best_non_choice_token_id": best_non_choice_token_id_cpu,
    }


def run_response_curve_policy(
    *,
    eval_rows: pd.DataFrame,
    eval_cache: dict[str, object],
    eval_feature_df: pd.DataFrame,
    selected_layers_df: pd.DataFrame,
    region_models: dict[tuple[str, int], dict[str, object]],
    step_fractions: list[float],
    max_delta_over_hidden_caps: list[float],
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
    validation_choice_logits = eval_cache["clean_choice_logits"]
    validation_best_non_choice_logit = eval_cache["clean_best_non_choice_logit"]
    validation_best_non_choice_token_id = eval_cache["clean_best_non_choice_token_id"]
    example_ids = [row["example_id"] for row in eval_cache["example_rows"]]
    example_id_to_index = {example_id: idx for idx, example_id in enumerate(example_ids)}

    selected_pairs = [
        (str(row.feature_name), int(row.layer_number))
        for row in selected_layers_df.itertuples(index=False)
    ]
    num_eval_batches = math.ceil(len(eval_rows) / INTERVENTION_BATCH_SIZE)
    total_steps = len(selected_pairs) * num_eval_batches * len(step_fractions) * 2 * len(max_delta_over_hidden_caps)
    rows: list[dict[str, object]] = []

    with tqdm(total=total_steps, desc=f"{eval_split_name} response-curve sweep") as pbar:
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

            current_region_all: list[str] = []
            target_good_all: list[float] = []
            full_distance_all: list[float] = []
            eligible_all: list[bool] = []
            for value in current_feature_all.tolist():
                current_region, target_good, full_distance = current_region_and_target(
                    current_value=float(value),
                    region_model=region_model,
                )
                current_region_all.append(current_region)
                target_good_all.append(np.nan if target_good is None else float(target_good))
                full_distance_all.append(np.nan if full_distance is None else float(full_distance))
                eligible_all.append(
                    bool(target_good is not None)
                    and abs(float(full_distance)) > 1e-8
                    and current_region not in {"good", "unsupported"}
                )

            for start in range(0, len(eval_rows), INTERVENTION_BATCH_SIZE):
                batch_df = eval_rows.iloc[start:start + INTERVENTION_BATCH_SIZE].reset_index(drop=True)
                batch_indices = [example_id_to_index[example_id] for example_id in batch_df["example_id"].tolist()]
                batch_current_region = [current_region_all[idx] for idx in batch_indices]
                batch_target_good = np.asarray([target_good_all[idx] for idx in batch_indices], dtype=np.float32)
                batch_full_distance = np.asarray([full_distance_all[idx] for idx in batch_indices], dtype=np.float32)
                batch_eligible = np.asarray([eligible_all[idx] for idx in batch_indices], dtype=bool)

                if not bool(batch_eligible.any()):
                    for max_delta_over_hidden_cap in max_delta_over_hidden_caps:
                        for step_fraction in step_fractions:
                            for direction in ["toward_good", "away_from_good"]:
                                for batch_index, row in batch_df.iterrows():
                                    global_index = batch_indices[batch_index]
                                    rows.append(
                                        {
                                            "example_id": row["example_id"],
                                            "feature_name": feature_name,
                                            "layer_number": layer_number,
                                            "step_fraction": float(step_fraction),
                                            "direction": direction,
                                            "max_delta_over_hidden_cap": float(max_delta_over_hidden_cap),
                                            "eligible_for_intervention": False,
                                            "current_region_label": str(batch_current_region[batch_index]),
                                            "current_feature_value": float(current_feature_all[global_index]),
                                            "target_good_value": float(batch_target_good[batch_index]) if math.isfinite(float(batch_target_good[batch_index])) else np.nan,
                                            "full_distance_to_good": float(batch_full_distance[batch_index]) if math.isfinite(float(batch_full_distance[batch_index])) else np.nan,
                                            "applied_feature_shift_fraction": float(step_fraction),
                                            "steered_feature_value_local": np.nan,
                                            "grad_l2_norm": 0.0,
                                            "raw_delta_l2_norm": 0.0,
                                            "raw_delta_over_token_hidden_l2": 0.0,
                                            "cap_scale": 1.0,
                                            "was_capped": False,
                                            "delta_l2_norm": 0.0,
                                            "delta_over_token_hidden_l2": 0.0,
                                            "token_hidden_l2_norm": np.nan,
                                            "steered_best_non_choice_token_id": int(validation_best_non_choice_token_id[global_index]),
                                            "steered_best_non_choice_logit": float(validation_best_non_choice_logit[global_index]),
                                            "steered_logit_A": float(validation_choice_logits[global_index, 0]),
                                            "steered_logit_B": float(validation_choice_logits[global_index, 1]),
                                            "steered_logit_C": float(validation_choice_logits[global_index, 2]),
                                            "steered_logit_D": float(validation_choice_logits[global_index, 3]),
                                            "steered_logit_E": float(validation_choice_logits[global_index, 4]),
                                        }
                                    )
                                pbar.update(1)
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
                token_hidden = eval_cache["hidden"][batch_indices, layer_number - 1, :].to(input_device)
                delta_basis = compute_directional_delta_basis(
                    token_hidden,
                    feature_name=feature_name,
                    layer_index_0based=layer_number - 1,
                    maybe_apply_final_norm=maybe_apply_final_norm,
                    lm_head_weight=lm_head_weight,
                    answer_id_tensor_lm_head=answer_id_tensor_lm_head.to(input_device),
                    vocab_size=vocab_size,
                    target_feature_value=batch_target_good,
                    intervention_mask=batch_eligible,
                )
                basis_delta = delta_basis["basis_delta"]

                for max_delta_over_hidden_cap in max_delta_over_hidden_caps:
                    for direction, direction_sign in [("toward_good", 1.0), ("away_from_good", -1.0)]:
                        for step_fraction in step_fractions:
                            delta_raw = (direction_sign * float(step_fraction)) * basis_delta
                            delta_cap = apply_delta_over_hidden_cap(
                                token_hidden,
                                delta_raw=delta_raw,
                                max_delta_over_hidden_cap=float(max_delta_over_hidden_cap),
                            )
                            delta = delta_cap["delta_capped"]
                            delta_stats = summarize_delta_application(
                                token_hidden,
                                delta=delta,
                                feature_name=feature_name,
                                layer_index_0based=layer_number - 1,
                                maybe_apply_final_norm=maybe_apply_final_norm,
                                lm_head_weight=lm_head_weight,
                                answer_id_tensor_lm_head=answer_id_tensor_lm_head.to(input_device),
                                vocab_size=vocab_size,
                                grad_l2_norm=delta_basis["grad_l2_norm"],
                            )
                            outputs = run_single_intervention_forward(
                                batch=batch,
                                decision_pos=decision_pos,
                                model=model,
                                steering_module=steering_module,
                                delta=delta,
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
                                        "step_fraction": float(step_fraction),
                                        "direction": direction,
                                        "max_delta_over_hidden_cap": float(max_delta_over_hidden_cap),
                                        "eligible_for_intervention": bool(batch_eligible[batch_index]),
                                        "current_region_label": str(batch_current_region[batch_index]),
                                        "current_feature_value": float(delta_basis["current_feature_value"][batch_index]),
                                        "target_good_value": float(batch_target_good[batch_index]) if math.isfinite(float(batch_target_good[batch_index])) else np.nan,
                                        "full_distance_to_good": float(delta_basis["full_distance_to_good"][batch_index]),
                                        "applied_feature_shift_fraction": float(direction_sign * step_fraction),
                                        "steered_feature_value_local": float(delta_stats["steered_feature_value_local"][batch_index]),
                                        "grad_l2_norm": float(delta_stats["grad_l2_norm"][batch_index]),
                                        "raw_delta_l2_norm": float(delta_cap["raw_delta_l2_norm"][batch_index]),
                                        "raw_delta_over_token_hidden_l2": float(delta_cap["raw_delta_over_token_hidden_l2"][batch_index]),
                                        "cap_scale": float(delta_cap["cap_scale"][batch_index]),
                                        "was_capped": bool(delta_cap["was_capped"][batch_index]),
                                        "delta_l2_norm": float(delta_stats["delta_l2_norm"][batch_index]),
                                        "delta_over_token_hidden_l2": float(delta_stats["delta_over_token_hidden_l2"][batch_index]),
                                        "token_hidden_l2_norm": float(delta_stats["token_hidden_l2_norm"][batch_index]),
                                        "steered_best_non_choice_token_id": int(outputs["best_non_choice_token_id"][batch_index]),
                                        "steered_best_non_choice_logit": float(outputs["best_non_choice_logit"][batch_index]),
                                        "steered_logit_A": float(outputs["choice_logits"][batch_index, 0]),
                                        "steered_logit_B": float(outputs["choice_logits"][batch_index, 1]),
                                        "steered_logit_C": float(outputs["choice_logits"][batch_index, 2]),
                                        "steered_logit_D": float(outputs["choice_logits"][batch_index, 3]),
                                        "steered_logit_E": float(outputs["choice_logits"][batch_index, 4]),
                                    }
                                )

                            pbar.update(1)

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    return pd.DataFrame(rows)


def parse_step_fractions(raw: str) -> list[float]:
    return parse_float_list(raw)


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
    parser.add_argument("--step-fractions", type=str, default="0.25,0.5,1.0,1.5")
    parser.add_argument(
        "--max-delta-over-hidden-caps",
        type=str,
        default=",".join(str(x) for x in DEFAULT_MAX_DELTA_OVER_HIDDEN_CAPS),
    )
    parser.add_argument("--good-threshold-log-ratio", type=float, default=GOOD_REGION_LOG_RATIO_THRESHOLD)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    fit_limit = args.fit_limit if args.fit_limit is not None else args.train_limit
    eval_limit = args.eval_limit if args.eval_limit is not None else args.validation_limit
    if args.fit_split == args.eval_split:
        raise ValueError("--fit-split and --eval-split must be different")
    step_fractions = parse_step_fractions(args.step_fractions)
    max_delta_over_hidden_caps = parse_float_list(args.max_delta_over_hidden_caps)

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
                "feature_names": FEATURE_NAMES,
                "top_k_layers_per_feature": int(args.top_k_layers_per_feature),
                "step_fractions": step_fractions,
                "max_delta_over_hidden_caps": max_delta_over_hidden_caps,
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
        feature_names=FEATURE_NAMES,
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
        feature_names=FEATURE_NAMES,
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
        feature_names=FEATURE_NAMES,
        top_k=int(args.top_k_layers_per_feature),
    )
    region_models, distribution_grid_df = build_distribution_models(
        fit_feature_df=fit_feature_df,
        selected_layers_df=selected_layers_df,
        log_ratio_threshold=float(args.good_threshold_log_ratio),
        grid_points=GRID_POINTS,
    )

    policy_outputs_df = run_response_curve_policy(
        eval_rows=eval_rows,
        eval_cache=eval_cache,
        eval_feature_df=eval_feature_df,
        selected_layers_df=selected_layers_df,
        region_models=region_models,
        step_fractions=step_fractions,
        max_delta_over_hidden_caps=max_delta_over_hidden_caps,
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
    separation_df.to_parquet(out_dir / "feature_layer_separation_summary.parquet", index=False)
    selected_layers_df.to_parquet(out_dir / "selected_layers_by_feature.parquet", index=False)
    distribution_grid_df.to_parquet(out_dir / "fit_distribution_grid.parquet", index=False)
    policy_outputs_df.to_parquet(out_dir / f"{args.eval_split}_response_curve_policy_outputs_raw.parquet", index=False)

    run_config = {
        "dataset": "csqa",
        "model_id": args.model_id,
        "seed": int(args.seed),
        "max_seq_len": int(args.max_seq_len),
        "fit_split": args.fit_split,
        "eval_split": args.eval_split,
        "fit_limit": fit_limit,
        "eval_limit": eval_limit,
        "method": "logit_feature_response_curve",
        "feature_names": FEATURE_NAMES,
        "candidate_layer_numbers": candidate_layer_numbers,
        "layer_selection_rule": "top_k_by_ks_statistic",
        "top_k_layers_per_feature": int(args.top_k_layers_per_feature),
        "selected_layers_by_feature": {
            feature_name: selected_layers_df.loc[selected_layers_df["feature_name"].eq(feature_name), "layer_number"].astype(int).tolist()
            for feature_name in FEATURE_NAMES
        },
        "step_fractions": step_fractions,
        "max_delta_over_hidden_caps": max_delta_over_hidden_caps,
        "good_region_log_ratio_threshold": float(args.good_threshold_log_ratio),
        "support_lower_quantile": float(SUPPORT_LOWER_QUANTILE),
        "support_upper_quantile": float(SUPPORT_UPPER_QUANTILE),
        "kde_bandwidth_multiplier": float(KDE_BANDWIDTH_MULTIPLIER),
        "log_ratio_smoothing_sigma_bandwidths": float(LOG_RATIO_SMOOTHING_SIGMA_BANDWIDTHS),
        "distribution_grid_points": int(GRID_POINTS),
        "extract_batch_size": EXTRACT_BATCH_SIZE,
        "readout_batch_size": READOUT_BATCH_SIZE,
        "intervention_batch_size": INTERVENTION_BATCH_SIZE,
        "last_layer_needs_final_norm": bool(last_layer_needs_final_norm),
    }
    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    print(f"[done] wrote {out_dir}")


if __name__ == "__main__":
    main()
