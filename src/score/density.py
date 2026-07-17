"""Estymacja rozkładów i definicja funkcji score.

Dla wartości cechy ``x`` estymowane są osobne gęstości dla odpowiedzi
poprawnych i błędnych. Funkcja używana w pracy ma postać

    s(x) = log p(x | correct) - log p(x | incorrect).

Obliczenia w tym module nie zależą od modelu Transformer ani od pipeline'u.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity

from src.score.constants import (
    KDE_BANDWIDTH_MULTIPLIER,
    KDE_JITTER_SCALE,
    LOG_RATIO_SMOOTHING_SIGMA_BANDWIDTHS,
    SUPPORT_LOWER_QUANTILE,
    SUPPORT_UPPER_QUANTILE,
)


def silverman_bandwidth(values: np.ndarray) -> float:
    """Reguła Silvermana z odpornym oszacowaniem skali."""
    values = np.asarray(values, dtype=float)
    n = values.shape[0]
    if n <= 1:
        return 0.1
    std = float(values.std(ddof=1))
    iqr = float(np.subtract(*np.percentile(values, [75, 25])))
    robust = iqr / 1.34 if iqr > 0 else std
    positive_scales = [scale for scale in (std, robust) if scale > 0]
    scale = min(positive_scales) if positive_scales else 1.0
    return float(max(0.9 * scale * (n ** (-1 / 5)), 1e-3))


def fit_kde(values: np.ndarray, bandwidth: float) -> KernelDensity:
    """Dopasuj jednowymiarowy estymator gęstości z jądrem Gaussa."""
    model = KernelDensity(kernel="gaussian", bandwidth=float(bandwidth))
    model.fit(values.reshape(-1, 1))
    return model


def gaussian_kernel_1d(sigma_grid: float) -> np.ndarray:
    if sigma_grid <= 1e-8:
        return np.array([1.0], dtype=np.float64)
    radius = max(1, int(math.ceil(3.0 * sigma_grid)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma_grid) ** 2)
    return kernel / np.sum(kernel)


def smooth_1d(values: np.ndarray, sigma_grid: float) -> np.ndarray:
    """Wygładź wartości na regularnej siatce bez zmiany długości tablicy."""
    kernel = gaussian_kernel_1d(float(sigma_grid))
    if kernel.shape[0] == 1:
        return values.copy()
    pad = kernel.shape[0] // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def build_distribution_models(
    *,
    fit_feature_df: pd.DataFrame,
    selected_layers_df: pd.DataFrame,
    log_ratio_threshold: float,
    grid_points: int,
) -> tuple[dict[tuple[str, int], dict[str, object]], pd.DataFrame]:
    """Dopasuj ``p_good``, ``p_bad`` i ``s(x)`` dla wybranych par cecha-warstwa."""
    models: dict[tuple[str, int], dict[str, object]] = {}
    rows: list[dict[str, object]] = []

    pairs = selected_layers_df[["feature_name", "layer_number"]].itertuples(index=False)
    for feature_name, layer_number in pairs:
        part = fit_feature_df.loc[
            fit_feature_df["feature_name"].eq(feature_name)
            & fit_feature_df["layer_number"].eq(layer_number)
        ]
        correct_values = part.loc[part["clean_is_correct"], "feature_value"].to_numpy(dtype=float)
        incorrect_values = part.loc[~part["clean_is_correct"], "feature_value"].to_numpy(dtype=float)
        pooled = np.concatenate([correct_values, incorrect_values])

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
            candidate_indices = supported_indices if supported_indices.size else np.arange(grid.shape[0])
            good_mask[int(candidate_indices[np.argmax(log_ratio[candidate_indices])])] = True

        region_label = np.full(grid.shape[0], "unsupported", dtype=object)
        region_label[supported_mask] = "neutral"
        region_label[bad_mask] = "bad"
        region_label[good_mask] = "good"

        key = (str(feature_name), int(layer_number))
        models[key] = {
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

        for idx, grid_x in enumerate(grid):
            rows.append(
                {
                    "feature_name": feature_name,
                    "layer_number": int(layer_number),
                    "grid_x": float(grid_x),
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


def add_score_derivative(
    region_models: dict[tuple[str, int], dict[str, object]],
    distribution_grid_df: pd.DataFrame,
) -> pd.DataFrame:
    """Dodaj numeryczną pochodną ``ds/dx`` do modeli i tabeli siatki."""
    output = distribution_grid_df.copy()
    output["score_derivative"] = np.nan
    for (feature_name, layer_number), model in region_models.items():
        grid = np.asarray(model["grid"], dtype=np.float64)
        score = np.asarray(model["log_ratio"], dtype=np.float64)
        score_derivative = np.gradient(score, grid)
        model["score_derivative"] = score_derivative
        mask = output["feature_name"].eq(feature_name) & output["layer_number"].eq(layer_number)
        output.loc[mask, "score_derivative"] = score_derivative.astype(np.float32)
    return output


def interpolate_score_state(
    feature_values: np.ndarray,
    *,
    region_model: dict[str, object],
) -> dict[str, np.ndarray]:
    """Odczytaj ``s(x)``, ``ds/dx`` i region dla obserwowanych wartości cechy."""
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

    interp_score = np.interp(values, grid, score, left=score[0], right=score[-1])
    interp_derivative = np.interp(
        values,
        grid,
        score_derivative,
        left=score_derivative[0],
        right=score_derivative[-1],
    )
    return {
        "supported": supported.astype(bool),
        "score_value": np.where(supported, interp_score, np.nan).astype(np.float32),
        "score_derivative": np.where(supported, interp_derivative, np.nan).astype(np.float32),
        "region_label": region_label_grid[nearest_idx].astype(object),
    }
