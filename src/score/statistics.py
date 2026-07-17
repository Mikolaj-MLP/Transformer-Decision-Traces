"""Statystyki diagnostyczne i wybór warstw."""

from __future__ import annotations

import numpy as np
import pandas as pd


def empirical_cdf(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Empiryczna dystrybuanta próby wyznaczona na zadanej siatce."""
    sorted_values = np.sort(values)
    return np.searchsorted(sorted_values, grid, side="right") / float(sorted_values.shape[0])


def compute_ks_statistic(correct_values: np.ndarray, incorrect_values: np.ndarray) -> float:
    """Dwustronna statystyka Kołmogorowa-Smirnowa między obiema grupami."""
    pooled_grid = np.sort(np.unique(np.concatenate([correct_values, incorrect_values], axis=0)))
    if pooled_grid.size == 0:
        return 0.0
    cdf_good = empirical_cdf(correct_values, pooled_grid)
    cdf_bad = empirical_cdf(incorrect_values, pooled_grid)
    return float(np.max(np.abs(cdf_good - cdf_bad)))


def build_separation_summary(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Podsumuj separację odpowiedzi poprawnych i błędnych dla cechy i warstwy."""
    rows: list[dict[str, object]] = []
    grouped_keys = sorted(
        feature_df[["feature_name", "layer_number"]].drop_duplicates().itertuples(index=False, name=None)
    )
    for feature_name, layer_number in grouped_keys:
        part = feature_df.loc[
            feature_df["feature_name"].eq(feature_name)
            & feature_df["layer_number"].eq(layer_number)
        ]
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
    """Wybierz dla każdej cechy warstwy o największej statystyce KS."""
    selected: list[pd.DataFrame] = []
    for feature_name in feature_names:
        part = separation_df.loc[separation_df["feature_name"].eq(feature_name)].copy()
        part = part.sort_values(["ks_statistic", "layer_number"], ascending=[False, True]).head(top_k)
        part["selection_rank"] = np.arange(1, len(part) + 1)
        selected.append(part)
    return pd.concat(selected, ignore_index=True)
