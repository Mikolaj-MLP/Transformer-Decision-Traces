"""Wczytywanie i podstawowe miary eksperymentu interwencyjnego."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.run_discovery import (
    FEATURE_LABELS,
    FEATURE_ORDER,
    RUN_SPECS,
    apply_model_order,
    default_data_dir,
    latest_control_runs_by_feature,
    read_run_config,
    split_tags,
)


def add_clean_predictions(clean: pd.DataFrame) -> pd.DataFrame:
    clean = clean.copy()
    columns = [f"clean_logit_{letter}" for letter in "ABCDE"]
    clean["clean_pred_idx"] = clean[columns].to_numpy().argmax(axis=1)
    clean["clean_is_correct"] = clean["clean_pred_idx"] == clean["true_choice_idx"]
    return clean


def annotate_outcomes(raw: pd.DataFrame) -> pd.DataFrame:
    """Dodaj poprawność, rescue/harm oraz zmiany score i marginesu."""
    raw = raw.copy()
    steered_columns = [f"steered_logit_{letter}" for letter in "ABCDE"]
    clean_columns = [f"clean_logit_{letter}" for letter in "ABCDE"]
    raw["steered_pred_idx"] = raw[steered_columns].to_numpy().argmax(axis=1)
    raw["steered_is_correct"] = raw["steered_pred_idx"] == raw["true_choice_idx"]
    raw["accuracy_delta"] = raw["steered_is_correct"].astype(int) - raw["clean_is_correct"].astype(int)
    raw["rescued"] = (~raw["clean_is_correct"]) & raw["steered_is_correct"]
    raw["harmed"] = raw["clean_is_correct"] & (~raw["steered_is_correct"])
    raw["score_gain"] = raw["steered_score_value_local"] - raw["current_score_value"]
    raw["feature_delta_local"] = raw["steered_feature_value_local"] - raw["current_feature_value"]

    truth = raw["true_choice_idx"].to_numpy(dtype=int)
    row_number = np.arange(len(raw))
    clean_logits = raw[clean_columns].to_numpy(dtype=float)
    steered_logits = raw[steered_columns].to_numpy(dtype=float)
    wrong = np.ones_like(clean_logits, dtype=bool)
    wrong[row_number, truth] = False
    raw["correct_choice_logit_change"] = (
        steered_logits[row_number, truth] - clean_logits[row_number, truth]
    )
    raw["correct_vs_best_wrong_gap_change"] = (
        steered_logits[row_number, truth]
        - np.where(wrong, steered_logits, -np.inf).max(axis=1)
        - clean_logits[row_number, truth]
        + np.where(wrong, clean_logits, -np.inf).max(axis=1)
    )
    raw["feature"] = raw["feature_name"].map(FEATURE_LABELS).fillna(raw["feature_name"])
    return raw


def load_intervention_run(label: str, feature_dirs: dict[str, Path]) -> dict[str, object]:
    """Scal finalne runy poszczególnych cech jednego modelu."""
    raw_frames, selected_frames, separation_frames, grid_frames = [], [], [], []
    clean_frames = []
    configs = []
    max_layer = 0
    for run_dir in dict.fromkeys(feature_dirs.values()):
        config = read_run_config(run_dir)
        configs.append(config)
        _, eval_tag = split_tags(config)
        local_features = [name for name, path in feature_dirs.items() if path == run_dir]
        clean = add_clean_predictions(
            pd.read_parquet(run_dir / f"{eval_tag}_clean_final_outputs.parquet")
        )
        raw = pd.read_parquet(run_dir / f"{eval_tag}_score_control_policy_outputs_raw.parquet")
        raw = raw[raw["feature_name"].isin(local_features)].merge(
            clean[["example_id", "true_choice_idx", "clean_is_correct"]],
            on="example_id",
            how="left",
        )
        raw_frames.append(annotate_outcomes(raw))
        clean_frames.append(clean)
        for filename, target in (
            ("selected_layers_by_feature.parquet", selected_frames),
            ("feature_layer_separation_summary.parquet", separation_frames),
            ("fit_distribution_grid.parquet", grid_frames),
        ):
            frame = pd.read_parquet(run_dir / filename)
            target.append(frame[frame["feature_name"].isin(local_features)].copy())
        max_layer = max(max_layer, int(separation_frames[-1]["layer_number"].max()))

    consistent_keys = [
        "fit_split",
        "eval_split",
        "fit_n_examples",
        "eval_n_examples",
        "top_k_layers_per_feature",
        "max_delta_over_hidden",
        "backtrack_scales",
    ]
    for key in consistent_keys:
        values = {json.dumps(config.get(key), sort_keys=True) for config in configs}
        if len(values) > 1:
            raise ValueError(f"Niespójny parametr {key!r} między runami modelu {label}")

    raw = pd.concat(raw_frames, ignore_index=True)
    selected = pd.concat(selected_frames, ignore_index=True)
    separation = pd.concat(separation_frames, ignore_index=True)
    grid = pd.concat(grid_frames, ignore_index=True)
    for frame in (raw, selected, separation, grid):
        frame["model"] = label
        frame["layer_pct"] = 100.0 * frame["layer_number"] / max_layer
        frame["feature"] = frame["feature_name"].map(FEATURE_LABELS).fillna(frame["feature_name"])
    return {
        "label": label,
        "config": configs[-1],
        "raw": raw,
        "accepted": raw[raw["accepted_intervention"]].copy(),
        "clean": pd.concat(clean_frames).drop_duplicates("example_id"),
        "selected": selected,
        "separation": separation,
        "grid": grid,
        "max_layer": max_layer,
    }


def load_all_intervention_runs(
    data_dir: Path | None = None,
    features: list[str] | None = None,
) -> list[dict[str, object]]:
    data_dir = data_dir or default_data_dir()
    features = features or FEATURE_ORDER
    return [
        load_intervention_run(label, latest_control_runs_by_feature(data_dir, fragment, features))
        for label, fragment in RUN_SPECS
    ]


def outcome_metrics(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Podstawowe miary raportowane w pracy."""
    summary = frame.groupby(group_cols, observed=True).agg(
        n=("example_id", "size"),
        clean_accuracy=("clean_is_correct", "mean"),
        steered_accuracy=("steered_is_correct", "mean"),
        delta_accuracy=("accuracy_delta", "mean"),
        rescue_rate=("rescued", "mean"),
        harm_rate=("harmed", "mean"),
        mean_score_gain=("score_gain", "mean"),
        mean_margin_change=("correct_vs_best_wrong_gap_change", "mean"),
        mean_relative_norm=("delta_over_token_hidden_l2", "mean"),
    ).reset_index()
    return apply_model_order(summary) if "model" in group_cols else summary


def paired_branch_metrics(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Sparowane różnice ascent względem descent i random dla tych samych przykładów."""
    index = list(dict.fromkeys(group_cols + ["example_id", "feature_name", "layer_number"]))
    wide = frame.pivot_table(
        index=index,
        columns="intervention_type",
        values=["steered_is_correct", "rescued", "harmed"],
        aggfunc="first",
    )
    wide.columns = [f"{metric}__{branch}" for metric, branch in wide.columns]
    wide = wide.reset_index().dropna()
    groups = wide.groupby(group_cols, observed=True) if group_cols else [((), wide)]
    rows = []
    for key, group in groups:
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_cols, key))
        row.update(
            {
                "matched_triplets": len(group),
                "ascent_minus_random_accuracy": (
                    group["steered_is_correct__ascent"].astype(int)
                    - group["steered_is_correct__random_same_norm"].astype(int)
                ).mean(),
                "ascent_minus_descent_accuracy": (
                    group["steered_is_correct__ascent"].astype(int)
                    - group["steered_is_correct__descent"].astype(int)
                ).mean(),
                "ascent_vs_random_harm_reduction": (
                    group["harmed__random_same_norm"].astype(int)
                    - group["harmed__ascent"].astype(int)
                ).mean(),
                "ascent_vs_descent_harm_reduction": (
                    group["harmed__descent"].astype(int)
                    - group["harmed__ascent"].astype(int)
                ).mean(),
            }
        )
        rows.append(row)
    output = pd.DataFrame(rows)
    return apply_model_order(output) if "model" in group_cols else output
