"""Wczytywanie wyników diagnostycznych i proste pochodne tabel."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.run_discovery import (
    FEATURE_LABELS,
    RUN_SPECS,
    default_data_dir,
    latest_diagnostic_run,
    read_run_config,
)


def logits_to_probabilities(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def add_clean_predictions(clean: pd.DataFrame) -> pd.DataFrame:
    clean = clean.copy()
    columns = [f"clean_logit_{letter}" for letter in "ABCDE"]
    logits = clean[columns].to_numpy(dtype=float)
    clean["clean_pred_idx"] = logits.argmax(axis=1)
    clean["clean_is_correct"] = clean["clean_pred_idx"] == clean["true_choice_idx"]
    return clean


def feature_values_from_readouts(
    readouts: pd.DataFrame,
    clean: pd.DataFrame,
) -> pd.DataFrame:
    """Odtwórz trzy badane cechy z zapisanych logitów warstwowych."""
    base = readouts.merge(
        clean[["example_id", "clean_is_correct"]],
        on="example_id",
        how="left",
        validate="many_to_one",
    )
    logits = base[[f"logit_{letter}" for letter in "ABCDE"]].to_numpy(dtype=float)
    probabilities = logits_to_probabilities(logits)
    log_probabilities = np.log(np.clip(probabilities, 1e-12, None))
    entropy = -(probabilities * log_probabilities).sum(axis=1)
    surprisal = -log_probabilities
    sorted_logits = np.sort(logits, axis=1)[:, ::-1]
    values = {
        "answer_choice_top1_top2_logit_gap": sorted_logits[:, 0] - sorted_logits[:, 1],
        "answer_choice_varentropy": (
            probabilities * (surprisal - entropy[:, None]) ** 2
        ).sum(axis=1),
        "answer_choice_entropy_normalized": entropy / math.log(5),
    }
    frames = []
    for feature_name, feature_values in values.items():
        frame = base[["example_id", "split", "layer_number", "clean_is_correct"]].copy()
        frame["feature_name"] = feature_name
        frame["feature_value"] = feature_values
        frame["feature"] = FEATURE_LABELS[feature_name]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def layerwise_predictions(
    readouts: pd.DataFrame,
    examples: pd.DataFrame,
    clean: pd.DataFrame,
) -> pd.DataFrame:
    """Poprawność prognozy warstwowej oraz zgodność z decyzją końcową."""
    output = readouts.merge(
        examples[["example_id", "correct_idx"]],
        on="example_id",
        how="left",
        validate="many_to_one",
    ).merge(
        clean[["example_id", "clean_pred_idx", "clean_is_correct"]],
        on="example_id",
        how="left",
        validate="many_to_one",
    )
    logits = output[[f"logit_{letter}" for letter in "ABCDE"]].to_numpy(dtype=float)
    output["layer_pred_idx"] = logits.argmax(axis=1)
    output["layer_is_correct"] = output["layer_pred_idx"] == output["correct_idx"]
    output["matches_final_prediction"] = output["layer_pred_idx"] == output["clean_pred_idx"]
    return output


def rank_layers(separation: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for _, frame in separation.groupby("feature_name", observed=True):
        frame = frame.sort_values(["ks_statistic", "layer_number"], ascending=[False, True]).copy()
        frame["selection_rank"] = np.arange(1, len(frame) + 1)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def load_diagnostic_run(label: str, run_dir: Path) -> dict[str, object]:
    config = read_run_config(run_dir)
    fit_split = config.get("fit_split", "train")
    eval_split = config.get("eval_split", "validation")
    fit_examples = pd.read_parquet(run_dir / f"{fit_split}_examples.parquet")
    eval_examples = pd.read_parquet(run_dir / f"{eval_split}_examples.parquet")
    fit_clean = add_clean_predictions(
        pd.read_parquet(run_dir / f"{fit_split}_clean_final_outputs.parquet")
    )
    eval_clean = add_clean_predictions(
        pd.read_parquet(run_dir / f"{eval_split}_clean_final_outputs.parquet")
    )
    fit_readouts = pd.read_parquet(run_dir / f"{fit_split}_layerwise_choice_readouts.parquet")
    eval_readouts = pd.read_parquet(run_dir / f"{eval_split}_layerwise_choice_readouts.parquet")
    separation = pd.read_parquet(run_dir / "feature_layer_separation_summary.parquet")
    grid = pd.read_parquet(run_dir / "fit_distribution_grid.parquet")
    selected = rank_layers(separation)
    max_layer = int(separation["layer_number"].max())

    for frame in (separation, selected, grid):
        frame["model"] = label
        frame["layer_pct"] = 100.0 * frame["layer_number"] / max_layer
        frame["feature"] = frame["feature_name"].map(FEATURE_LABELS).fillna(frame["feature_name"])

    return {
        "label": label,
        "config": config,
        "fit_split": fit_split,
        "eval_split": eval_split,
        "max_layer": max_layer,
        "fit_clean": fit_clean,
        "eval_clean": eval_clean,
        "fit_values": feature_values_from_readouts(fit_readouts, fit_clean),
        "eval_values": feature_values_from_readouts(eval_readouts, eval_clean),
        "eval_layerwise": layerwise_predictions(eval_readouts, eval_examples, eval_clean),
        "separation": separation,
        "selected": selected,
        "grid": grid,
    }


def load_all_diagnostic_runs(data_dir: Path | None = None) -> list[dict[str, object]]:
    data_dir = data_dir or default_data_dir()
    return [
        load_diagnostic_run(label, latest_diagnostic_run(data_dir, fragment))
        for label, fragment in RUN_SPECS
    ]


def attach_score_values(values: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    """Przypisz obserwacjom najbliższą wartość score na zapisanej siatce."""
    frames = []
    for (feature_name, layer_number), frame in values.groupby(
        ["feature_name", "layer_number"],
        observed=True,
    ):
        local_grid = grid.loc[
            grid["feature_name"].eq(feature_name)
            & grid["layer_number"].eq(layer_number),
            ["grid_x", "log_density_ratio", "region_label", "is_supported"],
        ].sort_values("grid_x")
        if local_grid.empty:
            continue
        frames.append(
            pd.merge_asof(
                frame.sort_values("feature_value"),
                local_grid,
                left_on="feature_value",
                right_on="grid_x",
                direction="nearest",
            )
        )
    return pd.concat(frames, ignore_index=True)
