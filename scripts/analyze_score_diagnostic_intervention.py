from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import rankdata, spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_score_experiment_results import (  # noqa: E402
    FEATURE_LABELS,
    load_final_data,
)


OUTPUT_DIR = REPO_ROOT / "docs" / "score_diagnostic_intervention"
FIGURE_DIR = REPO_ROOT / "docs" / "figures"

MODEL_DEPTHS = {
    "Qwen 0.5B": 24,
    "Qwen 3B": 36,
    "Qwen 7B": 28,
    "Llama 1B": 16,
    "Llama 3B": 28,
}

OUTCOMES = {
    "mean_score_gain": "Średnia zmiana score",
    "mean_gap_change": "Średnia zmiana marginesu logitowego",
    "delta_accuracy": "Zmiana accuracy",
    "balanced_accuracy_delta": "Zrównoważony efekt accuracy",
}

FEATURE_MARKERS = {
    "answer_choice_entropy_normalized": "o",
    "answer_choice_top1_top2_logit_gap": "s",
    "answer_choice_varentropy": "^",
}


def build_cells(
    raw: pd.DataFrame,
    top_k: int,
    source_split: str,
) -> pd.DataFrame:
    selected = raw[
        (raw["intervention_type"] == "ascent")
        & (raw["selection_rank"] <= top_k)
    ].copy()
    if source_split != "all":
        selected = selected[selected["source_split"] == source_split].copy()

    cells = (
        selected.groupby(
            ["model", "feature_name", "layer_number"], observed=True
        )
        .agg(
            rows=("example_id", "size"),
            ks_statistic=("ks_statistic", "first"),
            selection_rank=("selection_rank", "first"),
            mean_score_gain=("score_gain", "mean"),
            mean_gap_change=("gap_change", "mean"),
            delta_accuracy=("accuracy_delta", "mean"),
            clean_accuracy=("clean_is_correct", "mean"),
            n_correct=("clean_is_correct", "sum"),
            rescues=("rescued", "sum"),
            harms=("harmed", "sum"),
            rescue_rate=("rescued", "mean"),
            harm_rate=("harmed", "mean"),
        )
        .reset_index()
    )
    cells["feature"] = cells["feature_name"].map(FEATURE_LABELS)
    cells["layer_pct"] = cells.apply(
        lambda row: row["layer_number"] / MODEL_DEPTHS[row["model"]], axis=1
    )
    cells["cluster_id"] = (
        cells["model"].astype(str) + " | " + cells["feature_name"].astype(str)
    )
    cells["n_wrong"] = cells["rows"] - cells["n_correct"]
    cells["rescue_rate_given_wrong"] = cells["rescues"] / cells["n_wrong"]
    cells["harm_rate_given_correct"] = cells["harms"] / cells["n_correct"]
    cells["balanced_accuracy_delta"] = 0.5 * (
        cells["rescue_rate_given_wrong"] - cells["harm_rate_given_correct"]
    )
    return cells


def centered_group_ranks(
    frame: pd.DataFrame,
    column: str,
    cluster_column: str,
) -> np.ndarray:
    values = np.empty(len(frame), dtype=float)
    for _, indices in frame.groupby(cluster_column, observed=True).groups.items():
        positions = frame.index.get_indexer(indices)
        ranked = rankdata(frame.loc[indices, column].to_numpy(dtype=float))
        values[positions] = ranked - ranked.mean()
    return values


def association(
    frame: pd.DataFrame,
    outcome: str,
    method: str,
    cluster_column: str = "cluster_id",
) -> float:
    if method == "raw_spearman":
        return float(spearmanr(frame["ks_statistic"], frame[outcome]).statistic)

    working = frame.reset_index(drop=True).copy()
    x_rank = centered_group_ranks(working, "ks_statistic", cluster_column)
    y_rank = centered_group_ranks(working, outcome, cluster_column)
    if method == "within_rank":
        return float(np.corrcoef(x_rank, y_rank)[0, 1])
    if method != "within_rank_depth_adjusted":
        raise ValueError(f"Nieznana metoda: {method}")

    depth_rank = centered_group_ranks(working, "layer_pct", cluster_column)
    design = np.column_stack([np.ones(len(working)), depth_rank])
    residual_x = x_rank - design @ np.linalg.lstsq(design, x_rank, rcond=None)[0]
    residual_y = y_rank - design @ np.linalg.lstsq(design, y_rank, rcond=None)[0]
    return float(np.corrcoef(residual_x, residual_y)[0, 1])


def cluster_bootstrap(
    cells: pd.DataFrame,
    iterations: int = 4000,
    seed: int = 20260715,
) -> pd.DataFrame:
    methods = [
        "raw_spearman",
        "within_rank",
        "within_rank_depth_adjusted",
    ]
    working = cells.reset_index(drop=True).copy()
    cluster_names = working["cluster_id"].drop_duplicates().tolist()
    cluster_codes = pd.Categorical(
        working["cluster_id"], categories=cluster_names
    ).codes
    cluster_indices = [
        np.flatnonzero(cluster_codes == code) for code in range(len(cluster_names))
    ]
    x_raw = working["ks_statistic"].to_numpy(dtype=float)
    depth_rank = centered_group_ranks(working, "layer_pct", "cluster_id")
    x_rank = centered_group_ranks(working, "ks_statistic", "cluster_id")
    y_raw = {
        outcome: working[outcome].to_numpy(dtype=float) for outcome in OUTCOMES
    }
    y_rank = {
        outcome: centered_group_ranks(working, outcome, "cluster_id")
        for outcome in OUTCOMES
    }
    rng = np.random.default_rng(seed)
    estimates = {
        (outcome, method): [] for outcome in OUTCOMES for method in methods
    }

    for _ in range(iterations):
        sampled_codes = rng.integers(
            0, len(cluster_names), size=len(cluster_names)
        )
        expanded_indices = np.concatenate(
            [cluster_indices[code] for code in sampled_codes]
        )
        expanded_x_rank = x_rank[expanded_indices]
        expanded_depth_rank = depth_rank[expanded_indices]
        for outcome in OUTCOMES:
            expanded_y_rank = y_rank[outcome][expanded_indices]
            raw_value = spearmanr(
                x_raw[expanded_indices], y_raw[outcome][expanded_indices]
            ).statistic
            within_value = np.corrcoef(
                expanded_x_rank, expanded_y_rank
            )[0, 1]
            design = np.column_stack(
                [np.ones(len(expanded_indices)), expanded_depth_rank]
            )
            residual_x = expanded_x_rank - design @ np.linalg.lstsq(
                design, expanded_x_rank, rcond=None
            )[0]
            residual_y = expanded_y_rank - design @ np.linalg.lstsq(
                design, expanded_y_rank, rcond=None
            )[0]
            adjusted_value = np.corrcoef(residual_x, residual_y)[0, 1]
            estimates[(outcome, "raw_spearman")].append(float(raw_value))
            estimates[(outcome, "within_rank")].append(float(within_value))
            estimates[(outcome, "within_rank_depth_adjusted")].append(
                float(adjusted_value)
            )

    rows = []
    for outcome in OUTCOMES:
        for method in methods:
            boot = np.asarray(estimates[(outcome, method)], dtype=float)
            rows.append(
                {
                    "outcome": outcome,
                    "outcome_label": OUTCOMES[outcome],
                    "association": method,
                    "estimate": association(cells, outcome, method),
                    "cluster_bootstrap_ci_low": float(np.nanpercentile(boot, 2.5)),
                    "cluster_bootstrap_ci_high": float(np.nanpercentile(boot, 97.5)),
                    "bootstrap_iterations": iterations,
                    "clusters": cells["cluster_id"].nunique(),
                    "cells": len(cells),
                }
            )
    return pd.DataFrame(rows)


def split_robustness(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for top_k in [4, 8]:
        for source_split in ["validation", "all"]:
            cells = build_cells(raw, top_k=top_k, source_split=source_split)
            for outcome in OUTCOMES:
                raw_rho, raw_p = spearmanr(cells["ks_statistic"], cells[outcome])
                rows.append(
                    {
                        "top_k": top_k,
                        "source_split": source_split,
                        "outcome": outcome,
                        "cells": len(cells),
                        "raw_spearman_rho": float(raw_rho),
                        "raw_spearman_p_naive": float(raw_p),
                        "within_rank": association(cells, outcome, "within_rank"),
                        "within_rank_depth_adjusted": association(
                            cells, outcome, "within_rank_depth_adjusted"
                        ),
                    }
                )
    return pd.DataFrame(rows)


def stage_summary(cells: pd.DataFrame, top_k: int) -> pd.DataFrame:
    positive_gap = cells["mean_gap_change"] > 0
    positive_accuracy = cells["delta_accuracy"] > 0
    zero_accuracy = cells["delta_accuracy"] == 0
    positive_balanced = cells["balanced_accuracy_delta"] > 0
    zero_balanced = cells["balanced_accuracy_delta"] == 0
    total_rows = int(cells["rows"].sum())
    total_correct = int(cells["n_correct"].sum())
    total_wrong = int(cells["n_wrong"].sum())
    total_rescues = int(cells["rescues"].sum())
    total_harms = int(cells["harms"].sum())
    rescue_given_wrong = total_rescues / total_wrong
    harm_given_correct = total_harms / total_correct
    return pd.DataFrame(
        [
            {
                "top_k": top_k,
                "cells": len(cells),
                "score_gain_positive_cells": int((cells["mean_score_gain"] > 0).sum()),
                "margin_gain_positive_cells": int(positive_gap.sum()),
                "accuracy_gain_positive_cells": int(positive_accuracy.sum()),
                "accuracy_unchanged_cells": int(zero_accuracy.sum()),
                "accuracy_loss_cells": int((cells["delta_accuracy"] < 0).sum()),
                "balanced_gain_positive_cells": int(positive_balanced.sum()),
                "balanced_gain_unchanged_cells": int(zero_balanced.sum()),
                "balanced_gain_negative_cells": int(
                    (cells["balanced_accuracy_delta"] < 0).sum()
                ),
                "positive_margin_without_accuracy_gain": int(
                    (positive_gap & ~positive_accuracy).sum()
                ),
                "macro_mean_score_gain": float(cells["mean_score_gain"].mean()),
                "macro_mean_gap_change": float(cells["mean_gap_change"].mean()),
                "macro_mean_delta_accuracy": float(cells["delta_accuracy"].mean()),
                "macro_mean_balanced_accuracy_delta": float(
                    cells["balanced_accuracy_delta"].mean()
                ),
                "micro_rows": total_rows,
                "micro_clean_accuracy": total_correct / total_rows,
                "micro_rescues": total_rescues,
                "micro_harms": total_harms,
                "micro_rescue_rate_given_wrong": rescue_given_wrong,
                "micro_harm_rate_given_correct": harm_given_correct,
                "micro_observed_delta_accuracy": (
                    total_rescues - total_harms
                ) / total_rows,
                "micro_balanced_accuracy_delta": 0.5
                * (rescue_given_wrong - harm_given_correct),
                "required_rescue_to_harm_rate_ratio": total_correct / total_wrong,
                "observed_rescue_to_harm_rate_ratio": (
                    rescue_given_wrong / harm_given_correct
                ),
            }
        ]
    )


def model_conditional_summary(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for top_k in [4, 8]:
        selected = raw[
            (raw["intervention_type"] == "ascent")
            & (raw["selection_rank"] <= top_k)
            & (raw["source_split"] == "validation")
        ]
        for model, group in selected.groupby("model", observed=True):
            total_rows = len(group)
            correct = int(group["clean_is_correct"].sum())
            wrong = total_rows - correct
            rescues = int(group["rescued"].sum())
            harms = int(group["harmed"].sum())
            clean_accuracy = correct / total_rows
            rescue_given_wrong = rescues / wrong
            harm_given_correct = harms / correct
            rows.append(
                {
                    "top_k": top_k,
                    "model": model,
                    "rows": total_rows,
                    "clean_accuracy": clean_accuracy,
                    "correct_rows": correct,
                    "wrong_rows": wrong,
                    "rescues": rescues,
                    "harms": harms,
                    "rescue_rate_given_wrong": rescue_given_wrong,
                    "harm_rate_given_correct": harm_given_correct,
                    "observed_delta_accuracy": (rescues - harms) / total_rows,
                    "balanced_accuracy_delta": 0.5
                    * (rescue_given_wrong - harm_given_correct),
                    "required_rescue_to_harm_rate_ratio": (
                        clean_accuracy / (1.0 - clean_accuracy)
                    ),
                    "observed_rescue_to_harm_rate_ratio": (
                        rescue_given_wrong / harm_given_correct
                    ),
                }
            )
    return pd.DataFrame(rows)


def make_model_figure(model_summary: pd.DataFrame) -> Path:
    primary = model_summary[model_summary["top_k"] == 4].copy()
    primary = primary.sort_values("clean_accuracy").reset_index(drop=True)
    y = np.arange(len(primary))
    natural = 100 * primary["observed_delta_accuracy"].to_numpy()
    balanced = 100 * primary["balanced_accuracy_delta"].to_numpy()

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
        }
    )
    figure, axis = plt.subplots(figsize=(8.8, 4.8), constrained_layout=True)
    for index in range(len(primary)):
        axis.plot(
            [natural[index], balanced[index]],
            [y[index], y[index]],
            color="#CBD5E1",
            linewidth=3.0,
            solid_capstyle="round",
            zorder=1,
        )
    axis.scatter(
        natural,
        y,
        color="#D97706",
        s=72,
        label="Efekt na naturalnym rozkładzie",
        zorder=3,
    )
    axis.scatter(
        balanced,
        y,
        color="#2563EB",
        marker="D",
        s=62,
        label="Efekt po zrównoważeniu",
        zorder=3,
    )
    for index, (natural_value, balanced_value) in enumerate(zip(natural, balanced)):
        axis.annotate(
            f"{natural_value:+.2f}".replace(".", ","),
            (natural_value, y[index]),
            xytext=(-7, 9),
            textcoords="offset points",
            ha="right",
            color="#92400E",
            fontsize=8.5,
        )
        axis.annotate(
            f"{balanced_value:+.2f}".replace(".", ","),
            (balanced_value, y[index]),
            xytext=(7, 9),
            textcoords="offset points",
            ha="left",
            color="#1D4ED8",
            fontsize=8.5,
        )

    labels = [
        f"{row.model}  (bazowa accuracy: {100 * row.clean_accuracy:.1f}%)".replace(
            ".", ","
        )
        for row in primary.itertuples(index=False)
    ]
    axis.set_yticks(y, labels)
    axis.axvline(0, color="#64748B", linewidth=1.0, linestyle="--")
    axis.grid(axis="x", color="#E5E7EB", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.set_xlabel("Zmiana accuracy (p.p.)")
    axis.set_title(
        "Naturalny i zrównoważony efekt ascent według modelu (TOP_K = 4)",
        loc="left",
        fontweight="bold",
    )
    axis.legend(loc="lower right", frameon=True, fontsize=9)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    path = FIGURE_DIR / "score_intervention_by_model_base_accuracy.png"
    figure.savefig(path, dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def make_figure(cells: pd.DataFrame) -> Path:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.7), constrained_layout=True)
    color_min = 100 * cells["layer_pct"].min()
    color_max = 100 * cells["layer_pct"].max()
    last_scatter = None

    panels = [
        (
            "mean_gap_change",
            "Średnia zmiana marginesu logitowego",
            "A. Reakcja marginesu decyzyjnego",
            1.0,
        ),
        (
            "balanced_accuracy_delta",
            "Zrównoważony efekt accuracy (p.p.)",
            "B. Efekt po zdyskontowaniu bazowej accuracy",
            100.0,
        ),
    ]
    for axis, (outcome, y_label, title, scale) in zip(axes, panels):
        for feature_name, marker in FEATURE_MARKERS.items():
            subset = cells[cells["feature_name"] == feature_name]
            last_scatter = axis.scatter(
                subset["ks_statistic"],
                scale * subset[outcome],
                c=100 * subset["layer_pct"],
                cmap="viridis",
                vmin=color_min,
                vmax=color_max,
                marker=marker,
                s=52,
                alpha=0.84,
                edgecolor="white",
                linewidth=0.55,
            )
        rho = spearmanr(cells["ks_statistic"], cells[outcome]).statistic
        axis.axhline(0, color="#6B7280", linewidth=0.9, linestyle="--", zorder=0)
        axis.grid(True, color="#E5E7EB", linewidth=0.7, alpha=0.85)
        axis.set_axisbelow(True)
        axis.set_xlabel("Siła diagnostyczna warstwy (statystyka KS)")
        axis.set_ylabel(y_label)
        axis.set_title(title, loc="left", fontweight="bold")
        axis.text(
            0.03,
            0.95,
            rf"$\rho_s$ = {rho:.3f}".replace(".", ","),
            transform=axis.transAxes,
            ha="left",
            va="top",
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": "#D1D5DB",
                "alpha": 0.9,
            },
        )

    most_harmful = cells.loc[cells["balanced_accuracy_delta"].idxmin()]
    axes[1].annotate(
        f"{most_harmful['model']}, varentropy, L{int(most_harmful['layer_number'])}",
        xy=(
            most_harmful["ks_statistic"],
            100 * most_harmful["balanced_accuracy_delta"],
        ),
        xytext=(12, 15),
        textcoords="offset points",
        fontsize=8,
        arrowprops={"arrowstyle": "->", "color": "#4B5563", "lw": 0.8},
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            color="none",
            markerfacecolor="#64748B",
            markeredgecolor="white",
            markersize=7,
            label=FEATURE_LABELS[feature_name],
        )
        for feature_name, marker in FEATURE_MARKERS.items()
    ]
    axes[0].legend(
        handles=legend_handles,
        title="Cecha",
        loc="lower right",
        frameon=True,
        fontsize=8,
        title_fontsize=8.5,
    )
    colorbar = figure.colorbar(last_scatter, ax=axes, shrink=0.90, pad=0.02)
    colorbar.set_label("Względna głębokość warstwy (%)")

    path = FIGURE_DIR / "score_diagnostic_vs_intervention_top4_validation.png"
    figure.savefig(path, dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    raw, _, _ = load_final_data()

    cells_by_k = {
        top_k: build_cells(raw, top_k=top_k, source_split="validation")
        for top_k in [4, 8]
    }
    correlation_frames = []
    stage_frames = []
    for top_k, cells in cells_by_k.items():
        cells.to_csv(OUTPUT_DIR / f"cells_top{top_k}_validation.csv", index=False)
        correlations = cluster_bootstrap(cells)
        correlations.insert(0, "top_k", top_k)
        correlation_frames.append(correlations)
        stage_frames.append(stage_summary(cells, top_k))

    correlation_table = pd.concat(correlation_frames, ignore_index=True)
    correlation_table.to_csv(OUTPUT_DIR / "correlation_analysis.csv", index=False)
    stage_table = pd.concat(stage_frames, ignore_index=True)
    stage_table.to_csv(OUTPUT_DIR / "effect_stage_summary.csv", index=False)
    robustness = split_robustness(raw)
    robustness.to_csv(OUTPUT_DIR / "split_robustness.csv", index=False)
    figure_path = make_figure(cells_by_k[4])
    model_summary = model_conditional_summary(raw)
    model_summary.to_csv(OUTPUT_DIR / "model_conditional_summary.csv", index=False)
    model_figure_path = make_model_figure(model_summary)

    metadata = {
        "primary_analysis": "TOP_K=4, validation, ascent, accepted interventions",
        "robustness_analysis": "TOP_K=8 and full evaluation set",
        "unit_of_analysis": "model-feature-layer cell",
        "top4_validation_cells": len(cells_by_k[4]),
        "top8_validation_cells": len(cells_by_k[8]),
        "clusters": int(cells_by_k[4]["cluster_id"].nunique()),
        "figure": str(figure_path.relative_to(REPO_ROOT)),
        "model_figure": str(model_figure_path.relative_to(REPO_ROOT)),
        "interpretation_note": (
            "Surowa korelacja KS ze zmianą marginesu jest w dużej mierze związana "
            "z głębokością warstwy; po rangowym uwzględnieniu głębokości w obrębie "
            "trajektorii model-cecha zależność zanika."
        ),
    }
    with (OUTPUT_DIR / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    print(f"Zapisano dane: {OUTPUT_DIR}")
    print(f"Zapisano wykres: {figure_path}")
    print(stage_table.to_string(index=False))
    print(
        correlation_table[
            [
                "top_k",
                "outcome",
                "association",
                "estimate",
                "cluster_bootstrap_ci_low",
                "cluster_bootstrap_ci_high",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
