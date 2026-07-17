from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_score_experiment_results import (
    FEATURE_LABELS,
    INTERVENTION_LABELS,
    MODEL_SPECS,
    load_final_data,
)


OUTPUT_DIR = REPO_ROOT / "docs" / "layer_intervention_analysis"
FIGURE_DIR = REPO_ROOT / "docs" / "figures"

TOP_K = 8
BOOTSTRAP_ITERATIONS = 3000
BOOTSTRAP_SEED = 20260715

MODEL_ORDER = list(MODEL_SPECS)
FEATURE_ORDER = list(FEATURE_LABELS)
INTERVENTION_ORDER = ["ascent", "random_same_norm", "descent"]
INTERVENTION_COLORS = {
    "ascent": "#2563eb",
    "random_same_norm": "#6b7280",
    "descent": "#dc2626",
}


def prepare_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw, selected, clean = load_final_data()
    raw = raw[raw["selection_rank"] <= TOP_K].copy()
    selected = selected[selected["selection_rank"] <= TOP_K].copy()

    model_last_layer = selected.groupby("model", observed=True)["layer_number"].max()
    raw["last_layer"] = raw["model"].map(model_last_layer).astype(int)
    selected["last_layer"] = selected["model"].map(model_last_layer).astype(int)
    raw["layer_pct"] = 100.0 * raw["layer_number"] / raw["last_layer"]
    selected["layer_pct"] = 100.0 * selected["layer_number"] / selected["last_layer"]
    raw["is_final_layer"] = raw["layer_number"] == raw["last_layer"]
    selected["is_final_layer"] = selected["layer_number"] == selected["last_layer"]

    clean_columns = [f"clean_logit_{letter}" for letter in "ABCDE"]
    clean_prediction = raw[clean_columns].to_numpy(dtype=float).argmax(axis=1)
    raw["changed_answer"] = clean_prediction != raw["steered_pred_idx"].to_numpy()

    clean = clean.copy()
    logits = clean[clean_columns].to_numpy(dtype=float)
    ordered = np.sort(logits, axis=1)
    clean["clean_top1_top2_gap"] = ordered[:, -1] - ordered[:, -2]
    return raw, selected, clean


def selection_summaries(selected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_model = selected.groupby("model", observed=True).agg(
        selected_cells=("layer_number", "size"),
        min_depth_pct=("layer_pct", "min"),
        mean_depth_pct=("layer_pct", "mean"),
        median_depth_pct=("layer_pct", "median"),
        max_depth_pct=("layer_pct", "max"),
        final_layer_cells=("is_final_layer", "sum"),
    ).reset_index()
    by_model["model"] = pd.Categorical(by_model["model"], MODEL_ORDER, ordered=True)
    by_model = by_model.sort_values("model").reset_index(drop=True)

    by_rank = selected.groupby("selection_rank", observed=True).agg(
        selected_cells=("layer_number", "size"),
        mean_depth_pct=("layer_pct", "mean"),
        median_depth_pct=("layer_pct", "median"),
        mean_ks=("ks_statistic", "mean"),
    ).reset_index().sort_values("selection_rank")
    return by_model, by_rank


def layer_profile(raw: pd.DataFrame, source_split: str) -> pd.DataFrame:
    source = raw if source_split == "all" else raw[raw["source_split"] == source_split]
    rows: list[dict] = []
    group_columns = [
        "model",
        "intervention_type",
        "layer_number",
        "last_layer",
        "layer_pct",
    ]
    for keys, group in source.groupby(group_columns, observed=True):
        initially_correct = group["clean_is_correct"]
        initially_wrong = ~initially_correct
        rows.append(
            {
                **dict(zip(group_columns, keys)),
                "source_split": source_split,
                "rows": len(group),
                "delta_accuracy": group["accuracy_delta"].mean(),
                "harm_rate_given_correct": group.loc[initially_correct, "harmed"].mean(),
                "rescue_rate_given_wrong": group.loc[initially_wrong, "rescued"].mean(),
                "change_rate": group["changed_answer"].mean(),
                "mean_score_gain": group["score_gain"].mean(),
                "mean_gap_change": group["gap_change"].mean(),
            }
        )
    return pd.DataFrame(rows)


def depth_bin_profile(raw: pd.DataFrame, source_split: str) -> pd.DataFrame:
    source = raw if source_split == "all" else raw[raw["source_split"] == source_split]
    source = source.copy()
    source["depth_bin"] = pd.cut(
        source["layer_pct"],
        bins=[-0.1, 50.0, 80.0, 101.0],
        labels=["do 50%", "50–80%", "powyżej 80%"],
    )
    rows: list[dict] = []
    for keys, group in source.groupby(
        ["model", "intervention_type", "depth_bin"], observed=True
    ):
        initially_correct = group["clean_is_correct"]
        initially_wrong = ~initially_correct
        rows.append(
            {
                "model": keys[0],
                "intervention_type": keys[1],
                "depth_bin": keys[2],
                "source_split": source_split,
                "rows": len(group),
                "selected_layers": group["layer_number"].nunique(),
                "delta_accuracy": group["accuracy_delta"].mean(),
                "harm_rate_given_correct": group.loc[
                    initially_correct, "harmed"
                ].mean(),
                "rescue_rate_given_wrong": group.loc[
                    initially_wrong, "rescued"
                ].mean(),
                "change_rate": group["changed_answer"].mean(),
                "mean_score_gain": group["score_gain"].mean(),
                "mean_gap_change": group["gap_change"].mean(),
            }
        )
    return pd.DataFrame(rows)


def rank_profiles(
    raw: pd.DataFrame, source_split: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = raw if source_split == "all" else raw[raw["source_split"] == source_split]
    rows: list[dict] = []
    keys = ["model", "feature_name", "selection_rank", "intervention_type"]
    for group_keys, group in source.groupby(keys, observed=True):
        initially_correct = group["clean_is_correct"]
        initially_wrong = ~initially_correct
        rows.append(
            {
                **dict(zip(keys, group_keys)),
                "source_split": source_split,
                "rows": len(group),
                "layer_pct": group["layer_pct"].iloc[0],
                "delta_accuracy": group["accuracy_delta"].mean(),
                "harm_rate_given_correct": group.loc[
                    initially_correct, "harmed"
                ].mean(),
                "rescue_rate_given_wrong": group.loc[
                    initially_wrong, "rescued"
                ].mean(),
                "change_rate": group["changed_answer"].mean(),
                "mean_score_gain": group["score_gain"].mean(),
                "mean_gap_change": group["gap_change"].mean(),
            }
        )
    cells = pd.DataFrame(rows)
    macro = cells.groupby(
        ["source_split", "intervention_type", "selection_rank"], observed=True
    ).agg(
        model_feature_cells=("model", "size"),
        mean_depth_pct=("layer_pct", "mean"),
        delta_accuracy=("delta_accuracy", "mean"),
        harm_rate_given_correct=("harm_rate_given_correct", "mean"),
        rescue_rate_given_wrong=("rescue_rate_given_wrong", "mean"),
        change_rate=("change_rate", "mean"),
        mean_score_gain=("mean_score_gain", "mean"),
        mean_gap_change=("mean_gap_change", "mean"),
    ).reset_index()
    return cells, macro


PAIR_METRICS = [
    "accuracy_delta",
    "harmed",
    "rescued",
    "changed_answer",
    "score_gain",
    "gap_change",
]


def final_nonfinal_pairs(raw: pd.DataFrame, source_split: str) -> pd.DataFrame:
    source = raw if source_split == "all" else raw[raw["source_split"] == source_split]
    keys = ["model", "example_id", "feature_name", "intervention_type"]
    value_columns = PAIR_METRICS + ["clean_is_correct"]

    final = (
        source[source["is_final_layer"]]
        .groupby(keys, observed=True)[value_columns]
        .mean()
        .add_suffix("_final")
    )
    nonfinal = (
        source[~source["is_final_layer"]]
        .groupby(keys, observed=True)[value_columns]
        .mean()
        .add_suffix("_nonfinal")
    )
    pairs = final.join(nonfinal, how="inner").reset_index()
    pairs["source_split"] = source_split
    for metric in PAIR_METRICS:
        pairs[f"{metric}_diff"] = (
            pairs[f"{metric}_final"] - pairs[f"{metric}_nonfinal"]
        )
    return pairs


def metric_subset(pairs: pd.DataFrame, metric: str) -> pd.DataFrame:
    pairs = pairs.dropna(
        subset=[
            f"{metric}_final",
            f"{metric}_nonfinal",
            f"{metric}_diff",
        ]
    )
    if metric == "harmed":
        return pairs[pairs["clean_is_correct_final"] > 0.5]
    if metric == "rescued":
        return pairs[pairs["clean_is_correct_final"] < 0.5]
    return pairs


def summarize_final_nonfinal(pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_rows: list[dict] = []
    macro_rows: list[dict] = []
    for variant in INTERVENTION_ORDER:
        variant_pairs = pairs[pairs["intervention_type"] == variant]
        for metric in PAIR_METRICS:
            part = metric_subset(variant_pairs, metric)
            example_level = part.groupby(
                ["model", "example_id"], observed=True
            )[
                [f"{metric}_final", f"{metric}_nonfinal", f"{metric}_diff"]
            ].mean().reset_index()
            for model, group in example_level.groupby("model", observed=True):
                model_rows.append(
                    {
                        "source_split": pairs["source_split"].iloc[0],
                        "intervention_type": variant,
                        "metric": metric,
                        "model": model,
                        "clusters": group["example_id"].nunique(),
                        "final": group[f"{metric}_final"].mean(),
                        "nonfinal": group[f"{metric}_nonfinal"].mean(),
                        "difference": group[f"{metric}_diff"].mean(),
                    }
                )

            model_summary = pd.DataFrame(model_rows)
            current = model_summary[
                (model_summary["source_split"] == pairs["source_split"].iloc[0])
                & (model_summary["intervention_type"] == variant)
                & (model_summary["metric"] == metric)
            ]
            macro_rows.append(
                {
                    "source_split": pairs["source_split"].iloc[0],
                    "intervention_type": variant,
                    "metric": metric,
                    "models": current["model"].nunique(),
                    "clusters_model_example": example_level[
                        ["model", "example_id"]
                    ].drop_duplicates().shape[0],
                    "final_macro_models": current["final"].mean(),
                    "nonfinal_macro_models": current["nonfinal"].mean(),
                    "difference_macro_models": current["difference"].mean(),
                }
            )
    return pd.DataFrame(model_rows), pd.DataFrame(macro_rows)


def bootstrap_final_nonfinal(pairs: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows: list[dict] = []
    split = pairs["source_split"].iloc[0]
    for variant in INTERVENTION_ORDER:
        variant_pairs = pairs[pairs["intervention_type"] == variant]
        for metric in PAIR_METRICS:
            part = metric_subset(variant_pairs, metric)
            example_level = part.groupby(
                ["model", "example_id"], observed=True
            )[
                [f"{metric}_final", f"{metric}_nonfinal", f"{metric}_diff"]
            ].mean().reset_index()

            per_model_boot: list[np.ndarray] = []
            per_model_points: list[np.ndarray] = []
            for model in MODEL_ORDER:
                values = example_level[example_level["model"] == model][
                    [f"{metric}_final", f"{metric}_nonfinal", f"{metric}_diff"]
                ].to_numpy(dtype=float)
                if not len(values):
                    continue
                draw = rng.integers(
                    0,
                    len(values),
                    size=(BOOTSTRAP_ITERATIONS, len(values)),
                )
                per_model_boot.append(values[draw].mean(axis=1))
                per_model_points.append(values.mean(axis=0))

            boot = np.stack(per_model_boot).mean(axis=0)
            point = np.stack(per_model_points).mean(axis=0)
            rows.append(
                {
                    "source_split": split,
                    "intervention_type": variant,
                    "metric": metric,
                    "models": len(per_model_points),
                    "clusters_model_example": example_level[
                        ["model", "example_id"]
                    ].drop_duplicates().shape[0],
                    "final_macro_models": point[0],
                    "nonfinal_macro_models": point[1],
                    "difference_macro_models": point[2],
                    "difference_ci_low": np.quantile(boot[:, 2], 0.025),
                    "difference_ci_high": np.quantile(boot[:, 2], 0.975),
                    "iterations": BOOTSTRAP_ITERATIONS,
                }
            )
    return pd.DataFrame(rows)


def baseline_margin_vs_final_harm(
    raw: pd.DataFrame, clean: pd.DataFrame, source_split: str
) -> tuple[pd.DataFrame, dict[str, float]]:
    clean_source = clean if source_split == "all" else clean[clean["source_split"] == source_split]
    clean_correct = clean_source[clean_source["clean_is_correct"]]
    margin = clean_correct.groupby("model", observed=True).agg(
        clean_correct_examples=("example_id", "nunique"),
        mean_clean_gap=("clean_top1_top2_gap", "mean"),
        median_clean_gap=("clean_top1_top2_gap", "median"),
    )

    source = raw if source_split == "all" else raw[raw["source_split"] == source_split]
    final_descent = source[
        source["is_final_layer"]
        & source["clean_is_correct"]
        & (source["intervention_type"] == "descent")
    ]
    harm_by_example = final_descent.groupby(
        ["model", "example_id"], observed=True
    )["harmed"].mean().reset_index()
    harm = harm_by_example.groupby("model", observed=True).agg(
        accepted_correct_examples=("example_id", "nunique"),
        final_descent_harm_rate=("harmed", "mean"),
    )
    result = margin.join(harm).reset_index()
    result["model"] = pd.Categorical(result["model"], MODEL_ORDER, ordered=True)
    result = result.sort_values("model").reset_index(drop=True)

    rho_median, p_median = spearmanr(
        result["median_clean_gap"], result["final_descent_harm_rate"]
    )
    rho_mean, p_mean = spearmanr(
        result["mean_clean_gap"], result["final_descent_harm_rate"]
    )
    correlations = {
        "spearman_median_gap_vs_harm": float(rho_median),
        "naive_p_median_gap_vs_harm": float(p_median),
        "spearman_mean_gap_vs_harm": float(rho_mean),
        "naive_p_mean_gap_vs_harm": float(p_mean),
        "model_count": int(len(result)),
    }
    return result, correlations


def plot_selected_layers(selected: pd.DataFrame) -> Path:
    frame = selected.copy()
    frame["feature_label"] = frame["feature_name"].map(FEATURE_LABELS)
    frame["row_label"] = frame["model"].astype(str) + " — " + frame["feature_label"]
    ordered_labels = [
        f"{model} — {FEATURE_LABELS[feature]}"
        for model in MODEL_ORDER
        for feature in FEATURE_ORDER
    ]
    frame["row_label"] = pd.Categorical(
        frame["row_label"], categories=ordered_labels[::-1], ordered=True
    )
    frame = frame.sort_values(["row_label", "selection_rank"])

    figure, axis = plt.subplots(figsize=(11.2, 8.0))
    rank_palette = [
        "#1d4ed8",
        "#3b82f6",
        "#60a5fa",
        "#93c5fd",
        "#fbbf24",
        "#f59e0b",
        "#f97316",
        "#dc2626",
    ]
    rank_colors = {
        rank: rank_palette[rank - 1] for rank in range(1, TOP_K + 1)
    }
    y_lookup = {label: index for index, label in enumerate(ordered_labels[::-1])}
    for rank in range(1, TOP_K + 1):
        group = frame[frame["selection_rank"] == rank]
        axis.scatter(
            group["layer_pct"],
            group["row_label"].astype(str).map(y_lookup),
            s=62,
            color=rank_colors[rank],
            edgecolor="white",
            linewidth=0.7,
            label=f"ranga KS {rank}",
            zorder=3,
        )
    for boundary in [50, 80, 100]:
        axis.axvline(boundary, color="#9ca3af", linestyle="--", linewidth=0.9, zorder=1)
    axis.set_yticks(range(len(ordered_labels)))
    axis.set_yticklabels(ordered_labels[::-1], fontsize=9)
    axis.set_xlim(0, 101.5)
    axis.set_xlabel("Znormalizowana głębokość warstwy [%]")
    axis.set_title(
        "Położenie ośmiu najwyżej sklasyfikowanych warstw dla każdej pary model–cecha"
    )
    axis.grid(axis="x", alpha=0.2)
    axis.legend(loc="lower left", ncol=2, fontsize=9, frameon=True)
    figure.tight_layout()
    path = FIGURE_DIR / "layer_intervention_top8_selected_depth.png"
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_layer_harm(profile: pd.DataFrame) -> Path:
    data = profile[profile["source_split"] == "validation"].copy()
    figure, axes = plt.subplots(3, 2, figsize=(12.0, 11.0), sharex=True, sharey=True)
    axes = axes.ravel()
    for axis, model in zip(axes, MODEL_ORDER):
        model_data = data[data["model"] == model]
        for variant in INTERVENTION_ORDER:
            group = model_data[model_data["intervention_type"] == variant].sort_values(
                "layer_pct"
            )
            axis.plot(
                group["layer_pct"],
                100.0 * group["harm_rate_given_correct"],
                marker="o",
                linewidth=2.0,
                markersize=4.5,
                color=INTERVENTION_COLORS[variant],
                label=INTERVENTION_LABELS[variant],
            )
        axis.set_title(model)
        axis.set_xlim(0, 101)
        axis.set_ylim(-2, 100)
        axis.set_xticks([0, 20, 40, 60, 80, 100])
        axis.grid(alpha=0.22)
        axis.set_xlabel("Głębokość [%]")
        axis.set_ylabel("Zepsute odpowiedzi bazowo poprawne [%]")
    axes[-1].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[-1].legend(handles, labels, loc="center", frameon=True, title="Wariant")
    figure.suptitle(
        "Destrukcyjność interwencji w obrębie ośmiu wybranych warstw\n"
        "zbiór walidacyjny, agregacja po cechach",
        y=0.995,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    path = FIGURE_DIR / "layer_intervention_harm_by_depth_top8_validation.png"
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_margin_vs_harm(frame: pd.DataFrame) -> Path:
    figure, axis = plt.subplots(figsize=(8.8, 5.8))
    axis.scatter(
        frame["median_clean_gap"],
        100.0 * frame["final_descent_harm_rate"],
        s=95,
        color="#dc2626",
        edgecolor="black",
        linewidth=0.6,
        zorder=3,
    )
    offsets = {
        "Llama 1B": (8, 6),
        "Llama 3B": (8, -18),
        "Qwen 0.5B": (8, -18),
        "Qwen 3B": (-78, 9),
        "Qwen 7B": (8, 8),
    }
    for row in frame.itertuples(index=False):
        axis.annotate(
            str(row.model),
            (row.median_clean_gap, 100.0 * row.final_descent_harm_rate),
            xytext=offsets[str(row.model)],
            textcoords="offset points",
            fontsize=10,
        )
    axis.set_xlabel("Mediana bazowej luki top-1/top-2 dla poprawnych odpowiedzi")
    axis.set_ylabel("Odpowiedzi zepsute przez descent w warstwie końcowej [%]")
    axis.set_title("Bazowy margines decyzji a podatność warstwy końcowej")
    axis.set_xlim(left=0)
    axis.set_ylim(-3, 95)
    axis.grid(alpha=0.22)
    figure.tight_layout()
    path = FIGURE_DIR / "layer_intervention_margin_vs_final_descent_harm.png"
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    raw, selected, clean = prepare_data()
    selection_model, selection_rank = selection_summaries(selected)
    profiles = pd.concat(
        [layer_profile(raw, split) for split in ["validation", "all"]],
        ignore_index=True,
    )
    depth_bins = pd.concat(
        [depth_bin_profile(raw, split) for split in ["validation", "all"]],
        ignore_index=True,
    )
    depth_bin_macro = depth_bins.groupby(
        ["source_split", "intervention_type", "depth_bin"], observed=True
    ).agg(
        models=("model", "nunique"),
        delta_accuracy=("delta_accuracy", "mean"),
        harm_rate_given_correct=("harm_rate_given_correct", "mean"),
        rescue_rate_given_wrong=("rescue_rate_given_wrong", "mean"),
        change_rate=("change_rate", "mean"),
        mean_score_gain=("mean_score_gain", "mean"),
        mean_gap_change=("mean_gap_change", "mean"),
    ).reset_index()
    rank_cell_frames: list[pd.DataFrame] = []
    rank_macro_frames: list[pd.DataFrame] = []
    for split in ["validation", "all"]:
        rank_cells, rank_macro = rank_profiles(raw, split)
        rank_cell_frames.append(rank_cells)
        rank_macro_frames.append(rank_macro)
    rank_cells = pd.concat(rank_cell_frames, ignore_index=True)
    rank_macro = pd.concat(rank_macro_frames, ignore_index=True)

    all_model_summary: list[pd.DataFrame] = []
    all_macro_summary: list[pd.DataFrame] = []
    all_bootstrap: list[pd.DataFrame] = []
    for split in ["validation", "all"]:
        pairs = final_nonfinal_pairs(raw, split)
        model_summary, macro_summary = summarize_final_nonfinal(pairs)
        bootstrap = bootstrap_final_nonfinal(pairs)
        all_model_summary.append(model_summary)
        all_macro_summary.append(macro_summary)
        all_bootstrap.append(bootstrap)

    model_summary = pd.concat(all_model_summary, ignore_index=True)
    macro_summary = pd.concat(all_macro_summary, ignore_index=True)
    bootstrap = pd.concat(all_bootstrap, ignore_index=True)
    margin_harm, margin_correlations = baseline_margin_vs_final_harm(
        raw, clean, "validation"
    )

    outputs = {
        "selection_by_model": selection_model,
        "selection_by_rank": selection_rank,
        "layer_profile": profiles,
        "depth_bin_profile": depth_bins,
        "depth_bin_macro": depth_bin_macro,
        "rank_cell_profile": rank_cells,
        "rank_macro": rank_macro,
        "final_vs_nonfinal_model": model_summary,
        "final_vs_nonfinal_macro": macro_summary,
        "final_vs_nonfinal_bootstrap": bootstrap,
        "margin_vs_final_harm": margin_harm,
    }
    for name, frame in outputs.items():
        frame.to_csv(OUTPUT_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")

    figure_paths = {
        "selected_layers": str(plot_selected_layers(selected)),
        "harm_by_depth": str(plot_layer_harm(profiles)),
        "margin_vs_harm": str(plot_margin_vs_harm(margin_harm)),
    }
    metadata = {
        "top_k": TOP_K,
        "primary_split": "validation",
        "validation_examples_per_model": int(
            clean[clean["source_split"] == "validation"]
            .groupby("model", observed=True)["example_id"]
            .nunique()
            .min()
        ),
        "top8_branch_rows_all": int(len(raw)),
        "top8_accepted_triplets_all": int(len(raw) // 3),
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "margin_harm_correlations": margin_correlations,
        "figures": figure_paths,
    }
    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print("\n## SELECTED DEPTH BY MODEL")
    print(selection_model.to_string(index=False))
    print("\n## FINAL VS NONFINAL: VALIDATION MACRO + BOOTSTRAP")
    print(
        bootstrap[bootstrap["source_split"] == "validation"].to_string(index=False)
    )
    print("\n## MARGIN VS FINAL DESCENT HARM")
    print(margin_harm.to_string(index=False))


if __name__ == "__main__":
    main()
