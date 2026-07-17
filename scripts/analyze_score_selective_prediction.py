from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_score_experiment_results import (  # noqa: E402
    MODEL_SPECS,
    load_final_data,
    roc_auc_binary,
)


OUTPUT_DIR = REPO_ROOT / "docs" / "score_selective_prediction"
FIGURE_DIR = REPO_ROOT / "docs" / "figures"

MODEL_ORDER = list(MODEL_SPECS)
MODEL_COLORS = {
    "Qwen 0.5B": "#60a5fa",
    "Qwen 3B": "#2563eb",
    "Qwen 7B": "#1e3a8a",
    "Llama 1B": "#f59e0b",
    "Llama 3B": "#dc2626",
}
TOP_K_VALUES = [4, 8]
COVERAGES = [1.0, 0.8, 0.6, 0.4, 0.2]
BOOTSTRAP_ITERATIONS = 3000
BOOTSTRAP_SEED = 20260715


def example_scores(raw: pd.DataFrame, clean: pd.DataFrame, top_k: int) -> pd.DataFrame:
    source = raw[
        (raw["selection_rank"] <= top_k)
        & (raw["source_split"] == "validation")
        & raw["current_supported"]
        & raw["current_score_value"].notna()
    ].copy()
    source = source.drop_duplicates(
        ["model", "example_id", "feature_name", "layer_number"]
    )
    result = source.groupby(["model", "example_id"], observed=True).agg(
        mean_score=("current_score_value", "mean"),
        median_score=("current_score_value", "median"),
        supported_measurements=("current_score_value", "size"),
    ).reset_index()
    truth = clean[clean["source_split"] == "validation"][
        ["model", "example_id", "clean_is_correct"]
    ].drop_duplicates(["model", "example_id"])
    result = result.merge(
        truth,
        on=["model", "example_id"],
        how="inner",
        validate="one_to_one",
    )
    result["top_k"] = top_k
    result["model"] = pd.Categorical(result["model"], MODEL_ORDER, ordered=True)
    return result.sort_values(["model", "example_id"]).reset_index(drop=True)


def auc_point_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for model in MODEL_ORDER:
        group = frame[frame["model"] == model]
        rows.append(
            {
                "top_k": int(frame["top_k"].iloc[0]),
                "model": model,
                "examples": len(group),
                "auc": roc_auc_binary(
                    group["clean_is_correct"].to_numpy(),
                    group["mean_score"].to_numpy(),
                ),
            }
        )
    result = pd.DataFrame(rows)
    result = pd.concat(
        [
            result,
            pd.DataFrame(
                [
                    {
                        "top_k": int(frame["top_k"].iloc[0]),
                        "model": "Makro po modelach",
                        "examples": result["examples"].sum(),
                        "auc": result["auc"].mean(),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    return result


def coverage_point_summary(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    for model in MODEL_ORDER:
        group = frame[frame["model"] == model].sort_values(
            "mean_score", ascending=False
        )
        for coverage in COVERAGES:
            count = max(1, int(round(len(group) * coverage)))
            retained = group.head(count)
            rows.append(
                {
                    "top_k": int(frame["top_k"].iloc[0]),
                    "model": model,
                    "coverage": coverage,
                    "retained_examples": count,
                    "accuracy": retained["clean_is_correct"].mean(),
                    "minimum_retained_score": retained["mean_score"].min(),
                }
            )
    model_summary = pd.DataFrame(rows)
    macro = model_summary.groupby(["top_k", "coverage"], observed=True).agg(
        models=("model", "nunique"),
        retained_examples=("retained_examples", "sum"),
        accuracy_macro_models=("accuracy", "mean"),
    ).reset_index()
    return model_summary, macro


def quintile_summary(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = frame.copy()
    frame["score_quintile"] = frame.groupby("model", observed=True)[
        "mean_score"
    ].transform(
        lambda values: pd.qcut(
            values,
            5,
            labels=[1, 2, 3, 4, 5],
            duplicates="drop",
        )
    )
    by_model = frame.groupby(
        ["top_k", "model", "score_quintile"], observed=True
    ).agg(
        examples=("example_id", "size"),
        accuracy=("clean_is_correct", "mean"),
        mean_score=("mean_score", "mean"),
    ).reset_index()
    macro = by_model.groupby(["top_k", "score_quintile"], observed=True).agg(
        models=("model", "nunique"),
        examples=("examples", "sum"),
        accuracy_macro_models=("accuracy", "mean"),
        mean_score_macro_models=("mean_score", "mean"),
    ).reset_index()
    return by_model, macro


def bootstrap_auc_and_coverage(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(BOOTSTRAP_SEED + int(frame["top_k"].iloc[0]))
    auc_boot_by_model: dict[str, np.ndarray] = {}
    coverage_boot_by_model: dict[str, np.ndarray] = {}
    auc_rows: list[dict] = []
    coverage_rows: list[dict] = []

    for model in MODEL_ORDER:
        group = frame[frame["model"] == model]
        scores = group["mean_score"].to_numpy(dtype=float)
        labels = group["clean_is_correct"].to_numpy(dtype=bool)
        auc_boot = np.empty(BOOTSTRAP_ITERATIONS, dtype=float)
        coverage_boot = np.empty((BOOTSTRAP_ITERATIONS, len(COVERAGES)), dtype=float)
        for iteration in range(BOOTSTRAP_ITERATIONS):
            indices = rng.integers(0, len(group), size=len(group))
            sample_scores = scores[indices]
            sample_labels = labels[indices]
            auc_boot[iteration] = roc_auc_binary(sample_labels, sample_scores)
            order = np.argsort(sample_scores)[::-1]
            sorted_labels = sample_labels[order]
            for coverage_index, coverage in enumerate(COVERAGES):
                count = max(1, int(round(len(group) * coverage)))
                coverage_boot[iteration, coverage_index] = sorted_labels[:count].mean()

        auc_boot_by_model[model] = auc_boot
        coverage_boot_by_model[model] = coverage_boot
        auc_point = roc_auc_binary(labels, scores)
        auc_rows.append(
            {
                "top_k": int(frame["top_k"].iloc[0]),
                "model": model,
                "examples": len(group),
                "auc": auc_point,
                "ci_low": np.nanquantile(auc_boot, 0.025),
                "ci_high": np.nanquantile(auc_boot, 0.975),
                "iterations": BOOTSTRAP_ITERATIONS,
            }
        )
        point_order = np.argsort(scores)[::-1]
        point_labels = labels[point_order]
        for coverage_index, coverage in enumerate(COVERAGES):
            count = max(1, int(round(len(group) * coverage)))
            coverage_rows.append(
                {
                    "top_k": int(frame["top_k"].iloc[0]),
                    "model": model,
                    "coverage": coverage,
                    "accuracy": point_labels[:count].mean(),
                    "ci_low": np.quantile(
                        coverage_boot[:, coverage_index], 0.025
                    ),
                    "ci_high": np.quantile(
                        coverage_boot[:, coverage_index], 0.975
                    ),
                    "iterations": BOOTSTRAP_ITERATIONS,
                }
            )

    macro_auc_boot = np.stack(
        [auc_boot_by_model[model] for model in MODEL_ORDER]
    ).mean(axis=0)
    macro_auc_point = np.mean([row["auc"] for row in auc_rows])
    auc_rows.append(
        {
            "top_k": int(frame["top_k"].iloc[0]),
            "model": "Makro po modelach",
            "examples": len(frame),
            "auc": macro_auc_point,
            "ci_low": np.nanquantile(macro_auc_boot, 0.025),
            "ci_high": np.nanquantile(macro_auc_boot, 0.975),
            "iterations": BOOTSTRAP_ITERATIONS,
        }
    )

    macro_coverage_boot = np.stack(
        [coverage_boot_by_model[model] for model in MODEL_ORDER]
    ).mean(axis=0)
    _, macro_points = coverage_point_summary(frame)
    for coverage_index, coverage in enumerate(COVERAGES):
        point_row = macro_points[macro_points["coverage"] == coverage].iloc[0]
        coverage_rows.append(
            {
                "top_k": int(frame["top_k"].iloc[0]),
                "model": "Makro po modelach",
                "coverage": coverage,
                "accuracy": point_row["accuracy_macro_models"],
                "ci_low": np.quantile(
                    macro_coverage_boot[:, coverage_index], 0.025
                ),
                "ci_high": np.quantile(
                    macro_coverage_boot[:, coverage_index], 0.975
                ),
                "iterations": BOOTSTRAP_ITERATIONS,
            }
        )
    return pd.DataFrame(auc_rows), pd.DataFrame(coverage_rows)


def plot_selective_prediction(
    auc: pd.DataFrame, coverage: pd.DataFrame, top_k: int = 4
) -> Path:
    auc_data = auc[auc["top_k"] == top_k].copy()
    coverage_data = coverage[coverage["top_k"] == top_k].copy()
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.4))

    for model in MODEL_ORDER:
        group = coverage_data[coverage_data["model"] == model].sort_values("coverage")
        axes[0].plot(
            100.0 * group["coverage"],
            100.0 * group["accuracy"],
            marker="o",
            linewidth=1.8,
            color=MODEL_COLORS[model],
            label=model,
        )
    macro = coverage_data[coverage_data["model"] == "Makro po modelach"].sort_values(
        "coverage"
    )
    axes[0].plot(
        100.0 * macro["coverage"],
        100.0 * macro["accuracy"],
        marker="o",
        linewidth=3.0,
        color="#111827",
        label="Makro po modelach",
    )
    axes[0].set_xlabel("Pokrycie — odsetek zachowanych odpowiedzi [%]")
    axes[0].set_ylabel("Trafność zachowanych odpowiedzi [%]")
    axes[0].set_title("Trafność a pokrycie")
    axes[0].set_xlim(15, 105)
    axes[0].set_ylim(45, 100)
    axes[0].grid(alpha=0.22)
    axes[0].legend(fontsize=8, frameon=True)

    model_auc = auc_data[auc_data["model"].isin(MODEL_ORDER)].copy()
    model_auc["model"] = pd.Categorical(
        model_auc["model"], categories=MODEL_ORDER, ordered=True
    )
    model_auc = model_auc.sort_values("model")
    x = np.arange(len(model_auc))
    axes[1].bar(
        x,
        model_auc["auc"],
        color=[MODEL_COLORS[str(model)] for model in model_auc["model"]],
        edgecolor="black",
        linewidth=0.5,
    )
    axes[1].errorbar(
        x,
        model_auc["auc"],
        yerr=np.vstack(
            [
                model_auc["auc"] - model_auc["ci_low"],
                model_auc["ci_high"] - model_auc["auc"],
            ]
        ),
        fmt="none",
        color="#111827",
        capsize=4,
        linewidth=1.2,
    )
    axes[1].axhline(0.5, color="#9ca3af", linestyle="--", linewidth=1.0)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(model_auc["model"].astype(str), rotation=25, ha="right")
    axes[1].set_ylabel("AUC")
    axes[1].set_ylim(0.5, 0.85)
    axes[1].set_title("Rozróżnianie odpowiedzi poprawnych i błędnych")
    axes[1].grid(axis="y", alpha=0.22)

    figure.suptitle(
        f"Diagnostyczna użyteczność funkcji score — zbiór walidacyjny, TOP_K = {top_k}",
        y=1.01,
    )
    figure.tight_layout()
    path = FIGURE_DIR / f"score_selective_prediction_top{top_k}_validation.png"
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    raw, _, clean = load_final_data()

    example_frames: list[pd.DataFrame] = []
    auc_frames: list[pd.DataFrame] = []
    coverage_model_frames: list[pd.DataFrame] = []
    coverage_macro_frames: list[pd.DataFrame] = []
    quintile_model_frames: list[pd.DataFrame] = []
    quintile_macro_frames: list[pd.DataFrame] = []
    auc_bootstrap_frames: list[pd.DataFrame] = []
    coverage_bootstrap_frames: list[pd.DataFrame] = []

    for top_k in TOP_K_VALUES:
        examples = example_scores(raw, clean, top_k)
        auc = auc_point_summary(examples)
        coverage_model, coverage_macro = coverage_point_summary(examples)
        quintile_model, quintile_macro = quintile_summary(examples)
        auc_bootstrap, coverage_bootstrap = bootstrap_auc_and_coverage(examples)

        example_frames.append(examples)
        auc_frames.append(auc)
        coverage_model_frames.append(coverage_model)
        coverage_macro_frames.append(coverage_macro)
        quintile_model_frames.append(quintile_model)
        quintile_macro_frames.append(quintile_macro)
        auc_bootstrap_frames.append(auc_bootstrap)
        coverage_bootstrap_frames.append(coverage_bootstrap)

    outputs = {
        "example_scores": pd.concat(example_frames, ignore_index=True),
        "auc": pd.concat(auc_frames, ignore_index=True),
        "coverage_by_model": pd.concat(coverage_model_frames, ignore_index=True),
        "coverage_macro": pd.concat(coverage_macro_frames, ignore_index=True),
        "quintile_by_model": pd.concat(quintile_model_frames, ignore_index=True),
        "quintile_macro": pd.concat(quintile_macro_frames, ignore_index=True),
        "auc_bootstrap": pd.concat(auc_bootstrap_frames, ignore_index=True),
        "coverage_bootstrap": pd.concat(
            coverage_bootstrap_frames, ignore_index=True
        ),
    }
    for name, frame in outputs.items():
        frame.to_csv(OUTPUT_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")

    figure = plot_selective_prediction(
        outputs["auc_bootstrap"], outputs["coverage_bootstrap"], top_k=4
    )
    metadata = {
        "primary_top_k": 4,
        "sensitivity_top_k": 8,
        "split": "validation",
        "examples_per_model": 1221,
        "score_aggregation": "mean current_score_value over supported unique model-example-feature-layer measurements",
        "coverage_policy": "retain the highest-score fraction separately within each model",
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "figure": str(figure),
    }
    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print("\n## AUC WITH BOOTSTRAP")
    print(outputs["auc_bootstrap"].to_string(index=False))
    print("\n## COVERAGE MACRO")
    print(outputs["coverage_bootstrap"][outputs["coverage_bootstrap"]["model"] == "Makro po modelach"].to_string(index=False))
    print("\n## QUINTILES MACRO")
    print(outputs["quintile_macro"].to_string(index=False))


if __name__ == "__main__":
    main()
