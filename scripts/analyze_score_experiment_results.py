from __future__ import annotations

import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(
    os.environ.get("TRANSFORMER_DECISION_TRACES_DATA", REPO_ROOT.parent / "data")
)
DIAGNOSTICS_ROOT = DATA_ROOT / "diagnostics_clean"
FIGURE_DIR = REPO_ROOT / "docs" / "figures"
OUTPUT_DIR = REPO_ROOT / "docs" / "score_experiment_analysis"

MODEL_SPECS = {
    "Qwen 0.5B": "Qwen-Qwen2.5-0.5B-Instruct",
    "Qwen 3B": "Qwen-Qwen2.5-3B-Instruct",
    "Qwen 7B": "Qwen-Qwen2.5-7B-Instruct",
    "Llama 1B": "meta-llama-Llama-3.2-1B-Instruct",
    "Llama 3B": "meta-llama-Llama-3.2-3B-Instruct",
}

FEATURE_LABELS = {
    "answer_choice_entropy_normalized": "Entropia",
    "answer_choice_top1_top2_logit_gap": "Luka top-1/top-2",
    "answer_choice_varentropy": "Varentropy",
}

INTERVENTION_LABELS = {
    "ascent": "Ascent",
    "descent": "Descent",
    "random_same_norm": "Losowa",
}


def read_config(path: Path) -> dict:
    with (path / "run_config.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def find_final_feature_runs(fragment: str) -> dict[str, Path]:
    feature_to_dir: dict[str, Path] = {}
    for path in sorted(DATA_ROOT.glob(f"*_{fragment}_csqa_logit_feature_score_control_pipeline")):
        config = read_config(path)
        if config.get("fit_n_examples") != 3000 or config.get("eval_n_examples") != 2000:
            continue
        if config.get("max_delta_over_hidden") != 0.01:
            continue
        for feature_name in config.get("feature_names", []):
            feature_to_dir[feature_name] = path
    missing = set(FEATURE_LABELS) - set(feature_to_dir)
    if missing:
        raise FileNotFoundError(f"Brak końcowych runów dla {fragment}: {sorted(missing)}")
    return feature_to_dir


def find_validation_ids() -> set[str]:
    diagnostic_dir = sorted(
        DIAGNOSTICS_ROOT.glob("*_csqa_logit_feature_diagnostics_pipeline")
    )[0]
    examples = pd.read_parquet(diagnostic_dir / "validation_examples.parquet")
    return set(examples["example_id"].astype(str))


def add_outcomes(raw: pd.DataFrame, clean: pd.DataFrame) -> pd.DataFrame:
    clean = clean.copy()
    clean_columns = [f"clean_logit_{choice}" for choice in "ABCDE"]
    clean["clean_pred_idx"] = clean[clean_columns].to_numpy().argmax(axis=1)
    clean["clean_is_correct"] = clean["clean_pred_idx"] == clean["true_choice_idx"]
    raw = raw.merge(
        clean[["example_id", "true_choice_idx", "clean_is_correct"]],
        on="example_id",
        how="left",
        validate="many_to_one",
    )

    steered_columns = [f"steered_logit_{choice}" for choice in "ABCDE"]
    raw["steered_pred_idx"] = raw[steered_columns].to_numpy().argmax(axis=1)
    raw["steered_is_correct"] = raw["steered_pred_idx"] == raw["true_choice_idx"]
    raw["accuracy_delta"] = (
        raw["steered_is_correct"].astype(int) - raw["clean_is_correct"].astype(int)
    )
    raw["rescued"] = (~raw["clean_is_correct"]) & raw["steered_is_correct"]
    raw["harmed"] = raw["clean_is_correct"] & (~raw["steered_is_correct"])
    raw["score_gain"] = raw["steered_score_value_local"] - raw["current_score_value"]

    truth = raw["true_choice_idx"].to_numpy(dtype=int)
    row_number = np.arange(len(raw))
    clean_logits = raw[clean_columns].to_numpy(dtype=float)
    steered_logits = raw[steered_columns].to_numpy(dtype=float)
    wrong_mask = np.ones_like(clean_logits, dtype=bool)
    wrong_mask[row_number, truth] = False
    raw["correct_choice_logit_change"] = (
        steered_logits[row_number, truth] - clean_logits[row_number, truth]
    )
    raw["best_wrong_logit_change"] = (
        np.where(wrong_mask, steered_logits, -np.inf).max(axis=1)
        - np.where(wrong_mask, clean_logits, -np.inf).max(axis=1)
    )
    raw["gap_change"] = (
        raw["correct_choice_logit_change"] - raw["best_wrong_logit_change"]
    )
    return raw


def load_final_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    validation_ids = find_validation_ids()
    raw_frames: list[pd.DataFrame] = []
    selected_frames: list[pd.DataFrame] = []
    clean_frames: list[pd.DataFrame] = []

    for model, fragment in MODEL_SPECS.items():
        feature_runs = find_final_feature_runs(fragment)
        for run_dir in dict.fromkeys(feature_runs.values()):
            config = read_config(run_dir)
            run_features = [
                feature for feature, mapped_dir in feature_runs.items() if mapped_dir == run_dir
            ]
            tag = config["eval_split_tag"]
            clean = pd.read_parquet(run_dir / f"{tag}_clean_final_outputs.parquet")
            raw = pd.read_parquet(
                run_dir / f"{tag}_score_control_policy_outputs_raw.parquet"
            )
            raw = raw[raw["feature_name"].isin(run_features)].copy()
            raw = add_outcomes(raw, clean)
            raw["model"] = model
            raw["source_split"] = np.where(
                raw["example_id"].astype(str).isin(validation_ids),
                "validation",
                "train_topup",
            )
            raw_frames.append(raw)

            selected = pd.read_parquet(run_dir / "selected_layers_by_feature.parquet")
            selected = selected[selected["feature_name"].isin(run_features)].copy()
            selected["model"] = model
            selected_frames.append(selected)

            clean = clean.copy()
            clean_columns = [f"clean_logit_{choice}" for choice in "ABCDE"]
            clean["clean_pred_idx"] = clean[clean_columns].to_numpy().argmax(axis=1)
            clean["clean_is_correct"] = clean["clean_pred_idx"] == clean["true_choice_idx"]
            clean["model"] = model
            clean["source_split"] = np.where(
                clean["example_id"].astype(str).isin(validation_ids),
                "validation",
                "train_topup",
            )
            clean_frames.append(clean)

    raw = pd.concat(raw_frames, ignore_index=True)
    selected = pd.concat(selected_frames, ignore_index=True).drop_duplicates(
        ["model", "feature_name", "layer_number"]
    )
    raw = raw.merge(
        selected[
            ["model", "feature_name", "layer_number", "selection_rank", "ks_statistic"]
        ],
        on=["model", "feature_name", "layer_number"],
        how="left",
        validate="many_to_one",
    )
    clean = pd.concat(clean_frames, ignore_index=True).drop_duplicates(
        ["model", "example_id"]
    )
    raw["feature"] = raw["feature_name"].map(FEATURE_LABELS)
    raw["intervention"] = raw["intervention_type"].map(INTERVENTION_LABELS)
    return raw, selected, clean


def summarize_acceptance(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = [("Wszystkie", "Wszystkie", raw)]
    groups.extend(
        (model, FEATURE_LABELS[feature_name], group)
        for (model, feature_name), group in raw.groupby(
            ["model", "feature_name"], observed=True
        )
    )
    for model, feature, group in groups:
        selected_layer_count = group[
            ["model", "feature_name", "layer_number"]
        ].drop_duplicates().shape[0]
        # Końcowy protokół przewidywał dokładnie 2000 przykładów dla każdej
        # komórki model–cecha–warstwa. Brak przykładu w pliku wynikowym oznacza
        # brak zaakceptowanej interwencji, a nie mniejszy plan eksperymentalny.
        example_count = 2000
        planned_triplets = selected_layer_count * example_count
        recorded_triplets = group[
            ["model", "example_id", "feature_name", "layer_number"]
        ].drop_duplicates().shape[0]
        rows.append(
            {
                "model": model,
                "feature": feature,
                "selected_layers": selected_layer_count,
                "examples": example_count,
                "planned_example_layer_pairs": planned_triplets,
                "recorded_accepted_pairs": recorded_triplets,
                "retention_rate": recorded_triplets / planned_triplets,
                "recorded_branch_rows": len(group),
            }
        )
    return pd.DataFrame(rows)


def summarize_outcomes(accepted: pd.DataFrame) -> pd.DataFrame:
    group_columns = ["source_split", "intervention_type"]
    summary = accepted.groupby(group_columns, observed=True).agg(
        rows=("example_id", "size"),
        unique_examples=("example_id", "nunique"),
        clean_accuracy=("clean_is_correct", "mean"),
        steered_accuracy=("steered_is_correct", "mean"),
        delta_accuracy=("accuracy_delta", "mean"),
        rescued=("rescued", "sum"),
        harmed=("harmed", "sum"),
        mean_score_gain=("score_gain", "mean"),
        mean_gap_change=("gap_change", "mean"),
    ).reset_index()
    summary["rescue_rate_all"] = summary["rescued"] / summary["rows"]
    summary["harm_rate_all"] = summary["harmed"] / summary["rows"]
    return summary


def summarize_outcomes_by(
    accepted: pd.DataFrame, group_columns: list[str]
) -> pd.DataFrame:
    summary = accepted.groupby(group_columns + ["intervention_type"], observed=True).agg(
        rows=("example_id", "size"),
        unique_examples=("example_id", "nunique"),
        clean_accuracy=("clean_is_correct", "mean"),
        steered_accuracy=("steered_is_correct", "mean"),
        delta_accuracy=("accuracy_delta", "mean"),
        rescue_rate=("rescued", "mean"),
        harm_rate=("harmed", "mean"),
        mean_score_gain=("score_gain", "mean"),
        mean_gap_change=("gap_change", "mean"),
    ).reset_index()
    return summary


def summarize_conditional_outcomes(accepted: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (source_split, intervention_type), group in accepted.groupby(
        ["source_split", "intervention_type"], observed=True
    ):
        initially_wrong = ~group["clean_is_correct"]
        initially_correct = group["clean_is_correct"]
        rows.append(
            {
                "source_split": source_split,
                "intervention_type": intervention_type,
                "initially_wrong_rows": int(initially_wrong.sum()),
                "initially_correct_rows": int(initially_correct.sum()),
                "rescue_rate_given_wrong": float(
                    group.loc[initially_wrong, "rescued"].mean()
                ),
                "harm_rate_given_correct": float(
                    group.loc[initially_correct, "harmed"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def matched_triplets(accepted: pd.DataFrame) -> pd.DataFrame:
    keys = ["model", "example_id", "source_split", "feature_name", "layer_number"]
    value_columns = [
        "steered_is_correct",
        "rescued",
        "harmed",
        "score_gain",
        "gap_change",
    ]
    wide = accepted.set_index(keys + ["intervention_type"])[value_columns].unstack(
        "intervention_type"
    )
    wide.columns = [f"{metric}__{branch}" for metric, branch in wide.columns]
    wide = wide.reset_index()
    required = [
        f"steered_is_correct__{branch}"
        for branch in ("ascent", "descent", "random_same_norm")
    ]
    wide = wide.dropna(subset=required).copy()
    wide["ascent_minus_random_accuracy"] = (
        wide["steered_is_correct__ascent"].astype(int)
        - wide["steered_is_correct__random_same_norm"].astype(int)
    )
    wide["ascent_minus_descent_accuracy"] = (
        wide["steered_is_correct__ascent"].astype(int)
        - wide["steered_is_correct__descent"].astype(int)
    )
    wide["ascent_minus_random_rescue"] = (
        wide["rescued__ascent"].astype(int)
        - wide["rescued__random_same_norm"].astype(int)
    )
    wide["random_minus_ascent_harm"] = (
        wide["harmed__random_same_norm"].astype(int)
        - wide["harmed__ascent"].astype(int)
    )
    return wide


def bootstrap_cluster_mean(
    frame: pd.DataFrame,
    value_column: str,
    cluster_column: str = "example_id",
    iterations: int = 5000,
    seed: int = 42,
) -> dict[str, float]:
    cluster_means = frame.groupby(cluster_column, observed=True)[value_column].mean()
    values = cluster_means.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    boot = np.empty(iterations, dtype=float)
    for index in range(iterations):
        boot[index] = rng.choice(values, size=len(values), replace=True).mean()
    return {
        "estimate": float(values.mean()),
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
        "clusters": int(len(values)),
    }


def paired_bootstrap_summary(wide: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source_split in ["all", "validation", "train_topup"]:
        part = wide if source_split == "all" else wide[wide["source_split"] == source_split]
        for metric in [
            "ascent_minus_random_accuracy",
            "ascent_minus_descent_accuracy",
            "ascent_minus_random_rescue",
            "random_minus_ascent_harm",
        ]:
            result = bootstrap_cluster_mean(part, metric)
            result.update({"source_split": source_split, "metric": metric})
            rows.append(result)
    return pd.DataFrame(rows)


def roc_auc_binary(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    good = np.isfinite(scores)
    labels = labels[good]
    scores = scores[good]
    n_positive = int(labels.sum())
    n_negative = int((~labels).sum())
    if not n_positive or not n_negative:
        return math.nan
    ranks = rankdata(scores, method="average")
    return float(
        (ranks[labels].sum() - n_positive * (n_positive + 1) / 2)
        / (n_positive * n_negative)
    )


def score_triage_summary(raw: pd.DataFrame, clean: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_rows = raw[
        raw["current_supported"] & raw["current_score_value"].notna()
    ].drop_duplicates(["model", "example_id", "feature_name", "layer_number"])
    example_scores = score_rows.groupby(
        ["model", "example_id", "source_split"], observed=True
    ).agg(
        mean_score=("current_score_value", "mean"),
        median_score=("current_score_value", "median"),
        supported_measurements=("current_score_value", "size"),
    ).reset_index()
    example_scores = example_scores.merge(
        clean[["model", "example_id", "clean_is_correct"]],
        on=["model", "example_id"],
        how="left",
        validate="one_to_one",
    )

    auc_rows = []
    coverage_rows = []
    for source_split in ["all", "validation", "train_topup"]:
        source = (
            example_scores
            if source_split == "all"
            else example_scores[example_scores["source_split"] == source_split]
        )
        for model in ["Wszystkie", *MODEL_SPECS]:
            group = source if model == "Wszystkie" else source[source["model"] == model]
            auc_rows.append(
                {
                    "source_split": source_split,
                    "model": model,
                    "n": len(group),
                    "auc": roc_auc_binary(
                        group["clean_is_correct"].to_numpy(),
                        group["mean_score"].to_numpy(),
                    ),
                }
            )
            ordered = group.sort_values("mean_score", ascending=False)
            for coverage in [1.0, 0.8, 0.6, 0.4, 0.2]:
                count = max(1, int(round(len(ordered) * coverage)))
                kept = ordered.head(count)
                coverage_rows.append(
                    {
                        "source_split": source_split,
                        "model": model,
                        "coverage": coverage,
                        "n": count,
                        "accuracy": kept["clean_is_correct"].mean(),
                        "score_threshold": kept["mean_score"].min(),
                    }
                )
    return pd.DataFrame(auc_rows), pd.DataFrame(coverage_rows)


def effect_vs_diagnostic(
    accepted: pd.DataFrame, selected: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ascent = accepted[accepted["intervention_type"] == "ascent"]
    effects = ascent.groupby(
        ["model", "feature_name", "layer_number"], observed=True
    ).agg(
        rows=("example_id", "size"),
        delta_accuracy=("accuracy_delta", "mean"),
        mean_score_gain=("score_gain", "mean"),
        mean_gap_change=("gap_change", "mean"),
        rescue_rate=("rescued", "mean"),
        harm_rate=("harmed", "mean"),
    ).reset_index()
    effects = effects.merge(
        selected[["model", "feature_name", "layer_number", "ks_statistic"]],
        on=["model", "feature_name", "layer_number"],
        how="left",
        validate="one_to_one",
    )
    max_layers = selected.groupby("model")["layer_number"].max().to_dict()
    effects["layer_pct"] = effects.apply(
        lambda row: 100.0 * row["layer_number"] / max_layers[row["model"]], axis=1
    )
    rows = []
    for x_column in ["ks_statistic", "layer_pct"]:
        for y_column in [
            "delta_accuracy",
            "mean_score_gain",
            "mean_gap_change",
            "rescue_rate",
            "harm_rate",
        ]:
            rho, p_value = spearmanr(effects[x_column], effects[y_column])
            rows.append(
                {
                    "x": x_column,
                    "y": y_column,
                    "rho": rho,
                    "p_value_naive": p_value,
                    "n_layer_feature_model_cells": len(effects),
                }
            )
    return effects, pd.DataFrame(rows)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / exponential.sum(axis=1, keepdims=True)


def ece_score(probabilities: np.ndarray, truth: np.ndarray, bins: int = 10) -> float:
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    correct = prediction == truth
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            result += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(result)


def probability_metrics(logits: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    probabilities = softmax(logits.astype(float))
    row_number = np.arange(len(truth))
    one_hot = np.zeros_like(probabilities)
    one_hot[row_number, truth] = 1.0
    return {
        "accuracy": float((probabilities.argmax(axis=1) == truth).mean()),
        "mean_confidence": float(probabilities.max(axis=1).mean()),
        "nll": float(-np.log(probabilities[row_number, truth].clip(1e-12)).mean()),
        "brier": float(np.square(probabilities - one_hot).sum(axis=1).mean()),
        "ece_10_bins": ece_score(probabilities, truth),
    }


def calibration_summary(accepted: pd.DataFrame) -> pd.DataFrame:
    keys = ["model", "example_id", "feature_name", "layer_number"]
    branch_counts = accepted.groupby(keys, observed=True)["intervention_type"].nunique()
    valid_keys = branch_counts[branch_counts == 3].index
    indexed = accepted.set_index(keys)
    paired = indexed.loc[indexed.index.isin(valid_keys)].reset_index()
    rows = []
    clean = paired.drop_duplicates(keys)
    clean_logits = clean[[f"clean_logit_{choice}" for choice in "ABCDE"]].to_numpy()
    metrics = probability_metrics(clean_logits, clean["true_choice_idx"].to_numpy(dtype=int))
    metrics.update({"variant": "clean", "rows": len(clean)})
    rows.append(metrics)
    for branch, group in paired.groupby("intervention_type", observed=True):
        logits = group[[f"steered_logit_{choice}" for choice in "ABCDE"]].to_numpy()
        metrics = probability_metrics(logits, group["true_choice_idx"].to_numpy(dtype=int))
        metrics.update({"variant": branch, "rows": len(group)})
        rows.append(metrics)
    return pd.DataFrame(rows)


def load_cap_sweep() -> pd.DataFrame:
    frames = []
    allowed_models = {
        "Qwen/Qwen2.5-3B-Instruct": "Qwen 3B",
        "meta-llama/Llama-3.2-3B-Instruct": "Llama 3B",
    }
    for run_dir in sorted(DATA_ROOT.glob("*_csqa_logit_feature_score_control_pipeline")):
        if not run_dir.name.startswith("20260708-") or run_dir.name >= "20260708-120000":
            continue
        config = read_config(run_dir)
        model_id = config.get("model_id")
        if model_id not in allowed_models:
            continue
        if config.get("fit_split") != "validation" or config.get("eval_split") != "train":
            continue
        if config.get("fit_limit") != 1000 or config.get("eval_limit") != 650:
            continue
        cap = config.get("max_delta_over_hidden")
        if cap not in {0.0025, 0.005, 0.01, 0.015}:
            continue
        raw_path = run_dir / "train_score_control_policy_outputs_raw.parquet"
        if not raw_path.exists():
            continue
        clean = pd.read_parquet(run_dir / "train_clean_final_outputs.parquet")
        raw = add_outcomes(pd.read_parquet(raw_path), clean)
        raw["model"] = allowed_models[model_id]
        raw["cap"] = float(cap)
        frames.append(raw)
    if len(frames) != 8:
        raise RuntimeError(f"Oczekiwano 8 runów cap sweep, znaleziono {len(frames)}")
    return pd.concat(frames, ignore_index=True)


def cap_sweep_summary(raw: pd.DataFrame) -> pd.DataFrame:
    accepted = raw[raw["accepted_intervention"]].copy()
    return accepted.groupby(
        ["model", "cap", "intervention_type"], observed=True
    ).agg(
        rows=("example_id", "size"),
        unique_examples=("example_id", "nunique"),
        delta_accuracy=("accuracy_delta", "mean"),
        rescue_rate=("rescued", "mean"),
        harm_rate=("harmed", "mean"),
        mean_score_gain=("score_gain", "mean"),
        mean_gap_change=("gap_change", "mean"),
        mean_norm=("delta_over_token_hidden_l2", "mean"),
    ).reset_index()


def baseline_accuracy_summary(clean: pd.DataFrame) -> pd.DataFrame:
    return clean.groupby(["model", "source_split"], observed=True).agg(
        examples=("example_id", "nunique"),
        accuracy=("clean_is_correct", "mean"),
    ).reset_index()


def plot_triage(coverage: pd.DataFrame) -> Path:
    data = coverage[
        (coverage["source_split"] == "validation")
        & (coverage["model"] != "Wszystkie")
    ]
    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    for model, group in data.groupby("model", observed=True):
        axis.plot(
            100 * group["coverage"],
            100 * group["accuracy"],
            marker="o",
            linewidth=2,
            label=model,
        )
    axis.set_xlabel("Pokrycie: odsetek odpowiedzi pozostawionych bez eskalacji [%]")
    axis.set_ylabel("Trafność w pozostawionej części [%]")
    axis.set_title("Eksploracyjne wykorzystanie średniego score do selekcji odpowiedzi")
    axis.grid(alpha=0.25)
    axis.legend(ncol=2, frameon=False)
    figure.tight_layout()
    path = FIGURE_DIR / "score_risk_coverage.png"
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_diagnostic_vs_causal(effects: pd.DataFrame) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 4.1))
    colors = {
        "answer_choice_entropy_normalized": "#3b82f6",
        "answer_choice_top1_top2_logit_gap": "#f97316",
        "answer_choice_varentropy": "#10b981",
    }
    for feature_name, group in effects.groupby("feature_name", observed=True):
        label = FEATURE_LABELS[feature_name]
        color = colors[feature_name]
        axes[0].scatter(
            group["ks_statistic"],
            100 * group["delta_accuracy"],
            alpha=0.72,
            s=34,
            label=label,
            color=color,
        )
        axes[1].scatter(
            group["ks_statistic"],
            group["mean_gap_change"],
            alpha=0.72,
            s=34,
            label=label,
            color=color,
        )
    axes[0].axhline(0.0, color="black", linewidth=1, alpha=0.5)
    axes[0].set_ylabel("Zmiana trafności po ascent [p.p.]")
    axes[1].axhline(0.0, color="black", linewidth=1, alpha=0.5)
    axes[1].set_ylabel("Zmiana marginesu: poprawna − najlepsza błędna")
    for axis in axes:
        axis.set_xlabel("Statystyka KS w próbie dopasowania")
        axis.grid(alpha=0.22)
    axes[0].legend(frameon=False, fontsize=9)
    figure.suptitle("Siła diagnostyczna warstwy a skutek interwencji ascent")
    figure.tight_layout()
    path = FIGURE_DIR / "score_diagnostic_vs_intervention.png"
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_cap_sweep(summary: pd.DataFrame) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 4.1), sharey=True)
    branch_style = {
        "ascent": ("#2563eb", "o"),
        "descent": ("#dc2626", "s"),
        "random_same_norm": ("#6b7280", "^"),
    }
    for axis, model in zip(axes, ["Qwen 3B", "Llama 3B"]):
        model_data = summary[summary["model"] == model]
        for branch, group in model_data.groupby("intervention_type", observed=True):
            color, marker = branch_style[branch]
            axis.plot(
                100 * group["cap"],
                100 * group["delta_accuracy"],
                marker=marker,
                linewidth=2,
                color=color,
                label=INTERVENTION_LABELS[branch],
            )
        axis.axhline(0.0, color="black", linewidth=1, alpha=0.5)
        axis.set_title(model)
        axis.set_xlabel("Limit normy perturbacji względem stanu [%]")
        axis.grid(alpha=0.22)
    axes[0].set_ylabel("Zmiana trafności [p.p.]")
    axes[1].legend(frameon=False)
    figure.suptitle("Eksploracyjny test skali perturbacji (odrębny protokół 1000/650)")
    figure.tight_layout()
    path = FIGURE_DIR / "score_perturbation_norm_sweep.png"
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return path


def save_frame(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(OUTPUT_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    full_raw, selected, clean = load_final_data()
    raw = full_raw[full_raw["selection_rank"] <= 4].copy()
    selected_analysis = selected[selected["selection_rank"] <= 4].copy()
    accepted = raw[raw["accepted_intervention"]].copy()
    acceptance = summarize_acceptance(raw)
    outcomes = summarize_outcomes(accepted)
    outcomes_by_model = summarize_outcomes_by(accepted, ["model"])
    outcomes_by_feature = summarize_outcomes_by(accepted, ["feature"])
    conditional = summarize_conditional_outcomes(accepted)
    wide = matched_triplets(accepted)
    bootstrap = paired_bootstrap_summary(wide)
    auc, coverage = score_triage_summary(raw, clean)
    effects, correlations = effect_vs_diagnostic(accepted, selected_analysis)
    calibration = calibration_summary(accepted)
    baseline = baseline_accuracy_summary(clean)

    cap_raw = load_cap_sweep()
    cap_summary = cap_sweep_summary(cap_raw)

    frames = {
        "acceptance": acceptance,
        "outcomes_by_split": outcomes,
        "outcomes_by_model": outcomes_by_model,
        "outcomes_by_feature": outcomes_by_feature,
        "conditional_outcomes": conditional,
        "paired_bootstrap": bootstrap,
        "score_auc": auc,
        "score_risk_coverage": coverage,
        "diagnostic_vs_causal_cells": effects,
        "diagnostic_vs_causal_correlations": correlations,
        "calibration_exploratory": calibration,
        "baseline_accuracy": baseline,
        "cap_sweep": cap_summary,
    }
    for name, frame in frames.items():
        save_frame(frame, name)

    figures = {
        "risk_coverage": str(plot_triage(coverage)),
        "diagnostic_vs_causal": str(plot_diagnostic_vs_causal(effects)),
        "cap_sweep": str(plot_cap_sweep(cap_summary)),
    }

    metadata = {
        "full_top8_raw_rows": int(len(full_raw)),
        "analysis_top_k": 4,
        "analysis_top4_raw_rows": int(len(raw)),
        "analysis_top4_accepted_rows": int(len(accepted)),
        "final_unique_model_examples": int(
            clean[["model", "example_id"]].drop_duplicates().shape[0]
        ),
        "validation_examples_per_model": int(
            clean[clean["source_split"] == "validation"]
            .groupby("model")["example_id"]
            .nunique()
            .iloc[0]
        ),
        "train_topup_examples_per_model": int(
            clean[clean["source_split"] == "train_topup"]
            .groupby("model")["example_id"]
            .nunique()
            .iloc[0]
        ),
        "matched_triplets": int(len(wide)),
        "cap_sweep_raw_rows": int(len(cap_raw)),
        "figures": figures,
    }
    with (OUTPUT_DIR / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    for name in [
        "outcomes_by_split",
        "conditional_outcomes",
        "paired_bootstrap",
        "score_auc",
        "diagnostic_vs_causal_correlations",
        "calibration_exploratory",
        "cap_sweep",
    ]:
        print(f"\n## {name}\n{frames[name].to_string(index=False)}")


if __name__ == "__main__":
    main()
