"""Wspólne nazwy i wyszukiwanie kompletnych runów eksperymentu."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd


RUN_SPECS = [
    ("Qwen 0.5B", "Qwen-Qwen2.5-0.5B-Instruct"),
    ("Qwen 3B", "Qwen-Qwen2.5-3B-Instruct"),
    ("Qwen 7B", "Qwen-Qwen2.5-7B-Instruct"),
    ("Llama 1B", "meta-llama-Llama-3.2-1B-Instruct"),
    ("Llama 3B", "meta-llama-Llama-3.2-3B-Instruct"),
]
MODEL_ORDER = [label for label, _ in RUN_SPECS]

FEATURE_ORDER = [
    "answer_choice_top1_top2_logit_gap",
    "answer_choice_varentropy",
    "answer_choice_entropy_normalized",
]
FEATURE_LABELS = {
    "answer_choice_top1_top2_logit_gap": "luka logitów top-1/top-2",
    "answer_choice_varentropy": "varentropy odpowiedzi",
    "answer_choice_entropy_normalized": "entropia odpowiedzi",
}
INTERVENTION_ORDER = ["ascent", "descent", "random_same_norm"]
INTERVENTION_LABELS = {
    "ascent": "ascent",
    "descent": "descent",
    "random_same_norm": "random",
}


def find_repo_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    return next((path for path in (start, *start.parents) if (path / "src").is_dir()), start)


def default_data_dir() -> Path:
    repo_root = find_repo_root()
    return Path(os.environ.get("TRANSFORMER_DECISION_TRACES_DATA", repo_root.parent / "data"))


def read_run_config(run_dir: Path) -> dict:
    path = run_dir / "run_config.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def split_tags(config: dict) -> tuple[str, str]:
    return (
        config.get("fit_split_tag", config.get("fit_split", "train")),
        config.get("eval_split_tag", config.get("eval_split", "validation")),
    )


def ordered_features(names) -> list[str]:
    unique = list(pd.unique(pd.Series(names).dropna()))
    return [name for name in FEATURE_ORDER if name in unique] + sorted(
        name for name in unique if name not in FEATURE_ORDER
    )


def apply_model_order(frame: pd.DataFrame, column: str = "model") -> pd.DataFrame:
    frame = frame.copy()
    frame[column] = pd.Categorical(frame[column], categories=MODEL_ORDER, ordered=True)
    return frame


def latest_diagnostic_run(data_dir: Path, model_fragment: str) -> Path:
    candidates = sorted(
        path
        for path in data_dir.rglob(f"*_{model_fragment}_csqa_logit_feature_diagnostics_pipeline")
        if path.is_dir()
    )
    if not candidates:
        raise FileNotFoundError(f"Brak kompletnego runu diagnostycznego dla {model_fragment}")
    return candidates[-1]


def latest_control_runs_by_feature(
    data_dir: Path,
    model_fragment: str,
    required_features: list[str] | None = None,
) -> dict[str, Path]:
    """Znajdź najnowszy kompletny run każdej cechy dla danego modelu."""
    candidates = sorted(
        path
        for path in data_dir.rglob(f"*_{model_fragment}_csqa_logit_feature_score_control_pipeline")
        if path.is_dir()
    )
    feature_to_dir: dict[str, Path] = {}
    for run_dir in candidates:
        config = read_run_config(run_dir)
        _, eval_tag = split_tags(config)
        required_files = [
            run_dir / f"{eval_tag}_score_control_policy_outputs_raw.parquet",
            run_dir / f"{eval_tag}_clean_final_outputs.parquet",
            run_dir / "selected_layers_by_feature.parquet",
            run_dir / "feature_layer_separation_summary.parquet",
            run_dir / "fit_distribution_grid.parquet",
        ]
        if not all(path.exists() for path in required_files):
            continue
        features = config.get("feature_names") or pd.read_parquet(
            run_dir / "selected_layers_by_feature.parquet",
            columns=["feature_name"],
        )["feature_name"].dropna().unique()
        for feature_name in features:
            if required_features is None or feature_name in required_features:
                feature_to_dir[str(feature_name)] = run_dir

    required_features = required_features or FEATURE_ORDER
    missing = [name for name in required_features if name not in feature_to_dir]
    if missing:
        raise FileNotFoundError(f"Brak runów {model_fragment} dla cech: {', '.join(missing)}")
    return {name: feature_to_dir[name] for name in ordered_features(required_features)}
