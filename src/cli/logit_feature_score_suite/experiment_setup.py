"""Konfiguracja wejścia/wyjścia wspólna dla pipeline'ów score."""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import pandas as pd

from src.data.load_csqa import load_csqa
from src.score.constants import (
    DEFAULT_BACKTRACK_SCALES,
    DEFAULT_FEATURE_NAMES,
    DEFAULT_MAX_DELTA_OVER_HIDDEN,
    DEFAULT_TRAIN_LIMIT,
    GOOD_REGION_LOG_RATIO_THRESHOLD,
    GRID_POINTS,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


def now_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def slugify_model_id(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", model_id).strip("-")


def resolve_out_dir(
    out_dir: str | None,
    model_id: str,
    pipeline_name: str,
    *,
    output_group: str | None = None,
) -> Path:
    if out_dir is None:
        run_name = f"{now_id()}_{slugify_model_id(model_id)}_{pipeline_name}"
        return REPO_ROOT / "data" / "generated" / (output_group or pipeline_name) / run_name
    path = Path(out_dir)
    return path if path.is_absolute() else REPO_ROOT / path


def all_layer_numbers(num_layers: int) -> list[int]:
    return list(range(1, num_layers + 1))


def resolve_split_limit(
    *,
    split_name: str,
    explicit_limit: int | None,
    train_limit: int | None,
    validation_limit: int | None,
    default_train_limit: int | None,
) -> int | None:
    """Rozwiąż kompatybilne stare i nowe argumenty limitu próby."""
    if explicit_limit is not None:
        return explicit_limit
    if split_name == "train" and train_limit is not None:
        return train_limit
    if split_name == "validation" and validation_limit is not None:
        return validation_limit
    return default_train_limit if split_name == "train" else None


def split_artifact_tag(split_name: str, role: str, same_source_split: bool) -> str:
    return f"{split_name}_{role}" if same_source_split else split_name


def load_fit_and_eval_rows(
    *,
    fit_split: str,
    eval_split: str,
    fit_limit: int | None,
    eval_limit: int | None,
    seed: int,
    eval_top_up_from_fit_split: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, bool, dict[str, object]]:
    """Wczytaj rozłączne próby fit/eval zgodnie z protokołem eksperymentu."""
    same_source_split = fit_split == eval_split
    if same_source_split:
        needed = None if fit_limit is None or eval_limit is None else fit_limit + eval_limit
        source_rows = load_csqa(split=fit_split, limit=needed).copy()
        source_rows = source_rows.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        fit_stop = len(source_rows) if fit_limit is None else min(fit_limit, len(source_rows))
        eval_stop = len(source_rows) if eval_limit is None else min(fit_stop + eval_limit, len(source_rows))
        fit_rows = source_rows.iloc[:fit_stop].copy().reset_index(drop=True)
        eval_rows = source_rows.iloc[fit_stop:eval_stop].copy().reset_index(drop=True)
        if fit_rows.empty:
            raise ValueError("Empty fit split after partitioning the shared source split")
        if eval_rows.empty:
            raise ValueError("Empty eval split; reduce --fit-limit or --eval-limit")
        return fit_rows, eval_rows, True, {
            "fit_selection_strategy": "deterministic_shuffle_then_disjoint_slice",
            "eval_selection_strategy": "deterministic_shuffle_then_disjoint_slice",
            "eval_top_up_from_fit_split": False,
            "eval_top_up_count": 0,
        }

    if not eval_top_up_from_fit_split:
        fit_rows = load_csqa(split=fit_split, limit=fit_limit).copy().reset_index(drop=True)
        eval_rows = load_csqa(split=eval_split, limit=eval_limit).copy().reset_index(drop=True)
        return fit_rows, eval_rows, False, {
            "fit_selection_strategy": "direct_split_load",
            "eval_selection_strategy": "direct_split_load",
            "eval_top_up_from_fit_split": False,
            "eval_top_up_count": 0,
        }

    fit_source = load_csqa(split=fit_split, limit=None).copy()
    fit_source = fit_source.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    fit_stop = len(fit_source) if fit_limit is None else min(fit_limit, len(fit_source))
    fit_rows = fit_source.iloc[:fit_stop].copy().reset_index(drop=True)
    unused_fit_rows = fit_source.iloc[fit_stop:].copy().reset_index(drop=True)
    eval_rows = load_csqa(split=eval_split, limit=eval_limit).copy().reset_index(drop=True)

    top_up_count = 0
    if eval_limit is not None and len(eval_rows) < eval_limit:
        top_up = unused_fit_rows.iloc[: eval_limit - len(eval_rows)].copy().reset_index(drop=True)
        top_up_count = len(top_up)
        eval_rows = pd.concat([eval_rows, top_up], ignore_index=True)
    if fit_rows.empty or eval_rows.empty:
        raise ValueError("Empty fit or eval split after composing experiment rows")
    return fit_rows, eval_rows, False, {
        "fit_selection_strategy": "deterministic_shuffle_then_prefix",
        "eval_selection_strategy": "base_eval_plus_unused_fit_topup",
        "eval_top_up_from_fit_split": True,
        "eval_top_up_count": top_up_count,
    }


def parse_float_list(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values or any(value <= 0 for value in values):
        raise ValueError("All values must be positive")
    return values


def parse_positive_scale_list(raw: str) -> list[float]:
    values = parse_float_list(raw)
    if any(value > 1.0 + 1e-12 for value in values):
        raise ValueError("Backtrack scales must be in (0, 1]")
    return values


def parse_feature_names(raw: str) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [value for value in values if value not in DEFAULT_FEATURE_NAMES]
    if not values:
        raise ValueError("No feature names provided")
    if unknown:
        raise ValueError(f"Unknown feature names: {unknown}. Allowed: {DEFAULT_FEATURE_NAMES}")
    return values


def intervention_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the CSQA logit-feature score experiment.")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--max-seq-len", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fit-split", default="train")
    parser.add_argument("--eval-split", default="train")
    parser.add_argument("--fit-limit", type=int, default=None)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--eval-top-up-from-fit-split", action="store_true")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--validation-limit", type=int, default=None)
    parser.add_argument("--top-k-layers-per-feature", type=int, default=3)
    parser.add_argument("--feature-names", default=",".join(DEFAULT_FEATURE_NAMES))
    parser.add_argument("--max-delta-over-hidden", type=float, default=DEFAULT_MAX_DELTA_OVER_HIDDEN)
    parser.add_argument(
        "--backtrack-scales",
        default=",".join(str(value) for value in DEFAULT_BACKTRACK_SCALES),
    )
    parser.add_argument("--good-threshold-log-ratio", type=float, default=GOOD_REGION_LOG_RATIO_THRESHOLD)
    return parser


def diagnostics_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run all-layer CSQA feature diagnostics without interventions."
    )
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--max-seq-len", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fit-split", default="train")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--fit-limit", type=int, default=None)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--validation-limit", type=int, default=None)
    parser.add_argument("--feature-names", default=",".join(DEFAULT_FEATURE_NAMES))
    parser.add_argument("--good-threshold-log-ratio", type=float, default=GOOD_REGION_LOG_RATIO_THRESHOLD)
    parser.add_argument("--grid-points", type=int, default=GRID_POINTS)
    return parser


def intervention_limits(args: argparse.Namespace) -> tuple[int | None, int | None]:
    return (
        resolve_split_limit(
            split_name=args.fit_split,
            explicit_limit=args.fit_limit,
            train_limit=args.train_limit,
            validation_limit=args.validation_limit,
            default_train_limit=DEFAULT_TRAIN_LIMIT,
        ),
        resolve_split_limit(
            split_name=args.eval_split,
            explicit_limit=args.eval_limit,
            train_limit=args.train_limit,
            validation_limit=args.validation_limit,
            default_train_limit=DEFAULT_TRAIN_LIMIT,
        ),
    )


def diagnostics_limits(args: argparse.Namespace) -> tuple[int | None, int | None]:
    return (
        resolve_split_limit(
            split_name=args.fit_split,
            explicit_limit=args.fit_limit,
            train_limit=args.train_limit,
            validation_limit=args.validation_limit,
            default_train_limit=None,
        ),
        resolve_split_limit(
            split_name=args.eval_split,
            explicit_limit=args.eval_limit,
            train_limit=args.train_limit,
            validation_limit=args.validation_limit,
            default_train_limit=None,
        ),
    )
