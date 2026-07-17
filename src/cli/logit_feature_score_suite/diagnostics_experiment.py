"""Orkiestracja diagnostyki wszystkich warstw bez interwencji."""

from __future__ import annotations

import gc
import json

import numpy as np
import pandas as pd
import torch

from src.cli.logit_feature_score_suite.experiment_core import (
    extract_model_state,
    fit_score_models,
)
from src.cli.logit_feature_score_suite.experiment_setup import (
    diagnostics_limits,
    diagnostics_parser,
    parse_feature_names,
    resolve_out_dir,
)
from src.cli.logit_feature_score_suite.model_runtime import build_layerwise_choice_readout_table
from src.data.load_csqa import load_csqa
from src.score.constants import (
    EXTRACT_BATCH_SIZE,
    KDE_BANDWIDTH_MULTIPLIER,
    LOG_RATIO_SMOOTHING_SIGMA_BANDWIDTHS,
    READOUT_BATCH_SIZE,
    SUPPORT_LOWER_QUANTILE,
    SUPPORT_UPPER_QUANTILE,
)


def run_diagnostics_experiment(argv: list[str] | None = None) -> None:
    args = diagnostics_parser().parse_args(argv)
    if args.fit_split == args.eval_split:
        raise ValueError("--fit-split and --eval-split must be different")
    if args.grid_points <= 8:
        raise ValueError("--grid-points must be greater than 8")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    fit_limit, eval_limit = diagnostics_limits(args)
    feature_names = parse_feature_names(args.feature_names)
    fit_rows = load_csqa(split=args.fit_split, limit=fit_limit).copy()
    eval_rows = load_csqa(split=args.eval_split, limit=eval_limit).copy()

    preview = {
        "dataset": "csqa",
        "model_id": args.model_id,
        "fit_split": args.fit_split,
        "eval_split": args.eval_split,
        "fit_limit": fit_limit,
        "eval_limit": eval_limit,
        "max_seq_len": args.max_seq_len,
        "seed": args.seed,
        "feature_names": feature_names,
        "distribution_layers": "all",
        "good_threshold_log_ratio": float(args.good_threshold_log_ratio),
        "grid_points": int(args.grid_points),
        "support_lower_quantile": SUPPORT_LOWER_QUANTILE,
        "support_upper_quantile": SUPPORT_UPPER_QUANTILE,
        "kde_bandwidth_multiplier": KDE_BANDWIDTH_MULTIPLIER,
        "log_ratio_smoothing_sigma_bandwidths": LOG_RATIO_SMOOTHING_SIGMA_BANDWIDTHS,
    }
    print("[config]", json.dumps(preview, indent=2))

    state = extract_model_state(
        model_id=args.model_id,
        fit_rows=fit_rows,
        eval_rows=eval_rows,
        fit_split_tag=args.fit_split,
        eval_split_tag=args.eval_split,
        feature_names=feature_names,
        max_seq_len=args.max_seq_len,
    )
    score_fit = fit_score_models(
        state,
        feature_names=feature_names,
        top_k_layers=len(state["layer_numbers"]),
        log_ratio_threshold=args.good_threshold_log_ratio,
        grid_points=args.grid_points,
    )

    readout_kwargs = {
        "maybe_apply_final_norm": state["readout"]["maybe_apply_final_norm"],
        "lm_head_weight": state["readout"]["lm_head_weight"],
        "answer_id_tensor_lm_head": state["readout"]["answer_id_tensor_lm_head"],
        "input_device": state["input_device"],
    }
    fit_readouts = build_layerwise_choice_readout_table(
        cache=state["fit_cache"],
        split_name=args.fit_split,
        **readout_kwargs,
    )
    eval_readouts = build_layerwise_choice_readout_table(
        cache=state["eval_cache"],
        split_name=args.eval_split,
        **readout_kwargs,
    )

    out_dir = resolve_out_dir(
        args.out_dir,
        args.model_id,
        "csqa_logit_feature_diagnostics_pipeline",
        output_group="logit_feature_diagnostics_pipeline",
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(state["fit_cache"]["example_rows"]).to_parquet(
        out_dir / f"{args.fit_split}_examples.parquet",
        index=False,
    )
    pd.DataFrame(state["eval_cache"]["example_rows"]).to_parquet(
        out_dir / f"{args.eval_split}_examples.parquet",
        index=False,
    )
    pd.DataFrame(state["fit_cache"]["clean_output_rows"]).to_parquet(
        out_dir / f"{args.fit_split}_clean_final_outputs.parquet",
        index=False,
    )
    pd.DataFrame(state["eval_cache"]["clean_output_rows"]).to_parquet(
        out_dir / f"{args.eval_split}_clean_final_outputs.parquet",
        index=False,
    )
    fit_readouts.to_parquet(
        out_dir / f"{args.fit_split}_layerwise_choice_readouts.parquet",
        index=False,
    )
    eval_readouts.to_parquet(
        out_dir / f"{args.eval_split}_layerwise_choice_readouts.parquet",
        index=False,
    )
    score_fit["separation"].to_parquet(
        out_dir / "feature_layer_separation_summary.parquet",
        index=False,
    )
    score_fit["distribution_grid"].to_parquet(
        out_dir / "fit_distribution_grid.parquet",
        index=False,
    )

    saved_tables = [
        f"{args.fit_split}_examples.parquet",
        f"{args.eval_split}_examples.parquet",
        f"{args.fit_split}_clean_final_outputs.parquet",
        f"{args.eval_split}_clean_final_outputs.parquet",
        f"{args.fit_split}_layerwise_choice_readouts.parquet",
        f"{args.eval_split}_layerwise_choice_readouts.parquet",
        "feature_layer_separation_summary.parquet",
        "fit_distribution_grid.parquet",
    ]
    run_config = {
        **preview,
        "method": "logit_feature_diagnostics",
        "candidate_layer_numbers": state["layer_numbers"],
        "distribution_build_rule": "all_layers_from_feature_layer_separation_summary",
        "distribution_grid_points": int(args.grid_points),
        "extract_batch_size": EXTRACT_BATCH_SIZE,
        "readout_batch_size": READOUT_BATCH_SIZE,
        "last_layer_needs_final_norm": bool(state["readout"]["last_layer_needs_final_norm"]),
        "saved_tables": saved_tables,
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    print(f"[done] Wrote outputs to {out_dir}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
