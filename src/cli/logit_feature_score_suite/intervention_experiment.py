"""Orkiestracja końcowego eksperymentu; bez definicji matematycznych."""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from src.cli.logit_feature_score_suite.control_policy import run_score_control_policy
from src.cli.logit_feature_score_suite.experiment_core import (
    extract_model_state,
    fit_score_models,
    save_base_tables,
)
from src.cli.logit_feature_score_suite.experiment_setup import (
    intervention_limits,
    intervention_parser,
    load_fit_and_eval_rows,
    parse_feature_names,
    parse_positive_scale_list,
    resolve_out_dir,
    split_artifact_tag,
)
from src.cli.logit_feature_score_suite.score_policy import run_score_policy
from src.score.constants import (
    CONTROL_INTERVENTION_TYPES,
    GRID_POINTS,
    KDE_BANDWIDTH_MULTIPLIER,
    LOG_RATIO_SMOOTHING_SIGMA_BANDWIDTHS,
    SUPPORT_LOWER_QUANTILE,
    SUPPORT_UPPER_QUANTILE,
)


def _split_tags(
    args: argparse.Namespace,
    same_source_split: bool,
    split_plan: dict[str, object],
) -> tuple[str, str]:
    fit_tag = split_artifact_tag(args.fit_split, "fit", same_source_split)
    eval_tag = (
        f"{args.eval_split}_plus_{args.fit_split}_topup"
        if split_plan["eval_top_up_from_fit_split"]
        else split_artifact_tag(args.eval_split, "eval", same_source_split)
    )
    return fit_tag, eval_tag


def _config(
    *,
    args: argparse.Namespace,
    controls: bool,
    feature_names: list[str],
    backtrack_scales: list[float],
    fit_limit: int | None,
    eval_limit: int | None,
    fit_rows,
    eval_rows,
    fit_tag: str,
    eval_tag: str,
    same_source_split: bool,
    split_plan: dict[str, object],
    candidate_layers: list[int] | None = None,
) -> dict[str, object]:
    config = {
        "dataset": "csqa",
        "model_id": args.model_id,
        "seed": int(args.seed),
        "max_seq_len": int(args.max_seq_len),
        "fit_split": args.fit_split,
        "eval_split": args.eval_split,
        "fit_split_tag": fit_tag,
        "eval_split_tag": eval_tag,
        "fit_limit": fit_limit,
        "eval_limit": eval_limit,
        "same_source_split_partition": same_source_split,
        "fit_selection_strategy": split_plan["fit_selection_strategy"],
        "eval_selection_strategy": split_plan["eval_selection_strategy"],
        "eval_top_up_from_fit_split": bool(split_plan["eval_top_up_from_fit_split"]),
        "eval_top_up_count": int(split_plan["eval_top_up_count"]),
        "fit_n_examples": len(fit_rows),
        "eval_n_examples": len(eval_rows),
        "method": "logit_feature_score_control" if controls else "logit_feature_score_ascent",
        "feature_names": feature_names,
        "layer_selection_rule": "top_k_by_ks_statistic",
        "top_k_layers_per_feature": int(args.top_k_layers_per_feature),
        "max_delta_over_hidden": float(args.max_delta_over_hidden),
        "backtrack_scales": backtrack_scales,
        "good_threshold_log_ratio": float(args.good_threshold_log_ratio),
        "support_lower_quantile": SUPPORT_LOWER_QUANTILE,
        "support_upper_quantile": SUPPORT_UPPER_QUANTILE,
        "kde_bandwidth_multiplier": KDE_BANDWIDTH_MULTIPLIER,
        "log_ratio_smoothing_sigma_bandwidths": LOG_RATIO_SMOOTHING_SIGMA_BANDWIDTHS,
    }
    if same_source_split:
        config["shared_split_partition_strategy"] = "deterministic_shuffle_then_disjoint_slice"
    else:
        config["shared_split_partition_strategy"] = None
    if controls:
        config["intervention_types"] = list(CONTROL_INTERVENTION_TYPES)
    if candidate_layers is not None:
        config["candidate_layer_numbers"] = candidate_layers
    return config


def run_intervention_experiment(
    argv: list[str] | None = None,
    *,
    controls: bool,
) -> None:
    """Uruchom wspólny pipeline w wariancie ascent lub z kontrolami."""
    args = intervention_parser().parse_args(argv)
    if args.max_delta_over_hidden <= 0:
        raise ValueError("--max-delta-over-hidden must be positive")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    feature_names = parse_feature_names(args.feature_names)
    backtrack_scales = parse_positive_scale_list(args.backtrack_scales)
    fit_limit, eval_limit = intervention_limits(args)
    fit_rows, eval_rows, same_source_split, split_plan = load_fit_and_eval_rows(
        fit_split=args.fit_split,
        eval_split=args.eval_split,
        fit_limit=fit_limit,
        eval_limit=eval_limit,
        seed=args.seed,
        eval_top_up_from_fit_split=args.eval_top_up_from_fit_split,
    )
    fit_tag, eval_tag = _split_tags(args, same_source_split, split_plan)

    preview = _config(
        args=args,
        controls=controls,
        feature_names=feature_names,
        backtrack_scales=backtrack_scales,
        fit_limit=fit_limit,
        eval_limit=eval_limit,
        fit_rows=fit_rows,
        eval_rows=eval_rows,
        fit_tag=fit_tag,
        eval_tag=eval_tag,
        same_source_split=same_source_split,
        split_plan=split_plan,
    )
    print("[config]", json.dumps(preview, indent=2))

    state = extract_model_state(
        model_id=args.model_id,
        fit_rows=fit_rows,
        eval_rows=eval_rows,
        fit_split_tag=fit_tag,
        eval_split_tag=eval_tag,
        feature_names=feature_names,
        max_seq_len=args.max_seq_len,
    )
    score_fit = fit_score_models(
        state,
        feature_names=feature_names,
        top_k_layers=args.top_k_layers_per_feature,
        log_ratio_threshold=args.good_threshold_log_ratio,
        grid_points=GRID_POINTS,
    )

    policy = run_score_control_policy if controls else run_score_policy
    policy_outputs = policy(
        eval_rows=eval_rows,
        eval_cache=state["eval_cache"],
        eval_feature_df=state["eval_features"],
        selected_layers_df=score_fit["selected_layers"],
        region_models=score_fit["region_models"],
        backtrack_scales=backtrack_scales,
        max_delta_over_hidden=args.max_delta_over_hidden,
        tok=state["tokenizer"],
        model=state["model"],
        input_device=state["input_device"],
        decoder_layers=state["decoder_layers"],
        answer_id_tensor_cpu=state["readout"]["answer_id_tensor_cpu"],
        answer_id_tensor_lm_head=state["readout"]["answer_id_tensor_lm_head"],
        lm_head_weight=state["readout"]["lm_head_weight"],
        maybe_apply_final_norm=state["readout"]["maybe_apply_final_norm"],
        vocab_size=state["vocab_size"],
        max_seq_len=args.max_seq_len,
        eval_split_name=eval_tag,
    )

    pipeline_name = (
        "csqa_logit_feature_score_control_pipeline"
        if controls
        else "csqa_logit_feature_score_pipeline"
    )
    output_group = (
        "logit_feature_score_control_pipeline"
        if controls
        else "logit_feature_score_pipeline"
    )
    out_dir = resolve_out_dir(
        args.out_dir,
        args.model_id,
        pipeline_name,
        output_group=output_group,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    save_base_tables(
        out_dir=out_dir,
        state=state,
        score_fit=score_fit,
        fit_split_tag=fit_tag,
        eval_split_tag=eval_tag,
    )
    policy_file = (
        f"{eval_tag}_score_control_policy_outputs_raw.parquet"
        if controls
        else f"{eval_tag}_score_ascent_policy_outputs_raw.parquet"
    )
    policy_outputs.to_parquet(out_dir / policy_file, index=False)

    run_config = _config(
        args=args,
        controls=controls,
        feature_names=feature_names,
        backtrack_scales=backtrack_scales,
        fit_limit=fit_limit,
        eval_limit=eval_limit,
        fit_rows=fit_rows,
        eval_rows=eval_rows,
        fit_tag=fit_tag,
        eval_tag=eval_tag,
        same_source_split=same_source_split,
        split_plan=split_plan,
        candidate_layers=state["layer_numbers"],
    )
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    print(f"[done] Wrote outputs to {out_dir}")
