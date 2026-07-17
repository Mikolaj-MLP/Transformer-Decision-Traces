"""Wspólne, liniowe kroki techniczne pipeline'ów score."""

from __future__ import annotations

import pandas as pd

from src.cli.logit_feature_score_suite.experiment_setup import all_layer_numbers
from src.cli.logit_feature_score_suite.model_runtime import (
    build_feature_table,
    extract_split_cache,
    get_input_device,
    load_model_and_tokenizer,
    prepare_readout_context,
)
from src.csqa.common import get_decoder_layers
from src.score.constants import EXTRACT_BATCH_SIZE
from src.score.density import add_score_derivative, build_distribution_models
from src.score.statistics import build_separation_summary, select_top_k_layers_by_feature


def extract_model_state(
    *,
    model_id: str,
    fit_rows: pd.DataFrame,
    eval_rows: pd.DataFrame,
    fit_split_tag: str,
    eval_split_tag: str,
    feature_names: list[str],
    max_seq_len: int,
) -> dict[str, object]:
    """Załaduj model, wykonaj bazowe forwardy i oblicz cechy wszystkich warstw."""
    for frame in (fit_rows, eval_rows):
        frame["prompt_len_chars"] = frame["text"].str.len()

    model, tokenizer = load_model_and_tokenizer(model_id)
    input_device = get_input_device(model)
    decoder_layers = get_decoder_layers(model)
    num_layers = len(decoder_layers)
    layer_numbers = all_layer_numbers(num_layers)
    vocab_size = int(model.config.vocab_size)

    readout = prepare_readout_context(
        model=model,
        tok=tokenizer,
        input_device=input_device,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        probe_rows=eval_rows,
    )
    cache_kwargs = {
        "tok": tokenizer,
        "model": model,
        "input_device": input_device,
        "answer_id_tensor_cpu": readout["answer_id_tensor_cpu"],
        "answer_id_tensor_lm_head": readout["answer_id_tensor_lm_head"],
        "answer_choice_weight": readout["answer_choice_weight"],
        "final_norm": readout["final_norm"],
        "num_layers": num_layers,
        "max_seq_len": max_seq_len,
        "batch_size": EXTRACT_BATCH_SIZE,
        "last_layer_needs_final_norm": readout["last_layer_needs_final_norm"],
    }
    fit_cache = extract_split_cache(fit_rows, split_name=fit_split_tag, **cache_kwargs)
    eval_cache = extract_split_cache(eval_rows, split_name=eval_split_tag, **cache_kwargs)

    feature_kwargs = {
        "feature_names": feature_names,
        "maybe_apply_final_norm": readout["maybe_apply_final_norm"],
        "lm_head_weight": readout["lm_head_weight"],
        "answer_id_tensor_lm_head": readout["answer_id_tensor_lm_head"],
        "input_device": input_device,
        "vocab_size": vocab_size,
        "active_layer_numbers": layer_numbers,
    }
    fit_features = build_feature_table(
        split_name=fit_split_tag,
        cache=fit_cache,
        **feature_kwargs,
    )
    eval_features = build_feature_table(
        split_name=eval_split_tag,
        cache=eval_cache,
        **feature_kwargs,
    )
    return {
        "model": model,
        "tokenizer": tokenizer,
        "input_device": input_device,
        "decoder_layers": decoder_layers,
        "layer_numbers": layer_numbers,
        "vocab_size": vocab_size,
        "readout": readout,
        "fit_cache": fit_cache,
        "eval_cache": eval_cache,
        "fit_features": fit_features,
        "eval_features": eval_features,
    }


def fit_score_models(
    state: dict[str, object],
    *,
    feature_names: list[str],
    top_k_layers: int,
    log_ratio_threshold: float,
    grid_points: int,
) -> dict[str, object]:
    """Wybierz warstwy oraz dopasuj gęstości i funkcję score."""
    separation = build_separation_summary(state["fit_features"])
    selected_layers = select_top_k_layers_by_feature(
        separation,
        feature_names=feature_names,
        top_k=top_k_layers,
    )
    region_models, distribution_grid = build_distribution_models(
        fit_feature_df=state["fit_features"],
        selected_layers_df=selected_layers,
        log_ratio_threshold=log_ratio_threshold,
        grid_points=grid_points,
    )
    distribution_grid = add_score_derivative(region_models, distribution_grid)
    return {
        "separation": separation,
        "selected_layers": selected_layers,
        "region_models": region_models,
        "distribution_grid": distribution_grid,
    }


def save_base_tables(
    *,
    out_dir,
    state: dict[str, object],
    score_fit: dict[str, object],
    fit_split_tag: str,
    eval_split_tag: str,
) -> None:
    """Zapisz artefakty wspólne dla eksperymentów ascent i control."""
    pd.DataFrame(state["fit_cache"]["example_rows"]).to_parquet(
        out_dir / f"{fit_split_tag}_examples.parquet",
        index=False,
    )
    pd.DataFrame(state["eval_cache"]["example_rows"]).to_parquet(
        out_dir / f"{eval_split_tag}_examples.parquet",
        index=False,
    )
    pd.DataFrame(state["fit_cache"]["clean_output_rows"]).to_parquet(
        out_dir / f"{fit_split_tag}_clean_final_outputs.parquet",
        index=False,
    )
    pd.DataFrame(state["eval_cache"]["clean_output_rows"]).to_parquet(
        out_dir / f"{eval_split_tag}_clean_final_outputs.parquet",
        index=False,
    )
    state["fit_features"].to_parquet(
        out_dir / f"{fit_split_tag}_univariate_feature_values.parquet",
        index=False,
    )
    state["eval_features"].to_parquet(
        out_dir / f"{eval_split_tag}_univariate_feature_values.parquet",
        index=False,
    )
    score_fit["separation"].to_parquet(
        out_dir / "feature_layer_separation_summary.parquet",
        index=False,
    )
    score_fit["selected_layers"].to_parquet(
        out_dir / "selected_layers_by_feature.parquet",
        index=False,
    )
    score_fit["distribution_grid"].to_parquet(
        out_dir / "fit_distribution_grid.parquet",
        index=False,
    )
