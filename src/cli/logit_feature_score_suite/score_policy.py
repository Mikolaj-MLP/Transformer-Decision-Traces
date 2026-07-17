"""Pętla wykonawcza wariantu score-ascent."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.cli.logit_feature_score_suite.model_runtime import run_single_intervention_forward
from src.cli.logit_feature_score_suite.policy_common import (
    build_output_row,
    prepare_batch,
    release_batch_memory,
    true_choice_tensor,
)
from src.score.constants import INTERVENTION_BATCH_SIZE
from src.score.density import interpolate_score_state
from src.score.intervention import (
    build_full_cap_delta,
    compute_score_ascent_unit_delta,
    select_score_ascent_delta,
)


def run_score_policy(
    *,
    eval_rows: pd.DataFrame,
    eval_cache: dict[str, object],
    eval_feature_df: pd.DataFrame,
    selected_layers_df: pd.DataFrame,
    region_models: dict[tuple[str, int], dict[str, object]],
    backtrack_scales: list[float],
    max_delta_over_hidden: float,
    tok: AutoTokenizer,
    model: AutoModelForCausalLM,
    input_device: torch.device,
    decoder_layers,
    answer_id_tensor_cpu: torch.Tensor,
    answer_id_tensor_lm_head: torch.Tensor,
    lm_head_weight: torch.Tensor,
    maybe_apply_final_norm,
    vocab_size: int,
    max_seq_len: int,
    eval_split_name: str,
) -> pd.DataFrame:
    """Wykonaj ascent dla każdej wybranej pary cecha-warstwa."""
    lookup = eval_feature_df.set_index(["example_id", "feature_name", "layer_number"])
    example_ids = [row["example_id"] for row in eval_cache["example_rows"]]
    id_to_index = {example_id: idx for idx, example_id in enumerate(example_ids)}
    selected_pairs = [
        (str(row.feature_name), int(row.layer_number))
        for row in selected_layers_df.itertuples(index=False)
    ]
    total_steps = len(selected_pairs) * math.ceil(len(eval_rows) / INTERVENTION_BATCH_SIZE)
    rows: list[dict[str, object]] = []

    with tqdm(total=total_steps, desc=f"{eval_split_name} score-ascent sweep") as progress:
        for feature_name, layer_number in selected_pairs:
            region_model = region_models[(feature_name, layer_number)]
            current_feature_all = np.asarray(
                [lookup.loc[(example_id, feature_name, layer_number), "feature_value"] for example_id in example_ids],
                dtype=np.float32,
            )
            current_state_all = interpolate_score_state(current_feature_all, region_model=region_model)

            for start in range(0, len(eval_rows), INTERVENTION_BATCH_SIZE):
                batch_df = eval_rows.iloc[start : start + INTERVENTION_BATCH_SIZE].reset_index(drop=True)
                batch_indices = [id_to_index[value] for value in batch_df["example_id"].tolist()]
                current_score = current_state_all["score_value"][batch_indices].astype(np.float32)
                current_derivative = current_state_all["score_derivative"][batch_indices].astype(np.float32)
                current_supported = current_state_all["supported"][batch_indices].astype(bool)
                current_region = current_state_all["region_label"][batch_indices]
                eligible = current_supported & np.isfinite(current_derivative) & (np.abs(current_derivative) > 1e-8)

                batch, decision_pos = prepare_batch(
                    tok,
                    batch_df["text"].tolist(),
                    max_seq_len,
                    input_device,
                )
                true_choice_idx = true_choice_tensor(batch_df, input_device)
                token_hidden = eval_cache["hidden"][batch_indices, layer_number - 1, :].to(input_device)
                ascent_basis = compute_score_ascent_unit_delta(
                    token_hidden,
                    feature_name=feature_name,
                    layer_index_0based=layer_number - 1,
                    maybe_apply_final_norm=maybe_apply_final_norm,
                    lm_head_weight=lm_head_weight,
                    answer_id_tensor_lm_head=answer_id_tensor_lm_head.to(input_device),
                    vocab_size=vocab_size,
                    current_score_derivative=current_derivative,
                    intervention_mask=eligible,
                )
                effective_eligible = eligible & (ascent_basis["score_grad_l2_norm"] > 0.0)
                raw_delta = build_full_cap_delta(
                    token_hidden,
                    unit_delta=ascent_basis["unit_delta"],
                    intervention_mask=effective_eligible,
                    max_delta_over_hidden=max_delta_over_hidden,
                )
                selected = select_score_ascent_delta(
                    token_hidden,
                    delta_raw=raw_delta["delta_raw"],
                    current_score_value=current_score,
                    intervention_mask=effective_eligible,
                    feature_name=feature_name,
                    layer_index_0based=layer_number - 1,
                    maybe_apply_final_norm=maybe_apply_final_norm,
                    lm_head_weight=lm_head_weight,
                    answer_id_tensor_lm_head=answer_id_tensor_lm_head.to(input_device),
                    vocab_size=vocab_size,
                    region_model=region_model,
                    backtrack_scales=backtrack_scales,
                )
                outputs = run_single_intervention_forward(
                    batch=batch,
                    decision_pos=decision_pos,
                    model=model,
                    steering_module=decoder_layers[layer_number - 1],
                    delta=selected["delta"],
                    true_choice_idx=true_choice_idx,
                    answer_id_tensor_cpu=answer_id_tensor_cpu,
                )
                branch_stats = {
                    "steered_feature_value_local": selected["steered_feature_value_local"],
                    "steered_score_value_local": selected["steered_score_value_local"],
                    "steered_supported": selected["steered_supported"],
                    "steered_region_label": selected["steered_region_label"],
                    "delta_l2_norm": selected["delta_l2_norm"],
                    "delta_over_token_hidden_l2": selected["delta_over_token_hidden_l2"],
                    "token_hidden_l2_norm": selected["token_hidden_l2_norm"],
                }
                for batch_index, row in batch_df.iterrows():
                    global_index = batch_indices[batch_index]
                    rows.append(
                        build_output_row(
                            example_id=row["example_id"],
                            feature_name=feature_name,
                            layer_number=layer_number,
                            max_delta_over_hidden=max_delta_over_hidden,
                            eligible=effective_eligible[batch_index],
                            accepted=selected["accepted_intervention"][batch_index],
                            batch_index=batch_index,
                            global_index=global_index,
                            current_supported=current_supported,
                            current_region=current_region,
                            current_score=current_score,
                            current_score_derivative=current_derivative,
                            ascent_basis=ascent_basis,
                            raw_delta_stats=raw_delta,
                            branch_stats=branch_stats,
                            accepted_step_scale=selected["accepted_step_scale"],
                            outputs=outputs,
                            clean_choice_logits=eval_cache["clean_choice_logits"],
                            clean_best_non_choice_logit=eval_cache["clean_best_non_choice_logit"],
                            clean_best_non_choice_token_id=eval_cache["clean_best_non_choice_token_id"],
                        )
                    )
                progress.update(1)
                release_batch_memory()
    return pd.DataFrame(rows)
