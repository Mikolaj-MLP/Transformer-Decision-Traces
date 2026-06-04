from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings(
    "ignore",
    message=r"The default value for l1_ratios will change from None to \(0\.0,\) in version 1\.10\..*",
    category=FutureWarning,
    module=r"sklearn\.linear_model\._logistic",
)
warnings.filterwarnings(
    "ignore",
    message=r"'penalty' was deprecated in version 1\.8 and will be removed in 1\.10\..*",
    category=FutureWarning,
    module=r"sklearn\.linear_model\._logistic",
)
warnings.filterwarnings(
    "ignore",
    message=r"The fitted attributes of LogisticRegressionCV will be simplified in scikit-learn 1\.10 to remove redundancy\..*",
    category=FutureWarning,
    module=r"sklearn\.linear_model\._logistic",
)

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cli.run_csqa_adaptive_contrastive_pipeline import (  # noqa: E402
    EXTRACT_BATCH_SIZE,
    READOUT_BATCH_SIZE,
    choice_logits_to_entropy,
    choice_logits_to_gap,
    choice_logits_to_pred_idx,
    choose_model_dtype_and_device_map,
    extract_split_cache,
    get_input_device,
    prepare_readout_context,
)
from src.csqa.common import (  # noqa: E402
    encode_prompts,
    get_decoder_layers,
    repack_output_hidden,
    select_full_logits_at_decision,
    summarize_decision_logits,
    unpack_output_hidden,
)
from src.data.load_mcqa import SUPPORTED_DATASETS, load_mcqa  # noqa: E402
from src.data.load_aqua_rat import (  # noqa: E402
    DEFAULT_AQUA_SPLIT_SEED,
    DEFAULT_AQUA_TEST_SIZE,
    DEFAULT_AQUA_TRAIN_SIZE,
    DEFAULT_AQUA_VALIDATION_SIZE,
)


LETTERS = ["A", "B", "C", "D", "E"]
FEATURE_NAMES = [
    "answer_choice_entropy_normalized",
    "answer_choice_top1_top2_logit_gap",
    "answer_choice_top1_probability",
    "answer_choice_varentropy",
    "full_vocab_entropy_normalized",
]
INTERVENTION_BATCH_SIZE = 2
DETECTOR_C_GRID = np.logspace(-3, 2, 12)
GRAD_NORM_EPS = 1e-12
TARGET_LOWER_QUANTILE = 0.25
TARGET_UPPER_QUANTILE = 0.75


def now_id() -> str:
    import time

    return time.strftime("%Y%m%d-%H%M%S")


def repo_root() -> Path:
    return REPO_ROOT


def slugify_model_id(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", model_id).strip("-")


def resolve_out_dir(out_dir: str | None, model_id: str, dataset_name: str) -> Path:
    root = repo_root()
    if out_dir is None:
        run_name = f"{now_id()}_{slugify_model_id(model_id)}_{dataset_name}_logit_feature_steering_pipeline"
        return root / "data" / "generated" / "logit_feature_steering_pipeline" / run_name
    path = Path(out_dir)
    return path if path.is_absolute() else (root / path)


def maybe_clone_to_float_cpu(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy().astype(np.float32)


def resolve_hf_token() -> str | None:
    for key in ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"]:
        value = os.getenv(key)
        if value:
            return value
    return None


def choice_logits_to_top1_prob(choice_logits: np.ndarray) -> np.ndarray:
    probs = torch.softmax(torch.from_numpy(choice_logits), dim=1).numpy()
    return probs.max(axis=1)


def choice_logits_to_varentropy(choice_logits: np.ndarray) -> np.ndarray:
    probs = torch.softmax(torch.from_numpy(choice_logits), dim=1).numpy()
    log_probs = np.log(np.clip(probs, 1e-12, None))
    surprisal = -log_probs
    entropy = (probs * surprisal).sum(axis=1, keepdims=True)
    return (probs * (surprisal - entropy) ** 2).sum(axis=1)


def compute_feature_tensor(
    *,
    feature_name: str,
    full_logits: torch.Tensor,
    choice_logits: torch.Tensor,
    vocab_size: int,
) -> torch.Tensor:
    if feature_name == "answer_choice_entropy_normalized":
        choice_log_probs = F.log_softmax(choice_logits.float(), dim=-1)
        choice_probs = torch.exp(choice_log_probs)
        return -(choice_probs * choice_log_probs).sum(dim=-1) / math.log(len(LETTERS))

    if feature_name == "answer_choice_top1_top2_logit_gap":
        sorted_choice_logits = torch.sort(choice_logits.float(), dim=-1, descending=True).values
        return sorted_choice_logits[:, 0] - sorted_choice_logits[:, 1]

    if feature_name == "answer_choice_top1_probability":
        choice_probs = F.softmax(choice_logits.float(), dim=-1)
        return torch.max(choice_probs, dim=-1).values

    if feature_name == "answer_choice_varentropy":
        choice_log_probs = F.log_softmax(choice_logits.float(), dim=-1)
        choice_probs = torch.exp(choice_log_probs)
        surprisal = -choice_log_probs
        entropy = (choice_probs * surprisal).sum(dim=-1, keepdim=True)
        return (choice_probs * (surprisal - entropy) ** 2).sum(dim=-1)

    if feature_name == "full_vocab_entropy_normalized":
        full_log_probs = F.log_softmax(full_logits.float(), dim=-1)
        full_probs = torch.exp(full_log_probs)
        return -(full_probs * full_log_probs).sum(dim=-1) / math.log(vocab_size)

    raise ValueError(f"Unknown feature_name: {feature_name}")


def summarize_logit_features(
    *,
    full_logits: torch.Tensor,
    choice_logits: torch.Tensor,
    vocab_size: int,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for feature_name in FEATURE_NAMES:
        out[feature_name] = maybe_clone_to_float_cpu(
            compute_feature_tensor(
                feature_name=feature_name,
                full_logits=full_logits,
                choice_logits=choice_logits,
                vocab_size=vocab_size,
            )
        )
    return out


def build_feature_table(
    *,
    split_name: str,
    cache: dict[str, object],
    maybe_apply_final_norm,
    lm_head_weight: torch.Tensor,
    answer_id_tensor_lm_head: torch.Tensor,
    input_device: torch.device,
    vocab_size: int,
) -> pd.DataFrame:
    hidden = cache["hidden"]
    clean_is_correct = cache["clean_is_correct"]
    example_rows = pd.DataFrame(cache["example_rows"])[["example_id", "split"]]
    rows: list[dict[str, object]] = []

    for layer_index in tqdm(range(hidden.shape[1]), desc=f"{split_name} feature extraction"):
        layer_hidden = hidden[:, layer_index, :]
        feature_blocks: dict[str, list[np.ndarray]] = {feature_name: [] for feature_name in FEATURE_NAMES}

        for start in range(0, layer_hidden.shape[0], READOUT_BATCH_SIZE):
            end = start + READOUT_BATCH_SIZE
            hidden_batch = layer_hidden[start:end].to(input_device)
            readout = maybe_apply_final_norm(hidden_batch.float(), layer_index)
            full_logits = torch.matmul(
                readout.to(lm_head_weight.dtype),
                lm_head_weight.T,
            ).float()
            choice_logits = full_logits.index_select(1, answer_id_tensor_lm_head)
            feature_batch = summarize_logit_features(
                full_logits=full_logits,
                choice_logits=choice_logits,
                vocab_size=vocab_size,
            )
            for feature_name, values in feature_batch.items():
                feature_blocks[feature_name].append(values)

            del hidden_batch
            del readout
            del full_logits
            del choice_logits

        feature_values_by_name = {
            feature_name: np.concatenate(blocks, axis=0)
            for feature_name, blocks in feature_blocks.items()
        }

        for example_index, example_id in enumerate(example_rows["example_id"].tolist()):
            final_error = int(not bool(clean_is_correct[example_index]))
            for feature_name in FEATURE_NAMES:
                rows.append(
                    {
                        "example_id": example_id,
                        "split": split_name,
                        "layer_number": layer_index + 1,
                        "feature_name": feature_name,
                        "feature_value": float(feature_values_by_name[feature_name][example_index]),
                        "final_error": final_error,
                        "clean_is_correct": bool(clean_is_correct[example_index]),
                    }
                )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return pd.DataFrame(rows)


def fit_univariate_detectors(
    *,
    fit_feature_df: pd.DataFrame,
    eval_feature_df: pd.DataFrame,
    fit_split_name: str,
    eval_split_name: str,
) -> tuple[dict[tuple[str, int], object], pd.DataFrame, pd.DataFrame]:
    detector_models: dict[tuple[str, int], object] = {}
    coefficient_rows: list[dict[str, object]] = []
    output_rows: list[dict[str, object]] = []

    grouped_keys = sorted(
        fit_feature_df[["feature_name", "layer_number"]].drop_duplicates().itertuples(index=False, name=None)
    )
    for feature_name, layer_number in tqdm(grouped_keys, desc="detector training"):
        fit_part = fit_feature_df.loc[
            fit_feature_df["feature_name"].eq(feature_name)
            & fit_feature_df["layer_number"].eq(layer_number)
        ].copy()
        eval_part = eval_feature_df.loc[
            eval_feature_df["feature_name"].eq(feature_name)
            & eval_feature_df["layer_number"].eq(layer_number)
        ].copy()

        X_fit = fit_part[["feature_value"]].to_numpy(dtype=float)
        y_fit = fit_part["final_error"].to_numpy(dtype=int)
        X_eval = eval_part[["feature_value"]].to_numpy(dtype=float)
        y_eval = eval_part["final_error"].to_numpy(dtype=int)

        pipe = make_pipeline(
            StandardScaler(),
            LogisticRegressionCV(
                Cs=DETECTOR_C_GRID,
                cv=5,
                penalty="l1",
                solver="saga",
                scoring="average_precision",
                max_iter=5000,
                n_jobs=-1,
                random_state=42,
                refit=True,
            ),
        )
        pipe.fit(X_fit, y_fit)
        detector_models[(feature_name, int(layer_number))] = pipe

        scaler = pipe.named_steps["standardscaler"]
        model = pipe.named_steps["logisticregressioncv"]
        selected_c = float(np.ravel(model.C_)[0])

        fit_prob = pipe.predict_proba(X_fit)[:, 1]
        eval_prob = pipe.predict_proba(X_eval)[:, 1]
        fit_pred = (fit_prob >= 0.5).astype(int)
        eval_pred = (eval_prob >= 0.5).astype(int)

        coefficient_rows.append(
            {
                "feature_name": feature_name,
                "layer_number": int(layer_number),
                "coefficient": float(model.coef_.ravel()[0]),
                "abs_coefficient": float(abs(model.coef_.ravel()[0])),
                "is_nonzero": bool(model.coef_.ravel()[0] != 0.0),
                "selected_c": selected_c,
                "intercept": float(model.intercept_[0]),
                "feature_mean": float(scaler.mean_[0]),
                "feature_scale": float(scaler.scale_[0]),
            }
        )

        for part_name, part, prob, pred in [
            (fit_split_name, fit_part, fit_prob, fit_pred),
            (eval_split_name, eval_part, eval_prob, eval_pred),
        ]:
            for row_index, example_id in enumerate(part["example_id"].tolist()):
                output_rows.append(
                    {
                        "split": part_name,
                        "example_id": example_id,
                        "feature_name": feature_name,
                        "layer_number": int(layer_number),
                        "feature_value": float(part["feature_value"].iloc[row_index]),
                        "final_error": int(part["final_error"].iloc[row_index]),
                        "clean_is_correct": bool(part["clean_is_correct"].iloc[row_index]),
                        "detector_probability_error": float(prob[row_index]),
                    }
                )

    return (
        detector_models,
        pd.DataFrame(coefficient_rows),
        pd.DataFrame(output_rows),
    )


def build_feature_target_stats(fit_feature_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped_keys = sorted(
        fit_feature_df[["feature_name", "layer_number"]].drop_duplicates().itertuples(index=False, name=None)
    )
    for feature_name, layer_number in grouped_keys:
        part = fit_feature_df.loc[
            fit_feature_df["feature_name"].eq(feature_name)
            & fit_feature_df["layer_number"].eq(layer_number)
        ].copy()
        correct_values = part.loc[part["clean_is_correct"], "feature_value"].to_numpy(dtype=float)
        incorrect_values = part.loc[~part["clean_is_correct"], "feature_value"].to_numpy(dtype=float)
        rows.append(
            {
                "feature_name": feature_name,
                "layer_number": int(layer_number),
                "target_lower": float(np.quantile(correct_values, TARGET_LOWER_QUANTILE)),
                "target_median": float(np.median(correct_values)),
                "target_upper": float(np.quantile(correct_values, TARGET_UPPER_QUANTILE)),
                "correct_mean": float(correct_values.mean()),
                "incorrect_mean": float(incorrect_values.mean()),
                "correct_std": float(correct_values.std(ddof=0)),
                "incorrect_std": float(incorrect_values.std(ddof=0)),
                "correct_min": float(correct_values.min()),
                "correct_max": float(correct_values.max()),
            }
        )
    return pd.DataFrame(rows)


def compute_feature_from_token_hidden(
    token_hidden: torch.Tensor,
    *,
    feature_name: str,
    layer_index_0based: int,
    maybe_apply_final_norm,
    lm_head_weight: torch.Tensor,
    answer_id_tensor_lm_head: torch.Tensor,
    vocab_size: int,
) -> torch.Tensor:
    readout = maybe_apply_final_norm(token_hidden, layer_index_0based)
    full_logits = torch.matmul(
        readout.to(lm_head_weight.dtype),
        lm_head_weight.T,
    ).float()
    choice_logits = full_logits.index_select(1, answer_id_tensor_lm_head)
    return compute_feature_tensor(
        feature_name=feature_name,
        full_logits=full_logits,
        choice_logits=choice_logits,
        vocab_size=vocab_size,
    )


def compute_feature_steering_delta(
    token_hidden: torch.Tensor,
    *,
    feature_name: str,
    layer_index_0based: int,
    maybe_apply_final_norm,
    lm_head_weight: torch.Tensor,
    answer_id_tensor_lm_head: torch.Tensor,
    vocab_size: int,
    target_lower: float,
    target_upper: float,
    intervention_mask: np.ndarray,
) -> dict[str, object]:
    with torch.enable_grad():
        base = token_hidden.detach().clone().requires_grad_(True)
        current_feature = compute_feature_from_token_hidden(
            base,
            feature_name=feature_name,
            layer_index_0based=layer_index_0based,
            maybe_apply_final_norm=maybe_apply_final_norm,
            lm_head_weight=lm_head_weight,
            answer_id_tensor_lm_head=answer_id_tensor_lm_head,
            vocab_size=vocab_size,
        )
        target_feature = current_feature.detach().clamp(min=float(target_lower), max=float(target_upper))
        desired_shift = target_feature - current_feature.detach()

        deltas: list[torch.Tensor] = []
        grad_norms: list[float] = []
        for batch_index in range(base.shape[0]):
            if (not bool(intervention_mask[batch_index])) or (abs(float(desired_shift[batch_index].item())) <= 1e-8):
                deltas.append(torch.zeros_like(base[batch_index]))
                grad_norms.append(0.0)
                continue

            grad_full = torch.autograd.grad(
                current_feature[batch_index],
                base,
                retain_graph=(batch_index < (base.shape[0] - 1)),
                create_graph=False,
                allow_unused=False,
            )[0]
            grad_i = grad_full[batch_index]
            grad_norm_sq = torch.dot(grad_i.float(), grad_i.float())
            grad_norm = float(torch.sqrt(torch.clamp(grad_norm_sq, min=0.0)).item())
            if (not math.isfinite(grad_norm)) or grad_norm_sq.item() <= GRAD_NORM_EPS:
                deltas.append(torch.zeros_like(base[batch_index]))
                grad_norms.append(0.0)
                continue

            delta_i = (desired_shift[batch_index] / grad_norm_sq) * grad_i
            deltas.append(delta_i.detach())
            grad_norms.append(grad_norm)

        delta = torch.stack(deltas, dim=0)
        steered_feature = compute_feature_from_token_hidden(
            base + delta,
            feature_name=feature_name,
            layer_index_0based=layer_index_0based,
            maybe_apply_final_norm=maybe_apply_final_norm,
            lm_head_weight=lm_head_weight,
            answer_id_tensor_lm_head=answer_id_tensor_lm_head,
            vocab_size=vocab_size,
        ).detach()

    token_hidden_l2 = token_hidden.detach().float().norm(dim=-1)
    delta_l2 = delta.detach().float().norm(dim=-1)
    delta_over_hidden = delta_l2 / token_hidden_l2.clamp_min(1e-12)

    return {
        "delta": delta.detach(),
        "steered_feature_value_local": steered_feature.cpu().numpy().astype(np.float32),
        "grad_l2_norm": np.asarray(grad_norms, dtype=np.float32),
        "delta_l2_norm": delta_l2.detach().cpu().numpy().astype(np.float32),
        "delta_over_token_hidden_l2": delta_over_hidden.detach().cpu().numpy().astype(np.float32),
        "token_hidden_l2_norm": token_hidden_l2.detach().cpu().numpy().astype(np.float32),
    }


def run_validation_policy(
    *,
    eval_rows: pd.DataFrame,
    eval_cache: dict[str, object],
    eval_detector_outputs_df: pd.DataFrame,
    target_stats_df: pd.DataFrame,
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
    detector_lookup = eval_detector_outputs_df.set_index(["example_id", "feature_name", "layer_number"])
    target_lookup = target_stats_df.set_index(["feature_name", "layer_number"])

    validation_choice_logits = eval_cache["clean_choice_logits"]
    validation_best_non_choice_logit = eval_cache["clean_best_non_choice_logit"]
    validation_best_non_choice_token_id = eval_cache["clean_best_non_choice_token_id"]
    example_ids = [row["example_id"] for row in eval_cache["example_rows"]]
    example_id_to_index = {example_id: idx for idx, example_id in enumerate(example_ids)}

    rows: list[dict[str, object]] = []
    total_steps = len(decoder_layers) * len(FEATURE_NAMES)

    with tqdm(total=total_steps, desc=f"{eval_split_name} policy sweep") as pbar:
        for feature_name in FEATURE_NAMES:
            for layer_number in range(1, len(decoder_layers) + 1):
                steering_module = decoder_layers[layer_number - 1]
                target_row = target_lookup.loc[(feature_name, layer_number)]
                target_lower = float(target_row["target_lower"])
                target_upper = float(target_row["target_upper"])

                detector_prob_all = np.array(
                    [
                        float(detector_lookup.loc[(example_id, feature_name, layer_number), "detector_probability_error"])
                        for example_id in example_ids
                    ],
                    dtype=np.float32,
                )
                detector_pred_all = detector_prob_all >= 0.5

                for start in range(0, len(eval_rows), INTERVENTION_BATCH_SIZE):
                    batch_df = eval_rows.iloc[start:start + INTERVENTION_BATCH_SIZE].reset_index(drop=True)
                    batch_indices = [example_id_to_index[example_id] for example_id in batch_df["example_id"].tolist()]
                    batch_detector_prob = detector_prob_all[batch_indices]
                    batch_detector_pred = detector_pred_all[batch_indices]

                    if not bool(batch_detector_pred.any()):
                        for batch_index, row in batch_df.iterrows():
                            global_index = batch_indices[batch_index]
                            rows.append(
                                {
                                    "example_id": row["example_id"],
                                    "feature_name": feature_name,
                                    "layer_number": layer_number,
                                    "detector_probability_error": float(batch_detector_prob[batch_index]),
                                    "steered_feature_value_local": np.nan,
                                    "grad_l2_norm": 0.0,
                                    "delta_l2_norm": 0.0,
                                    "delta_over_token_hidden_l2": 0.0,
                                    "token_hidden_l2_norm": np.nan,
                                    "steered_best_non_choice_token_id": int(validation_best_non_choice_token_id[global_index]),
                                    "steered_best_non_choice_logit": float(validation_best_non_choice_logit[global_index]),
                                    "steered_logit_A": float(validation_choice_logits[global_index, 0]),
                                    "steered_logit_B": float(validation_choice_logits[global_index, 1]),
                                    "steered_logit_C": float(validation_choice_logits[global_index, 2]),
                                    "steered_logit_D": float(validation_choice_logits[global_index, 3]),
                                    "steered_logit_E": float(validation_choice_logits[global_index, 4]),
                                }
                            )
                        continue

                    batch_cpu = encode_prompts(batch_df["text"].tolist(), tok, max_seq_len)
                    decision_pos = batch_cpu.pop("decision_pos")
                    batch_cpu.pop("prompt_token_count")
                    batch = {k: v.to(input_device) for k, v in batch_cpu.items()}
                    decision_pos = decision_pos.to(input_device)
                    true_choice_idx = torch.tensor(
                        [LETTERS.index(str(x)) for x in batch_df["answerKey"].tolist()],
                        dtype=torch.long,
                        device=input_device,
                    )
                    steering_stats: dict[str, np.ndarray] = {}

                    def steering_hook(module, inputs, output):
                        hidden = unpack_output_hidden(output)
                        row_idx = torch.arange(hidden.shape[0], device=hidden.device)
                        token_hidden = hidden[row_idx, decision_pos]
                        stats = compute_feature_steering_delta(
                            token_hidden,
                            feature_name=feature_name,
                            layer_index_0based=layer_number - 1,
                            maybe_apply_final_norm=maybe_apply_final_norm,
                            lm_head_weight=lm_head_weight,
                            answer_id_tensor_lm_head=answer_id_tensor_lm_head.to(hidden.device),
                            vocab_size=vocab_size,
                            target_lower=target_lower,
                            target_upper=target_upper,
                            intervention_mask=batch_detector_pred,
                        )
                        steering_stats.update({key: value for key, value in stats.items() if key != "delta"})
                        hidden_out = hidden.clone()
                        hidden_out[row_idx, decision_pos] = token_hidden + stats["delta"].to(hidden.device, dtype=hidden.dtype)
                        return repack_output_hidden(output, hidden_out)

                    handle = steering_module.register_forward_hook(steering_hook)
                    try:
                        with torch.no_grad():
                            out = model(**batch, return_dict=True, use_cache=False)
                    finally:
                        handle.remove()

                    full_logits = select_full_logits_at_decision(out.logits, decision_pos)
                    metrics = summarize_decision_logits(
                        full_logits,
                        true_choice_idx,
                        answer_id_tensor_cpu.to(full_logits.device),
                    )
                    choice_logits_cpu = metrics["choice_logits"].detach().cpu().numpy().astype(np.float32)
                    best_non_choice_logit_cpu = metrics["best_non_choice_logit"].detach().cpu().numpy().astype(np.float32)
                    masked_logits = full_logits.clone()
                    masked_logits[:, answer_id_tensor_cpu.to(full_logits.device)] = -torch.inf
                    best_non_choice_token_id_cpu = torch.argmax(masked_logits, dim=-1).detach().cpu().numpy().astype(np.int64)

                    for batch_index, row in batch_df.iterrows():
                        rows.append(
                            {
                                "example_id": row["example_id"],
                                "feature_name": feature_name,
                                "layer_number": layer_number,
                                "detector_probability_error": float(batch_detector_prob[batch_index]),
                                "steered_feature_value_local": float(steering_stats["steered_feature_value_local"][batch_index]),
                                "grad_l2_norm": float(steering_stats["grad_l2_norm"][batch_index]),
                                "delta_l2_norm": float(steering_stats["delta_l2_norm"][batch_index]),
                                "delta_over_token_hidden_l2": float(steering_stats["delta_over_token_hidden_l2"][batch_index]),
                                "token_hidden_l2_norm": float(steering_stats["token_hidden_l2_norm"][batch_index]),
                                "steered_best_non_choice_token_id": int(best_non_choice_token_id_cpu[batch_index]),
                                "steered_best_non_choice_logit": float(best_non_choice_logit_cpu[batch_index]),
                                "steered_logit_A": float(choice_logits_cpu[batch_index, 0]),
                                "steered_logit_B": float(choice_logits_cpu[batch_index, 1]),
                                "steered_logit_C": float(choice_logits_cpu[batch_index, 2]),
                                "steered_logit_D": float(choice_logits_cpu[batch_index, 3]),
                                "steered_logit_E": float(choice_logits_cpu[batch_index, 4]),
                            }
                        )

                pbar.update(1)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    return pd.DataFrame(rows)

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="csqa", choices=SUPPORTED_DATASETS)
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--max-seq-len", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fit-split", type=str, default="validation")
    parser.add_argument("--eval-split", type=str, default="train")
    parser.add_argument("--fit-limit", type=int, default=None)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--validation-limit", type=int, default=None)
    parser.add_argument("--aqua-train-size", type=int, default=DEFAULT_AQUA_TRAIN_SIZE)
    parser.add_argument("--aqua-validation-size", type=int, default=DEFAULT_AQUA_VALIDATION_SIZE)
    parser.add_argument("--aqua-test-size", type=int, default=DEFAULT_AQUA_TEST_SIZE)
    parser.add_argument("--aqua-split-seed", type=int, default=DEFAULT_AQUA_SPLIT_SEED)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    fit_limit = args.fit_limit if args.fit_limit is not None else args.train_limit
    eval_limit = args.eval_limit if args.eval_limit is not None else args.validation_limit
    if args.fit_split == args.eval_split:
        raise ValueError("--fit-split and --eval-split must be different")

    print(
        "[config]",
        json.dumps(
            {
                "dataset": args.dataset,
                "model_id": args.model_id,
                "fit_split": args.fit_split,
                "eval_split": args.eval_split,
                "fit_limit": fit_limit,
                "eval_limit": eval_limit,
                "max_seq_len": args.max_seq_len,
                "seed": args.seed,
                "aqua_train_size": args.aqua_train_size,
                "aqua_validation_size": args.aqua_validation_size,
                "aqua_test_size": args.aqua_test_size,
                "aqua_split_seed": args.aqua_split_seed,
            },
            indent=2,
        ),
    )

    fit_rows = load_mcqa(
        args.dataset,
        split=args.fit_split,
        limit=fit_limit,
        aqua_train_size=args.aqua_train_size,
        aqua_validation_size=args.aqua_validation_size,
        aqua_test_size=args.aqua_test_size,
        aqua_split_seed=args.aqua_split_seed,
    ).copy()
    eval_rows = load_mcqa(
        args.dataset,
        split=args.eval_split,
        limit=eval_limit,
        aqua_train_size=args.aqua_train_size,
        aqua_validation_size=args.aqua_validation_size,
        aqua_test_size=args.aqua_test_size,
        aqua_split_seed=args.aqua_split_seed,
    ).copy()
    for frame in [fit_rows, eval_rows]:
        frame["prompt_len_chars"] = frame["text"].str.len()

    model_dtype, device_map = choose_model_dtype_and_device_map()

    hf_token = resolve_hf_token()

    tok = AutoTokenizer.from_pretrained(args.model_id, token=hf_token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        token=hf_token,
        dtype=model_dtype,
        device_map=device_map,
        attn_implementation="eager",
    )
    model.eval()

    input_device = get_input_device(model)
    decoder_layers = get_decoder_layers(model)
    num_layers = len(decoder_layers)
    vocab_size = int(model.config.vocab_size)

    readout_ctx = prepare_readout_context(
        model=model,
        tok=tok,
        input_device=input_device,
        num_layers=num_layers,
        max_seq_len=args.max_seq_len,
        probe_rows=eval_rows,
    )
    final_norm = readout_ctx["final_norm"]
    answer_id_tensor_cpu = readout_ctx["answer_id_tensor_cpu"]
    answer_id_tensor_lm_head = readout_ctx["answer_id_tensor_lm_head"]
    answer_choice_weight = readout_ctx["answer_choice_weight"]
    lm_head_weight = readout_ctx["lm_head_weight"]
    maybe_apply_final_norm = readout_ctx["maybe_apply_final_norm"]
    last_layer_needs_final_norm = readout_ctx["last_layer_needs_final_norm"]

    fit_cache = extract_split_cache(
        fit_rows,
        split_name=args.fit_split,
        tok=tok,
        model=model,
        input_device=input_device,
        answer_id_tensor_cpu=answer_id_tensor_cpu,
        answer_id_tensor_lm_head=answer_id_tensor_lm_head,
        answer_choice_weight=answer_choice_weight,
        final_norm=final_norm,
        num_layers=num_layers,
        max_seq_len=args.max_seq_len,
        batch_size=EXTRACT_BATCH_SIZE,
        last_layer_needs_final_norm=last_layer_needs_final_norm,
    )
    eval_cache = extract_split_cache(
        eval_rows,
        split_name=args.eval_split,
        tok=tok,
        model=model,
        input_device=input_device,
        answer_id_tensor_cpu=answer_id_tensor_cpu,
        answer_id_tensor_lm_head=answer_id_tensor_lm_head,
        answer_choice_weight=answer_choice_weight,
        final_norm=final_norm,
        num_layers=num_layers,
        max_seq_len=args.max_seq_len,
        batch_size=EXTRACT_BATCH_SIZE,
        last_layer_needs_final_norm=last_layer_needs_final_norm,
    )

    fit_feature_df = build_feature_table(
        split_name=args.fit_split,
        cache=fit_cache,
        maybe_apply_final_norm=maybe_apply_final_norm,
        lm_head_weight=lm_head_weight,
        answer_id_tensor_lm_head=answer_id_tensor_lm_head,
        input_device=input_device,
        vocab_size=vocab_size,
    )
    eval_feature_df = build_feature_table(
        split_name=args.eval_split,
        cache=eval_cache,
        maybe_apply_final_norm=maybe_apply_final_norm,
        lm_head_weight=lm_head_weight,
        answer_id_tensor_lm_head=answer_id_tensor_lm_head,
        input_device=input_device,
        vocab_size=vocab_size,
    )

    detector_models, detector_coefficients_df, detector_outputs_df = fit_univariate_detectors(
        fit_feature_df=fit_feature_df,
        eval_feature_df=eval_feature_df,
        fit_split_name=args.fit_split,
        eval_split_name=args.eval_split,
    )
    del detector_models

    target_stats_df = build_feature_target_stats(fit_feature_df)

    validation_policy_outputs_raw_df = run_validation_policy(
        eval_rows=eval_rows,
        eval_cache=eval_cache,
        eval_detector_outputs_df=detector_outputs_df.loc[detector_outputs_df["split"].eq(args.eval_split)].copy(),
        target_stats_df=target_stats_df,
        tok=tok,
        model=model,
        input_device=input_device,
        decoder_layers=decoder_layers,
        answer_id_tensor_cpu=answer_id_tensor_cpu,
        answer_id_tensor_lm_head=answer_id_tensor_lm_head,
        lm_head_weight=lm_head_weight,
        maybe_apply_final_norm=maybe_apply_final_norm,
        vocab_size=vocab_size,
        max_seq_len=args.max_seq_len,
        eval_split_name=args.eval_split,
    )

    out_dir = resolve_out_dir(args.out_dir, args.model_id, args.dataset)
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(fit_cache["example_rows"]).to_parquet(out_dir / f"{args.fit_split}_examples.parquet", index=False)
    pd.DataFrame(eval_cache["example_rows"]).to_parquet(out_dir / f"{args.eval_split}_examples.parquet", index=False)
    pd.DataFrame(fit_cache["clean_output_rows"]).to_parquet(out_dir / f"{args.fit_split}_clean_final_outputs.parquet", index=False)
    pd.DataFrame(eval_cache["clean_output_rows"]).to_parquet(out_dir / f"{args.eval_split}_clean_final_outputs.parquet", index=False)
    fit_feature_df.to_parquet(out_dir / f"{args.fit_split}_univariate_feature_values.parquet", index=False)
    eval_feature_df.to_parquet(out_dir / f"{args.eval_split}_univariate_feature_values.parquet", index=False)
    detector_coefficients_df.to_parquet(out_dir / "detector_coefficients.parquet", index=False)
    detector_outputs_df.to_parquet(out_dir / "detector_outputs.parquet", index=False)
    target_stats_df.to_parquet(out_dir / "feature_target_stats.parquet", index=False)
    validation_policy_outputs_raw_df.to_parquet(out_dir / f"{args.eval_split}_policy_outputs_raw.parquet", index=False)

    run_config = {
        "dataset": args.dataset,
        "model_id": args.model_id,
        "seed": int(args.seed),
        "max_seq_len": int(args.max_seq_len),
        "fit_split": args.fit_split,
        "eval_split": args.eval_split,
        "fit_limit": fit_limit,
        "eval_limit": eval_limit,
        "aqua_train_size": int(args.aqua_train_size),
        "aqua_validation_size": int(args.aqua_validation_size),
        "aqua_test_size": int(args.aqua_test_size),
        "aqua_split_seed": int(args.aqua_split_seed),
        "method": "univariate_logit_feature_steering",
        "feature_names": FEATURE_NAMES,
        "target_lower_quantile": TARGET_LOWER_QUANTILE,
        "target_upper_quantile": TARGET_UPPER_QUANTILE,
        "extract_batch_size": EXTRACT_BATCH_SIZE,
        "readout_batch_size": READOUT_BATCH_SIZE,
        "intervention_batch_size": INTERVENTION_BATCH_SIZE,
        "last_layer_needs_final_norm": bool(last_layer_needs_final_norm),
    }
    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    print(f"[done] wrote {out_dir}")


if __name__ == "__main__":
    main()
