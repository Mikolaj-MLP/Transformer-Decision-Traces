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
from src.data.load_csqa import load_csqa  # noqa: E402


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


def resolve_out_dir(out_dir: str | None, model_id: str) -> Path:
    root = repo_root()
    if out_dir is None:
        run_name = f"{now_id()}_{slugify_model_id(model_id)}_csqa_logit_feature_steering_pipeline"
        return root / "data" / "generated" / "csqa_logit_feature_steering_pipeline" / run_name
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
    train_feature_df: pd.DataFrame,
    validation_feature_df: pd.DataFrame,
) -> tuple[dict[tuple[str, int], object], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    detector_models: dict[tuple[str, int], object] = {}
    coefficient_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    output_rows: list[dict[str, object]] = []

    grouped_keys = sorted(
        train_feature_df[["feature_name", "layer_number"]].drop_duplicates().itertuples(index=False, name=None)
    )
    for feature_name, layer_number in tqdm(grouped_keys, desc="detector training"):
        train_part = train_feature_df.loc[
            train_feature_df["feature_name"].eq(feature_name)
            & train_feature_df["layer_number"].eq(layer_number)
        ].copy()
        validation_part = validation_feature_df.loc[
            validation_feature_df["feature_name"].eq(feature_name)
            & validation_feature_df["layer_number"].eq(layer_number)
        ].copy()

        X_train = train_part[["feature_value"]].to_numpy(dtype=float)
        y_train = train_part["final_error"].to_numpy(dtype=int)
        X_validation = validation_part[["feature_value"]].to_numpy(dtype=float)
        y_validation = validation_part["final_error"].to_numpy(dtype=int)

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
        pipe.fit(X_train, y_train)
        detector_models[(feature_name, int(layer_number))] = pipe

        scaler = pipe.named_steps["standardscaler"]
        model = pipe.named_steps["logisticregressioncv"]
        selected_c = float(np.ravel(model.C_)[0])

        train_prob = pipe.predict_proba(X_train)[:, 1]
        validation_prob = pipe.predict_proba(X_validation)[:, 1]
        train_pred = (train_prob >= 0.5).astype(int)
        validation_pred = (validation_prob >= 0.5).astype(int)

        validation_tp = int(np.sum((validation_pred == 1) & (y_validation == 1)))
        validation_fp = int(np.sum((validation_pred == 1) & (y_validation == 0)))
        validation_tn = int(np.sum((validation_pred == 0) & (y_validation == 0)))
        validation_fn = int(np.sum((validation_pred == 0) & (y_validation == 1)))

        summary_rows.append(
            {
                "feature_name": feature_name,
                "layer_number": int(layer_number),
                "selected_c": selected_c,
                "train_error_rate": float(y_train.mean()),
                "validation_error_rate": float(y_validation.mean()),
                "train_roc_auc_error": float(roc_auc_score(y_train, train_prob)),
                "train_pr_auc_error": float(average_precision_score(y_train, train_prob)),
                "validation_roc_auc_error": float(roc_auc_score(y_validation, validation_prob)),
                "validation_pr_auc_error": float(average_precision_score(y_validation, validation_prob)),
                "train_flag_rate": float(train_pred.mean()),
                "validation_flag_rate": float(validation_pred.mean()),
                "validation_precision": float(validation_tp / max(validation_tp + validation_fp, 1)),
                "validation_recall": float(validation_tp / max(validation_tp + validation_fn, 1)),
                "validation_false_positive_rate": float(validation_fp / max(validation_fp + validation_tn, 1)),
                "validation_tp": validation_tp,
                "validation_fp": validation_fp,
                "validation_tn": validation_tn,
                "validation_fn": validation_fn,
            }
        )

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
            ("train", train_part, train_prob, train_pred),
            ("validation", validation_part, validation_prob, validation_pred),
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
        pd.DataFrame(summary_rows),
        pd.DataFrame(output_rows),
    )


def build_feature_target_stats(train_feature_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped_keys = sorted(
        train_feature_df[["feature_name", "layer_number"]].drop_duplicates().itertuples(index=False, name=None)
    )
    for feature_name, layer_number in grouped_keys:
        part = train_feature_df.loc[
            train_feature_df["feature_name"].eq(feature_name)
            & train_feature_df["layer_number"].eq(layer_number)
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
        "current_feature_value": current_feature.detach().cpu().numpy().astype(np.float32),
        "target_feature_value": target_feature.detach().cpu().numpy().astype(np.float32),
        "steered_feature_value_local": steered_feature.cpu().numpy().astype(np.float32),
        "requested_feature_delta": desired_shift.detach().cpu().numpy().astype(np.float32),
        "grad_l2_norm": np.asarray(grad_norms, dtype=np.float32),
        "delta_l2_norm": delta_l2.detach().cpu().numpy().astype(np.float32),
        "delta_over_token_hidden_l2": delta_over_hidden.detach().cpu().numpy().astype(np.float32),
        "token_hidden_l2_norm": token_hidden_l2.detach().cpu().numpy().astype(np.float32),
    }


def run_validation_policy(
    *,
    validation_rows: pd.DataFrame,
    validation_cache: dict[str, object],
    validation_detector_outputs_df: pd.DataFrame,
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
) -> pd.DataFrame:
    detector_lookup = validation_detector_outputs_df.set_index(["example_id", "feature_name", "layer_number"])
    target_lookup = target_stats_df.set_index(["feature_name", "layer_number"])

    validation_choice_logits = validation_cache["clean_choice_logits"]
    validation_best_non_choice_logit = validation_cache["clean_best_non_choice_logit"]
    validation_best_non_choice_token_id = validation_cache["clean_best_non_choice_token_id"]
    example_ids = [row["example_id"] for row in validation_cache["example_rows"]]
    example_id_to_index = {example_id: idx for idx, example_id in enumerate(example_ids)}

    rows: list[dict[str, object]] = []
    total_steps = len(decoder_layers) * len(FEATURE_NAMES)

    with tqdm(total=total_steps, desc="validation policy sweep") as pbar:
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
                detector_pred_all = np.array(
                    [
                        bool(detector_lookup.loc[(example_id, feature_name, layer_number), "detector_predicted_error"])
                        for example_id in example_ids
                    ],
                    dtype=bool,
                )

                for start in range(0, len(validation_rows), INTERVENTION_BATCH_SIZE):
                    batch_df = validation_rows.iloc[start:start + INTERVENTION_BATCH_SIZE].reset_index(drop=True)
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


def summarize_policy_results(
    *,
    validation_clean_outputs_df: pd.DataFrame,
    policy_outputs_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = policy_outputs_df.merge(
        validation_clean_outputs_df.rename(
            columns={
                "clean_logit_A": "baseline_logit_A",
                "clean_logit_B": "baseline_logit_B",
                "clean_logit_C": "baseline_logit_C",
                "clean_logit_D": "baseline_logit_D",
                "clean_logit_E": "baseline_logit_E",
                "best_non_choice_token_id": "baseline_best_non_choice_token_id",
                "best_non_choice_logit": "baseline_best_non_choice_logit",
            }
        ),
        on="example_id",
        how="left",
        validate="many_to_one",
    )

    baseline_choice_logits = merged[
        ["baseline_logit_A", "baseline_logit_B", "baseline_logit_C", "baseline_logit_D", "baseline_logit_E"]
    ].to_numpy(dtype=np.float32)
    steered_choice_logits = merged[
        ["steered_logit_A", "steered_logit_B", "steered_logit_C", "steered_logit_D", "steered_logit_E"]
    ].to_numpy(dtype=np.float32)
    true_choice_idx = merged["true_choice_idx"].to_numpy(dtype=np.int64)

    clean_pred_idx = choice_logits_to_pred_idx(baseline_choice_logits)
    steered_pred_idx = choice_logits_to_pred_idx(steered_choice_logits)
    clean_is_correct = clean_pred_idx == true_choice_idx
    steered_is_correct = steered_pred_idx == true_choice_idx
    detector_predicted_error = merged["detector_probability_error"].to_numpy(dtype=float) >= 0.5

    merged["clean_predicted_choice_idx"] = clean_pred_idx
    merged["steered_predicted_choice_idx"] = steered_pred_idx
    merged["clean_is_correct"] = clean_is_correct
    merged["steered_is_correct"] = steered_is_correct
    merged["detector_predicted_error"] = detector_predicted_error
    merged["prediction_changed"] = clean_pred_idx != steered_pred_idx
    merged["rescued_error"] = (~clean_is_correct) & steered_is_correct
    merged["harmed_correct"] = clean_is_correct & (~steered_is_correct)
    merged["clean_answer_choice_entropy"] = choice_logits_to_entropy(baseline_choice_logits)
    merged["steered_answer_choice_entropy"] = choice_logits_to_entropy(steered_choice_logits)
    merged["clean_answer_choice_top1_top2_logit_gap"] = choice_logits_to_gap(baseline_choice_logits)
    merged["steered_answer_choice_top1_top2_logit_gap"] = choice_logits_to_gap(steered_choice_logits)
    merged["clean_answer_choice_top1_probability"] = choice_logits_to_top1_prob(baseline_choice_logits)
    merged["steered_answer_choice_top1_probability"] = choice_logits_to_top1_prob(steered_choice_logits)
    merged["clean_answer_choice_varentropy"] = choice_logits_to_varentropy(baseline_choice_logits)
    merged["steered_answer_choice_varentropy"] = choice_logits_to_varentropy(steered_choice_logits)
    merged["delta_answer_choice_entropy"] = merged["steered_answer_choice_entropy"] - merged["clean_answer_choice_entropy"]
    merged["delta_answer_choice_top1_top2_logit_gap"] = (
        merged["steered_answer_choice_top1_top2_logit_gap"] - merged["clean_answer_choice_top1_top2_logit_gap"]
    )
    merged["delta_answer_choice_top1_probability"] = (
        merged["steered_answer_choice_top1_probability"] - merged["clean_answer_choice_top1_probability"]
    )
    merged["delta_answer_choice_varentropy"] = (
        merged["steered_answer_choice_varentropy"] - merged["clean_answer_choice_varentropy"]
    )

    validation_choice_logits = validation_clean_outputs_df[
        ["clean_logit_A", "clean_logit_B", "clean_logit_C", "clean_logit_D", "clean_logit_E"]
    ].to_numpy(dtype=np.float32)
    validation_true = validation_clean_outputs_df["true_choice_idx"].to_numpy(dtype=np.int64)
    validation_clean_correct = choice_logits_to_pred_idx(validation_choice_logits) == validation_true
    clean_accuracy = float(validation_clean_correct.mean())
    n_total = int(len(validation_clean_outputs_df))
    n_clean_correct = int(validation_clean_correct.sum())
    n_clean_incorrect = int(n_total - n_clean_correct)

    summary_rows: list[dict[str, object]] = []
    for (feature_name, layer_number), part in merged.groupby(["feature_name", "layer_number"], sort=True):
        steered_correct = part["steered_is_correct"].to_numpy(dtype=bool)
        rescued = part["rescued_error"].to_numpy(dtype=bool)
        harmed = part["harmed_correct"].to_numpy(dtype=bool)
        flagged = part["detector_predicted_error"].to_numpy(dtype=bool)
        steered_accuracy = float(steered_correct.mean())
        rescued_count = int(rescued.sum())
        harmed_count = int(harmed.sum())
        summary_rows.append(
            {
                "feature_name": feature_name,
                "layer_number": int(layer_number),
                "n_total": n_total,
                "n_clean_correct": n_clean_correct,
                "n_clean_incorrect": n_clean_incorrect,
                "clean_accuracy": clean_accuracy,
                "steered_accuracy": steered_accuracy,
                "accuracy_delta": steered_accuracy - clean_accuracy,
                "flagged_count": int(flagged.sum()),
                "flag_rate": float(flagged.mean()),
                "rescued_count": rescued_count,
                "harmed_count": harmed_count,
                "net_gain_count": int(rescued_count - harmed_count),
                "rescued_rate_among_incorrect": float(rescued_count / max(n_clean_incorrect, 1)),
                "harmed_rate_among_correct": float(harmed_count / max(n_clean_correct, 1)),
                "prediction_changed_count": int(part["prediction_changed"].sum()),
                "mean_requested_feature_delta": float(part["requested_feature_delta"].mean()),
                "mean_abs_requested_feature_delta": float(part["requested_feature_delta"].abs().mean()),
                "mean_grad_l2_norm": float(part["grad_l2_norm"].mean()),
                "mean_delta_l2_norm": float(part["delta_l2_norm"].mean()),
                "mean_delta_over_token_hidden_l2": float(part["delta_over_token_hidden_l2"].mean()),
                "mean_detector_probability_error": float(part["detector_probability_error"].mean()),
                "flagged_precision": float(
                    part.loc[part["detector_predicted_error"], "clean_is_correct"].rsub(1).mean()
                    if int(flagged.sum()) > 0
                    else np.nan
                ),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["net_gain_count", "accuracy_delta", "rescued_count", "harmed_count", "mean_delta_l2_norm"],
        ascending=[False, False, False, True, True],
    ).reset_index(drop=True)
    return merged, summary_df


def select_best_policy(policy_summary_df: pd.DataFrame) -> dict[str, object]:
    best_row = policy_summary_df.iloc[0]
    return {
        "feature_name": str(best_row["feature_name"]),
        "layer_number": int(best_row["layer_number"]),
        "net_gain_count": int(best_row["net_gain_count"]),
        "accuracy_delta": float(best_row["accuracy_delta"]),
        "rescued_count": int(best_row["rescued_count"]),
        "harmed_count": int(best_row["harmed_count"]),
        "flagged_count": int(best_row["flagged_count"]),
        "mean_delta_l2_norm": float(best_row["mean_delta_l2_norm"]),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--max-seq-len", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--validation-limit", type=int, default=None)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_rows = load_csqa(split="train", limit=args.train_limit).copy()
    validation_rows = load_csqa(split="validation", limit=args.validation_limit).copy()
    for frame in [train_rows, validation_rows]:
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
    )
    final_norm = readout_ctx["final_norm"]
    answer_id_tensor_cpu = readout_ctx["answer_id_tensor_cpu"]
    answer_id_tensor_lm_head = readout_ctx["answer_id_tensor_lm_head"]
    answer_choice_weight = readout_ctx["answer_choice_weight"]
    lm_head_weight = readout_ctx["lm_head_weight"]
    maybe_apply_final_norm = readout_ctx["maybe_apply_final_norm"]
    last_layer_needs_final_norm = readout_ctx["last_layer_needs_final_norm"]

    train_cache = extract_split_cache(
        train_rows,
        split_name="train",
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
    validation_cache = extract_split_cache(
        validation_rows,
        split_name="validation",
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

    train_feature_df = build_feature_table(
        split_name="train",
        cache=train_cache,
        maybe_apply_final_norm=maybe_apply_final_norm,
        lm_head_weight=lm_head_weight,
        answer_id_tensor_lm_head=answer_id_tensor_lm_head,
        input_device=input_device,
        vocab_size=vocab_size,
    )
    validation_feature_df = build_feature_table(
        split_name="validation",
        cache=validation_cache,
        maybe_apply_final_norm=maybe_apply_final_norm,
        lm_head_weight=lm_head_weight,
        answer_id_tensor_lm_head=answer_id_tensor_lm_head,
        input_device=input_device,
        vocab_size=vocab_size,
    )

    detector_models, detector_coefficients_df, detector_summary_df, detector_outputs_df = fit_univariate_detectors(
        train_feature_df=train_feature_df,
        validation_feature_df=validation_feature_df,
    )
    del detector_models

    target_stats_df = build_feature_target_stats(train_feature_df)

    validation_policy_outputs_raw_df = run_validation_policy(
        validation_rows=validation_rows,
        validation_cache=validation_cache,
        validation_detector_outputs_df=detector_outputs_df.loc[detector_outputs_df["split"].eq("validation")].copy(),
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
    )

    validation_clean_outputs_df = pd.DataFrame(validation_cache["clean_output_rows"])
    validation_policy_outputs_derived_df, validation_policy_summary_df = summarize_policy_results(
        validation_clean_outputs_df=validation_clean_outputs_df,
        policy_outputs_df=validation_policy_outputs_raw_df,
    )
    best_policy = select_best_policy(validation_policy_summary_df)

    detector_outputs_raw_df = detector_outputs_df.copy()
    policy_outputs_raw_df = validation_policy_outputs_raw_df.copy()

    out_dir = resolve_out_dir(args.out_dir, args.model_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(train_cache["example_rows"]).to_parquet(out_dir / "train_examples.parquet", index=False)
    pd.DataFrame(validation_cache["example_rows"]).to_parquet(out_dir / "validation_examples.parquet", index=False)
    pd.DataFrame(train_cache["clean_output_rows"]).to_parquet(out_dir / "train_clean_final_outputs.parquet", index=False)
    validation_clean_outputs_df.to_parquet(out_dir / "validation_clean_final_outputs.parquet", index=False)
    train_feature_df.to_parquet(out_dir / "train_univariate_feature_values.parquet", index=False)
    validation_feature_df.to_parquet(out_dir / "validation_univariate_feature_values.parquet", index=False)
    detector_coefficients_df.to_parquet(out_dir / "detector_coefficients.parquet", index=False)
    detector_outputs_raw_df.to_parquet(out_dir / "detector_outputs.parquet", index=False)
    target_stats_df.to_parquet(out_dir / "feature_target_stats.parquet", index=False)
    policy_outputs_raw_df.to_parquet(out_dir / "validation_policy_outputs_raw.parquet", index=False)

    run_config = {
        "model_id": args.model_id,
        "seed": int(args.seed),
        "max_seq_len": int(args.max_seq_len),
        "train_limit": args.train_limit,
        "validation_limit": args.validation_limit,
        "method": "univariate_logit_feature_steering",
        "feature_names": FEATURE_NAMES,
        "target_lower_quantile": TARGET_LOWER_QUANTILE,
        "target_upper_quantile": TARGET_UPPER_QUANTILE,
        "extract_batch_size": EXTRACT_BATCH_SIZE,
        "readout_batch_size": READOUT_BATCH_SIZE,
        "intervention_batch_size": INTERVENTION_BATCH_SIZE,
        "last_layer_needs_final_norm": bool(last_layer_needs_final_norm),
        "best_policy": best_policy,
    }
    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    print(f"[done] wrote {out_dir}")


if __name__ == "__main__":
    main()
