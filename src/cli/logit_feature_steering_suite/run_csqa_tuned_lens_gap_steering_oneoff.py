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

REPO_ROOT = Path(__file__).resolve().parents[3]
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
from src.cli.extract_csqa_trace_feature_tables import train_tuned_lenses  # noqa: E402
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
FEATURE_NAME = "answer_choice_top1_top2_logit_gap"
FEATURE_NAMES = [FEATURE_NAME]
CONTROL_INTERVENTION_TYPES = [
    "wrong_direction",
    "random_perp",
]
INTERVENTION_BATCH_SIZE = 2
DETECTOR_C_GRID = np.logspace(-3, 2, 12)
GRAD_NORM_EPS = 1e-12
TARGET_LOWER_QUANTILE = 0.25
TARGET_UPPER_QUANTILE = 0.75
DEFAULT_TRAIN_LIMIT = 2000


def now_id() -> str:
    import time

    return time.strftime("%Y%m%d-%H%M%S")


def repo_root() -> Path:
    return REPO_ROOT


def default_limit_for_split(split_name: str) -> int | None:
    return DEFAULT_TRAIN_LIMIT if split_name == "train" else None


def slugify_model_id(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", model_id).strip("-")


def resolve_out_dir(out_dir: str | None, model_id: str) -> Path:
    root = repo_root()
    if out_dir is None:
        run_name = f"{now_id()}_{slugify_model_id(model_id)}_csqa_tuned_lens_gap_steering_oneoff"
        return root / "data" / "generated" / "tuned_lens_gap_steering_oneoff" / run_name
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


def latter_half_layer_numbers(num_layers: int) -> list[int]:
    start_layer = (num_layers // 2) + 1
    return list(range(start_layer, num_layers + 1))


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


def compute_tuned_lens_readout(
    hidden_batch: torch.Tensor,
    *,
    layer_index_0based: int,
    layer_lens,
    final_norm,
    maybe_apply_final_norm,
) -> torch.Tensor:
    hidden_batch = hidden_batch.float()
    if layer_lens is None:
        return maybe_apply_final_norm(hidden_batch, layer_index_0based)

    readout = layer_lens(hidden_batch)
    if final_norm is not None:
        readout = final_norm(readout)
    return readout


def build_feature_table(
    *,
    split_name: str,
    cache: dict[str, object],
    lenses: list[object | None],
    final_norm,
    maybe_apply_final_norm,
    lm_head_weight: torch.Tensor,
    answer_id_tensor_lm_head: torch.Tensor,
    input_device: torch.device,
    vocab_size: int,
    active_layer_numbers: list[int] | None = None,
) -> pd.DataFrame:
    hidden = cache["hidden"]
    clean_is_correct = cache["clean_is_correct"]
    example_rows = pd.DataFrame(cache["example_rows"])[["example_id", "split"]]
    rows: list[dict[str, object]] = []

    layer_indices = (
        [layer_number - 1 for layer_number in active_layer_numbers]
        if active_layer_numbers is not None
        else list(range(hidden.shape[1]))
    )

    for layer_index in tqdm(layer_indices, desc=f"{split_name} feature extraction"):
        layer_hidden = hidden[:, layer_index, :]
        feature_blocks: dict[str, list[np.ndarray]] = {feature_name: [] for feature_name in FEATURE_NAMES}
        layer_lens = lenses[layer_index]
        if layer_lens is not None:
            layer_lens = layer_lens.to(input_device)
            layer_lens.eval()
            for param in layer_lens.parameters():
                param.requires_grad_(False)

        for start in range(0, layer_hidden.shape[0], READOUT_BATCH_SIZE):
            end = start + READOUT_BATCH_SIZE
            hidden_batch = layer_hidden[start:end].to(input_device)
            readout = compute_tuned_lens_readout(
                hidden_batch.float(),
                layer_index_0based=layer_index,
                layer_lens=layer_lens,
                final_norm=final_norm,
                maybe_apply_final_norm=maybe_apply_final_norm,
            )
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
        if layer_lens is not None:
            layer_lens = layer_lens.cpu()

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
                class_weight="balanced",
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
    layer_lens,
    final_norm,
    maybe_apply_final_norm,
    lm_head_weight: torch.Tensor,
    answer_id_tensor_lm_head: torch.Tensor,
    vocab_size: int,
) -> torch.Tensor:
    readout = compute_tuned_lens_readout(
        token_hidden,
        layer_index_0based=layer_index_0based,
        layer_lens=layer_lens,
        final_norm=final_norm,
        maybe_apply_final_norm=maybe_apply_final_norm,
    )
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
    layer_lens,
    final_norm,
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
            layer_lens=layer_lens,
            final_norm=final_norm,
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

    token_hidden_l2 = token_hidden.detach().float().norm(dim=-1)
    delta_l2 = delta.detach().float().norm(dim=-1)
    delta_over_hidden = delta_l2 / token_hidden_l2.clamp_min(1e-12)

    return {
        "delta": delta.detach(),
        "current_feature_value": current_feature.detach().cpu().numpy().astype(np.float32),
        "target_feature_value": target_feature.detach().cpu().numpy().astype(np.float32),
        "desired_shift": desired_shift.detach().cpu().numpy().astype(np.float32),
        "grad_l2_norm": np.asarray(grad_norms, dtype=np.float32),
        "delta_l2_norm": delta_l2.detach().cpu().numpy().astype(np.float32),
        "delta_over_token_hidden_l2": delta_over_hidden.detach().cpu().numpy().astype(np.float32),
        "token_hidden_l2_norm": token_hidden_l2.detach().cpu().numpy().astype(np.float32),
    }


def summarize_delta_application(
    token_hidden: torch.Tensor,
    *,
    delta: torch.Tensor,
    feature_name: str,
    layer_index_0based: int,
    layer_lens,
    final_norm,
    maybe_apply_final_norm,
    lm_head_weight: torch.Tensor,
    answer_id_tensor_lm_head: torch.Tensor,
    vocab_size: int,
    grad_l2_norm: np.ndarray,
) -> dict[str, object]:
    steered_feature = compute_feature_from_token_hidden(
        token_hidden + delta,
        feature_name=feature_name,
        layer_index_0based=layer_index_0based,
        layer_lens=layer_lens,
        final_norm=final_norm,
        maybe_apply_final_norm=maybe_apply_final_norm,
        lm_head_weight=lm_head_weight,
        answer_id_tensor_lm_head=answer_id_tensor_lm_head,
        vocab_size=vocab_size,
    ).detach()

    token_hidden_l2 = token_hidden.detach().float().norm(dim=-1)
    delta_l2 = delta.detach().float().norm(dim=-1)
    delta_over_hidden = delta_l2 / token_hidden_l2.clamp_min(1e-12)

    return {
        "steered_feature_value_local": steered_feature.cpu().numpy().astype(np.float32),
        "grad_l2_norm": np.asarray(grad_l2_norm, dtype=np.float32),
        "delta_l2_norm": delta_l2.detach().cpu().numpy().astype(np.float32),
        "delta_over_token_hidden_l2": delta_over_hidden.detach().cpu().numpy().astype(np.float32),
        "token_hidden_l2_norm": token_hidden_l2.detach().cpu().numpy().astype(np.float32),
    }


def build_random_perp_delta(
    targeted_delta: torch.Tensor,
    *,
    intervention_mask: np.ndarray,
    generator: torch.Generator,
) -> torch.Tensor:
    random_delta = torch.zeros_like(targeted_delta)
    for batch_index in range(targeted_delta.shape[0]):
        if not bool(intervention_mask[batch_index]):
            continue

        basis = targeted_delta[batch_index].detach().float().cpu()
        basis_norm = float(basis.norm().item())
        if (not math.isfinite(basis_norm)) or basis_norm <= 1e-12:
            continue

        random_vec = None
        for _ in range(4):
            candidate = torch.randn(basis.shape, generator=generator, dtype=torch.float32)
            candidate = candidate - (torch.dot(candidate, basis) / torch.dot(basis, basis)) * basis
            candidate_norm = float(candidate.norm().item())
            if math.isfinite(candidate_norm) and candidate_norm > 1e-12:
                random_vec = candidate / candidate_norm
                break

        if random_vec is None:
            continue

        random_delta[batch_index] = (basis_norm * random_vec).to(
            device=targeted_delta.device,
            dtype=targeted_delta.dtype,
        )

    return random_delta


def run_single_intervention_forward(
    *,
    batch: dict[str, torch.Tensor],
    decision_pos: torch.Tensor,
    model: AutoModelForCausalLM,
    steering_module,
    delta: torch.Tensor,
    true_choice_idx: torch.Tensor,
    answer_id_tensor_cpu: torch.Tensor,
) -> dict[str, np.ndarray]:
    def steering_hook(module, inputs, output):
        hidden = unpack_output_hidden(output)
        row_idx = torch.arange(hidden.shape[0], device=hidden.device)
        hidden_out = hidden.clone()
        hidden_out[row_idx, decision_pos] = hidden[row_idx, decision_pos] + delta.to(
            hidden.device,
            dtype=hidden.dtype,
        )
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

    return {
        "choice_logits": choice_logits_cpu,
        "best_non_choice_logit": best_non_choice_logit_cpu,
        "best_non_choice_token_id": best_non_choice_token_id_cpu,
    }


def run_validation_policy(
    *,
    eval_rows: pd.DataFrame,
    eval_cache: dict[str, object],
    eval_detector_outputs_df: pd.DataFrame,
    target_stats_df: pd.DataFrame,
    lenses: list[object | None],
    final_norm,
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
    random_generator: torch.Generator,
    active_layer_numbers: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    detector_lookup = eval_detector_outputs_df.set_index(["example_id", "feature_name", "layer_number"])
    target_lookup = target_stats_df.set_index(["feature_name", "layer_number"])

    validation_choice_logits = eval_cache["clean_choice_logits"]
    validation_best_non_choice_logit = eval_cache["clean_best_non_choice_logit"]
    validation_best_non_choice_token_id = eval_cache["clean_best_non_choice_token_id"]
    example_ids = [row["example_id"] for row in eval_cache["example_rows"]]
    example_id_to_index = {example_id: idx for idx, example_id in enumerate(example_ids)}

    targeted_rows: list[dict[str, object]] = []
    control_rows: list[dict[str, object]] = []
    total_steps = len(active_layer_numbers) * len(FEATURE_NAMES)

    with tqdm(total=total_steps, desc=f"{eval_split_name} policy sweep") as pbar:
        for feature_name in FEATURE_NAMES:
            for layer_number in active_layer_numbers:
                steering_module = decoder_layers[layer_number - 1]
                layer_lens = lenses[layer_number - 1]
                if layer_lens is not None:
                    layer_lens = layer_lens.to(input_device)
                    layer_lens.eval()
                    for param in layer_lens.parameters():
                        param.requires_grad_(False)
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
                            targeted_rows.append(
                                {
                                    "example_id": row["example_id"],
                                    "intervention_type": "targeted",
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
                            for intervention_type in CONTROL_INTERVENTION_TYPES:
                                control_rows.append(
                                    {
                                        "example_id": row["example_id"],
                                        "intervention_type": intervention_type,
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
                    token_hidden = eval_cache["hidden"][batch_indices, layer_number - 1, :].to(input_device)
                    targeted_plan = compute_feature_steering_delta(
                        token_hidden,
                        feature_name=feature_name,
                        layer_index_0based=layer_number - 1,
                        layer_lens=layer_lens,
                        final_norm=final_norm,
                        maybe_apply_final_norm=maybe_apply_final_norm,
                        lm_head_weight=lm_head_weight,
                        answer_id_tensor_lm_head=answer_id_tensor_lm_head.to(input_device),
                        vocab_size=vocab_size,
                        target_lower=target_lower,
                        target_upper=target_upper,
                        intervention_mask=batch_detector_pred,
                    )
                    targeted_delta = targeted_plan["delta"]
                    wrong_delta = -targeted_delta
                    random_perp_delta = build_random_perp_delta(
                        targeted_delta,
                        intervention_mask=batch_detector_pred,
                        generator=random_generator,
                    )

                    targeted_stats = summarize_delta_application(
                        token_hidden,
                        delta=targeted_delta,
                        feature_name=feature_name,
                        layer_index_0based=layer_number - 1,
                        layer_lens=layer_lens,
                        final_norm=final_norm,
                        maybe_apply_final_norm=maybe_apply_final_norm,
                        lm_head_weight=lm_head_weight,
                        answer_id_tensor_lm_head=answer_id_tensor_lm_head.to(input_device),
                        vocab_size=vocab_size,
                        grad_l2_norm=targeted_plan["grad_l2_norm"],
                    )
                    wrong_stats = summarize_delta_application(
                        token_hidden,
                        delta=wrong_delta,
                        feature_name=feature_name,
                        layer_index_0based=layer_number - 1,
                        layer_lens=layer_lens,
                        final_norm=final_norm,
                        maybe_apply_final_norm=maybe_apply_final_norm,
                        lm_head_weight=lm_head_weight,
                        answer_id_tensor_lm_head=answer_id_tensor_lm_head.to(input_device),
                        vocab_size=vocab_size,
                        grad_l2_norm=targeted_plan["grad_l2_norm"],
                    )
                    random_stats = summarize_delta_application(
                        token_hidden,
                        delta=random_perp_delta,
                        feature_name=feature_name,
                        layer_index_0based=layer_number - 1,
                        layer_lens=layer_lens,
                        final_norm=final_norm,
                        maybe_apply_final_norm=maybe_apply_final_norm,
                        lm_head_weight=lm_head_weight,
                        answer_id_tensor_lm_head=answer_id_tensor_lm_head.to(input_device),
                        vocab_size=vocab_size,
                        grad_l2_norm=targeted_plan["grad_l2_norm"],
                    )

                    targeted_outputs = run_single_intervention_forward(
                        batch=batch,
                        decision_pos=decision_pos,
                        model=model,
                        steering_module=steering_module,
                        delta=targeted_delta,
                        true_choice_idx=true_choice_idx,
                        answer_id_tensor_cpu=answer_id_tensor_cpu,
                    )
                    wrong_outputs = run_single_intervention_forward(
                        batch=batch,
                        decision_pos=decision_pos,
                        model=model,
                        steering_module=steering_module,
                        delta=wrong_delta,
                        true_choice_idx=true_choice_idx,
                        answer_id_tensor_cpu=answer_id_tensor_cpu,
                    )
                    random_outputs = run_single_intervention_forward(
                        batch=batch,
                        decision_pos=decision_pos,
                        model=model,
                        steering_module=steering_module,
                        delta=random_perp_delta,
                        true_choice_idx=true_choice_idx,
                        answer_id_tensor_cpu=answer_id_tensor_cpu,
                    )

                    for batch_index, row in batch_df.iterrows():
                        targeted_rows.append(
                            {
                                "example_id": row["example_id"],
                                "intervention_type": "targeted",
                                "feature_name": feature_name,
                                "layer_number": layer_number,
                                "detector_probability_error": float(batch_detector_prob[batch_index]),
                                "steered_feature_value_local": float(targeted_stats["steered_feature_value_local"][batch_index]),
                                "grad_l2_norm": float(targeted_stats["grad_l2_norm"][batch_index]),
                                "delta_l2_norm": float(targeted_stats["delta_l2_norm"][batch_index]),
                                "delta_over_token_hidden_l2": float(targeted_stats["delta_over_token_hidden_l2"][batch_index]),
                                "token_hidden_l2_norm": float(targeted_stats["token_hidden_l2_norm"][batch_index]),
                                "steered_best_non_choice_token_id": int(targeted_outputs["best_non_choice_token_id"][batch_index]),
                                "steered_best_non_choice_logit": float(targeted_outputs["best_non_choice_logit"][batch_index]),
                                "steered_logit_A": float(targeted_outputs["choice_logits"][batch_index, 0]),
                                "steered_logit_B": float(targeted_outputs["choice_logits"][batch_index, 1]),
                                "steered_logit_C": float(targeted_outputs["choice_logits"][batch_index, 2]),
                                "steered_logit_D": float(targeted_outputs["choice_logits"][batch_index, 3]),
                                "steered_logit_E": float(targeted_outputs["choice_logits"][batch_index, 4]),
                            }
                        )
                        for intervention_type, stats_dict, output_dict in [
                            ("wrong_direction", wrong_stats, wrong_outputs),
                            ("random_perp", random_stats, random_outputs),
                        ]:
                            control_rows.append(
                                {
                                    "example_id": row["example_id"],
                                    "intervention_type": intervention_type,
                                    "feature_name": feature_name,
                                    "layer_number": layer_number,
                                    "detector_probability_error": float(batch_detector_prob[batch_index]),
                                    "steered_feature_value_local": float(stats_dict["steered_feature_value_local"][batch_index]),
                                    "grad_l2_norm": float(stats_dict["grad_l2_norm"][batch_index]),
                                    "delta_l2_norm": float(stats_dict["delta_l2_norm"][batch_index]),
                                    "delta_over_token_hidden_l2": float(stats_dict["delta_over_token_hidden_l2"][batch_index]),
                                    "token_hidden_l2_norm": float(stats_dict["token_hidden_l2_norm"][batch_index]),
                                    "steered_best_non_choice_token_id": int(output_dict["best_non_choice_token_id"][batch_index]),
                                    "steered_best_non_choice_logit": float(output_dict["best_non_choice_logit"][batch_index]),
                                    "steered_logit_A": float(output_dict["choice_logits"][batch_index, 0]),
                                    "steered_logit_B": float(output_dict["choice_logits"][batch_index, 1]),
                                    "steered_logit_C": float(output_dict["choice_logits"][batch_index, 2]),
                                    "steered_logit_D": float(output_dict["choice_logits"][batch_index, 3]),
                                    "steered_logit_E": float(output_dict["choice_logits"][batch_index, 4]),
                                }
                            )

                pbar.update(1)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if layer_lens is not None:
                    layer_lens = layer_lens.cpu()

    return pd.DataFrame(targeted_rows), pd.DataFrame(control_rows)

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--max-seq-len", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fit-split", type=str, default="train")
    parser.add_argument("--eval-split", type=str, default="validation")
    parser.add_argument("--fit-limit", type=int, default=None)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--validation-limit", type=int, default=None)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    fit_limit = (
        args.fit_limit
        if args.fit_limit is not None
        else (
            args.train_limit
            if args.fit_split == "train" and args.train_limit is not None
            else (
                args.validation_limit
                if args.fit_split == "validation" and args.validation_limit is not None
                else default_limit_for_split(args.fit_split)
            )
        )
    )
    eval_limit = (
        args.eval_limit
        if args.eval_limit is not None
        else (
            args.train_limit
            if args.eval_split == "train" and args.train_limit is not None
            else (
                args.validation_limit
                if args.eval_split == "validation" and args.validation_limit is not None
                else default_limit_for_split(args.eval_split)
            )
        )
    )
    if args.fit_split == args.eval_split:
        raise ValueError("--fit-split and --eval-split must be different")

    print(
        "[config]",
        json.dumps(
            {
                "dataset": "csqa",
                "model_id": args.model_id,
                "fit_split": args.fit_split,
                "eval_split": args.eval_split,
                "fit_limit": fit_limit,
                "eval_limit": eval_limit,
                "max_seq_len": args.max_seq_len,
                "seed": args.seed,
            },
            indent=2,
        ),
    )

    fit_rows = load_csqa(split=args.fit_split, limit=fit_limit).copy()
    eval_rows = load_csqa(split=args.eval_split, limit=eval_limit).copy()
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
    active_layer_numbers = list(range(1, num_layers + 1))
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
    hidden_size = int(model.lm_head.weight.shape[1])

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

    lenses, lens_state_dicts, tuned_lens_history_df = train_tuned_lenses(
        train_hidden=fit_cache["hidden"],
        teacher_choice_probs_train=fit_cache["final_choice_probs"],
        hidden_size=hidden_size,
        num_layers=num_layers,
        answer_choice_weight=answer_choice_weight,
        train_device=input_device,
    )

    fit_feature_df = build_feature_table(
        split_name=args.fit_split,
        cache=fit_cache,
        lenses=lenses,
        final_norm=final_norm,
        maybe_apply_final_norm=maybe_apply_final_norm,
        lm_head_weight=lm_head_weight,
        answer_id_tensor_lm_head=answer_id_tensor_lm_head,
        input_device=input_device,
        vocab_size=vocab_size,
        active_layer_numbers=active_layer_numbers,
    )
    eval_feature_df = build_feature_table(
        split_name=args.eval_split,
        cache=eval_cache,
        lenses=lenses,
        final_norm=final_norm,
        maybe_apply_final_norm=maybe_apply_final_norm,
        lm_head_weight=lm_head_weight,
        answer_id_tensor_lm_head=answer_id_tensor_lm_head,
        input_device=input_device,
        vocab_size=vocab_size,
        active_layer_numbers=active_layer_numbers,
    )

    detector_models, detector_coefficients_df, detector_outputs_df = fit_univariate_detectors(
        fit_feature_df=fit_feature_df,
        eval_feature_df=eval_feature_df,
        fit_split_name=args.fit_split,
        eval_split_name=args.eval_split,
    )
    del detector_models

    target_stats_df = build_feature_target_stats(fit_feature_df)

    control_random_generator = torch.Generator(device="cpu")
    control_random_generator.manual_seed(args.seed)

    validation_policy_outputs_raw_df, validation_policy_control_outputs_raw_df = run_validation_policy(
        eval_rows=eval_rows,
        eval_cache=eval_cache,
        eval_detector_outputs_df=detector_outputs_df.loc[detector_outputs_df["split"].eq(args.eval_split)].copy(),
        target_stats_df=target_stats_df,
        lenses=lenses,
        final_norm=final_norm,
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
        random_generator=control_random_generator,
        active_layer_numbers=active_layer_numbers,
    )

    out_dir = resolve_out_dir(args.out_dir, args.model_id)
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
    tuned_lens_history_df.to_parquet(out_dir / "tuned_lens_training_history.parquet", index=False)
    torch.save(lens_state_dicts, out_dir / "tuned_lens_state.pt")
    validation_policy_outputs_raw_df.to_parquet(out_dir / f"{args.eval_split}_policy_outputs_raw.parquet", index=False)
    validation_policy_control_outputs_raw_df.to_parquet(
        out_dir / f"{args.eval_split}_policy_control_outputs_raw.parquet",
        index=False,
    )

    run_config = {
        "dataset": "csqa",
        "model_id": args.model_id,
        "seed": int(args.seed),
        "max_seq_len": int(args.max_seq_len),
        "fit_split": args.fit_split,
        "eval_split": args.eval_split,
        "fit_limit": fit_limit,
        "eval_limit": eval_limit,
        "method": "univariate_tuned_lens_gap_steering_oneoff",
        "readout_method": "tuned_lens",
        "detector_type": "l1_logistic_regression_cv",
        "detector_class_weight": "balanced",
        "feature_names": FEATURE_NAMES,
        "active_layer_numbers": active_layer_numbers,
        "layer_selection_rule": "all_layers",
        "control_intervention_types": CONTROL_INTERVENTION_TYPES,
        "target_lower_quantile": TARGET_LOWER_QUANTILE,
        "target_upper_quantile": TARGET_UPPER_QUANTILE,
        "extract_batch_size": EXTRACT_BATCH_SIZE,
        "readout_batch_size": READOUT_BATCH_SIZE,
        "intervention_batch_size": INTERVENTION_BATCH_SIZE,
        "last_layer_needs_final_norm": bool(last_layer_needs_final_norm),
        "tuned_lens_batch_size": 64,
        "tuned_lens_max_epochs": 10,
        "tuned_lens_patience": 2,
    }
    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    print(f"[done] wrote {out_dir}")


if __name__ == "__main__":
    main()
