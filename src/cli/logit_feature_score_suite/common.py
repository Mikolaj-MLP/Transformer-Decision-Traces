from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.neighbors import KernelDensity
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.csqa.common import (
    build_answer_token_ids,
    encode_prompts,
    get_final_norm_module,
    repack_output_hidden,
    select_full_logits_at_decision,
    summarize_decision_logits,
    unpack_output_hidden,
)
from src.data.load_csqa import load_csqa


LETTERS = ["A", "B", "C", "D", "E"]
EXTRACT_BATCH_SIZE = 4
READOUT_BATCH_SIZE = 64
GRID_POINTS = 512
GOOD_REGION_LOG_RATIO_THRESHOLD = math.log(1.5)
GRAD_NORM_EPS = 1e-12
KDE_JITTER_SCALE = 1e-6
SUPPORT_LOWER_QUANTILE = 0.01
SUPPORT_UPPER_QUANTILE = 0.99
KDE_BANDWIDTH_MULTIPLIER = 1.5
LOG_RATIO_SMOOTHING_SIGMA_BANDWIDTHS = 1.0


def get_input_device(model: AutoModelForCausalLM) -> torch.device:
    for param in model.parameters():
        if param.device.type != "meta":
            return param.device
    return torch.device("cpu")


def choose_model_dtype_and_device_map() -> tuple[torch.dtype, str | None]:
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16, "auto"
        return torch.float16, "auto"
    return torch.float32, None


def resolve_hf_token() -> str | None:
    for key in ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"]:
        value = os.getenv(key)
        if value:
            return value
    return None


def maybe_clone_to_float_cpu(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy().astype(np.float32)


def prepare_readout_context(
    *,
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    input_device: torch.device,
    num_layers: int,
    max_seq_len: int,
    probe_rows: pd.DataFrame | None = None,
) -> dict[str, object]:
    final_norm = get_final_norm_module(model)
    lm_head_weight = model.lm_head.weight.detach()
    lm_head_device = lm_head_weight.device
    answer_token_ids = build_answer_token_ids(tok)
    answer_ids = [answer_token_ids[letter] for letter in LETTERS]
    answer_id_tensor_cpu = torch.tensor(answer_ids, dtype=torch.long)
    answer_id_tensor_lm_head = answer_id_tensor_cpu.to(lm_head_device)
    answer_choice_weight = lm_head_weight.index_select(0, answer_id_tensor_lm_head)

    if probe_rows is None:
        probe_rows = load_csqa(split="validation", limit=1).copy()
    else:
        probe_rows = probe_rows.iloc[:1].copy()
    probe_cpu = encode_prompts(probe_rows["text"].tolist(), tok, max_seq_len)
    probe_pos = int(probe_cpu["decision_pos"][0].item())
    probe_batch = {
        key: value.to(input_device)
        for key, value in probe_cpu.items()
        if key not in ["decision_pos", "prompt_token_count"]
    }
    with torch.inference_mode():
        probe_out = model(**probe_batch, output_hidden_states=True, return_dict=True, use_cache=False)

    raw_last = probe_out.hidden_states[-1][0, probe_pos].float()
    target_choice_logits = probe_out.logits[0, probe_pos, answer_id_tensor_lm_head].float().detach().cpu()
    raw_choice_logits = torch.mv(answer_choice_weight.detach().float().cpu(), raw_last.detach().cpu())

    if final_norm is not None:
        normed_last = final_norm(raw_last.unsqueeze(0)).squeeze(0)
        normed_choice_logits = torch.mv(answer_choice_weight.detach().float().cpu(), normed_last.detach().cpu())
        raw_err = torch.mean(torch.abs(raw_choice_logits - target_choice_logits)).item()
        normed_err = torch.mean(torch.abs(normed_choice_logits - target_choice_logits)).item()
        last_layer_needs_final_norm = bool(normed_err < raw_err)
    else:
        last_layer_needs_final_norm = False

    def maybe_apply_final_norm(hidden: torch.Tensor, layer_index_0based: int) -> torch.Tensor:
        if final_norm is None:
            return hidden
        if layer_index_0based < (num_layers - 1):
            return final_norm(hidden)
        if last_layer_needs_final_norm:
            return final_norm(hidden)
        return hidden

    return {
        "final_norm": final_norm,
        "answer_id_tensor_cpu": answer_id_tensor_cpu,
        "answer_id_tensor_lm_head": answer_id_tensor_lm_head,
        "answer_choice_weight": answer_choice_weight,
        "lm_head_weight": lm_head_weight,
        "last_layer_needs_final_norm": last_layer_needs_final_norm,
        "maybe_apply_final_norm": maybe_apply_final_norm,
    }


def extract_split_cache(
    frame: pd.DataFrame,
    *,
    split_name: str,
    tok: AutoTokenizer,
    model: AutoModelForCausalLM,
    input_device: torch.device,
    answer_id_tensor_cpu: torch.Tensor,
    answer_id_tensor_lm_head: torch.Tensor,
    answer_choice_weight: torch.Tensor,
    final_norm,
    num_layers: int,
    max_seq_len: int,
    batch_size: int,
    last_layer_needs_final_norm: bool,
) -> dict[str, object]:
    hidden_blocks: list[torch.Tensor] = []
    final_choice_prob_blocks: list[torch.Tensor] = []
    true_choice_idx_blocks: list[torch.Tensor] = []
    example_rows: list[dict[str, object]] = []
    clean_output_rows: list[dict[str, object]] = []
    clean_choice_logits_blocks: list[np.ndarray] = []
    clean_best_non_choice_logit_blocks: list[np.ndarray] = []
    clean_best_non_choice_token_id_blocks: list[np.ndarray] = []
    clean_is_correct_blocks: list[np.ndarray] = []

    def maybe_apply_final_norm(hidden: torch.Tensor, layer_index_0based: int) -> torch.Tensor:
        if final_norm is None:
            return hidden
        if layer_index_0based < (num_layers - 1):
            return final_norm(hidden)
        if last_layer_needs_final_norm:
            return final_norm(hidden)
        return hidden

    for start in tqdm(
        range(0, len(frame), batch_size),
        total=int(math.ceil(len(frame) / batch_size)),
        desc=f"{split_name} hidden extraction",
    ):
        batch_df = frame.iloc[start:start + batch_size].reset_index(drop=True)
        batch_cpu = encode_prompts(batch_df["text"].tolist(), tok, max_seq_len)
        decision_pos = batch_cpu.pop("decision_pos")
        prompt_token_count = batch_cpu.pop("prompt_token_count")
        batch = {key: value.to(input_device) for key, value in batch_cpu.items()}
        decision_pos = decision_pos.to(input_device)
        true_choice_idx = torch.tensor(batch_df["correct_idx"].tolist(), dtype=torch.long)

        with torch.inference_mode():
            out = model(**batch, output_hidden_states=True, return_dict=True, use_cache=False)

        row_idx = torch.arange(len(batch_df), device=input_device)
        per_layer_hidden = []
        for layer_index in range(num_layers):
            hidden = out.hidden_states[layer_index + 1][row_idx, decision_pos].detach().cpu().to(torch.float16)
            per_layer_hidden.append(hidden)

        final_raw = out.hidden_states[-1][row_idx, decision_pos]
        final_readout = maybe_apply_final_norm(final_raw, num_layers - 1)
        final_choice_logits = torch.matmul(
            final_readout.to(answer_choice_weight.dtype),
            answer_choice_weight.T,
        ).float()
        final_choice_probs = torch.softmax(final_choice_logits, dim=-1).detach().cpu()

        hidden_blocks.append(torch.stack(per_layer_hidden, dim=1))
        final_choice_prob_blocks.append(final_choice_probs)
        true_choice_idx_blocks.append(true_choice_idx)

        final_logits = out.logits[row_idx, decision_pos].float().detach().cpu()
        masked_logits = final_logits.clone()
        masked_logits[:, answer_id_tensor_cpu] = -torch.inf
        best_non_choice_logit, best_non_choice_token_id = torch.max(masked_logits, dim=-1)
        choice_logits = final_logits.index_select(1, answer_id_tensor_cpu)
        predicted_choice_idx = torch.argmax(choice_logits, dim=-1)
        is_correct = predicted_choice_idx.eq(true_choice_idx)

        clean_choice_logits_blocks.append(choice_logits.numpy().astype(np.float32))
        clean_best_non_choice_logit_blocks.append(best_non_choice_logit.numpy().astype(np.float32))
        clean_best_non_choice_token_id_blocks.append(best_non_choice_token_id.numpy().astype(np.int64))
        clean_is_correct_blocks.append(is_correct.numpy())

        for batch_index, row in batch_df.iterrows():
            example_rows.append(
                {
                    "example_id": row["example_id"],
                    "split": split_name,
                    "text": row["text"],
                    "answerKey": row["answerKey"],
                    "correct_idx": int(row["correct_idx"]),
                    "prompt_len_chars": int(row["prompt_len_chars"]),
                    "prompt_token_count": int(prompt_token_count[batch_index].item()),
                    "decision_pos": int(decision_pos[batch_index].item()),
                }
            )
            clean_output_rows.append(
                {
                    "example_id": row["example_id"],
                    "true_choice_idx": int(row["correct_idx"]),
                    "clean_logit_A": float(choice_logits[batch_index, 0].item()),
                    "clean_logit_B": float(choice_logits[batch_index, 1].item()),
                    "clean_logit_C": float(choice_logits[batch_index, 2].item()),
                    "clean_logit_D": float(choice_logits[batch_index, 3].item()),
                    "clean_logit_E": float(choice_logits[batch_index, 4].item()),
                    "best_non_choice_token_id": int(best_non_choice_token_id[batch_index].item()),
                    "best_non_choice_logit": float(best_non_choice_logit[batch_index].item()),
                }
            )

    return {
        "hidden": torch.cat(hidden_blocks, dim=0),
        "final_choice_probs": torch.cat(final_choice_prob_blocks, dim=0),
        "true_choice_idx": torch.cat(true_choice_idx_blocks, dim=0),
        "clean_choice_logits": np.concatenate(clean_choice_logits_blocks, axis=0),
        "clean_best_non_choice_logit": np.concatenate(clean_best_non_choice_logit_blocks, axis=0),
        "clean_best_non_choice_token_id": np.concatenate(clean_best_non_choice_token_id_blocks, axis=0),
        "clean_is_correct": np.concatenate(clean_is_correct_blocks, axis=0).astype(bool),
        "example_rows": example_rows,
        "clean_output_rows": clean_output_rows,
    }


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
    feature_names: list[str],
    full_logits: torch.Tensor,
    choice_logits: torch.Tensor,
    vocab_size: int,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for feature_name in feature_names:
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
    feature_names: list[str],
    maybe_apply_final_norm,
    lm_head_weight: torch.Tensor,
    answer_id_tensor_lm_head: torch.Tensor,
    input_device: torch.device,
    vocab_size: int,
    active_layer_numbers: list[int],
) -> pd.DataFrame:
    hidden = cache["hidden"]
    clean_is_correct = cache["clean_is_correct"]
    example_rows = pd.DataFrame(cache["example_rows"])[["example_id", "split"]]
    rows: list[dict[str, object]] = []

    layer_indices = [layer_number - 1 for layer_number in active_layer_numbers]
    for layer_index in tqdm(layer_indices, desc=f"{split_name} feature extraction"):
        layer_hidden = hidden[:, layer_index, :]
        feature_blocks: dict[str, list[np.ndarray]] = {feature_name: [] for feature_name in feature_names}

        for start in range(0, layer_hidden.shape[0], READOUT_BATCH_SIZE):
            end = start + READOUT_BATCH_SIZE
            hidden_batch = layer_hidden[start:end].to(input_device)
            readout = maybe_apply_final_norm(hidden_batch.float(), layer_index)
            full_logits = torch.matmul(readout.to(lm_head_weight.dtype), lm_head_weight.T).float()
            choice_logits = full_logits.index_select(1, answer_id_tensor_lm_head)
            feature_batch = summarize_logit_features(
                feature_names=feature_names,
                full_logits=full_logits,
                choice_logits=choice_logits,
                vocab_size=vocab_size,
            )
            for feature_name, values in feature_batch.items():
                feature_blocks[feature_name].append(values)

        feature_values_by_name = {
            feature_name: np.concatenate(blocks, axis=0)
            for feature_name, blocks in feature_blocks.items()
        }

        example_ids = example_rows["example_id"].tolist()
        for example_index, example_id in enumerate(example_ids):
            final_error = int(not bool(clean_is_correct[example_index]))
            for feature_name in feature_names:
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

    return pd.DataFrame(rows)


def empirical_cdf(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    sorted_values = np.sort(values)
    return np.searchsorted(sorted_values, grid, side="right") / float(sorted_values.shape[0])


def compute_ks_statistic(correct_values: np.ndarray, incorrect_values: np.ndarray) -> float:
    pooled_grid = np.sort(np.unique(np.concatenate([correct_values, incorrect_values], axis=0)))
    if pooled_grid.size == 0:
        return 0.0
    cdf_good = empirical_cdf(correct_values, pooled_grid)
    cdf_bad = empirical_cdf(incorrect_values, pooled_grid)
    return float(np.max(np.abs(cdf_good - cdf_bad)))


def build_separation_summary(feature_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped_keys = sorted(
        feature_df[["feature_name", "layer_number"]].drop_duplicates().itertuples(index=False, name=None)
    )
    for feature_name, layer_number in grouped_keys:
        part = feature_df.loc[
            feature_df["feature_name"].eq(feature_name)
            & feature_df["layer_number"].eq(layer_number)
        ].copy()
        correct_values = part.loc[part["clean_is_correct"], "feature_value"].to_numpy(dtype=float)
        incorrect_values = part.loc[~part["clean_is_correct"], "feature_value"].to_numpy(dtype=float)
        rows.append(
            {
                "feature_name": feature_name,
                "layer_number": int(layer_number),
                "ks_statistic": compute_ks_statistic(correct_values, incorrect_values),
                "correct_mean": float(correct_values.mean()),
                "incorrect_mean": float(incorrect_values.mean()),
                "correct_median": float(np.median(correct_values)),
                "incorrect_median": float(np.median(incorrect_values)),
                "correct_std": float(correct_values.std(ddof=0)),
                "incorrect_std": float(incorrect_values.std(ddof=0)),
                "n_correct": int(correct_values.shape[0]),
                "n_incorrect": int(incorrect_values.shape[0]),
            }
        )
    return pd.DataFrame(rows)


def select_top_k_layers_by_feature(
    separation_df: pd.DataFrame,
    *,
    feature_names: list[str],
    top_k: int,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for feature_name in feature_names:
        part = separation_df.loc[separation_df["feature_name"].eq(feature_name)].copy()
        part = part.sort_values(["ks_statistic", "layer_number"], ascending=[False, True]).head(top_k).copy()
        part["selection_rank"] = np.arange(1, len(part) + 1)
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


def silverman_bandwidth(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    n = values.shape[0]
    if n <= 1:
        return 0.1
    std = float(values.std(ddof=1)) if n > 1 else 0.0
    iqr = float(np.subtract(*np.percentile(values, [75, 25])))
    robust = iqr / 1.34 if iqr > 0 else std
    scale = min(x for x in [std, robust] if x > 0) if any(x > 0 for x in [std, robust]) else 1.0
    bw = 0.9 * scale * (n ** (-1 / 5))
    return float(max(bw, 1e-3))


def fit_kde(values: np.ndarray, bandwidth: float) -> KernelDensity:
    model = KernelDensity(kernel="gaussian", bandwidth=float(bandwidth))
    model.fit(values.reshape(-1, 1))
    return model


def gaussian_kernel_1d(sigma_grid: float) -> np.ndarray:
    if sigma_grid <= 1e-8:
        return np.array([1.0], dtype=np.float64)
    radius = max(1, int(math.ceil(3.0 * sigma_grid)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma_grid) ** 2)
    kernel /= np.sum(kernel)
    return kernel


def smooth_1d(values: np.ndarray, sigma_grid: float) -> np.ndarray:
    kernel = gaussian_kernel_1d(float(sigma_grid))
    if kernel.shape[0] == 1:
        return values.copy()
    pad = kernel.shape[0] // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def parse_float_list(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("No float values provided")
    if any(value <= 0 for value in values):
        raise ValueError("All values must be positive")
    return values


def build_distribution_models(
    *,
    fit_feature_df: pd.DataFrame,
    selected_layers_df: pd.DataFrame,
    log_ratio_threshold: float,
    grid_points: int,
) -> tuple[dict[tuple[str, int], dict[str, object]], pd.DataFrame]:
    models: dict[tuple[str, int], dict[str, object]] = {}
    rows: list[dict[str, object]] = []

    for feature_name, layer_number in selected_layers_df[["feature_name", "layer_number"]].itertuples(index=False):
        part = fit_feature_df.loc[
            fit_feature_df["feature_name"].eq(feature_name)
            & fit_feature_df["layer_number"].eq(layer_number)
        ].copy()
        correct_values = part.loc[part["clean_is_correct"], "feature_value"].to_numpy(dtype=float)
        incorrect_values = part.loc[~part["clean_is_correct"], "feature_value"].to_numpy(dtype=float)
        pooled = np.concatenate([correct_values, incorrect_values], axis=0)

        if np.unique(pooled).shape[0] <= 1:
            pooled = pooled + np.random.default_rng(42).normal(0.0, KDE_JITTER_SCALE, size=pooled.shape[0])
            correct_values = pooled[: correct_values.shape[0]]
            incorrect_values = pooled[correct_values.shape[0] :]

        bandwidth = KDE_BANDWIDTH_MULTIPLIER * silverman_bandwidth(pooled)
        kde_good = fit_kde(correct_values, bandwidth)
        kde_bad = fit_kde(incorrect_values, bandwidth)

        data_min = float(np.min(pooled))
        data_max = float(np.max(pooled))
        span = max(data_max - data_min, bandwidth * 6.0, 1e-3)
        pad = max(span * 0.15, bandwidth * 3.0)
        grid = np.linspace(data_min - pad, data_max + pad, grid_points, dtype=np.float64)

        log_p_good = kde_good.score_samples(grid.reshape(-1, 1))
        log_p_bad = kde_bad.score_samples(grid.reshape(-1, 1))
        log_ratio_raw = log_p_good - log_p_bad
        grid_step = float(grid[1] - grid[0]) if grid.shape[0] > 1 else 1.0
        sigma_x = max(bandwidth * LOG_RATIO_SMOOTHING_SIGMA_BANDWIDTHS, grid_step)
        sigma_grid = sigma_x / max(grid_step, 1e-12)
        log_ratio = smooth_1d(log_ratio_raw, sigma_grid)

        support_low = float(np.quantile(pooled, SUPPORT_LOWER_QUANTILE))
        support_high = float(np.quantile(pooled, SUPPORT_UPPER_QUANTILE))
        supported_mask = (grid >= support_low) & (grid <= support_high)

        good_mask = supported_mask & (log_ratio >= log_ratio_threshold)
        bad_mask = supported_mask & (log_ratio <= -log_ratio_threshold)
        if not bool(good_mask.any()):
            supported_indices = np.where(supported_mask)[0]
            candidate_indices = supported_indices if supported_indices.size > 0 else np.arange(grid.shape[0])
            max_idx = int(candidate_indices[np.argmax(log_ratio[candidate_indices])])
            good_mask[max_idx] = True

        region_label = np.full(grid.shape[0], "unsupported", dtype=object)
        region_label[supported_mask] = "neutral"
        region_label[bad_mask] = "bad"
        region_label[good_mask] = "good"

        models[(feature_name, int(layer_number))] = {
            "grid": grid,
            "log_p_good": log_p_good,
            "log_p_bad": log_p_bad,
            "log_ratio": log_ratio,
            "log_ratio_raw": log_ratio_raw,
            "region_label": region_label,
            "good_mask": good_mask,
            "supported_mask": supported_mask,
            "bandwidth": bandwidth,
            "support_low": support_low,
            "support_high": support_high,
            "smoothing_sigma_grid": float(sigma_grid),
        }

        for idx in range(grid.shape[0]):
            rows.append(
                {
                    "feature_name": feature_name,
                    "layer_number": int(layer_number),
                    "grid_x": float(grid[idx]),
                    "log_p_good": float(log_p_good[idx]),
                    "log_p_bad": float(log_p_bad[idx]),
                    "p_good": float(math.exp(log_p_good[idx])),
                    "p_bad": float(math.exp(log_p_bad[idx])),
                    "log_density_ratio_raw": float(log_ratio_raw[idx]),
                    "log_density_ratio": float(log_ratio[idx]),
                    "region_label": str(region_label[idx]),
                    "is_supported": bool(supported_mask[idx]),
                    "bandwidth": float(bandwidth),
                    "support_low": support_low,
                    "support_high": support_high,
                    "smoothing_sigma_grid": float(sigma_grid),
                }
            )

    return models, pd.DataFrame(rows)


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
    full_logits = torch.matmul(readout.to(lm_head_weight.dtype), lm_head_weight.T).float()
    choice_logits = full_logits.index_select(1, answer_id_tensor_lm_head)
    return compute_feature_tensor(
        feature_name=feature_name,
        full_logits=full_logits,
        choice_logits=choice_logits,
        vocab_size=vocab_size,
    )


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
