"""Techniczna obsługa modelu, readoutu warstw i forwardów interwencyjnych."""

from __future__ import annotations

import gc
import math
import os
from collections.abc import Callable

import numpy as np
import pandas as pd
import torch
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
from src.score.constants import LETTERS, READOUT_BATCH_SIZE
from src.score.features import summarize_logit_features


def get_input_device(model: AutoModelForCausalLM) -> torch.device:
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    return torch.device("cpu")


def choose_model_dtype_and_device_map() -> tuple[torch.dtype, str | None]:
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16, "auto"
        return torch.float16, "auto"
    return torch.float32, None


def resolve_hf_token() -> str | None:
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        if value := os.getenv(key):
            return value
    return None


def load_model_and_tokenizer(model_id: str) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Załaduj model w precyzji odpowiedniej dla dostępnego urządzenia."""
    model_dtype, device_map = choose_model_dtype_and_device_map()
    token = resolve_hf_token()
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=token,
        dtype=model_dtype,
        device_map=device_map,
        attn_implementation="eager",
    )
    model.eval()
    return model, tokenizer


def prepare_readout_context(
    *,
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    input_device: torch.device,
    num_layers: int,
    max_seq_len: int,
    probe_rows: pd.DataFrame | None = None,
) -> dict[str, object]:
    """Przygotuj readout logitów i ustal sposób normalizacji ostatniej warstwy."""
    final_norm = get_final_norm_module(model)
    lm_head_weight = model.lm_head.weight.detach()
    answer_token_ids = build_answer_token_ids(tok)
    answer_ids = [answer_token_ids[letter] for letter in LETTERS]
    answer_id_tensor_cpu = torch.tensor(answer_ids, dtype=torch.long)
    answer_id_tensor_lm_head = answer_id_tensor_cpu.to(lm_head_weight.device)
    answer_choice_weight = lm_head_weight.index_select(0, answer_id_tensor_lm_head)

    probe_rows = (
        load_csqa(split="validation", limit=1).copy()
        if probe_rows is None
        else probe_rows.iloc[:1].copy()
    )
    probe_cpu = encode_prompts(probe_rows["text"].tolist(), tok, max_seq_len)
    probe_pos = int(probe_cpu["decision_pos"][0].item())
    probe_batch = {
        key: value.to(input_device)
        for key, value in probe_cpu.items()
        if key not in ("decision_pos", "prompt_token_count")
    }
    with torch.inference_mode():
        probe_out = model(
            **probe_batch,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

    raw_last = probe_out.hidden_states[-1][0, probe_pos].float()
    target_choice_logits = probe_out.logits[
        0,
        probe_pos,
        answer_id_tensor_lm_head,
    ].float().detach().cpu()
    raw_choice_logits = torch.mv(answer_choice_weight.detach().float().cpu(), raw_last.detach().cpu())

    if final_norm is not None:
        normed_last = final_norm(raw_last.unsqueeze(0)).squeeze(0)
        normed_choice_logits = torch.mv(
            answer_choice_weight.detach().float().cpu(),
            normed_last.detach().cpu(),
        )
        raw_error = torch.mean(torch.abs(raw_choice_logits - target_choice_logits)).item()
        normed_error = torch.mean(torch.abs(normed_choice_logits - target_choice_logits)).item()
        last_layer_needs_final_norm = bool(normed_error < raw_error)
    else:
        last_layer_needs_final_norm = False

    def maybe_apply_final_norm(hidden: torch.Tensor, layer_index_0based: int) -> torch.Tensor:
        if final_norm is None:
            return hidden
        if layer_index_0based < num_layers - 1 or last_layer_needs_final_norm:
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
    """Wykonaj bazowy forward i zachowaj stany potrzebne dalszym analizom."""
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
        if layer_index_0based < num_layers - 1 or last_layer_needs_final_norm:
            return final_norm(hidden)
        return hidden

    for start in tqdm(
        range(0, len(frame), batch_size),
        total=int(math.ceil(len(frame) / batch_size)),
        desc=f"{split_name} hidden extraction",
    ):
        batch_df = frame.iloc[start : start + batch_size].reset_index(drop=True)
        batch_cpu = encode_prompts(batch_df["text"].tolist(), tok, max_seq_len)
        decision_pos = batch_cpu.pop("decision_pos")
        prompt_token_count = batch_cpu.pop("prompt_token_count")
        batch = {key: value.to(input_device) for key, value in batch_cpu.items()}
        decision_pos = decision_pos.to(input_device)
        true_choice_idx = torch.tensor(batch_df["correct_idx"].tolist(), dtype=torch.long)

        with torch.inference_mode():
            output = model(
                **batch,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )

        row_idx = torch.arange(len(batch_df), device=input_device)
        per_layer_hidden = [
            output.hidden_states[layer_index + 1][row_idx, decision_pos]
            .detach()
            .cpu()
            .to(torch.float16)
            for layer_index in range(num_layers)
        ]

        final_raw = output.hidden_states[-1][row_idx, decision_pos]
        final_readout = maybe_apply_final_norm(final_raw, num_layers - 1)
        final_choice_logits = torch.matmul(
            final_readout.to(answer_choice_weight.dtype),
            answer_choice_weight.T,
        ).float()
        final_choice_probs = torch.softmax(final_choice_logits, dim=-1).detach().cpu()

        hidden_blocks.append(torch.stack(per_layer_hidden, dim=1))
        final_choice_prob_blocks.append(final_choice_probs)
        true_choice_idx_blocks.append(true_choice_idx)

        final_logits = output.logits[row_idx, decision_pos].float().detach().cpu()
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
                    **{
                        f"clean_logit_{letter}": float(choice_logits[batch_index, idx].item())
                        for idx, letter in enumerate(LETTERS)
                    },
                    "best_non_choice_token_id": int(best_non_choice_token_id[batch_index].item()),
                    "best_non_choice_logit": float(best_non_choice_logit[batch_index].item()),
                }
            )

    return {
        "hidden": torch.cat(hidden_blocks),
        "final_choice_probs": torch.cat(final_choice_prob_blocks),
        "true_choice_idx": torch.cat(true_choice_idx_blocks),
        "clean_choice_logits": np.concatenate(clean_choice_logits_blocks).astype(np.float32),
        "clean_best_non_choice_logit": np.concatenate(clean_best_non_choice_logit_blocks).astype(np.float32),
        "clean_best_non_choice_token_id": np.concatenate(clean_best_non_choice_token_id_blocks).astype(np.int64),
        "clean_is_correct": np.concatenate(clean_is_correct_blocks).astype(bool),
        "example_rows": example_rows,
        "clean_output_rows": clean_output_rows,
    }


def build_feature_table(
    *,
    split_name: str,
    cache: dict[str, object],
    feature_names: list[str],
    maybe_apply_final_norm: Callable[[torch.Tensor, int], torch.Tensor],
    lm_head_weight: torch.Tensor,
    answer_id_tensor_lm_head: torch.Tensor,
    input_device: torch.device,
    vocab_size: int,
    active_layer_numbers: list[int],
) -> pd.DataFrame:
    """Oblicz wartości badanych cech dla wszystkich aktywnych warstw."""
    hidden = cache["hidden"]
    clean_is_correct = cache["clean_is_correct"]
    example_ids = pd.DataFrame(cache["example_rows"])["example_id"].tolist()
    rows: list[dict[str, object]] = []

    for layer_index in tqdm(
        [number - 1 for number in active_layer_numbers],
        desc=f"{split_name} feature extraction",
    ):
        layer_hidden = hidden[:, layer_index, :]
        blocks: dict[str, list[np.ndarray]] = {name: [] for name in feature_names}
        for start in range(0, layer_hidden.shape[0], READOUT_BATCH_SIZE):
            hidden_batch = layer_hidden[start : start + READOUT_BATCH_SIZE].to(input_device)
            readout = maybe_apply_final_norm(hidden_batch.float(), layer_index)
            full_logits = torch.matmul(readout.to(lm_head_weight.dtype), lm_head_weight.T).float()
            choice_logits = full_logits.index_select(1, answer_id_tensor_lm_head)
            values = summarize_logit_features(
                feature_names=feature_names,
                full_logits=full_logits,
                choice_logits=choice_logits,
                vocab_size=vocab_size,
            )
            for feature_name, feature_values in values.items():
                blocks[feature_name].append(feature_values)

        values_by_feature = {
            feature_name: np.concatenate(feature_blocks)
            for feature_name, feature_blocks in blocks.items()
        }
        for example_index, example_id in enumerate(example_ids):
            for feature_name in feature_names:
                rows.append(
                    {
                        "example_id": example_id,
                        "split": split_name,
                        "layer_number": layer_index + 1,
                        "feature_name": feature_name,
                        "feature_value": float(values_by_feature[feature_name][example_index]),
                        "final_error": int(not bool(clean_is_correct[example_index])),
                        "clean_is_correct": bool(clean_is_correct[example_index]),
                    }
                )
    return pd.DataFrame(rows)


def build_layerwise_choice_readout_table(
    *,
    cache: dict[str, object],
    split_name: str,
    maybe_apply_final_norm: Callable[[torch.Tensor, int], torch.Tensor],
    lm_head_weight: torch.Tensor,
    answer_id_tensor_lm_head: torch.Tensor,
    input_device: torch.device,
) -> pd.DataFrame:
    """Zapisz logity pięciu odpowiedzi i najlepszego tokenu spoza odpowiedzi."""
    hidden = cache["hidden"]
    row_meta = pd.DataFrame(cache["example_rows"])[["example_id", "split"]].to_dict("records")
    rows: list[dict[str, object]] = []

    for layer_index in tqdm(range(hidden.shape[1]), desc=f"{split_name} layerwise readouts"):
        layer_hidden = hidden[:, layer_index, :]
        for start in range(0, layer_hidden.shape[0], READOUT_BATCH_SIZE):
            end = start + READOUT_BATCH_SIZE
            hidden_batch = layer_hidden[start:end].to(input_device).float()
            readout = maybe_apply_final_norm(hidden_batch, layer_index)
            full_logits = torch.matmul(readout.to(lm_head_weight.dtype), lm_head_weight.T).float()
            masked_logits = full_logits.clone()
            masked_logits[:, answer_id_tensor_lm_head] = -torch.inf
            best_logit, best_token = torch.max(masked_logits, dim=-1)
            choice_logits = full_logits.index_select(1, answer_id_tensor_lm_head)

            choice_cpu = choice_logits.detach().cpu().numpy().astype(np.float32)
            best_logit_cpu = best_logit.detach().cpu().numpy().astype(np.float32)
            best_token_cpu = best_token.detach().cpu().numpy().astype(np.int64)
            for local_index, global_index in enumerate(range(start, min(end, layer_hidden.shape[0]))):
                rows.append(
                    {
                        **row_meta[global_index],
                        "layer_number": layer_index + 1,
                        "best_non_choice_token_id": int(best_token_cpu[local_index]),
                        "best_non_choice_logit": float(best_logit_cpu[local_index]),
                        **{
                            f"logit_{letter}": float(choice_cpu[local_index, idx])
                            for idx, letter in enumerate(LETTERS)
                        },
                    }
                )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return pd.DataFrame(rows)


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
    """Wstrzyknij perturbację przez hook i wykonaj pełny forward modelu."""
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
            output = model(**batch, return_dict=True, use_cache=False)
    finally:
        handle.remove()

    full_logits = select_full_logits_at_decision(output.logits, decision_pos)
    metrics = summarize_decision_logits(
        full_logits,
        true_choice_idx,
        answer_id_tensor_cpu.to(full_logits.device),
    )
    masked_logits = full_logits.clone()
    masked_logits[:, answer_id_tensor_cpu.to(full_logits.device)] = -torch.inf
    return {
        "choice_logits": metrics["choice_logits"].detach().cpu().numpy().astype(np.float32),
        "best_non_choice_logit": metrics["best_non_choice_logit"].detach().cpu().numpy().astype(np.float32),
        "best_non_choice_token_id": torch.argmax(masked_logits, dim=-1).detach().cpu().numpy().astype(np.int64),
    }
