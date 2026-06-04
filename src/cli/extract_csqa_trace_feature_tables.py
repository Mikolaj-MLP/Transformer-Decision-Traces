from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.csqa.common import (  # noqa: E402
    AffineTranslator,
    build_answer_token_ids,
    build_choice_token_spans,
    encode_prompts,
    get_decoder_layers,
    get_final_norm_module,
)
from src.data.load_mcqa import SUPPORTED_DATASETS, load_mcqa  # noqa: E402
from src.data.load_aqua_rat import (  # noqa: E402
    DEFAULT_AQUA_SPLIT_SEED,
    DEFAULT_AQUA_TEST_SIZE,
    DEFAULT_AQUA_TRAIN_SIZE,
    DEFAULT_AQUA_VALIDATION_SIZE,
)


LETTERS = ["A", "B", "C", "D", "E"]
EXTRACT_BATCH_SIZE = 4
ATTENTION_BATCH_SIZE = 1
READOUT_BATCH_SIZE = 64
TUNED_LENS_BATCH_SIZE = 64
TUNED_LENS_MAX_EPOCHS = 10
TUNED_LENS_PATIENCE = 2
TUNED_LENS_LR = 1e-3
TUNED_LENS_WEIGHT_DECAY = 1e-5


def now_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def repo_root() -> Path:
    return REPO_ROOT


def slugify_model_id(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", model_id).strip("-")


def resolve_out_dir(out_dir: str | None, model_id: str, dataset_name: str) -> Path:
    root = repo_root()
    if out_dir is None:
        run_name = f"{now_id()}_{slugify_model_id(model_id)}_{dataset_name}_trace_feature_tables"
        return root / "data" / "generated" / "trace_feature_tables" / run_name
    path = Path(out_dir)
    return path if path.is_absolute() else (root / path)


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
        probe_rows = load_mcqa("csqa", split="validation", limit=1).copy()
    else:
        probe_rows = probe_rows.iloc[:1].copy()
    probe_cpu = encode_prompts(probe_rows["text"].tolist(), tok, max_seq_len)
    probe_pos = int(probe_cpu["decision_pos"][0].item())
    probe_batch = {
        k: v.to(input_device)
        for k, v in probe_cpu.items()
        if k not in ["decision_pos", "prompt_token_count"]
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


def extract_split_hidden_cache(
    frame: pd.DataFrame,
    *,
    split_name: str,
    tok: AutoTokenizer,
    model: AutoModelForCausalLM,
    input_device: torch.device,
    answer_choice_weight: torch.Tensor,
    num_layers: int,
    max_seq_len: int,
    batch_size: int,
    maybe_apply_final_norm,
) -> dict[str, object]:
    hidden_blocks: list[torch.Tensor] = []
    final_choice_prob_blocks: list[torch.Tensor] = []
    example_rows: list[dict[str, object]] = []

    for start in tqdm(
        range(0, len(frame), batch_size),
        total=int(math.ceil(len(frame) / batch_size)),
        desc=f"{split_name} hidden extraction",
    ):
        batch_df = frame.iloc[start:start + batch_size].reset_index(drop=True)
        batch_cpu = encode_prompts(batch_df["text"].tolist(), tok, max_seq_len)
        decision_pos = batch_cpu.pop("decision_pos")
        prompt_token_count = batch_cpu.pop("prompt_token_count")
        batch = {k: v.to(input_device) for k, v in batch_cpu.items()}
        decision_pos = decision_pos.to(input_device)

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

    return {
        "hidden": torch.cat(hidden_blocks, dim=0),
        "final_choice_probs": torch.cat(final_choice_prob_blocks, dim=0),
        "example_rows": example_rows,
    }


def train_tuned_lenses(
    *,
    train_hidden: torch.Tensor,
    teacher_choice_probs_train: torch.Tensor,
    hidden_size: int,
    num_layers: int,
    answer_choice_weight: torch.Tensor,
    train_device: torch.device,
) -> tuple[list[AffineTranslator | None], dict[int, dict[str, torch.Tensor]], pd.DataFrame]:
    lenses: list[AffineTranslator | None] = []
    lens_state_dicts: dict[int, dict[str, torch.Tensor]] = {}
    train_history_rows: list[dict[str, object]] = []

    for layer_index in tqdm(range(num_layers), desc="tuned lens training"):
        if layer_index == (num_layers - 1):
            lenses.append(None)
            continue

        lens = AffineTranslator(hidden_size).to(train_device)
        optimizer = torch.optim.AdamW(
            lens.parameters(),
            lr=TUNED_LENS_LR,
            weight_decay=TUNED_LENS_WEIGHT_DECAY,
        )
        dataset = TensorDataset(train_hidden[:, layer_index, :], teacher_choice_probs_train)
        dataloader = DataLoader(dataset, batch_size=TUNED_LENS_BATCH_SIZE, shuffle=True)

        best_loss = float("inf")
        epochs_without_improvement = 0

        for epoch in range(1, TUNED_LENS_MAX_EPOCHS + 1):
            epoch_losses: list[float] = []
            lens.train()
            for xb, teacher_probs_b in dataloader:
                xb = xb.to(train_device).float()
                teacher_probs_b = teacher_probs_b.to(train_device)
                pred_readout = lens(xb)
                pred_choice_logits = torch.matmul(
                    pred_readout.to(answer_choice_weight.dtype),
                    answer_choice_weight.T,
                ).float()
                pred_log_probs = torch.log_softmax(pred_choice_logits, dim=-1)
                loss = F.kl_div(pred_log_probs, teacher_probs_b, reduction="batchmean")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.item()))

            mean_epoch_loss = float(np.mean(epoch_losses))
            train_history_rows.append(
                {
                    "layer_number": layer_index + 1,
                    "epoch": epoch,
                    "mean_kl_loss": mean_epoch_loss,
                }
            )

            if mean_epoch_loss < (best_loss - 1e-6):
                best_loss = mean_epoch_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= TUNED_LENS_PATIENCE:
                break

        lens.eval()
        lens_cpu = lens.cpu()
        lenses.append(lens_cpu)
        lens_state_dicts[layer_index + 1] = {k: v.detach().cpu() for k, v in lens_cpu.state_dict().items()}
        del lens
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return lenses, lens_state_dicts, pd.DataFrame(train_history_rows)


def build_layerwise_readout_table(
    *,
    cache: dict[str, object],
    split_name: str,
    lenses: list[AffineTranslator | None],
    final_norm,
    maybe_apply_final_norm,
    lm_head_weight: torch.Tensor,
    answer_id_tensor_lm_head: torch.Tensor,
    input_device: torch.device,
) -> pd.DataFrame:
    hidden = cache["hidden"]
    example_rows = cache["example_rows"]
    row_meta_records = pd.DataFrame(example_rows)[["example_id", "split", "correct_idx"]].to_dict("records")
    rows: list[dict[str, object]] = []

    for layer_index in tqdm(range(hidden.shape[1]), desc=f"{split_name} layerwise readouts"):
        layer_hidden = hidden[:, layer_index, :]

        for readout_method in ["direct_readout", "tuned_lens"]:
            lens = lenses[layer_index]
            if readout_method == "tuned_lens" and lens is not None:
                lens = lens.to(input_device)

            for start in range(0, layer_hidden.shape[0], READOUT_BATCH_SIZE):
                end = start + READOUT_BATCH_SIZE
                hidden_batch = layer_hidden[start:end].to(input_device).float()

                if readout_method == "direct_readout":
                    readout = maybe_apply_final_norm(hidden_batch, layer_index)
                else:
                    if lens is None:
                        readout = maybe_apply_final_norm(hidden_batch, layer_index)
                    else:
                        with torch.inference_mode():
                            readout = lens(hidden_batch)
                        if final_norm is not None:
                            readout = final_norm(readout)

                full_logits = torch.matmul(
                    readout.to(lm_head_weight.dtype),
                    lm_head_weight.T,
                ).float()
                masked_logits = full_logits.clone()
                masked_logits[:, answer_id_tensor_lm_head] = -torch.inf
                best_non_choice_logit, best_non_choice_token_id = torch.max(masked_logits, dim=-1)
                choice_logits = full_logits.index_select(1, answer_id_tensor_lm_head)

                choice_logits_cpu = choice_logits.detach().cpu().numpy().astype(np.float32)
                best_non_choice_logit_cpu = best_non_choice_logit.detach().cpu().numpy().astype(np.float32)
                best_non_choice_token_id_cpu = best_non_choice_token_id.detach().cpu().numpy().astype(np.int64)

                for local_index, global_index in enumerate(range(start, min(end, layer_hidden.shape[0]))):
                    meta = row_meta_records[global_index]
                    rows.append(
                        {
                            "example_id": meta["example_id"],
                            "split": meta["split"],
                            "layer_number": layer_index + 1,
                            "readout_method": readout_method,
                            "true_choice_idx": int(meta["correct_idx"]),
                            "best_non_choice_token_id": int(best_non_choice_token_id_cpu[local_index]),
                            "best_non_choice_logit": float(best_non_choice_logit_cpu[local_index]),
                            "logit_A": float(choice_logits_cpu[local_index, 0]),
                            "logit_B": float(choice_logits_cpu[local_index, 1]),
                            "logit_C": float(choice_logits_cpu[local_index, 2]),
                            "logit_D": float(choice_logits_cpu[local_index, 3]),
                            "logit_E": float(choice_logits_cpu[local_index, 4]),
                        }
                    )

                del hidden_batch
                del readout
                del full_logits
                del masked_logits
                del best_non_choice_logit
                del best_non_choice_token_id
                del choice_logits

            if readout_method == "tuned_lens" and lens is not None:
                lens = lens.cpu()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return pd.DataFrame(rows)


def build_headwise_attention_entropy_table(
    frame: pd.DataFrame,
    *,
    split_name: str,
    tok: AutoTokenizer,
    model: AutoModelForCausalLM,
    input_device: torch.device,
    max_seq_len: int,
    num_layers: int,
) -> pd.DataFrame:
    choice_span_map = {
        row["example_id"]: build_choice_token_spans(
            text=row["text"],
            tok=tok,
            max_seq_len=max_seq_len,
        )
        for _, row in frame[["example_id", "text"]].iterrows()
    }

    rows: list[dict[str, object]] = []
    for start in tqdm(
        range(0, len(frame), ATTENTION_BATCH_SIZE),
        total=int(math.ceil(len(frame) / ATTENTION_BATCH_SIZE)),
        desc=f"{split_name} headwise attention entropy",
    ):
        batch_df = frame.iloc[start:start + ATTENTION_BATCH_SIZE].reset_index(drop=True)
        batch_cpu = encode_prompts(batch_df["text"].tolist(), tok, max_seq_len)
        decision_pos = batch_cpu.pop("decision_pos")
        prompt_token_count = batch_cpu.pop("prompt_token_count")
        padded_seq_len = int(batch_cpu["input_ids"].shape[1])
        batch = {k: v.to(input_device) for k, v in batch_cpu.items()}
        decision_pos = decision_pos.to(input_device)

        with torch.inference_mode():
            out = model(**batch, output_attentions=True, return_dict=True, use_cache=False)

        for batch_index, row in batch_df.iterrows():
            example_id = row["example_id"]
            valid_len = int(prompt_token_count[batch_index].item())
            pad_offset = padded_seq_len - valid_len
            pos = int(decision_pos[batch_index].item())
            _ = choice_span_map[example_id]

            for layer_index in range(num_layers):
                attn_row_full = out.attentions[layer_index][batch_index, :, pos, :].float().detach().cpu().numpy()
                attn_row_valid = attn_row_full[:, pad_offset:pad_offset + valid_len]
                squared_mass = np.sum(attn_row_valid ** 2, axis=1)
                if valid_len > 1:
                    head_renyi2_entropy = -np.log(np.clip(squared_mass, 1e-12, None)) / math.log(valid_len)
                else:
                    head_renyi2_entropy = np.zeros(attn_row_full.shape[0], dtype=np.float32)

                for head_index in range(attn_row_full.shape[0]):
                    rows.append(
                        {
                            "example_id": example_id,
                            "split": split_name,
                            "layer_number": layer_index + 1,
                            "head_number": head_index + 1,
                            "head_renyi2_entropy_normalized": float(head_renyi2_entropy[head_index]),
                        }
                    )

    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="csqa", choices=SUPPORTED_DATASETS)
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--max-seq-len", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--validation-limit", type=int, default=None)
    parser.add_argument("--aqua-train-size", type=int, default=DEFAULT_AQUA_TRAIN_SIZE)
    parser.add_argument("--aqua-validation-size", type=int, default=DEFAULT_AQUA_VALIDATION_SIZE)
    parser.add_argument("--aqua-test-size", type=int, default=DEFAULT_AQUA_TEST_SIZE)
    parser.add_argument("--aqua-split-seed", type=int, default=DEFAULT_AQUA_SPLIT_SEED)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_rows = load_mcqa(
        args.dataset,
        split="train",
        limit=args.train_limit,
        aqua_train_size=args.aqua_train_size,
        aqua_validation_size=args.aqua_validation_size,
        aqua_test_size=args.aqua_test_size,
        aqua_split_seed=args.aqua_split_seed,
    ).copy()
    validation_rows = load_mcqa(
        args.dataset,
        split="validation",
        limit=args.validation_limit,
        aqua_train_size=args.aqua_train_size,
        aqua_validation_size=args.aqua_validation_size,
        aqua_test_size=args.aqua_test_size,
        aqua_split_seed=args.aqua_split_seed,
    ).copy()
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
    hidden_size = int(model.lm_head.weight.shape[1])

    readout_ctx = prepare_readout_context(
        model=model,
        tok=tok,
        input_device=input_device,
        num_layers=num_layers,
        max_seq_len=args.max_seq_len,
        probe_rows=validation_rows,
    )
    final_norm = readout_ctx["final_norm"]
    answer_choice_weight = readout_ctx["answer_choice_weight"]
    answer_id_tensor_lm_head = readout_ctx["answer_id_tensor_lm_head"]
    lm_head_weight = readout_ctx["lm_head_weight"]
    maybe_apply_final_norm = readout_ctx["maybe_apply_final_norm"]

    train_cache = extract_split_hidden_cache(
        train_rows,
        split_name="train",
        tok=tok,
        model=model,
        input_device=input_device,
        answer_choice_weight=answer_choice_weight,
        num_layers=num_layers,
        max_seq_len=args.max_seq_len,
        batch_size=EXTRACT_BATCH_SIZE,
        maybe_apply_final_norm=maybe_apply_final_norm,
    )
    validation_cache = extract_split_hidden_cache(
        validation_rows,
        split_name="validation",
        tok=tok,
        model=model,
        input_device=input_device,
        answer_choice_weight=answer_choice_weight,
        num_layers=num_layers,
        max_seq_len=args.max_seq_len,
        batch_size=EXTRACT_BATCH_SIZE,
        maybe_apply_final_norm=maybe_apply_final_norm,
    )

    lenses, lens_state_dicts, tuned_lens_history_df = train_tuned_lenses(
        train_hidden=train_cache["hidden"],
        teacher_choice_probs_train=train_cache["final_choice_probs"].float(),
        hidden_size=hidden_size,
        num_layers=num_layers,
        answer_choice_weight=answer_choice_weight,
        train_device=input_device,
    )

    layerwise_train_df = build_layerwise_readout_table(
        cache=train_cache,
        split_name="train",
        lenses=lenses,
        final_norm=final_norm,
        maybe_apply_final_norm=maybe_apply_final_norm,
        lm_head_weight=lm_head_weight,
        answer_id_tensor_lm_head=answer_id_tensor_lm_head,
        input_device=input_device,
    )
    layerwise_validation_df = build_layerwise_readout_table(
        cache=validation_cache,
        split_name="validation",
        lenses=lenses,
        final_norm=final_norm,
        maybe_apply_final_norm=maybe_apply_final_norm,
        lm_head_weight=lm_head_weight,
        answer_id_tensor_lm_head=answer_id_tensor_lm_head,
        input_device=input_device,
    )
    layerwise_df = pd.concat([layerwise_train_df, layerwise_validation_df], ignore_index=True)

    headwise_train_df = build_headwise_attention_entropy_table(
        train_rows,
        split_name="train",
        tok=tok,
        model=model,
        input_device=input_device,
        max_seq_len=args.max_seq_len,
        num_layers=num_layers,
    )
    headwise_validation_df = build_headwise_attention_entropy_table(
        validation_rows,
        split_name="validation",
        tok=tok,
        model=model,
        input_device=input_device,
        max_seq_len=args.max_seq_len,
        num_layers=num_layers,
    )
    headwise_df = pd.concat([headwise_train_df, headwise_validation_df], ignore_index=True)

    examples_df = pd.DataFrame(train_cache["example_rows"] + validation_cache["example_rows"])

    out_dir = resolve_out_dir(args.out_dir, args.model_id, args.dataset)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    examples_df.to_parquet(out_dir / "examples.parquet", index=False)
    layerwise_df.to_parquet(out_dir / "layerwise_readouts.parquet", index=False)
    headwise_df.to_parquet(out_dir / "headwise_attention_entropy.parquet", index=False)
    tuned_lens_history_df.to_parquet(out_dir / "tuned_lens_training_history.parquet", index=False)
    torch.save(lens_state_dicts, out_dir / "tuned_lens_state.pt")

    run_config = {
        "dataset": args.dataset,
        "model_id": args.model_id,
        "max_seq_len": int(args.max_seq_len),
        "seed": int(args.seed),
        "train_limit": None if args.train_limit is None else int(args.train_limit),
        "validation_limit": None if args.validation_limit is None else int(args.validation_limit),
        "aqua_train_size": int(args.aqua_train_size),
        "aqua_validation_size": int(args.aqua_validation_size),
        "aqua_test_size": int(args.aqua_test_size),
        "aqua_split_seed": int(args.aqua_split_seed),
        "num_layers": int(num_layers),
        "hidden_size": int(hidden_size),
        "extract_batch_size": EXTRACT_BATCH_SIZE,
        "attention_batch_size": ATTENTION_BATCH_SIZE,
        "readout_batch_size": READOUT_BATCH_SIZE,
        "tuned_lens_batch_size": TUNED_LENS_BATCH_SIZE,
        "tuned_lens_max_epochs": TUNED_LENS_MAX_EPOCHS,
        "tuned_lens_patience": TUNED_LENS_PATIENCE,
        "tuned_lens_lr": TUNED_LENS_LR,
        "tuned_lens_weight_decay": TUNED_LENS_WEIGHT_DECAY,
        "layerwise_table": "layerwise_readouts.parquet",
        "headwise_table": "headwise_attention_entropy.parquet",
    }
    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    print(f"[done] wrote {out_dir}")


if __name__ == "__main__":
    main()
