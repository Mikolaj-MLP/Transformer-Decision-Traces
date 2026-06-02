from __future__ import annotations

import argparse
import gc
import json
import math
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
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.legacy.cli.extract_csqa_base_layerwise import (
    AffineTranslator,
    build_answer_token_ids,
    build_choice_token_spans,
    encode_prompts,
    get_decoder_layers,
    get_final_norm_module,
    true_choice_logit_minus_best_other_choice_logit,
)
from src.data.load_csqa import load_csqa


LETTERS = ["A", "B", "C", "D", "E"]
EXTRACT_BATCH_SIZE = 4
READOUT_BATCH_SIZE = 32
ATTENTION_BATCH_SIZE = 1
TUNED_LENS_BATCH_SIZE = 64
TUNED_LENS_EPOCHS = 2
TUNED_LENS_LR = 1e-3
TUNED_LENS_WEIGHT_DECAY = 1e-5


def now_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def repo_root() -> Path:
    return REPO_ROOT


def slugify_model_id(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", model_id).strip("-")


def resolve_out_dir(out_dir: str | None, model_id: str) -> Path:
    root = repo_root()
    if out_dir is None:
        run_name = f"{now_id()}_{slugify_model_id(model_id)}_csqa_base_train_layerwise"
        return root / "data" / "generated" / "csqa_base_train_layerwise" / run_name
    path = Path(out_dir)
    return path if path.is_absolute() else (root / path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--max-seq-len", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_rows = load_csqa(split="train", limit=None).copy()
    train_rows["prompt_len_chars"] = train_rows["text"].str.len()

    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            model_dtype = torch.bfloat16
        else:
            model_dtype = torch.float16
        device_map = "auto"
    else:
        model_dtype = torch.float32
        device_map = None

    tok = AutoTokenizer.from_pretrained(args.model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=model_dtype,
        device_map=device_map,
        attn_implementation="eager",
    )
    model.eval()

    input_device = model.device if hasattr(model, "device") else next(model.parameters()).device
    decoder_layers = get_decoder_layers(model)
    final_norm = get_final_norm_module(model)
    num_layers = len(decoder_layers)
    hidden_size = int(model.lm_head.weight.shape[1])

    answer_token_ids = build_answer_token_ids(tok)
    answer_ids = [answer_token_ids[letter] for letter in LETTERS]
    answer_id_tensor_cpu = torch.tensor(answer_ids, dtype=torch.long)

    lm_head_weight = model.lm_head.weight.detach()
    lm_head_device = lm_head_weight.device
    answer_id_tensor_lm_head = answer_id_tensor_cpu.to(lm_head_device)
    answer_choice_weight = lm_head_weight.index_select(0, answer_id_tensor_lm_head)

    probe_cpu = encode_prompts(train_rows["text"].head(1).tolist(), tok, args.max_seq_len)
    probe_pos = int(probe_cpu["decision_pos"][0].item())
    probe_batch = {k: v.to(input_device) for k, v in probe_cpu.items() if k not in ["decision_pos", "prompt_token_count"]}
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

    def summarize_hidden_readout(
        hidden_batch: torch.Tensor,
        layer_index_0based: int,
        true_choice_idx_batch: torch.Tensor,
        final_choice_prob_batch: torch.Tensor,
        *,
        always_apply_final_norm_if_available: bool,
    ) -> dict[str, torch.Tensor]:
        readout = hidden_batch.float()
        if always_apply_final_norm_if_available:
            if final_norm is not None:
                readout = final_norm(readout)
        else:
            readout = maybe_apply_final_norm(readout, layer_index_0based)

        full_logits = torch.matmul(
            readout.to(lm_head_weight.dtype),
            lm_head_weight.T,
        ).float()
        full_log_probs = torch.log_softmax(full_logits, dim=-1)
        full_probs = torch.exp(full_log_probs)
        full_entropy = -(full_probs * full_log_probs).sum(dim=-1)
        predicted_vocab_token_id = torch.argmax(full_logits, dim=-1)

        masked_logits = full_logits.clone()
        masked_logits[:, answer_id_tensor_lm_head] = -torch.inf
        best_non_choice_logit, best_non_choice_token_id = torch.max(masked_logits, dim=-1)

        choice_logits = full_logits.index_select(1, answer_id_tensor_lm_head)
        choice_log_probs = torch.log_softmax(choice_logits, dim=-1)
        choice_probs = torch.exp(choice_log_probs)
        predicted_choice_index = torch.argmax(choice_logits, dim=-1)
        sorted_choice_logits = torch.sort(choice_logits, dim=-1, descending=True).values
        choice_gap = sorted_choice_logits[:, 0] - sorted_choice_logits[:, 1]
        row_idx = torch.arange(choice_logits.shape[0], device=choice_logits.device)
        true_choice_prob = choice_probs[row_idx, true_choice_idx_batch]
        true_choice_rank = 1 + (
            choice_logits > choice_logits[row_idx, true_choice_idx_batch][:, None]
        ).sum(dim=-1)
        true_choice_gap = true_choice_logit_minus_best_other_choice_logit(
            choice_logits,
            true_choice_idx_batch,
        )
        choice_kl_to_final = torch.sum(
            final_choice_prob_batch
            * (
                torch.log(torch.clamp(final_choice_prob_batch, min=1e-12))
                - choice_log_probs
            ),
            dim=-1,
        )
        choice_entropy = -(choice_probs * choice_log_probs).sum(dim=-1)

        return {
            "predicted_choice_idx": predicted_choice_index,
            "predicted_choice_is_correct": predicted_choice_index.eq(true_choice_idx_batch),
            "predicted_vocab_token_id": predicted_vocab_token_id,
            "best_non_choice_token_id": best_non_choice_token_id,
            "best_non_choice_logit": best_non_choice_logit,
            "full_vocab_entropy": full_entropy,
            "answer_choice_entropy": choice_entropy,
            "answer_choice_top1_top2_logit_gap": choice_gap,
            "true_answer_logit_minus_best_other_choice_logit": true_choice_gap,
            "true_answer_probability_within_choices": true_choice_prob,
            "true_answer_rank_within_choices": true_choice_rank,
            "choice_kl_to_final": choice_kl_to_final,
            "logit_A": choice_logits[:, 0],
            "logit_B": choice_logits[:, 1],
            "logit_C": choice_logits[:, 2],
            "logit_D": choice_logits[:, 3],
            "logit_E": choice_logits[:, 4],
        }

    def extract_split_cache(frame: pd.DataFrame):
        hidden_blocks: list[torch.Tensor] = []
        final_choice_prob_blocks: list[torch.Tensor] = []
        true_choice_idx_blocks: list[torch.Tensor] = []
        example_rows: list[dict[str, object]] = []
        final_output_rows: list[dict[str, object]] = []

        for start in range(0, len(frame), EXTRACT_BATCH_SIZE):
            batch_df = frame.iloc[start:start + EXTRACT_BATCH_SIZE].reset_index(drop=True)
            batch_cpu = encode_prompts(batch_df["text"].tolist(), tok, args.max_seq_len)
            decision_pos = batch_cpu.pop("decision_pos")
            prompt_token_count = batch_cpu.pop("prompt_token_count")
            batch = {k: v.to(input_device) for k, v in batch_cpu.items()}
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

            for batch_index, row in batch_df.iterrows():
                example_rows.append(
                    {
                        "example_id": row["example_id"],
                        "text": row["text"],
                        "answerKey": row["answerKey"],
                        "correct_idx": int(row["correct_idx"]),
                        "prompt_len_chars": int(row["prompt_len_chars"]),
                        "prompt_token_count": int(prompt_token_count[batch_index].item()),
                        "decision_pos": int(decision_pos[batch_index].item()),
                    }
                )

            final_logits = out.logits[row_idx, decision_pos].float().detach().cpu()
            masked_logits = final_logits.clone()
            masked_logits[:, answer_id_tensor_cpu] = -torch.inf
            best_non_choice_logit, best_non_choice_token_id = torch.max(masked_logits, dim=-1)

            choice_logits = final_logits.index_select(1, answer_id_tensor_cpu)
            for batch_index, row in batch_df.iterrows():
                final_output_rows.append(
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
            "example_id": frame["example_id"].tolist(),
            "answerKey": frame["answerKey"].astype(str).tolist(),
            "example_rows": example_rows,
            "final_output_rows": final_output_rows,
        }

    train_cache = extract_split_cache(train_rows)
    teacher_choice_probs_train = train_cache["final_choice_probs"].float()

    train_device = input_device
    lenses: list[AffineTranslator | None] = []
    lens_state_dicts: dict[int, dict[str, torch.Tensor]] = {}
    train_history_rows: list[dict[str, object]] = []

    for layer_index in range(num_layers):
        if layer_index == (num_layers - 1):
            lenses.append(None)
            continue

        lens = AffineTranslator(hidden_size).to(train_device)
        optimizer = torch.optim.AdamW(
            lens.parameters(),
            lr=TUNED_LENS_LR,
            weight_decay=TUNED_LENS_WEIGHT_DECAY,
        )

        dataset = TensorDataset(train_cache["hidden"][:, layer_index, :], teacher_choice_probs_train)
        dataloader = DataLoader(dataset, batch_size=TUNED_LENS_BATCH_SIZE, shuffle=True)

        for epoch in range(1, TUNED_LENS_EPOCHS + 1):
            epoch_losses = []
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

            train_history_rows.append(
                {
                    "layer_number": layer_index + 1,
                    "epoch": epoch,
                    "mean_kl_loss": float(np.mean(epoch_losses)),
                }
            )

        lens.eval()
        lens_cpu = lens.cpu()
        lenses.append(lens_cpu)
        lens_state_dicts[layer_index + 1] = {k: v.detach().cpu() for k, v in lens_cpu.state_dict().items()}
        del lens
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    train_history_df = pd.DataFrame(train_history_rows)

    train_examples_df = pd.DataFrame(train_cache["example_rows"]).drop_duplicates(subset=["example_id"]).reset_index(drop=True)
    train_clean_final_df = pd.DataFrame(train_cache["final_output_rows"])

    choice_span_map = {
        row["example_id"]: build_choice_token_spans(
            text=row["text"],
            tok=tok,
            max_seq_len=args.max_seq_len,
        )
        for _, row in train_rows[["example_id", "text"]].iterrows()
    }

    attention_rows: list[dict[str, object]] = []
    for start in range(0, len(train_rows), ATTENTION_BATCH_SIZE):
        batch_df = train_rows.iloc[start:start + ATTENTION_BATCH_SIZE].reset_index(drop=True)
        batch_cpu = encode_prompts(batch_df["text"].tolist(), tok, args.max_seq_len)
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
            choice_spans = choice_span_map[example_id]

            for layer_index in range(num_layers):
                attn_row_full = out.attentions[layer_index][batch_index, :, pos, :].float().detach().cpu().numpy()
                attn_row_valid = attn_row_full[:, pad_offset:pad_offset + valid_len]
                squared_mass = np.sum(attn_row_valid ** 2, axis=1)
                if valid_len > 1:
                    head_renyi2_entropy = -np.log(np.clip(squared_mass, 1e-12, None)) / math.log(valid_len)
                else:
                    head_renyi2_entropy = np.zeros(attn_row_full.shape[0], dtype=np.float32)

                choice_mass_by_head = np.zeros((attn_row_full.shape[0], len(LETTERS)), dtype=np.float32)
                for choice_index, (span_start, span_end) in enumerate(choice_spans):
                    abs_start = pad_offset + span_start
                    abs_end = pad_offset + span_end
                    if abs_end > abs_start:
                        choice_mass_by_head[:, choice_index] = attn_row_full[:, abs_start:abs_end].sum(axis=1)

                for head_index in range(choice_mass_by_head.shape[0]):
                    attention_rows.append(
                        {
                            "example_id": example_id,
                            "layer_number": layer_index + 1,
                            "head_number": head_index + 1,
                            "head_renyi2_entropy_normalized": float(head_renyi2_entropy[head_index]),
                            "attention_choice_mass_A": float(choice_mass_by_head[head_index, 0]),
                            "attention_choice_mass_B": float(choice_mass_by_head[head_index, 1]),
                            "attention_choice_mass_C": float(choice_mass_by_head[head_index, 2]),
                            "attention_choice_mass_D": float(choice_mass_by_head[head_index, 3]),
                            "attention_choice_mass_E": float(choice_mass_by_head[head_index, 4]),
                        }
                    )

    train_attention_outputs_df = pd.DataFrame(attention_rows)

    layerwise_rows: list[dict[str, object]] = []
    true_choice_idx_train = train_cache["true_choice_idx"].numpy().astype(np.int64)
    final_choice_probs_train = train_cache["final_choice_probs"].numpy().astype(np.float32)
    lenses_eval = [lens.to(train_device) if lens is not None else None for lens in lenses]

    for method_name in ["direct_readout", "tuned_lens"]:
        for layer_index in range(num_layers):
            for start in range(0, len(train_rows), READOUT_BATCH_SIZE):
                stop = min(start + READOUT_BATCH_SIZE, len(train_rows))
                hidden_batch = train_cache["hidden"][start:stop, layer_index, :].to(train_device).float()

                if method_name == "direct_readout":
                    readout = maybe_apply_final_norm(hidden_batch, layer_index)
                else:
                    if layer_index == (num_layers - 1):
                        readout = maybe_apply_final_norm(hidden_batch, layer_index)
                    else:
                        readout = lenses_eval[layer_index](hidden_batch)

                metrics = summarize_hidden_readout(
                    readout,
                    layer_index,
                    torch.tensor(true_choice_idx_train[start:stop], dtype=torch.long, device=train_device),
                    torch.as_tensor(final_choice_probs_train[start:stop], device=train_device),
                    always_apply_final_norm_if_available=False,
                )

                metrics_cpu = {k: v.detach().cpu() for k, v in metrics.items()}
                for batch_index in range(stop - start):
                    global_index = start + batch_index
                    true_idx = int(true_choice_idx_train[global_index])
                    layerwise_rows.append(
                        {
                            "example_id": train_cache["example_id"][global_index],
                            "layer_number": layer_index + 1,
                            "readout_method": method_name,
                            "true_choice_idx": true_idx,
                            "best_non_choice_token_id": int(metrics_cpu["best_non_choice_token_id"][batch_index].item()),
                            "best_non_choice_logit": float(metrics_cpu["best_non_choice_logit"][batch_index].item()),
                            "logit_A": float(metrics_cpu["logit_A"][batch_index].item()),
                            "logit_B": float(metrics_cpu["logit_B"][batch_index].item()),
                            "logit_C": float(metrics_cpu["logit_C"][batch_index].item()),
                            "logit_D": float(metrics_cpu["logit_D"][batch_index].item()),
                            "logit_E": float(metrics_cpu["logit_E"][batch_index].item()),
                        }
                    )

    train_layerwise_outputs_df = pd.DataFrame(layerwise_rows)

    for lens in lenses_eval:
        if lens is not None:
            lens.cpu()

    output_dir = resolve_out_dir(args.out_dir, args.model_id)
    tmp_dir = output_dir.parent / f"_tmp_{output_dir.name}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=False)

    train_examples_df.to_parquet(tmp_dir / "train_examples.parquet", index=False)
    train_clean_final_df.to_parquet(tmp_dir / "train_clean_final_outputs.parquet", index=False)
    train_layerwise_outputs_df.to_parquet(tmp_dir / "train_layerwise_outputs.parquet", index=False)
    train_attention_outputs_df.to_parquet(tmp_dir / "train_attention_outputs.parquet", index=False)
    train_history_df.to_parquet(tmp_dir / "train_tuned_lens_training_history.parquet", index=False)
    torch.save(
        {
            "model_id": args.model_id,
            "num_layers": num_layers,
            "hidden_size": hidden_size,
            "state_dict_by_layer_number": lens_state_dicts,
        },
        tmp_dir / "train_tuned_lens_state.pt",
    )

    run_config = {
        "model_id": args.model_id,
        "train_split": "train",
        "max_seq_len": int(args.max_seq_len),
        "seed": int(args.seed),
        "extract_batch_size": EXTRACT_BATCH_SIZE,
        "readout_batch_size": READOUT_BATCH_SIZE,
        "attention_batch_size": ATTENTION_BATCH_SIZE,
        "tuned_lens_batch_size": TUNED_LENS_BATCH_SIZE,
        "tuned_lens_epochs": TUNED_LENS_EPOCHS,
        "tuned_lens_lr": TUNED_LENS_LR,
        "tuned_lens_weight_decay": TUNED_LENS_WEIGHT_DECAY,
        "answer_token_ids": answer_token_ids,
        "num_train_examples": int(len(train_rows)),
        "num_layers": int(num_layers),
        "hidden_size": int(hidden_size),
        "model_dtype": str(model_dtype),
        "last_layer_needs_final_norm": bool(last_layer_needs_final_norm),
        "tuned_lens_target": "final_answer_choice_distribution",
        "attention_target": "decision_position_per_head_choice_attention_masses_plus_head_entropy",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with (tmp_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    shutil.move(tmp_dir.as_posix(), output_dir.as_posix())
    print(f"[done] wrote {output_dir}")


if __name__ == "__main__":
    main()
