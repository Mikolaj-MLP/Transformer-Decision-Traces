from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
from transformers import (
    AutoModelForMultipleChoice,
    AutoTokenizer,
    DataCollatorForMultipleChoice,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.data.load_csqa import load_csqa


def now_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


class CSQAMultipleChoiceDataset(Dataset):
    def __init__(self, rows: List[Dict], tokenizer, max_len: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        ex = self.rows[idx]
        question = ex["question"]
        choices = ex["choices"]  # list[str], len=5

        enc = self.tokenizer(
            [question] * len(choices),
            choices,
            truncation=True,
            max_length=self.max_len,
            padding=False,  # dynamic padding in collator
        )

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": int(ex["label_idx"]),
        }


def build_rows(split: str, limit: int | None = None) -> List[Dict]:
    df = load_csqa(split=split, limit=limit).reset_index(drop=True)
    rows = []
    for _, r in df.iterrows():
        question = str(r["text"]).splitlines()[0].removeprefix("Q: ").strip()
        choices = [c["text"] for c in r["csqa_choices"]]
        labels = [c["label"] for c in r["csqa_choices"]]
        label_idx = labels.index(r["answerKey"])
        rows.append(
            {
                "example_id": r["example_id"],
                "question": question,
                "choices": choices,
                "label_idx": int(label_idx),
            }
        )
    return rows


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {"acc": float(accuracy_score(labels, preds))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="roberta-base")
    ap.add_argument("--train_split", type=str, default="train", choices=["train"])
    ap.add_argument("--eval_split", type=str, default="validation", choices=["validation", "test"])
    ap.add_argument("--train_limit", type=int, default=None)
    ap.add_argument("--eval_limit", type=int, default=None)
    ap.add_argument("--max_seq_len", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--eval_steps", type=int, default=500)
    ap.add_argument("--save_steps", type=int, default=500)
    ap.add_argument("--save_total_limit", type=int, default=2)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--out_dir", type=str, default=None)
    args = ap.parse_args()

    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_id = args.model
    run_id = f"{now_id()}_{Path(model_id).name}_csqa_mcq_ft"
    out_dir = Path(args.out_dir or f"checkpoints/{run_id}")
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForMultipleChoice.from_pretrained(model_id, attn_implementation="eager")

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    train_rows = build_rows(args.train_split, args.train_limit)
    eval_rows = build_rows(args.eval_split, args.eval_limit)
    if len(train_rows) == 0 or len(eval_rows) == 0:
        raise RuntimeError("Empty train/eval rows.")

    train_ds = CSQAMultipleChoiceDataset(train_rows, tok, args.max_seq_len)
    eval_ds = CSQAMultipleChoiceDataset(eval_rows, tok, args.max_seq_len)
    collator = DataCollatorForMultipleChoice(tokenizer=tok, pad_to_multiple_of=8)

    use_fp16 = bool(args.fp16 and torch.cuda.is_available())
    use_bf16 = bool(args.bf16 and torch.cuda.is_available())

    targs = TrainingArguments(
        output_dir=str(out_dir),
        overwrite_output_dir=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=50,
        report_to="none",
        load_best_model_at_end=False,
        fp16=use_fp16,
        bf16=use_bf16,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tok,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    train_metrics = trainer.train().metrics
    eval_metrics = trainer.evaluate()

    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))

    meta = {
        "run_id": run_id,
        "task": "csqa_multiple_choice",
        "base_model": model_id,
        "train_split": args.train_split,
        "eval_split": args.eval_split,
        "train_n": len(train_rows),
        "eval_n": len(eval_rows),
        "max_seq_len": args.max_seq_len,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "epochs": args.epochs,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "fp16": use_fp16,
        "bf16": use_bf16,
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "train_metrics": {k: float(v) for k, v in train_metrics.items() if isinstance(v, (int, float))},
        "eval_metrics": {k: float(v) for k, v in eval_metrics.items() if isinstance(v, (int, float))},
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with (out_dir / "finetune_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[done] saved model to {out_dir}")
    print(f"[done] eval metrics: {eval_metrics}")


if __name__ == "__main__":
    main()
