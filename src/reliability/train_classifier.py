# src/reliability/train_classifier.py
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

#  Data helpers (binary valence from GoEmotions
def _map_valence(label_names):
    POS = {"admiration","amusement","approval","caring","desire","excitement",
           "gratitude","joy","love","optimism","pride","relief"}
    NEG = {"anger","annoyance","disappointment","disapproval","disgust","embarrassment",
           "fear","grief","nervousness","realization","remorse","sadness"}
    s = set(label_names)
    pos = len(s & POS) > 0
    neg = len(s & NEG) > 0
    if pos and not neg: return 1
    if neg and not pos: return 0
    return None  # neutral / mixed → drop for binary setup

def _make_split_csvs(processed_parquet: str, out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(processed_parquet)
    rows = []
    for _, r in df.iterrows():
        lab = _map_valence(list(r["labels_names"]))
        if lab is None:
            continue
        rows.append({
            "example_id": r["example_id"],
            "text": r["text"],
            "label": int(lab),
            "split": r["split"],
        })
    d = pd.DataFrame(rows)
    for split in ["train", "validation", "test"]:
        d[d["split"] == split][["example_id","text","label"]].to_csv(
            out / f"goemotions_valence_{split}.csv", index=False
        )
    return out

def _build_hf_dataset(train_csv, val_csv, test_csv, tok):
    def _load(csv_path):
        df = pd.read_csv(csv_path)
        dset = Dataset.from_pandas(df)
        def enc(ex):
            toks = tok(ex["text"], truncation=True, padding="max_length", max_length=128)
            toks["labels"] = ex["label"]
            return toks
        return dset.map(enc, batched=True, remove_columns=dset.column_names)
    return DatasetDict({
        "train": _load(train_csv),
        "validation": _load(val_csv),
        "test": _load(test_csv),
    })

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_go", default="data/processed/go_emotions.parquet", type=str)
    ap.add_argument("--data_dir",    default="data/reliability", type=str)
    ap.add_argument("--model",       default="roberta-base", type=str)
    ap.add_argument("--out_dir",     default="runs/classifier", type=str)
    ap.add_argument("--epochs",      default=2, type=int)
    ap.add_argument("--batch_size",  default=16, type=int)
    args = ap.parse_args()

    # CSV splits (binary valence)
    csv_dir = _make_split_csvs(args.processed_go, args.data_dir)
    train_csv = csv_dir / "goemotions_valence_train.csv"
    val_csv   = csv_dir / "goemotions_valence_validation.csv"
    test_csv  = csv_dir / "goemotions_valence_test.csv"

    # Tokenizer & datasets
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    dsd = _build_hf_dataset(train_csv, val_csv, test_csv, tok)

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)

    # TrainingArguments with version-compat fallback
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    base_kwargs = dict(
        output_dir=str(out_dir / "hf"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=2e-5,
        report_to=[],           # disable W&B/etc by default
        logging_steps=50,
    )
    try:
        tr_args = TrainingArguments(
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            **base_kwargs
        )
    except TypeError:
        # Older transformers that don't accept evaluation/save strategy
        tr_args = TrainingArguments(**base_kwargs)

    def compute_metrics(p):
        preds = p.predictions.argmax(-1)
        labels = p.label_ids
        return {
            "acc": accuracy_score(labels, preds),
            "f1":  f1_score(labels, preds)
        }

    trainer = Trainer(
        model=model,
        args=tr_args,
        train_dataset=dsd["train"],
        eval_dataset=dsd["validation"],
        tokenizer=tok,
        compute_metrics=compute_metrics
    )

    trainer.train()

    # Predictions on validation split -> probabilities, errors
    val_raw = trainer.predict(dsd["validation"])
    val_df = pd.read_csv(val_csv)

    # Stable softmax (NumPy)
    logits = val_raw.predictions
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    probs = exp / exp.sum(axis=1, keepdims=True)
    p_pos = probs[:, 1]                   # P(label=1 positive)
    y_pred = (p_pos >= 0.5).astype(int)
    y_true = val_raw.label_ids
    err    = (y_pred != y_true).astype(int)

    out = pd.DataFrame({
        "example_id": val_df["example_id"].values,
        "text":       val_df["text"].values,
        "y_true":     y_true,
        "y_pred":     y_pred,
        "p_pos":      p_pos,
        "error":      err
    })
    (out_dir / "val_predictions.csv").parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "val_predictions.csv", index=False)
    print("[done] wrote", out_dir / "val_predictions.csv")

if __name__ == "__main__":
    main()
