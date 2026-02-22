from __future__ import annotations

import hashlib
from typing import Optional

import pandas as pd
from datasets import load_dataset

from src.data.load_csqa import load_csqa
from src.data.load_text import load_go_emotions, load_ud_ewt


def _stable_id(prefix: str, split: str, idx: int, text: str) -> str:
    h = hashlib.sha1(f"{prefix}::{split}::{idx}::{text}".encode("utf-8")).hexdigest()
    return h[:16]


def load_nexttok_texts(
    dataset: str = "ud_ewt",
    split: str = "validation",
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """
    Return a DataFrame with columns:
      - example_id
      - text
    """
    if dataset == "ud_ewt":
        df = load_ud_ewt(token_level=False)
        out = df[df["split"] == split][["example_id", "text"]].reset_index(drop=True)
    elif dataset == "go_emotions":
        df, _ = load_go_emotions()
        out = df[df["split"] == split][["example_id", "text"]].reset_index(drop=True)
    elif dataset == "csqa":
        df = load_csqa(split=split, limit=limit)
        out = df[["example_id", "text"]].reset_index(drop=True)
    elif dataset in {"wikitext2", "wikitext103"}:
        cfg_name = "wikitext-2-raw-v1" if dataset == "wikitext2" else "wikitext-103-raw-v1"
        split_map = {"train": "train", "validation": "validation", "test": "test"}
        ds = load_dataset("wikitext", cfg_name, split=split_map[split])
        rows = []
        for i, ex in enumerate(ds):
            text = str(ex.get("text", "")).strip()
            if not text:
                continue
            rows.append(
                {
                    "example_id": _stable_id(dataset, split, i, text),
                    "text": text,
                }
            )
            if limit is not None and len(rows) >= int(limit):
                break
        out = pd.DataFrame(rows)
    else:
        raise ValueError("dataset must be one of: ud_ewt, go_emotions, csqa, wikitext2, wikitext103")

    if limit is not None and dataset != "csqa":
        out = out.head(int(limit)).reset_index(drop=True)
    return out.reset_index(drop=True)
