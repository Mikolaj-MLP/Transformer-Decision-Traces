# src/data/load_csqa.py
from __future__ import annotations

import hashlib
from typing import Optional, List, Dict, Any

import pandas as pd
from datasets import load_dataset


def _stable_id(split: str, idx: int, q: str) -> str:
    """
    Deterministic ID independent of HF example 'id' field.
    Using split + row index + question text hash.
    """
    h = hashlib.md5(f"{split}::{idx}::{q}".encode("utf-8")).hexdigest()
    return h


def _format_csqa_prompt(question: str, choices: List[Dict[str, str]]) -> str:
    """
    Format prompt exactly like:
    Q: ...
    Choices:
    A: ...
    ...
    Answer:
    """
    lines = [f"Q: {question}", "Choices:"]
    for ch in choices:
        lines.append(f"{ch['label']}: {ch['text']}")
    lines.append("Answer:")
    return "\n".join(lines)


def load_csqa(
    split: str = "validation",
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """
    Returns DataFrame with columns:
      - example_id: stable hash
      - text: prompt (question + choices + Answer:)
      - answerKey: 'A'..'E'
      - correct_idx: 0..4
      - csqa_choices: list[{'label': 'A', 'text': '...'}, ...] length 5
    """
    ds = load_dataset("commonsense_qa", split=split)

    n = len(ds) if limit is None else min(int(limit), len(ds))
    if limit is not None:
        ds = ds.select(range(n))

    rows: List[Dict[str, Any]] = []
    for i, ex in enumerate(ds):
        q = ex["question"]
        labels = list(ex["choices"]["label"])
        texts = list(ex["choices"]["text"])
        choices = [{"label": l, "text": t} for l, t in zip(labels, texts)]

        if len(choices) != 5:
            raise ValueError(f"CSQA example {i} has {len(choices)} choices (expected 5).")

        ans = ex["answerKey"]
        try:
            correct_idx = labels.index(ans)
        except ValueError as e:
            raise ValueError(f"answerKey '{ans}' not found in labels {labels} (example {i}).") from e

        rows.append(
            {
                "example_id": _stable_id(split, i, q),
                "text": _format_csqa_prompt(q, choices),
                "answerKey": ans,
                "correct_idx": int(correct_idx),
                "csqa_choices": choices,
            }
        )

    df = pd.DataFrame(rows)

    # sanity
    if len(df) != n:
        raise RuntimeError(f"Expected {n} rows, got {len(df)}.")
    if not df["csqa_choices"].apply(lambda x: isinstance(x, list) and len(x) == 5).all():
        raise RuntimeError("csqa_choices must be a list of length 5 for every row.")

    return df
