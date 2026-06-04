from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


AQUA_URLS = {
    "official_train": "https://raw.githubusercontent.com/google-deepmind/AQuA/master/train.json",
    "official_dev": "https://raw.githubusercontent.com/google-deepmind/AQuA/master/dev.json",
    "official_test": "https://raw.githubusercontent.com/google-deepmind/AQuA/master/test.json",
}

DEFAULT_AQUA_TRAIN_SIZE = 4_000
DEFAULT_AQUA_VALIDATION_SIZE = 1_500
DEFAULT_AQUA_TEST_SIZE = 4_000
DEFAULT_AQUA_SPLIT_SEED = 42
LETTERS = ["A", "B", "C", "D", "E"]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _cache_dir() -> Path:
    path = _repo_root() / "data" / "cache" / "aqua_rat"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _stable_id(source_split: str, idx: int, question: str) -> str:
    h = hashlib.md5(f"aqua_rat::{source_split}::{idx}::{question}".encode("utf-8")).hexdigest()
    return h


def _format_prompt(question: str, choices: List[Dict[str, str]]) -> str:
    lines = [f"Q: {question}", "Choices:"]
    for ch in choices:
        lines.append(f"{ch['label']}: {ch['text']}")
    lines.append("Answer:")
    return "\n".join(lines)


def _download_if_missing(url: str, path: Path) -> None:
    if path.exists():
        return
    with urllib.request.urlopen(url, timeout=120) as response:
        data = response.read()
    path.write_bytes(data)


def _parse_option(raw_option: str) -> Dict[str, str]:
    match = re.match(r"^\s*([A-E])\)\s*(.*)\s*$", str(raw_option))
    if not match:
        raise ValueError(f"Could not parse AQuA-RAT option: {raw_option!r}")
    label, text = match.group(1), match.group(2)
    return {"label": label, "text": text}


def _load_official_rows() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    cache = _cache_dir()

    for source_split, url in AQUA_URLS.items():
        local_path = cache / f"{source_split}.json"
        _download_if_missing(url, local_path)

        with local_path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                ex = json.loads(line)
                question = str(ex["question"]).strip()
                choices = [_parse_option(opt) for opt in ex["options"]]
                if len(choices) != 5:
                    raise ValueError(f"AQuA-RAT example {source_split}:{idx} has {len(choices)} choices (expected 5).")

                labels = [choice["label"] for choice in choices]
                answer_key = str(ex["correct"]).strip()
                if answer_key not in labels:
                    raise ValueError(
                        f"AQuA-RAT answer {answer_key!r} not found in labels {labels} for {source_split}:{idx}."
                    )
                correct_idx = labels.index(answer_key)

                rows.append(
                    {
                        "example_id": _stable_id(source_split, idx, question),
                        "text": _format_prompt(question, choices),
                        "answerKey": answer_key,
                        "correct_idx": int(correct_idx),
                        "mcqa_choices": choices,
                        "source_split": source_split,
                        "rationale": str(ex.get("rationale", "")),
                    }
                )

    return pd.DataFrame(rows)


def _build_repartitioned_splits(
    *,
    train_size: int,
    validation_size: int,
    test_size: int,
    split_seed: int,
) -> pd.DataFrame:
    all_rows = _load_official_rows().copy()
    total_needed = int(train_size) + int(validation_size) + int(test_size)
    if total_needed > len(all_rows):
        raise ValueError(
            f"Requested AQuA-RAT repartition size {total_needed} exceeds available rows {len(all_rows)}."
        )

    rng = np.random.default_rng(int(split_seed))
    perm = rng.permutation(len(all_rows))
    shuffled = all_rows.iloc[perm].reset_index(drop=True)

    train_end = int(train_size)
    validation_end = train_end + int(validation_size)
    test_end = validation_end + int(test_size)

    split_labels = (
        ["train"] * train_end
        + ["validation"] * int(validation_size)
        + ["test"] * int(test_size)
    )

    repartitioned = shuffled.iloc[:test_end].copy()
    repartitioned["split"] = split_labels
    return repartitioned


def load_aqua_rat(
    split: str = "validation",
    limit: Optional[int] = None,
    *,
    train_size: int = DEFAULT_AQUA_TRAIN_SIZE,
    validation_size: int = DEFAULT_AQUA_VALIDATION_SIZE,
    test_size: int = DEFAULT_AQUA_TEST_SIZE,
    split_seed: int = DEFAULT_AQUA_SPLIT_SEED,
) -> pd.DataFrame:
    """
    Returns a deterministic repartitioned AQuA-RAT dataframe with the same core
    columns as the CSQA loader:
      - example_id
      - text
      - answerKey
      - correct_idx
      - mcqa_choices
    """
    normalized_split = str(split).lower()
    if normalized_split == "dev":
        normalized_split = "validation"
    if normalized_split not in {"train", "validation", "test"}:
        raise ValueError(f"Unsupported AQuA-RAT split: {split}")

    repartitioned = _build_repartitioned_splits(
        train_size=train_size,
        validation_size=validation_size,
        test_size=test_size,
        split_seed=split_seed,
    )

    part = repartitioned.loc[repartitioned["split"].eq(normalized_split)].reset_index(drop=True)
    if limit is not None:
        part = part.iloc[: min(int(limit), len(part))].reset_index(drop=True)

    if not part["mcqa_choices"].apply(lambda x: isinstance(x, list) and len(x) == 5).all():
        raise RuntimeError("mcqa_choices must be a list of length 5 for every AQuA-RAT row.")

    return part
