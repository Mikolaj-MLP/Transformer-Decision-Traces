from __future__ import annotations

from typing import Optional

import pandas as pd

from src.data.load_aqua_rat import (
    DEFAULT_AQUA_SPLIT_SEED,
    DEFAULT_AQUA_TEST_SIZE,
    DEFAULT_AQUA_TRAIN_SIZE,
    DEFAULT_AQUA_VALIDATION_SIZE,
    load_aqua_rat,
)
from src.data.load_csqa import load_csqa


SUPPORTED_DATASETS = ("csqa", "aqua_rat")


def load_mcqa(
    dataset: str,
    *,
    split: str,
    limit: Optional[int] = None,
    aqua_train_size: int = DEFAULT_AQUA_TRAIN_SIZE,
    aqua_validation_size: int = DEFAULT_AQUA_VALIDATION_SIZE,
    aqua_test_size: int = DEFAULT_AQUA_TEST_SIZE,
    aqua_split_seed: int = DEFAULT_AQUA_SPLIT_SEED,
) -> pd.DataFrame:
    normalized = str(dataset).lower()
    if normalized == "csqa":
        return load_csqa(split=split, limit=limit)
    if normalized == "aqua_rat":
        return load_aqua_rat(
            split=split,
            limit=limit,
            train_size=aqua_train_size,
            validation_size=aqua_validation_size,
            test_size=aqua_test_size,
            split_seed=aqua_split_seed,
        )
    raise ValueError(f"Unsupported dataset: {dataset}")
