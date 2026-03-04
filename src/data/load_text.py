# src/data/load_text.py
import json
import pandas as pd
from typing import Literal, Tuple

UD_PATH = "data/processed/ud_ewt.parquet"
GO_PATH = "data/processed/go_emotions.parquet"
GO_LABELS = "data/processed/go_emotions.labels.json"

def load_ud_ewt(path: str = UD_PATH, token_level: bool = False) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if not token_level:
        return df

    keep = ["Tense","Number","Person","Mood","VerbForm","Case","Gender","Degree","Aspect","Polarity"]
    rows = []
    for _, r in df.iterrows():
        feats_list = json.loads(r["feats_json"])
        for i, tok in enumerate(r["tokens"]):
            feats = feats_list[i] if i < len(feats_list) else {}
            rows.append({
                "dataset": r["dataset"],
                "split": r["split"],
                "example_id": r["example_id"],
                "text": r["text"],
                "tok": tok,
                "tok_idx": i,
                "offset_start": r["token_offsets"][i][0],
                "offset_end": r["token_offsets"][i][1],
                "upos": r["upos"][i],
                "xpos": r["xpos"][i],
                "head": r["head"][i],
                "deprel": r["deprel"][i],
                **{f"m_{k}": feats.get(k) for k in keep},
            })
    return pd.DataFrame(rows)

def load_go_emotions(path: str = GO_PATH) -> Tuple[pd.DataFrame, list]:
    df = pd.read_parquet(path)
    with open(GO_LABELS, "r", encoding="utf-8") as f:
        labels = json.load(f)
    return df, labels
