# src/data/prepare_datasets.py
import os, json, hashlib
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import pandas as pd

# optional HF path
def _maybe_import_datasets():
    try:
        from datasets import load_dataset 
        return load_dataset
    except Exception:
        return None

HF_load_dataset = _maybe_import_datasets()

# utils 
def ensure_dir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)

def stable_id(prefix: str, split: str, idx: int, text: str) -> str:
    h = hashlib.sha1()
    h.update((prefix + "::" + split + "::" + str(idx) + "::" + text).encode("utf-8"))
    return h.hexdigest()[:16]

def reconstruct_text_and_offsets(tokens: List[str], space_after_flags: Optional[List[bool]]) -> Tuple[str, List[Tuple[int,int]]]:
    text_parts, offsets, cursor = [], [], 0
    N = len(tokens)
    for i, tok in enumerate(tokens):
        text_parts.append(tok)
        start, end = cursor, cursor + len(tok)
        offsets.append((start, end))
        cursor = end
        if i != N - 1:
            sep = "" if (space_after_flags and not space_after_flags[i]) else " "
            text_parts.append(sep)
            cursor += len(sep)
    return ("".join(text_parts), offsets)

# UD English-EWT via GitHub 
def _download_ud_ewt_files(tmp_dir: Path, tag: str = "r2.14") -> Dict[str, Path]:
    import requests
    ensure_dir(tmp_dir.as_posix())
    base = f"https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/{tag}"
    files = {
        "train": "en_ewt-ud-train.conllu",
        "validation": "en_ewt-ud-dev.conllu",
        "test": "en_ewt-ud-test.conllu",
    }
    out = {}
    for split, fname in files.items():
        url = f"{base}/{fname}"
        dst = tmp_dir / fname
        if not dst.exists():
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            dst.write_bytes(r.content)
        out[split] = dst
    return out

def _parse_conllu_sentence(sent) -> Dict[str, List]:
    tokens, upos, xpos, head, deprel = [], [], [], [], []
    feats_list, space_after_flags = [], []
    for tok in sent:
        tok_id = tok["id"]
        if not isinstance(tok_id, int):
            continue
        tokens.append(tok["form"])
        upos.append(tok.get("upostag"))
        xpos.append(tok.get("xpostag"))
        head.append(tok.get("head"))
        deprel.append(tok.get("deprel"))
        feats_list.append(tok.get("feats") or {})
        misc = tok.get("misc") or {}
        # SpaceAfter=No --> False; otherwise True
        space_after_flags.append(False if misc.get("SpaceAfter") == "No" else True)
    return {
        "tokens": tokens,
        "upos": upos,
        "xpos": xpos,
        "head": head,
        "deprel": deprel,
        "feats_list": feats_list,
        "space_after_flags": space_after_flags,
    }

def _prepare_ud_ewt_from_github(out_dir: str) -> str:
    from conllu import parse_incr
    tmp = Path("data/raw/ud_ewt_conllu")
    paths = _download_ud_ewt_files(tmp)

    keep_feats = {"Tense","Number","Person","Mood","VerbForm","Case","Gender","Degree","Aspect","Polarity"}
    rows = []
    for split, fpath in paths.items():
        with fpath.open("r", encoding="utf-8") as f:
            idx = 0
            for sent in parse_incr(f):
                parsed = _parse_conllu_sentence(sent)
                tokens = parsed["tokens"]
                if not tokens:
                    continue
                text, offsets = reconstruct_text_and_offsets(tokens, parsed["space_after_flags"])
                ex_id = stable_id("ud_ewt", split, idx, text)
                # reduce feats per token into a per-token dict with selected keys
                feats_subset = []
                for d in parsed["feats_list"]:
                    sub = {k: d.get(k) for k in keep_feats}
                    feats_subset.append(sub)
                rows.append({
                    "dataset": "ud_ewt",
                    "split": split,
                    "example_id": ex_id,
                    "text": text,
                    "tokens": tokens,
                    "token_offsets": offsets,
                    "upos": parsed["upos"],
                    "xpos": parsed["xpos"],
                    "head": parsed["head"],
                    "deprel": parsed["deprel"],
                    "feats_per_token": feats_subset,
                })
                idx += 1
    df = pd.DataFrame(rows)
    # compact JSON column for feats
    df["feats_json"] = df["feats_per_token"].apply(json.dumps)
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, "ud_ewt.parquet")
    df.drop(columns=["feats_per_token"]).to_parquet(out_path, index=False)
    return out_path

def prepare_ud_ewt(out_dir: str = "data/processed") -> str:
    """
    Try HF first; if the environment disallows dataset scripts,
    fall back to GitHub + CoNLL-U parsing so this works everywhere.
    Overcomplicated but should work once i start using AWS Instances to run the code for better performence
    """
    if HF_load_dataset is not None:
        try:
            ds = HF_load_dataset("universal_dependencies", "en_ewt")  # may raise in Datasets >=4   !  ! ! ! !! !  ! ! !! 
            rows = []
            keep_feats = ["Tense","Number","Person","Mood","VerbForm","Case","Gender","Degree","Aspect","Polarity"]
            for split in ["train", "validation", "test"]:
                d = ds[split]
                for i in range(len(d)):
                    ex = d[i]
                    tokens = ex["tokens"]
                    upos = ex.get("upos")
                    xpos = ex.get("xpos")
                    feats_raw = ex.get("feats")        # list[str] per token or per-sent string depending on version
                    head = ex.get("head")
                    deprel = ex.get("deprel")
                    misc = ex.get("misc")
                    # misc is list[str] like "SpaceAfter=No"; transform to flags
                    flags = []
                    if misc:
                        for m in misc:
                            flags.append(False if (m and "SpaceAfter=No" in m) else True)
                    else:
                        flags = [True] * len(tokens)
                    text, offsets = reconstruct_text_and_offsets(tokens, flags)
                    ex_id = stable_id("ud_ewt", split, i, text)

                    # feats parsing (handles per-token "A=B|C=D" strings)
                    feats_subset = []
                    if isinstance(feats_raw, list):
                        for f in feats_raw:
                            dct = {}
                            if f:
                                for kv in f.split("|"):
                                    if "=" in kv:
                                        k, v = kv.split("=", 1)
                                        if k in keep_feats:
                                            dct[k] = v
                            feats_subset.append(dct)
                    else:
                        feats_subset = [{} for _ in tokens]

                    rows.append({
                        "dataset": "ud_ewt",
                        "split": split,
                        "example_id": ex_id,
                        "text": text,
                        "tokens": tokens,
                        "token_offsets": offsets,
                        "upos": upos,
                        "xpos": xpos,
                        "head": head,
                        "deprel": deprel,
                        "feats_per_token": feats_subset,
                    })
            df = pd.DataFrame(rows)
            df["feats_json"] = df["feats_per_token"].apply(json.dumps)
            ensure_dir(out_dir)
            out_path = os.path.join(out_dir, "ud_ewt.parquet")
            df.drop(columns=["feats_per_token"]).to_parquet(out_path, index=False)
            return out_path
        except Exception:
            # fall back to GitHub
            return _prepare_ud_ewt_from_github(out_dir)
    else:
        return _prepare_ud_ewt_from_github(out_dir)

# GoEmotions
def _load_goemotions_hf():
    if HF_load_dataset is None:
        raise RuntimeError("datasets not installed")
    return HF_load_dataset("go_emotions")

def _load_goemotions_fallback():
    # Parquet mirror with simplified schema
    if HF_load_dataset is None:
        raise RuntimeError("datasets not installed")
    return HF_load_dataset("SetFit/go_emotions")

def prepare_goemotions(out_dir: str = "data/processed") -> str:
    print("[go_emotions] loading…")
    ensure_dir(out_dir)
    ds = None
    try:
        ds = _load_goemotions_hf()
    except Exception:
        ds = _load_goemotions_fallback()

    # label names
    try:
        label_names = ds["train"].features["labels"].feature.names
    except Exception:
        # mirror fallback
        label_names = ds["train"].features["labels"].feature.names
    n_labels = len(label_names)

    from sklearn.preprocessing import MultiLabelBinarizer
    mlb = MultiLabelBinarizer(classes=list(range(n_labels)))

    parts = []
    for split in ["train", "validation", "test"]:
        print(f"[go_emotions] processing split: {split}")
        pdf = ds[split].to_pandas()  # vectorized arrow-->pandas (much faster than per-item indexing)
        labels = pdf["labels"].tolist()
        Y = mlb.fit_transform(labels)  # multihot matrix (n_samples x n_labels)

        # build rows
        ex_ids = [stable_id("go_emotions", split, i, t) for i, t in enumerate(pdf["text"].values)]
        out = pd.DataFrame({
            "dataset": "go_emotions",
            "split": split,
            "example_id": ex_ids,
            "text": pdf["text"].values,
        })
        # store labels as lists for portability
        out["labels_idx"] = labels
        out["labels_names"] = [[label_names[j] for j in lst] for lst in labels]
        out["labels_multihot"] = [row.tolist() for row in Y]
        parts.append(out)

    df = pd.concat(parts, ignore_index=True)
    out_path = os.path.join(out_dir, "go_emotions.parquet")
    df.to_parquet(out_path, index=False)
    with open(os.path.join(out_dir, "go_emotions.labels.json"), "w", encoding="utf-8") as f:
        json.dump(label_names, f, ensure_ascii=False, indent=2)
    print(f"[go_emotions] wrote {out_path} ({len(df)} rows)")
    return out_path


# entrypoint
def prepare_all(out_dir: str = "data/processed") -> Dict[str, str]:
    return {"ud_ewt": prepare_ud_ewt(out_dir), "go_emotions": prepare_goemotions(out_dir)}

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="data/processed")
    ap.add_argument("--which", type=str, default="all", choices=["all","ud","go"])
    args = ap.parse_args()
    if args.which == "all":
        print(prepare_all(args.out_dir))
    elif args.which == "ud":
        print({"ud_ewt": prepare_ud_ewt(args.out_dir)})
    else:
        print({"go_emotions": prepare_goemotions(args.out_dir)})
