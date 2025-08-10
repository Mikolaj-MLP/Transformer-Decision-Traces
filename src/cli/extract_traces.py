# src/cli/extract_traces.py
from __future__ import annotations
import os, json, time, argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig

from src.data.load_text import load_ud_ewt, load_go_emotions
from src.models.load import load_base
from src.models.hooks import QKVHooks, MLPHooks
from src.utils.zarrio import create_array
import zarr


def now_id():
    return time.strftime("%Y%m%d-%H%M%S")


def get_texts(dataset: str, split: str, limit: int) -> pd.DataFrame:
    if dataset == "ud_ewt":
        df = load_ud_ewt(token_level=False)
        df = df[df["split"] == split].head(limit).copy()
        return df[["example_id", "text"]].reset_index(drop=True)
    elif dataset == "go_emotions":
        df, _ = load_go_emotions()
        df = df[df["split"] == split].head(limit).copy()
        return df[["example_id", "text"]].reset_index(drop=True)
    else:
        raise ValueError("dataset must be 'ud_ewt' or 'go_emotions'")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="roberta-base")
    ap.add_argument("--dataset", type=str, choices=["ud_ewt", "go_emotions"], default="ud_ewt")
    ap.add_argument("--split", type=str, default="validation")
    ap.add_argument("--limit", type=int, default=32)
    ap.add_argument("--max_seq_len", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--capture", nargs="+", default=["attn", "qkv"], choices=["attn", "qkv", "mlp"])
    ap.add_argument("--mlp_pool", type=str, default="mean", choices=["mean", "max", "none"])
    ap.add_argument("--out_dir", type=str, default=None)
    args = ap.parse_args()

    texts = get_texts(args.dataset, args.split, args.limit)
    tok, model, device = load_base(args.model)

    run_id = f"{now_id()}_{args.model}_{args.dataset}_{args.split}_n{len(texts)}"
    out_root = args.out_dir or f"traces/{run_id}"
    Path(out_root).mkdir(parents=True, exist_ok=True)

    # Tokenize once
    enc = tok(
        texts["text"].tolist(),
        padding="max_length",
        truncation=True,
        max_length=args.max_seq_len,
        return_tensors="pt",
        return_offsets_mapping=True,   # FastTokenizer char offsets
    )


    # Shapes
    cfg = AutoConfig.from_pretrained(args.model)
    L, H = int(cfg.num_hidden_layers), int(cfg.num_attention_heads)
    T = int(args.max_seq_len)
    d_head = int(cfg.hidden_size) // H
    N = len(texts)

    # Storage
    attn_arr = None
    if "attn" in args.capture:
        attn_arr = create_array(
            os.path.join(out_root, "attn.zarr"),
            "attn",
            shape=(N, L, H, T, T),
            chunks=(1, 1, 1, T, T),
            dtype="f2",
        )

    q_arr = k_arr = v_arr = None
    if "qkv" in args.capture:
        q_arr = create_array(
            os.path.join(out_root, "qkv.zarr"),
            "q",
            shape=(N, L, H, T, d_head),
            chunks=(1, 1, 1, T, d_head),
            dtype="f2",
        )
        k_arr = create_array(
            os.path.join(out_root, "qkv.zarr"),
            "k",
            shape=(N, L, H, T, d_head),
            chunks=(1, 1, 1, T, d_head),
            dtype="f2",
        )
        v_arr = create_array(
            os.path.join(out_root, "qkv.zarr"),
            "v",
            shape=(N, L, H, T, d_head),
            chunks=(1, 1, 1, T, d_head),
            dtype="f2",
        )

    mlp_arr = None
    mlp_pool = args.mlp_pool
    if "mlp" in args.capture:
        if mlp_pool == "none":
            # Warning: big! use carefully
            # We don't know D_intermediate ahead of time without a dry-run; do one small pass
            pass

    # Hooks
    qkv_hooks = QKVHooks(model) if "qkv" in args.capture else None
    mlp_hooks = MLPHooks(model) if "mlp" in args.capture else None

    # Iterate
    n = len(texts)
    with torch.no_grad():
        for i in range(0, n, args.batch_size):
            sl = slice(i, min(i + args.batch_size, n))
            # Only pass model-accepted keys (e.g., input_ids, attention_mask, token_type_ids)
            model_inputs = set(getattr(tok, "model_input_names", ["input_ids","attention_mask","token_type_ids","position_ids"]))
            batch = {k: enc[k][sl].to(device) for k in model_inputs if k in enc}


            if qkv_hooks:
                qkv_hooks.clear()
            if mlp_hooks:
                mlp_hooks.clear()

            out = model(**batch, output_attentions=("attn" in args.capture))

            if attn_arr is not None:
                # out.attentions: list[L] of (B, H, T, T)
                attns = [a.detach().to("cpu").numpy().astype(np.float16) for a in out.attentions]
                attn_b = np.stack(attns, axis=1)  # (B, L, H, T, T)
                attn_arr[sl, :, :, :, :] = attn_b

            if qkv_hooks is not None:
                q, k, v = qkv_hooks.stack()  # (B, L, H, T, d)
                q_arr[sl, :, :, :, :] = q.numpy().astype(np.float16)
                k_arr[sl, :, :, :, :] = k.numpy().astype(np.float16)
                v_arr[sl, :, :, :, :] = v.numpy().astype(np.float16)

            if mlp_hooks is not None:
                pooled = mlp_hooks.stack(pool=None if mlp_pool == "none" else mlp_pool)
                if mlp_arr is None:
                    # Create array now that we know D (or pooled rank)
                    if pooled.ndim == 4:
                        # (B, L, T, D)
                        D = int(pooled.shape[-1])
                        mlp_arr = create_array(
                            os.path.join(out_root, "mlp.zarr"),
                            "mlp",
                            shape=(N, L, T, D),
                            chunks=(1, 1, T, D),
                            dtype="f2",
                        )
                    else:
                        # (B, L, T)
                        mlp_arr = create_array(
                            os.path.join(out_root, "mlp.zarr"),
                            "mlp",
                            shape=(N, L, T),
                            chunks=(1, 1, T),
                            dtype="f2",
                        )
                mlp_arr[sl, ...] = pooled.numpy().astype(np.float16)

    # Save per-example metadata (lists for portability)
    df_tok = pd.DataFrame({
        "example_id": texts["example_id"],
        "text": texts["text"],
        "input_ids": enc["input_ids"].cpu().tolist(),
        "attention_mask": enc["attention_mask"].cpu().tolist(),
        "offset_mapping": enc["offset_mapping"].cpu().tolist(),  # (T,2) or (0,0) for specials
    })
    # decoded tokens
    df_tok["tokens"] = [tok.convert_ids_to_tokens(row) for row in df_tok["input_ids"]]
    df_tok.to_parquet(os.path.join(out_root, "tokens.parquet"), index=False)

    # Meta 
    meta = {
        "run_id": os.path.basename(out_root),
        "model": args.model,
        "dataset": args.dataset,
        "split": args.split,
        "n_examples": N,
        "max_seq_len": T,
        "num_layers": L,
        "num_heads": H,
        "head_dim": d_head,
        "capture": args.capture,
        "mlp_pool": args.mlp_pool if "mlp" in args.capture else None,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(out_root, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[done] wrote {out_root}")



if __name__ == "__main__":
    main()
