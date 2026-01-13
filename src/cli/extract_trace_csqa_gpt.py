# src/cli/extract_trace_csqa_gpt.py
from __future__ import annotations

import argparse, json, shutil, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig

from src.data.load_csqa import load_csqa
from src.models.load import load_base
from src.models.gpt2_hooks import GPT2QKVHooksLite, GPT2ResidualHooksLite
from src.utils.zarrio import create_array


def now_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, default=None, help="local fine-tuned checkpoint dir")
    ap.add_argument("--split", type=str, default="validation", choices=["train", "validation", "test"])
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"])
    ap.add_argument("--capture", nargs="+", default=["attn", "qkv", "hidden", "resid"],
                    choices=["attn", "qkv", "hidden", "resid"])
    ap.add_argument("--limit", type=int, default=None, help="debug only; default=full split")
    ap.add_argument("--max_seq_len", type=str, default="auto",
                    help="int (e.g. 128) or 'auto' to fit full prompts in this split")
    args = ap.parse_args()

    # data 
    df = load_csqa(split=args.split, limit=args.limit)  # limit=None => full split 
    df = df.reset_index(drop=True)
    N = len(df)
    if N == 0:
        raise RuntimeError("No CSQA examples loaded.")

    # model
    model_name = "gpt2"
    model_id = args.model_path or model_name
    tok, model, device = load_base(model_name, model_path=args.model_path)
    cfg = AutoConfig.from_pretrained(model_id)

    # GPT-2: create PAD token by aliasing EOS
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    # Decoder-only: left padding keeps "end of prompt" aligned at the right edge
    tok.padding_side = "left"

    # choose max_seq_len
    if args.max_seq_len.strip().lower() == "auto":
        # measure true token lengths (no padding) to avoid truncation
        lens = []
        for t in df["text"].tolist():
            ids = tok(t, add_special_tokens=False, truncation=False)["input_ids"]
            lens.append(len(ids))
        max_len = max(lens)
        # Hard upper bound: GPT-2 context window
        ctx = int(getattr(cfg, "n_ctx", 1024))
        T = min(max_len, ctx)
    else:
        T = int(args.max_seq_len)

    #tokenize (fixed length => deterministic )
    enc = tok(
        df["text"].tolist(),
        padding="max_length",
        truncation=True,
        max_length=T,
        return_tensors="pt",
        return_offsets_mapping=True,
    )

    # dims
    L = int(getattr(cfg, "n_layer", 12))
    H = int(getattr(cfg, "n_head", 12))
    D = int(getattr(cfg, "n_embd", 768))
    d_head = D // H

    zdtype = "f2" if args.dtype == "float16" else "f4"
    np_dtype = np.float16 if args.dtype == "float16" else np.float32

    # run dirs
    run_id = f"{now_id()}_{model_name}_csqa_{args.split}_n{N}"
    final_root = Path(args.out_dir or f"traces/{run_id}")
    tmp_root = final_root.parent / f"_tmp_{final_root.name}"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_root.mkdir(parents=True, exist_ok=False)

    def p(*parts): return (tmp_root / Path(*parts)).as_posix()

    #zarr arrays 
    attn_arr = q_arr = k_arr = v_arr = None
    if "attn" in args.capture:
        attn_arr = create_array(p("dec_self_attn.zarr"), "attn",
                                shape=(N, L, H, T, T),
                                chunks=(1, 1, 1, T, T), dtype=zdtype)
    if "qkv" in args.capture:
        q_arr = create_array(p("dec_self_qkv.zarr"), "q",
                             shape=(N, L, H, T, d_head),
                             chunks=(1, 1, 1, T, d_head), dtype=zdtype)
        k_arr = create_array(p("dec_self_qkv.zarr"), "k",
                             shape=(N, L, H, T, d_head),
                             chunks=(1, 1, 1, T, d_head), dtype=zdtype)
        v_arr = create_array(p("dec_self_qkv.zarr"), "v",
                             shape=(N, L, H, T, d_head),
                             chunks=(1, 1, 1, T, d_head), dtype=zdtype)

    hidden_arr = None
    if "hidden" in args.capture:
        hidden_arr = create_array(p("dec_hidden.zarr"), "h",
                                  shape=(N, L + 1, T, D),
                                  chunks=(1, 1, T, D), dtype=zdtype)

    res_embed = res_pre = res_post_attn = res_post_mlp = None
    if "resid" in args.capture:
        res_embed = create_array(p("dec_res_embed.zarr"), "x",
                                 shape=(N, T, D), chunks=(1, T, D), dtype=zdtype)
        res_pre = create_array(p("dec_res_pre_attn.zarr"), "x",
                               shape=(N, L, T, D), chunks=(1, 1, T, D), dtype=zdtype)
        res_post_attn = create_array(p("dec_res_post_attn.zarr"), "x",
                                     shape=(N, L, T, D), chunks=(1, 1, T, D), dtype=zdtype)
        res_post_mlp = create_array(p("dec_res_post_mlp.zarr"), "x",
                                    shape=(N, L, T, D), chunks=(1, 1, T, D), dtype=zdtype)

    #hooks
    qkv_hooks = GPT2QKVHooksLite(model) if "qkv" in args.capture else None
    resid_hooks = GPT2ResidualHooksLite(model) if "resid" in args.capture else None


    #forward passes
    model.eval()
    with torch.no_grad():
        for i in range(0, N, args.batch_size):
            sl = slice(i, min(i + args.batch_size, N))
            print(f"[batch] {sl.start}-{sl.stop} / {N}")

            batch = {
                "input_ids": enc["input_ids"][sl].to(device),
                "attention_mask": enc["attention_mask"][sl].to(device),
            }

            if qkv_hooks: qkv_hooks.clear()

            out = model(
                **batch,
                output_attentions=("attn" in args.capture),
                output_hidden_states=("hidden" in args.capture),
                return_dict=True,
            )

            if "attn" in args.capture and out.attentions is not None:
                A = np.stack([a.detach().cpu().numpy() for a in out.attentions], axis=1)  # (B,L,H,T,T)
                attn_arr[sl] = A.astype(np_dtype)

            if "qkv" in args.capture and qkv_hooks is not None:
                q, k, v = qkv_hooks.stack()  # torch (B,L,H,T,d_head)
                q_arr[sl] = q.detach().cpu().numpy().astype(np_dtype)
                k_arr[sl] = k.detach().cpu().numpy().astype(np_dtype)
                v_arr[sl] = v.detach().cpu().numpy().astype(np_dtype)

            if "hidden" in args.capture and hidden_arr is not None and out.hidden_states is not None:
                Hs = torch.stack([h.detach().cpu() for h in out.hidden_states], dim=1)  # (B,L+1,T,D)
                hidden_arr[sl] = Hs.numpy().astype(np_dtype)

            if "resid" in args.capture and resid_hooks is not None:
                emb = resid_hooks.pop_embed()               # (B,T,D)
                pre, pattn, post = resid_hooks.pop_layers() # dict layer->(B,T,D)

                if emb is not None:
                    res_embed[sl] = emb.detach().cpu().numpy().astype(np_dtype)

                def stack_layers(dct, B):
                    xs = []
                    for li in range(L):
                        x = dct.get(li)
                        if x is None:
                            x = torch.zeros((B, T, D), device=device, dtype=emb.dtype if emb is not None else torch.float16)
                        xs.append(x)
                    return torch.stack(xs, dim=1).detach().cpu().numpy()

                B = sl.stop - sl.start
                res_pre[sl] = stack_layers(pre, B).astype(np_dtype)
                res_post_attn[sl] = stack_layers(pattn, B).astype(np_dtype)
                res_post_mlp[sl] = stack_layers(post, B).astype(np_dtype)

    # tokens.parquet 
    df_tok = pd.DataFrame({
        "example_id": df["example_id"].tolist(),
        "text": df["text"].tolist(),
        "input_ids": enc["input_ids"].cpu().tolist(),
        "attention_mask": enc["attention_mask"].cpu().tolist(),
        "offset_mapping": enc["offset_mapping"].cpu().tolist(),
    })
    df_tok["tokens"] = [tok.convert_ids_to_tokens(row) for row in df_tok["input_ids"]]
    df_tok["answerKey"] = df["answerKey"].tolist()
    df_tok["csqa_choices"] = df["csqa_choices"].tolist()
    df_tok.to_parquet(tmp_root / "tokens.parquet", index=False)

    meta = {
        "run_id": final_root.name,
        "model": model_id,
        "arch": "dec",
        "dataset": "csqa",
        "split": args.split,
        "n_examples": N,
        "max_seq_len": T,
        "num_layers": L,
        "num_heads": H,
        "head_dim": d_head,
        "dtype": args.dtype,
        "capture": args.capture,
        "has_targets": None,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with (tmp_root / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if final_root.exists():
        shutil.rmtree(final_root)
    shutil.move(tmp_root.as_posix(), final_root.as_posix())
    print(f"[done] wrote {final_root}")


if __name__ == "__main__":
    main()
