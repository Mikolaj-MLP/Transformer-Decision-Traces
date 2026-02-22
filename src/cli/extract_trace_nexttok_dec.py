from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig

from src.models.gpt2_hooks import GPT2QKVHooksLite, GPT2ResidualHooksLite
from src.models.load import load_base
from src.utils.nexttok_texts import load_nexttok_texts
from src.utils.zarrio import create_array


def now_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_out_dir(out_dir: str | None, run_id: str) -> Path:
    root = _repo_root()
    if out_dir is None:
        return root / "traces" / run_id
    p = Path(out_dir)
    return p if p.is_absolute() else (root / p)


def _is_decoder_only(cfg) -> bool:
    model_type = str(getattr(cfg, "model_type", "")).lower()
    decoder_families = {
        "gpt2",
        "gptj",
        "gpt_neox",
        "llama",
        "mistral",
        "falcon",
        "opt",
        "bloom",
        "qwen2",
        "xglm",
        "mpt",
        "phi",
        "gemma",
        "mixtral",
    }
    is_encdec = bool(getattr(cfg, "is_encoder_decoder", False))
    return (not is_encdec) and (
        bool(getattr(cfg, "is_decoder", False)) or model_type in decoder_families
    )


def _context_window(cfg, fallback: int = 1024) -> int:
    for name in ("n_positions", "max_position_embeddings", "n_ctx"):
        v = getattr(cfg, name, None)
        if isinstance(v, int) and v > 0:
            return int(v)
    return int(fallback)


def _safe_int(x) -> int:
    if x is None:
        return -1
    return int(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="gpt2")
    ap.add_argument("--model_path", type=str, default=None, help="local checkpoint dir")
    ap.add_argument(
        "--dataset",
        type=str,
        default="ud_ewt",
        choices=["ud_ewt", "go_emotions", "csqa", "wikitext2", "wikitext103"],
    )
    ap.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["train", "validation", "test"],
    )
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"])
    ap.add_argument(
        "--capture",
        nargs="+",
        default=["attn", "qkv", "hidden", "resid"],
        choices=["attn", "qkv", "hidden", "resid"],
    )
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--max_seq_len",
        type=str,
        default="auto",
        help="int (e.g. 128) or 'auto' to fit full prompts in this split",
    )
    ap.add_argument(
        "--next_topk",
        type=int,
        default=5,
        help="store top-k next-token predictions per example",
    )
    args = ap.parse_args()

    df = load_nexttok_texts(dataset=args.dataset, split=args.split, limit=args.limit)
    df = df.reset_index(drop=True)
    N = len(df)
    if N == 0:
        raise RuntimeError("No examples loaded.")

    model_id = args.model_path or args.model
    tok, model, device = load_base(args.model, model_path=args.model_path)
    cfg = AutoConfig.from_pretrained(model_id)
    if not _is_decoder_only(cfg):
        raise ValueError(f"Model '{model_id}' is not decoder-only.")

    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    if args.max_seq_len.strip().lower() == "auto":
        lens = []
        for text in df["text"].tolist():
            ids = tok(text, add_special_tokens=False, truncation=False)["input_ids"]
            lens.append(len(ids))
        T = min(max(lens), _context_window(cfg))
    else:
        T = int(args.max_seq_len)

    tok_kwargs = {
        "padding": "max_length",
        "truncation": True,
        "max_length": T,
        "return_tensors": "pt",
    }
    if getattr(tok, "is_fast", False):
        tok_kwargs["return_offsets_mapping"] = True

    enc = tok(df["text"].tolist(), **tok_kwargs)

    L = _safe_int(getattr(cfg, "n_layer", getattr(cfg, "num_hidden_layers", 0)))
    H = _safe_int(getattr(cfg, "n_head", getattr(cfg, "num_attention_heads", 0)))
    D = _safe_int(getattr(cfg, "n_embd", getattr(cfg, "hidden_size", 0)))
    if L <= 0 or H <= 0 or D <= 0:
        raise RuntimeError("Could not infer decoder dimensions from config.")
    d_head = D // H

    zdtype = "f2" if args.dtype == "float16" else "f4"
    np_dtype = np.float16 if args.dtype == "float16" else np.float32
    topk = max(1, int(args.next_topk))

    run_id = f"{now_id()}_{Path(model_id).name}_{args.dataset}_{args.split}_n{N}_nexttok"
    final_root = _resolve_out_dir(args.out_dir, run_id)
    tmp_root = final_root.parent / f"_tmp_{final_root.name}"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_root.mkdir(parents=True, exist_ok=False)

    def p(*parts):
        return (tmp_root / Path(*parts)).as_posix()

    attn_arr = q_arr = k_arr = v_arr = None
    if "attn" in args.capture:
        attn_arr = create_array(
            p("dec_self_attn.zarr"),
            "attn",
            shape=(N, L, H, T, T),
            chunks=(1, 1, 1, T, T),
            dtype=zdtype,
        )
    if "qkv" in args.capture:
        q_arr = create_array(
            p("dec_self_qkv.zarr"),
            "q",
            shape=(N, L, H, T, d_head),
            chunks=(1, 1, 1, T, d_head),
            dtype=zdtype,
        )
        k_arr = create_array(
            p("dec_self_qkv.zarr"),
            "k",
            shape=(N, L, H, T, d_head),
            chunks=(1, 1, 1, T, d_head),
            dtype=zdtype,
        )
        v_arr = create_array(
            p("dec_self_qkv.zarr"),
            "v",
            shape=(N, L, H, T, d_head),
            chunks=(1, 1, 1, T, d_head),
            dtype=zdtype,
        )

    hidden_arr = None
    if "hidden" in args.capture:
        hidden_arr = create_array(
            p("dec_hidden.zarr"),
            "h",
            shape=(N, L + 1, T, D),
            chunks=(1, 1, T, D),
            dtype=zdtype,
        )

    res_embed = res_pre = res_post_attn = res_post_mlp = None
    if "resid" in args.capture:
        res_embed = create_array(
            p("dec_res_embed.zarr"),
            "x",
            shape=(N, T, D),
            chunks=(1, T, D),
            dtype=zdtype,
        )
        res_pre = create_array(
            p("dec_res_pre_attn.zarr"),
            "x",
            shape=(N, L, T, D),
            chunks=(1, 1, T, D),
            dtype=zdtype,
        )
        res_post_attn = create_array(
            p("dec_res_post_attn.zarr"),
            "x",
            shape=(N, L, T, D),
            chunks=(1, 1, T, D),
            dtype=zdtype,
        )
        res_post_mlp = create_array(
            p("dec_res_post_mlp.zarr"),
            "x",
            shape=(N, L, T, D),
            chunks=(1, 1, T, D),
            dtype=zdtype,
        )

    qkv_hooks = GPT2QKVHooksLite(model) if "qkv" in args.capture else None
    resid_hooks = GPT2ResidualHooksLite(model) if "resid" in args.capture else None

    next_pos = [-1] * N
    next_true_id = [-1] * N
    next_true_token = [None] * N
    next_pred_id = [-1] * N
    next_pred_token = [None] * N
    next_correct = [None] * N
    next_pred_prob = [None] * N
    next_true_prob = [None] * N
    next_entropy = [None] * N
    next_topk_ids = [None] * N
    next_topk_tokens = [None] * N
    next_topk_probs = [None] * N

    model.eval()
    try:
        with torch.no_grad():
            for i in range(0, N, args.batch_size):
                sl = slice(i, min(i + args.batch_size, N))
                print(f"[batch] {sl.start}-{sl.stop} / {N}")

                batch = {
                    "input_ids": enc["input_ids"][sl].to(device),
                    "attention_mask": enc["attention_mask"][sl].to(device),
                }
                if qkv_hooks:
                    qkv_hooks.clear()

                out = model(
                    **batch,
                    output_attentions=("attn" in args.capture),
                    output_hidden_states=("hidden" in args.capture),
                    return_dict=True,
                )

                if "attn" in args.capture and out.attentions is not None:
                    A = np.stack([a.detach().cpu().numpy() for a in out.attentions], axis=1)
                    attn_arr[sl] = A.astype(np_dtype)

                if "qkv" in args.capture and qkv_hooks is not None:
                    q, k, v = qkv_hooks.stack()
                    q_arr[sl] = q.detach().cpu().numpy().astype(np_dtype)
                    k_arr[sl] = k.detach().cpu().numpy().astype(np_dtype)
                    v_arr[sl] = v.detach().cpu().numpy().astype(np_dtype)

                if "hidden" in args.capture and hidden_arr is not None and out.hidden_states is not None:
                    Hs = torch.stack([h.detach().cpu() for h in out.hidden_states], dim=1)
                    hidden_arr[sl] = Hs.numpy().astype(np_dtype)

                if "resid" in args.capture and resid_hooks is not None:
                    emb = resid_hooks.pop_embed()
                    pre, pattn, post = resid_hooks.pop_layers()

                    if emb is not None:
                        res_embed[sl] = emb.detach().cpu().numpy().astype(np_dtype)

                    def stack_layers(dct, bsz):
                        xs = []
                        base_dtype = emb.dtype if emb is not None else torch.float16
                        for li in range(L):
                            x = dct.get(li)
                            if x is None:
                                x = torch.zeros((bsz, T, D), device=device, dtype=base_dtype)
                            xs.append(x)
                        return torch.stack(xs, dim=1).detach().cpu().numpy()

                    B = sl.stop - sl.start
                    res_pre[sl] = stack_layers(pre, B).astype(np_dtype)
                    res_post_attn[sl] = stack_layers(pattn, B).astype(np_dtype)
                    res_post_mlp[sl] = stack_layers(post, B).astype(np_dtype)

                logits = out.logits.detach().float().cpu()
                ids_cpu = batch["input_ids"].detach().cpu()
                mask_cpu = batch["attention_mask"].detach().cpu()
                B = ids_cpu.shape[0]
                kcur = min(topk, logits.shape[-1])

                for bi in range(B):
                    row_i = sl.start + bi
                    active = int(mask_cpu[bi].sum().item())
                    if active < 2:
                        continue

                    p_next = active - 2
                    true_id = int(ids_cpu[bi, p_next + 1].item())
                    vec = logits[bi, p_next]
                    probs = torch.softmax(vec, dim=-1)

                    pred_id = int(torch.argmax(vec).item())
                    pred_prob = float(probs[pred_id].item())
                    true_prob = float(probs[true_id].item())
                    ent = float((-(probs * torch.log(probs + 1e-12))).sum().item())

                    tv, ti = torch.topk(vec, k=kcur, dim=-1)
                    top_ids = [int(x) for x in ti.tolist()]
                    top_probs = [float(probs[t].item()) for t in top_ids]

                    next_pos[row_i] = int(p_next)
                    next_true_id[row_i] = true_id
                    next_true_token[row_i] = tok.convert_ids_to_tokens([true_id])[0]
                    next_pred_id[row_i] = pred_id
                    next_pred_token[row_i] = tok.convert_ids_to_tokens([pred_id])[0]
                    next_correct[row_i] = bool(pred_id == true_id)
                    next_pred_prob[row_i] = pred_prob
                    next_true_prob[row_i] = true_prob
                    next_entropy[row_i] = ent
                    next_topk_ids[row_i] = top_ids
                    next_topk_tokens[row_i] = tok.convert_ids_to_tokens(top_ids)
                    next_topk_probs[row_i] = top_probs
    finally:
        if qkv_hooks is not None:
            qkv_hooks.remove()
        if resid_hooks is not None:
            resid_hooks.remove()

    base = {
        "example_id": df["example_id"].tolist(),
        "text": df["text"].tolist(),
        "input_ids": enc["input_ids"].cpu().tolist(),
        "attention_mask": enc["attention_mask"].cpu().tolist(),
    }
    if "offset_mapping" in enc:
        base["offset_mapping"] = enc["offset_mapping"].cpu().tolist()
    else:
        base["offset_mapping"] = [None] * N

    df_tok = pd.DataFrame(base)
    df_tok["tokens"] = [tok.convert_ids_to_tokens(row) for row in df_tok["input_ids"]]

    df_tok["next_pos"] = next_pos
    df_tok["next_true_id"] = next_true_id
    df_tok["next_true_token"] = next_true_token
    df_tok["next_pred_id"] = next_pred_id
    df_tok["next_pred_token"] = next_pred_token
    df_tok["next_correct"] = next_correct
    df_tok["next_pred_prob"] = next_pred_prob
    df_tok["next_true_prob"] = next_true_prob
    df_tok["next_entropy"] = next_entropy
    df_tok["next_topk_ids"] = next_topk_ids
    df_tok["next_topk_tokens"] = next_topk_tokens
    df_tok["next_topk_probs"] = next_topk_probs
    df_tok.to_parquet(tmp_root / "tokens.parquet", index=False)

    valid_mask = df_tok["next_pos"] >= 0
    n_valid = int(valid_mask.sum())
    next_acc = float(df_tok.loc[valid_mask, "next_correct"].mean()) if n_valid > 0 else None
    next_true_prob_mean = (
        float(df_tok.loc[valid_mask, "next_true_prob"].mean()) if n_valid > 0 else None
    )

    meta = {
        "run_id": final_root.name,
        "model": model_id,
        "arch": "dec",
        "objective": "next_token",
        "dataset": args.dataset,
        "split": args.split,
        "n_examples": N,
        "max_seq_len": T,
        "num_layers": L,
        "num_heads": H,
        "head_dim": d_head,
        "dtype": args.dtype,
        "capture": args.capture,
        "next_topk": topk,
        "next_token_examples": n_valid,
        "next_token_acc": next_acc,
        "next_token_true_prob_mean": next_true_prob_mean,
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
