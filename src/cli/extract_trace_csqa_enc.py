from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig, AutoModelForMultipleChoice, AutoTokenizer

from src.data.load_csqa import load_csqa
from src.models.encoder_hooks import EncoderQKVHooksLite, EncoderResidualHooksLite
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


def _get_ctx_window(cfg, fallback: int = 512) -> int:
    for name in ("max_position_embeddings", "n_positions"):
        v = getattr(cfg, name, None)
        if isinstance(v, int) and v > 0:
            return int(v)
    return int(fallback)


def _extract_question(prompt_text: str) -> str:
    lines = prompt_text.splitlines()
    if lines and lines[0].startswith("Q: "):
        return lines[0][3:].strip()
    return prompt_text.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="roberta-base")
    ap.add_argument("--model_path", type=str, default=None, help="local fine-tuned checkpoint dir")
    ap.add_argument("--split", type=str, default="validation", choices=["train", "validation", "test"])
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"])
    ap.add_argument(
        "--capture",
        nargs="+",
        default=["attn", "qkv", "hidden", "resid"],
        choices=["attn", "qkv", "hidden", "resid"],
    )
    ap.add_argument("--limit", type=int, default=None, help="debug only; default=full split")
    ap.add_argument(
        "--max_seq_len",
        type=str,
        default="auto",
        help="int (e.g. 128) or 'auto' based on model max positions",
    )
    args = ap.parse_args()

    model_id = args.model_path or args.model
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    cfg = AutoConfig.from_pretrained(model_id)
    model = AutoModelForMultipleChoice.from_pretrained(model_id, attn_implementation="eager")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    df = load_csqa(split=args.split, limit=args.limit).reset_index(drop=True)
    N = len(df)
    if N == 0:
        raise RuntimeError("No CSQA examples loaded.")

    C = int(len(df.iloc[0]["csqa_choices"]))
    if C <= 1:
        raise RuntimeError("Expected at least 2 choices per example.")

    labels = [ch["label"] for ch in df.iloc[0]["csqa_choices"]]
    label2idx = {lab: i for i, lab in enumerate(labels)}
    y_true = np.array([label2idx[x] for x in df["answerKey"].tolist()], dtype=np.int64)

    if args.max_seq_len.strip().lower() == "auto":
        T = min(_get_ctx_window(cfg), 256)
    else:
        T = int(args.max_seq_len)

    L = int(getattr(cfg, "num_hidden_layers", 12))
    H = int(getattr(cfg, "num_attention_heads", 12))
    D = int(getattr(cfg, "hidden_size", 768))
    d_head = D // H

    zdtype = "f2" if args.dtype == "float16" else "f4"
    np_dtype = np.float16 if args.dtype == "float16" else np.float32

    run_id = f"{now_id()}_{Path(model_id).name}_csqa_{args.split}_n{N}_encmcq"
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
            p("enc_mc_attn.zarr"),
            "attn",
            shape=(N, C, L, H, T, T),
            chunks=(1, C, 1, 1, T, T),
            dtype=zdtype,
        )
    if "qkv" in args.capture:
        q_arr = create_array(
            p("enc_mc_qkv.zarr"),
            "q",
            shape=(N, C, L, H, T, d_head),
            chunks=(1, C, 1, 1, T, d_head),
            dtype=zdtype,
        )
        k_arr = create_array(
            p("enc_mc_qkv.zarr"),
            "k",
            shape=(N, C, L, H, T, d_head),
            chunks=(1, C, 1, 1, T, d_head),
            dtype=zdtype,
        )
        v_arr = create_array(
            p("enc_mc_qkv.zarr"),
            "v",
            shape=(N, C, L, H, T, d_head),
            chunks=(1, C, 1, 1, T, d_head),
            dtype=zdtype,
        )

    hidden_arr = None
    if "hidden" in args.capture:
        hidden_arr = create_array(
            p("enc_mc_hidden.zarr"),
            "h",
            shape=(N, C, L + 1, T, D),
            chunks=(1, C, 1, T, D),
            dtype=zdtype,
        )

    res_embed = res_pre = res_post_attn = res_post_mlp = None
    if "resid" in args.capture:
        res_embed = create_array(
            p("enc_mc_res_embed.zarr"),
            "x",
            shape=(N, C, T, D),
            chunks=(1, C, T, D),
            dtype=zdtype,
        )
        res_pre = create_array(
            p("enc_mc_res_pre_attn.zarr"),
            "x",
            shape=(N, C, L, T, D),
            chunks=(1, C, 1, T, D),
            dtype=zdtype,
        )
        res_post_attn = create_array(
            p("enc_mc_res_post_attn.zarr"),
            "x",
            shape=(N, C, L, T, D),
            chunks=(1, C, 1, T, D),
            dtype=zdtype,
        )
        res_post_mlp = create_array(
            p("enc_mc_res_post_mlp.zarr"),
            "x",
            shape=(N, C, L, T, D),
            chunks=(1, C, 1, T, D),
            dtype=zdtype,
        )

    qkv_hooks = EncoderQKVHooksLite(model) if "qkv" in args.capture else None
    resid_hooks = EncoderResidualHooksLite(model) if "resid" in args.capture else None

    pred_idx = np.full(N, -1, dtype=np.int64)
    pred_label = [None] * N
    is_correct = np.zeros(N, dtype=bool)
    choice_logits_all = [None] * N
    choice_probs_all = [None] * N

    input_ids_all = [None] * N
    attention_mask_all = [None] * N
    offset_mapping_all = [None] * N
    tokens_all = [None] * N
    question_all = [None] * N

    model.eval()
    try:
        with torch.no_grad():
            for i in range(0, N, args.batch_size):
                sl = slice(i, min(i + args.batch_size, N))
                bdf = df.iloc[sl]
                B = len(bdf)
                print(f"[batch] {sl.start}-{sl.stop} / {N}")

                questions = [_extract_question(t) for t in bdf["text"].tolist()]
                choices_text = [[ch["text"] for ch in chs] for chs in bdf["csqa_choices"].tolist()]

                first_sentences = []
                second_sentences = []
                for qi, q in enumerate(questions):
                    for cj in range(C):
                        first_sentences.append(q)
                        second_sentences.append(choices_text[qi][cj])

                tok_kwargs = dict(
                    padding="max_length",
                    truncation=True,
                    max_length=T,
                    return_tensors="pt",
                )
                if getattr(tok, "is_fast", False):
                    tok_kwargs["return_offsets_mapping"] = True

                enc = tok(first_sentences, second_sentences, **tok_kwargs)
                ids = enc["input_ids"].view(B, C, T)
                mask = enc["attention_mask"].view(B, C, T)

                batch = {
                    "input_ids": ids.to(device),
                    "attention_mask": mask.to(device),
                }
                if qkv_hooks:
                    qkv_hooks.clear()

                out = model(
                    **batch,
                    output_attentions=("attn" in args.capture),
                    output_hidden_states=("hidden" in args.capture),
                    return_dict=True,
                )

                logits = out.logits.detach().cpu()  # (B,C)
                probs = torch.softmax(logits, dim=-1)
                pidx = torch.argmax(logits, dim=-1)
                pred_idx[sl] = pidx.numpy().astype(np.int64)
                for bi in range(B):
                    row_i = sl.start + bi
                    pred_label[row_i] = labels[int(pred_idx[row_i])]
                    is_correct[row_i] = bool(pred_idx[row_i] == y_true[row_i])
                    choice_logits_all[row_i] = logits[bi].numpy().astype(np.float32).tolist()
                    choice_probs_all[row_i] = probs[bi].numpy().astype(np.float32).tolist()

                if "attn" in args.capture and out.attentions is not None:
                    # tuple[L] of (B*C,H,T,T)
                    A = np.stack([a.detach().cpu().numpy().reshape(B, C, H, T, T) for a in out.attentions], axis=2)
                    attn_arr[sl] = A.astype(np_dtype)  # (B,C,L,H,T,T)

                if "qkv" in args.capture and qkv_hooks is not None:
                    q, k, v = qkv_hooks.stack()  # (B*C,L,H,T,d)
                    q = q.numpy().reshape(B, C, L, H, T, d_head).astype(np_dtype)
                    k = k.numpy().reshape(B, C, L, H, T, d_head).astype(np_dtype)
                    v = v.numpy().reshape(B, C, L, H, T, d_head).astype(np_dtype)
                    q_arr[sl] = q
                    k_arr[sl] = k
                    v_arr[sl] = v

                if "hidden" in args.capture and hidden_arr is not None and out.hidden_states is not None:
                    Hs = torch.stack([h.detach().cpu() for h in out.hidden_states], dim=1)  # (B*C,L+1,T,D)
                    Hs = Hs.numpy().reshape(B, C, L + 1, T, D).astype(np_dtype)
                    hidden_arr[sl] = Hs

                if "resid" in args.capture and resid_hooks is not None:
                    emb = resid_hooks.pop_embed()  # (B*C,T,D)
                    pre, pattn, post = resid_hooks.pop_layers()

                    if emb is not None:
                        res_embed[sl] = emb.detach().cpu().numpy().reshape(B, C, T, D).astype(np_dtype)

                    def stack_layers(dct):
                        xs = []
                        for li in range(L):
                            x = dct.get(li)
                            if x is None:
                                x = torch.zeros((B * C, T, D), device=device, dtype=torch.float16)
                            xs.append(x)
                        return torch.stack(xs, dim=1).detach().cpu().numpy().reshape(B, C, L, T, D)

                    res_pre[sl] = stack_layers(pre).astype(np_dtype)
                    res_post_attn[sl] = stack_layers(pattn).astype(np_dtype)
                    res_post_mlp[sl] = stack_layers(post).astype(np_dtype)

                ids_np = ids.cpu().numpy().tolist()
                mask_np = mask.cpu().numpy().tolist()
                if "offset_mapping" in enc:
                    offs_np = enc["offset_mapping"].view(B, C, T, 2).cpu().numpy().tolist()
                else:
                    offs_np = [None] * B
                for bi in range(B):
                    row_i = sl.start + bi
                    input_ids_all[row_i] = ids_np[bi]
                    attention_mask_all[row_i] = mask_np[bi]
                    offset_mapping_all[row_i] = offs_np[bi] if offs_np is not None else None
                    tokens_all[row_i] = [tok.convert_ids_to_tokens(choice_ids) for choice_ids in ids_np[bi]]
                    question_all[row_i] = questions[bi]
    finally:
        if qkv_hooks is not None:
            qkv_hooks.remove()
        if resid_hooks is not None:
            resid_hooks.remove()

    df_tok = pd.DataFrame(
        {
            "example_id": df["example_id"].tolist(),
            "text": df["text"].tolist(),
            "question": question_all,
            "answerKey": df["answerKey"].tolist(),
            "label_idx": y_true.tolist(),
            "pred_idx": pred_idx.tolist(),
            "pred_label": pred_label,
            "is_correct": is_correct.tolist(),
            "choice_labels": [labels] * N,
            "choice_texts": [[ch["text"] for ch in chs] for chs in df["csqa_choices"].tolist()],
            "choice_logits": choice_logits_all,
            "choice_probs": choice_probs_all,
            "input_ids": input_ids_all,
            "attention_mask": attention_mask_all,
            "offset_mapping": offset_mapping_all,
            "tokens": tokens_all,
        }
    )
    df_tok.to_parquet(tmp_root / "tokens.parquet", index=False)

    acc = float(is_correct.mean())
    meta = {
        "run_id": final_root.name,
        "model": model_id,
        "arch": "enc",
        "objective": "mcq_classification",
        "dataset": "csqa",
        "split": args.split,
        "n_examples": N,
        "n_choices": C,
        "max_seq_len": T,
        "num_layers": L,
        "num_heads": H,
        "head_dim": d_head,
        "dtype": args.dtype,
        "capture": args.capture,
        "acc": acc,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with (tmp_root / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if final_root.exists():
        shutil.rmtree(final_root)
    shutil.move(tmp_root.as_posix(), final_root.as_posix())
    print(f"[done] wrote {final_root} | acc={acc:.4f}")


if __name__ == "__main__":
    main()
