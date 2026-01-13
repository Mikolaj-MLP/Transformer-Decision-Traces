# src/cli/extract_traces.py
from __future__ import annotations
import os, json, time, argparse, shutil
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig
from datasets import load_dataset
from src.data.load_csqa import load_csqa
from src.data.load_text import load_ud_ewt, load_go_emotions
from src.models.load import load_base
from src.models.hooks import QKVHooks, GPT2QKVHooks, T5QKVHooks, MLPHooks, ResidualHooks
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

    elif dataset == "csqa":

        df = load_csqa(split=split, limit=limit)
        # keep csqa metadata columns!
        keep = ["example_id", "text", "answerKey", "correct_idx", "csqa_choices"]
        return df[keep].reset_index(drop=True)
    else:
        raise ValueError("dataset must be 'ud_ewt' or 'go_emotions' or 'csqa'")



def parse_index_list(lst):
    if lst is None:
        return None
    if isinstance(lst, (list, tuple)):
        return [int(x) for x in lst]
    s = str(lst).replace(",", " ").split()
    return [int(x) for x in s]

def _validate_indices(name: str, idxs, maxn: int):
    """Drop out-of-range indices with a warning; error if nothing left."""
    if idxs is None:
        return None
    valid = [i for i in idxs if 0 <= i < maxn]
    if len(valid) < len(idxs):
        dropped = [i for i in idxs if i not in valid]
        print(f"[warn] dropping out-of-range {name} {dropped}; valid range = 0..{maxn-1}")
    if not valid:
        raise ValueError(f"No valid indices left for {name}; requested {idxs}, but max is {maxn-1}.")
    return valid

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, default=None, help="local checkpoint dir; overrides --model")
    ap.add_argument("--model", type=str, default="roberta-base")
    #ap.add_argument("--dataset", type=str, choices=["ud_ewt", "go_emotions"], default="ud_ewt")
    ap.add_argument("--dataset", type=str, choices=["ud_ewt", "go_emotions", "csqa"], default="ud_ewt")
    ap.add_argument("--split", type=str, default="validation")
    ap.add_argument("--limit", type=int, default=32)
    ap.add_argument("--max_seq_len", type=int, default=128)
    ap.add_argument("--dec_max_len", type=int, default=64, help="decoder seq length when targets are provided")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument(
        "--capture",
        nargs="+",
        default=["attn", "qkv"],
        choices=["attn", "qkv", "mlp", "hidden", "resid"],  # <--- added 'resid'
    )
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"])

    # NEW: fine-grained selectors (side-specific)
    ap.add_argument("--layers", type=str, default=None, help="global layers fallback, e.g. '0 1 6 11'")
    ap.add_argument("--heads", type=str, default=None, help="global heads fallback, e.g. '0 3 7'")
    ap.add_argument("--layers-enc", type=str, default=None, help="encoder layers, e.g. '0,6,11'")
    ap.add_argument("--heads-enc", type=str, default=None, help="encoder heads, e.g. '0,3,7'")
    ap.add_argument("--layers-dec", type=str, default=None, help="decoder layers, e.g. '0,6,11'")
    ap.add_argument("--heads-dec", type=str, default=None, help="decoder heads, e.g. '0,3,7'")

    ap.add_argument("--targets_file", type=str, default=None, help="(enc-dec only) one target per line; aligns with inputs")
    ap.add_argument("--out_dir", type=str, default=None)
    args = ap.parse_args()

    texts = get_texts(args.dataset, args.split, args.limit)

    tok, model, device = load_base(args.model, model_path=args.model_path)
    cfg = AutoConfig.from_pretrained(args.model_path or args.model)

    #GPT-2 padding policy 
    # GPT-2 has no native pad token -> use eos as pad.
    if getattr(tok, "pad_token_id", None) is None:
        if getattr(tok, "eos_token_id", None) is None:
            raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id.")
        tok.pad_token = tok.eos_token

    # Force RIGHT padding so active tokens start at index 0.
    tok.padding_side = "right"

    # warn if something tries to override it later
    if tok.padding_side != "right":
        warnings.warn(f"tokenizer.padding_side is {tok.padding_side} (expected right)")

    # ARCH DETECTION
    is_encdec = bool(getattr(cfg, "is_encoder_decoder", False))
    model_type = str(getattr(cfg, "model_type", "")).lower()
    # treat these families as decoder-only even if cfg.is_decoder isn't set
    _decoder_families = {
        "gpt2","gptj","gpt_neox","llama","mistral","falcon","opt","bloom",
        "qwen2","xglm","mpt","phi","gemma","mixtral"
    }
    is_decoder_only = (not is_encdec) and (
        bool(getattr(cfg, "is_decoder", False)) or model_type in _decoder_families
    )
    arch = "dec" if is_decoder_only else ("encdec" if is_encdec else "enc")

    # --- TOKENIZER PADDING POLICY ---
    # Ensure a PAD token exists (GPT-2 et al. don't ship one)
    if getattr(tok, "pad_token_id", None) is None and getattr(tok, "eos_token_id", None) is not None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    # Tokenize encoder/source texts
    enc = tok(
        texts["text"].tolist(),
        padding="max_length",
        truncation=True,
        max_length=args.max_seq_len,
        return_tensors="pt",
        return_offsets_mapping=True,
    )

    # Decoder inputs for enc-dec 
    dec = None
    if arch == "encdec" and args.targets_file and os.path.exists(args.targets_file):
        with open(args.targets_file, "r", encoding="utf-8") as f:
            tgt_lines = [ln.rstrip("\n") for ln in f][: len(texts)]
        if len(tgt_lines) < len(texts):
            raise ValueError(f"targets_file has {len(tgt_lines)} lines, need {len(texts)}.")
        dec = tok(
            tgt_lines,
            padding="max_length",
            truncation=True,
            max_length=args.dec_max_len,
            return_tensors="pt",
            return_offsets_mapping=True,
        )

    # Shapes and dims per arch
    if arch == "enc":
        L_enc = int(cfg.num_hidden_layers)
        H_enc = int(cfg.num_attention_heads)
        T_enc = int(args.max_seq_len)
        d_head_enc = int(cfg.hidden_size) // H_enc
        D_enc = int(cfg.hidden_size)
        L_dec = H_dec = T_dec = d_head_dec = D_dec = 0
    elif arch == "dec":
        try:
            L_dec = int(getattr(cfg, "n_layer", cfg.num_hidden_layers))
            H_dec = int(getattr(cfg, "n_head", cfg.num_attention_heads))
            d_head_dec = int(getattr(cfg, "n_embd", cfg.hidden_size)) // H_dec
            D_dec = int(getattr(cfg, "n_embd", cfg.hidden_size))
        except Exception:
            L_dec = int(cfg.num_hidden_layers); H_dec = int(cfg.num_attention_heads)
            D_dec = int(cfg.hidden_size); d_head_dec = D_dec // H_dec
        T_dec = int(args.max_seq_len)
        L_enc = H_enc = T_enc = d_head_enc = D_enc = 0
    else:  # encdec (e.g., T5)
        L_enc = int(getattr(cfg, "num_layers", getattr(cfg, "num_encoder_layers", 12)))
        L_dec = int(getattr(cfg, "num_decoder_layers", L_enc))
        H_enc = H_dec = int(getattr(cfg, "num_heads", 12))
        d_model = int(getattr(cfg, "d_model", 512))
        d_head_enc = d_model // H_enc
        d_head_dec = d_head_enc
        D_enc = D_dec = d_model
        T_enc = int(args.max_seq_len)
        T_dec = int(args.dec_max_len)

    #per-side indices (fall back to global, then default all) and validate
    _raw_layers_enc = parse_index_list(args.layers_enc or args.layers)
    _raw_heads_enc  = parse_index_list(args.heads_enc  or args.heads)
    _raw_layers_dec = parse_index_list(args.layers_dec or args.layers)
    _raw_heads_dec  = parse_index_list(args.heads_dec  or args.heads)

    layer_idx_enc = _validate_indices("layers-enc", _raw_layers_enc, L_enc) if _raw_layers_enc is not None else list(range(L_enc or 0))
    head_idx_enc  = _validate_indices("heads-enc",  _raw_heads_enc,  H_enc) if _raw_heads_enc  is not None else list(range(H_enc or 0))
    layer_idx_dec = _validate_indices("layers-dec", _raw_layers_dec, L_dec) if _raw_layers_dec is not None else list(range(L_dec or 0))
    head_idx_dec  = _validate_indices("heads-dec",  _raw_heads_dec,  H_dec) if _raw_heads_dec  is not None else list(range(H_dec or 0))

    # Dtype for storage
    dtype = "f2" if args.dtype == "float16" else "f4"

    # Run id / atomic dirs
    run_id = f"{now_id()}_{args.model}_{args.dataset}_{args.split}_n{len(texts)}"
    final_root = Path(args.out_dir or f"traces/{run_id}")
    tmp_root = final_root.parent / f"_tmp_{final_root.name}"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_root.mkdir(parents=True, exist_ok=False)

    # Prepare storage under tmp_root
    def p(*parts): return os.path.join(tmp_root.as_posix(), *parts)

    # ATTENTION & QKV arrays per arch
    if arch == "enc":
        attn_arr = q_arr = k_arr = v_arr = None
        if "attn" in args.capture and L_enc and H_enc:
            attn_arr = create_array(p("attn.zarr"), "attn",
                                    shape=(len(texts), len(layer_idx_enc), len(head_idx_enc), T_enc, T_enc),
                                    chunks=(1, 1, 1, T_enc, T_enc), dtype=dtype)
        if "qkv" in args.capture and L_enc and H_enc:
            q_arr = create_array(p("qkv.zarr"), "q",
                                 shape=(len(texts), len(layer_idx_enc), len(head_idx_enc), T_enc, d_head_enc),
                                 chunks=(1, 1, 1, T_enc, d_head_enc), dtype=dtype)
            k_arr = create_array(p("qkv.zarr"), "k",
                                 shape=(len(texts), len(layer_idx_enc), len(head_idx_enc), T_enc, d_head_enc),
                                 chunks=(1, 1, 1, T_enc, d_head_enc), dtype=dtype)
            v_arr = create_array(p("qkv.zarr"), "v",
                                 shape=(len(texts), len(layer_idx_enc), len(head_idx_enc), T_enc, d_head_enc),
                                 chunks=(1, 1, 1, T_enc, d_head_enc), dtype=dtype)
        qkv_enc = QKVHooks(model) if "qkv" in args.capture and L_enc else None
        qkv_dec = None
        qkv_t5  = None
    elif arch == "dec":
        attn_arr = q_arr = k_arr = v_arr = None
        if "attn" in args.capture and L_dec and H_dec:
            attn_arr = create_array(p("dec_self_attn.zarr"), "attn",
                                    shape=(len(texts), len(layer_idx_dec), len(head_idx_dec), T_dec, T_dec),
                                    chunks=(1, 1, 1, T_dec, T_dec), dtype=dtype)
        if "qkv" in args.capture and L_dec and H_dec:
            q_arr = create_array(p("dec_self_qkv.zarr"), "q",
                                 shape=(len(texts), len(layer_idx_dec), len(head_idx_dec), T_dec, d_head_dec),
                                 chunks=(1, 1, 1, T_dec, d_head_dec), dtype=dtype)
            k_arr = create_array(p("dec_self_qkv.zarr"), "k",
                                 shape=(len(texts), len(layer_idx_dec), len(head_idx_dec), T_dec, d_head_dec),
                                 chunks=(1, 1, 1, T_dec, d_head_dec), dtype=dtype)
            v_arr = create_array(p("dec_self_qkv.zarr"), "v",
                                 shape=(len(texts), len(layer_idx_dec), len(head_idx_dec), T_dec, d_head_dec),
                                 chunks=(1, 1, 1, T_dec, d_head_dec), dtype=dtype)
        qkv_enc = None
        qkv_dec = GPT2QKVHooks(model) if "qkv" in args.capture and L_dec else None
        qkv_t5  = None
    else:
        enc_attn = enc_q = enc_k = enc_v = None
        decs_attn = decs_q = decs_k = decs_v = None
        decc_attn = decc_q = decc_k = decc_v = None
        if "attn" in args.capture and L_enc and H_enc:
            enc_attn = create_array(p("enc_self_attn.zarr"), "attn",
                                    shape=(len(texts), len(layer_idx_enc), len(head_idx_enc), T_enc, T_enc),
                                    chunks=(1, 1, 1, T_enc, T_enc), dtype=dtype)
            if dec is not None and L_dec and H_dec:
                decs_attn = create_array(p("dec_self_attn.zarr"), "attn",
                                         shape=(len(texts), len(layer_idx_dec), len(head_idx_dec), T_dec, T_dec),
                                         chunks=(1, 1, 1, T_dec, T_dec), dtype=dtype)
                decc_attn = create_array(p("dec_cross_attn.zarr"), "attn",
                                         shape=(len(texts), len(layer_idx_dec), len(head_idx_dec), T_dec, T_enc),
                                         chunks=(1, 1, 1, T_dec, T_enc), dtype=dtype)
        if "qkv" in args.capture and L_enc and H_enc:
            enc_q = create_array(p("enc_self_qkv.zarr"), "q",
                                 shape=(len(texts), len(layer_idx_enc), len(head_idx_enc), T_enc, d_head_enc),
                                 chunks=(1, 1, 1, T_enc, d_head_enc), dtype=dtype)
            enc_k = create_array(p("enc_self_qkv.zarr"), "k",
                                 shape=(len(texts), len(layer_idx_enc), len(head_idx_enc), T_enc, d_head_enc),
                                 chunks=(1, 1, 1, T_enc, d_head_enc), dtype=dtype)
            enc_v = create_array(p("enc_self_qkv.zarr"), "v",
                                 shape=(len(texts), len(layer_idx_enc), len(head_idx_enc), T_enc, d_head_enc),
                                 chunks=(1, 1, 1, T_enc, d_head_enc), dtype=dtype)
            if dec is not None and L_dec and H_dec:
                decs_q = create_array(p("dec_self_qkv.zarr"), "q",
                                      shape=(len(texts), len(layer_idx_dec), len(head_idx_dec), T_dec, d_head_dec),
                                      chunks=(1, 1, 1, T_dec, d_head_dec), dtype=dtype)
                decs_k = create_array(p("dec_self_qkv.zarr"), "k",
                                      shape=(len(texts), len(layer_idx_dec), len(head_idx_dec), T_dec, d_head_dec),
                                      chunks=(1, 1, 1, T_dec, d_head_dec), dtype=dtype)
                decs_v = create_array(p("dec_self_qkv.zarr"), "v",
                                      shape=(len(texts), len(layer_idx_dec), len(head_idx_dec), T_dec, d_head_dec),
                                      chunks=(1, 1, 1, T_dec, d_head_dec), dtype=dtype)
                decc_q = create_array(p("dec_cross_qkv.zarr"), "q",
                                      shape=(len(texts), len(layer_idx_dec), len(head_idx_dec), T_dec, d_head_dec),
                                      chunks=(1, 1, 1, T_dec, d_head_dec), dtype=dtype)
                decc_k = create_array(p("dec_cross_qkv.zarr"), "k",
                                      shape=(len(texts), len(layer_idx_dec), len(head_idx_dec), T_enc, d_head_dec),
                                      chunks=(1, 1, 1, T_enc, d_head_dec), dtype=dtype)
                decc_v = create_array(p("dec_cross_qkv.zarr"), "v",
                                      shape=(len(texts), len(layer_idx_dec), len(head_idx_dec), T_enc, d_head_dec),
                                      chunks=(1, 1, 1, T_enc, d_head_dec), dtype=dtype)
        qkv_t5  = T5QKVHooks(model) if "qkv" in args.capture else None
        qkv_enc = qkv_dec = None

    # Hidden states (optional)
    enc_hidden_arr = dec_hidden_arr = None
    if "hidden" in args.capture:
        if arch == "enc":
            enc_hidden_arr = create_array(p("hidden.zarr"), "h",
                                          shape=(len(texts), L_enc + 1, T_enc, D_enc),
                                          chunks=(1, 1, T_enc, D_enc), dtype=dtype)
        elif arch == "dec":
            dec_hidden_arr = create_array(p("dec_hidden.zarr"), "h",
                                          shape=(len(texts), L_dec + 1, T_dec, D_dec),
                                          chunks=(1, 1, T_dec, D_dec), dtype=dtype)
        else:
            enc_hidden_arr = create_array(p("enc_hidden.zarr"), "h",
                                          shape=(len(texts), L_enc + 1, T_enc, D_enc),
                                          chunks=(1, 1, T_enc, D_enc), dtype=dtype)
            if dec is not None:
                dec_hidden_arr = create_array(p("dec_hidden.zarr"), "h",
                                              shape=(len(texts), L_dec + 1, T_dec, D_dec),
                                              chunks=(1, 1, T_dec, D_dec), dtype=dtype)

    # Residual streams 
    # Encoder-only: res_embed / res_pre_attn / res_post_attn / res_post_mlp
    # Decoder-only: dec_res_*
    # Enc-Dec:      enc_res_* and dec_res_*
    res_enc_embed = res_enc_pre = res_enc_pattn = res_enc_post = None
    res_dec_embed = res_dec_pre = res_dec_pattn = res_dec_post = None
    if "resid" in args.capture:
        if arch == "enc":
            res_enc_embed  = create_array(p("res_embed.zarr"), "x",
                                          shape=(len(texts), T_enc, D_enc), chunks=(1, T_enc, D_enc), dtype=dtype)
            res_enc_pre    = create_array(p("res_pre_attn.zarr"), "x",
                                          shape=(len(texts), len(layer_idx_enc), T_enc, D_enc),
                                          chunks=(1, 1, T_enc, D_enc), dtype=dtype)
            res_enc_pattn  = create_array(p("res_post_attn.zarr"), "x",
                                          shape=(len(texts), len(layer_idx_enc), T_enc, D_enc),
                                          chunks=(1, 1, T_enc, D_enc), dtype=dtype)
            res_enc_post   = create_array(p("res_post_mlp.zarr"), "x",
                                          shape=(len(texts), len(layer_idx_enc), T_enc, D_enc),
                                          chunks=(1, 1, T_enc, D_enc), dtype=dtype)
        elif arch == "dec":
            res_dec_embed  = create_array(p("dec_res_embed.zarr"), "x",
                                          shape=(len(texts), T_dec, D_dec), chunks=(1, T_dec, D_dec), dtype=dtype)
            res_dec_pre    = create_array(p("dec_res_pre_attn.zarr"), "x",
                                          shape=(len(texts), len(layer_idx_dec), T_dec, D_dec),
                                          chunks=(1, 1, T_dec, D_dec), dtype=dtype)
            res_dec_pattn  = create_array(p("dec_res_post_attn.zarr"), "x",
                                          shape=(len(texts), len(layer_idx_dec), T_dec, D_dec),
                                          chunks=(1, 1, T_dec, D_dec), dtype=dtype)
            res_dec_post   = create_array(p("dec_res_post_mlp.zarr"), "x",
                                          shape=(len(texts), len(layer_idx_dec), T_dec, D_dec),
                                          chunks=(1, 1, T_dec, D_dec), dtype=dtype)
        else:
            res_enc_embed  = create_array(p("enc_res_embed.zarr"), "x",
                                          shape=(len(texts), T_enc, D_enc), chunks=(1, T_enc, D_enc), dtype=dtype)
            res_enc_pre    = create_array(p("enc_res_pre_attn.zarr"), "x",
                                          shape=(len(texts), len(layer_idx_enc), T_enc, D_enc),
                                          chunks=(1, 1, T_enc, D_enc), dtype=dtype)
            res_enc_pattn  = create_array(p("enc_res_post_attn.zarr"), "x",
                                          shape=(len(texts), len(layer_idx_enc), T_enc, D_enc),
                                          chunks=(1, 1, T_enc, D_enc), dtype=dtype)
            res_enc_post   = create_array(p("enc_res_post_mlp.zarr"), "x",
                                          shape=(len(texts), len(layer_idx_enc), T_enc, D_enc),
                                          chunks=(1, 1, T_enc, D_enc), dtype=dtype)
            if dec is not None:
                res_dec_embed  = create_array(p("dec_res_embed.zarr"), "x",
                                              shape=(len(texts), T_dec, D_dec), chunks=(1, T_dec, D_dec), dtype=dtype)
                res_dec_pre    = create_array(p("dec_res_pre_attn.zarr"), "x",
                                              shape=(len(texts), len(layer_idx_dec), T_dec, D_dec),
                                              chunks=(1, 1, T_dec, D_dec), dtype=dtype)
                res_dec_pattn  = create_array(p("dec_res_post_attn.zarr"), "x",
                                              shape=(len(texts), len(layer_idx_dec), T_dec, D_dec),
                                              chunks=(1, 1, T_dec, D_dec), dtype=dtype)
                res_dec_post   = create_array(p("dec_res_post_mlp.zarr"), "x",
                                              shape=(len(texts), len(layer_idx_dec), T_dec, D_dec),
                                              chunks=(1, 1, T_dec, D_dec), dtype=dtype)

    # Hooks
    qkv_enc = qkv_dec = qkv_t5 = None
    if "qkv" in args.capture:
        if arch == "enc":
            qkv_enc = QKVHooks(model) if L_enc else None
        elif arch == "dec":
            qkv_dec = GPT2QKVHooks(model) if L_dec else None
        else:
            qkv_t5  = T5QKVHooks(model)
    resid_hooks = ResidualHooks(model, arch) if "resid" in args.capture else None

    # Iterate
    N = len(texts)
    with torch.no_grad():
        for i in range(0, N, args.batch_size):
            sl = slice(i, min(i + args.batch_size, N))
            print(f"[batch] {sl.start}-{sl.stop} / {N}")

            # Build batch for this slice
            batch = {k: enc[k][sl].to(device) for k in ["input_ids", "attention_mask"] if k in enc}

            # Encoder–decoder models: attach decoder inputs
            if arch == "encdec":
                if dec is not None:
                    # Real targets provided
                    batch["decoder_input_ids"] = dec["input_ids"][sl].to(device)
                    if "attention_mask" in dec:
                        batch["decoder_attention_mask"] = dec["attention_mask"][sl].to(device)
                else:
                    # No targets ; provide a 1-token dummy so T5 runs the encoder cleanly
                    start_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
                    B = sl.stop - sl.start
                    d_ids = torch.full((B, 1), int(start_id), device=device, dtype=torch.long)
                    batch["decoder_input_ids"] = d_ids
                    batch["decoder_attention_mask"] = torch.ones_like(d_ids)

            # clear hooks
            if qkv_enc: qkv_enc.clear()
            if qkv_dec: qkv_dec.clear()
            if qkv_t5:  qkv_t5.clear()

            out = model(
                **batch,
                output_attentions=("attn" in args.capture),
                output_hidden_states=("hidden" in args.capture),
                return_dict=True,
            )

            # attentions
            if "attn" in args.capture:
                if arch == "enc" and out.attentions is not None:
                    attn_b = np.stack([a.detach().cpu().numpy() for a in out.attentions], axis=1)
                    attn_b = attn_b[:, layer_idx_enc][:, :, head_idx_enc]
                    attn_arr[sl] = attn_b.astype(np.float16 if dtype == "f2" else np.float32)
                elif arch == "dec" and out.attentions is not None:
                    attn_b = np.stack([a.detach().cpu().numpy() for a in out.attentions], axis=1)
                    attn_b = attn_b[:, layer_idx_dec][:, :, head_idx_dec]
                    attn_arr[sl] = attn_b.astype(np.float16 if dtype == "f2" else np.float32)
                elif arch == "encdec":
                    if getattr(out, "encoder_attentions", None) is not None and L_enc:
                        Eb = np.stack([a.detach().cpu().numpy() for a in out.encoder_attentions], axis=1)
                        Eb = Eb[:, layer_idx_enc][:, :, head_idx_enc]
                        enc_attn[sl] = Eb.astype(np.float16 if dtype == "f2" else np.float32)
                    if dec is not None and getattr(out, "decoder_attentions", None) is not None and L_dec:
                        Db = np.stack([a.detach().cpu().numpy() for a in out.decoder_attentions], axis=1)
                        Db = Db[:, layer_idx_dec][:, :, head_idx_dec]
                        decs_attn[sl] = Db.astype(np.float16 if dtype == "f2" else np.float32)
                    if dec is not None and getattr(out, "cross_attentions", None) is not None and L_dec:
                        Xb = np.stack([a.detach().cpu().numpy() for a in out.cross_attentions], axis=1)
                        Xb = Xb[:, layer_idx_dec][:, :, head_idx_dec]
                        decc_attn[sl] = Xb.astype(np.float16 if dtype == "f2" else np.float32)

            # QKV
            if "qkv" in args.capture:
                if arch == "enc" and qkv_enc is not None:
                    q, k, v = qkv_enc.stack()
                    q = q[:, layer_idx_enc][:, :, head_idx_enc]; k = k[:, layer_idx_enc][:, :, head_idx_enc]; v = v[:, layer_idx_enc][:, :, head_idx_enc]
                    q_arr[sl] = q.numpy().astype(np.float16 if dtype == "f2" else np.float32)
                    k_arr[sl] = k.numpy().astype(np.float16 if dtype == "f2" else np.float32)
                    v_arr[sl] = v.numpy().astype(np.float16 if dtype == "f2" else np.float32)
                elif arch == "dec" and qkv_dec is not None:
                    q, k, v = qkv_dec.stack()
                    q = q[:, layer_idx_dec][:, :, head_idx_dec]; k = k[:, layer_idx_dec][:, :, head_idx_dec]; v = v[:, layer_idx_dec][:, :, head_idx_dec]
                    q_arr[sl] = q.numpy().astype(np.float16 if dtype == "f2" else np.float32)
                    k_arr[sl] = k.numpy().astype(np.float16 if dtype == "f2" else np.float32)
                    v_arr[sl] = v.numpy().astype(np.float16 if dtype == "f2" else np.float32)
                elif arch == "encdec" and qkv_t5 is not None:
                    qE, kE, vE = qkv_t5.stack("enc_self")
                    qE = qE[:, layer_idx_enc][:, :, head_idx_enc]; kE = kE[:, layer_idx_enc][:, :, head_idx_enc]; vE = vE[:, layer_idx_enc][:, :, head_idx_enc]
                    enc_q[sl] = qE.numpy().astype(np.float16 if dtype == "f2" else np.float32)
                    enc_k[sl] = kE.numpy().astype(np.float16 if dtype == "f2" else np.float32)
                    enc_v[sl] = vE.numpy().astype(np.float16 if dtype == "f2" else np.float32)
                    if dec is not None and L_dec and H_dec:
                        qDs, kDs, vDs = qkv_t5.stack("dec_self")
                        qDs = qDs[:, layer_idx_dec][:, :, head_idx_dec]; kDs = kDs[:, layer_idx_dec][:, :, head_idx_dec]; vDs = vDs[:, layer_idx_dec][:, :, head_idx_dec]
                        decs_q[sl] = qDs.numpy().astype(np.float16 if dtype == "f2" else np.float32)
                        decs_k[sl] = kDs.numpy().astype(np.float16 if dtype == "f2" else np.float32)
                        decs_v[sl] = vDs.numpy().astype(np.float16 if dtype == "f2" else np.float32)
                        qX, kX, vX = qkv_t5.stack("dec_cross")
                        qX = qX[:, layer_idx_dec][:, :, head_idx_dec]
                        decc_q[sl] = qX.numpy().astype(np.float16 if dtype == "f2" else np.float32)
                        decc_k[sl] = kX.numpy().astype(np.float16 if dtype == "f2" else np.float32)
                        decc_v[sl] = vX.numpy().astype(np.float16 if dtype == "f2" else np.float32)

            # Hidden states
            if "hidden" in args.capture:
                if arch == "enc" and enc_hidden_arr is not None and out.hidden_states is not None:
                    Hs = torch.stack([h.detach().cpu() for h in out.hidden_states], dim=1)  # (B,L+1,T,D)
                    enc_hidden_arr[sl] = Hs.numpy().astype(np.float16 if dtype == "f2" else np.float32)
                elif arch == "dec" and dec_hidden_arr is not None and out.hidden_states is not None:
                    Hs = torch.stack([h.detach().cpu() for h in out.hidden_states], dim=1)
                    dec_hidden_arr[sl] = Hs.numpy().astype(np.float16 if dtype == "f2" else np.float32)
                elif arch == "encdec":
                    if enc_hidden_arr is not None and getattr(out, "encoder_hidden_states", None) is not None:
                        Hse = torch.stack([h.detach().cpu() for h in out.encoder_hidden_states], dim=1)
                        enc_hidden_arr[sl] = Hse.numpy().astype(np.float16 if dtype == "f2" else np.float32)
                    if dec_hidden_arr is not None and getattr(out, "decoder_hidden_states", None) is not None:
                        Hsd = torch.stack([h.detach().cpu() for h in out.decoder_hidden_states], dim=1)
                        dec_hidden_arr[sl] = Hsd.numpy().astype(np.float16 if dtype == "f2" else np.float32)

            # Residual stream checkpoints
            if "resid" in args.capture and resid_hooks is not None:
                emb = resid_hooks.pop_embed()
                pre, pattn, post = resid_hooks.pop_layers()  # dicts: layer_id -> (B,T,D)

                if arch == "enc":
                    if emb is not None and res_enc_embed is not None:
                        res_enc_embed[sl] = emb.detach().cpu().numpy().astype(np.float16 if dtype == "f2" else np.float32)
                    if res_enc_pre is not None:
                        pre_list = [pre.get(li, None) for li in layer_idx_enc]
                        if any(x is not None for x in pre_list):
                            pre_list = [x if x is not None else torch.zeros((sl.stop-sl.start, T_enc, D_enc), device=device, dtype=emb.dtype if emb is not None else out.last_hidden_state.dtype) for x in pre_list]
                            X = torch.stack(pre_list, dim=1).detach().cpu().numpy()
                            res_enc_pre[sl] = X.astype(np.float16 if dtype == "f2" else np.float32)
                    if res_enc_pattn is not None:
                        pa_list = [pattn.get(li, None) for li in layer_idx_enc]
                        if any(x is not None for x in pa_list):
                            pa_list = [x if x is not None else torch.zeros((sl.stop-sl.start, T_enc, D_enc), device=device, dtype=emb.dtype if emb is not None else out.last_hidden_state.dtype) for x in pa_list]
                            X = torch.stack(pa_list, dim=1).detach().cpu().numpy()
                            res_enc_pattn[sl] = X.astype(np.float16 if dtype == "f2" else np.float32)
                    if res_enc_post is not None:
                        po_list = [post.get(li, None) for li in layer_idx_enc]
                        if any(x is not None for x in po_list):
                            po_list = [x if x is not None else torch.zeros((sl.stop-sl.start, T_enc, D_enc), device=device, dtype=emb.dtype if emb is not None else out.last_hidden_state.dtype) for x in po_list]
                            X = torch.stack(po_list, dim=1).detach().cpu().numpy()
                            res_enc_post[sl] = X.astype(np.float16 if dtype == "f2" else np.float32)

                elif arch == "dec":
                    if emb is not None and res_dec_embed is not None:
                        res_dec_embed[sl] = emb.detach().cpu().numpy().astype(np.float16 if dtype == "f2" else np.float32)
                    if res_dec_pre is not None:
                        pre_list = [pre.get(li, None) for li in layer_idx_dec]
                        if any(x is not None for x in pre_list):
                            pre_list = [x if x is not None else torch.zeros((sl.stop-sl.start, T_dec, D_dec), device=device, dtype=emb.dtype if emb is not None else out.last_hidden_state.dtype) for x in pre_list]
                            X = torch.stack(pre_list, dim=1).detach().cpu().numpy()
                            res_dec_pre[sl] = X.astype(np.float16 if dtype == "f2" else np.float32)
                    if res_dec_pattn is not None:
                        pa_list = [pattn.get(li, None) for li in layer_idx_dec]
                        if any(x is not None for x in pa_list):
                            pa_list = [x if x is not None else torch.zeros((sl.stop-sl.start, T_dec, D_dec), device=device, dtype=emb.dtype if emb is not None else out.last_hidden_state.dtype) for x in pa_list]
                            X = torch.stack(pa_list, dim=1).detach().cpu().numpy()
                            res_dec_pattn[sl] = X.astype(np.float16 if dtype == "f2" else np.float32)
                    if res_dec_post is not None:
                        po_list = [post.get(li, None) for li in layer_idx_dec]
                        if any(x is not None for x in po_list):
                            po_list = [x if x is not None else torch.zeros((sl.stop-sl.start, T_dec, D_dec), device=device, dtype=emb.dtype if emb is not None else out.last_hidden_state.dtype) for x in po_list]
                            X = torch.stack(po_list, dim=1).detach().cpu().numpy()
                            res_dec_post[sl] = X.astype(np.float16 if dtype == "f2" else np.float32)

                else:  # encdec
                    # Encoder side
                    if res_enc_embed is not None and emb is not None:
                        # For encdec, ResidualHooks.embed is the encoder embedding output (only run encoder once per forward)
                        res_enc_embed[sl] = emb.detach().cpu().numpy().astype(np.float16 if dtype == "f2" else np.float32)
                    if res_enc_pre is not None:
                        pre_list = [pre.get(li, None) for li in layer_idx_enc]
                        if any(x is not None for x in pre_list):
                            pre_list = [x if x is not None else torch.zeros((sl.stop-sl.start, T_enc, D_enc), device=device, dtype=emb.dtype if emb is not None else out.last_hidden_state.dtype) for x in pre_list]
                            X = torch.stack(pre_list, dim=1).detach().cpu().numpy()
                            res_enc_pre[sl] = X.astype(np.float16 if dtype == "f2" else np.float32)
                    if res_enc_pattn is not None:
                        pa_list = [pattn.get(li, None) for li in layer_idx_enc]
                        if any(x is not None for x in pa_list):
                            pa_list = [x if x is not None else torch.zeros((sl.stop-sl.start, T_enc, D_enc), device=device, dtype=emb.dtype if emb is not None else out.last_hidden_state.dtype) for x in pa_list]
                            X = torch.stack(pa_list, dim=1).detach().cpu().numpy()
                            res_enc_pattn[sl] = X.astype(np.float16 if dtype == "f2" else np.float32)
                    if res_enc_post is not None:
                        po_list = [post.get(li, None) for li in layer_idx_enc]
                        if any(x is not None for x in po_list):
                            po_list = [x if x is not None else torch.zeros((sl.stop-sl.start, T_enc, D_enc), device=device, dtype=emb.dtype if emb is not None else out.last_hidden_state.dtype) for x in po_list]
                            X = torch.stack(po_list, dim=1).detach().cpu().numpy()
                            res_enc_post[sl] = X.astype(np.float16 if dtype == "f2" else np.float32)

                    # Decoder side 
                    if dec is not None:
                        if res_dec_embed is not None and emb is not None:
                            # Note: many enc-dec impls don't expose decoder "embed" via the same hook; ResidualHooks approximates it if available.
                            pass  
                        if res_dec_pre is not None:
                            pre_list = [pre.get(li, None) for li in layer_idx_dec]
                            if any(x is not None for x in pre_list):
                                pre_list = [x if x is not None else torch.zeros((sl.stop-sl.start, T_dec, D_dec), device=device, dtype=out.last_hidden_state.dtype) for x in pre_list]
                                X = torch.stack(pre_list, dim=1).detach().cpu().numpy()
                                res_dec_pre[sl] = X.astype(np.float16 if dtype == "f2" else np.float32)
                        if res_dec_pattn is not None:
                            pa_list = [pattn.get(li, None) for li in layer_idx_dec]
                            if any(x is not None for x in pa_list):
                                pa_list = [x if x is not None else torch.zeros((sl.stop-sl.start, T_dec, D_dec), device=device, dtype=out.last_hidden_state.dtype) for x in pa_list]
                                X = torch.stack(pa_list, dim=1).detach().cpu().numpy()
                                res_dec_pattn[sl] = X.astype(np.float16 if dtype == "f2" else np.float32)
                        if res_dec_post is not None:
                            po_list = [post.get(li, None) for li in layer_idx_dec]
                            if any(x is not None for x in po_list):
                                po_list = [x if x is not None else torch.zeros((sl.stop-sl.start, T_dec, D_dec), device=device, dtype=out.last_hidden_state.dtype) for x in po_list]
                                X = torch.stack(po_list, dim=1).detach().cpu().numpy()
                                res_dec_post[sl] = X.astype(np.float16 if dtype == "f2" else np.float32)

    # Save tokens + meta under tmp_root
    df_tok = pd.DataFrame({
        "example_id": texts["example_id"],
        "text": texts["text"],
        "input_ids": enc["input_ids"].cpu().tolist(),
        "attention_mask": enc["attention_mask"].cpu().tolist(),
        "offset_mapping": enc["offset_mapping"].cpu().tolist(),
    })
    # Ensure tokens length == T for every row
    input_ids_list = df_tok["input_ids"].tolist()
    df_tok["tokens"] = [tok.convert_ids_to_tokens(ids) for ids in input_ids_list]
    # Hard assertion (fail fast if something is wrong)
    T = len(input_ids_list[0]) if input_ids_list else 0
    assert all(isinstance(t, list) and len(t) == T for t in df_tok["tokens"]), \
        "tokens column must be list-of-length T for every example"
    # CSQA extras 
    if "answerKey" in texts.columns:
        df_tok["answerKey"] = texts["answerKey"].tolist()
    if "csqa_choices" in texts.columns:
        df_tok["csqa_choices"] = texts["csqa_choices"].tolist()
    # /
    if arch == "encdec" and dec is not None:
        df_tok["dec_input_ids"] = dec["input_ids"].cpu().tolist()
        df_tok["dec_attention_mask"] = dec["attention_mask"].cpu().tolist() if "attention_mask" in dec else None
        df_tok["dec_offset_mapping"] = dec["offset_mapping"].cpu().tolist()
        df_tok["dec_tokens"] = [tok.convert_ids_to_tokens(row) for row in df_tok["dec_input_ids"]]
    df_tok.to_parquet(tmp_root / "tokens.parquet", index=False)

    meta = {
        "run_id": final_root.name,
        "model": args.model,
        "arch": arch,
        "dataset": args.dataset,
        "split": args.split,
        "n_examples": len(texts),
        "max_seq_len": {"enc": T_enc, "dec": T_dec} if arch == "encdec" else (T_enc or T_dec),
        "num_layers": int(getattr(cfg, "num_hidden_layers", getattr(cfg, "num_layers", getattr(cfg, "n_layer", 12)))),
        "num_heads": int(getattr(cfg, "num_attention_heads", getattr(cfg, "num_heads", getattr(cfg, "n_head", 12)))),
        "head_dim": int(getattr(cfg, "hidden_size", getattr(cfg, "d_model", getattr(cfg, "n_embd", 768)))) // int(getattr(cfg, "num_attention_heads", getattr(cfg, "num_heads", getattr(cfg, "n_head", 12)))),
        "layers_stored": {"enc": layer_idx_enc, "dec": layer_idx_dec} if arch in ("encdec", "dec") else layer_idx_enc,
        "heads_stored": {"enc": head_idx_enc, "dec": head_idx_dec} if arch in ("encdec", "dec") else head_idx_enc,
        "dtype": args.dtype,
        "capture": args.capture,
        "has_targets": bool(dec is not None) if arch == "encdec" else None,
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
