# src/traces/store.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
import zarr


def _zget(g, name: str):
    try:
        return g[name]     # zarr v3+
    except Exception:
        try:
            return g.get(name, None)  # zarr v2
        except Exception:
            return None

class TraceStore:
    """
    Read-only accessor for a trace run directory.
    Encoder-only:
      - attn.zarr/attn
      - qkv.zarr/{q,k,v}
      - hidden.zarr/h (optional)
      - res_embed.zarr/x
      - res_pre_attn.zarr/x
      - res_post_attn.zarr/x
      - res_post_mlp.zarr/x

    Decoder-only:
      - dec_self_attn.zarr/attn
      - dec_self_qkv.zarr/{q,k,v}
      - dec_hidden.zarr/h (optional)
      - dec_res_embed.zarr/x
      - dec_res_pre_attn.zarr/x
      - dec_res_post_attn.zarr/x
      - dec_res_post_mlp.zarr/x

    Encoder–Decoder:
      - enc_self_attn.zarr/attn, enc_self_qkv.zarr/{q,k,v}
      - dec_self_attn.zarr/attn, dec_self_qkv.zarr/{q,k,v}
      - dec_cross_attn.zarr/attn, dec_cross_qkv.zarr/{q,k,v}
      - enc_hidden.zarr/h, dec_hidden.zarr/h (optional)
      - enc_res_* and dec_res_* (as above)
    """

    # init & meta 
    def __init__(self, run_dir: str):
        self.run_dir = Path(run_dir)
        if not self.run_dir.exists():
            raise FileNotFoundError(run_dir)

        with (self.run_dir / "meta.json").open("r", encoding="utf-8") as f:
            self.meta = json.load(f)
        self.tokens = pd.read_parquet(self.run_dir / "tokens.parquet")

        self.index_path = self.run_dir / "index.json"
        if self.index_path.exists():
            with self.index_path.open("r", encoding="utf-8") as f:
                self._ex2row = json.load(f)
        else:
            self._ex2row = {eid: i for i, eid in enumerate(self.tokens["example_id"].tolist())}

        # arrays
        self._attn_enc = self._q_enc = self._k_enc = self._v_enc = None
        self._attn_dec = self._q_dec = self._k_dec = self._v_dec = None
        self._attn_x   = self._q_x   = self._k_x   = self._v_x   = None  # cross-attn (dec→enc)
        self._hidden_enc = self._hidden_dec = None
        self._mc_attn = self._mc_q = self._mc_k = self._mc_v = None
        self._mc_hidden = None

        # residuals
        self._res_enc_embed = self._res_enc_pre = self._res_enc_pattn = self._res_enc_post = None
        self._res_dec_embed = self._res_dec_pre = self._res_dec_pattn = self._res_dec_post = None
        self._res_mc_embed = self._res_mc_pre = self._res_mc_pattn = self._res_mc_post = None

        self._load_arrays_if_exist()

    # properties
    @property
    def arch(self) -> str:
        return str(self.meta.get("arch", ""))

    @property
    def n(self) -> int:
        return int(self.meta.get("n_examples", len(self.tokens)))

    @property
    def T(self) -> int | Dict[str, int]:
        return self.meta.get("max_seq_len", 0)

    # Back-compat (single numbers). Prefer arrays() for exact per-side shapes.
    @property
    def L(self) -> int:
        return int(self.meta.get("num_layers", 0))

    @property
    def H(self) -> int:
        return int(self.meta.get("num_heads", 0))

    @property
    def d_head(self) -> int:
        return int(self.meta.get("head_dim", 0))

    def arrays(self) -> Dict[str, Tuple[int, ...]]:
        out: Dict[str, Tuple[int, ...]] = {}

        # Encoder self
        if self._attn_enc is not None: out["enc_self_attn"] = tuple(self._attn_enc.shape)
        if self._q_enc   is not None: out["enc_self_q"]    = tuple(self._q_enc.shape)
        if self._k_enc   is not None: out["enc_self_k"]    = tuple(self._k_enc.shape)
        if self._v_enc   is not None: out["enc_self_v"]    = tuple(self._v_enc.shape)

        # Decoder self
        if self._attn_dec is not None: out["dec_self_attn"] = tuple(self._attn_dec.shape)
        if self._q_dec    is not None: out["dec_self_q"]    = tuple(self._q_dec.shape)
        if self._k_dec    is not None: out["dec_self_k"]    = tuple(self._k_dec.shape)
        if self._v_dec    is not None: out["dec_self_v"]    = tuple(self._v_dec.shape)

        # Decoder cross
        if self._attn_x   is not None: out["dec_cross_attn"] = tuple(self._attn_x.shape)
        if self._q_x      is not None: out["dec_cross_q"]    = tuple(self._q_x.shape)
        if self._k_x      is not None: out["dec_cross_k"]    = tuple(self._k_x.shape)
        if self._v_x      is not None: out["dec_cross_v"]    = tuple(self._v_x.shape)

        # Hidden
        if self._hidden_enc is not None: out["enc_hidden"] = tuple(self._hidden_enc.shape)
        if self._hidden_dec is not None: out["dec_hidden"] = tuple(self._hidden_dec.shape)

        # Encoder MCQ
        if self._mc_attn is not None: out["enc_mc_attn"] = tuple(self._mc_attn.shape)
        if self._mc_q    is not None: out["enc_mc_q"]    = tuple(self._mc_q.shape)
        if self._mc_k    is not None: out["enc_mc_k"]    = tuple(self._mc_k.shape)
        if self._mc_v    is not None: out["enc_mc_v"]    = tuple(self._mc_v.shape)
        if self._mc_hidden is not None: out["enc_mc_hidden"] = tuple(self._mc_hidden.shape)

        # Residuals (encoder side)
        if self._res_enc_embed is not None: out["enc_res_embed"]    = tuple(self._res_enc_embed.shape)   # (N,T,D)
        if self._res_enc_pre   is not None: out["enc_res_pre_attn"] = tuple(self._res_enc_pre.shape)     # (N,L_sel,T,D)
        if self._res_enc_pattn is not None: out["enc_res_post_attn"]= tuple(self._res_enc_pattn.shape)   # (N,L_sel,T,D)
        if self._res_enc_post  is not None: out["enc_res_post_mlp"] = tuple(self._res_enc_post.shape)    # (N,L_sel,T,D)

        # Residuals (decoder side)
        if self._res_dec_embed is not None: out["dec_res_embed"]    = tuple(self._res_dec_embed.shape)   # (N,T,D)
        if self._res_dec_pre   is not None: out["dec_res_pre_attn"] = tuple(self._res_dec_pre.shape)     # (N,L_sel,T,D)
        if self._res_dec_pattn is not None: out["dec_res_post_attn"]= tuple(self._res_dec_pattn.shape)   # (N,L_sel,T,D)
        if self._res_dec_post  is not None: out["dec_res_post_mlp"] = tuple(self._res_dec_post.shape)    # (N,L_sel,T,D)

        # Residuals (encoder MCQ side)
        if self._res_mc_embed is not None: out["enc_mc_res_embed"] = tuple(self._res_mc_embed.shape)
        if self._res_mc_pre   is not None: out["enc_mc_res_pre_attn"] = tuple(self._res_mc_pre.shape)
        if self._res_mc_pattn is not None: out["enc_mc_res_post_attn"] = tuple(self._res_mc_pattn.shape)
        if self._res_mc_post  is not None: out["enc_mc_res_post_mlp"] = tuple(self._res_mc_post.shape)

        return out

    # meta helpers
    def has(self, example_id: str) -> bool:
        return example_id in self._ex2row

    def row(self, example_id: str) -> int:
        try:
            return int(self._ex2row[example_id])
        except KeyError:
            raise KeyError(f"example_id not found: {example_id}")

    def text(self, example_id: str) -> str:
        return str(self.tokens.iloc[self.row(example_id)]["text"])

    def encodings(self, example_id: str) -> Dict[str, Any]:
        i = self.row(example_id)
        r = self.tokens.iloc[i]
        out: Dict[str, Any] = {}
        for k in [
            "input_ids", "attention_mask", "offset_mapping", "tokens",
            "dec_input_ids", "dec_attention_mask", "dec_offset_mapping", "dec_tokens",
        ]:
            if k in r and r[k] is not None:
                out[k] = r[k]
        return out

    # accessors: attn/qkv/hidden
    def attn(self, example_id: str, layer: Optional[int] = None, head: Optional[int] = None,
             side: str = "enc", kind: str = "self") -> np.ndarray:
        """
        Returns:
          If layer is None: (L_sel, H_sel, Tq, Tk)
          If layer is int and head is None: (H_sel, Tq, Tk) for that layer
          If layer is int and head is int: (Tq, Tk) for that layer/head
        """
        arr = self._pick_attn(side, kind)
        if arr is None:
            raise RuntimeError("Requested attention array not present.")
        i = self.row(example_id)
        x = arr[i]  # (L,H,Tq,Tk)
        if layer is not None:
            x = x[layer]
            if head is not None:
                x = x[head]
        elif head is not None:
            x = x[:, head]
        return np.asarray(x)

    def qkv(self, example_id: str, which: str = "q", layer: Optional[int] = None, head: Optional[int] = None,
            side: str = "enc", kind: str = "self") -> np.ndarray:
        """
        Returns:
          If layer is None: (L_sel, H_sel, T, d)
          If layer is int and head is None: (H_sel, T, d) for that layer
          If layer is int and head is int: (T, d) for that layer/head
        """
        arr = self._pick_qkv(which, side, kind)
        if arr is None:
            raise RuntimeError(f"Requested {which.upper()} array not present.")
        i = self.row(example_id)
        x = arr[i]  # (L,H,T,d)
        if layer is not None:
            x = x[layer]
            if head is not None:
                x = x[head]
        elif head is not None:
            x = x[:, head]
        return np.asarray(x)

    def hidden(self, example_id: str, side: str = "enc") -> np.ndarray:
        """
        Returns hidden states tensor:
          - Encoder: (L+1, T, D)
          - Decoder: (L+1, T, D)
        """
        if side == "enc" and self._hidden_enc is not None:
            return np.asarray(self._hidden_enc[self.row(example_id)])
        if side == "dec" and self._hidden_dec is not None:
            return np.asarray(self._hidden_dec[self.row(example_id)])
        raise RuntimeError("Requested hidden states not present.")

    # accessors: encoder MCQ (choice-aware)
    def mcq_attn(self, example_id: str, choice_idx: int, layer: Optional[int] = None, head: Optional[int] = None) -> np.ndarray:
        """
        Encoder MCQ attention accessor.
        Stored shape: (N, C, L, H, T, T)
        """
        if self._mc_attn is None:
            raise RuntimeError("Requested enc_mc_attn not present.")
        i = self.row(example_id)
        x = self._mc_attn[i, choice_idx]  # (L,H,T,T)
        if layer is not None:
            x = x[layer]
            if head is not None:
                x = x[head]
        elif head is not None:
            x = x[:, head]
        return np.asarray(x)

    def mcq_qkv(self, example_id: str, choice_idx: int, which: str = "q", layer: Optional[int] = None, head: Optional[int] = None) -> np.ndarray:
        """
        Encoder MCQ Q/K/V accessor.
        Stored shape: (N, C, L, H, T, d)
        """
        arr = {"q": self._mc_q, "k": self._mc_k, "v": self._mc_v}[which]
        if arr is None:
            raise RuntimeError(f"Requested enc_mc_{which} not present.")
        i = self.row(example_id)
        x = arr[i, choice_idx]  # (L,H,T,d)
        if layer is not None:
            x = x[layer]
            if head is not None:
                x = x[head]
        elif head is not None:
            x = x[:, head]
        return np.asarray(x)

    def mcq_hidden(self, example_id: str, choice_idx: int) -> np.ndarray:
        """
        Encoder MCQ hidden accessor.
        Stored shape: (N, C, L+1, T, D)
        """
        if self._mc_hidden is None:
            raise RuntimeError("Requested enc_mc_hidden not present.")
        return np.asarray(self._mc_hidden[self.row(example_id), choice_idx])

    # accessors: residual streams
    def resid_embed(self, example_id: str, side: str = "enc") -> np.ndarray:
        """
        Return the embedding-output residual stream for the given side.
        Shape: (T, D)
        """
        i = self.row(example_id)
        if side == "enc" and self._res_enc_embed is not None:
            return np.asarray(self._res_enc_embed[i])
        if side == "dec" and self._res_dec_embed is not None:
            return np.asarray(self._res_dec_embed[i])
        raise RuntimeError("Requested residual embed not present.")

    def resid(self, example_id: str, stage: str, layer: Optional[int] = None, side: str = "enc") -> np.ndarray:
        """
        stage ∈ {"pre_attn", "post_attn", "post_mlp"}
        Returns:
          If layer is None: (L_sel, T, D)
          If layer is int:  (T, D) for that layer
        """
        i = self.row(example_id)
        arr = self._pick_resid(stage=stage, side=side)
        if arr is None:
            raise RuntimeError(f"Requested residual stage '{stage}' not present for side='{side}'.")
        x = arr[i]  # (L_sel, T, D)
        if layer is not None:
            x = x[layer]
        return np.asarray(x)

    def mcq_resid_embed(self, example_id: str, choice_idx: int) -> np.ndarray:
        """
        Encoder MCQ embedding residual stream.
        Stored shape: (N, C, T, D)
        """
        if self._res_mc_embed is None:
            raise RuntimeError("Requested enc_mc_res_embed not present.")
        return np.asarray(self._res_mc_embed[self.row(example_id), choice_idx])

    def mcq_resid(self, example_id: str, choice_idx: int, stage: str, layer: Optional[int] = None) -> np.ndarray:
        """
        stage in {"pre_attn", "post_attn", "post_mlp"}
        Stored shape for stage arrays: (N, C, L, T, D)
        """
        arr = None
        if stage == "pre_attn":
            arr = self._res_mc_pre
        elif stage == "post_attn":
            arr = self._res_mc_pattn
        elif stage == "post_mlp":
            arr = self._res_mc_post
        if arr is None:
            raise RuntimeError(f"Requested enc_mc residual stage '{stage}' not present.")
        x = arr[self.row(example_id), choice_idx]  # (L,T,D)
        if layer is not None:
            x = x[layer]
        return np.asarray(x)

    # internals: selectors
    def _pick_attn(self, side: str, kind: str):
        if side == "enc" and kind == "self":  return self._attn_enc
        if side == "dec" and kind == "self":  return self._attn_dec
        if side == "dec" and kind == "cross": return self._attn_x
        return None

    def _pick_qkv(self, which: str, side: str, kind: str):
        if side == "enc" and kind == "self":
            return {"q": self._q_enc, "k": self._k_enc, "v": self._v_enc}[which]
        if side == "dec" and kind == "self":
            return {"q": self._q_dec, "k": self._k_dec, "v": self._v_dec}[which]
        if side == "dec" and kind == "cross":
            return {"q": self._q_x, "k": self._k_x, "v": self._v_x}[which]
        return None

    def _pick_resid(self, stage: str, side: str):
        if side == "enc":
            if stage == "pre_attn":  return self._res_enc_pre
            if stage == "post_attn": return self._res_enc_pattn
            if stage == "post_mlp":  return self._res_enc_post
        elif side == "dec":
            if stage == "pre_attn":  return self._res_dec_pre
            if stage == "post_attn": return self._res_dec_pattn
            if stage == "post_mlp":  return self._res_dec_post
        return None

    #loader
    def _load_arrays_if_exist(self) -> None:
        rd = self.run_dir

        # encoder-only back-compat
        p = rd / "attn.zarr"
        if p.exists(): self._attn_enc = zarr.open(p, mode="r")["attn"]
        p = rd / "qkv.zarr"
        if p.exists():
            g = zarr.open(p, mode="r")
            self._q_enc = _zget(g, "q"); self._k_enc = _zget(g, "k"); self._v_enc = _zget(g, "v")
        p = rd / "hidden.zarr"
        if p.exists(): self._hidden_enc = zarr.open(p, mode="r")["h"]

        # decoder-only (GPT-2)
        p = rd / "dec_self_attn.zarr"
        if p.exists(): self._attn_dec = zarr.open(p, mode="r")["attn"]
        p = rd / "dec_self_qkv.zarr"
        if p.exists():
            g = zarr.open(p, mode="r")
            self._q_dec = _zget(g, "q"); self._k_dec = _zget(g, "k"); self._v_dec = _zget(g, "v")
        p = rd / "dec_hidden.zarr"
        if p.exists(): self._hidden_dec = zarr.open(p, mode="r")["h"]

        # encoder MCQ
        p = rd / "enc_mc_attn.zarr"
        if p.exists(): self._mc_attn = zarr.open(p, mode="r")["attn"]
        p = rd / "enc_mc_qkv.zarr"
        if p.exists():
            g = zarr.open(p, mode="r")
            self._mc_q = _zget(g, "q"); self._mc_k = _zget(g, "k"); self._mc_v = _zget(g, "v")
        p = rd / "enc_mc_hidden.zarr"
        if p.exists(): self._mc_hidden = zarr.open(p, mode="r")["h"]

        # encoder–decoder (T5)
        p = rd / "enc_self_attn.zarr"
        if p.exists(): self._attn_enc = zarr.open(p, mode="r")["attn"]
        p = rd / "enc_self_qkv.zarr"
        if p.exists():
            g = zarr.open(p, mode="r")
            self._q_enc = _zget(g, "q"); self._k_enc = _zget(g, "k"); self._v_enc = _zget(g, "v")

        p = rd / "dec_self_attn.zarr"
        if p.exists(): self._attn_dec = zarr.open(p, mode="r")["attn"]
        p = rd / "dec_self_qkv.zarr"
        if p.exists():
            g = zarr.open(p, mode="r")
            self._q_dec = _zget(g, "q"); self._k_dec = _zget(g, "k"); self._v_dec = _zget(g, "v")

        p = rd / "dec_cross_attn.zarr"
        if p.exists(): self._attn_x = zarr.open(p, mode="r")["attn"]
        p = rd / "dec_cross_qkv.zarr"
        if p.exists():
            g = zarr.open(p, mode="r")
            self._q_x = _zget(g, "q"); self._k_x = _zget(g, "k"); self._v_x = _zget(g, "v")

        p = rd / "enc_hidden.zarr"
        if p.exists(): self._hidden_enc = zarr.open(p, mode="r")["h"]
        p = rd / "dec_hidden.zarr"
        if p.exists(): self._hidden_dec = zarr.open(p, mode="r")["h"]

        # residuals: encoder-only or enc side 
        # Encoder-only run uses plain names; enc-dec runs use enc_* names.
        # load both when present; last assignment wins (enc_* will overwrite if both exist !!!!!!!!!!).
        # Embedding (N,T,D)
        for cand in ["res_embed.zarr", "enc_res_embed.zarr"]:
            p = rd / cand
            if p.exists():
                self._res_enc_embed = zarr.open(p, mode="r")["x"]
        # Layer checkpoints (N,L_sel,T,D)
        for fname, attr in [
            ("res_pre_attn.zarr",   "_res_enc_pre"),
            ("res_post_attn.zarr",  "_res_enc_pattn"),
            ("res_post_mlp.zarr",   "_res_enc_post"),
            ("enc_res_pre_attn.zarr",   "_res_enc_pre"),
            ("enc_res_post_attn.zarr",  "_res_enc_pattn"),
            ("enc_res_post_mlp.zarr",   "_res_enc_post"),
        ]:
            p = rd / fname
            if p.exists():
                setattr(self, attr, zarr.open(p, mode="r")["x"])

        # residuals: decoder side
        for fname, attr in [
            ("dec_res_embed.zarr",      "_res_dec_embed"),
            ("dec_res_pre_attn.zarr",   "_res_dec_pre"),
            ("dec_res_post_attn.zarr",  "_res_dec_pattn"),
            ("dec_res_post_mlp.zarr",   "_res_dec_post"),
        ]:
            p = rd / fname
            if p.exists():
                node = zarr.open(p, mode="r")["x"]
                setattr(self, attr, node)

        # residuals: encoder MCQ side
        for fname, attr in [
            ("enc_mc_res_embed.zarr",      "_res_mc_embed"),
            ("enc_mc_res_pre_attn.zarr",   "_res_mc_pre"),
            ("enc_mc_res_post_attn.zarr",  "_res_mc_pattn"),
            ("enc_mc_res_post_mlp.zarr",   "_res_mc_post"),
        ]:
            p = rd / fname
            if p.exists():
                setattr(self, attr, zarr.open(p, mode="r")["x"])

    # utilities
    def build_index(self, persist: bool = True) -> Dict[str, int]:
        self._ex2row = {eid: i for i, eid in enumerate(self.tokens["example_id"].tolist())}
        if persist:
            with self.index_path.open("w", encoding="utf-8") as f:
                json.dump(self._ex2row, f, indent=2)
        return self._ex2row
