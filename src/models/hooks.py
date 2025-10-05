# src/models/hooks.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn


#  Encoder-only (BERT/RoBERTa) 

def _has_proj(m: nn.Module, kind: str) -> bool:
    if kind == "q":
        return hasattr(m, "query") or hasattr(m, "q_proj")
    if kind == "k":
        return hasattr(m, "key") or hasattr(m, "k_proj")
    if kind == "v":
        return hasattr(m, "value") or hasattr(m, "v_proj")
    return False

def _project(m: nn.Module, hidden_states: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "q":
        return m.query(hidden_states) if hasattr(m, "query") else m.q_proj(hidden_states)
    if kind == "k":
        return m.key(hidden_states) if hasattr(m, "key") else m.k_proj(hidden_states)
    if kind == "v":
        return m.value(hidden_states) if hasattr(m, "value") else m.v_proj(hidden_states)
    raise ValueError(kind)

def _num_heads_and_dim(m: nn.Module, q: torch.Tensor) -> Tuple[int, int]:
    if hasattr(m, "num_attention_heads") and hasattr(m, "attention_head_size"):
        return int(m.num_attention_heads), int(m.attention_head_size)
    hidden = q.shape[-1]
    if hasattr(m, "num_attention_heads"):
        H = int(m.num_attention_heads)
        return H, hidden // H
    raise AttributeError("Cannot infer num_attention_heads/attention_head_size")

def _to_heads(x: torch.Tensor, H: int, d: int) -> torch.Tensor:
    # (B, T, H*d) -> (B, H, T, d)
    B, T, _ = x.shape
    return x.view(B, T, H, d).permute(0, 2, 1, 3).contiguous()

@dataclass
class QKVBuffers:
    q: Dict[int, torch.Tensor]
    k: Dict[int, torch.Tensor]
    v: Dict[int, torch.Tensor]
    def clear(self) -> None:
        self.q.clear(); self.k.clear(); self.v.clear()

class QKVHooks:
    """Encoder-only Q/K/V from attention.self modules (BERT/RoBERTa)."""
    def __init__(self, model: nn.Module):
        self.model = model
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.layers: List[nn.Module] = []
        self.buffers = QKVBuffers(q={}, k={}, v={})
        # Discover attention.self modules
        for name, module in model.named_modules():
            if "attention" in name and "self" in name:
                if _has_proj(module, "q") and _has_proj(module, "k") and _has_proj(module, "v"):
                    self.layers.append(module)
        if not self.layers:
            raise RuntimeError("No encoder self-attention modules with query/key/value found.")
        for layer_idx, layer in enumerate(self.layers):
            h = layer.register_forward_hook(self._make_hook(layer_idx))
            self.handles.append(h)

    def _make_hook(self, layer_idx: int):
        def hook(mod: nn.Module, inputs, output):
            hidden_states: torch.Tensor = inputs[0]
            with torch.no_grad():
                q_lin = _project(mod, hidden_states, "q")
                k_lin = _project(mod, hidden_states, "k")
                v_lin = _project(mod, hidden_states, "v")
                H, d = _num_heads_and_dim(mod, q_lin)
                self.buffers.q[layer_idx] = _to_heads(q_lin, H, d).detach().cpu()
                self.buffers.k[layer_idx] = _to_heads(k_lin, H, d).detach().cpu()
                self.buffers.v[layer_idx] = _to_heads(v_lin, H, d).detach().cpu()
        return hook

    def clear(self) -> None: self.buffers.clear()
    @property
    def L(self) -> int: return len(self.layers)

    def stack(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(self.buffers.q) != self.L:
            missing = [i for i in range(self.L) if i not in self.buffers.q]
            raise RuntimeError(f"Missing QKV for layers: {missing}")
        q = torch.stack([self.buffers.q[i] for i in range(self.L)], dim=1)  # (B,L,H,T,d)
        k = torch.stack([self.buffers.k[i] for i in range(self.L)], dim=1)
        v = torch.stack([self.buffers.v[i] for i in range(self.L)], dim=1)
        return q, k, v

    def remove(self) -> None:
        for h in self.handles: h.remove()
        self.handles.clear()


# Decoder-only (GPT-2)

def _gpt2_heads_dims(m: nn.Module) -> Tuple[int, int]:
    if hasattr(m, "num_heads") and hasattr(m, "head_dim"):
        return int(m.num_heads), int(m.head_dim)
    if hasattr(m, "n_head") and hasattr(m, "split_size"):
        H = int(m.n_head)
        D = (int(m.split_size) // 3) // H
        return H, D
    if hasattr(m, "c_attn") and hasattr(m.c_attn, "weight"):
        out = m.c_attn.weight.shape[0]  # 3*embed_dim
        embed_dim = out // 3
        if hasattr(m, "n_head"):
            H = int(m.n_head)
            return H, embed_dim // H
    raise AttributeError("Cannot infer GPT-2 num_heads/head_dim")

def _to_heads_gpt2(x: torch.Tensor, H: int, d: int) -> torch.Tensor:
    B, T, _ = x.shape
    return x.view(B, T, H, d).permute(0, 2, 1, 3).contiguous()

class GPT2QKVHooks:
    """Decoder-only Q/K/V from GPT-2 self-attention modules (no cross-attn)."""
    def __init__(self, model: nn.Module):
        self.model = model
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.layers: List[nn.Module] = []
        self.q: Dict[int, torch.Tensor] = {}
        self.k: Dict[int, torch.Tensor] = {}
        self.v: Dict[int, torch.Tensor] = {}
        for name, module in model.named_modules():
            if hasattr(module, "c_attn") and not getattr(module, "is_cross_attention", False):
                self.layers.append(module)
        if not self.layers:
            raise RuntimeError("No GPT-2 decoder self-attention modules found.")
        for layer_idx, layer in enumerate(self.layers):
            h = layer.register_forward_hook(self._make_hook(layer_idx))
            self.handles.append(h)

    def _make_hook(self, layer_idx: int):
        def hook(mod: nn.Module, inputs, output):
            hidden_states: torch.Tensor = inputs[0]  # (B,T,D)
            with torch.no_grad():
                qkv = mod.c_attn(hidden_states)      # (B,T,3*D)
                H, d = _gpt2_heads_dims(mod)
                D = H * d
                q_lin, k_lin, v_lin = qkv.split(D, dim=-1)
                self.q[layer_idx] = _to_heads_gpt2(q_lin, H, d).detach().cpu()
                self.k[layer_idx] = _to_heads_gpt2(k_lin, H, d).detach().cpu()
                self.v[layer_idx] = _to_heads_gpt2(v_lin, H, d).detach().cpu()
        return hook

    def clear(self) -> None:
        self.q.clear(); self.k.clear(); self.v.clear()

    @property
    def L(self) -> int: return len(self.layers)

    def stack(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(self.q) != self.L:
            missing = [i for i in range(self.L) if i not in self.q]
            raise RuntimeError(f"Missing QKV for layers: {missing}")
        q = torch.stack([self.q[i] for i in range(self.L)], dim=1)  # (B,L,H,T,d)
        k = torch.stack([self.k[i] for i in range(self.L)], dim=1)
        v = torch.stack([self.v[i] for i in range(self.L)], dim=1)
        return q, k, v

    def remove(self) -> None:
        for h in self.handles: h.remove()
        self.handles.clear()


# Encoder–Decoder (T5)

def _t5_heads_dims(m: nn.Module) -> Tuple[int, int]:
    # T5Attention exposes n_heads and key_value_proj_dim
    if hasattr(m, "n_heads") and hasattr(m, "key_value_proj_dim"):
        return int(m.n_heads), int(m.key_value_proj_dim)
    # Fallback: try relative dims
    if hasattr(m, "n_heads") and hasattr(m, "inner_dim"):
        H = int(m.n_heads)
        d = int(m.inner_dim) // H
        return H, d
    raise AttributeError("Cannot infer T5 heads/d_head")

def _to_heads_t5(x: torch.Tensor, H: int, d: int) -> torch.Tensor:
    B, T, _ = x.shape
    return x.view(B, T, H, d).permute(0, 2, 1, 3).contiguous()

class T5QKVHooks:
    """
    Collect Q/K/V from T5Attention modules for:
      - encoder self-attn
      - decoder self-attn
      - decoder cross-attn
    """
    def __init__(self, model: nn.Module):
        self.enc_self: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self.dec_self: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self.dec_cross: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self.enc_layers: List[int] = []
        self.dec_self_layers: List[int] = []
        self.dec_cross_layers: List[int] = []
        self.handles: List[torch.utils.hooks.RemovableHandle] = []

        # Register on all T5Attention modules (Self & Cross)
        layer_counters = {"enc": 0, "dec_self": 0, "dec_cross": 0}
        for name, module in model.named_modules():
            # Heuristic: T5Attention modules have attributes q,k,v and (maybe) is_cross_attention
            if not (hasattr(module, "q") and hasattr(module, "k") and hasattr(module, "v")):
                continue
            is_cross = bool(getattr(module, "is_cross_attention", False))
            is_dec = bool(getattr(module, "is_decoder", False))
            if is_cross and is_dec:
                idx = layer_counters["dec_cross"]; layer_counters["dec_cross"] += 1
                self.dec_cross_layers.append(idx)
                self.handles.append(module.register_forward_hook(self._make_hook(kind="dec_cross", layer_idx=idx)))
            elif (not is_cross) and is_dec:
                idx = layer_counters["dec_self"]; layer_counters["dec_self"] += 1
                self.dec_self_layers.append(idx)
                self.handles.append(module.register_forward_hook(self._make_hook(kind="dec_self", layer_idx=idx)))
            elif (not is_cross) and (not is_dec):
                idx = layer_counters["enc"]; layer_counters["enc"] += 1
                self.enc_layers.append(idx)
                self.handles.append(module.register_forward_hook(self._make_hook(kind="enc", layer_idx=idx)))
            # else: unexpected

    def _make_hook(self, kind: str, layer_idx: int):
        def hook(mod: nn.Module, inputs, output):
            # T5Attention.forward(hidden_states, mask=None, key_value_states=None, ...)
            hidden_states: torch.Tensor = inputs[0]
            key_value_states: Optional[torch.Tensor] = None
            if len(inputs) >= 3:
                key_value_states = inputs[2]
            with torch.no_grad():
                H, d = _t5_heads_dims(mod)
                if kind == "dec_cross" and key_value_states is not None:
                    q_lin = mod.q(hidden_states)
                    k_lin = mod.k(key_value_states)
                    v_lin = mod.v(key_value_states)
                else:
                    # self-attn on encoder or decoder
                    q_lin = mod.q(hidden_states)
                    k_lin = mod.k(hidden_states if key_value_states is None else key_value_states)
                    v_lin = mod.v(hidden_states if key_value_states is None else key_value_states)
                qh = _to_heads_t5(q_lin, H, d).detach().cpu()
                kh = _to_heads_t5(k_lin, H, d).detach().cpu()
                vh = _to_heads_t5(v_lin, H, d).detach().cpu()
                if kind == "enc":
                    self.enc_self[layer_idx] = (qh, kh, vh)
                elif kind == "dec_self":
                    self.dec_self[layer_idx] = (qh, kh, vh)
                elif kind == "dec_cross":
                    self.dec_cross[layer_idx] = (qh, kh, vh)
        return hook

    def clear(self) -> None:
        self.enc_self.clear(); self.dec_self.clear(); self.dec_cross.clear()

    def remove(self) -> None:
        for h in self.handles: h.remove()
        self.handles.clear()

    def stack(self, which: str):
        """
        which ∈ {"enc_self","dec_self","dec_cross"} → returns (q,k,v) each (B,L,H,T,[T_enc?],d) flattened to (B,L,H,T,d)
        For cross, returns (B,L,H,T_dec,d) for Q and (B,L,H,T_enc,d) for K/V when consumed later per-need.
        Here we keep consistent (B,L,H,T,d) per-tensor so downstream can store by token length.
        """
        if which == "enc_self":
            layers = sorted(self.enc_self.keys())
            q = torch.stack([self.enc_self[i][0] for i in layers], dim=1)
            k = torch.stack([self.enc_self[i][1] for i in layers], dim=1)
            v = torch.stack([self.enc_self[i][2] for i in layers], dim=1)
            return q, k, v
        if which == "dec_self":
            layers = sorted(self.dec_self.keys())
            q = torch.stack([self.dec_self[i][0] for i in layers], dim=1)
            k = torch.stack([self.dec_self[i][1] for i in layers], dim=1)
            v = torch.stack([self.dec_self[i][2] for i in layers], dim=1)
            return q, k, v
        if which == "dec_cross":
            layers = sorted(self.dec_cross.keys())
            q = torch.stack([self.dec_cross[i][0] for i in layers], dim=1)
            k = torch.stack([self.dec_cross[i][1] for i in layers], dim=1)
            v = torch.stack([self.dec_cross[i][2] for i in layers], dim=1)
            return q, k, v
        raise ValueError(which)


# MLP activations (encoder/decoder both)

class MLPHooks:
    """
    Capture intermediate (post-activation) outputs from each Transformer layer's MLP.
    On each forward() call, .stack(pool="mean"|"max"|None) returns:
      - if pool is None: (B, L, T, D_intermediate)
      - else: (B, L, T) pooled scalars
    Works for BERT/RoBERTa (.intermediate), GPT-2 (.mlp), and T5 (.DenseReluDense).
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self.layers: List[nn.Module] = []
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.buffers: Dict[int, torch.Tensor] = {}

        # encoders (BERT/RoBERTa)
        for name, module in model.named_modules():
            if name.endswith(".intermediate"):
                self.layers.append(module)
        # GPT-2 style
        if not self.layers:
            for name, module in model.named_modules():
                if name.endswith(".mlp"):
                    self.layers.append(module)
        # T5 style feed-forward
        if not self.layers:
            for name, module in model.named_modules():
                if name.endswith(".DenseReluDense"):
                    self.layers.append(module)

        if not self.layers:
            raise RuntimeError("Could not find MLP/Intermediate modules to hook.")

        for layer_idx, layer in enumerate(self.layers):
            h = layer.register_forward_hook(self._make_hook(layer_idx))
            self.handles.append(h)

    def _make_hook(self, layer_idx: int):
        def hook(mod: nn.Module, inputs, output):
            act: torch.Tensor = output if isinstance(output, torch.Tensor) else inputs[0]
            self.buffers[layer_idx] = act.detach().to("cpu")
        return hook

    def clear(self) -> None:
        self.buffers.clear()

    @property
    def L(self) -> int:
        return len(self.buffers) if self.buffers else 0

    def stack(self, pool: Optional[str] = "mean") -> torch.Tensor:
        if not self.buffers:
            raise RuntimeError("No MLP activations captured yet.")
        keys = sorted(self.buffers.keys())
        acts = [self.buffers[i] for i in keys]  # list of (B, T, D)
        x = torch.stack(acts, dim=1)  # (B, L, T, D)
        if pool is None:
            return x
        if pool == "mean":
            return x.mean(dim=-1)  # (B, L, T)
        if pool == "max":
            return x.amax(dim=-1)  # (B, L, T)
        raise ValueError(f"Unknown pool: {pool}")

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()
