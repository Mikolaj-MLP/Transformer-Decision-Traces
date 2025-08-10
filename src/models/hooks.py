# src/models/hooks.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn


def _has_proj(m: nn.Module, kind: str) -> bool:
    # Support both classic BERT/RoBERTa (.query/.key/.value) and some sdpa variants (.q_proj/.k_proj/.v_proj)
    if kind == "q":
        return hasattr(m, "query") or hasattr(m, "q_proj")
    if kind == "k":
        return hasattr(m, "key") or hasattr(m, "k_proj")
    if kind == "v":
        return hasattr(m, "value") or hasattr(m, "v_proj")
    return False


def _project(m: nn.Module, hidden_states: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "q":
        if hasattr(m, "query"):
            return m.query(hidden_states)
        return m.q_proj(hidden_states)
    if kind == "k":
        if hasattr(m, "key"):
            return m.key(hidden_states)
        return m.k_proj(hidden_states)
    if kind == "v":
        if hasattr(m, "value"):
            return m.value(hidden_states)
        return m.v_proj(hidden_states)
    raise ValueError(kind)


def _num_heads_and_dim(m: nn.Module, q: torch.Tensor) -> Tuple[int, int]:
    # Prefer module attributes; fall back to infer from tensor shape
    if hasattr(m, "num_attention_heads") and hasattr(m, "attention_head_size"):
        return int(m.num_attention_heads), int(m.attention_head_size)
    # Infer: last dim is H * d_head
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
        self.q.clear()
        self.k.clear()
        self.v.clear()


class QKVHooks:
    """
    Attach forward hooks to all self-attention blocks in an HF encoder-only model
    (e.g., BERT/RoBERTa). On each forward() call, call .stack() to get
    (B, L, H, T, d_head) tensors for q, k, v.

    Usage:
        hooks = QKVHooks(model)
        ...
        hooks.clear()
        _ = model(**inputs)
        q, k, v = hooks.stack()
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.layers: List[nn.Module] = []
        self.buffers = QKVBuffers(q={}, k={}, v={})

        # Discover attention.self modules in order
        for name, module in model.named_modules():
            # Typical path: encoder.layer.<i>.attention.self
            if "attention" in name and "self" in name and isinstance(module, nn.Module):
                if _has_proj(module, "q") and _has_proj(module, "k") and _has_proj(module, "v"):
                    self.layers.append(module)

        if not self.layers:
            raise RuntimeError("Could not locate any self-attention modules with query/key/value projections.")

        # Register hooks
        for layer_idx, layer in enumerate(self.layers):
            h = layer.register_forward_hook(self._make_hook(layer_idx))
            self.handles.append(h)

    def _make_hook(self, layer_idx: int):
        def hook(mod: nn.Module, inputs, output):
            # inputs[0] is hidden_states for HF attention blocks
            hidden_states: torch.Tensor = inputs[0]
            with torch.no_grad():
                q_lin = _project(mod, hidden_states, "q")
                k_lin = _project(mod, hidden_states, "k")
                v_lin = _project(mod, hidden_states, "v")
                H, d = _num_heads_and_dim(mod, q_lin)
                qh = _to_heads(q_lin, H, d)
                kh = _to_heads(k_lin, H, d)
                vh = _to_heads(v_lin, H, d)
                # Store on CPU to ease memory pressure between batches
                self.buffers.q[layer_idx] = qh.detach().to("cpu")
                self.buffers.k[layer_idx] = kh.detach().to("cpu")
                self.buffers.v[layer_idx] = vh.detach().to("cpu")
        return hook

    def clear(self) -> None:
        self.buffers.clear()

    @property
    def L(self) -> int:
        return len(self.layers)

    def stack(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Stack per-layer dicts into (B, L, H, T, d) tensors."""
        if len(self.buffers.q) != self.L:
            missing = [i for i in range(self.L) if i not in self.buffers.q]
            raise RuntimeError(f"Missing QKV for layers: {missing}")
        # Preserve layer order
        q_list = [self.buffers.q[i] for i in range(self.L)]
        k_list = [self.buffers.k[i] for i in range(self.L)]
        v_list = [self.buffers.v[i] for i in range(self.L)]
        # Shapes: each (B, H, T, d)
        q = torch.stack(q_list, dim=1)  # (B, L, H, T, d)
        k = torch.stack(k_list, dim=1)
        v = torch.stack(v_list, dim=1)
        return q, k, v

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()


# ---- Optional: MLP (intermediate) activations ----

class MLPHooks:
    """
    Capture intermediate (post-activation) outputs from each Transformer layer's MLP.
    Warning: full tensors are large; default to pooling to reduce size.

    On each forward() call, call .stack(pool="mean"|"max"|None) to obtain:
      - if pool is None: (B, L, T, D_intermediate)
      - else: (B, L, T) pooled scalars
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self.layers: List[nn.Module] = []
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.buffers: Dict[int, torch.Tensor] = {}

        # Typical path: encoder.layer.<i>.intermediate (Bert/RoBERTa)
        for name, module in model.named_modules():
            if name.endswith(".intermediate"):
                self.layers.append(module)

        if not self.layers:
            # Some models expose .mlp instead; try that
            for name, module in model.named_modules():
                if name.endswith(".mlp"):
                    self.layers.append(module)

        if not self.layers:
            raise RuntimeError("Could not find MLP intermediate modules to hook.")

        for layer_idx, layer in enumerate(self.layers):
            h = layer.register_forward_hook(self._make_hook(layer_idx))
            self.handles.append(h)

    def _make_hook(self, layer_idx: int):
        def hook(mod: nn.Module, inputs, output):
            # In HF BERT/RoBERTa, .intermediate returns post-activation activations (B, T, D_intermediate)
            act: torch.Tensor = output
            self.buffers[layer_idx] = act.detach().to("cpu")
        return hook

    def clear(self) -> None:
        self.buffers.clear()

    @property
    def L(self) -> int:
        return len(self.layers)

    def stack(self, pool: Optional[str] = "mean") -> torch.Tensor:
        if len(self.buffers) != self.L:
            missing = [i for i in range(self.L) if i not in self.buffers]
            raise RuntimeError(f"Missing MLP activations for layers: {missing}")
        acts = [self.buffers[i] for i in range(self.L)]  # list of (B, T, D)
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
