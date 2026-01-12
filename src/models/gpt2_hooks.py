# src/models/gpt2_hooks.py
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn


def _to_heads(x: torch.Tensor, H: int, d: int) -> torch.Tensor:
    # (B,T,H*d) -> (B,H,T,d)
    B, T, _ = x.shape
    return x.view(B, T, H, d).permute(0, 2, 1, 3).contiguous()


class GPT2QKVHooksLite:
    """
    Collect Q/K/V for GPT-2 from each block.attn.c_attn(hidden_states).
    Stores tensors on CPU: (B, L, H, T, d_head) after stack().
    """
    def __init__(self, model: nn.Module):
        self.model = model
        core = getattr(model, "transformer", model)
        if not hasattr(core, "h"):
            raise RuntimeError("Expected GPT-2-like model with transformer.h blocks.")
        self.core = core
        self.blocks: List[nn.Module] = list(core.h)
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.q: Dict[int, torch.Tensor] = {}
        self.k: Dict[int, torch.Tensor] = {}
        self.v: Dict[int, torch.Tensor] = {}

        for li, block in enumerate(self.blocks):
            attn = getattr(block, "attn", None)
            if attn is None or not hasattr(attn, "c_attn"):
                raise RuntimeError(f"Block {li} has no attn.c_attn (not GPT-2).")
            self.handles.append(attn.register_forward_hook(self._make_hook(li)))

        # infer H, D 
        self.H = int(getattr(self.core.config, "n_head", getattr(self.core, "n_head", 12))) if hasattr(self.core, "config") else 12
        self.D = int(getattr(self.core.config, "n_embd", getattr(self.core, "n_embd", 768))) if hasattr(self.core, "config") else 768
        self.d_head = self.D // self.H

    def _make_hook(self, li: int):
        def hook(mod: nn.Module, inputs, output):
            # inputs[0] is hidden_states (B,T,D)
            x: torch.Tensor = inputs[0]
            with torch.no_grad():
                qkv = mod.c_attn(x)  # (B,T,3*D)
                D = self.D
                q_lin, k_lin, v_lin = qkv.split(D, dim=-1)
                self.q[li] = _to_heads(q_lin, self.H, self.d_head).detach().cpu()
                self.k[li] = _to_heads(k_lin, self.H, self.d_head).detach().cpu()
                self.v[li] = _to_heads(v_lin, self.H, self.d_head).detach().cpu()
        return hook

    def clear(self) -> None:
        self.q.clear(); self.k.clear(); self.v.clear()

    def stack(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        L = len(self.blocks)
        if len(self.q) != L:
            missing = [i for i in range(L) if i not in self.q]
            raise RuntimeError(f"Missing QKV for layers: {missing}")
        q = torch.stack([self.q[i] for i in range(L)], dim=1)  # (B,L,H,T,d)
        k = torch.stack([self.k[i] for i in range(L)], dim=1)
        v = torch.stack([self.v[i] for i in range(L)], dim=1)
        return q, k, v

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()


class GPT2ResidualHooksLite:
    """
    Capture GPT-2 residual checkpoints:
      - embed: output of transformer.drop (wte+wpe + dropout) => true resid stream after embeddings
      - pre[li]: input to block li (resid before ln_1)
      - post_attn[li]: x_pre + attn_out (resid after attn residual add)
      - post_mlp[li]: output of block (resid after mlp residual add)
    """
    def __init__(self, model: nn.Module):
        self.model = model
        core = getattr(model, "transformer", model)
        if not hasattr(core, "h"):
            raise RuntimeError("Expected GPT-2 model with transformer.h blocks.")
        self.core = core
        self.blocks: List[nn.Module] = list(core.h)

        self.embed: Optional[torch.Tensor] = None
        self.pre: Dict[int, torch.Tensor] = {}
        self.pattn: Dict[int, torch.Tensor] = {}
        self.post: Dict[int, torch.Tensor] = {}

        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        # embed stream = output of transformer.drop
        if hasattr(core, "drop"):
            self.handles.append(core.drop.register_forward_hook(self._hook_embed))
        else:
            raise RuntimeError("GPT-2 core has no transformer.drop")

        for li, block in enumerate(self.blocks):
            self.handles.append(block.register_forward_pre_hook(self._make_pre_hook(li)))
            self.handles.append(block.register_forward_hook(self._make_post_hook(li)))
            attn = getattr(block, "attn", None)
            if attn is None:
                raise RuntimeError(f"Block {li} has no attn module.")
            self.handles.append(attn.register_forward_hook(self._make_attn_hook(li)))

    def _hook_embed(self, mod, inp, out):
        o = out[0] if isinstance(out, (tuple, list)) else out
        if torch.is_tensor(o):
            self.embed = o.detach()

    def _make_pre_hook(self, li: int):
        def _pre(mod, inp):
            x = inp[0]
            self.pre[li] = x.detach()
        return _pre

    def _make_attn_hook(self, li: int):
        # GPT2Attention.forward returns (attn_output, present, (attn_weights))
        def _attn(mod, inp, out):
            x_pre = self.pre.get(li, None)
            o = out[0] if isinstance(out, (tuple, list)) else out
            if not torch.is_tensor(o):
                return
            if x_pre is not None:
                self.pattn[li] = (x_pre + o).detach()
            else:
                self.pattn[li] = o.detach()
        return _attn

    def _make_post_hook(self, li: int):
        def _post(mod, inp, out):
            o = out[0] if isinstance(out, (tuple, list)) else out
            if torch.is_tensor(o):
                self.post[li] = o.detach()
        return _post

    def pop_embed(self) -> Optional[torch.Tensor]:
        x = self.embed
        self.embed = None
        return x

    def pop_layers(self):
        pre, pattn, post = self.pre, self.pattn, self.post
        self.pre = {}; self.pattn = {}; self.post = {}
        return pre, pattn, post

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()
