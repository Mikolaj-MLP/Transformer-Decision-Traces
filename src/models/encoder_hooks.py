from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


def _to_heads(x: torch.Tensor, H: int, d: int) -> torch.Tensor:
    # (B,T,H*d) -> (B,H,T,d)
    B, T, _ = x.shape
    return x.view(B, T, H, d).permute(0, 2, 1, 3).contiguous()


class EncoderQKVHooksLite:
    """
    Collect Q/K/V from encoder self-attention modules (BERT/RoBERTa-style).
    Stored per layer as CPU tensors, stack() -> (B, L, H, T, d_head).
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.core = self._find_core(model)
        self.layers: List[nn.Module] = list(self.core.encoder.layer)
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.q: Dict[int, torch.Tensor] = {}
        self.k: Dict[int, torch.Tensor] = {}
        self.v: Dict[int, torch.Tensor] = {}

        self.H = int(getattr(self.core.config, "num_attention_heads", 12))
        self.D = int(getattr(self.core.config, "hidden_size", 768))
        self.d_head = self.D // self.H

        for li, layer in enumerate(self.layers):
            self_attn = layer.attention.self
            self.handles.append(self_attn.register_forward_hook(self._make_hook(li)))

    @staticmethod
    def _find_core(model: nn.Module):
        # RobertaForMultipleChoice -> model.roberta
        # BertForMultipleChoice    -> model.bert
        for name in ("roberta", "bert", "deberta", "electra"):
            core = getattr(model, name, None)
            if core is not None and hasattr(core, "encoder") and hasattr(core.encoder, "layer"):
                return core
        if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
            return model
        raise RuntimeError("Could not locate encoder core with encoder.layer modules.")

    def _make_hook(self, li: int):
        def hook(mod: nn.Module, inputs, _output):
            x = inputs[0]  # (B,T,D)
            with torch.no_grad():
                q_lin = mod.query(x)
                k_lin = mod.key(x)
                v_lin = mod.value(x)
                self.q[li] = _to_heads(q_lin, self.H, self.d_head).detach().cpu()
                self.k[li] = _to_heads(k_lin, self.H, self.d_head).detach().cpu()
                self.v[li] = _to_heads(v_lin, self.H, self.d_head).detach().cpu()

        return hook

    def clear(self) -> None:
        self.q.clear()
        self.k.clear()
        self.v.clear()

    def stack(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        L = len(self.layers)
        if len(self.q) != L:
            missing = [i for i in range(L) if i not in self.q]
            raise RuntimeError(f"Missing QKV for layers: {missing}")
        q = torch.stack([self.q[i] for i in range(L)], dim=1)
        k = torch.stack([self.k[i] for i in range(L)], dim=1)
        v = torch.stack([self.v[i] for i in range(L)], dim=1)
        return q, k, v

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()


class EncoderResidualHooksLite:
    """
    Capture encoder residual checkpoints:
      - embed: output of embeddings
      - pre[li]: input to encoder layer li
      - post_attn[li]: output of attention submodule in layer li
      - post_mlp[li]: output of full encoder layer li
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.core = EncoderQKVHooksLite._find_core(model)
        self.layers: List[nn.Module] = list(self.core.encoder.layer)

        self.embed: Optional[torch.Tensor] = None
        self.pre: Dict[int, torch.Tensor] = {}
        self.pattn: Dict[int, torch.Tensor] = {}
        self.post: Dict[int, torch.Tensor] = {}
        self.handles: List[torch.utils.hooks.RemovableHandle] = []

        if not hasattr(self.core, "embeddings"):
            raise RuntimeError("Encoder core has no embeddings module.")
        self.handles.append(self.core.embeddings.register_forward_hook(self._hook_embed))

        for li, layer in enumerate(self.layers):
            self.handles.append(layer.register_forward_pre_hook(self._make_pre_hook(li)))
            self.handles.append(layer.attention.register_forward_hook(self._make_attn_hook(li)))
            self.handles.append(layer.register_forward_hook(self._make_post_hook(li)))

    def _hook_embed(self, _mod, _inp, out):
        o = out[0] if isinstance(out, (tuple, list)) else out
        if torch.is_tensor(o):
            self.embed = o.detach()

    def _make_pre_hook(self, li: int):
        def _pre(_mod, inp):
            self.pre[li] = inp[0].detach()

        return _pre

    def _make_attn_hook(self, li: int):
        def _attn(_mod, _inp, out):
            o = out[0] if isinstance(out, (tuple, list)) else out
            if torch.is_tensor(o):
                self.pattn[li] = o.detach()

        return _attn

    def _make_post_hook(self, li: int):
        def _post(_mod, _inp, out):
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
        self.pre = {}
        self.pattn = {}
        self.post = {}
        return pre, pattn, post

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()
