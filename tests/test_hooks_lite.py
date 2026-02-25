import unittest

import torch
import torch.nn as nn

from src.models.encoder_hooks import EncoderQKVHooksLite, EncoderResidualHooksLite
from src.models.gpt2_hooks import GPT2QKVHooksLite, GPT2ResidualHooksLite


class _DummyGPT2Attention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.c_attn = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        # Return tuple so hooks exercise tuple-output branch.
        return (self.proj(x),)


class _DummyGPT2Block(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attn = _DummyGPT2Attention(dim)
        self.mlp = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        attn_out = self.attn(x)[0]
        x = x + attn_out
        x = x + self.mlp(x)
        return x


class _DummyGPT2Core(nn.Module):
    def __init__(self, layers: int, heads: int, dim: int, vocab: int = 32):
        super().__init__()
        self.config = type("Cfg", (), {"n_head": heads, "n_embd": dim})()
        self.wte = nn.Embedding(vocab, dim)
        self.drop = nn.Identity()
        self.h = nn.ModuleList([_DummyGPT2Block(dim) for _ in range(layers)])

    def forward(self, input_ids):
        x = self.drop(self.wte(input_ids))
        for blk in self.h:
            x = blk(x)
        return x


class _DummyGPT2Model(nn.Module):
    def __init__(self, layers: int = 2, heads: int = 2, dim: int = 8):
        super().__init__()
        self.transformer = _DummyGPT2Core(layers=layers, heads=heads, dim=dim)

    def forward(self, input_ids):
        return self.transformer(input_ids)


class _DummyEncoderSelf(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        # Build context from value projection to keep shape semantics.
        return self.value(x)


class _DummyEncoderAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.self = _DummyEncoderSelf(dim)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return self.out(self.self(x))


class _DummyEncoderLayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attention = _DummyEncoderAttention(dim)
        self.ff = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        x = self.attention(x)
        x = self.ff(x)
        return x


class _DummyEncoderCore(nn.Module):
    def __init__(self, layers: int, heads: int, dim: int, vocab: int = 32):
        super().__init__()
        self.config = type("Cfg", (), {"num_attention_heads": heads, "hidden_size": dim})()
        self.embeddings = nn.Embedding(vocab, dim)
        self.encoder = type("Enc", (), {})()
        self.encoder.layer = nn.ModuleList([_DummyEncoderLayer(dim) for _ in range(layers)])

    def forward(self, input_ids):
        x = self.embeddings(input_ids)
        for layer in self.encoder.layer:
            x = layer(x)
        return x


class _DummyEncoderModel(nn.Module):
    def __init__(self, layers: int = 2, heads: int = 2, dim: int = 8):
        super().__init__()
        self.roberta = _DummyEncoderCore(layers=layers, heads=heads, dim=dim)

    def forward(self, input_ids):
        return self.roberta(input_ids)


class TestGpt2HooksLite(unittest.TestCase):
    def test_qkv_stack_shape(self):
        model = _DummyGPT2Model(layers=2, heads=2, dim=8)
        hooks = GPT2QKVHooksLite(model)
        try:
            input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
            _ = model(input_ids)
            q, k, v = hooks.stack()
            self.assertEqual(tuple(q.shape), (2, 2, 2, 3, 4))
            self.assertEqual(tuple(k.shape), (2, 2, 2, 3, 4))
            self.assertEqual(tuple(v.shape), (2, 2, 2, 3, 4))
        finally:
            hooks.remove()

    def test_residual_hooks_capture(self):
        model = _DummyGPT2Model(layers=2, heads=2, dim=8)
        hooks = GPT2ResidualHooksLite(model)
        try:
            input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
            _ = model(input_ids)

            emb = hooks.pop_embed()
            pre, pattn, post = hooks.pop_layers()
            self.assertEqual(tuple(emb.shape), (1, 3, 8))
            self.assertEqual(sorted(pre.keys()), [0, 1])
            self.assertEqual(sorted(pattn.keys()), [0, 1])
            self.assertEqual(sorted(post.keys()), [0, 1])
            self.assertEqual(tuple(pre[0].shape), (1, 3, 8))
            self.assertEqual(tuple(post[1].shape), (1, 3, 8))
        finally:
            hooks.remove()


class TestEncoderHooksLite(unittest.TestCase):
    def test_qkv_stack_shape(self):
        model = _DummyEncoderModel(layers=2, heads=2, dim=8)
        hooks = EncoderQKVHooksLite(model)
        try:
            input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
            _ = model(input_ids)
            q, k, v = hooks.stack()
            self.assertEqual(tuple(q.shape), (2, 2, 2, 3, 4))
            self.assertEqual(tuple(k.shape), (2, 2, 2, 3, 4))
            self.assertEqual(tuple(v.shape), (2, 2, 2, 3, 4))
        finally:
            hooks.remove()

    def test_residual_hooks_capture(self):
        model = _DummyEncoderModel(layers=2, heads=2, dim=8)
        hooks = EncoderResidualHooksLite(model)
        try:
            input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
            _ = model(input_ids)
            emb = hooks.pop_embed()
            pre, pattn, post = hooks.pop_layers()
            self.assertEqual(tuple(emb.shape), (1, 3, 8))
            self.assertEqual(sorted(pre.keys()), [0, 1])
            self.assertEqual(sorted(pattn.keys()), [0, 1])
            self.assertEqual(sorted(post.keys()), [0, 1])
            self.assertEqual(tuple(pattn[0].shape), (1, 3, 8))
        finally:
            hooks.remove()

    def test_find_core_errors_for_invalid_model(self):
        with self.assertRaises(RuntimeError):
            EncoderQKVHooksLite._find_core(nn.Linear(4, 4))


if __name__ == "__main__":
    unittest.main()

