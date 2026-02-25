import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

from src.traces_utils.store import TraceStore


class TestTraceStoreMinimalRuns(unittest.TestCase):
    def _write_tokens_parquet(self, run_dir: Path, df: pd.DataFrame) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(run_dir / "tokens.parquet", index=False)

    def _write_meta(self, run_dir: Path, meta: dict) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def test_decoder_hidden_and_encodings(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"

            meta = {
                "run_id": "test_dec",
                "model": "gpt2",
                "arch": "dec",
                "dataset": "dummy",
                "split": "test",
                "n_examples": 2,
                "max_seq_len": 4,
                "num_layers": 1,
                "num_heads": 1,
                "head_dim": 2,
                "dtype": "float32",
                "capture": ["hidden"],
            }
            self._write_meta(run_dir, meta)

            df = pd.DataFrame(
                {
                    "example_id": ["e0", "e1"],
                    "text": ["hello", "world"],
                    "input_ids": [[1, 2, 3, 4], [5, 6, 7, 8]],
                    "attention_mask": [[0, 1, 1, 1], [1, 1, 1, 1]],
                    "offset_mapping": [[(0, 1)] * 4, [(0, 1)] * 4],
                    "tokens": [["a", "b", "c", "d"], ["e", "f", "g", "h"]],
                }
            )
            self._write_tokens_parquet(run_dir, df)

            # Create a minimal dec_hidden.zarr/h array.
            # Stored shape in runs: (N, L+1, T, D)
            N, Lp1, T, D = 2, 2, 4, 6
            zpath = run_dir / "dec_hidden.zarr"
            g = zarr.open(zpath.as_posix(), mode="a")
            g.create_dataset("h", shape=(N, Lp1, T, D), chunks=(1, 1, T, D), dtype="f4")
            g["h"][:] = np.arange(N * Lp1 * T * D, dtype=np.float32).reshape(N, Lp1, T, D)

            st = TraceStore(str(run_dir))
            self.assertEqual(st.arch, "dec")
            self.assertIn("dec_hidden", st.arrays())

            enc = st.encodings("e0")
            self.assertIn("input_ids", enc)
            self.assertIn("attention_mask", enc)
            self.assertIn("tokens", enc)

            h = st.hidden("e1", side="dec")
            self.assertEqual(h.shape, (Lp1, T, D))

    def test_encoder_mcq_hidden_and_residuals(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            N, C, L, T, D = 3, 5, 2, 4, 6

            meta = {
                "run_id": "test_encmcq",
                "model": "roberta-base",
                "arch": "enc",
                "objective": "mcq_classification",
                "dataset": "csqa",
                "split": "test",
                "n_examples": N,
                "n_choices": C,
                "max_seq_len": T,
                "num_layers": L,
                "num_heads": 2,
                "head_dim": 3,
                "dtype": "float32",
                "capture": ["hidden", "resid"],
            }
            self._write_meta(run_dir, meta)

            df = pd.DataFrame(
                {
                    "example_id": [f"e{i}" for i in range(N)],
                    "text": ["x"] * N,
                    "question": ["q"] * N,
                    "answerKey": ["A"] * N,
                    "label_idx": [0] * N,
                    "pred_idx": [0] * N,
                    "pred_label": ["A"] * N,
                    "is_correct": [True] * N,
                    "choice_labels": [["A", "B", "C", "D", "E"]] * N,
                    "choice_texts": [["a", "b", "c", "d", "e"]] * N,
                    "choice_logits": [[0, 0, 0, 0, 0]] * N,
                    "choice_probs": [[0.2, 0.2, 0.2, 0.2, 0.2]] * N,
                    "input_ids": [[[]] * C] * N,
                    "attention_mask": [[[]] * C] * N,
                    "offset_mapping": [[None] * C] * N,
                    "tokens": [[[]] * C] * N,
                }
            )
            self._write_tokens_parquet(run_dir, df)

            # enc_mc_hidden: (N, C, L+1, T, D)
            hpath = run_dir / "enc_mc_hidden.zarr"
            hg = zarr.open(hpath.as_posix(), mode="a")
            hg.create_dataset(
                "h", shape=(N, C, L + 1, T, D), chunks=(1, C, 1, T, D), dtype="f4"
            )
            hg["h"][:] = np.zeros((N, C, L + 1, T, D), dtype=np.float32)

            # residuals: embed (N, C, T, D); stage arrays (N, C, L, T, D)
            for fname, key, shape in [
                ("enc_mc_res_embed.zarr", "x", (N, C, T, D)),
                ("enc_mc_res_pre_attn.zarr", "x", (N, C, L, T, D)),
                ("enc_mc_res_post_attn.zarr", "x", (N, C, L, T, D)),
                ("enc_mc_res_post_mlp.zarr", "x", (N, C, L, T, D)),
            ]:
                g = zarr.open((run_dir / fname).as_posix(), mode="a")
                g.create_dataset(key, shape=shape, chunks=(1, C) + shape[2:], dtype="f4")
                g[key][:] = np.zeros(shape, dtype=np.float32)

            st = TraceStore(str(run_dir))
            keys = st.arrays().keys()
            self.assertIn("enc_mc_hidden", keys)
            self.assertIn("enc_mc_res_embed", keys)

            hid = st.mcq_hidden("e0", choice_idx=0)
            self.assertEqual(hid.shape, (L + 1, T, D))

            emb = st.mcq_resid_embed("e0", choice_idx=2)
            self.assertEqual(emb.shape, (T, D))

            pre = st.mcq_resid("e1", choice_idx=1, stage="pre_attn")
            self.assertEqual(pre.shape, (L, T, D))

    def test_build_index_persists(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            meta = {"run_id": "test", "arch": "dec", "n_examples": 2}
            self._write_meta(run_dir, meta)
            df = pd.DataFrame({"example_id": ["a", "b"], "text": ["t1", "t2"]})
            self._write_tokens_parquet(run_dir, df)

            st = TraceStore(str(run_dir))
            idx = st.build_index(persist=True)
            self.assertEqual(idx["a"], 0)
            self.assertTrue((run_dir / "index.json").exists())

    def test_decoder_attn_qkv_and_residual_accessors(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            N, L, H, T, D = 2, 3, 2, 4, 8
            d_head = D // H

            self._write_meta(
                run_dir,
                {
                    "run_id": "test_dec_full",
                    "arch": "dec",
                    "n_examples": N,
                    "max_seq_len": T,
                    "num_layers": L,
                    "num_heads": H,
                    "head_dim": d_head,
                },
            )
            self._write_tokens_parquet(
                run_dir,
                pd.DataFrame(
                    {
                        "example_id": ["e0", "e1"],
                        "text": ["t0", "t1"],
                        "input_ids": [[1, 2, 3, 4], [5, 6, 7, 8]],
                        "attention_mask": [[1, 1, 1, 1], [1, 1, 1, 1]],
                        "offset_mapping": [[(0, 1)] * T, [(0, 1)] * T],
                        "tokens": [["a", "b", "c", "d"], ["e", "f", "g", "h"]],
                    }
                ),
            )

            ga = zarr.open((run_dir / "dec_self_attn.zarr").as_posix(), mode="a")
            ga.create_dataset("attn", shape=(N, L, H, T, T), chunks=(1, 1, 1, T, T), dtype="f4")
            ga["attn"][:] = np.arange(N * L * H * T * T, dtype=np.float32).reshape(N, L, H, T, T)

            gqkv = zarr.open((run_dir / "dec_self_qkv.zarr").as_posix(), mode="a")
            for key in ("q", "k", "v"):
                gqkv.create_dataset(
                    key, shape=(N, L, H, T, d_head), chunks=(1, 1, 1, T, d_head), dtype="f4"
                )
                gqkv[key][:] = np.ones((N, L, H, T, d_head), dtype=np.float32)

            for fname in ("dec_res_embed.zarr", "dec_res_pre_attn.zarr", "dec_res_post_attn.zarr", "dec_res_post_mlp.zarr"):
                g = zarr.open((run_dir / fname).as_posix(), mode="a")
                shape = (N, T, D) if fname == "dec_res_embed.zarr" else (N, L, T, D)
                chunks = (1, T, D) if fname == "dec_res_embed.zarr" else (1, 1, T, D)
                g.create_dataset("x", shape=shape, chunks=chunks, dtype="f4")
                g["x"][:] = np.zeros(shape, dtype=np.float32)

            st = TraceStore(str(run_dir))
            self.assertTrue(st.has("e0"))
            self.assertEqual(st.row("e1"), 1)
            self.assertEqual(st.text("e0"), "t0")

            a_full = st.attn("e0", side="dec", kind="self")
            self.assertEqual(a_full.shape, (L, H, T, T))
            a_lh = st.attn("e0", side="dec", kind="self", layer=1, head=1)
            self.assertEqual(a_lh.shape, (T, T))

            q = st.qkv("e1", which="q", side="dec", kind="self")
            self.assertEqual(q.shape, (L, H, T, d_head))
            q_l = st.qkv("e1", which="q", side="dec", kind="self", layer=0)
            self.assertEqual(q_l.shape, (H, T, d_head))

            re = st.resid_embed("e0", side="dec")
            rp = st.resid("e0", side="dec", stage="pre_attn")
            self.assertEqual(re.shape, (T, D))
            self.assertEqual(rp.shape, (L, T, D))

    def test_missing_array_raises(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            self._write_meta(run_dir, {"run_id": "x", "arch": "dec", "n_examples": 1})
            self._write_tokens_parquet(run_dir, pd.DataFrame({"example_id": ["e0"], "text": ["t"]}))
            st = TraceStore(str(run_dir))
            with self.assertRaises(RuntimeError):
                st.attn("e0", side="dec", kind="self")


if __name__ == "__main__":
    unittest.main()
