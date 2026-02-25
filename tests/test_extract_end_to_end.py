import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from transformers import AutoConfig

from src.cli import extract_trace_csqa_enc
from src.cli import extract_trace_csqa_gpt
from src.cli import extract_trace_nexttok_dec
from src.cli import extract_traces
from src.traces_utils.store import TraceStore


def make_csqa_df(n=2):
    labels = ["A", "B", "C", "D", "E"]
    choices = [{"label": lab, "text": f"choice_{lab}"} for lab in labels]
    rows = []
    for i in range(n):
        rows.append(
            {
                "example_id": f"csqa_{i}",
                "text": f"Q: question_{i}\nChoices:\nA: a\nB: b\nC: c\nD: d\nE: e\nAnswer:",
                "answerKey": labels[i % len(labels)],
                "correct_idx": i % len(labels),
                "csqa_choices": choices,
            }
        )
    return pd.DataFrame(rows)


class TestExtractScriptsEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gpt_model = "gpt2"
        cls.enc_model = "roberta-base"
        try:
            AutoConfig.from_pretrained(cls.gpt_model, local_files_only=True)
            AutoConfig.from_pretrained(cls.enc_model, local_files_only=True)
        except Exception as exc:
            raise unittest.SkipTest(
                "Skipping end-to-end extract tests: required HF model files are not cached locally "
                f"({exc})."
            )

    def _run_main_offline(self, fn):
        # Keep tests deterministic and independent from network availability.
        with patch.dict(
            os.environ,
            {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            },
            clear=False,
        ):
            fn()

    def test_extract_trace_csqa_gpt_hidden(self):
        csqa_df = make_csqa_df(3)
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "csqa_gpt_run"
            with patch("src.cli.extract_trace_csqa_gpt.load_csqa", return_value=csqa_df), patch(
                "sys.argv",
                [
                    "extract_trace_csqa_gpt.py",
                    "--split",
                    "validation",
                    "--batch_size",
                    "2",
                    "--limit",
                    "3",
                    "--model_path",
                    self.gpt_model,
                    "--max_seq_len",
                    "32",
                    "--capture",
                    "hidden",
                    "--out_dir",
                    str(out_dir),
                ],
            ):
                self._run_main_offline(extract_trace_csqa_gpt.main)

            st = TraceStore(str(out_dir))
            self.assertEqual(st.meta["arch"], "dec")
            self.assertEqual(st.meta["n_examples"], 3)
            self.assertEqual(st.meta["model"], self.gpt_model)
            self.assertEqual(st.meta["capture"], ["hidden"])
            self.assertIn("dec_hidden", st.arrays())

            shape = st.arrays()["dec_hidden"]
            self.assertEqual(shape[0], 3)
            self.assertEqual(shape[1], st.meta["num_layers"] + 1)
            self.assertEqual(shape[2], st.meta["max_seq_len"])
            self.assertIn("answerKey", st.tokens.columns)
            self.assertIn("csqa_choices", st.tokens.columns)

    def test_extract_trace_nexttok_dec_hidden(self):
        nexttok_df = pd.DataFrame(
            {
                "example_id": ["n0", "n1"],
                "text": ["alpha beta gamma", "delta epsilon zeta"],
            }
        )
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "nexttok_run"
            with patch(
                "src.cli.extract_trace_nexttok_dec.load_nexttok_texts", return_value=nexttok_df
            ), patch(
                "sys.argv",
                [
                    "extract_trace_nexttok_dec.py",
                    "--model",
                    self.gpt_model,
                    "--dataset",
                    "ud_ewt",
                    "--split",
                    "validation",
                    "--batch_size",
                    "2",
                    "--max_seq_len",
                    "32",
                    "--capture",
                    "hidden",
                    "--out_dir",
                    str(out_dir),
                ],
            ):
                self._run_main_offline(extract_trace_nexttok_dec.main)

            st = TraceStore(str(out_dir))
            self.assertEqual(st.meta["arch"], "dec")
            self.assertEqual(st.meta["objective"], "next_token")
            self.assertEqual(st.meta["model"], self.gpt_model)
            self.assertIn("dec_hidden", st.arrays())

            shape = st.arrays()["dec_hidden"]
            self.assertEqual(shape[0], 2)
            self.assertEqual(shape[1], st.meta["num_layers"] + 1)
            self.assertEqual(shape[2], st.meta["max_seq_len"])

            required = {"next_pos", "next_true_id", "next_pred_id", "next_correct", "next_entropy"}
            self.assertTrue(required.issubset(set(st.tokens.columns)))
            self.assertTrue((st.tokens["next_pos"] >= 0).any())

    def test_extract_trace_csqa_enc_hidden(self):
        csqa_df = make_csqa_df(2)
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "enc_run"
            with patch("src.cli.extract_trace_csqa_enc.load_csqa", return_value=csqa_df), patch(
                "sys.argv",
                [
                    "extract_trace_csqa_enc.py",
                    "--model_path",
                    self.enc_model,
                    "--split",
                    "validation",
                    "--batch_size",
                    "2",
                    "--limit",
                    "2",
                    "--max_seq_len",
                    "32",
                    "--capture",
                    "hidden",
                    "--out_dir",
                    str(out_dir),
                ],
            ):
                self._run_main_offline(extract_trace_csqa_enc.main)

            st = TraceStore(str(out_dir))
            self.assertEqual(st.meta["arch"], "enc")
            self.assertEqual(st.meta["objective"], "mcq_classification")
            self.assertEqual(st.meta["model"], self.enc_model)
            self.assertEqual(st.meta["n_choices"], 5)
            self.assertIn("enc_mc_hidden", st.arrays())

            shape = st.arrays()["enc_mc_hidden"]
            self.assertEqual(shape[0], 2)
            self.assertEqual(shape[1], 5)
            self.assertEqual(shape[2], st.meta["num_layers"] + 1)
            self.assertEqual(shape[3], st.meta["max_seq_len"])

            for col in ("choice_logits", "choice_probs", "pred_idx", "is_correct"):
                self.assertIn(col, st.tokens.columns)

    def test_extract_traces_generic_encoder_hidden(self):
        ud_df = pd.DataFrame(
            {
                "split": ["validation", "validation"],
                "example_id": ["u0", "u1"],
                "text": ["tiny sample one", "tiny sample two"],
            }
        )
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "generic_run"
            with patch("src.cli.extract_traces.load_ud_ewt", return_value=ud_df), patch(
                "sys.argv",
                [
                    "extract_traces.py",
                    "--model",
                    self.enc_model,
                    "--dataset",
                    "ud_ewt",
                    "--split",
                    "validation",
                    "--limit",
                    "2",
                    "--max_seq_len",
                    "32",
                    "--batch_size",
                    "2",
                    "--capture",
                    "hidden",
                    "--out_dir",
                    str(out_dir),
                ],
            ):
                self._run_main_offline(extract_traces.main)

            st = TraceStore(str(out_dir))
            self.assertEqual(st.meta["arch"], "enc")
            self.assertEqual(st.meta["model"], self.enc_model)
            self.assertIn("enc_hidden", st.arrays())

            shape = st.arrays()["enc_hidden"]
            self.assertEqual(shape[0], 2)
            self.assertEqual(shape[1], st.meta["num_layers"] + 1)
            self.assertEqual(shape[2], st.meta["max_seq_len"])
            self.assertTrue({"example_id", "text", "input_ids", "tokens"}.issubset(st.tokens.columns))


if __name__ == "__main__":
    unittest.main()
