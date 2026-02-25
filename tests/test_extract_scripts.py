import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.cli import extract_trace_csqa_enc as enc_cli
from src.cli import extract_trace_csqa_gpt as gpt_cli
from src.cli import extract_trace_nexttok_dec as nexttok_cli
from src.cli import extract_traces as generic_cli
from src.cli import index_traces


class TestExtractScriptsHelpers(unittest.TestCase):
    def test_resolve_out_dir_defaults(self):
        out1 = gpt_cli._resolve_out_dir(None, "r1")
        out2 = enc_cli._resolve_out_dir(None, "r2")
        out3 = nexttok_cli._resolve_out_dir(None, "r3")
        out4 = generic_cli._resolve_out_dir(None, "r4")
        self.assertTrue(str(out1).lower().endswith(str(Path("traces/r1")).lower()))
        self.assertTrue(str(out2).lower().endswith(str(Path("traces/r2")).lower()))
        self.assertTrue(str(out3).lower().endswith(str(Path("traces/r3")).lower()))
        self.assertTrue(str(out4).lower().endswith(str(Path("traces/r4")).lower()))

    def test_resolve_out_dir_relative(self):
        out = enc_cli._resolve_out_dir("traces/local_run", "ignored")
        self.assertTrue(str(out).lower().endswith(str(Path("traces/local_run")).lower()))

    def test_extract_question(self):
        self.assertEqual(enc_cli._extract_question("Q: What is this?\nChoices:\nA: x"), "What is this?")
        self.assertEqual(enc_cli._extract_question("Plain prompt"), "Plain prompt")

    def test_get_ctx_window_fallback_and_priority(self):
        class Cfg:
            max_position_embeddings = 2048
            n_positions = 1024

        self.assertEqual(enc_cli._get_ctx_window(Cfg(), fallback=512), 2048)

        class Empty:
            pass

        self.assertEqual(enc_cli._get_ctx_window(Empty(), fallback=333), 333)

    def test_decoder_only_detection(self):
        class DecCfg:
            model_type = "gpt2"
            is_encoder_decoder = False
            is_decoder = False

        class EncCfg:
            model_type = "roberta"
            is_encoder_decoder = False
            is_decoder = False

        self.assertTrue(nexttok_cli._is_decoder_only(DecCfg()))
        self.assertFalse(nexttok_cli._is_decoder_only(EncCfg()))

    def test_context_window_and_safe_int(self):
        class Cfg:
            n_positions = 4096
            max_position_embeddings = 0
            n_ctx = 1024

        self.assertEqual(nexttok_cli._context_window(Cfg(), fallback=777), 4096)
        self.assertEqual(nexttok_cli._safe_int(None), -1)
        self.assertEqual(nexttok_cli._safe_int(7), 7)

    def test_parse_index_list_and_validate_indices(self):
        self.assertIsNone(generic_cli.parse_index_list(None))
        self.assertEqual(generic_cli.parse_index_list("1, 2 3"), [1, 2, 3])
        self.assertEqual(generic_cli.parse_index_list([4, "5"]), [4, 5])

        valid = generic_cli._validate_indices("layers", [0, 2, 9], maxn=3)
        self.assertEqual(valid, [0, 2])

        with self.assertRaises(ValueError):
            generic_cli._validate_indices("layers", [9], maxn=3)

    def test_get_texts_variants(self):
        ud = pd.DataFrame(
            {
                "split": ["train", "validation"],
                "example_id": ["a", "b"],
                "text": ["t1", "t2"],
            }
        )
        with patch("src.cli.extract_traces.load_ud_ewt", return_value=ud):
            out = generic_cli.get_texts("ud_ewt", "validation", 10)
            self.assertEqual(list(out.columns), ["example_id", "text"])
            self.assertEqual(len(out), 1)

        go = pd.DataFrame(
            {
                "split": ["validation"],
                "example_id": ["x"],
                "text": ["happy"],
            }
        )
        with patch("src.cli.extract_traces.load_go_emotions", return_value=(go, ["joy"])):
            out = generic_cli.get_texts("go_emotions", "validation", 10)
            self.assertEqual(list(out.columns), ["example_id", "text"])
            self.assertEqual(len(out), 1)

        csqa = pd.DataFrame(
            {
                "example_id": ["c1"],
                "text": ["Q: q"],
                "answerKey": ["A"],
                "correct_idx": [0],
                "csqa_choices": [[{"label": "A", "text": "x"}] * 5],
            }
        )
        with patch("src.cli.extract_traces.load_csqa", return_value=csqa):
            out = generic_cli.get_texts("csqa", "validation", 10)
            self.assertEqual(
                list(out.columns),
                ["example_id", "text", "answerKey", "correct_idx", "csqa_choices"],
            )

        with self.assertRaises(ValueError):
            generic_cli.get_texts("unknown", "validation", 10)


class TestIndexTracesCli(unittest.TestCase):
    def test_main_writes_index(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "meta.json").write_text(
                json.dumps({"run_id": "x", "arch": "dec", "n_examples": 2}, indent=2),
                encoding="utf-8",
            )
            pd.DataFrame({"example_id": ["e0", "e1"], "text": ["a", "b"]}).to_parquet(
                run_dir / "tokens.parquet", index=False
            )

            with patch("sys.argv", ["index_traces.py", str(run_dir), "--validate"]):
                index_traces.main()

            self.assertTrue((run_dir / "index.json").exists())


if __name__ == "__main__":
    unittest.main()

