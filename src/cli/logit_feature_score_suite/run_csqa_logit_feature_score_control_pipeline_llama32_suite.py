from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cli.logit_feature_score_suite.run_csqa_logit_feature_score_control_pipeline import main as base_main
from src.csqa.model_presets import resolve_llama32_instruct_model_id


DEFAULT_SIZES = ["1B", "3B"]
DEFAULT_FIT_LIMIT = "3000"
DEFAULT_EVAL_LIMIT = "2000"
DEFAULT_TOP_K = "8"
DEFAULT_MAX_DELTA_OVER_HIDDEN = "0.01"
DEFAULT_FEATURE_NAMES = "answer_choice_top1_top2_logit_gap,answer_choice_varentropy"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the CSQA logit-feature score-control pipeline for a Meta Llama 3.2 Instruct model suite."
    )
    parser.add_argument(
        "--sizes",
        type=str,
        default=",".join(DEFAULT_SIZES),
        help="Comma-separated Llama 3.2 Instruct sizes to run, for example: 1B,3B",
    )
    args, passthrough = parser.parse_known_args()

    sizes = [item.strip().upper() for item in args.sizes.split(",") if item.strip()]
    if not sizes:
        raise ValueError("No Llama sizes provided")

    for size in sizes:
        model_id = resolve_llama32_instruct_model_id(size)
        print(f"[suite] starting {size} -> {model_id}")
        cmd = [
            "--model-id",
            model_id,
            "--fit-limit",
            DEFAULT_FIT_LIMIT,
            "--top-k-layers-per-feature",
            DEFAULT_TOP_K,
            "--feature-names",
            DEFAULT_FEATURE_NAMES,
            "--max-delta-over-hidden",
            DEFAULT_MAX_DELTA_OVER_HIDDEN,
            *passthrough,
        ]
        if DEFAULT_EVAL_LIMIT is not None:
            cmd.extend(["--eval-limit", DEFAULT_EVAL_LIMIT])
        base_main(cmd)


if __name__ == "__main__":
    main()
