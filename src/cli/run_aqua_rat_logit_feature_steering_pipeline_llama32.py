from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cli.run_csqa_logit_feature_steering_pipeline import main as base_main
from src.csqa.model_presets import resolve_llama32_instruct_model_id


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the AQuA-RAT logit-feature steering pipeline for Meta Llama 3.2 Instruct."
    )
    parser.add_argument("--size", type=str, default="1B", choices=["1B", "3B"])
    args, passthrough = parser.parse_known_args()

    model_id = resolve_llama32_instruct_model_id(args.size)
    base_main(["--dataset", "aqua_rat", "--model-id", model_id, *passthrough])


if __name__ == "__main__":
    main()
