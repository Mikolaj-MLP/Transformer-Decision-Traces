from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cli.run_csqa_feature_band_steering_pipeline import main as base_main
from src.csqa.model_presets import resolve_qwen25_instruct_model_id


DEFAULT_SIZES = ["0.5B", "3B", "7B"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the CSQA feature-band steering pipeline for a Qwen 2.5 Instruct model suite."
    )
    parser.add_argument(
        "--sizes",
        type=str,
        default=",".join(DEFAULT_SIZES),
        help="Comma-separated Qwen2.5 Instruct sizes to run, for example: 0.5B,3B,7B",
    )
    args, passthrough = parser.parse_known_args()

    sizes = [item.strip().upper() for item in args.sizes.split(",") if item.strip()]
    if not sizes:
        raise ValueError("No Qwen sizes provided")

    for size in sizes:
        model_id = resolve_qwen25_instruct_model_id(size)
        print(f"[suite] starting {size} -> {model_id}")
        base_main(["--model-id", model_id, *passthrough])


if __name__ == "__main__":
    main()
