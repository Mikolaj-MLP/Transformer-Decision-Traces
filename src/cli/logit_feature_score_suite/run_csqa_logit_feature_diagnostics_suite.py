from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cli.logit_feature_score_suite.run_csqa_logit_feature_diagnostics_pipeline import main as base_main
from src.csqa.model_presets import resolve_llama32_instruct_model_id, resolve_qwen25_instruct_model_id


DEFAULT_QWEN_SIZES = ["0.5B", "3B", "7B"]
DEFAULT_LLAMA_SIZES = ["1B", "3B"]
DEFAULT_FIT_LIMIT = None
DEFAULT_EVAL_LIMIT = None
DEFAULT_FEATURE_NAMES = (
    "answer_choice_entropy_normalized,"
    "answer_choice_top1_top2_logit_gap,"
    "answer_choice_varentropy"
)


def parse_sizes(raw: str) -> list[str]:
    return [item.strip().upper() for item in raw.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the CSQA logit-feature diagnostics pipeline for both Qwen 2.5 and Llama 3.2 model families."
    )
    parser.add_argument(
        "--families",
        type=str,
        default="qwen,llama",
        help="Comma-separated families to run: qwen,llama",
    )
    parser.add_argument(
        "--qwen-sizes",
        type=str,
        default=",".join(DEFAULT_QWEN_SIZES),
        help="Comma-separated Qwen2.5 Instruct sizes, for example: 0.5B,3B,7B",
    )
    parser.add_argument(
        "--llama-sizes",
        type=str,
        default=",".join(DEFAULT_LLAMA_SIZES),
        help="Comma-separated Llama 3.2 Instruct sizes, for example: 1B,3B",
    )
    args, passthrough = parser.parse_known_args()

    families = [item.strip().lower() for item in args.families.split(",") if item.strip()]
    if not families:
        raise ValueError("No model families provided")

    run_specs: list[tuple[str, str]] = []
    if "qwen" in families:
        qwen_sizes = parse_sizes(args.qwen_sizes)
        if not qwen_sizes:
            raise ValueError("No Qwen sizes provided")
        for size in qwen_sizes:
            run_specs.append((f"qwen-{size}", resolve_qwen25_instruct_model_id(size)))
    if "llama" in families:
        llama_sizes = parse_sizes(args.llama_sizes)
        if not llama_sizes:
            raise ValueError("No Llama sizes provided")
        for size in llama_sizes:
            run_specs.append((f"llama-{size}", resolve_llama32_instruct_model_id(size)))

    if not run_specs:
        raise ValueError("No models selected to run")

    for label, model_id in run_specs:
        print(f"[suite] starting {label} -> {model_id}")
        cmd = [
            "--model-id",
            model_id,
            "--feature-names",
            DEFAULT_FEATURE_NAMES,
            *passthrough,
        ]
        if DEFAULT_FIT_LIMIT is not None:
            cmd.extend(["--fit-limit", DEFAULT_FIT_LIMIT])
        if DEFAULT_EVAL_LIMIT is not None:
            cmd.extend(["--eval-limit", DEFAULT_EVAL_LIMIT])
        base_main(cmd)


if __name__ == "__main__":
    main()
