from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cli.logit_feature_score_suite.run_csqa_logit_feature_score_control_pipeline import main as base_main
from src.csqa.model_presets import resolve_llama32_instruct_model_id, resolve_qwen25_instruct_model_id


DEFAULT_QWEN_SIZES = ["3B"]
DEFAULT_LLAMA_SIZES = ["3B"]
DEFAULT_CAPS = ["0.0025", "0.005", "0.01"]
DEFAULT_FIT_LIMIT = "2000"
DEFAULT_EVAL_LIMIT = None
DEFAULT_TOP_K = "6"
DEFAULT_FEATURE_NAMES = "answer_choice_top1_top2_logit_gap,answer_choice_varentropy"


def parse_csv(raw: str) -> list[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def resolve_model_runs(*, family: str, sizes: list[str]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    family_key = str(family).strip().lower()
    if family_key == "qwen":
        for size in sizes:
            out.append((f"qwen-{size.upper()}", resolve_qwen25_instruct_model_id(size)))
        return out
    if family_key == "llama":
        for size in sizes:
            out.append((f"llama-{size.upper()}", resolve_llama32_instruct_model_id(size)))
        return out
    raise ValueError(f"Unsupported family: {family}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a cap sweep for the CSQA logit-feature score-control pipeline."
    )
    parser.add_argument(
        "--families",
        type=str,
        default="qwen,llama",
        help="Comma-separated families to run: qwen, llama",
    )
    parser.add_argument(
        "--qwen-sizes",
        type=str,
        default=",".join(DEFAULT_QWEN_SIZES),
        help="Comma-separated Qwen 2.5 Instruct sizes, for example: 0.5B,3B,7B",
    )
    parser.add_argument(
        "--llama-sizes",
        type=str,
        default=",".join(DEFAULT_LLAMA_SIZES),
        help="Comma-separated Llama 3.2 Instruct sizes, for example: 1B,3B",
    )
    parser.add_argument(
        "--caps",
        type=str,
        default=",".join(DEFAULT_CAPS),
        help="Comma-separated max-delta-over-hidden caps, for example: 0.0025,0.005,0.01",
    )
    args, passthrough = parser.parse_known_args()

    families = [item.lower() for item in parse_csv(args.families)]
    if not families:
        raise ValueError("No model families provided")

    caps = parse_csv(args.caps)
    if not caps:
        raise ValueError("No cap values provided")

    qwen_sizes = parse_csv(args.qwen_sizes)
    llama_sizes = parse_csv(args.llama_sizes)

    planned_runs: list[tuple[str, str, str]] = []
    for family in families:
        if family == "qwen":
            planned_runs.extend((label, model_id, cap) for label, model_id in resolve_model_runs(family=family, sizes=qwen_sizes) for cap in caps)
            continue
        if family == "llama":
            planned_runs.extend((label, model_id, cap) for label, model_id in resolve_model_runs(family=family, sizes=llama_sizes) for cap in caps)
            continue
        raise ValueError(f"Unsupported family in --families: {family}")

    for label, model_id, cap in planned_runs:
        print(f"[cap-sweep] starting {label} | cap={cap} -> {model_id}")
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
            cap,
            *passthrough,
        ]
        if DEFAULT_EVAL_LIMIT is not None:
            cmd.extend(["--eval-limit", DEFAULT_EVAL_LIMIT])
        base_main(cmd)


if __name__ == "__main__":
    main()
