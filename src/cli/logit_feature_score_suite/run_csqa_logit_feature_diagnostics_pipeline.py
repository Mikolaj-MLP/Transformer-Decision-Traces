"""CLI diagnostyki cech logitowych i funkcji score."""

from __future__ import annotations

from src.cli.logit_feature_score_suite.diagnostics_experiment import (
    run_diagnostics_experiment,
)


def main(argv: list[str] | None = None) -> None:
    run_diagnostics_experiment(argv)


if __name__ == "__main__":
    main()
