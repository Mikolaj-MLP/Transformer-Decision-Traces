"""CLI dla końcowego eksperymentu z kontrolami kierunku interwencji."""

from __future__ import annotations

from src.cli.logit_feature_score_suite.intervention_experiment import (
    run_intervention_experiment,
)


def main(argv: list[str] | None = None) -> None:
    run_intervention_experiment(argv, controls=True)


if __name__ == "__main__":
    main()
