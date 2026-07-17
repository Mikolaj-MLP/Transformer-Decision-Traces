from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import torch

from src.score.density import add_score_derivative, build_distribution_models, interpolate_score_state
from src.score.features import compute_feature_tensor
from src.score.intervention import build_full_cap_delta, build_random_same_norm_delta
from src.score.statistics import compute_ks_statistic


class ScoreResearchTests(unittest.TestCase):
    def test_uniform_choice_logits_have_max_entropy_and_zero_varentropy(self) -> None:
        choice_logits = torch.zeros((2, 5))
        full_logits = choice_logits.clone()

        entropy = compute_feature_tensor(
            feature_name="answer_choice_entropy_normalized",
            full_logits=full_logits,
            choice_logits=choice_logits,
            vocab_size=5,
        )
        gap = compute_feature_tensor(
            feature_name="answer_choice_top1_top2_logit_gap",
            full_logits=full_logits,
            choice_logits=choice_logits,
            vocab_size=5,
        )
        varentropy = compute_feature_tensor(
            feature_name="answer_choice_varentropy",
            full_logits=full_logits,
            choice_logits=choice_logits,
            vocab_size=5,
        )

        torch.testing.assert_close(entropy, torch.ones(2))
        torch.testing.assert_close(gap, torch.zeros(2))
        torch.testing.assert_close(varentropy, torch.zeros(2))

    def test_ks_is_one_for_disjoint_samples(self) -> None:
        statistic = compute_ks_statistic(
            np.array([-2.0, -1.0]),
            np.array([1.0, 2.0]),
        )
        self.assertAlmostEqual(statistic, 1.0)

    def test_density_ratio_prefers_values_from_correct_distribution(self) -> None:
        rng = np.random.default_rng(7)
        correct = rng.normal(1.0, 0.15, size=80)
        incorrect = rng.normal(-1.0, 0.15, size=80)
        feature_table = pd.DataFrame(
            {
                "feature_name": "feature",
                "layer_number": 1,
                "feature_value": np.concatenate([correct, incorrect]),
                "clean_is_correct": [True] * len(correct) + [False] * len(incorrect),
            }
        )
        selected = pd.DataFrame({"feature_name": ["feature"], "layer_number": [1]})
        models, grid = build_distribution_models(
            fit_feature_df=feature_table,
            selected_layers_df=selected,
            log_ratio_threshold=0.1,
            grid_points=256,
        )
        grid = add_score_derivative(models, grid)
        state = interpolate_score_state(
            np.array([-1.0, 1.0]),
            region_model=models[("feature", 1)],
        )
        self.assertLess(state["score_value"][0], state["score_value"][1])
        self.assertEqual(len(grid), 256)

    def test_random_control_matches_norm_and_is_orthogonal(self) -> None:
        torch.manual_seed(11)
        reference = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        random_delta = build_random_same_norm_delta(
            reference,
            intervention_mask=np.array([True, True]),
        )
        torch.testing.assert_close(random_delta.norm(dim=-1), reference.norm(dim=-1))
        dot = (random_delta * reference).sum(dim=-1)
        torch.testing.assert_close(dot, torch.zeros_like(dot), atol=1e-6, rtol=0.0)

    def test_cap_is_relative_to_hidden_norm(self) -> None:
        hidden = torch.tensor([[3.0, 4.0], [0.0, 2.0]])
        unit = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        result = build_full_cap_delta(
            hidden,
            unit_delta=unit,
            intervention_mask=np.array([True, True]),
            max_delta_over_hidden=0.01,
        )
        np.testing.assert_allclose(
            result["raw_delta_over_token_hidden_l2"],
            np.array([0.01, 0.01]),
            rtol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
