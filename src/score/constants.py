"""Jawne parametry końcowego eksperymentu opartego na funkcji score."""

from __future__ import annotations

import math


LETTERS = ["A", "B", "C", "D", "E"]

DEFAULT_FEATURE_NAMES = [
    "answer_choice_entropy_normalized",
    "answer_choice_top1_top2_logit_gap",
    "answer_choice_varentropy",
]

EXTRACT_BATCH_SIZE = 4
READOUT_BATCH_SIZE = 64
INTERVENTION_BATCH_SIZE = 2
DEFAULT_TRAIN_LIMIT = 2000

GRID_POINTS = 512
GOOD_REGION_LOG_RATIO_THRESHOLD = math.log(1.5)
KDE_JITTER_SCALE = 1e-6
SUPPORT_LOWER_QUANTILE = 0.01
SUPPORT_UPPER_QUANTILE = 0.99
KDE_BANDWIDTH_MULTIPLIER = 1.5
LOG_RATIO_SMOOTHING_SIGMA_BANDWIDTHS = 1.0

DEFAULT_MAX_DELTA_OVER_HIDDEN = 0.005
DEFAULT_BACKTRACK_SCALES = [1.0, 0.5]
GRAD_NORM_EPS = 1e-12
SCORE_IMPROVEMENT_EPS = 1e-6
RANDOM_ORTHO_EPS = 1e-8
CONTROL_INTERVENTION_TYPES = ("ascent", "descent", "random_same_norm")
