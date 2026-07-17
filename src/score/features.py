"""Definicje badanych cech logitowych.

Ten plik zawiera wyłącznie część matematyczną. Nie ładuje danych, modeli ani
nie zapisuje wyników. Wszystkie cechy są liczone dla logitów pięciu odpowiedzi
CSQA, poza opcjonalną entropią pełnego słownika.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F

from src.score.constants import LETTERS


def compute_feature_tensor(
    *,
    feature_name: str,
    full_logits: torch.Tensor,
    choice_logits: torch.Tensor,
    vocab_size: int,
) -> torch.Tensor:
    """Oblicz wskazaną cechę dla każdego przykładu w batchu.

    Entropia to ``-sum(p_i log p_i) / log(5)``. Luka logitowa jest różnicą
    dwóch największych logitów odpowiedzi. Varentropy to wariancja informacji
    ``-log(p_i)`` względem rozkładu prawdopodobieństw odpowiedzi.
    """
    if feature_name == "answer_choice_entropy_normalized":
        choice_log_probs = F.log_softmax(choice_logits.float(), dim=-1)
        choice_probs = torch.exp(choice_log_probs)
        return -(choice_probs * choice_log_probs).sum(dim=-1) / math.log(len(LETTERS))

    if feature_name == "answer_choice_top1_top2_logit_gap":
        sorted_choice_logits = torch.sort(choice_logits.float(), dim=-1, descending=True).values
        return sorted_choice_logits[:, 0] - sorted_choice_logits[:, 1]

    if feature_name == "answer_choice_top1_probability":
        choice_probs = F.softmax(choice_logits.float(), dim=-1)
        return torch.max(choice_probs, dim=-1).values

    if feature_name == "answer_choice_varentropy":
        choice_log_probs = F.log_softmax(choice_logits.float(), dim=-1)
        choice_probs = torch.exp(choice_log_probs)
        surprisal = -choice_log_probs
        entropy = (choice_probs * surprisal).sum(dim=-1, keepdim=True)
        return (choice_probs * (surprisal - entropy) ** 2).sum(dim=-1)

    if feature_name == "full_vocab_entropy_normalized":
        full_log_probs = F.log_softmax(full_logits.float(), dim=-1)
        full_probs = torch.exp(full_log_probs)
        return -(full_probs * full_log_probs).sum(dim=-1) / math.log(vocab_size)

    raise ValueError(f"Unknown feature_name: {feature_name}")


def summarize_logit_features(
    *,
    feature_names: list[str],
    full_logits: torch.Tensor,
    choice_logits: torch.Tensor,
    vocab_size: int,
) -> dict[str, np.ndarray]:
    """Zwróć kilka cech jako tablice NumPy gotowe do tabel wynikowych."""
    return {
        feature_name: compute_feature_tensor(
            feature_name=feature_name,
            full_logits=full_logits,
            choice_logits=choice_logits,
            vocab_size=vocab_size,
        )
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
        for feature_name in feature_names
    }


def compute_feature_from_token_hidden(
    token_hidden: torch.Tensor,
    *,
    feature_name: str,
    layer_index_0based: int,
    maybe_apply_final_norm: Callable[[torch.Tensor, int], torch.Tensor],
    lm_head_weight: torch.Tensor,
    answer_id_tensor_lm_head: torch.Tensor,
    vocab_size: int,
) -> torch.Tensor:
    """Przeprowadź stan ukryty przez readout i oblicz jedną cechę."""
    readout = maybe_apply_final_norm(token_hidden, layer_index_0based)
    full_logits = torch.matmul(readout.to(lm_head_weight.dtype), lm_head_weight.T).float()
    choice_logits = full_logits.index_select(1, answer_id_tensor_lm_head)
    return compute_feature_tensor(
        feature_name=feature_name,
        full_logits=full_logits,
        choice_logits=choice_logits,
        vocab_size=vocab_size,
    )
