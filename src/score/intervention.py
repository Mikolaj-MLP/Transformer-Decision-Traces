"""Matematyka interwencji w stanie ukrytym.

Kierunek ascent jest gradientem złożenia ``s(f(h))`` po stanie ukrytym ``h``:

    grad_h s(f(h)) = (ds/df) * grad_h f(h).

Kierunek jest normalizowany, a długość kroku ograniczana relatywnie do normy
stanu ukrytego. Akceptowany jest pierwszy krok z listy backtrackingu, który
pozostaje w obszarze wspieranym i zwiększa wartość ``score``.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import torch

from src.score.constants import GRAD_NORM_EPS, RANDOM_ORTHO_EPS, SCORE_IMPROVEMENT_EPS
from src.score.density import interpolate_score_state
from src.score.features import compute_feature_from_token_hidden


def compute_score_ascent_unit_delta(
    token_hidden: torch.Tensor,
    *,
    feature_name: str,
    layer_index_0based: int,
    maybe_apply_final_norm: Callable[[torch.Tensor, int], torch.Tensor],
    lm_head_weight: torch.Tensor,
    answer_id_tensor_lm_head: torch.Tensor,
    vocab_size: int,
    current_score_derivative: np.ndarray,
    intervention_mask: np.ndarray,
) -> dict[str, object]:
    """Wyznacz jednostkowy gradient ``s(f(h))`` dla każdego przykładu."""
    score_derivative_tensor = torch.as_tensor(
        current_score_derivative,
        device=token_hidden.device,
        dtype=torch.float32,
    )
    with torch.enable_grad():
        base = token_hidden.detach().clone().requires_grad_(True)
        current_feature = compute_feature_from_token_hidden(
            base,
            feature_name=feature_name,
            layer_index_0based=layer_index_0based,
            maybe_apply_final_norm=maybe_apply_final_norm,
            lm_head_weight=lm_head_weight,
            answer_id_tensor_lm_head=answer_id_tensor_lm_head,
            vocab_size=vocab_size,
        )

        unit_deltas: list[torch.Tensor] = []
        feature_grad_norms: list[float] = []
        score_grad_norms: list[float] = []
        for batch_index in range(base.shape[0]):
            derivative = float(score_derivative_tensor[batch_index].item())
            inactive = (
                not bool(intervention_mask[batch_index])
                or not math.isfinite(derivative)
                or abs(derivative) <= 1e-8
            )
            if inactive:
                unit_deltas.append(torch.zeros_like(base[batch_index]))
                feature_grad_norms.append(0.0)
                score_grad_norms.append(0.0)
                continue

            grad_full = torch.autograd.grad(
                current_feature[batch_index],
                base,
                retain_graph=batch_index < base.shape[0] - 1,
                create_graph=False,
                allow_unused=False,
            )[0]
            feature_grad = grad_full[batch_index]
            feature_grad_norm = float(feature_grad.detach().float().norm().item())
            score_grad = score_derivative_tensor[batch_index] * feature_grad
            score_grad_norm = float(score_grad.detach().float().norm().item())

            if not math.isfinite(score_grad_norm) or score_grad_norm <= GRAD_NORM_EPS:
                unit_deltas.append(torch.zeros_like(base[batch_index]))
                feature_grad_norms.append(feature_grad_norm if math.isfinite(feature_grad_norm) else 0.0)
                score_grad_norms.append(0.0)
                continue

            unit_deltas.append(score_grad / score_grad.detach().float().norm().clamp_min(1e-12))
            feature_grad_norms.append(feature_grad_norm if math.isfinite(feature_grad_norm) else 0.0)
            score_grad_norms.append(score_grad_norm)

        unit_delta = torch.stack(unit_deltas).detach()

    token_hidden_l2 = token_hidden.detach().float().norm(dim=-1)
    return {
        "unit_delta": unit_delta,
        "current_feature_value": current_feature.detach().cpu().numpy().astype(np.float32),
        "feature_grad_l2_norm": np.asarray(feature_grad_norms, dtype=np.float32),
        "score_grad_l2_norm": np.asarray(score_grad_norms, dtype=np.float32),
        "token_hidden_l2_norm": token_hidden_l2.detach().cpu().numpy().astype(np.float32),
    }


def build_full_cap_delta(
    token_hidden: torch.Tensor,
    *,
    unit_delta: torch.Tensor,
    intervention_mask: np.ndarray,
    max_delta_over_hidden: float,
) -> dict[str, object]:
    """Nadaj kierunkowi długość ``cap * ||h||``."""
    token_hidden_l2 = token_hidden.detach().float().norm(dim=-1)
    scale = token_hidden_l2 * float(max_delta_over_hidden)
    mask = torch.as_tensor(intervention_mask, device=token_hidden.device, dtype=torch.float32)
    delta_raw = unit_delta * (scale * mask).unsqueeze(-1).to(unit_delta.dtype)
    raw_delta_l2 = delta_raw.detach().float().norm(dim=-1)
    raw_delta_over_hidden = raw_delta_l2 / token_hidden_l2.clamp_min(1e-12)
    return {
        "delta_raw": delta_raw.detach(),
        "raw_delta_l2_norm": raw_delta_l2.detach().cpu().numpy().astype(np.float32),
        "raw_delta_over_token_hidden_l2": raw_delta_over_hidden.detach().cpu().numpy().astype(np.float32),
        "token_hidden_l2_norm": token_hidden_l2.detach().cpu().numpy().astype(np.float32),
    }


def evaluate_candidate_delta(
    token_hidden: torch.Tensor,
    *,
    delta: torch.Tensor,
    feature_name: str,
    layer_index_0based: int,
    maybe_apply_final_norm: Callable[[torch.Tensor, int], torch.Tensor],
    lm_head_weight: torch.Tensor,
    answer_id_tensor_lm_head: torch.Tensor,
    vocab_size: int,
    region_model: dict[str, object],
) -> dict[str, np.ndarray]:
    """Oblicz cechę i score lokalnego stanu ``h + delta`` bez pełnego forwardu."""
    steered_feature = compute_feature_from_token_hidden(
        token_hidden + delta,
        feature_name=feature_name,
        layer_index_0based=layer_index_0based,
        maybe_apply_final_norm=maybe_apply_final_norm,
        lm_head_weight=lm_head_weight,
        answer_id_tensor_lm_head=answer_id_tensor_lm_head,
        vocab_size=vocab_size,
    ).detach()
    feature_values = steered_feature.cpu().numpy().astype(np.float32)
    score_state = interpolate_score_state(feature_values, region_model=region_model)
    return {
        "feature_value": feature_values,
        "score_value": score_state["score_value"].astype(np.float32),
        "supported": score_state["supported"].astype(bool),
        "region_label": score_state["region_label"],
    }


def select_score_ascent_delta(
    token_hidden: torch.Tensor,
    *,
    delta_raw: torch.Tensor,
    current_score_value: np.ndarray,
    intervention_mask: np.ndarray,
    feature_name: str,
    layer_index_0based: int,
    maybe_apply_final_norm: Callable[[torch.Tensor, int], torch.Tensor],
    lm_head_weight: torch.Tensor,
    answer_id_tensor_lm_head: torch.Tensor,
    vocab_size: int,
    region_model: dict[str, object],
    backtrack_scales: list[float],
) -> dict[str, object]:
    """Wybierz pierwszy wspierany krok, który faktycznie zwiększa score."""
    batch_size = token_hidden.shape[0]
    accepted_delta = torch.zeros_like(delta_raw)
    accepted_feature_value = np.full(batch_size, np.nan, dtype=np.float32)
    accepted_score_value = np.full(batch_size, np.nan, dtype=np.float32)
    accepted_supported = np.zeros(batch_size, dtype=bool)
    accepted_step_scale = np.zeros(batch_size, dtype=np.float32)
    accepted_region_label = np.asarray(["unsupported"] * batch_size, dtype=object)
    accepted_mask = np.zeros(batch_size, dtype=bool)

    current_score_value = np.asarray(current_score_value, dtype=np.float32)
    unresolved = np.asarray(intervention_mask, dtype=bool).copy()
    for scale in backtrack_scales:
        if not bool(unresolved.any()):
            break
        candidate_delta = delta_raw * float(scale)
        candidate = evaluate_candidate_delta(
            token_hidden,
            delta=candidate_delta,
            feature_name=feature_name,
            layer_index_0based=layer_index_0based,
            maybe_apply_final_norm=maybe_apply_final_norm,
            lm_head_weight=lm_head_weight,
            answer_id_tensor_lm_head=answer_id_tensor_lm_head,
            vocab_size=vocab_size,
            region_model=region_model,
        )
        improved = (
            unresolved
            & candidate["supported"]
            & np.isfinite(candidate["score_value"])
            & (candidate["score_value"] > current_score_value + SCORE_IMPROVEMENT_EPS)
        )
        if not bool(improved.any()):
            continue
        indices = np.where(improved)[0]
        accepted_delta[indices] = candidate_delta[indices]
        accepted_feature_value[indices] = candidate["feature_value"][indices]
        accepted_score_value[indices] = candidate["score_value"][indices]
        accepted_supported[indices] = candidate["supported"][indices]
        accepted_step_scale[indices] = float(scale)
        accepted_region_label[indices] = candidate["region_label"][indices]
        accepted_mask[indices] = True
        unresolved[indices] = False

    norm_stats = compute_delta_norm_stats(token_hidden, delta=accepted_delta)
    return {
        "delta": accepted_delta.detach(),
        "accepted_intervention": accepted_mask,
        "accepted_step_scale": accepted_step_scale,
        "steered_feature_value_local": accepted_feature_value,
        "steered_score_value_local": accepted_score_value,
        "steered_supported": accepted_supported,
        "steered_region_label": accepted_region_label,
        **norm_stats,
    }


def build_random_same_norm_delta(
    reference_delta: torch.Tensor,
    *,
    intervention_mask: np.ndarray,
) -> torch.Tensor:
    """Losowa kontrola ortogonalna do ascent i o identycznej normie."""
    reference_delta = reference_delta.detach()
    ref_float = reference_delta.float()
    ref_norm = ref_float.norm(dim=-1, keepdim=True)

    random_vector = torch.randn_like(reference_delta)
    random_float = random_vector.float()
    ref_norm_sq = (ref_float * ref_float).sum(dim=-1, keepdim=True).clamp_min(1e-12)
    projection = (random_float * ref_float).sum(dim=-1, keepdim=True) / ref_norm_sq
    orthogonal = random_vector - projection.to(random_vector.dtype) * reference_delta
    orthogonal_norm = orthogonal.float().norm(dim=-1, keepdim=True)

    fallback = random_vector / random_float.norm(dim=-1, keepdim=True).clamp_min(1e-12).to(random_vector.dtype)
    orthogonal_unit = orthogonal / orthogonal_norm.clamp_min(1e-12).to(orthogonal.dtype)
    use_fallback = orthogonal_norm <= RANDOM_ORTHO_EPS
    random_unit = torch.where(use_fallback.expand_as(orthogonal_unit), fallback, orthogonal_unit)

    random_delta = random_unit * ref_norm.to(random_unit.dtype)
    mask = torch.as_tensor(intervention_mask, device=random_delta.device, dtype=random_delta.dtype).unsqueeze(-1)
    return (random_delta * mask).detach()


def compute_delta_norm_stats(
    token_hidden: torch.Tensor,
    *,
    delta: torch.Tensor,
) -> dict[str, np.ndarray]:
    """Norma perturbacji i jej stosunek do normy stanu ukrytego."""
    delta_l2 = delta.detach().float().norm(dim=-1)
    token_hidden_l2 = token_hidden.detach().float().norm(dim=-1)
    delta_over_hidden = delta_l2 / token_hidden_l2.clamp_min(1e-12)
    return {
        "delta_l2_norm": delta_l2.detach().cpu().numpy().astype(np.float32),
        "delta_over_token_hidden_l2": delta_over_hidden.detach().cpu().numpy().astype(np.float32),
        "token_hidden_l2_norm": token_hidden_l2.detach().cpu().numpy().astype(np.float32),
    }
