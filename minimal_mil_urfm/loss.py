"""Bag BCE + soft-top-k auxiliary loss.

Bag loss is binary cross-entropy on the bag logit.

The soft-top-k auxiliary loss encourages the top-K instance logits within
each bag to align with the bag label. It is computed as a temperature-
softmax-weighted average of per-instance BCE losses across the top-K
instances by score. Helps spatial localization in attention rollout
without lesion-level annotations.

Reference: the companion paper (in submission).
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def bag_bce(bag_logit: torch.Tensor, bag_label: torch.Tensor,
            pos_weight: float = 1.0) -> torch.Tensor:
    """Binary cross-entropy on bag-level logit + scalar label."""
    return F.binary_cross_entropy_with_logits(
        bag_logit, bag_label.float(),
        pos_weight=torch.tensor(pos_weight, device=bag_logit.device, dtype=bag_logit.dtype),
    )


def soft_topk_aux_loss(
    instance_logits: torch.Tensor,
    bag_label: torch.Tensor,
    k: int = 5,
    temperature: float = 0.5,
) -> torch.Tensor:
    """Soft top-K auxiliary loss for one bag.

    Picks the top min(k, N_frames) instance logits, applies a softmax over
    them with `temperature`, then computes a softmax-weighted BCE of each
    of those K against the bag label.

    Returns 0 if the bag has fewer than 1 instance (defensive).
    """
    n = instance_logits.shape[0]
    if n == 0:
        return torch.zeros((), device=instance_logits.device,
                           dtype=instance_logits.dtype)

    k_eff = min(k, n)
    top_logits, _idx = torch.topk(instance_logits, k=k_eff, largest=True, sorted=False)

    # Per-instance BCE
    target = bag_label.float().expand_as(top_logits)
    bce = F.binary_cross_entropy_with_logits(top_logits, target, reduction="none")

    # Softmax-weighted average across the top-K
    weights = torch.softmax(top_logits / max(temperature, 1e-6), dim=0)
    return (weights * bce).sum()


def bag_bce_with_soft_topk(
    bag_logit: torch.Tensor,
    instance_logits: torch.Tensor,
    bag_label: torch.Tensor,
    pos_weight: float = 1.0,
    aux_weight: float = 0.3,
    k: int = 5,
    temperature: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combined loss for one bag.

    Returns
    -------
    total  : ()    bag_bce + aux_weight * soft_topk
    bag    : ()    bag_bce component (for logging)
    aux    : ()    soft_topk component (for logging)
    """
    bag = bag_bce(bag_logit, bag_label, pos_weight=pos_weight)
    if aux_weight <= 0:
        zero = torch.zeros_like(bag)
        return bag, bag, zero
    aux = soft_topk_aux_loss(instance_logits, bag_label, k=k, temperature=temperature)
    return bag + aux_weight * aux, bag, aux
