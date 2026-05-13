"""Bag BCE + soft-top-k auxiliary loss.

Bag loss is binary cross-entropy on the bag logit (output of the MIL
aggregator).

The soft-top-k auxiliary loss provides a second bag-level supervision
signal through a different aggregation path: it takes the top-K instance
logits, softmax-weights them with a temperature into a single aggregated
logit, and applies BCE between that aggregate and the bag label. In the
zero-temperature limit this reduces to BCE on the hard-max instance; at
large temperature it tends toward BCE on the mean of the top-K. Helps
spatial localization in attention rollout without lesion-level annotations.

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
    """Soft top-K auxiliary loss for one bag (BCE on the soft-pooled logit).

    Picks the top min(k, N_frames) instance logits, softmax-weights them
    with `temperature` into a single aggregated logit, then applies BCE
    between that aggregate and the bag label.

    Formally::

        agg = sum_i w_i * s_i        where  w_i = softmax(s_i / tau)
        loss = BCE(agg, y_bag)

    This matches the formulation used to train the companion paper's
    model; do not confuse it with the alternative `sum_i w_i * BCE(s_i, y)`
    weighted-sum-of-per-instance-BCEs formulation, which has different
    gradients.

    Returns 0 if the bag has fewer than 1 instance (defensive).
    """
    n = instance_logits.shape[0]
    if n == 0:
        return torch.zeros((), device=instance_logits.device,
                           dtype=instance_logits.dtype)

    k_eff = min(k, n)
    top_logits, _idx = torch.topk(instance_logits, k=k_eff, largest=True, sorted=False)

    # Softmax-weighted aggregation of top-K instance logits → one bag-level logit
    weights = torch.softmax(top_logits / max(temperature, 1e-6), dim=0)
    aggregated_logit = (weights * top_logits).sum()

    # BCE on the aggregated logit
    return F.binary_cross_entropy_with_logits(
        aggregated_logit, bag_label.float()
    )


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
