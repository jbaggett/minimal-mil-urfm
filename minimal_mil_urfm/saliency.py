"""Attention rollout saliency (Abnar & Zuidema, 2020).

Given the per-block attention matrices from a ViT, attention rollout
composes them with a residual term to estimate token-level influence on
the CLS prediction. Output is a per-patch saliency map.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn.functional as F


def attention_rollout(
    attns: List[torch.Tensor],
    head_fusion: str = "mean",
    discard_ratio: float = 0.0,
    return_layers: int | None = None,
) -> torch.Tensor:
    """Compose per-block attention into a CLS→patches saliency vector.

    Parameters
    ----------
    attns : list of (B, n_heads, T, T)
        One attention tensor per ViT block, where T = 1 + N_patches.
    head_fusion : "mean" | "max"
        How to reduce heads inside each block.
    discard_ratio : float in [0, 1)
        Fraction of LOWEST-attention values to zero out per block (helps
        focus saliency). 0 means no thresholding.
    return_layers : int or None
        If given, only use the LAST ``return_layers`` blocks (sometimes
        gives sharper saliency for medical images per Wollek et al. 2023).

    Returns
    -------
    rollout : (B, N_patches)
        CLS-to-patch attention after composition, normalized so that
        the patch axis sums to 1 per sample.
    """
    if not attns:
        raise ValueError("attns is empty; ensure return_attn=True in the forward pass")
    if return_layers is not None:
        attns = attns[-return_layers:]
    B, H, T, _ = attns[0].shape
    device = attns[0].device

    # Reduce heads + add residual + renormalize
    reduced = []
    for a in attns:
        if head_fusion == "mean":
            a = a.mean(dim=1)
        elif head_fusion == "max":
            a = a.amax(dim=1)
        else:
            raise ValueError(f"head_fusion={head_fusion!r}")
        if discard_ratio > 0:
            flat = a.view(B, -1)
            k = int(flat.shape[1] * discard_ratio)
            if k > 0:
                vals, idx = torch.topk(flat, k=k, largest=False)
                mask = torch.ones_like(flat)
                mask.scatter_(1, idx, 0)
                a = (flat * mask).view(B, T, T)
        # Add identity (residual stream)
        eye = torch.eye(T, device=device).unsqueeze(0).expand(B, T, T)
        a = a + eye
        a = a / a.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        reduced.append(a)

    rollout = reduced[0]
    for a in reduced[1:]:
        rollout = a @ rollout

    cls_to_patches = rollout[:, 0, 1:]  # (B, N_patches)
    cls_to_patches = cls_to_patches / cls_to_patches.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    return cls_to_patches


def saliency_map_per_frame(
    cls_to_patches: torch.Tensor,
    patch_size: int,
    img_size: int,
) -> torch.Tensor:
    """Reshape a per-patch saliency vector into a 2D map at image resolution.

    Parameters
    ----------
    cls_to_patches : (B, N_patches) — output of attention_rollout
    patch_size : ViT patch size (16 for B/16)
    img_size   : input resolution the ViT was run at

    Returns (B, img_size, img_size) saliency map, bilinearly upsampled.
    """
    side = img_size // patch_size
    assert cls_to_patches.shape[-1] == side * side, (
        f"patch count mismatch: {cls_to_patches.shape[-1]} != {side*side}"
    )
    grid = cls_to_patches.view(-1, 1, side, side)
    up = F.interpolate(grid, size=(img_size, img_size), mode="bilinear",
                       align_corners=False)
    return up.squeeze(1)


def peak_xy(saliency_map_2d: torch.Tensor) -> Tuple[int, int]:
    """Return (x, y) pixel of the maximum-saliency point in a 2D map."""
    h, w = saliency_map_2d.shape[-2:]
    idx = int(saliency_map_2d.flatten().argmax().item())
    y, x = divmod(idx, w)
    return x, y
