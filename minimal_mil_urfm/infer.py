"""Inference with optional test-time augmentation.

Scores one bag at a time. TTA = N augmented forward passes per bag,
averaging the bag-level probability across passes.
"""
from __future__ import annotations

from typing import Callable, Iterable, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import BagDataset, identity_collate
from .model import MILModel


def _hflip_tensor(bag: torch.Tensor) -> torch.Tensor:
    return torch.flip(bag, dims=[-1])


def _vflip_tensor(bag: torch.Tensor) -> torch.Tensor:
    return torch.flip(bag, dims=[-2])


def _jitter_tensor(bag: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Multiplicative contrast jitter around the per-bag mean."""
    if abs(scale - 1.0) < 1e-9:
        return bag
    m = bag.mean()
    return (bag - m) * scale + m


def _tta_pass(bag: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    if bool(rng.random() < 0.5):
        bag = _hflip_tensor(bag)
    if bool(rng.random() < 0.5):
        bag = _vflip_tensor(bag)
    bag = _jitter_tensor(bag, scale=float(rng.uniform(0.9, 1.1)))
    return bag


@torch.no_grad()
def predict(
    model: MILModel,
    ds: BagDataset,
    *,
    device: str = None,
    tta_passes: int = 1,
    tta_seed: int = 0,
):
    """Score every bag in ``ds`` and return a dict with arrays of
    ``bag_ids``, ``y_true``, ``y_prob``. With tta_passes>1, averages
    probabilities across TTA passes."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval().to(device)
    rng = np.random.default_rng(tta_seed)

    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        collate_fn=identity_collate, num_workers=2)
    bag_ids, y_true, y_prob = [], [], []
    for batch in tqdm(loader, desc=f"infer (tta={tta_passes})", leave=False):
        bag, label, bag_id = batch[0]
        bag = bag.to(device, non_blocking=True)
        probs = []
        for _ in range(max(1, tta_passes)):
            xb = _tta_pass(bag, rng) if tta_passes > 1 else bag
            bag_logit, _il, _a, _ba = model(xb)
            probs.append(float(torch.sigmoid(bag_logit).item()))
        bag_ids.append(bag_id)
        y_true.append(int(label))
        y_prob.append(float(np.mean(probs)))
    return {
        "bag_ids": bag_ids,
        "y_true": np.asarray(y_true, dtype=int),
        "y_prob": np.asarray(y_prob, dtype=float),
    }
