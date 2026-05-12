"""Bag dataset for MIL fine-tuning.

Reads a manifest CSV that conforms to the public-bus-mil schema
(see https://github.com/jbaggett/public-bus-mil), and yields one bag at a
time as (frames_tensor, bag_label, bag_id). Frames are loaded from the
``images_dir`` that the manifest refers to.

Expected manifest columns:
  - bag_id          : str
  - label           : 0 | 1
  - fold            : int 1..5  (optional; only needed if you use fold_split)
  - patient_id      : str       (carried through; not used here)
  - dataset         : str       (carried through; not used here)

Plus a frames CSV with:
  - bag_id, frame_idx_in_bag, src_path

The dataset takes care of:
  - center-letterbox + resize to (img_size, img_size) with gray fill
  - ImageNet mean/std normalization (matches the default for MAE-pretrained ViTs)
  - returning frames as a single tensor (N, 3, H, W) per bag

For training, wrap with a standard ``torch.utils.data.DataLoader`` using
``batch_size=1`` and ``collate_fn=identity_collate`` so each batch is one bag.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def center_letterbox_to(img: Image.Image, size: int, fill: int = 128) -> Image.Image:
    """Resize aspect-preserving to fit inside (size, size); pad with gray.

    Matches the v7 / public-bus-mil convention used to produce input
    frames for the URFM-trained models. Symmetric padding (centered).
    """
    img = img.convert("L")  # grayscale
    w, h = img.size
    scale = size / max(w, h)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    img = img.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("L", (size, size), fill)
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    canvas.paste(img, (pad_x, pad_y))
    return canvas


def to_3channel_imagenet_tensor(img_pil: Image.Image) -> torch.Tensor:
    """Grayscale PIL -> (3, H, W) float tensor with ImageNet normalization."""
    arr = np.array(img_pil, dtype=np.float32) / 255.0   # (H, W)
    arr = np.stack([arr, arr, arr], axis=-1)            # (H, W, 3)
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = arr.transpose(2, 0, 1)                        # (3, H, W)
    return torch.from_numpy(arr).contiguous()


class BagDataset(Dataset):
    """Iterates over MIL bags from a public-bus-mil-style manifest.

    Parameters
    ----------
    manifest_csv : path
        Bag-level CSV (see schema in module docstring).
    frames_csv : path
        Frame-level CSV (paths relative to ``images_dir``).
    images_dir : path
        Root directory; ``frames.src_path`` joins under this.
    img_size : int
        Output canvas size. URFM/USF-MAE/UltraFedFM/ImageNet ViT-B/16 = 224.
    fold_filter : (int, str) or None
        e.g., ``(5, "test")`` to select bags in fold 5; ``(5, "train")`` to
        select bags NOT in fold 5; or ``None`` for all bags.
    transform_pre_norm : optional callable
        Frame-level augmentation applied to the PIL image BEFORE
        letterbox + ImageNet norm. Receives and returns a PIL image.
    """

    def __init__(
        self,
        manifest_csv: str | Path,
        frames_csv: str | Path,
        images_dir: str | Path,
        img_size: int = 224,
        fold_filter: Optional[tuple] = None,
        transform_pre_norm: Optional[Callable[[Image.Image], Image.Image]] = None,
    ):
        self.manifest = pd.read_csv(manifest_csv)
        self.frames = pd.read_csv(frames_csv)
        self.images_dir = Path(images_dir)
        self.img_size = img_size
        self.transform = transform_pre_norm

        if fold_filter is not None:
            fold, role = fold_filter
            if role not in ("test", "train"):
                raise ValueError("fold_filter role must be 'test' or 'train'")
            mask = self.manifest["fold"] == fold
            if role == "train":
                mask = ~mask
            self.manifest = self.manifest[mask].reset_index(drop=True)

        # Build a per-bag index into frames.csv
        self._frames_by_bag = {
            bag_id: g.sort_values("frame_idx_in_bag")["src_path"].tolist()
            for bag_id, g in self.frames.groupby("bag_id")
        }

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int):
        row = self.manifest.iloc[idx]
        bag_id = str(row["bag_id"])
        label = int(row["label"])
        frame_paths = self._frames_by_bag.get(bag_id, [])
        if not frame_paths:
            raise RuntimeError(f"No frames found for bag {bag_id}")
        tensors = []
        for rel in frame_paths:
            pil = Image.open(self.images_dir / rel).convert("L")
            if self.transform is not None:
                pil = self.transform(pil)
            pil = center_letterbox_to(pil, self.img_size)
            tensors.append(to_3channel_imagenet_tensor(pil))
        bag = torch.stack(tensors, dim=0)  # (N, 3, H, W)
        return bag, label, bag_id


def identity_collate(batch):
    """For DataLoader: pass bags through unmolested (no padding/stacking)."""
    # When batch_size=1, batch is a list of length 1.
    return batch
