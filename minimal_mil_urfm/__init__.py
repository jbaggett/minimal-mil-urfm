"""minimal-mil-urfm — a clean reference implementation of MIL fine-tuning
for breast ultrasound on top of any ViT-B/16-compatible timm backbone.

Headline use case: fine-tune URFM (Kang et al. 2025,
https://github.com/sonovision-ai/URFM) with LoRA rank-8 adapters and a
DSMIL aggregation head on the public-bus-mil benchmark
(https://github.com/jbaggett/public-bus-mil) using 5-fold patient-grouped
cross-validation.

The backbone is supplied by the user via a timm model name (`vit_base_patch16_224`
by default) plus an optional checkpoint path. Several ultrasound-pretrained
backbones share this architecture and load directly with `--backbone-weights`:

  - URFM           (Kang et al. 2025)        — recommended default
  - USF-MAE        (per their public release)
  - UltraFedFM     (per their public release)
  - Plain ImageNet ViT-B/16                   — useful for ablation

`USFM` (Universal Salient-feature ultrasound Foundation Model, Jiao et al.)
is NOT directly compatible because it uses a different ViT variant; load it
separately by adapting `minimal_mil_urfm.backbone.load_backbone`.

The code is intentionally minimal — single-file modules where reasonable, no
internal abstraction layers — to make the recipe legible and easy to fork.
"""
__version__ = "0.1.0"

from .model import MILModel
from .loss import bag_bce_with_soft_topk
from .data import BagDataset
from .saliency import attention_rollout
from .splits import fold_split

__all__ = [
    "MILModel",
    "bag_bce_with_soft_topk",
    "BagDataset",
    "attention_rollout",
    "fold_split",
]
