"""MIL model: ViT backbone + LoRA adapters + DSMIL aggregation head.

The intent is to make the recipe maximally legible in ~300 lines:
  - load any timm ViT-B/16-compatible backbone
  - wrap the attention QKV projections with LoRA adapters via peft
  - extract the CLS embedding for each frame in a bag
  - aggregate frame embeddings into a bag-level score via DSMIL
  - return per-frame instance scores for the soft-top-k auxiliary loss
"""
from __future__ import annotations

from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model

from .backbone import ViTBackbone, load_backbone


class DSMIL(nn.Module):
    """DSMIL aggregation (Li, Li, Eliceiri 2021).

    Two-stream: (1) instance-level classifier scores every instance;
    (2) the *critical* instance (highest score) attends to all other
    instances to produce a bag-level score. The two streams are fused.

    Inputs
    ------
    feats : (N, D)         frame embeddings (one bag at a time)

    Returns
    -------
    bag_logit       : ()        bag-level scalar logit
    instance_logits : (N,)      per-instance logits (used by aux loss)
    attn_weights    : (N,)      attention weights of the bag stream (sum to 1)
    """

    def __init__(self, feat_dim: int, hidden_dim: int = 128, dropout: float = 0.0):
        super().__init__()
        # Instance-level classifier
        self.inst_fc = nn.Linear(feat_dim, 1)
        # Bag-level transform
        self.q = nn.Linear(feat_dim, hidden_dim)
        self.v = nn.Linear(feat_dim, hidden_dim)
        self.bag_fc = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feats = self.dropout(feats)
        instance_logits = self.inst_fc(feats).squeeze(-1)  # (N,)

        # Identify the critical instance (highest logit)
        crit_idx = instance_logits.argmax()
        crit_feat = feats[crit_idx]                       # (D,)

        # Bag stream: attention from critical instance to all instances
        q = self.q(crit_feat)                              # (H,)
        v = self.v(feats)                                  # (N, H)
        # Attention scores (scaled dot product)
        scores = (v @ q) / (q.shape[-1] ** 0.5)            # (N,)
        attn = torch.softmax(scores, dim=0)                # (N,)
        bag_feat = (attn.unsqueeze(-1) * v).sum(dim=0)     # (H,)
        bag_logit = self.bag_fc(bag_feat).squeeze(-1)      # ()

        # DSMIL paper averages the bag-stream logit with the critical-
        # instance logit; that's the fusion step.
        bag_logit = 0.5 * bag_logit + 0.5 * instance_logits[crit_idx]
        return bag_logit, instance_logits, attn


class MILModel(nn.Module):
    """End-to-end MIL classifier: ViT-LoRA encoder + DSMIL.

    Parameters
    ----------
    timm_model_name : str
        timm model name for the backbone (default ``vit_base_patch16_224``).
    backbone_weights : str, optional
        Path to a custom checkpoint (e.g., URFM .pth) or ``hf:repo/file.pth``.
        If ``None``, timm's pretrained weights (typically ImageNet) are used.
    img_size : int
        Backbone input resolution. URFM uses 224.
    lora_rank : int
        LoRA rank (default 8 per Paper 2). Set to 0 to disable LoRA and
        train with full backbone fine-tuning (or with the backbone frozen,
        depending on ``freeze_backbone``).
    lora_alpha : int
        LoRA scaling factor (default = rank).
    lora_dropout : float
        Dropout inside LoRA adapters (default 0.1).
    lora_target_modules : list of str
        Module-name suffixes to inject LoRA into. timm ViT uses ``qkv`` and
        ``proj`` in each attention block; both are conventional targets.
    freeze_backbone : bool
        If True (the default with LoRA enabled), the backbone is frozen
        except for the LoRA adapters. If False, all backbone parameters
        train. With ``lora_rank=0`` and ``freeze_backbone=False`` you get
        a standard full-fine-tune baseline.
    dsmil_hidden_dim : int
    dsmil_dropout : float
    """

    def __init__(
        self,
        timm_model_name: str = "vit_base_patch16_224",
        backbone_weights: Optional[str] = None,
        img_size: int = 224,
        lora_rank: int = 8,
        lora_alpha: int = 8,
        lora_dropout: float = 0.1,
        lora_target_modules: Tuple[str, ...] = ("qkv", "proj"),
        freeze_backbone: bool = True,
        dsmil_hidden_dim: int = 128,
        dsmil_dropout: float = 0.0,
        pretrained: bool = True,
    ):
        super().__init__()
        self.backbone = load_backbone(
            timm_model_name=timm_model_name,
            backbone_weights=backbone_weights,
            img_size=img_size,
            pretrained=pretrained,
        )
        feat_dim = self.backbone.embed_dim

        if lora_rank > 0:
            cfg = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=list(lora_target_modules),
                bias="none",
            )
            # peft wraps the whole module; we want to wrap only the ViT.
            self.backbone.vit = get_peft_model(self.backbone.vit, cfg)
            # peft freezes everything except LoRA params by default; nothing
            # more to do if freeze_backbone=True. If freeze_backbone=False we
            # still want to train ALL backbone params alongside LoRA.
            if not freeze_backbone:
                for p in self.backbone.vit.parameters():
                    p.requires_grad = True
        else:
            # No LoRA — either full fine-tune or fully frozen
            for p in self.backbone.vit.parameters():
                p.requires_grad = not freeze_backbone

        self.head = DSMIL(
            feat_dim=feat_dim,
            hidden_dim=dsmil_hidden_dim,
            dropout=dsmil_dropout,
        )

    # ---------------------------------------------------------------------
    # Forward

    def _encode_frames(
        self, x: torch.Tensor, return_attn: bool = False
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Run the backbone on N frames and return CLS embeddings.

        Input  : (N, 3, H, W)
        Output : feats (N, D), attns list (length=blocks, each (N, heads, T, T))
                 or empty list if return_attn=False
        """
        tokens, attns = self.backbone(x, return_attn=return_attn)
        # CLS is token 0
        cls = tokens[:, 0, :]
        return cls, attns

    def forward(
        self,
        bag: torch.Tensor,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """One bag at a time.

        Input
        -----
        bag : (N_frames, 3, H, W)

        Returns
        -------
        bag_logit       : ()
        instance_logits : (N_frames,)
        attn_weights    : (N_frames,)        DSMIL bag-stream attention
        backbone_attns  : list of (N_frames, heads, T, T)
                          ViT block attentions if ``return_attn=True``,
                          else empty list
        """
        feats, backbone_attns = self._encode_frames(bag, return_attn=return_attn)
        bag_logit, instance_logits, attn = self.head(feats)
        return bag_logit, instance_logits, attn, backbone_attns

    # ---------------------------------------------------------------------
    # Trainable-parameter accounting (for transparency in reports)

    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def summary(self) -> str:
        tr = self.trainable_params()
        tot = self.total_params()
        return (
            f"MILModel: backbone={type(self.backbone.vit).__name__}, "
            f"feat_dim={self.backbone.embed_dim}, "
            f"trainable_params={tr:,} / {tot:,} ({100*tr/tot:.2f}%)"
        )
