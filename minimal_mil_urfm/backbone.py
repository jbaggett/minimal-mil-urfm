"""ViT-B/16 backbone loader for minimal-mil-urfm.

Wraps a timm Vision Transformer to expose:
  - per-patch token features (B, N_patches+1, D) with the CLS prepended
  - the attention weights from every block (for attention-rollout saliency)

Supports loading custom MAE-style checkpoints (e.g., URFM, USF-MAE,
UltraFedFM) by filtering out decoder/mask_token keys from a state_dict
and loading the encoder-only weights into a standard timm ViT.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import timm


# Architectures known to be drop-in compatible with timm vit_base_patch16_224.
COMPATIBLE_TIMM_MODELS = {
    "vit_base_patch16_224",
    "vit_base_patch16_224.mae",
    "vit_base_patch16_384",
}

# Documented ultrasound-pretrained checkpoints that load cleanly into a
# timm ViT-B/16 encoder via the MAE-key-filter loader below.
KNOWN_US_CHECKPOINTS = {
    "URFM-B/16":      "https://huggingface.co/QingboKang/URFM",
    "URFM-L/16":      "https://huggingface.co/QingboKang/URFM",
    "URFM-H/14":      "https://huggingface.co/QingboKang/URFM",
    "USF-MAE":        "(see USF-MAE source release)",
    "UltraFedFM":     "(see UltraFedFM source release)",
}


class ViTBackbone(nn.Module):
    """Wraps a timm ViT to expose token features + per-layer attention.

    Forward returns:
      tokens : (B, 1 + N_patches, D)        — pre-norm token sequence after
                                              all transformer blocks (incl. CLS)
      attns  : list of (B, n_heads, T, T)   — one attention tensor per block,
                                              already softmax'd. Empty list if
                                              ``return_attn=False`` to save mem.
    """

    def __init__(
        self,
        timm_model_name: str = "vit_base_patch16_224",
        img_size: int = 224,
        in_chans: int = 3,
        backbone_weights: Optional[str] = None,
        pretrained: bool = True,
    ):
        super().__init__()
        # If a local checkpoint is supplied, we don't need timm to also try
        # downloading ImageNet weights.
        use_timm_pretrained = pretrained and backbone_weights is None
        self.vit = timm.create_model(
            timm_model_name,
            pretrained=use_timm_pretrained,
            img_size=img_size,
            in_chans=in_chans,
            num_classes=0,  # remove classifier head
        )
        if backbone_weights is not None:
            self._load_backbone_weights(backbone_weights)
        self.embed_dim = self.vit.embed_dim
        self.n_blocks = len(self.vit.blocks)

        # Hook every block's attention softmax for attention-rollout
        self._attn_storage: List[torch.Tensor] = []
        self._hook_handles = []
        self._return_attn = False
        self._register_attention_hooks()

    def _load_backbone_weights(self, path_or_repo: str):
        """Load a state_dict from a local .pth or a HuggingFace repo file.

        Filters MAE-style decoder + mask_token keys so encoder-only weights
        flow into the timm ViT cleanly.
        """
        sd = _load_state_dict_any(path_or_repo)
        sd = _filter_mae_encoder_keys(sd)
        missing, unexpected = self.vit.load_state_dict(sd, strict=False)
        # Quiet expected misses (head.weight/head.bias since num_classes=0)
        missing = [k for k in missing if not k.startswith(("head.",))]
        if missing or unexpected:
            msg = []
            if missing:
                msg.append(f"  missing keys (first 5): {missing[:5]}")
            if unexpected:
                msg.append(f"  unexpected keys (first 5): {unexpected[:5]}")
            print(
                "[backbone] partial state_dict load (this can be normal for "
                "MAE → encoder-only loading):\n" + "\n".join(msg)
            )

    def _register_attention_hooks(self):
        """Hook each transformer block's softmax output via the attn_drop module.

        timm's Attention computes ``attn = softmax(QK^T/sqrt(d))`` and then
        applies an ``attn_drop`` Dropout. We pre-hook attn_drop so we
        receive the softmax tensor as its input. We also disable fused
        attention so the unfused (Python) path runs.
        """
        def make_hook(layer_idx: int):
            def hook(module, inputs):
                # `inputs` is a tuple; the softmax tensor is inputs[0].
                if not self._return_attn:
                    return
                self._attn_storage.append(inputs[0].detach())
            return hook

        for i, block in enumerate(self.vit.blocks):
            # Force timm Attention to use the unfused path so attn_drop runs
            if hasattr(block.attn, "fused_attn"):
                block.attn.fused_attn = False
            self._hook_handles.append(
                block.attn.attn_drop.register_forward_pre_hook(make_hook(i))
            )

    def forward(
        self,
        x: torch.Tensor,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        self._attn_storage = []
        self._return_attn = return_attn

        # Run the patch-embed + pos-embed + blocks (no head)
        # timm exposes forward_features which returns the full token sequence.
        tokens = self.vit.forward_features(x)
        attns = self._attn_storage if return_attn else []
        return tokens, attns


def load_backbone(
    timm_model_name: str = "vit_base_patch16_224",
    backbone_weights: Optional[str] = None,
    img_size: int = 224,
    in_chans: int = 3,
    pretrained: bool = True,
) -> ViTBackbone:
    """Convenience factory. See ``ViTBackbone`` docstring for details."""
    return ViTBackbone(
        timm_model_name=timm_model_name,
        img_size=img_size,
        in_chans=in_chans,
        backbone_weights=backbone_weights,
        pretrained=pretrained,
    )


# ----- utilities ----------------------------------------------------------

def _load_state_dict_any(path_or_repo: str) -> dict:
    """Load a state_dict from a local file or a string of the form
    ``hf:repo_id/file.pth`` (HuggingFace Hub)."""
    if path_or_repo.startswith("hf:"):
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise ImportError(
                "huggingface_hub required for hf: paths; "
                "pip install minimal-mil-urfm[hf]"
            ) from e
        spec = path_or_repo[3:]  # strip "hf:"
        if "/" not in spec:
            raise ValueError(f"Bad hf: spec '{path_or_repo}'. Use hf:org/repo/file.pth")
        repo_id, filename = spec.rsplit("/", 1) if spec.count("/") >= 2 else (spec, None)
        # Allow hf:repo_id:filename or hf:repo_id/filename
        if filename is None or ".pth" not in filename and ".pt" not in filename:
            # Treat last component as the filename
            parts = spec.split("/")
            repo_id = "/".join(parts[:-1])
            filename = parts[-1]
        path = hf_hub_download(repo_id=repo_id, filename=filename)
    else:
        path = path_or_repo
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        raise ValueError(f"Cannot extract state_dict from {path}; got {type(obj)}")
    # Prefer EMA weights (typically cleaner — no `module.` prefix), then
    # state_dict, then the dict itself. URFM's `model` key is a string
    # (the architecture name) so we never want to use it as a state_dict.
    for key in ("ema_state_dict", "state_dict", "model_ema", "weights"):
        if key in obj and isinstance(obj[key], dict):
            return obj[key]
    # Some MAE checkpoints store weights directly under "model"
    if "model" in obj and isinstance(obj["model"], dict):
        return obj["model"]
    # Last resort: treat the whole dict as a state_dict iff it has only
    # tensor values (no metadata fields).
    if all(hasattr(v, "shape") for v in obj.values()):
        return obj
    raise ValueError(
        f"Cannot find a state_dict in {path}; top-level keys: {list(obj.keys())}"
    )


def _filter_mae_encoder_keys(sd: dict) -> dict:
    """Drop MAE decoder + mask-token keys; remap if needed."""
    keep = {}
    for k, v in sd.items():
        # MAE decoder keys
        if k.startswith("decoder_"):
            continue
        if k == "mask_token":
            continue
        # Some MAE checkpoints prefix with "encoder." or "module."
        if k.startswith("module."):
            k = k[len("module."):]
        if k.startswith("encoder."):
            k = k[len("encoder."):]
        keep[k] = v
    return keep
