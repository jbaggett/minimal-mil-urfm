"""End-to-end smoke tests with synthetic data — no GPU and no downloads.

These exercise the full model + loss + saliency + (mini) training loop
to catch shape/typing/regression bugs without requiring URFM weights or
the public-bus-mil benchmark.
"""
from __future__ import annotations

import torch
import pytest

from minimal_mil_urfm.model import MILModel
from minimal_mil_urfm.loss import bag_bce_with_soft_topk
from minimal_mil_urfm.saliency import attention_rollout, saliency_map_per_frame, peak_xy


@pytest.fixture(scope="module")
def model_cpu():
    """Build a small MILModel with LoRA + DSMIL on CPU.

    We use `pretrained=False` to skip the timm checkpoint download.
    """
    torch.manual_seed(0)
    m = MILModel(
        timm_model_name="vit_base_patch16_224",
        backbone_weights=None,
        img_size=224,
        lora_rank=4,
        lora_alpha=4,
        lora_dropout=0.0,
        freeze_backbone=True,
        pretrained=False,
    )
    return m


def test_forward_shapes(model_cpu):
    """One bag of 4 frames in → bag scalar + 4 instance logits out."""
    bag = torch.randn(4, 3, 224, 224)
    bag_logit, inst, attn, attns = model_cpu(bag)
    assert bag_logit.shape == ()
    assert inst.shape == (4,)
    assert attn.shape == (4,)
    # By default we don't ask for backbone attentions
    assert attns == []


def test_forward_with_attentions(model_cpu):
    bag = torch.randn(3, 3, 224, 224)
    _bl, _il, _a, attns = model_cpu(bag, return_attn=True)
    assert len(attns) == 12, "ViT-B/16 has 12 blocks"
    # Each attention tensor: (N=3 frames, heads=12, T=197, T=197)
    for a in attns:
        assert a.dim() == 4
        assert a.shape[0] == 3
        assert a.shape[-1] == a.shape[-2] == 197


def test_loss_runs_and_is_finite(model_cpu):
    bag = torch.randn(4, 3, 224, 224)
    label = torch.tensor(1.0)
    bag_logit, inst, _a, _ba = model_cpu(bag)
    total, l_bag, l_aux = bag_bce_with_soft_topk(bag_logit, inst, label)
    assert torch.isfinite(total)
    assert torch.isfinite(l_bag)
    assert torch.isfinite(l_aux)
    # Backprop should run and produce gradients in trainable params
    total.backward()
    n_with_grad = sum(
        1 for p in model_cpu.parameters()
        if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0
    )
    assert n_with_grad > 0


def test_saliency_pipeline(model_cpu):
    bag = torch.randn(2, 3, 224, 224)
    _bl, _il, _a, attns = model_cpu(bag, return_attn=True)
    rollout = attention_rollout(attns)
    assert rollout.shape == (2, 14 * 14), "ViT-B/16 @ 224 has 14×14 patches"
    sal_map = saliency_map_per_frame(rollout, patch_size=16, img_size=224)
    assert sal_map.shape == (2, 224, 224)
    x, y = peak_xy(sal_map[0])
    assert 0 <= x < 224 and 0 <= y < 224


def test_trainable_param_count_with_lora(model_cpu):
    """With LoRA rank=4 + frozen backbone, trainable params should be a
    small fraction of total (≪ 1% for a ViT-B/16)."""
    tr = model_cpu.trainable_params()
    tot = model_cpu.total_params()
    assert tr < tot, "should not be training the whole backbone"
    assert tr / tot < 0.05, f"LoRA fraction unexpectedly high: {tr/tot:.4f}"


def test_full_finetune_mode():
    """No LoRA, freeze_backbone=False → ~all params trainable."""
    m = MILModel(
        timm_model_name="vit_base_patch16_224",
        pretrained=False,
        lora_rank=0,
        freeze_backbone=False,
    )
    tr, tot = m.trainable_params(), m.total_params()
    assert tr / tot > 0.99


def test_linear_probe_mode():
    """No LoRA, frozen backbone → trainable params are just the DSMIL head."""
    m = MILModel(
        timm_model_name="vit_base_patch16_224",
        pretrained=False,
        lora_rank=0,
        freeze_backbone=True,
    )
    tr, tot = m.trainable_params(), m.total_params()
    # DSMIL head is tiny (< 0.1% of a ViT-B/16)
    assert tr / tot < 0.01, f"linear probe trainable fraction too high: {tr/tot:.4f}"
