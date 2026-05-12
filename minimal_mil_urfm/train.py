"""Single-fold training loop.

This is intentionally minimal: AdamW + cosine schedule with warmup,
bag-by-bag iteration (batch_size=1 over bags), bag CE + soft top-k loss,
validation each epoch, checkpoint at best validation AUROC.

Exposes one function ``train_one_fold`` that's called by the multi-fold
driver in ``scripts/train_on_public_bus_mil.py``.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from .model import MILModel
from .loss import bag_bce_with_soft_topk
from .data import BagDataset, identity_collate


@dataclass
class TrainConfig:
    # Optimizer
    lr: float = 2e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 2
    n_epochs: int = 15
    grad_clip_norm: float = 1.0
    # Class balance
    pos_weight: float = 1.0          # adjust if your cohort is imbalanced
    # Auxiliary loss
    aux_weight: float = 0.3
    soft_topk_k: int = 5
    soft_topk_temperature: float = 0.5
    # Misc
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    log_every_n_bags: int = 50
    ckpt_dir: Optional[str] = None


def cosine_with_warmup(step: int, total_steps: int, warmup_steps: int,
                      base_lr: float, min_lr: float = 1e-6) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


def _run_one_epoch_train(model, loader, opt, cfg, step_state):
    model.train()
    total_loss = total_bag = total_aux = 0.0
    n_bags = 0
    pbar = tqdm(loader, desc=f"train epoch {step_state['epoch']:02d}", leave=False)
    for batch in pbar:
        # batch is a list of length 1 thanks to identity_collate
        bag, label, _bag_id = batch[0]
        bag = bag.to(cfg.device, non_blocking=True)
        label_t = torch.tensor(label, device=cfg.device, dtype=torch.float32)

        # LR schedule (per-step)
        lr_now = cosine_with_warmup(
            step_state["step"], step_state["total_steps"],
            step_state["warmup_steps"], cfg.lr,
        )
        for g in opt.param_groups:
            g["lr"] = lr_now

        bag_logit, instance_logits, _attn, _bbattn = model(bag)
        loss, l_bag, l_aux = bag_bce_with_soft_topk(
            bag_logit, instance_logits, label_t,
            pos_weight=cfg.pos_weight, aux_weight=cfg.aux_weight,
            k=cfg.soft_topk_k, temperature=cfg.soft_topk_temperature,
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip_norm and cfg.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                cfg.grad_clip_norm,
            )
        opt.step()

        total_loss += float(loss.item())
        total_bag  += float(l_bag.item())
        total_aux  += float(l_aux.item())
        n_bags     += 1
        step_state["step"] += 1
        pbar.set_postfix({"loss": f"{loss.item():.3f}", "lr": f"{lr_now:.2e}"})
    return {
        "loss": total_loss / max(1, n_bags),
        "bag_loss": total_bag / max(1, n_bags),
        "aux_loss": total_aux / max(1, n_bags),
        "n_bags": n_bags,
    }


@torch.no_grad()
def _run_one_epoch_eval(model, loader, cfg):
    model.eval()
    y_true, y_prob, bag_ids = [], [], []
    for batch in tqdm(loader, desc="val", leave=False):
        bag, label, bag_id = batch[0]
        bag = bag.to(cfg.device, non_blocking=True)
        bag_logit, _il, _a, _ba = model(bag)
        prob = float(torch.sigmoid(bag_logit).item())
        y_true.append(int(label))
        y_prob.append(prob)
        bag_ids.append(bag_id)
    if len(set(y_true)) < 2:
        auroc = float("nan")
        auprc = float("nan")
    else:
        auroc = float(roc_auc_score(y_true, y_prob))
        auprc = float(average_precision_score(y_true, y_prob))
    return {
        "auroc": auroc, "auprc": auprc,
        "y_true": y_true, "y_prob": y_prob, "bag_ids": bag_ids,
    }


def train_one_fold(
    train_ds: BagDataset,
    val_ds: BagDataset,
    *,
    timm_model_name: str = "vit_base_patch16_224",
    backbone_weights: Optional[str] = None,
    img_size: int = 224,
    lora_rank: int = 8,
    lora_alpha: int = 8,
    lora_dropout: float = 0.1,
    freeze_backbone: bool = True,
    pretrained: bool = True,
    cfg: Optional[TrainConfig] = None,
):
    """Train a single fold; return the final-val + best-val metrics + per-bag
    val predictions, plus the trained model (in memory)."""
    if cfg is None:
        cfg = TrainConfig()

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    model = MILModel(
        timm_model_name=timm_model_name,
        backbone_weights=backbone_weights,
        img_size=img_size,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        freeze_backbone=freeze_backbone,
        pretrained=pretrained,
    ).to(cfg.device)
    print(model.summary())

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)

    train_loader = DataLoader(
        train_ds, batch_size=1, shuffle=True, num_workers=2,
        collate_fn=identity_collate, pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=2,
        collate_fn=identity_collate, pin_memory=torch.cuda.is_available(),
    )

    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * cfg.n_epochs
    warmup_steps = steps_per_epoch * cfg.warmup_epochs
    step_state = {"step": 0, "total_steps": total_steps,
                  "warmup_steps": warmup_steps, "epoch": 0}

    best = {"epoch": -1, "auroc": -1.0, "auprc": -1.0,
            "y_true": [], "y_prob": [], "bag_ids": []}
    history = []
    for epoch in range(1, cfg.n_epochs + 1):
        step_state["epoch"] = epoch
        t0 = time.time()
        tr = _run_one_epoch_train(model, train_loader, opt, cfg, step_state)
        ev = _run_one_epoch_eval(model, val_loader, cfg)
        dt = time.time() - t0
        print(
            f"  epoch {epoch:02d}: train_loss={tr['loss']:.4f} "
            f"(bag={tr['bag_loss']:.4f} aux={tr['aux_loss']:.4f}) "
            f"val_auroc={ev['auroc']:.4f} val_auprc={ev['auprc']:.4f} "
            f"[{dt:.1f}s]"
        )
        history.append({"epoch": epoch, **tr,
                        "val_auroc": ev["auroc"], "val_auprc": ev["auprc"]})
        if ev["auroc"] > best["auroc"]:
            best = {"epoch": epoch, **ev}
            if cfg.ckpt_dir:
                ckpt = Path(cfg.ckpt_dir) / f"best_epoch{epoch:02d}.pt"
                ckpt.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), ckpt)

    return {
        "model": model,
        "best": best,
        "history": history,
    }
