#!/usr/bin/env python3
"""5-fold CV training on the public-bus-mil benchmark.

For each of folds 1..5, train on the other four (with one of them held
out as validation) and evaluate on the held-out fold. Aggregate AUROC
mean and SD across folds.

Usage examples
--------------
    # ImageNet ViT-B/16 baseline (no extra downloads)
    python scripts/train_on_public_bus_mil.py \
        --manifest /path/to/benchmark/manifest.csv \
        --frames   /path/to/benchmark/frames.csv \
        --images   /path/to/benchmark/images \
        --out      ./results/imagenet_vit \
        --epochs 15 --lora-rank 8

    # URFM-pretrained ViT-B/16 (after downloading from
    # https://huggingface.co/QingboKang/URFM)
    python scripts/train_on_public_bus_mil.py \
        --manifest /path/to/benchmark/manifest.csv \
        --frames   /path/to/benchmark/frames.csv \
        --images   /path/to/benchmark/images \
        --backbone-weights /path/to/mae_vit_base_patch16_dec768d8b_all_biomedclip_1199.pth \
        --out      ./results/urfm_vit \
        --epochs 15 --lora-rank 8

    # Quick smoke test on a single fold (fold 5 as test, fold 4 as val, fold 3 as train)
    python scripts/train_on_public_bus_mil.py \
        --manifest ... --frames ... --images ... --out ./results/smoke \
        --epochs 2 --folds 5 --train-folds-override 3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from minimal_mil_urfm.data import BagDataset
from minimal_mil_urfm.train import TrainConfig, train_one_fold
from minimal_mil_urfm.infer import predict


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, help="bag-level CSV (public-bus-mil)")
    p.add_argument("--frames", required=True, help="frame-level CSV (public-bus-mil)")
    p.add_argument("--images", required=True, help="root dir for frame paths")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4, 5],
                   help="which folds to use as TEST (default: all five)")
    p.add_argument("--train-folds-override", type=int, default=None,
                   help="(debug) restrict training pool to this many folds")
    p.add_argument("--timm-model", default="vit_base_patch16_224")
    p.add_argument("--backbone-weights", default=None,
                   help="path to a URFM/USF-MAE/UltraFedFM .pth, or hf:repo/file.pth")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=8)
    p.add_argument("--lora-dropout", type=float, default=0.1)
    p.add_argument("--full-finetune", action="store_true",
                   help="(no LoRA) train all backbone params")
    p.add_argument("--linear-probe", action="store_true",
                   help="(no LoRA, frozen backbone) train only the DSMIL head")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=int, default=2)
    p.add_argument("--aux-weight", type=float, default=0.3)
    p.add_argument("--soft-topk-k", type=int, default=5)
    p.add_argument("--soft-topk-temperature", type=float, default=0.5)
    p.add_argument("--pos-weight", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None,
                   help="cuda or cpu; auto-detect if not specified")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve LoRA / fine-tune mode
    if args.linear_probe:
        lora_rank, freeze = 0, True
    elif args.full_finetune:
        lora_rank, freeze = 0, False
    else:
        lora_rank, freeze = args.lora_rank, True

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Backbone: {args.timm_model}  weights={args.backbone_weights or '(timm default)'}")
    print(f"Mode: {'linear-probe' if args.linear_probe else 'full-fine-tune' if args.full_finetune else f'LoRA rank={lora_rank}'}")

    fold_results = []
    for test_fold in args.folds:
        # val fold = next-lower (wrap)
        val_fold = test_fold - 1 if test_fold > 1 else 5
        print(f"\n=== Fold {test_fold} as test (val=fold{val_fold}) ===")

        train_filter = ("__exclude__", {test_fold, val_fold})  # placeholder marker
        # BagDataset only supports single-fold filtering; we apply it via the
        # underlying manifest by hand here.
        train_ds = BagDataset(
            manifest_csv=args.manifest, frames_csv=args.frames,
            images_dir=args.images, img_size=args.img_size,
            fold_filter=None,
        )
        # Trim manifest: train = NOT in {test, val}
        m = train_ds.manifest
        train_ds.manifest = m[~m["fold"].isin([test_fold, val_fold])].reset_index(drop=True)
        if args.train_folds_override is not None:
            # Keep only the first N rows for debugging
            train_ds.manifest = train_ds.manifest.head(args.train_folds_override * 50).reset_index(drop=True)

        val_ds = BagDataset(
            manifest_csv=args.manifest, frames_csv=args.frames,
            images_dir=args.images, img_size=args.img_size,
            fold_filter=(val_fold, "test"),
        )
        test_ds = BagDataset(
            manifest_csv=args.manifest, frames_csv=args.frames,
            images_dir=args.images, img_size=args.img_size,
            fold_filter=(test_fold, "test"),
        )
        print(f"  train n={len(train_ds)}, val n={len(val_ds)}, test n={len(test_ds)}")

        cfg = TrainConfig(
            lr=args.lr, weight_decay=args.weight_decay,
            warmup_epochs=args.warmup_epochs, n_epochs=args.epochs,
            pos_weight=args.pos_weight,
            aux_weight=args.aux_weight,
            soft_topk_k=args.soft_topk_k,
            soft_topk_temperature=args.soft_topk_temperature,
            device=device, seed=args.seed,
            ckpt_dir=str(out_dir / f"fold{test_fold}/ckpts"),
        )

        result = train_one_fold(
            train_ds, val_ds,
            timm_model_name=args.timm_model,
            backbone_weights=args.backbone_weights,
            img_size=args.img_size,
            lora_rank=lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            freeze_backbone=freeze,
            pretrained=True,
            cfg=cfg,
        )

        # Test-set inference at best-val checkpoint (model is already loaded in memory)
        test_pred = predict(result["model"], test_ds, device=device, tta_passes=1)
        if len(set(test_pred["y_true"])) >= 2:
            from sklearn.metrics import roc_auc_score, average_precision_score
            test_auroc = float(roc_auc_score(test_pred["y_true"], test_pred["y_prob"]))
            test_auprc = float(average_precision_score(test_pred["y_true"], test_pred["y_prob"]))
        else:
            test_auroc = float("nan"); test_auprc = float("nan")

        fold_dir = out_dir / f"fold{test_fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        with open(fold_dir / "history.json", "w") as f:
            json.dump(result["history"], f, indent=2)
        pd.DataFrame({"bag_id": test_pred["bag_ids"],
                      "y_true": test_pred["y_true"],
                      "y_prob": test_pred["y_prob"]}
                    ).to_csv(fold_dir / "test_predictions.csv", index=False)
        summary = {
            "test_fold": test_fold,
            "val_fold": val_fold,
            "best_val_epoch": result["best"]["epoch"],
            "best_val_auroc": result["best"]["auroc"],
            "test_auroc": test_auroc,
            "test_auprc": test_auprc,
        }
        with open(fold_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  fold {test_fold}: test AUROC={test_auroc:.4f}  AUPRC={test_auprc:.4f}")
        fold_results.append(summary)

    # Aggregate
    aurocs = np.array([r["test_auroc"] for r in fold_results
                       if not np.isnan(r["test_auroc"])])
    auprcs = np.array([r["test_auprc"] for r in fold_results
                       if not np.isnan(r["test_auprc"])])
    aggregate = {
        "n_folds": len(aurocs),
        "test_auroc_mean": float(aurocs.mean()) if len(aurocs) else float("nan"),
        "test_auroc_sd":   float(aurocs.std(ddof=1)) if len(aurocs) > 1 else 0.0,
        "test_auprc_mean": float(auprcs.mean()) if len(auprcs) else float("nan"),
        "test_auprc_sd":   float(auprcs.std(ddof=1)) if len(auprcs) > 1 else 0.0,
        "per_fold": fold_results,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(aggregate, f, indent=2)
    print("\n=== Cross-validation summary ===")
    print(f"  test AUROC: {aggregate['test_auroc_mean']:.4f} "
          f"± {aggregate['test_auroc_sd']:.4f} ({len(aurocs)} folds)")
    print(f"  test AUPRC: {aggregate['test_auprc_mean']:.4f} "
          f"± {aggregate['test_auprc_sd']:.4f}")
    print(f"  Full summary: {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
