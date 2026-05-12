# minimal-mil-urfm

A **minimal, self-contained reference implementation** of multiple-instance-
learning fine-tuning for breast ultrasound on top of any ViT-B/16-compatible
backbone — including **URFM** (Kang et al. 2025), **USF-MAE**, **UltraFedFM**,
or a plain ImageNet ViT.

Designed to pair with the [`public-bus-mil` benchmark](https://github.com/jbaggett/public-bus-mil)
as a complete fine-tuning recipe you can adapt and report against.

The code is intentionally tight (~1,000 lines) so the recipe is legible and
easy to fork. No internal Mayo or institution-specific code; everything
runs on public weights + public data.

---

## What's in the box

| Module | Purpose | Lines |
|---|---|---|
| `minimal_mil_urfm/backbone.py` | timm ViT-B/16 loader with MAE-state-dict filter for URFM-style weights | ~170 |
| `minimal_mil_urfm/model.py` | LoRA adapter wrapping + DSMIL aggregation head | ~180 |
| `minimal_mil_urfm/loss.py` | Bag BCE + soft top-k auxiliary loss | ~80 |
| `minimal_mil_urfm/data.py` | `public-bus-mil`-format bag dataset | ~150 |
| `minimal_mil_urfm/splits.py` | Fold-based split helper | ~50 |
| `minimal_mil_urfm/train.py` | Single-fold training loop (AdamW + cosine + warmup) | ~190 |
| `minimal_mil_urfm/infer.py` | Bag-level inference with optional TTA | ~80 |
| `minimal_mil_urfm/saliency.py` | Attention rollout from ViT block attentions | ~100 |
| `scripts/train_on_public_bus_mil.py` | 5-fold CV training driver | ~190 |
| `tests/` | Smoke + loss-math tests (no GPU, no downloads) | ~150 |

---

## Backbone compatibility

| Pretrained backbone | Drop-in? | Source |
|---|---|---|
| **URFM-B/16** (Kang et al. 2025) | ✅ | <https://huggingface.co/QingboKang/URFM> |
| **URFM-L/16**, **URFM-H/14** | ✅ (use the matching `--timm-model`) | same |
| **USF-MAE** | ✅ | per their public release |
| **UltraFedFM** | ✅ | per their public release |
| Plain **ImageNet ViT-B/16** | ✅ (the timm default) | timm |
| **USFM** (Jiao et al.) | ❌ | uses a different ViT variant; adapt `backbone.py` to load |

For URFM, USF-MAE, and UltraFedFM, the MAE-pretrained encoder weights load
directly into a standard timm `vit_base_patch16_224` after stripping the
decoder + mask-token keys. `backbone.py` does that filtering automatically.

---

## Install

```bash
pip install -e .
# Or with the benchmark + dev test dependencies:
pip install -e ".[benchmark,dev]"
```

The benchmark extra installs the [`public-bus-mil`](https://github.com/jbaggett/public-bus-mil)
package, which provides the bag-construction CLI you'll need to build the
training/evaluation cohort from the BUV + WHBUS public sources.

---

## Quick start — 5-fold CV on public-bus-mil

1. **Build the benchmark** (one-time; ~5 minutes if you already have the
   BUV + WHBUS source downloads):

   ```bash
   # See https://github.com/jbaggett/public-bus-mil for download links
   pip install public-bus-mil
   build-bus-mil-bags \
       --dataset both \
       --buv-extracted /path/to/BUV_Extracted \
       --buv-annotations-train /path/to/imagenet_vid_train_15frames.json \
       --buv-annotations-val   /path/to/imagenet_vid_val.json \
       --whbus /path/to/WHBUS/buvimgs \
       --out ./external_test_v1/
   ```

2. **Download URFM-B/16 weights** (optional; if you skip this, you get an
   ImageNet ViT-B/16 baseline):

   ```bash
   pip install huggingface_hub
   huggingface-cli download QingboKang/URFM \
       mae_vit_base_patch16_dec768d8b_all_biomedclip_1199.pth \
       --local-dir ./weights/urfm
   ```

3. **Run 5-fold CV training**:

   ```bash
   python scripts/train_on_public_bus_mil.py \
       --manifest ./external_test_v1/manifest.csv \
       --frames   ./external_test_v1/frames.csv \
       --images   ./external_test_v1/images \
       --backbone-weights ./weights/urfm/mae_vit_base_patch16_dec768d8b_all_biomedclip_1199.pth \
       --out      ./results/urfm_lora8 \
       --epochs   15 \
       --lora-rank 8
   ```

   Output:
   ```
   results/urfm_lora8/
   ├── fold1/{ckpts, history.json, test_predictions.csv, summary.json}
   ├── fold2/...
   ├── fold3/...
   ├── fold4/...
   ├── fold5/...
   └── summary.json    # aggregate AUROC mean ± SD across the 5 folds
   ```

A single fold takes ~10-20 minutes on an A100 (~1-2 hours on a consumer
GPU); the full 5-fold sweep is ~1-2 hours / ~6-10 hours respectively.

### Fine-tune-recipe variants (sweep these to compare approaches)

```bash
# ImageNet ViT-B/16 baseline (no extra weights download)
python scripts/train_on_public_bus_mil.py ... --out ./results/imagenet_lora8 \
    --epochs 15 --lora-rank 8

# URFM, full fine-tune (no LoRA, all backbone params trainable)
python scripts/train_on_public_bus_mil.py ... --backbone-weights ... \
    --out ./results/urfm_full --epochs 15 --full-finetune

# URFM, linear probe (frozen backbone, only DSMIL head trains)
python scripts/train_on_public_bus_mil.py ... --backbone-weights ... \
    --out ./results/urfm_linear --epochs 15 --linear-probe

# URFM, LoRA rank 16 (vs default rank 8)
python scripts/train_on_public_bus_mil.py ... --backbone-weights ... \
    --out ./results/urfm_lora16 --epochs 15 --lora-rank 16 --lora-alpha 16
```

This gives you a clean head-to-head comparison of fine-tuning approaches on
the same patient-grouped folds — the comparison the public-bus-mil
benchmark is designed for.

---

## How the model is wired

```
input bag (N frames, 3×224×224)
        │
        ▼
   ViT-B/16 backbone
   (URFM or timm ImageNet)
        │  ← LoRA adapters (rank 8) on qkv + proj projections
        ▼
   per-frame CLS embedding (N × 768)
        │
        ▼
        DSMIL head
   ┌─────────────────┐
   │ instance scores │ → soft top-k auxiliary loss
   │      ▼          │
   │ critical inst.  │
   │   attends all   │
   │      ▼          │
   │  bag logit      │ → bag BCE loss
   └─────────────────┘
        │
        ▼
   bag score (probability of malignancy)
```

**For saliency**, pass `return_attn=True` through the model. The backbone
exposes per-block attention weights; `saliency.attention_rollout` composes
them into a CLS-to-patch saliency map. See `tests/test_smoke.py::test_saliency_pipeline`
for a minimal example.

---

## Hyperparameters (default values match the recipe used in Paper 2)

| Knob | Default | CLI flag |
|---|---|---|
| Backbone | timm `vit_base_patch16_224` | `--timm-model` |
| Backbone weights | timm pretrained (ImageNet) | `--backbone-weights /path/to/URFM.pth` |
| Image size | 224 | `--img-size` |
| LoRA rank | 8 | `--lora-rank` |
| LoRA alpha | 8 | `--lora-alpha` |
| LoRA dropout | 0.1 | `--lora-dropout` |
| LoRA target modules | `qkv`, `proj` | (hard-coded in `model.py`) |
| Epochs | 15 | `--epochs` |
| Warmup epochs | 2 | `--warmup-epochs` |
| Learning rate | 2 × 10⁻⁴ (cosine decay) | `--lr` |
| Weight decay | 1 × 10⁻⁴ | `--weight-decay` |
| Gradient clip | L2 norm 1.0 | (hard-coded in `train.py`) |
| Soft top-k k | 5 | `--soft-topk-k` |
| Soft top-k temperature | 0.5 | `--soft-topk-temperature` |
| Auxiliary loss weight (λ) | 0.3 | `--aux-weight` |
| Positive-class weight (BCE) | 1.0 | `--pos-weight` |
| Random seed | 42 | `--seed` |

---

## Limitations / what this code is NOT

- **Not a SOTA implementation.** This is a clean reference, not a tuned
  competitive baseline. With 366 bags in `public-bus-mil` you should
  expect AUROC roughly in the 0.75–0.85 range; clinical-scale training
  (tens of thousands of bags) reaches 0.92+ on the same architecture
  per Paper 2 but requires private data.
- **Not optimized.** Bag iteration is one-at-a-time with no gradient
  accumulation. Adding a real bag-aware sampler + multi-GPU would
  matter for larger cohorts.
- **No mixed-precision** by default. Add `torch.cuda.amp` if you need
  more headroom.
- **Single-bag batches.** Bag sizes vary across the benchmark; padding
  every bag to the max would waste compute. The cost is no batch-level
  parallelism within an iteration; with 366 bags this is fine.
- **No exhaustive callbacks or logging.** Per-epoch print + JSON
  history. Wire in `wandb` or `tensorboard` if needed.

---

## Tests

```bash
pip install -e ".[dev]"
pytest tests/
```

Should report `13 passed` (all on CPU, no downloads).

---

## License

MIT — see `LICENSE`. URFM weights are Apache-2.0 (see the URFM repo);
public-bus-mil source datasets have their own licenses (see DATA_SOURCES.md
in that repo).

---

## Citation

If you use this code, please cite:

- The backbone you're using (URFM, USF-MAE, UltraFedFM, etc.) per its
  source paper.
- The `public-bus-mil` benchmark (separate Zenodo DOI; see its README).
- (Optional) Paper 2 if your fine-tuning recipe specifically uses the
  DSMIL + soft-top-k combination introduced there.
