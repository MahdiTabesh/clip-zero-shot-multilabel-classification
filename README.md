# CLIP Zero‑Shot Multi‑Label Classification — Rare Traffic Classes

**What this is:** My compact project using **CLIP** to perform **multi‑label classification** on a small, imbalanced traffic dataset.
Because I only had **~100 labeled training images**, I built a **zero‑shot** pipeline: descriptive prompts → CLIP scoring →
**per‑class threshold calibration**. This serves as a strong baseline for future fine‑tuning.

## Highlights
- Patch‑level scoring for large frames; cosine similarity to class prompts
- Prompt engineering (2–5 per class) improves separation
- **Result:** Mean Balanced Accuracy (train split) ≈ **65%** with pure zero‑shot + thresholds

## Next steps
- Add more labeled data and rebalance
- Linear‑probe or LoRA/PEFT fine‑tune CLIP
- Per‑class temperature/Platt scaling; prompt ensembles; test‑time augmentation