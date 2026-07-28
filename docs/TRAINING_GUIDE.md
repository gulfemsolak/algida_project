# Training Guide

## Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA 11.8+ recommended (CPU training works but is slow)
- ~20 GB free disk space for data + model weights

---

## Step 1 — Environment Setup

```bash
# Clone and enter the project
cd icecream-shelf-detector

# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify GPU availability
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Step 2 — Data Preparation Checklist

- [ ] Raw shelf photos collected (minimum 200 images recommended)
- [ ] Images placed in `data/raw/`
- [ ] All images are JPG or PNG
- [ ] Images are labelled (see `docs/ANNOTATION_GUIDE.md`)
- [ ] Label `.txt` files placed in `data/annotations/`

---

## Step 3 — Annotate with Roboflow

1. Go to [roboflow.com](https://roboflow.com) and create a free account
2. Create a new project → **Object Detection**
3. Upload images from `data/raw/`
4. Label each image using the class names from `config/config.yaml`:
   - `magnum_classic`, `magnum_almond`, `magnum_white`
   - `cornetto_vanilla`, `cornetto_chocolate`, `cornetto_strawberry`
   - `popsicle_fruit`, `popsicle_chocolate`
   - `sandwich_ice_cream`, `cup_ice_cream`, `empty_slot`
5. Export as **YOLOv8 format**
6. Place exported `.txt` files in `data/annotations/`

See `docs/ANNOTATION_GUIDE.md` for detailed per-class guidelines.

---

## Step 4 — Run Augmentation

```bash
python -m src.data.augmentor \
    --raw-images  data/raw/ \
    --raw-labels  data/annotations/ \
    --out-images  data/augmented/images/ \
    --out-labels  data/augmented/labels/
```

This generates 5 augmented versions per image by default (configurable in `config/config.yaml` → `augmentation.num_augmented_per_image`).

---

## Step 5 — Validate Dataset

```bash
python -m src.data.validator \
    --images-dir data/raw/ \
    --labels-dir data/annotations/ \
    --report     reports/validation_report.json
```

Review `reports/validation_report.json` and fix any issues before proceeding.

---

## Step 6 — Build Splits

```bash
python -m src.data.dataset_builder \
    --raw-images data/augmented/images/ \
    --raw-labels data/augmented/labels/
```

This creates `data/splits/` with `train/`, `val/`, `test/` folders and writes `config/dataset.yaml`.

---

## Step 7 — Start Training

```bash
# Quick one-liner:
./scripts/train.sh yolov8m

# Or manually:
python -m src.models.trainer --model yolov8m --config config/config.yaml
```

Training logs appear in the terminal. Weights are saved to `runs/<experiment>/weights/`.

### Resuming interrupted training

```bash
python -m src.models.trainer \
    --model yolov8m \
    --resume \
    --resume-weights runs/<experiment>/weights/last.pt
```

---

## Step 8 — Monitor Training

TensorBoard is not required — training curves are saved as `results.csv` in `runs/<experiment>/`.

View them in the dashboard:
```bash
./scripts/demo.sh
# → Model Metrics page → select your run
```

---

## Step 9 — Evaluate

```bash
./scripts/evaluate.sh runs/<experiment>/weights/best.pt test
```

This writes `reports/evaluation_report.json` and `reports/evaluation_report.html`.

---

## Step 10 — Compare Models

Train nano and small variants, then compare:

```bash
python -m src.models.model_comparison \
    --nano-weights   runs/<exp_nano>/weights/best.pt \
    --small-weights  runs/<exp_small>/weights/best.pt \
    --medium-weights runs/<exp_medium>/weights/best.pt
```

Results saved to `reports/model_comparison/comparison_report.json`.

---

## Common Issues and Fixes

| Issue | Fix |
|-------|-----|
| `CUDA out of memory` | Reduce `batch_size` in `config/config.yaml` (try 8 or 4) |
| Training immediately stops (patience=0) | Ensure `dataset.yaml` paths are absolute or correct relative to CWD |
| Low mAP after 100 epochs | Increase `epochs`, check class balance, add more training data |
| `dataset.yaml` not found | Run `dataset_builder.py` first |
| ImportError for ultralytics | Run `pip install ultralytics` |
| Augmentation produces black images | Check that label `.txt` files exist for all images |
