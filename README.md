# DeLR – Dual-encoder Landmark Regression

A PyTorch implementation of the **DeLR (Dual-encoder Landmark Regression)** architecture for cephalometric landmark detection, evaluated on two public datasets:

- **Aariz Cephalograms** — 1000 images, 29 annotated landmarks (700 / 150 / 150 train/valid/test).
- **CephAdoAdu Dataset** — 700 images, 10 landmarks, mixed adolescent + adult cohort (400 train / 300 test in the official splits; we hold out 10% of train as validation).

## Pretrained checkpoints

Trained weights are hosted on **Hugging Face**:

🔗 https://huggingface.co/datasets/emad2001/DeLR-Cephalometric-ConvNeXtV2

```bash
# Option 1 — Hugging Face CLI (recommended)
pip install huggingface_hub
huggingface-cli download emad2001/DeLR-Cephalometric-ConvNeXtV2 \
  checkpoints/CephAdoAdu/best_model.pt \
  checkpoints/Aariz_26/best_model.pt \
  --repo-type dataset --local-dir .

# Option 2 — Python
python - <<'PY'
from huggingface_hub import hf_hub_download
for sub in ("CephAdoAdu", "Aariz_26"):
    hf_hub_download(
        repo_id="emad2001/DeLR-Cephalometric-ConvNeXtV2",
        filename=f"checkpoints/{sub}/best_model.pt",
        repo_type="dataset",
        local_dir=".",
    )
PY
```

After download, the checkpoints land in `checkpoints/CephAdoAdu/best_model.pt` and `checkpoints/Aariz_26/best_model.pt` — exactly where `infer.py` expects them by default.

## Repository layout

```
.
├── train.py                   # training CLI (OneCycle / cosine / constant LR)
├── infer.py                   # test-set evaluation + JSON predictions
├── delr/
│   ├── __init__.py
│   ├── model.py               # DeLR / D-CeLR architecture (ConvNeXtV2 backbone)
│   ├── datasets.py            # Aariz + CephAdoAdu loaders
│   └── metrics.py             # MRE (mm + px) and SDR
├── scripts/
│   └── preresize_aariz.py     # one-off pre-resize for fast Aariz training
├── checkpoints/               # logs + predictions; .pt downloaded from HF
│   ├── CephAdoAdu/
│   │   ├── train.log          # full training history (3 phases, 243 epochs)
│   │   └── test_predictions.json
│   └── Aariz_26/
│       ├── train.log          # 200 epochs
│       └── test_predictions.json
├── requirements.txt
└── README.md
```

## Results

All numbers are on the **held-out test split**. Training: ConvNeXtV2-tiny backbone, input 1024×1024, batch 2 (T4 16 GB), augmentations on, AdamW + grad-clip 1.0.

### CephAdoAdu (10 landmarks, 300 test images)

| Metric | Value |
|---|---|
| MRE | **1.045 mm** (10.45 px) |
| SDR @ 2.0 mm | 87.53 % |
| SDR @ 2.5 mm | 92.37 % |
| SDR @ 3.0 mm | 95.27 % |
| SDR @ 4.0 mm | 97.63 % |

Validation MRE (40 imgs) was 0.998 mm — generalisation gap < 0.05 mm.
Pixel size per image is not provided by the dataset; we used a uniform 0.1 mm/px convention (configurable via `--pixel-size-mm`).

### Aariz (26 landmarks, 150 test images)

| Metric | Value |
|---|---|
| MRE | **1.073 mm** (validation; 200-epoch reference run) |
| SDR @ 2.0 mm | 87.0 % |
| SDR @ 2.5 mm | 92.1 % |
| SDR @ 3.0 mm | 94.8 % |
| SDR @ 4.0 mm | 97.2 % |

Pixel size is per-image from `cephalogram_machine_mappings.csv`.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

PyTorch ≥ 2.1 with CUDA is recommended.

## Datasets

### Aariz (29 landmarks, expanded layout)
```
<aariz_root>/
  train/
    Cephalograms/<id>.{png,jpg}
    Annotations/Cephalometric Landmarks/Junior Orthodontists/<id>.json
    Annotations/Cephalometric Landmarks/Senior Orthodontists/<id>.json
  valid/ ...
  test/  ...
  cephalogram_machine_mappings.csv     # per-image pixel-size (mm/px)
```
The loader averages junior + senior annotations. `--landmarks` selects the subset: `19` (standard eval), `26` (drops three soft-tissue points), `all` (29).

### CephAdoAdu
```
<cephadoadu_root>/
  final_splits.json                    # train/test image IDs + group
  adult/dataset/<id>.jpg
  adult/txt/<id>.txt                   # JSON list of 10 landmarks
  under_age/dataset/<id>.jpg
  under_age/txt/<id>.txt
```
The loader carves a deterministic 10 % validation split from `final_splits.json`'s train list (seed 42).

## Speeding up Aariz (recommended)

Source images are huge (some up to 95 MP). The dataset loader spends most of its time on PIL bicubic resizing during training, which makes the GPU sit idle ~50 % of the time. The fix is to pre-resize once:

```bash
python scripts/preresize_aariz.py \
  --src   /path/to/Aariz \
  --dst   /path/to/Aariz_resized_1024 \
  --size  1024 \
  --workers 4
```

This writes 1024×1024 PNGs (~430 MB total) plus an `original_sizes.json` sidecar, and symlinks the annotations + CSV. Subsequent training: pass `--preresized --dataset-root /path/to/Aariz_resized_1024`. Per-epoch time on a T4 dropped from ~730 s → ~370 s.

## Training

### CephAdoAdu — full reproduction recipe
The published checkpoint comes from a **three-phase** schedule (see `checkpoints/CephAdoAdu/train.log` for the merged log).

```bash
# Phase 1: 200-epoch OneCycle (max_lr=2e-4) — diverged at ~epoch 37, best at epoch 36 (MRE 7.75 mm)
python train.py \
  --dataset cephadoadu \
  --dataset-root "/path/to/CephAdoAdu Dataset" \
  --backbone convnextv2_tiny \
  --image-size 1024 --batch-size 2 \
  --epochs 200 --lr 2e-4 \
  --landmarks all --num-workers 2 \
  --output-dir outputs/cephadoadu_phase1

# Phase 2: resume from phase-1 best, constant lr=5e-5 for 200 epochs (best at epoch 72, MRE 1.145 mm)
python train.py \
  --dataset cephadoadu \
  --dataset-root "/path/to/CephAdoAdu Dataset" \
  --backbone convnextv2_tiny \
  --image-size 1024 --batch-size 2 \
  --epochs 200 --lr 5e-5 --scheduler constant \
  --resume outputs/cephadoadu_phase1/best_model.pt \
  --output-dir outputs/cephadoadu_phase2

# Phase 3: resume from phase-2 best, cosine 5e-5 → 1e-6 for 128 epochs (best at epoch 93, MRE 0.998 mm)
python train.py \
  --dataset cephadoadu \
  --dataset-root "/path/to/CephAdoAdu Dataset" \
  --backbone convnextv2_tiny \
  --image-size 1024 --batch-size 2 \
  --epochs 128 --lr 5e-5 --lr-min 1e-6 --scheduler cosine \
  --resume outputs/cephadoadu_phase2/best_model.pt \
  --output-dir outputs/cephadoadu_phase3
```

**Why three phases?** OneCycle's mid-training peak LR (≈2e-4) destabilised the auxiliary heatmap head when the small effective batch (2) couldn't smooth gradients. Switching to a smaller constant LR + later cosine annealing recovered training and pushed best validation MRE from 7.75 mm to 0.998 mm. The full loss / metric trajectory is in the merged `train.log`.

### Aariz (26 landmarks)
```bash
python train.py \
  --dataset aariz \
  --dataset-root /path/to/Aariz_resized_1024 --preresized \
  --backbone convnextv2_tiny \
  --image-size 1024 --batch-size 2 \
  --epochs 200 --lr 2e-4 \
  --landmarks 26 --num-workers 4 \
  --output-dir outputs/aariz_26
```

## Inference / test-set evaluation

```bash
# CephAdoAdu test split
python infer.py \
  --dataset cephadoadu \
  --dataset-root "/path/to/CephAdoAdu Dataset" \
  --split test \
  --backbone convnextv2_tiny --image-size 1024 --batch-size 2 \
  --checkpoint checkpoints/CephAdoAdu/best_model.pt \
  --output checkpoints/CephAdoAdu/test_predictions.json

# Aariz test split
python infer.py \
  --dataset aariz \
  --dataset-root /path/to/Aariz \
  --split test \
  --landmarks 26 \
  --backbone convnextv2_tiny --image-size 1024 --batch-size 2 \
  --checkpoint checkpoints/Aariz_26/best_model.pt \
  --output checkpoints/Aariz_26/test_predictions.json
```

Predictions are saved as `{image_id: [[x_orig, y_orig], ...]}` in **original-image pixel space** (already de-scaled from the network's 1024×1024 frame). MRE and SDR are printed.

## Citation

If this work helps your research, please cite the original DeLR / D-CeLR paper and the dataset releases (Aariz Cephalograms; CephAdoAdu).

## License

See repository licence file (or add one before publishing).
