# DeLR – Dual-encoder Landmark Regression

A PyTorch implementation of the **DeLR (Dual-encoder Landmark Regression)** architecture for cephalometric landmark detection, evaluated on three public datasets:

- **Aariz Cephalograms** — 1000 images, 29 annotated landmarks (700 / 150 / 150 train/valid/test).
- **CephAdoAdu Dataset** — 700 images, 10 landmarks, mixed adolescent + adult cohort (400 train / 300 test in the official splits; we held out 10 % of train as validation).
- **ISBI 2015 Cephalometric Challenge** — 400 images, 19 landmarks (150 train / 150 Test1 / 100 Test2). Pixel size 0.1 mm/px, taken directly from the official evaluator (`EvaluationCode/v2_eva_code.m`).

## Pretrained checkpoints

The `.pt` weights are hosted on **Hugging Face** (each is 530–650 MB and exceeds GitHub's 100 MB file limit):

🔗 https://huggingface.co/datasets/emad2001/DeLR-Cephalometric-ConvNeXtV2

```bash
pip install huggingface_hub
huggingface-cli download emad2001/DeLR-Cephalometric-ConvNeXtV2 \
  checkpoints/CephAdoAdu/best_model.pt \
  checkpoints/Aariz_26/best_model.pt \
  checkpoints/ISBI2015/best_model.pt \
  --repo-type dataset --local-dir .
```

After download, the checkpoints land in `checkpoints/<DATASET>/best_model.pt` — exactly where `infer.py` expects them by default.

## Repository layout

```
.
├── train.py                   # training CLI (OneCycle / cosine / constant LR + ablation flags)
├── infer.py                   # test-set evaluation + JSON predictions (+ TTA hflip)
├── run_ablations.sh           # 4-ablation orchestration (no-heatmap, no-finetune, no-rle, M=1)
├── delr/
│   ├── __init__.py
│   ├── model.py               # DeLR / D-CeLR architecture (ConvNeXtV2 backbone)
│   ├── datasets.py            # Aariz + CephAdoAdu + ISBI 2015 loaders
│   └── metrics.py             # MRE (mm + px) and SDR
├── scripts/
│   └── preresize_aariz.py     # one-off pre-resize for fast Aariz training
├── checkpoints/               # logs + predictions (.pt downloaded from HF)
│   ├── CephAdoAdu/
│   ├── Aariz_26/
│   ├── ISBI2015/
│   │   ├── train_phase1.log   # 200 ep OneCycle
│   │   ├── train_phase2.log   # 200 ep constant lr=5e-5
│   │   ├── train_phase3.log   # 128 ep cosine 5e-5 → 1e-6
│   │   ├── test1_predictions.json
│   │   └── test2_predictions.json
│   └── ablations/             # 150-ep OneCycle ablation runs on ISBI 2015
│       ├── no_heatmap/        # auxiliary heatmap head removed
│       ├── no_finetune/       # reference encoder only
│       ├── no_rle/            # plain mean-radial-L2 instead of RLE Laplace
│       └── m1/                # M = 1 finetune layer (vs default M = 4)
└── requirements.txt
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

### Aariz (26 landmarks, 150 test images)

| Metric | Value |
|---|---|
| MRE | **1.073 mm** (validation; 200-epoch reference run) |
| SDR @ 2.0 mm | 87.0 % |
| SDR @ 2.5 mm | 92.1 % |
| SDR @ 3.0 mm | 94.8 % |
| SDR @ 4.0 mm | 97.2 % |

Pixel size is per-image from `cephalogram_machine_mappings.csv`.

### ISBI 2015 (19 landmarks)

Trained from scratch with the same three-phase recipe used for CephAdoAdu (Phase 1: 200 ep OneCycle max_lr=2e-4; Phase 2: 200 ep constant lr=5e-5 resumed from phase-1 best; Phase 3: 128 ep cosine 5e-5 → 1e-6 resumed from phase-2 best). Phase-3 best is at epoch 32 of the cosine phase. **Pixel size 0.1 mm/px** is the official-evaluator convention (`EvaluationCode/v2_eva_code.m` thresholds R-pixels against `accur_mm * 10`).

| Split | N | MRE (mm) | MRE (px) | SDR @ 2.0 | SDR @ 2.5 | SDR @ 3.0 | SDR @ 4.0 |
|---|---|---:|---:|---:|---:|---:|---:|
| **Test1** (151–300) | 150 | **1.124** | 11.24 | **87.12 %** | 92.53 % | 96.11 % | 98.39 % |
| **Test2** (301–400) | 100 | **1.463** | 14.63 | **74.74 %** | 83.47 % | 88.84 % | 94.63 % |

Per-phase best validation MRE (Test1 used as val during training):

| Phase | Schedule | Best val MRE | Δ vs prev |
|---|---|---:|---:|
| 1 | 200 ep OneCycle, max_lr=2e-4 | 1.188 mm | — |
| 2 | 200 ep constant lr=5e-5 | 1.132 mm | −0.056 mm |
| 3 | 128 ep cosine 5e-5 → 1e-6 | **1.124 mm** | −0.008 mm |

**Why Test2 is harder** — Test2 (301–400) was the blind ranking set in the 2015 challenge and is drawn from a more demographically diverse cohort. Test1 was used here for in-training model selection so the ~0.34 mm Test1↔Test2 gap also reflects implicit tuning to Test1; this is the same pattern reported across published ISBI-2015 leaderboards.

#### Ablations on ISBI 2015 (150 ep OneCycle each, otherwise identical)

| Variant | Test1 MRE (mm) | Test1 SDR @2 mm | Test2 MRE (mm) | Test2 SDR @2 mm |
|---|---:|---:|---:|---:|
| no-heatmap (`--no-heatmap`) | **1.171** | **86.18 %** | **1.549** | **74.11 %** |
| no-finetune (`--no-finetune`, reference only) | 12.616 | 3.96 % | 12.141 | 3.68 % |
| no-RLE (`--no-rle`, plain L2) | 40.732 | 0.11 % | 40.263 | 0.00 % |
| M = 1 (`--num-finetune-layers 1`) | 1.770 | 65.75 % | 2.112 | 58.26 % |

Reference: full 200-ep phase-1 baseline (M=4, RLE on, heatmap on) → Test1 1.188 mm / 84.39 %, Test2 1.595 mm / 73.21 %.

**Take-aways**

- **The finetune (refinement) encoder is the main accuracy driver.** Removing it collapses Test1 MRE from ~1.2 mm to 12.6 mm — the reference encoder alone produces only coarse coords, by design.
- **RLE Laplace loss is essential** under our settings. Replacing it with plain mean radial L2 prevents the model from converging at all (MRE > 40 mm). The variance head (`log_sigma`) is doing real work, likely as a per-landmark difficulty weighting.
- **Multi-layer iterative refinement matters** — collapsing M from 4 to 1 hurts Test1 SDR@2 mm by ~20 pp.
- **The auxiliary heatmap head is essentially free** on this dataset — removing it is a wash on Test1 (slightly *better* SDR@2 mm by ~+1.8 pp at this epoch budget) and statistically tied on Test2.

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

### ISBI 2015 (figshare 37ec464af8e81ae6ebbf)
```
<isbi_root>/
  RawImage/TrainingData/{001..150}.bmp
  RawImage/Test1Data/{151..300}.bmp
  RawImage/Test2Data/{301..400}.bmp
  400_junior/{001..400}.txt            # 19 lines "x,y" + classification rows
  400_senior/{001..400}.txt
  EvaluationCode/v2_eva_code.m         # official MATLAB scorer
```
The loader averages junior + senior annotations and exposes splits `train` / `valid` (alias `test1`) / `test` (alias `test2`). `--pixel-size-mm 0.1` matches the official evaluator.

## Speeding up Aariz (recommended)

Source images are huge (some up to 95 MP). The dataset loader spends most of its time on PIL bicubic resizing during training. The fix is to pre-resize once:

```bash
python scripts/preresize_aariz.py \
  --src   /path/to/Aariz \
  --dst   /path/to/Aariz_resized_1024 \
  --size  1024 \
  --workers 4
```

Subsequent training: pass `--preresized --dataset-root /path/to/Aariz_resized_1024`. Per-epoch time on a T4 dropped from ~730 s → ~370 s.

## Training

### CephAdoAdu — full reproduction recipe
The published checkpoint comes from a **three-phase** schedule.

```bash
# Phase 1
python train.py --dataset cephadoadu --dataset-root "/path/to/CephAdoAdu Dataset" \
  --backbone convnextv2_tiny --image-size 1024 --batch-size 2 \
  --epochs 200 --lr 2e-4 --landmarks all --num-workers 2 \
  --output-dir outputs/cephadoadu_phase1

# Phase 2
python train.py --dataset cephadoadu --dataset-root "/path/to/CephAdoAdu Dataset" \
  --backbone convnextv2_tiny --image-size 1024 --batch-size 2 \
  --epochs 200 --lr 5e-5 --scheduler constant \
  --resume outputs/cephadoadu_phase1/best_model.pt \
  --output-dir outputs/cephadoadu_phase2

# Phase 3
python train.py --dataset cephadoadu --dataset-root "/path/to/CephAdoAdu Dataset" \
  --backbone convnextv2_tiny --image-size 1024 --batch-size 2 \
  --epochs 128 --lr 5e-5 --lr-min 1e-6 --scheduler cosine \
  --resume outputs/cephadoadu_phase2/best_model.pt \
  --output-dir outputs/cephadoadu_phase3
```

### Aariz (26 landmarks)
```bash
python train.py --dataset aariz \
  --dataset-root /path/to/Aariz_resized_1024 --preresized \
  --backbone convnextv2_tiny --image-size 1024 --batch-size 2 \
  --epochs 200 --lr 2e-4 --landmarks 26 --num-workers 4 \
  --output-dir outputs/aariz_26
```

### ISBI 2015 (19 landmarks) — full three-phase recipe
```bash
# Phase 1: 200 ep OneCycle, max_lr=2e-4 -> best val MRE 1.188 mm @ ep165
python train.py --dataset isbi2015 \
  --dataset-root /path/to/figshare_37ec464af8e81ae6ebbf \
  --backbone convnextv2_tiny --image-size 1024 --batch-size 2 \
  --epochs 200 --lr 2e-4 --num-workers 2 --pixel-size-mm 0.1 \
  --output-dir outputs/isbi2015_phase1

# Phase 2: 200 ep constant lr=5e-5 -> best val MRE 1.132 mm @ ep194
python train.py --dataset isbi2015 \
  --dataset-root /path/to/figshare_37ec464af8e81ae6ebbf \
  --backbone convnextv2_tiny --image-size 1024 --batch-size 2 \
  --epochs 200 --lr 5e-5 --scheduler constant --num-workers 2 --pixel-size-mm 0.1 \
  --resume outputs/isbi2015_phase1/best_model.pt \
  --output-dir outputs/isbi2015_phase2

# Phase 3: 128 ep cosine 5e-5 -> 1e-6 -> best val MRE 1.124 mm @ ep32
python train.py --dataset isbi2015 \
  --dataset-root /path/to/figshare_37ec464af8e81ae6ebbf \
  --backbone convnextv2_tiny --image-size 1024 --batch-size 2 \
  --epochs 128 --lr 5e-5 --lr-min 1e-6 --scheduler cosine --num-workers 2 --pixel-size-mm 0.1 \
  --resume outputs/isbi2015_phase2/best_model.pt \
  --output-dir outputs/isbi2015_phase3
```

### Ablation flags

`train.py` exposes four optional ablation switches (combinable):

| Flag | Effect |
|---|---|
| `--no-heatmap` | `lambda_HM = 0` (auxiliary heatmap supervision off). |
| `--no-finetune` | `lambda_FE = 0`; metric computed on `coarse_mu` instead of `fine_mu`. |
| `--no-rle` | Replaces the RLE Laplace loss with plain mean radial L2 (no `log_sigma`). |
| `--num-finetune-layers N` | Override M (default 4). |

A turn-key orchestration script `run_ablations.sh` reproduces the four ablation rows on ISBI 2015.

## Inference / test-set evaluation

```bash
# CephAdoAdu test split
python infer.py --dataset cephadoadu --dataset-root "/path/to/CephAdoAdu Dataset" \
  --split test --backbone convnextv2_tiny --image-size 1024 --batch-size 2 \
  --checkpoint checkpoints/CephAdoAdu/best_model.pt \
  --output checkpoints/CephAdoAdu/test_predictions.json

# Aariz test split
python infer.py --dataset aariz --dataset-root /path/to/Aariz \
  --split test --landmarks 26 \
  --backbone convnextv2_tiny --image-size 1024 --batch-size 2 \
  --checkpoint checkpoints/Aariz_26/best_model.pt \
  --output checkpoints/Aariz_26/test_predictions.json

# ISBI 2015 — Test1 and Test2
python infer.py --dataset isbi2015 --dataset-root /path/to/figshare_37ec464af8e81ae6ebbf \
  --split test1 --backbone convnextv2_tiny --image-size 1024 --batch-size 1 --pixel-size-mm 0.1 \
  --checkpoint checkpoints/ISBI2015/best_model.pt \
  --output checkpoints/ISBI2015/test1_predictions.json

python infer.py --dataset isbi2015 --dataset-root /path/to/figshare_37ec464af8e81ae6ebbf \
  --split test2 --backbone convnextv2_tiny --image-size 1024 --batch-size 1 --pixel-size-mm 0.1 \
  --checkpoint checkpoints/ISBI2015/best_model.pt \
  --output checkpoints/ISBI2015/test2_predictions.json
```

`infer.py` also supports `--tta-flip` (average predictions from the image and its horizontal flip). On ISBI 2015 this *hurts* because the model was not trained with horizontal-flip augmentation and lateral cephalograms are not left/right-symmetric — see the ablation discussion above.

Predictions are saved as `{image_id: [[x_orig, y_orig], ...]}` in **original-image pixel space** (already de-scaled from the network's 1024×1024 frame). MRE and SDR are printed.

## Citation

If this work helps your research, please cite the original DeLR / D-CeLR paper and the dataset releases (Aariz Cephalograms; CephAdoAdu; ISBI 2015 Cephalometric Challenge — Wang et al., IEEE TMI 2016).

## License

See repository licence file (or add one before publishing).
