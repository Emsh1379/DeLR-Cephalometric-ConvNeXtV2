import argparse
import time
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from delr import build_dcelr
from delr.datasets import (
    AarizCephalometricDataset,
    create_aariz_dataloaders,
    create_cephadoadu_dataloaders,
)
from delr.metrics import compute_mre_and_sdr, DEFAULT_THRESHOLDS_MM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DeLR (Dual-encoder Landmark Regression).")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/teamspace/studios/this_studio/Aariz/Aariz"),
        help="Root path containing train/valid/test folders.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="aariz",
        choices=["aariz", "cephadoadu"],
        help="Which dataset loader to use.",
    )
    parser.add_argument(
        "--pixel-size-mm",
        type=float,
        default=0.1,
        help="Assumed pixel size in mm for cephadoadu (no per-image mapping available).",
    )
    parser.add_argument(
        "--preresized",
        action="store_true",
        help="Aariz dataset is already pre-resized to image_size with original_sizes.json sidecar.",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--heatmap-sigma", type=float, default=1.8)
    parser.add_argument(
        "--landmarks",
        type=str,
        default="26",
        choices=["19", "26", "all"],
        help="Subset of landmarks to keep (19, 26, or all annotations).",
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable train-time geometric/color/noise augmentations.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize images to mean=0.5, std=0.5 after ToTensor.",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="convnextv2_base",
        help="Backbone architecture (e.g., convnextv2_base, convnextv2_large, resnet34).",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs"))
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to a checkpoint whose state_dict should be loaded before training.",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="onecycle",
        choices=["onecycle", "constant", "cosine"],
        help="LR schedule. 'cosine' anneals from --lr down to --lr-min over all steps.",
    )
    parser.add_argument(
        "--lr-min",
        type=float,
        default=1e-6,
        help="Minimum LR for cosine annealing (end-of-schedule lr).",
    )
    parser.add_argument(
        "--max-train-steps",
        type=int,
        default=None,
        help="Optional limit of batches per epoch (useful for debugging).",
    )
    parser.add_argument(
        "--max-val-steps",
        type=int,
        default=None,
        help="Optional limit of validation batches per epoch (useful for debugging).",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: OneCycleLR,
    device: torch.device,
    max_steps: Optional[int] = None,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_heatmap = 0.0
    total_re = 0.0
    total_fe = 0.0
    total_samples = 0

    for step_idx, (images, coords, heatmaps, _) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        coords = coords.to(device, non_blocking=True)
        heatmaps = heatmaps.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        out = model(images, gt_coords=coords, gt_heatmap=heatmaps)
        loss = out["loss_total"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        batch_size = images.size(0)
        total_samples += batch_size
        total_loss += loss.item() * batch_size
        total_heatmap += out["loss_hm"].item() * batch_size
        total_re += out["loss_re"].item() * batch_size
        total_fe += out["loss_fe"].item() * batch_size

        if max_steps is not None and step_idx >= max_steps:
            break

    denom = max(total_samples, 1)
    return {
        "loss": total_loss / denom,
        "loss_hm": total_heatmap / denom,
        "loss_re": total_re / denom,
        "loss_fe": total_fe / denom,
    }


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_steps: Optional[int] = None,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_heatmap = 0.0
    total_re = 0.0
    total_fe = 0.0
    total_samples = 0
    sum_mre_mm = 0.0
    sum_mre_px = 0.0
    total_points = 0.0
    threshold_keys = [str(thr).replace(".", "_") for thr in DEFAULT_THRESHOLDS_MM]
    sdr_hit_totals = {key: 0.0 for key in threshold_keys}

    for step_idx, (images, coords, heatmaps, metas) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        coords = coords.to(device, non_blocking=True)
        heatmaps = heatmaps.to(device, non_blocking=True)

        out = model(images, gt_coords=coords, gt_heatmap=heatmaps)
        batch_size = images.size(0)
        total_samples += batch_size
        total_loss += out["loss_total"].item() * batch_size
        total_heatmap += out["loss_hm"].item() * batch_size
        total_re += out["loss_re"].item() * batch_size
        total_fe += out["loss_fe"].item() * batch_size

        pred_metrics = compute_mre_and_sdr(out["fine_mu"], coords, metas, DEFAULT_THRESHOLDS_MM)
        sum_mre_mm += pred_metrics["sum_mre_mm"]
        sum_mre_px += pred_metrics["sum_mre_px"]
        total_points += pred_metrics["num_points"]
        for key in threshold_keys:
            sdr_hit_totals[key] += pred_metrics[f"sdr_hits_{key}mm"]

        if max_steps is not None and step_idx >= max_steps:
            break

    if total_samples == 0 or total_points == 0:
        return {
            "loss": float("nan"),
            "loss_hm": float("nan"),
            "loss_re": float("nan"),
            "loss_fe": float("nan"),
            "mre_mm": float("nan"),
            "mre_px": float("nan"),
            **{f"sdr_{key}mm": float("nan") for key in threshold_keys},
        }

    denom = total_samples
    metrics = {
        "loss": total_loss / denom,
        "loss_hm": total_heatmap / denom,
        "loss_re": total_re / denom,
        "loss_fe": total_fe / denom,
        "mre_mm": sum_mre_mm / total_points,
        "mre_px": sum_mre_px / total_points,
    }
    for key in threshold_keys:
        metrics[f"sdr_{key}mm"] = (sdr_hit_totals[key] / total_points) * 100.0
    return metrics


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    if args.dataset == "cephadoadu":
        train_loader, val_loader = create_cephadoadu_dataloaders(
            dataset_root=args.dataset_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            image_size=args.image_size,
            heatmap_sigma=args.heatmap_sigma,
            augment_train=not args.no_augment,
            normalize=args.normalize,
            default_pixel_size_mm=args.pixel_size_mm,
        )
    else:
        train_loader, val_loader = create_aariz_dataloaders(
            dataset_root=args.dataset_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            image_size=args.image_size,
            heatmap_sigma=args.heatmap_sigma,
            landmark_mode=args.landmarks,
            augment_train=not args.no_augment,
            normalize=args.normalize,
            preresized=args.preresized,
        )

    model = build_dcelr(
        num_landmarks=train_loader.dataset.num_landmarks,
        in_channels=1,
        backbone=args.backbone,
    ).to(device)

    resume_checkpoint = None
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        resume_checkpoint = ckpt
        state_dict = ckpt.get("state_dict", ckpt)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Resumed weights from {args.resume} (missing={len(missing)}, unexpected={len(unexpected)}).")
        if "val_metrics" in ckpt:
            print(f"Checkpoint val_metrics: {ckpt['val_metrics']}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = args.max_train_steps or len(train_loader)
    if args.scheduler == "onecycle":
        scheduler = OneCycleLR(
            optimizer,
            max_lr=args.lr,
            epochs=args.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.3,
            anneal_strategy="cos",
        )
    elif args.scheduler == "cosine":
        total_steps = args.epochs * steps_per_epoch
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=args.lr_min,
        )
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _step: 1.0)

    best_mre = float("inf")
    if resume_checkpoint is not None and "val_metrics" in resume_checkpoint:
        best_mre = resume_checkpoint["val_metrics"].get("mre_mm", best_mre)
        torch.save(resume_checkpoint, args.output_dir / "best_model.pt")
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            max_steps=args.max_train_steps,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            max_steps=args.max_val_steps,
        )
        elapsed = time.time() - start_time

        log_line = (
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Train loss: {train_metrics['loss']:.4f} "
            f"(hm={train_metrics['loss_hm']:.4f}, re={train_metrics['loss_re']:.4f}, fe={train_metrics['loss_fe']:.4f}) | "
            f"Val loss: {val_metrics['loss']:.4f} "
            f"(hm={val_metrics['loss_hm']:.4f}, re={val_metrics['loss_re']:.4f}, fe={val_metrics['loss_fe']:.4f}) | "
            f"Val MRE(mm): {val_metrics['mre_mm']:.3f} | "
            f"SDR%(2/2.5/3/4): "
            f"{val_metrics['sdr_2_0mm']:.1f}/"
            f"{val_metrics['sdr_2_5mm']:.1f}/"
            f"{val_metrics['sdr_3_0mm']:.1f}/"
            f"{val_metrics['sdr_4_0mm']:.1f} | "
            f"{elapsed:.1f}s"
        )
        print(log_line)

        if val_metrics["mre_mm"] < best_mre:
            best_mre = val_metrics["mre_mm"]
            best_state = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_metrics": val_metrics,
            }
            torch.save(best_state, args.output_dir / "best_model.pt")


if __name__ == "__main__":
    main()
