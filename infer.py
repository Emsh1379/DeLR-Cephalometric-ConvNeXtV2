import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import torch

from delr import build_dcelr
from delr.datasets import AarizCephalometricDataset, CephAdoAduDataset, ISBI2015Dataset
from delr.metrics import compute_mre_and_sdr, DEFAULT_THRESHOLDS_MM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference script for DeLR (Dual-encoder Landmark Regression).")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/teamspace/studios/this_studio/Aariz/Aariz"),
        help="Root directory of the Aariz dataset.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "valid", "test", "test1", "test2"],
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="aariz",
        choices=["aariz", "cephadoadu", "isbi2015"],
    )
    parser.add_argument("--landmarks", type=str, default="26", choices=["19", "26", "all"])
    parser.add_argument(
        "--pixel-size-mm",
        type=float,
        default=0.1,
        help="Assumed pixel size in mm for cephadoadu (no per-image mapping available).",
    )
    parser.add_argument(
        "--preresized",
        action="store_true",
        help="Aariz dataset is pre-resized to image_size with original_sizes.json sidecar.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=1024,
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to trained checkpoint (.pt).")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("./predictions.json"))
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples for quick smoke tests.")
    parser.add_argument(
        "--backbone",
        type=str,
        default="convnextv2_base",
        help="Backbone to instantiate the model with (e.g., convnextv2_base, convnextv2_tiny, resnet34).",
    )
    parser.add_argument(
        "--skip-metrics",
        action="store_true",
        help="Skip metric computation even if ground-truth annotations are available.",
    )
    parser.add_argument(
        "--tta-flip",
        action="store_true",
        help="Test-time augmentation: average predictions from the image and its horizontal flip.",
    )
    parser.add_argument(
        "--no-finetune",
        action="store_true",
        help="Use reference-encoder predictions (coarse_mu) instead of fine_mu.",
    )
    parser.add_argument(
        "--num-finetune-layers",
        type=int,
        default=4,
        help="M: number of finetune-encoder layers used to build the model (must match the checkpoint).",
    )
    parser.add_argument(
        "--fpn-refine",
        action="store_true",
        help="Enable high-resolution FPN sampling plus a local offset refinement head.",
    )
    parser.add_argument(
        "--fpn-dim",
        type=int,
        default=256,
        help="Channel width for the optional FPN local refinement branch.",
    )
    parser.add_argument(
        "--fpn-refine-level",
        type=str,
        default="p2",
        choices=["p2", "p3", "p4", "multi"],
        help="FPN level sampled by the local offset head. Use 'multi' for p2+p3+p4.",
    )
    parser.add_argument("--patch-refine", action="store_true")
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--patch-radius", type=float, default=32.0)
    parser.add_argument("--hr-heatmap-refine", action="store_true")
    return parser.parse_args()


def append_predictions(
    predictions: Dict[str, list],
    batch_coords: torch.Tensor,
    metas: Dict[str, torch.Tensor],
) -> None:
    for idx, coords_pred in enumerate(batch_coords):
        sx = metas["scale_x"][idx]
        sy = metas["scale_y"][idx]
        if isinstance(sx, torch.Tensor):
            sx = float(sx.item())
        if isinstance(sy, torch.Tensor):
            sy = float(sy.item())

        coords_orig = coords_pred.clone()
        coords_orig[:, 0] /= sx
        coords_orig[:, 1] /= sy
        image_id = metas["image_id"][idx]
        if isinstance(image_id, torch.Tensor):
            image_id = image_id.item()
        predictions[str(image_id)] = coords_orig.tolist()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    if args.dataset == "cephadoadu":
        dataset = CephAdoAduDataset(
            dataset_root=args.dataset_root,
            split=args.split,
            image_size=args.image_size,
            return_heatmap=False,
            default_pixel_size_mm=args.pixel_size_mm,
        )
    elif args.dataset == "isbi2015":
        dataset = ISBI2015Dataset(
            dataset_root=args.dataset_root,
            split=args.split,
            image_size=args.image_size,
            return_heatmap=False,
            default_pixel_size_mm=args.pixel_size_mm,
        )
    else:
        dataset = AarizCephalometricDataset(
            dataset_root=args.dataset_root,
            split=args.split,
            image_size=args.image_size,
            return_heatmap=False,
            landmark_mode=args.landmarks,
            preresized=args.preresized,
        )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_config = checkpoint.get("model_config", {}) if isinstance(checkpoint, dict) else {}
    use_fpn_refine = args.fpn_refine or bool(model_config.get("use_fpn_refine", False))
    fpn_dim = int(model_config.get("fpn_dim", args.fpn_dim))
    fpn_refine_level = str(model_config.get("fpn_refine_level", args.fpn_refine_level))
    use_patch_refine = args.patch_refine or bool(model_config.get("use_patch_refine", False))
    patch_size = int(model_config.get("patch_size", args.patch_size))
    patch_radius = float(model_config.get("patch_radius", args.patch_radius))
    use_hr_heatmap_refine = args.hr_heatmap_refine or bool(model_config.get("use_hr_heatmap_refine", False))
    model = build_dcelr(
        num_landmarks=dataset.num_landmarks,
        in_channels=1,
        backbone=model_config.get("backbone", args.backbone),
        num_layers_finetune=int(model_config.get("num_layers_finetune", max(1, args.num_finetune_layers))),
        use_fpn_refine=use_fpn_refine,
        fpn_dim=fpn_dim,
        fpn_refine_level=fpn_refine_level,
        use_patch_refine=use_patch_refine,
        patch_size=patch_size,
        patch_radius=patch_radius,
        use_hr_heatmap_refine=use_hr_heatmap_refine,
    ).to(device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    predictions: Dict[str, list] = {}
    sum_mre_mm = 0.0
    sum_mre_px = 0.0
    total_points = 0.0
    threshold_keys = [str(thr).replace(".", "_") for thr in DEFAULT_THRESHOLDS_MM]
    sdr_hits = {key: 0.0 for key in threshold_keys}

    processed = 0
    for batch in loader:
        if len(batch) == 3:
            images, coords, metas = batch
        else:
            raise ValueError("Unexpected batch format from dataset.")
        images = images.to(device, non_blocking=True)

        pred_key = "coarse_mu" if args.no_finetune else "fine_mu"
        with torch.no_grad():
            out = model(images)
            preds = out[pred_key]
            if args.tta_flip:
                images_flip = torch.flip(images, dims=[-1])
                out_flip = model(images_flip)
                preds_flip = out_flip[pred_key].clone()
                preds_flip[..., 0] = (args.image_size - 1) - preds_flip[..., 0]
                preds = (preds + preds_flip) * 0.5
        preds = preds.cpu()
        append_predictions(predictions, preds, metas)

        if not args.skip_metrics:
            metrics = compute_mre_and_sdr(preds, coords, metas, DEFAULT_THRESHOLDS_MM)
            sum_mre_mm += metrics["sum_mre_mm"]
            sum_mre_px += metrics["sum_mre_px"]
            total_points += metrics["num_points"]
            for key in threshold_keys:
                sdr_hits[key] += metrics[f"sdr_hits_{key}mm"]

        processed += len(images)
        if args.max_samples is not None and processed >= args.max_samples:
            break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(predictions, f, indent=2)

    print(f"Saved predictions to {args.output}")

    if not args.skip_metrics and total_points > 0:
        mre_mm = sum_mre_mm / total_points
        mre_px = sum_mre_px / total_points
        print(f"MRE(mm): {mre_mm:.3f} | MRE(px): {mre_px:.3f}")
        sdr_line = " / ".join(
            f"{(sdr_hits[key] / total_points) * 100.0:.2f}%@{key.replace('_', '.')}mm" for key in threshold_keys
        )
        print(f"SDR: {sdr_line}")


if __name__ == "__main__":
    main()
