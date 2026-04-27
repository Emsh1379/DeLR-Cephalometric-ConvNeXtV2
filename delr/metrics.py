"""
Metric helpers for Dual-encoder Landmark Regression.

The functions here operate on batched predictions/ground-truth coordinates in
resized image space and rely on the metadata dictionary returned by the
`AarizCephalometricDataset`.
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import torch

DEFAULT_THRESHOLDS_MM: Tuple[float, ...] = (2.0, 2.5, 3.0, 4.0)


def _as_float(value) -> float:
    """Convert scalars/tensors/lists from dataloader collate into floats."""
    if isinstance(value, torch.Tensor):
        return float(value.item())
    if isinstance(value, (list, tuple)):
        # handle case where collate turns floats into length-1 tensors lists
        return float(value[0])
    return float(value)


def compute_mre_and_sdr(
    preds: torch.Tensor,
    targets: torch.Tensor,
    metas: Dict[str, torch.Tensor],
    thresholds_mm: Iterable[float] = DEFAULT_THRESHOLDS_MM,
) -> Dict[str, float]:
    """
    Compute Mean Radial Error in millimetres and Success Detection Rates.

    Args:
        preds:   [B, K, 2] predicted coordinates in resized image space.
        targets:[B, K, 2] ground truth coordinates in resized image space.
        metas:   dictionary returned by `AarizCephalometricDataset` containing
                 `scale_x`, `scale_y`, and `pixel_size_mm` entries produced by the
                 dataloader's default collate function.
        thresholds_mm: iterable of thresholds (in millimetres) for SDR.

    Returns:
        dict containing:
            - mre_mm: mean radial error in millimetres.
            - mre_px: mean radial error in original pixel space.
            - sdr_{thr}mm: success detection rate (%) for each threshold.
    """
    diff = preds - targets  # [B,K,2] in resized coordinates
    thresholds = list(thresholds_mm)
    sdr_counts = [0.0 for _ in thresholds]
    total_landmarks = 0
    total_mre_mm = 0.0
    total_mre_px = 0.0

    batch_size = preds.shape[0]
    for idx in range(batch_size):
        sx = _as_float(metas["scale_x"][idx])
        sy = _as_float(metas["scale_y"][idx])
        pixel_size = _as_float(metas["pixel_size_mm"][idx])

        diff_sample = diff[idx]
        diff_x = diff_sample[:, 0] / sx
        diff_y = diff_sample[:, 1] / sy
        dist_px = torch.sqrt(diff_x**2 + diff_y**2)
        dist_mm = dist_px * pixel_size

        total_mre_mm += dist_mm.sum().item()
        total_mre_px += dist_px.sum().item()
        total_landmarks += dist_mm.numel()

        for thr_idx, thr in enumerate(thresholds):
            sdr_counts[thr_idx] += (dist_mm <= thr).sum().item()

    denom = max(total_landmarks, 1)
    metrics = {
        "mre_mm": total_mre_mm / denom,
        "mre_px": total_mre_px / denom,
        "num_points": denom,
        "sum_mre_mm": total_mre_mm,
        "sum_mre_px": total_mre_px,
    }
    for thr_idx, thr in enumerate(thresholds):
        base_key = str(thr).replace(".", "_")
        metrics[f"sdr_{base_key}mm"] = (sdr_counts[thr_idx] / denom) * 100.0
        metrics[f"sdr_hits_{base_key}mm"] = sdr_counts[thr_idx]
    return metrics
