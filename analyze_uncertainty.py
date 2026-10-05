"""
Predicted uncertainty vs observed error on the Aariz test set, and overlays of
the worst cases. Re-runs inference with the released checkpoint because the
prediction files store coordinates only.

    python analyze_uncertainty.py --aariz /path/to/Aariz \
        --checkpoint checkpoints/Aariz_26/best_model.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy import stats

import evaluate_manuscript_metrics as E
from delr import build_dcelr
from delr.datasets import AarizCephalometricDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aariz", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results" / "manuscript_metrics")
    ap.add_argument("--n-overlays", type=int, default=6)
    args = ap.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = AarizCephalometricDataset(args.aariz, "test", image_size=1024, return_heatmap=False, landmark_mode="26")
    loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False, num_workers=2)
    ck = torch.load(args.checkpoint, map_location=dev)
    cfg = ck.get("model_config", {}) if isinstance(ck, dict) else {}
    model = build_dcelr(num_landmarks=ds.num_landmarks, in_channels=1,
                        backbone=cfg.get("backbone", "convnextv2_tiny"),
                        num_layers_finetune=int(cfg.get("num_layers_finetune", 4))).to(dev)
    model.load_state_dict(ck.get("state_dict", ck))
    model.eval()

    pred, sig_raw, sig_mm = {}, {}, {}
    with torch.no_grad():
        for images, _, metas in loader:
            o = model(images.to(dev))
            mu, ls = o["fine_mu"].cpu().numpy().astype(np.float64), o["fine_log_sigma"].cpu().numpy()[..., 0].astype(np.float64)
            for b, image_id in enumerate(metas["image_id"]):
                sx, sy, px = float(metas["scale_x"][b]), float(metas["scale_y"][b]), float(metas["pixel_size_mm"][b])
                pred[image_id] = mu[b] / np.array([sx, sy])
                sig_raw[image_id] = np.exp(ls[b])                       # network (1024 x 1024) pixels
                sig_mm[image_id] = np.exp(ls[b]) * px * 0.5 * (1 / sx + 1 / sy)  # mm, mean of the two axis scales

    gt, px, names = E.load_aariz_gt(args.aariz, "test", 26)
    saved = json.loads((Path(__file__).resolve().parent / "checkpoints" / "Aariz_26" / "test_predictions.json").read_text())
    ids = sorted(gt)
    dmax = max(np.abs(pred[i] - np.asarray(saved[i])).max() for i in ids)
    err_saved = np.stack([np.linalg.norm(np.asarray(saved[i]) - gt[i], axis=1) * px[i] for i in ids])
    err = np.stack([np.linalg.norm(pred[i] - gt[i], axis=1) * px[i] for i in ids])
    s_raw = np.stack([sig_raw[i] for i in ids])
    s_mm = np.stack([sig_mm[i] for i in ids])
    np.save(out / "uncertainty_mm_aariz_test.npy", s_mm)
    np.save(out / "uncertainty_netpx_aariz_test.npy", s_raw)

    L = ["# Predicted uncertainty and failure cases (Aariz test set)\n",
         f"Inference re-run with `{args.checkpoint}`. Largest difference between re-run and saved predictions: "
         f"{dmax:.3f} original pixels; MRE re-run {err.mean():.4f} mm, saved predictions {err_saved.mean():.4f} mm. "
         f"All statistics below use the errors of the saved prediction files.\n",
         "Uncertainty = exp(predicted log-scale) of the last refinement update, converted from network pixels "
         "to millimetres with the image's pixel size and resize factors.\n"]
    e, u, ur = err_saved.ravel(), s_mm.ravel(), s_raw.ravel()
    L.append("## Correlation over all landmark predictions\n")
    L.append("| Uncertainty unit | n | Spearman rho | Pearson r |\n|---|---:|---:|---:|")
    L.append(f"| millimetres | {e.size} | {stats.spearmanr(u, e)[0]:.3f} | {stats.pearsonr(u, e)[0]:.3f} |")
    L.append(f"| network pixels (raw output) | {e.size} | {stats.spearmanr(ur, e)[0]:.3f} | {stats.pearsonr(ur, e)[0]:.3f} |\n")
    L.append(f"Uncertainty (mm): mean {u.mean():.3f}, median {np.median(u):.3f}. Error (mm): mean {e.mean():.3f}, "
             f"median {np.median(e):.3f}. Ratio mean error / mean uncertainty = {e.mean() / u.mean():.2f}; "
             f"median of per-prediction ratio = {np.median(e / u):.2f}; least-squares slope through origin = "
             f"{(e * u).sum() / (u * u).sum():.2f}.\n")
    L.append("## Deciles of predicted uncertainty (mm)\n")
    L.append("| Decile | n | Uncertainty range (mm) | Mean uncertainty (mm) | MRE (mm) | SDR 2 mm (%) | Error / uncertainty |\n|---:|---:|---|---:|---:|---:|---:|")
    order = np.argsort(u, kind="stable")
    for d, idx in enumerate(np.array_split(order, 10), start=1):
        L.append(f"| {d} | {len(idx)} | {u[idx].min():.3f}-{u[idx].max():.3f} | {u[idx].mean():.3f} | {e[idx].mean():.3f} | "
                 f"{(e[idx] <= 2).mean() * 100:.1f} | {e[idx].mean() / u[idx].mean():.2f} |")
    L.append("")

    machine = {}
    import csv
    with (args.aariz / "cephalogram_machine_mappings.csv").open() as f:
        for row in csv.DictReader(f):
            machine[row["cephalogram_id"]] = row["machine"].strip()
    k = err_saved.shape[1]
    L.append("## The 20 largest errors with their predicted uncertainty\n")
    L.append("| Rank | Image | Device | Landmark | Error (mm) | Predicted uncertainty (mm) | Uncertainty percentile |\n|---:|---|---|---|---:|---:|---:|")
    for rank, f in enumerate(np.argsort(-e)[:20], start=1):
        i, j = divmod(int(f), k)
        L.append(f"| {rank} | {ids[i]} | {machine[ids[i]]} | {names[j]} | {e[f]:.2f} | {u[f]:.3f} | {stats.percentileofscore(u, u[f]):.0f} |")
    L.append("")
    fails = e > 4
    L.append(f"Of the {int(fails.sum())} errors above 4 mm, {int((u[fails] > np.percentile(u, 90)).sum())} are in the top uncertainty "
             f"decile and {int((u[fails] > np.median(u)).sum())} are above the median uncertainty; median uncertainty of these "
             f"failures {np.median(u[fails]):.3f} mm versus {np.median(u[~fails]):.3f} mm for the other predictions.\n")

    # overlays of the worst images (by largest single error)
    stem2path = {p.stem: p for p in (args.aariz / "test" / "Cephalograms").iterdir()}
    worst_imgs = []
    for f in np.argsort(-e):
        i = int(f) // k
        if i not in worst_imgs:
            worst_imgs.append(i)
        if len(worst_imgs) >= args.n_overlays:
            break
    L.append("## Overlays\n")
    for n, i in enumerate(worst_imgs, start=1):
        im = Image.open(stem2path[ids[i]]).convert("RGB")
        dr = ImageDraw.Draw(im)
        r = max(4, im.width // 250)
        p, g = np.asarray(saved[ids[i]]), gt[ids[i]]
        for j in range(k):
            big = err_saved[i, j] > 4
            dr.line([tuple(g[j]), tuple(p[j])], fill=(255, 255, 0), width=max(2, r // 2))
            dr.ellipse([g[j][0] - r, g[j][1] - r, g[j][0] + r, g[j][1] + r], outline=(0, 255, 0), width=max(2, r // 2))
            dr.ellipse([p[j][0] - r, p[j][1] - r, p[j][0] + r, p[j][1] + r], fill=(255, 0, 0) if big else (0, 160, 255))
            if big:
                dr.text((p[j][0] + 2 * r, p[j][1] - 2 * r), f"{j + 1}", fill=(255, 0, 0))
        s = 1400 / max(im.size)
        im = im.resize((int(im.width * s), int(im.height * s)))
        name = f"overlay_aariz_{n}_{ids[i]}.jpg"
        im.save(out / name, quality=85)
        bad = ", ".join(f"{j + 1} {names[j]} {err_saved[i, j]:.1f} mm (u {s_mm[i, j]:.2f})" for j in np.argsort(-err_saved[i]) if err_saved[i, j] > 4)
        L.append(f"- `{name}` ({machine[ids[i]]}): {bad}")
    L.append("\nGreen circle: reference; filled dot: prediction (red and numbered if error > 4 mm); yellow line joins the two.\n")
    (out / "uncertainty_failures_aariz.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
