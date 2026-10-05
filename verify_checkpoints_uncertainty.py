"""
Re-run inference with the released checkpoints and check (a) that they reproduce
the saved prediction files and (b) how the predicted uncertainty relates to the
observed error, for every evaluation set.

    python verify_checkpoints_uncertainty.py --aariz ... --isbi ... --cephadoadu ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy import stats

import evaluate_manuscript_metrics as E
from delr import build_dcelr
from delr.datasets import AarizCephalometricDataset, CephAdoAduDataset, ISBI2015Dataset

ROOT = Path(__file__).resolve().parent


def run(ds, ckpt, dev):
    loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False, num_workers=2)
    ck = torch.load(ckpt, map_location=dev, weights_only=False)
    cfg = ck.get("model_config", {}) if isinstance(ck, dict) else {}
    model = build_dcelr(num_landmarks=ds.num_landmarks, in_channels=1, backbone=cfg.get("backbone", "convnextv2_tiny"),
                        num_layers_finetune=int(cfg.get("num_layers_finetune", 4))).to(dev)
    model.load_state_dict(ck.get("state_dict", ck))
    model.eval()
    pred, sig = {}, {}
    with torch.no_grad():
        for images, _, metas in loader:
            o = model(images.to(dev))
            mu = o["fine_mu"].cpu().numpy().astype(np.float64)
            ls = o["fine_log_sigma"].cpu().numpy()[..., 0].astype(np.float64)
            for b, image_id in enumerate(metas["image_id"]):
                sx, sy = float(metas["scale_x"][b]), float(metas["scale_y"][b])
                pred[str(image_id)] = mu[b] / np.array([sx, sy])
                sig[str(image_id)] = np.exp(ls[b]) * 0.5 * (1 / sx + 1 / sy)   # original pixels
    del model
    torch.cuda.empty_cache()
    return pred, sig


def report(label, tag, pred, sig, saved_path, gt, px, out):
    saved = json.loads(saved_path.read_text())
    ids = sorted(gt)
    dmax = max(np.abs(pred[i] - np.asarray(saved[i])).max() for i in ids)
    e_saved = np.stack([np.linalg.norm(np.asarray(saved[i]) - gt[i], axis=1) * px[i] for i in ids])
    e_new = np.stack([np.linalg.norm(pred[i] - gt[i], axis=1) * px[i] for i in ids])
    u = np.stack([sig[i] * px[i] for i in ids])
    np.save(out / f"uncertainty_mm_{tag}.npy", u)
    e, uu = e_saved.ravel(), u.ravel()
    order = np.argsort(uu, kind="stable")
    dec = np.array_split(order, 10)
    fails = e > 4
    row = [label, len(ids), f"{dmax:.3f}", f"{e_new.mean():.4f}", f"{e_saved.mean():.4f}",
           f"{stats.spearmanr(uu, e)[0]:.3f}", f"{stats.pearsonr(uu, e)[0]:.3f}",
           f"{e[dec[0]].mean():.2f} ({(e[dec[0]] <= 2).mean() * 100:.1f}%)",
           f"{e[dec[-1]].mean():.2f} ({(e[dec[-1]] <= 2).mean() * 100:.1f}%)",
           f"{e.mean() / uu.mean():.2f}",
           f"{int((uu[fails] > np.median(uu)).sum())}/{int(fails.sum())}",
           f"{int((uu[fails] > np.percentile(uu, 90)).sum())}/{int(fails.sum())}"]
    dec_rows = [[label, d, f"{uu[i].mean():.3f}", f"{e[i].mean():.3f}", f"{(e[i] <= 2).mean() * 100:.1f}", f"{e[i].mean() / uu[i].mean():.2f}"]
                for d, i in enumerate(dec, start=1)]
    return row, dec_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aariz", type=Path)
    ap.add_argument("--isbi", type=Path)
    ap.add_argument("--cephadoadu", type=Path)
    ap.add_argument("--output-dir", type=Path, default=ROOT / "results" / "manuscript_metrics")
    args = ap.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = ROOT / "checkpoints"
    rows, decs = [], []

    if args.aariz:
        ds = AarizCephalometricDataset(args.aariz, "test", return_heatmap=False, landmark_mode="26")
        pred, sig = run(ds, ck / "Aariz_26" / "best_model.pt", dev)
        gt, px, _ = E.load_aariz_gt(args.aariz, "test", 26)
        r, d = report("Aariz test", "aariz_test", pred, sig, ck / "Aariz_26" / "test_predictions.json", gt, px, out)
        rows.append(r); decs += d
    if args.isbi:
        for split, label in (("test1", "ISBI 2015 Test1"), ("test2", "ISBI 2015 Test2")):
            ds = ISBI2015Dataset(args.isbi, split, return_heatmap=False)
            pred, sig = run(ds, ck / "ISBI2015" / "best_model.pt", dev)
            gt, px, _ = E.load_isbi_gt(args.isbi, list(pred))
            r, d = report(label, f"isbi2015_{split}", pred, sig, ck / "ISBI2015" / f"{split}_predictions.json", gt, px, out)
            rows.append(r); decs += d
    if args.cephadoadu:
        ds = CephAdoAduDataset(args.cephadoadu, "test", return_heatmap=False)
        pred, sig = run(ds, ck / "CephAdoAdu" / "best_model.pt", dev)
        gt, px, _, _ = E.load_cephadoadu_gt(args.cephadoadu, "celda")
        r, d = report("CephAdoAdu validation part", "cephadoadu", pred, sig, ck / "CephAdoAdu" / "test_predictions.json", gt, px, out)
        rows.append(r); decs += d

    def table(h, rs):
        return "\n".join(["| " + " | ".join(h) + " |", "|" + "---|" * len(h)] + ["| " + " | ".join(map(str, r)) + " |" for r in rs])

    text = "\n".join([
        "# Released checkpoints: reproduction of the prediction files and predicted uncertainty\n",
        "Inference re-run with the released weights. 'Max diff' is the largest coordinate difference (original pixels) "
        "between the re-run and the saved prediction file. Uncertainty = exp(predicted log-scale) of the last refinement "
        "update, converted to millimetres. Correlations and deciles are over all landmark predictions and use the errors "
        "of the saved prediction files.\n",
        table(["Evaluation set", "Images", "Max diff (px)", "MRE re-run (mm)", "MRE saved (mm)", "Spearman rho", "Pearson r",
               "Most confident decile: MRE (SDR 2 mm)", "Least confident decile: MRE (SDR 2 mm)", "Mean error / mean uncertainty",
               "Errors > 4 mm with above-median uncertainty", "Errors > 4 mm in least confident decile"], rows),
        "\n## Deciles of predicted uncertainty\n",
        table(["Evaluation set", "Decile", "Mean uncertainty (mm)", "MRE (mm)", "SDR 2 mm (%)", "Error / uncertainty"], decs), ""])
    (out / "checkpoint_reproduction_uncertainty.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
