"""
Per-device accuracy (Aariz), failure cases named by landmark, and rank
correlations, recomputed from the saved prediction files.

Landmark names are never typed by hand for Aariz: they are read from the
"title" field of each annotation entry. ISBI 2015 uses the official challenge
order and CephAdoAdu the order in the CeLDA reference code.

Usage (same flags as evaluate_manuscript_metrics.py):
    python analyze_devices_failures.py --aariz ... --isbi ... --cephadoadu ...
"""

from __future__ import annotations

import argparse
import csv
import json
import zlib
from pathlib import Path

import numpy as np
from scipy import stats

import evaluate_manuscript_metrics as E

FAIL_MM = 4.0


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def failure_section(label, tag, ids, err, names, out_dir, extra=None, top=15):
    """extra: optional {image_id: str} shown as an additional column."""
    n, k = err.shape
    lines = [f"### {label}\n"]
    fails = err > FAIL_MM
    lines.append(f"{int(fails.sum())} of {n * k} landmark predictions ({fails.mean() * 100:.2f} %) exceed "
                 f"{FAIL_MM:.0f} mm; {int(fails.any(axis=1).sum())} of {n} images contain at least one.\n")

    order = np.argsort(-fails.sum(axis=0), kind="stable")
    rows = [[j + 1, names[j], int(fails[:, j].sum()), f"{fails[:, j].mean() * 100:.1f}",
             f"{err[:, j].mean():.3f}", f"{err[:, j].max():.2f}"] for j in order]
    lines.append(f"Failures (> {FAIL_MM:.0f} mm) by landmark, most frequent first:\n")
    lines.append(md_table(["#", "Landmark", f"n > {FAIL_MM:.0f} mm", "% of images", "MRE (mm)", "Max error (mm)"], rows) + "\n")

    flat = np.argsort(-err, axis=None)[:top]
    rows, csv_rows = [], []
    for rank, f in enumerate(flat, start=1):
        i, j = divmod(int(f), k)
        row = [rank, ids[i], j + 1, names[j], f"{err[i, j]:.2f}"]
        if extra is not None:
            row.append(extra[ids[i]])
        rows.append(row)
    header = ["Rank", "Image", "#", "Landmark", "Error (mm)"] + (["Group / device"] if extra is not None else [])
    lines.append(f"The {top} largest individual errors:\n")
    lines.append(md_table(header, rows) + "\n")

    img_mean = err.mean(axis=1)
    rows = []
    for i in np.argsort(-img_mean)[:5]:
        worst = int(np.argmax(err[i]))
        row = [ids[i], f"{img_mean[i]:.2f}", int(fails[i].sum()), f"{names[worst]} ({err[i, worst]:.2f} mm)"]
        if extra is not None:
            row.append(extra[ids[i]])
        rows.append(row)
    header = ["Image", "Image MRE (mm)", f"Landmarks > {FAIL_MM:.0f} mm", "Worst landmark"] + (["Group / device"] if extra is not None else [])
    lines.append("The 5 images with the highest mean error:\n")
    lines.append(md_table(header, rows) + "\n")

    with (out_dir / f"failures_gt{FAIL_MM:.0f}mm_{tag}.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "landmark_index", "landmark", "error_mm", "group_or_device"])
        for i, j in sorted(zip(*np.nonzero(fails)), key=lambda t: -err[t[0], t[1]]):
            w.writerow([ids[i], j + 1, names[j], f"{err[i, j]:.4f}", extra[ids[i]] if extra is not None else ""])
    return "\n".join(lines)


def rho_line(name, x, y):
    r, p = stats.spearmanr(x, y)
    pr, pp = stats.pearsonr(x, y)
    return [name, len(x), f"{r:+.3f}", f"{p:.4g}", f"{pr:+.3f}", f"{pp:.4g}"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aariz", type=Path)
    ap.add_argument("--isbi", type=Path)
    ap.add_argument("--cephadoadu", type=Path)
    ap.add_argument("--checkpoints", type=Path, default=Path(__file__).resolve().parent / "checkpoints")
    ap.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results" / "manuscript_metrics")
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()
    out_dir, ck = args.output_dir, args.checkpoints
    out_dir.mkdir(parents=True, exist_ok=True)
    rep = ["# Per-device accuracy, failure cases and rank correlations\n",
           "Recomputed from the saved prediction files. A failure is a radial error above 4 mm.\n"]
    rho_rows = []

    if args.aariz:
        gt, px, names = E.load_aariz_gt(args.aariz, "test", 26)
        ids, err = E.radial_errors_mm(ck / "Aariz_26" / "test_predictions.json", gt, px)
        machine, fmt = {}, {}
        with (args.aariz / "cephalogram_machine_mappings.csv").open() as f:
            for row in csv.DictReader(f):
                machine[row["cephalogram_id"]] = row["machine"].strip()
        dev = np.array([machine[i] for i in ids])

        rep.append("## Aariz - accuracy by imaging device (test set)\n")
        rows, total_n, total_sum = [], 0, 0.0
        for d in sorted(set(dev), key=lambda d: -(dev == d).sum()):
            e = err[dev == d]
            ps = sorted({px[i] for i, m in zip(ids, dev) if m == d})
            rng = np.random.default_rng([2025, zlib.crc32(d.encode())])
            lo, hi, _ = E.bootstrap_ci(e, args.n_boot, rng)
            v = E._stat_vector(e)
            rows.append([d, len(e), "/".join(f"{p:g}" for p in ps), f"{v[0]:.3f} +/- {e.std(ddof=1):.3f}",
                         f"{lo[0]:.3f}-{hi[0]:.3f}"] + [f"{x:.2f}" for x in v[1:]])
            total_n += len(e)
            total_sum += e.sum()
        v = E._stat_vector(err)
        rows.append(["**All**", len(err), "", f"{v[0]:.3f} +/- {err.std(ddof=1):.3f}", ""] + [f"{x:.2f}" for x in v[1:]])
        rep.append(md_table(["Device", "N images", "Pixel size (mm)", "MRE +/- SD (mm)", "MRE 95% CI"]
                            + [f"SDR@{t} (%)" for t in E.THRESHOLDS_MM], rows) + "\n")
        rep.append(f"Check: {len(set(dev))} devices, {total_n} images; image-weighted mean of the device MREs = "
                   f"{total_sum / (total_n * err.shape[1]):.3f} mm (overall {err.mean():.3f} mm). "
                   f"Kruskal-Wallis on per-image mean error across devices: H = "
                   f"{stats.kruskal(*[err[dev == d].mean(axis=1) for d in set(dev)]).statistic:.2f}, p = "
                   f"{stats.kruskal(*[err[dev == d].mean(axis=1) for d in set(dev)]).pvalue:.4g}.\n")

        # inter-observer distance (junior vs senior), same landmarks, mm
        base = args.aariz / "test" / "Annotations" / "Cephalometric Landmarks"
        io = []
        for i in ids:
            j = json.loads((base / "Junior Orthodontists" / f"{i}.json").read_text())["landmarks"][:26]
            s = json.loads((base / "Senior Orthodontists" / f"{i}.json").read_text())["landmarks"][:26]
            assert [p["title"] for p in j] == [p["title"] for p in s] == names
            jc = np.array([[p["value"]["x"], p["value"]["y"]] for p in j], dtype=float)
            sc = np.array([[p["value"]["x"], p["value"]["y"]] for p in s], dtype=float)
            io.append(np.linalg.norm(jc - sc, axis=1) * px[i])
        io = np.stack(io)
        np.save(out_dir / "interobserver_mm_aariz_test.npy", io)
        rho_rows.append(rho_line("Aariz: per-landmark model MRE vs junior-senior distance (26 landmarks)", err.mean(0), io.mean(0)))
        rho_rows.append(rho_line("Aariz: per-image model MRE vs junior-senior distance (150 images)", err.mean(1), io.mean(1)))
        rho_rows.append(rho_line("Aariz: all landmark instances, model error vs junior-senior distance", err.ravel(), io.ravel()))
        rho_rows.append(rho_line("Aariz: per-image model MRE vs pixel size", err.mean(1), np.array([px[i] for i in ids])))
        rep.append(f"Aariz inter-observer (junior vs senior) distance on the same landmarks: "
                   f"{io.mean():.3f} +/- {io.std(ddof=1):.3f} mm (model vs their mean: {err.mean():.3f} mm).\n")
        aariz_fail = failure_section("Aariz (test, 26 landmarks)", "aariz_test", ids, err, names, out_dir, extra=machine)

        lm = [[k + 1, names[k], f"{err[:, k].mean():.3f}", f"{io[:, k].mean():.3f}"] for k in range(26)]
        aariz_io_table = md_table(["#", "Landmark", "Model MRE (mm)", "Junior-senior distance (mm)"], lm)
    else:
        aariz_fail = aariz_io_table = None

    isbi_fail = []
    if args.isbi:
        for split, label in (("test1", "ISBI 2015 Test1"), ("test2", "ISBI 2015 Test2")):
            p = ck / "ISBI2015" / f"{split}_predictions.json"
            pid = list(json.loads(p.read_text()))
            gt, px, names = E.load_isbi_gt(args.isbi, pid)
            ids, err = E.radial_errors_mm(p, gt, px)
            io = np.stack([np.linalg.norm(E._parse_isbi_file(args.isbi / "400_junior" / f"{i}.txt").astype(float)
                                          - E._parse_isbi_file(args.isbi / "400_senior" / f"{i}.txt").astype(float), axis=1) * 0.1
                           for i in ids])
            rho_rows.append(rho_line(f"{label}: per-landmark model MRE vs junior-senior distance (19 landmarks)", err.mean(0), io.mean(0)))
            rho_rows.append(rho_line(f"{label}: per-image model MRE vs junior-senior distance", err.mean(1), io.mean(1)))
            isbi_fail.append(failure_section(f"{label} (19 landmarks)", f"isbi2015_{split}", ids, err, names, out_dir)
                             + f"\nInter-observer (junior vs senior) distance: {io.mean():.3f} +/- {io.std(ddof=1):.3f} mm.\n")

    ado_fail = None
    if args.cephadoadu:
        gt, px, names, group = E.load_cephadoadu_gt(args.cephadoadu, "celda")
        ids, err = E.radial_errors_mm(ck / "CephAdoAdu" / "test_predictions.json", gt, px)
        label = {"under_age": "adolescent", "adult": "adult"}
        ado_fail = failure_section("CephAdoAdu (official validation part, 10 landmarks)", "cephadoadu", ids, err, names, out_dir,
                                   extra={i: label[group[i]] for i in ids})
        g = np.array([group[i] for i in ids])
        f = err > FAIL_MM
        ado_fail += (f"\nFailures by age group: adolescent {int(f[g == 'under_age'].sum())} of {f[g == 'under_age'].size}, "
                     f"adult {int(f[g == 'adult'].sum())} of {f[g == 'adult'].size}.\n")
        rho_rows.append(rho_line("CephAdoAdu: per-image model MRE vs CeLDA mm-per-pixel factor", err.mean(1), np.array([px[i] for i in ids])))

    rep.append("## Failure cases (named by landmark)\n")
    for sec in [aariz_fail] + isbi_fail + [ado_fail]:
        if sec:
            rep.append(sec)
    rep.append("## Rank correlations\n")
    rep.append(md_table(["Quantity", "n", "Spearman rho", "p", "Pearson r", "p"], rho_rows) + "\n")
    if aariz_io_table:
        rep.append("### Aariz - model error and inter-observer distance by landmark\n")
        rep.append(aariz_io_table + "\n")
    text = "\n".join(rep)
    (out_dir / "devices_failures_correlations.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
