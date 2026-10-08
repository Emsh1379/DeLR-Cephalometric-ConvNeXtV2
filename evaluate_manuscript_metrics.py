"""
Manuscript metrics from the saved test-set predictions (no GPU / no images needed).

Recomputes, from `checkpoints/*/test*_predictions.json` and the ground-truth
annotations of each dataset:

  * Table 2  - MRE +/- SD and SDR@2/2.5/3/4 mm with 95 % bootstrap confidence
               intervals (image-level resampling) for every test split.
  * Table 5  - CephAdoAdu adolescent vs adult comparison (per group metrics,
               difference with bootstrap CI, Mann-Whitney U, per-landmark tests).
  * Supplement - full per-landmark tables (CSV + Markdown) for every split.

Usage:
    python evaluate_manuscript_metrics.py \
        --aariz /path/to/Aariz --isbi /path/to/isbi_root \
        --cephadoadu "/path/to/CephAdoAdu Dataset"

Any dataset flag may be omitted; that dataset is then skipped.
Ground truth is built exactly as in `delr/datasets.py` (junior/senior mean for
Aariz and ISBI 2015). Pixel size: per image for Aariz, 0.1 mm for ISBI 2015, and
for CephAdoAdu the official CeLDA convention (longest image side rescaled to
2048 px at 0.1 mm/px; needs the images to read their sizes) unless
`--cephadoadu-spacing fixed` is given.
"""

from __future__ import annotations

import argparse
import csv
import json
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

THRESHOLDS_MM: Tuple[float, ...] = (2.0, 2.5, 3.0, 4.0)

ISBI_LANDMARKS = [
    "Sella", "Nasion", "Orbitale", "Porion", "Subspinale (A point)",
    "Supramentale (B point)", "Pogonion", "Menton", "Gnathion", "Gonion",
    "Lower incisal incision", "Upper incisal incision", "Upper lip", "Lower lip",
    "Subnasale", "Soft tissue pogonion", "Posterior nasal spine",
    "Anterior nasal spine", "Articulare",
]


# --------------------------------------------------------------------------- #
# Ground-truth loaders (mirror delr/datasets.py)
# --------------------------------------------------------------------------- #
# Test images whose labrale inferius (index 24) and labrale superius (index 25)
# labels are interchanged in both the junior and senior annotation files: the
# reference upper lip lies below the lower lip. They are exchanged back on load.
AARIZ_SWAPPED_LIPS = {"cl5lg05un01hm074k3dwy9l5q"}


def load_aariz_gt(root: Path, split: str = "test", num_landmarks: int = 26):
    base = root / split / "Annotations" / "Cephalometric Landmarks"
    pixel_size: Dict[str, float] = {}
    with (root / "cephalogram_machine_mappings.csv").open() as f:
        for row in csv.DictReader(f):
            pixel_size[row["cephalogram_id"]] = float(row["pixel_size"])

    gt, px, names = {}, {}, None
    for junior_file in sorted((base / "Junior Orthodontists").glob("*.json")):
        stem = junior_file.stem
        junior = json.loads(junior_file.read_text())["landmarks"]
        senior = json.loads((base / "Senior Orthodontists" / f"{stem}.json").read_text())["landmarks"]
        j = np.array([[p["value"]["x"], p["value"]["y"]] for p in junior], dtype=np.float32)
        s = np.array([[p["value"]["x"], p["value"]["y"]] for p in senior], dtype=np.float32)
        gt[stem] = ((j + s) * 0.5)[:num_landmarks].astype(np.float64)
        if stem in AARIZ_SWAPPED_LIPS and num_landmarks >= 26:
            gt[stem][[24, 25]] = gt[stem][[25, 24]]
        px[stem] = pixel_size[stem]
        if names is None:
            names = [p["title"] for p in junior][:num_landmarks]
    return gt, px, names


def _parse_isbi_file(path: Path, num_landmarks: int = 19) -> np.ndarray:
    coords: List[List[float]] = []
    for line in path.read_text().splitlines():
        parts = line.strip().split(",")
        if len(parts) < 2:
            continue
        try:
            coords.append([float(parts[0]), float(parts[1])])
        except ValueError:
            continue
        if len(coords) >= num_landmarks:
            break
    return np.asarray(coords, dtype=np.float32)


def load_isbi_gt(root: Path, ids: Sequence[str], pixel_size_mm: float = 0.1):
    gt, px = {}, {}
    for image_id in ids:
        junior = _parse_isbi_file(root / "400_junior" / f"{image_id}.txt")
        senior = _parse_isbi_file(root / "400_senior" / f"{image_id}.txt")
        gt[image_id] = ((junior + senior) * 0.5).astype(np.float64)
        px[image_id] = pixel_size_mm
    return gt, px, ISBI_LANDMARKS


# Landmark order (annotation "type" 1..10) as defined in the official CeLDA code
# (ShanghaiTech-IMPACT/CeLDA, code/test.py).
CEPHADOADU_LANDMARKS = ["A", "ANS", "UI", "UIA", "Or", "P", "LI", "LIA", "Sn", "Pog"]
CELDA_REFERENCE_SIDE = 2048
CELDA_PIXEL_SIZE_MM = 0.1


def load_cephadoadu_gt(root: Path, spacing: str = "celda", num_landmarks: int = 10):
    """
    spacing="celda": official CephAdoAdu / CeLDA convention (code/test.py and
        infer_util.mappping_back): coordinates are mapped to the image rescaled so
        that its longest side is 2048 px and distances are multiplied by 0.1 mm,
        i.e. an effective pixel size of 0.1 * 2048 / max(H, W) mm per original pixel.
        The dataset ships no physical spacing and image sizes vary widely.
    spacing="fixed": 0.1 mm per original pixel (what train.py / infer.py log).
    """
    splits = json.loads((root / "final_splits.json").read_text())
    gt, px, group = {}, {}, {}
    for image_id, grp in splits["test"]:
        ann = root / grp / "txt" / f"{image_id}.txt"
        if not ann.exists():
            continue
        records = sorted(json.loads(ann.read_text()), key=lambda r: int(r["type"]))
        gt[image_id] = np.array(
            [[float(r["data"][0]["x"]), float(r["data"][0]["y"])] for r in records[:num_landmarks]],
            dtype=np.float64,
        )
        if spacing == "celda":
            from PIL import Image

            with Image.open(root / grp / "dataset" / f"{image_id}.jpg") as im:
                px[image_id] = CELDA_PIXEL_SIZE_MM * CELDA_REFERENCE_SIDE / max(im.size)
        elif spacing == "fixed":
            px[image_id] = CELDA_PIXEL_SIZE_MM
        else:
            raise ValueError(f"Unknown spacing='{spacing}'")
        group[image_id] = grp
    return gt, px, CEPHADOADU_LANDMARKS[:num_landmarks], group


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def radial_errors_mm(pred_path: Path, gt: Dict[str, np.ndarray], px: Dict[str, float]):
    preds = json.loads(pred_path.read_text())
    ids = sorted(i for i in preds if i in gt)
    missing = sorted(set(preds) ^ set(gt))
    if missing:
        print(f"  [warn] {len(missing)} ids present in only one of predictions / ground truth "
              f"(e.g. {missing[:3]})")
    errors = np.stack([
        np.linalg.norm(np.asarray(preds[i], dtype=np.float64) - gt[i], axis=1) * px[i] for i in ids
    ])
    return ids, errors  # [N, K] in mm


def _stat_vector(errors: np.ndarray) -> np.ndarray:
    """[MRE, SDR@thr...] for an [N, K] error matrix."""
    return np.array([errors.mean()] + [(errors <= t).mean() * 100.0 for t in THRESHOLDS_MM])


def bootstrap_ci(errors: np.ndarray, n_boot: int, rng: np.random.Generator, alpha: float = 0.05):
    """Percentile bootstrap over images (the independent sampling unit)."""
    n = errors.shape[0]
    per_image_sum = errors.sum(axis=1)
    per_image_hits = np.stack([(errors <= t).sum(axis=1) for t in THRESHOLDS_MM], axis=1)
    idx = rng.integers(0, n, size=(n_boot, n))
    denom = n * errors.shape[1]
    boots = np.concatenate(
        [per_image_sum[idx].sum(axis=1, keepdims=True) / denom,
         per_image_hits[idx].sum(axis=1) / denom * 100.0],
        axis=1,
    )
    lo = np.percentile(boots, 100 * alpha / 2, axis=0)
    hi = np.percentile(boots, 100 * (1 - alpha / 2), axis=0)
    return lo, hi, boots


def summarise(errors: np.ndarray, n_boot: int, rng: np.random.Generator) -> Dict[str, float]:
    point = _stat_vector(errors)
    lo, hi, _ = bootstrap_ci(errors, n_boot, rng)
    out = {
        "n_images": int(errors.shape[0]),
        "n_landmarks": int(errors.shape[1]),
        "mre_mm": point[0], "mre_ci_lo": lo[0], "mre_ci_hi": hi[0],
        "sd_mm": errors.std(ddof=1),
        "median_mm": float(np.median(errors)),
    }
    for k, t in enumerate(THRESHOLDS_MM, start=1):
        out[f"sdr_{t}"] = point[k]
        out[f"sdr_{t}_ci_lo"] = lo[k]
        out[f"sdr_{t}_ci_hi"] = hi[k]
    return out


def per_landmark_table(errors: np.ndarray, names: Sequence[str], n_boot: int, rng) -> List[Dict]:
    rows = []
    for k, name in enumerate(names):
        col = errors[:, k:k + 1]
        s = summarise(col, n_boot, rng)
        rows.append({"index": k + 1, "landmark": name, **s})
    return rows


def holm(pvals: Sequence[float]) -> np.ndarray:
    p = np.asarray(pvals, dtype=np.float64)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


# --------------------------------------------------------------------------- #
# Formatting / output
# --------------------------------------------------------------------------- #
def fmt_ci(v: float, lo: float, hi: float, nd: int = 2) -> str:
    return f"{v:.{nd}f} ({lo:.{nd}f}-{hi:.{nd}f})"


def table2_row(label: str, s: Dict[str, float]) -> str:
    cells = [label, str(s["n_images"]), str(s["n_landmarks"]),
             f"{s['mre_mm']:.3f} +/- {s['sd_mm']:.3f}",
             f"{s['mre_ci_lo']:.3f}-{s['mre_ci_hi']:.3f}"]
    cells += [fmt_ci(s[f"sdr_{t}"], s[f"sdr_{t}_ci_lo"], s[f"sdr_{t}_ci_hi"]) for t in THRESHOLDS_MM]
    return "| " + " | ".join(cells) + " |"


TABLE2_HEADER = (
    "| Dataset / split | N images | Landmarks | MRE +/- SD (mm) | MRE 95% CI (mm) | "
    + " | ".join(f"SDR@{t} mm % (95% CI)" for t in THRESHOLDS_MM) + " |\n"
    + "|---|---:|---:|---:|---:|" + "---:|" * len(THRESHOLDS_MM)
)


def write_landmark_tables(out_dir: Path, tag: str, rows: List[Dict]) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"per_landmark_{tag}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "| # | Landmark | MRE +/- SD (mm) | MRE 95% CI | Median (mm) | "
        + " | ".join(f"SDR@{t} (%)" for t in THRESHOLDS_MM) + " |",
        "|---:|---|---:|---:|---:|" + "---:|" * len(THRESHOLDS_MM),
    ]
    for r in rows:
        lines.append(
            f"| {r['index']} | {r['landmark']} | {r['mre_mm']:.3f} +/- {r['sd_mm']:.3f} | "
            f"{r['mre_ci_lo']:.3f}-{r['mre_ci_hi']:.3f} | {r['median_mm']:.3f} | "
            + " | ".join(f"{r[f'sdr_{t}']:.2f}" for t in THRESHOLDS_MM) + " |"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--aariz", type=Path, default=None, help="Aariz dataset root (contains train/valid/test).")
    parser.add_argument("--isbi", type=Path, default=None, help="ISBI 2015 root (contains 400_junior / 400_senior).")
    parser.add_argument("--cephadoadu", type=Path, default=None, help="CephAdoAdu root (contains final_splits.json).")
    parser.add_argument("--checkpoints", type=Path, default=Path(__file__).resolve().parent / "checkpoints")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results" / "manuscript_metrics")
    parser.add_argument("--cephadoadu-spacing", choices=("celda", "fixed"), default="celda",
                        help="mm conversion for CephAdoAdu: 'celda' = official convention (longest side "
                             "rescaled to 2048 px at 0.1 mm/px); 'fixed' = 0.1 mm per original pixel.")
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()

    def rng_for(tag: str) -> np.random.Generator:
        # Independent stream per analysis, so every CI is reproducible no matter
        # which dataset flags are passed or in which order they are evaluated.
        return np.random.default_rng([args.seed, zlib.crc32(tag.encode())])

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = args.checkpoints

    table2: List[str] = []
    supplement: List[str] = []
    summary_json: Dict[str, Dict] = {}
    report: List[str] = []

    def run_split(label: str, tag: str, pred_path: Path, gt, px, names):
        ids, errors = radial_errors_mm(pred_path, gt, px)
        s = summarise(errors, args.n_boot, rng_for(tag))
        summary_json[tag] = s
        table2.append(table2_row(label, s))
        rows = per_landmark_table(errors, names, args.n_boot, rng_for(tag + "/landmarks"))
        md = write_landmark_tables(out_dir, tag, rows)
        supplement.append(f"### {label} - per-landmark results (N = {len(ids)} images)\n\n{md}\n")
        np.save(out_dir / f"radial_errors_mm_{tag}.npy", errors)
        (out_dir / f"image_ids_{tag}.json").write_text(json.dumps(ids))
        return ids, errors

    if args.aariz is not None:
        gt, px, names = load_aariz_gt(args.aariz, "test", 26)
        run_split("Aariz (test)", "aariz_test", ckpt / "Aariz_26" / "test_predictions.json", gt, px, names)

    if args.isbi is not None:
        for split, label in (("test1", "ISBI 2015 Test1"), ("test2", "ISBI 2015 Test2")):
            pred_path = ckpt / "ISBI2015" / f"{split}_predictions.json"
            ids = list(json.loads(pred_path.read_text()).keys())
            gt, px, names = load_isbi_gt(args.isbi, ids)
            run_split(label, f"isbi2015_{split}", pred_path, gt, px, names)

    table5_md = ""
    if args.cephadoadu is not None:
        gt, px, names, group = load_cephadoadu_gt(args.cephadoadu, args.cephadoadu_spacing)
        ids, errors = run_split("CephAdoAdu (official validation split, all)", "cephadoadu_test",
                                ckpt / "CephAdoAdu" / "test_predictions.json", gt, px, names)
        grp = np.array([group[i] for i in ids])
        ado, adu = errors[grp == "under_age"], errors[grp == "adult"]
        s_ado = summarise(ado, args.n_boot, rng_for("cephadoadu_adolescent"))
        s_adu = summarise(adu, args.n_boot, rng_for("cephadoadu_adult"))
        summary_json["cephadoadu_adolescent"], summary_json["cephadoadu_adult"] = s_ado, s_adu
        table2.append(table2_row("CephAdoAdu - adolescent", s_ado))
        table2.append(table2_row("CephAdoAdu - adult", s_adu))

        # Difference (adolescent - adult): independent bootstrap of each group.
        _, _, b_ado = bootstrap_ci(ado, args.n_boot, rng_for("cephadoadu_diff/adolescent"))
        _, _, b_adu = bootstrap_ci(adu, args.n_boot, rng_for("cephadoadu_diff/adult"))
        diff = _stat_vector(ado) - _stat_vector(adu)
        d_lo, d_hi = np.percentile(b_ado - b_adu, [2.5, 97.5], axis=0)

        # Image-level tests (per-image mean radial error is the independent unit).
        img_ado, img_adu = ado.mean(axis=1), adu.mean(axis=1)
        u_stat, u_p = stats.mannwhitneyu(img_ado, img_adu, alternative="two-sided")
        t_stat, t_p = stats.ttest_ind(img_ado, img_adu, equal_var=False)
        pooled_sd = np.sqrt((img_ado.var(ddof=1) + img_adu.var(ddof=1)) / 2)
        cohen_d = (img_ado.mean() - img_adu.mean()) / pooled_sd
        # u_stat belongs to the adolescent sample: U / (n1 n2) = P(adolescent > adult),
        # so negative r means adolescents have the smaller errors.
        rank_biserial = 2.0 * u_stat / (len(img_ado) * len(img_adu)) - 1.0

        lm_p = [stats.mannwhitneyu(ado[:, k], adu[:, k], alternative="two-sided").pvalue
                for k in range(errors.shape[1])]
        lm_p_holm = holm(lm_p)

        lines = [
            "| Metric | Adolescent (n = %d) | Adult (n = %d) | Difference Ado - Adu (95%% CI) |" % (len(ado), len(adu)),
            "|---|---:|---:|---:|",
            f"| MRE +/- SD (mm) | {s_ado['mre_mm']:.3f} +/- {s_ado['sd_mm']:.3f} | "
            f"{s_adu['mre_mm']:.3f} +/- {s_adu['sd_mm']:.3f} | {diff[0]:+.3f} ({d_lo[0]:+.3f} to {d_hi[0]:+.3f}) |",
            f"| MRE 95% CI (mm) | {s_ado['mre_ci_lo']:.3f}-{s_ado['mre_ci_hi']:.3f} | "
            f"{s_adu['mre_ci_lo']:.3f}-{s_adu['mre_ci_hi']:.3f} | |",
            f"| Median radial error (mm) | {s_ado['median_mm']:.3f} | {s_adu['median_mm']:.3f} | |",
        ]
        for k, t in enumerate(THRESHOLDS_MM, start=1):
            lines.append(
                f"| SDR@{t} mm (%) | {fmt_ci(s_ado[f'sdr_{t}'], s_ado[f'sdr_{t}_ci_lo'], s_ado[f'sdr_{t}_ci_hi'])} | "
                f"{fmt_ci(s_adu[f'sdr_{t}'], s_adu[f'sdr_{t}_ci_lo'], s_adu[f'sdr_{t}_ci_hi'])} | "
                f"{diff[k]:+.2f} ({d_lo[k]:+.2f} to {d_hi[k]:+.2f}) |"
            )
        lines += [
            "",
            f"Image-level comparison of per-image mean radial error: Mann-Whitney U = {u_stat:.1f}, "
            f"p = {u_p:.4g} (rank-biserial r = {rank_biserial:+.3f}); Welch t = {t_stat:.3f}, p = {t_p:.4g}; "
            f"Cohen's d = {cohen_d:+.3f}.",
            "",
            "| # | Landmark | Adolescent MRE +/- SD (mm) | Adult MRE +/- SD (mm) | Diff (mm) | "
            "Adolescent SDR@2 (%) | Adult SDR@2 (%) | p (Mann-Whitney) | p (Holm) |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        lm_rows = []
        for k, name in enumerate(names):
            row = {
                "index": k + 1, "landmark": name,
                "adolescent_mre_mm": ado[:, k].mean(), "adolescent_sd_mm": ado[:, k].std(ddof=1),
                "adult_mre_mm": adu[:, k].mean(), "adult_sd_mm": adu[:, k].std(ddof=1),
                "diff_mm": ado[:, k].mean() - adu[:, k].mean(),
                "adolescent_sdr_2mm": (ado[:, k] <= 2.0).mean() * 100,
                "adult_sdr_2mm": (adu[:, k] <= 2.0).mean() * 100,
                "p_mannwhitney": lm_p[k], "p_holm": lm_p_holm[k],
            }
            lm_rows.append(row)
            lines.append(
                f"| {k + 1} | {name} | {row['adolescent_mre_mm']:.3f} +/- {row['adolescent_sd_mm']:.3f} | "
                f"{row['adult_mre_mm']:.3f} +/- {row['adult_sd_mm']:.3f} | {row['diff_mm']:+.3f} | "
                f"{row['adolescent_sdr_2mm']:.2f} | {row['adult_sdr_2mm']:.2f} | "
                f"{row['p_mannwhitney']:.4g} | {row['p_holm']:.4g} |"
            )
        with (out_dir / "table5_cephadoadu_per_landmark_by_group.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(lm_rows[0].keys()))
            writer.writeheader()
            writer.writerows(lm_rows)
        # Sensitivity of the CephAdoAdu numbers to the mm conversion.
        lines += ["", "Sensitivity to the pixel-to-mm conversion (point estimates):", "",
                  "| Conversion | Group | MRE (mm) | " + " | ".join(f"SDR@{t} (%)" for t in THRESHOLDS_MM) + " |",
                  "|---|---|---:|" + "---:|" * len(THRESHOLDS_MM)]
        pred_path = ckpt / "CephAdoAdu" / "test_predictions.json"
        for mode, label in (("celda", "CeLDA: longest side -> 2048 px, 0.1 mm/px"),
                            ("fixed", "0.1 mm per original pixel")):
            gt_m, px_m, _, _ = load_cephadoadu_gt(args.cephadoadu, mode)
            ids_m, err_m = radial_errors_mm(pred_path, gt_m, px_m)
            grp_m = np.array([group[i] for i in ids_m])
            for g_label, sel in (("all", slice(None)), ("adolescent", grp_m == "under_age"), ("adult", grp_m == "adult")):
                v = _stat_vector(err_m[sel])
                summary_json[f"cephadoadu_sensitivity/{mode}/{g_label}"] = {
                    "mre_mm": v[0], **{f"sdr_{t}": v[k] for k, t in enumerate(THRESHOLDS_MM, start=1)}}
                lines.append(f"| {label} | {g_label} | {v[0]:.3f} | " + " | ".join(f"{x:.2f}" for x in v[1:]) + " |")
        table5_md = "\n".join(lines)
        summary_json["cephadoadu_group_comparison"] = {
            "n_adolescent": int(len(ado)), "n_adult": int(len(adu)),
            "mre_diff_mm": diff[0], "mre_diff_ci_lo": d_lo[0], "mre_diff_ci_hi": d_hi[0],
            **{f"sdr_{t}_diff": diff[k] for k, t in enumerate(THRESHOLDS_MM, start=1)},
            **{f"sdr_{t}_diff_ci_lo": d_lo[k] for k, t in enumerate(THRESHOLDS_MM, start=1)},
            **{f"sdr_{t}_diff_ci_hi": d_hi[k] for k, t in enumerate(THRESHOLDS_MM, start=1)},
            "mannwhitney_u": u_stat, "mannwhitney_p": u_p, "rank_biserial": rank_biserial,
            "welch_t": t_stat, "welch_p": t_p, "cohen_d": cohen_d,
        }

    report.append("# Manuscript metrics\n")
    report.append(f"95 % confidence intervals: percentile bootstrap, {args.n_boot} resamples at the image level, "
                  f"seed {args.seed}. SD is the standard deviation of the radial error over all landmark "
                  f"instances. Ground truth is the junior/senior mean for Aariz and ISBI 2015. "
                  f"Pixel size: per image from cephalogram_machine_mappings.csv (Aariz), 0.1 mm (ISBI 2015), "
                  f"CephAdoAdu conversion = '{args.cephadoadu_spacing}' (see Table 5 sensitivity rows).\n")
    report.append("## Table 2 - overall accuracy with 95 % confidence intervals\n")
    report.append(TABLE2_HEADER + "\n" + "\n".join(table2) + "\n")
    if table5_md:
        report.append("## Table 5 - CephAdoAdu: adolescent vs adult\n")
        report.append(table5_md + "\n")
    report.append("## Supplementary per-landmark tables\n")
    report.extend(supplement)

    text = "\n".join(report)
    (out_dir / "manuscript_metrics.md").write_text(text)
    (out_dir / "summary.json").write_text(json.dumps(summary_json, indent=2, default=float))
    print(text)
    print(f"\nWrote results to {out_dir}")


if __name__ == "__main__":
    main()
