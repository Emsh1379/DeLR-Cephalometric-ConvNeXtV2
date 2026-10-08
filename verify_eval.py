"""Independent cross-checks of evaluate_manuscript_metrics.py (run from the repo root)."""
import json, sys, collections, subprocess
from pathlib import Path
import numpy as np, torch
from PIL import Image
from scipy import stats

sys.path.insert(0, ".")
import evaluate_manuscript_metrics as E
from delr.metrics import compute_mre_and_sdr
from delr.datasets import AarizCephalometricDataset, CephAdoAduDataset, ISBI2015Dataset

AARIZ = Path("/kaggle/working/data/aariz_raw/Aariz")
ISBI = Path("/kaggle/working/data/isbi")
ADO = Path("/kaggle/working/data/cephadoadu/content/drive/MyDrive/CephAdoAdu Dataset")
CK = Path("checkpoints")
ok = True
def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    detail = str(detail)
    print(("PASS " if cond else "FAIL ") + name + (" :: " + detail if detail else ""))

def repo_metric(preds, ds_samples_gt, px):
    """Feed the repo's own metric function (scale 1 => original pixel space)."""
    ids = sorted(preds)
    P = torch.tensor(np.stack([np.asarray(preds[i]) for i in ids]), dtype=torch.float64)
    G = torch.tensor(np.stack([ds_samples_gt[i] for i in ids]), dtype=torch.float64)
    metas = {"scale_x": [1.0] * len(ids), "scale_y": [1.0] * len(ids), "pixel_size_mm": [px[i] for i in ids]}
    return compute_mre_and_sdr(P, G, metas)

# ---------- 1. GT from the repo's dataset classes == GT from the eval script ----------
print("== 1. ground truth identical to repo loaders; metrics identical to delr.metrics ==")
ds = AarizCephalometricDataset(AARIZ, "test", return_heatmap=False, landmark_mode="26")
gt, px, names = E.load_aariz_gt(AARIZ, "test", 26)
d = max(np.abs(ds._load_annotations(p.stem)["coords"] - gt[p.stem]).max() for p in ds.samples)
check("aariz GT == loader GT", d < 1e-3 and len(gt) == len(ds.samples) == 150, f"max diff {d:.2e}px, n={len(gt)}")
check("aariz names == loader names", names == ds.landmark_names)
check("aariz pixel size == loader", all(px[p.stem] == ds.pixel_size_map[p.stem] for p in ds.samples),
      dict(collections.Counter(px.values())))
preds = json.load(open(CK / "Aariz_26/test_predictions.json"))
ids, err = E.radial_errors_mm(CK / "Aariz_26/test_predictions.json", gt, px)
m = repo_metric(preds, gt, px)
check("aariz MRE/SDR == delr.metrics", abs(m["mre_mm"] - err.mean()) < 1e-6 and abs(m["sdr_2_0mm"] - (err <= 2).mean() * 100) < 1e-9,
      f"{m['mre_mm']:.4f} / {m['sdr_2_0mm']:.2f}")
check("aariz ids: preds == test GT", set(preds) == set(gt) and err.shape == (150, 26) and np.isfinite(err).all())

# junior/senior title order consistency (loader averages by position)
base = AARIZ / "test/Annotations/Cephalometric Landmarks"
bad, nl = 0, collections.Counter()
ref = None
for jf in (base / "Junior Orthodontists").glob("*.json"):
    j = [p["title"] for p in json.load(open(jf))["landmarks"]]
    s = [p["title"] for p in json.load(open(base / "Senior Orthodontists" / jf.name))["landmarks"]]
    ref = ref or j
    bad += (j != s) or (j != ref)
    nl[len(j)] += 1
check("aariz landmark order identical in every junior+senior file", bad == 0, f"mismatching files={bad}, counts={dict(nl)}")
print("   aariz 26 names:", names)
print("   dropped (27-29):", ref[26:])

for split, n in (("test1", 150), ("test2", 100)):
    ds = ISBI2015Dataset(ISBI, split, return_heatmap=False)
    preds = json.load(open(CK / f"ISBI2015/{split}_predictions.json"))
    gt, px, _ = E.load_isbi_gt(ISBI, list(preds))
    d = max(np.abs(ds._load_annotation(p.stem) - gt[p.stem]).max() for p in ds.samples)
    ids, err = E.radial_errors_mm(CK / f"ISBI2015/{split}_predictions.json", gt, px)
    m = repo_metric(preds, gt, px)
    check(f"isbi {split} GT == loader GT, ids match", d < 1e-3 and {p.stem for p in ds.samples} == set(preds) and err.shape == (n, 19), f"max diff {d:.2e}")
    check(f"isbi {split} MRE/SDR == delr.metrics", abs(m["mre_mm"] - err.mean()) < 1e-6 and abs(m["sdr_2_0mm"] - (err <= 2).mean() * 100) < 1e-9, f"{m['mre_mm']:.4f} / {m['sdr_2_0mm']:.2f}")

ds = CephAdoAduDataset(ADO, "test", return_heatmap=False)
gt, px, names, group = E.load_cephadoadu_gt(ADO, "fixed")
preds = json.load(open(CK / "CephAdoAdu/test_predictions.json"))
d = max(np.abs(ds._load_annotation(a) - gt[i]).max() for _, a, i in ds.samples)
check("cephadoadu GT == loader GT, ids match", d < 1e-3 and {i for *_, i in ds.samples} == set(preds) == set(gt), f"max diff {d:.2e}, n={len(gt)}")
ids, err = E.radial_errors_mm(CK / "CephAdoAdu/test_predictions.json", gt, px)
m = repo_metric(preds, gt, px)
check("cephadoadu MRE/SDR == delr.metrics", abs(m["mre_mm"] - err.mean()) < 1e-6, f"{m['mre_mm']:.4f} / {m['sdr_2_0mm']:.2f}")
check("cephadoadu README numbers reproduced (1.045 / 87.53 / 92.37 / 95.27 / 97.63)",
      abs(err.mean() - 1.045) < 6e-4 and all(abs((err <= t).mean() * 100 - v) < 6e-3 for t, v in zip((2, 2.5, 3, 4), (87.53, 92.37, 95.27, 97.63))),
      [round(err.mean(), 4)] + [round((err <= t).mean() * 100, 2) for t in (2, 2.5, 3, 4)])
check("cephadoadu groups 150/150", collections.Counter(group.values()) == {"adult": 150, "under_age": 150}, dict(collections.Counter(group.values())))
# annotation structure: types unique 1..K, one point each; group folder is unambiguous
types, npts, both = collections.Counter(), collections.Counter(), 0
for i, g in group.items():
    rec = json.load(open(ADO / g / "txt" / f"{i}.txt"))
    types[tuple(sorted(int(r["type"]) for r in rec))] += 1
    npts[tuple(len(r["data"]) for r in rec)] += 1
    other = "adult" if g == "under_age" else "under_age"
    both += (ADO / other / "txt" / f"{i}.txt").exists()
check("cephadoadu every test annotation = types 1..10, one point each", list(types) == [tuple(range(1, 11))] and set(npts) == {(1,) * 10}, f"{dict(types)} {dict(npts)}")
check("cephadoadu no test id present in both age folders", both == 0, both)

# ---------- 2. README / log numbers ----------
print("== 2. numbers already published in README ==")
for tag, f, g, ref in (("isbi test1", "ISBI2015/test1_predictions.json", None, (1.124, 87.12, 92.53, 96.11, 98.39)),
                       ("isbi test2", "ISBI2015/test2_predictions.json", None, (1.463, 74.74, 83.47, 88.84, 94.63))):
    p = json.load(open(CK / f)); gt_, px_, _ = E.load_isbi_gt(ISBI, list(p)); _, e = E.radial_errors_mm(CK / f, gt_, px_)
    got = [e.mean()] + [(e <= t).mean() * 100 for t in (2, 2.5, 3, 4)]
    check(f"{tag} README reproduced", abs(got[0] - ref[0]) < 6e-4 and all(abs(a - b) < 6e-3 for a, b in zip(got[1:], ref[1:])), np.round(got, 3))
print(subprocess.run("grep -n -i -E 'test' checkpoints/Aariz_26/train.log | tail -5; grep -c . checkpoints/Aariz_26/train.log; tail -3 checkpoints/Aariz_26/train.log | cut -c1-400", shell=True, capture_output=True, text=True).stdout)

# ---------- 3. GT clipping in the loader (eval script does not clip) ----------
print("== 3. would the loader's coordinate clipping change any GT? ==")
def oob(gtd, path_of):
    n = 0
    for i, c in gtd.items():
        w, h = Image.open(path_of(i)).size
        n += int(((c[:, 0] < 0) | (c[:, 0] > w * (1 - 1 / 1024)) | (c[:, 1] < 0) | (c[:, 1] > h * (1 - 1 / 1024))).sum())
    return n
gt_a, _, _ = E.load_aariz_gt(AARIZ, "test", 26)
stem2path = {p.stem: p for p in (AARIZ / "test/Cephalograms").iterdir()}
check("aariz: no GT landmark outside clip range", oob(gt_a, lambda i: stem2path[i]) == 0)
check("cephadoadu: no GT landmark outside clip range", oob(gt, lambda i: ADO / group[i] / "dataset" / f"{i}.jpg") == 0)
ids12 = [f"{i:03d}" for i in range(151, 401)]
gt_i, _, _ = E.load_isbi_gt(ISBI, ids12)
check("isbi: no GT landmark outside clip range", oob(gt_i, lambda i: ISBI / "RawImage" / ("Test1Data" if int(i) <= 300 else "Test2Data") / f"{i}.bmp") == 0)

# ---------- 4. statistics ----------
print("== 4. statistics ==")
rng = np.random.default_rng(0)
lo, hi, boots = E.bootstrap_ci(err, 300, rng)
rng = np.random.default_rng(0)
idx = rng.integers(0, err.shape[0], size=(300, err.shape[0]))
naive = np.array([E._stat_vector(err[r]) for r in idx])
check("vectorised bootstrap == naive resample-and-recompute", np.allclose(naive, boots, atol=1e-10), f"max diff {np.abs(naive - boots).max():.1e}")
# percentile CI vs scipy BCa on per-image means (equal K per image => same estimator)
img = err.mean(axis=1)
r = stats.bootstrap((img,), np.mean, n_resamples=10000, method="BCa", random_state=1).confidence_interval
lo, hi, _ = E.bootstrap_ci(err, 10000, np.random.default_rng(1))
t = stats.t.interval(0.95, len(img) - 1, loc=img.mean(), scale=stats.sem(img))
print(f"   cephadoadu MRE CI: percentile {lo[0]:.3f}-{hi[0]:.3f} | BCa {r.low:.3f}-{r.high:.3f} | t-interval {t[0]:.3f}-{t[1]:.3f}")
check("percentile CI agrees with BCa and t-interval within 0.02 mm", max(abs(lo[0] - r.low), abs(hi[0] - r.high), abs(lo[0] - t[0]), abs(hi[0] - t[1])) < 0.02)
try:
    from statsmodels.stats.multitest import multipletests
    p = np.random.default_rng(3).uniform(0, 0.2, 10)
    check("Holm == statsmodels", np.allclose(E.holm(p), multipletests(p, method="holm")[1]))
except ImportError:
    p = np.array([0.01, 0.04, 0.03, 0.005]); check("Holm hand example", np.allclose(E.holm(p), [0.03, 0.06, 0.06, 0.02]))
g = np.array([group[i] for i in ids]); a, b = err[g == "under_age"].mean(1), err[g == "adult"].mean(1)
u = stats.mannwhitneyu(a, b).statistic
auc = (a[:, None] > b[None, :]).mean() + 0.5 * (a[:, None] == b[None, :]).mean()
check("U statistic belongs to adolescent sample (U/(n1 n2) == P[ado > adu])", abs(u / (len(a) * len(b)) - auc) < 1e-12, f"P(ado>adu)={auc:.3f}")
print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
