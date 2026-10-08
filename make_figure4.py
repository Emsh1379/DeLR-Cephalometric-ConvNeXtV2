"""Figure 4 - ablation study on ISBI 2015 (mean radial error, log scale).

Values are those of Table 6. The reference ("Full model") is the full model
trained with the same 150-epoch schedule as the variants
(results/training_runs/ablation_full150/metrics.json).
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator

HERE = os.path.dirname(os.path.abspath(__file__))
ref = json.load(open(os.path.join(HERE, "results", "training_runs", "ablation_full150", "metrics.json")))

labels = ["Full model", "No heatmap\nhead", "M = 1", "No\nrefinement", "No RLE\nloss"]
test1 = [round(ref["test1"]["mre_mm"], 3), 1.171, 1.770, 12.616, 40.732]
test2 = [round(ref["test2"]["mre_mm"], 3), 1.549, 2.112, 12.141, 40.263]

BLUE, ORANGE = "#2F6DB5", "#D9812B"
fig, ax = plt.subplots(figsize=(5.5, 3.2), dpi=300)
w = 0.36
x = range(len(labels))
b1 = ax.bar([i - w / 2 for i in x], test1, w, color=BLUE, label="Test1")
b2 = ax.bar([i + w / 2 for i in x], test2, w, color=ORANGE, label="Test2")
ax.axhline(2, color="#666666", ls="--", lw=0.8, zorder=0)
for bars in (b1, b2):
    for r in bars:
        ax.annotate(f"{r.get_height():.3f}", (r.get_x() + r.get_width() / 2, r.get_height()), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=6.5,
                    bbox=dict(facecolor="white", edgecolor="none", pad=0.4))
ax.set_yscale("log")
ax.set_ylim(0.8, 80)
ax.yaxis.set_major_locator(FixedLocator([1, 2, 5, 10, 20, 50]))
ax.yaxis.set_minor_locator(NullLocator())
ax.set_yticklabels(["1", "2", "5", "10", "20", "50"])
ax.set_ylabel("Mean radial error (mm, log scale)", fontsize=9)
ax.set_xticks(list(x))
ax.set_xticklabels(labels, fontsize=8)
ax.tick_params(axis="y", labelsize=9)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout()
out = os.path.join(HERE, "updated_manuscript", "Figure4_ablation.tif")
fig.savefig(out, dpi=300, pil_kwargs={"compression": "tiff_lzw"})
fig.savefig(os.path.join(HERE, "updated_manuscript", "Figure4_ablation_preview.png"), dpi=300)
print("saved", out, test1, test2)
