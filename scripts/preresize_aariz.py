#!/usr/bin/env python3
"""Pre-resize Aariz cephalograms to 1024x1024 PNG and write a sidecar JSON of
original sizes so the training loader can reconstruct the per-image mm scale.

Usage:
    python scripts/preresize_aariz.py \\
        --src /teamspace/studios/this_studio/Aariz \\
        --dst /teamspace/studios/this_studio/Aariz_resized_1024 \\
        --size 1024 \\
        --workers 8
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Tuple

from PIL import Image
from PIL.Image import Resampling

Image.MAX_IMAGE_PIXELS = None  # silence DecompressionBombWarning


def _resize_one(args: Tuple[Path, Path, int]) -> Tuple[str, int, int]:
    src_path, dst_path, size = args
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src_path) as im:
        orig_w, orig_h = im.size
        im = im.convert("L")
        im = im.resize((size, size), Resampling.BICUBIC)
        im.save(dst_path, format="PNG", optimize=False, compress_level=1)
    return src_path.stem, orig_w, orig_h


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dst", type=Path, required=True)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    args.dst.mkdir(parents=True, exist_ok=True)

    # Symlink (or copy) annotations and CSV — these live in the same parent.
    for split in ("train", "valid", "test"):
        (args.dst / split).mkdir(parents=True, exist_ok=True)
        ann_src = args.src / split / "Annotations"
        ann_dst = args.dst / split / "Annotations"
        if not ann_dst.exists():
            os.symlink(ann_src, ann_dst)
    csv_dst = args.dst / "cephalogram_machine_mappings.csv"
    if not csv_dst.exists():
        os.symlink(args.src / "cephalogram_machine_mappings.csv", csv_dst)

    jobs = []
    for split in ("train", "valid", "test"):
        src_dir = args.src / split / "Cephalograms"
        dst_dir = args.dst / split / "Cephalograms"
        for p in sorted(src_dir.iterdir()):
            if p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
                continue
            jobs.append((p, dst_dir / f"{p.stem}.png", args.size))

    print(f"Resizing {len(jobs)} images with {args.workers} workers ...", flush=True)
    sizes_index: dict[str, list[int]] = {}
    t0 = time.time()
    with mp.Pool(args.workers) as pool:
        done = 0
        for stem, w, h in pool.imap_unordered(_resize_one, jobs, chunksize=4):
            sizes_index[stem] = [w, h]
            done += 1
            if done % 100 == 0:
                rate = done / (time.time() - t0 + 1e-9)
                eta = (len(jobs) - done) / max(rate, 1e-9)
                print(f"  {done}/{len(jobs)}  rate={rate:.1f}/s  eta={eta:.0f}s", flush=True)

    out_json = args.dst / "original_sizes.json"
    out_json.write_text(json.dumps(sizes_index))
    print(f"Wrote {len(sizes_index)} entries to {out_json}", flush=True)
    print(f"Done in {time.time() - t0:.1f}s.", flush=True)


if __name__ == "__main__":
    main()
