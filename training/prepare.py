"""Prepare annotated datasets for training the learned segmentation engine.

Usage::

    python training/prepare.py --source datasets/bact_fluor --out datasets/prepared

Expects the Omnipose ``bact_fluor`` layout (``{train,test}_sorted/<group>/<name>.tif``
with matching ``<name>_masks.tif`` instance masks). Each image is rescaled so its
median cell width (from the expert masks) becomes ``TARGET_CELL_WIDTH`` pixels,
then the three training targets are computed. 15 % of the training images of
each group are held out for validation (model selection and threshold tuning);
the official test split is kept untouched for the final evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from skimage import io, measure

from cell_identifier.learned import TARGET_CELL_WIDTH, make_targets


def median_width(labels: np.ndarray) -> float:
    widths = [r.axis_minor_length for r in measure.regionprops(labels) if r.area >= 9]
    return float(np.median(widths)) if widths else TARGET_CELL_WIDTH


def rescale(image: np.ndarray, labels: np.ndarray, factor: float) -> tuple[np.ndarray, np.ndarray]:
    if abs(factor - 1) < 0.02:
        return image.astype(np.float32), labels
    h, w = labels.shape
    size = (max(32, round(w * factor)), max(32, round(h * factor)))
    interp = cv2.INTER_AREA if factor < 1 else cv2.INTER_LINEAR
    img = cv2.resize(image.astype(np.float32), size, interpolation=interp)
    lab = cv2.resize(labels.astype(np.float32), size, interpolation=cv2.INTER_NEAREST)
    return img, lab.astype(np.int32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("datasets/bact_fluor"))
    parser.add_argument("--out", type=Path, default=Path("datasets/prepared"))
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-width", type=float, default=TARGET_CELL_WIDTH)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    index: dict[str, list[dict[str, object]]] = {"train": [], "val": [], "test": []}
    for split_dir, split in (("train_sorted", "train"), ("test_sorted", "test")):
        groups = sorted(p for p in (args.source / split_dir).iterdir() if p.is_dir())
        for group in groups:
            masks = sorted(group.glob("*_masks.tif"))
            order = rng.permutation(len(masks))
            n_val = max(1, round(len(masks) * args.val_fraction)) if split == "train" else 0
            for rank, i in enumerate(order):
                mask_path = masks[i]
                target_split = "val" if rank < n_val else split
                image = io.imread(str(mask_path).replace("_masks.tif", ".tif"))
                labels = io.imread(mask_path).astype(np.int32)
                if image.ndim == 3:  # some files store a channel axis
                    image = image[..., 0] if image.shape[-1] <= 4 else image[0]
                width = median_width(labels)
                factor = args.target_width / width
                img, lab = rescale(image, labels, factor)
                targets = make_targets(lab)
                name = f"{group.name}__{mask_path.name.replace('_masks.tif', '')}"
                out = args.out / target_split / f"{name}.npz"
                out.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    out,
                    image=img.astype(np.float32),
                    labels=lab.astype(np.int32),
                    targets=targets.astype(np.float16),
                    p_low=np.float32(np.percentile(img, 1)),
                    p_high=np.float32(np.percentile(img, 99.8)),
                )
                index[target_split].append(
                    {
                        "name": name,
                        "group": group.name,
                        "source": str(mask_path),
                        "cell_width": round(width, 2),
                        "scale": round(factor, 4),
                        "shape": list(lab.shape),
                        "cells": int(lab.max()),
                    }
                )
    index["target_cell_width"] = args.target_width  # type: ignore[assignment]
    (args.out / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    for split, items in index.items():
        if not isinstance(items, list):
            continue
        print(f"{split}: {len(items)} images, {sum(int(i['cells']) for i in items)} cells")


if __name__ == "__main__":
    main()
