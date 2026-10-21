"""Tune post-processing on validation data, then score engines on the test set.

Usage::

    python training/evaluate.py --model runs/unet24/cell_unet.onnx --out runs/unet24

1. Threshold tuning: the probability and seed thresholds are grid-searched on
   the *validation* images (never on test images).
2. Test evaluation: every test image is analysed at its original resolution
   through the public :func:`cell_identifier.analyze` pipeline with
   (a) the classical engine, (b) the learned engine with automatic cell-width
   estimation and (c) the learned engine told the true median cell width.
   Scores are Cellpose-style average precision, AP = TP / (TP + FP + FN), at
   IoU 0.5 / 0.75 / 0.9, plus F1 at IoU 0.5.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from skimage import io, measure

sys.path.insert(0, str(Path(__file__).parent))
from cell_identifier import AnalysisConfig, analyze, match_labels
from cell_identifier.config import apply_overrides
from cell_identifier.evaluation import average_precision
from cell_identifier.learned import LearnedModel, normalize_input, postprocess
from train import load_split


def tune(
    model: LearnedModel, items: list[dict[str, np.ndarray]]
) -> tuple[float, float, float, list]:
    cache = [(model.predict(normalize_input(it["image"])), it["labels"]) for it in items]
    grid = []
    for prob in (0.35, 0.45, 0.55):
        for seed in (0.4, 0.5, 0.6):
            for conf in (0.0, 0.6, 0.7):
                labelled = [(postprocess(p, prob, seed, 4, conf), y) for p, y in cache]
                ap50 = np.mean([average_precision(a, y, 0.5) for a, y in labelled])
                ap75 = np.mean([average_precision(a, y, 0.75) for a, y in labelled])
                grid.append((float((ap50 + ap75) / 2), prob, seed, conf, float(ap50), float(ap75)))
    grid.sort(reverse=True)
    return grid[0][1], grid[0][2], grid[0][3], grid[:5]


def median_width(labels: np.ndarray) -> float:
    widths = [r.axis_minor_length for r in measure.regionprops(labels) if r.area >= 9]
    return float(np.median(widths))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("datasets/prepared"))
    parser.add_argument("--source", type=Path, default=Path("datasets/bact_fluor"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--skip-classical", action="store_true")
    args = parser.parse_args()

    model = LearnedModel(args.model)
    prob, seed, conf, top = tune(model, load_split(args.data, "val"))
    print(f"tuned on validation: prob={prob} seed={seed} min_confidence={conf}")
    for row in top:
        print(
            "   score {:.3f}  prob {:.2f} seed {:.2f} conf {:.2f}  AP50 {:.3f} AP75 {:.3f}".format(
                *row
            )
        )

    base = apply_overrides(
        AnalysisConfig(),
        [
            "segmentation.engine=classical",
            "filter.min_area=5",
            "filter.max_area=1000000",
            "classification.method=none",
        ],
    )
    learned = apply_overrides(
        base,
        [
            "segmentation.engine=learned",
            f"learned.model={json.dumps(str(args.model))}",
            f"learned.prob_threshold={prob}",
            f"learned.seed_threshold={seed}",
            f"learned.min_confidence={conf}",
        ],
    )
    engines = {"learned (auto width)": learned}
    if not args.skip_classical:
        engines = {"classical": base, **engines}

    scores: dict[str, dict[str, list]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    timing: dict[str, list[float]] = collections.defaultdict(list)
    for mask_path in sorted((args.source / "test_sorted").glob("*/*_masks.tif")):
        group = mask_path.parent.name
        image = io.imread(str(mask_path).replace("_masks.tif", ".tif"))
        truth = io.imread(mask_path).astype(np.int32)
        width = median_width(truth)
        runs = dict(engines)
        runs["learned (true width)"] = replace(
            learned, learned=replace(learned.learned, cell_width_px=width)
        )
        for name, config in runs.items():
            t0 = time.perf_counter()
            result = analyze(image, config)
            timing[name].append(time.perf_counter() - t0)
            row = [average_precision(result.labels, truth, t) for t in (0.5, 0.75, 0.9)]
            row.append(match_labels(result.labels, truth).f1)
            scores[name][group].append(row)
            scores[name]["ALL"].append(row)

    report: dict[str, object] = {
        "model": str(args.model),
        "prob_threshold": prob,
        "seed_threshold": seed,
        "min_confidence": conf,
        "validation_top": top,
        "test": {},
    }
    groups = [*sorted({g for s in scores.values() for g in s if g != "ALL"}), "ALL"]
    print(
        f"\n{'engine':24s} {'group':8s} {'n':>3s} {'AP50':>6s} {'AP75':>6s} {'AP90':>6s} {'F1':>6s}"
    )
    for name, by_group in scores.items():
        report["test"][name] = {}  # type: ignore[index]
        for group in groups:
            rows = np.array(by_group[group])
            mean = rows.mean(axis=0)
            report["test"][name][group] = {  # type: ignore[index]
                "n": len(rows),
                "ap50": mean[0],
                "ap75": mean[1],
                "ap90": mean[2],
                "f1": mean[3],
            }
            cols = " ".join(f"{v:6.3f}" for v in mean)
            print(f"{name:24s} {group:8s} {len(rows):3d} {cols}")
        print(f"{'':24s} mean time per image {np.mean(timing[name]):.2f} s")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "evaluation.json").write_text(
        json.dumps(report, indent=1, default=float), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
