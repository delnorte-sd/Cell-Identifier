"""Copy a trained, evaluated model into the package with self-describing metadata.

Usage::

    python training/bundle.py --run runs/unet12w8

Reads ``<run>/cell_unet.onnx``, ``<run>/cell_unet.json`` (written by
``train.py``), ``<run>/evaluation.json`` (written by ``evaluate.py``) and
``<run>/history.json``, and writes ``src/cell_identifier/models/cell_unet.onnx``
plus ``cell_unet.json``. The JSON records the working cell width, the
validation-tuned post-processing thresholds, the held-out test scores and the
training-data attribution; the package and the web documentation read it, so
the numbers they show are always those of the shipped model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

DEST = Path(__file__).resolve().parents[1] / "src" / "cell_identifier" / "models"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()

    onnx = args.run / "cell_unet.onnx"
    meta = json.loads((args.run / "cell_unet.json").read_text(encoding="utf-8"))
    evaluation = json.loads((args.run / "evaluation.json").read_text(encoding="utf-8"))
    history = json.loads((args.run / "history.json").read_text(encoding="utf-8"))
    test = evaluation["test"]

    metadata = {
        "name": "cell_unet",
        "architecture": f"U-Net, 4 levels, base width {meta['base_channels']}, 3 sigmoid outputs",
        "outputs": ["foreground", "boundary", "normalised_distance"],
        "target_cell_width": meta["target_cell_width"],
        "post_processing": {
            "prob_threshold": evaluation["prob_threshold"],
            "seed_threshold": evaluation["seed_threshold"],
            "min_confidence": evaluation["min_confidence"],
        },
        "training": {
            "iterations": int(history["args"]["iters"]),
            "batch_size": int(history["args"]["batch"]),
            "best_validation": max(history["history"], key=lambda h: h["ap50"] + h["ap75"]),
            "synthetic_images": int(history["args"].get("synthetic", 0)),
        },
        "test_scores": {
            engine: {group: {k: round(v, 4) for k, v in s.items()} for group, s in groups.items()}
            for engine, groups in test.items()
        },
        "data": {
            "name": "Omnipose bact_fluor (fluorescence bacteria, expert instance masks)",
            "url": "https://osf.io/xmury/",
            "license": "CC BY-NC 3.0",
            "citation": (
                "Cutler, K. J. et al. Omnipose: a high-precision morphology-independent "
                "solution for bacterial cell segmentation. Nature Methods 19, 1438-1448 (2022)."
            ),
            "split": "122 train / 21 validation / 75 test images (test never used for tuning)",
        },
        "sha256": hashlib.sha256(onnx.read_bytes()).hexdigest(),
        "bundled_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    DEST.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(onnx, DEST / "cell_unet.onnx")
    (DEST / "cell_unet.json").write_text(json.dumps(metadata, indent=1) + "\n", encoding="utf-8")
    size = (DEST / "cell_unet.onnx").stat().st_size / 1e6
    print(f"bundled {onnx} ({size:.1f} MB) -> {DEST}")


if __name__ == "__main__":
    main()
