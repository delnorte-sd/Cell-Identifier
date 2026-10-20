"""Command-line interface: ``cell-identifier <command> ...``.

Commands
--------
analyze    Analyse one or more images or folders and write reports.
generate   Create synthetic images with ground-truth labels.
benchmark  Generate synthetic images, analyse them and score the pipeline.
config     Print the default (or effective) configuration as JSON.
serve      Run the web interface (requires the ``web`` extra).

Exit codes: 0 success, 1 one or more images failed, 2 usage/configuration error.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import __version__
from .config import (
    AnalysisConfig,
    ConfigError,
    SyntheticConfig,
    apply_overrides,
    load_config_file,
)
from .evaluation import match_labels
from .image_io import SUPPORTED_EXTENSIONS, ImageValidationError, save_image
from .pipeline import AnalysisResult, analyze
from .reporting import write_batch_summary, write_outputs
from .synthetic import generate

__all__ = ["build_parser", "main"]

logger = logging.getLogger("cell_identifier")

EXIT_OK, EXIT_FAILURES, EXIT_USAGE = 0, 1, 2


def _collect_inputs(paths: Sequence[str], recursive: bool) -> list[Path]:
    """Expand files and directories into a sorted, de-duplicated list of image files."""
    found: dict[Path, None] = {}
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            pattern = "**/*" if recursive else "*"
            for candidate in sorted(path.glob(pattern)):
                if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_EXTENSIONS:
                    found[candidate] = None
        else:
            found[path] = None
    return list(found)


def _unique_dir(base: Path, stem: str, used: set[str]) -> Path:
    name, i = stem, 2
    while name in used:
        name, i = f"{stem}_{i}", i + 1
    used.add(name)
    return base / name


def _load_analysis_config(args: argparse.Namespace) -> AnalysisConfig:
    config = load_config_file(args.config, AnalysisConfig) if args.config else AnalysisConfig()
    return apply_overrides(config, args.set or [])


def _load_synthetic_config(args: argparse.Namespace) -> SyntheticConfig:
    config = load_config_file(args.config, SyntheticConfig) if args.config else SyntheticConfig()
    shortcuts = {
        "num_cells": args.num_cells,
        "shape": args.shape,
        "seed": args.seed,
        "width": args.size,
        "height": args.size,
    }
    overrides = [f"{k}={json.dumps(v)}" for k, v in shortcuts.items() if v is not None]
    if args.overlap is not None:
        overrides.append(f"allow_overlap={json.dumps(args.overlap)}")
    return apply_overrides(config, overrides + list(args.set or []))


def _cmd_analyze(args: argparse.Namespace) -> int:
    config = _load_analysis_config(args)
    inputs = _collect_inputs(args.inputs, args.recursive)
    if not inputs:
        print("error: no input images found", file=sys.stderr)
        return EXIT_USAGE
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[AnalysisResult | tuple[str, str]] = []
    used: set[str] = set()
    failures = 0
    for path in inputs:
        try:
            result = analyze(path, config)
        except (FileNotFoundError, ImageValidationError) as exc:
            failures += 1
            rows.append((str(path), str(exc)))
            print(f"FAILED  {path}: {exc}", file=sys.stderr)
            continue
        target = _unique_dir(out, path.stem, used)
        write_outputs(result, target, images=not args.no_images)
        if args.figure:
            from .figures import save_summary_figure

            save_summary_figure(result, target / "distributions.png")
        rows.append(result)
        s = result.summary()
        print(
            f"OK      {path}: {s['cell_count']} cells, "
            f"mean area {s['area_px']['mean']} px, "
            f"coverage {s['coverage_fraction']:.1%} -> {target}"
        )
    summary_path = write_batch_summary(rows, out / "summary.csv")
    print(f"Wrote {summary_path} ({len(inputs) - failures} ok, {failures} failed)")
    return EXIT_FAILURES if failures else EXIT_OK


def _cmd_generate(args: argparse.Namespace) -> int:
    config = _load_synthetic_config(args)
    out = Path(args.output)
    for index in range(args.count):
        seed = None if config.seed is None else config.seed + index
        synthetic = generate(apply_overrides(config, [f"seed={json.dumps(seed)}"]))
        stem = f"synthetic_{index:03d}" if args.count > 1 else "synthetic"
        save_image(out / f"{stem}.png", synthetic.image)
        save_image(out / f"{stem}_labels.png", synthetic.labels.astype("uint16"))
        (out / f"{stem}_ground_truth.json").write_text(
            json.dumps(synthetic.ground_truth_dict(), indent=2) + "\n", encoding="utf-8"
        )
        print(f"{out / (stem + '.png')}: {synthetic.cell_count}/{synthetic.requested} cells placed")
    return EXIT_OK


def _cmd_benchmark(args: argparse.Namespace) -> int:
    analysis = _load_analysis_config(args)
    synthetic = _load_synthetic_config(
        argparse.Namespace(**{**vars(args), "config": None, "set": []})
    )
    base_seed = synthetic.seed if synthetic.seed is not None else 0
    rows: list[dict[str, Any]] = []
    print(f"{'seed':>5} {'true':>5} {'found':>5} {'prec':>6} {'recall':>6} {'F1':>6} {'IoU':>5}")
    for index in range(args.count):
        sample = generate(apply_overrides(synthetic, [f"seed={base_seed + index}"]))
        result = analyze(sample.image, analysis)
        m = match_labels(result.labels, sample.labels)
        rows.append({"seed": base_seed + index, **m.to_dict()})
        print(
            f"{base_seed + index:>5} {m.true_count:>5} {m.predicted_count:>5} "
            f"{m.precision:>6.3f} {m.recall:>6.3f} {m.f1:>6.3f} {m.mean_iou:>5.2f}"
        )
    mean_f1 = sum(r["f1"] for r in rows) / len(rows)
    print(f"mean F1 = {mean_f1:.4f} over {len(rows)} images")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    return EXIT_OK


def _cmd_config(args: argparse.Namespace) -> int:
    if args.synthetic:
        config: Any = (
            load_config_file(args.config, SyntheticConfig) if args.config else SyntheticConfig()
        )
        config = apply_overrides(config, args.set or [])
    else:
        config = _load_analysis_config(args)
    print(json.dumps(config.to_dict(), indent=2))
    return EXIT_OK


def _cmd_serve(args: argparse.Namespace) -> int:
    try:
        from .web import create_app
    except ImportError:
        print(
            "error: the web interface needs Flask: pip install 'cell-identifier[web]'",
            file=sys.stderr,
        )
        return EXIT_USAGE
    create_app().run(host=args.host, port=args.port, debug=args.debug)
    return EXIT_OK


def _add_config_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-c", "--config", help="TOML or JSON configuration file")
    parser.add_argument(
        "-s",
        "--set",
        action="append",
        metavar="KEY=VALUE",
        help="override one setting, e.g. --set segmentation.threshold_method=triangle (repeatable)",
    )


def _add_synthetic_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-n", "--num-cells", type=int, help="cells per image")
    parser.add_argument("--shape", choices=["pill", "ellipse"], help="cell shape")
    parser.add_argument("--seed", type=int, help="random seed (incremented per image)")
    parser.add_argument("--size", type=int, help="square image side in pixels")
    overlap = parser.add_mutually_exclusive_group()
    overlap.add_argument("--overlap", dest="overlap", action="store_true", default=None)
    overlap.add_argument("--no-overlap", dest="overlap", action="store_false")
    parser.add_argument("--count", type=int, default=1, help="number of images")


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser (exposed for documentation and testing)."""
    parser = argparse.ArgumentParser(
        prog="cell-identifier",
        description="Detect, segment and measure cells in fluorescence microscopy images.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="more logging")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    p = sub.add_parser("analyze", help="analyse images and write reports")
    p.add_argument("inputs", nargs="+", help="image files and/or directories")
    p.add_argument("-o", "--output", default="results", help="output directory")
    p.add_argument("-r", "--recursive", action="store_true", help="search directories recursively")
    p.add_argument("--no-images", action="store_true", help="only write JSON/CSV reports")
    p.add_argument(
        "--figure", action="store_true", help="also plot distributions (needs matplotlib)"
    )
    _add_config_options(p)
    p.set_defaults(func=_cmd_analyze)

    p = sub.add_parser("generate", help="create synthetic images with ground truth")
    p.add_argument("-o", "--output", default="synthetic", help="output directory")
    _add_synthetic_options(p)
    _add_config_options(p)
    p.set_defaults(func=_cmd_generate)

    p = sub.add_parser("benchmark", help="score the pipeline on synthetic ground truth")
    _add_synthetic_options(p)
    _add_config_options(p)
    p.add_argument("--json", help="also write per-image metrics to this JSON file")
    p.set_defaults(func=_cmd_benchmark, count=5)

    p = sub.add_parser("config", help="print the effective configuration as JSON")
    p.add_argument("--synthetic", action="store_true", help="show generator settings instead")
    _add_config_options(p)
    p.set_defaults(func=_cmd_config)

    p = sub.add_parser("serve", help="run the web interface")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--debug", action="store_true", help="Flask debug mode (local use only)")
    p.set_defaults(func=_cmd_serve)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``cell-identifier`` console script."""
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING - 10 * min(args.verbose, 2),
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        return int(args.func(args))
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return EXIT_USAGE


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
