# Changelog

## 1.1.0 — Web workspace redesign

### Added
- Viewer-centred web workspace: the canvas viewer has real zoom (about the cursor), pan, fit and 1:1, with
  nearest-neighbour magnification so pixels are never interpolated.
- Independently composited layers from the API: original, processed (segmentation input), transparent label
  mask, transparent outlines and ground truth, plus an overlay opacity control.
- Draggable split comparison (original vs. annotated).
- Exact hover picking from a 24-bit label-ID image: the status bar shows coordinates, the pixel value and the cell
  under the cursor; click to select a cell and see all its measurements.
- Sortable, keyboard-navigable cell table, an area histogram of the real cells, and brightness-class
  proportions, all linked to the viewer selection.
- Settings grouped by pipeline stage, with background correction, noise floor, split sensitivity and pixel size
  now exposed. Results are marked stale when settings change; Ctrl+Enter re-runs.
- Validate mode applies the current analysis settings to seeded synthetic images and reports
  precision/recall/F1.
- Keyboard shortcuts, a help dialog, `aria-live` status, visible focus, and responsive layouts for laptop,
  tablet and phone.
- `docs/UI_DESIGN.md` design specification; `render_outline_layer`, `render_label_layer`,
  `encode_label_ids` and `render_processed` in the visualisation module.

### Changed
- The calm neutral interface uses an accessible sage accent (WCAG AA contrast) and Okabe–Ito data colours. The
  Font Awesome CDN dependency was removed in favour of a small inline icon set.
- `/api/analyze` responses include `image_info` and `layers`; `/api/generate` with `evaluate` returns the full
  analysis payload plus a ground-truth layer. Existing response keys are unchanged.

## 1.0.0 — Professional overhaul

### Added
- Installable `cell_identifier` package (src layout, `pyproject.toml`) with typed, documented modules for
  configuration, I/O, preprocessing, segmentation, measurement, classification, visualisation, reporting,
  synthetic generation and evaluation.
- `cell-identifier` CLI: `analyze` (files, folders, batch summary, config files, `--set` overrides),
  `generate`, `benchmark`, `config`, `serve`.
- Validated configuration dataclasses with TOML/JSON loading; every former magic number is now a documented
  parameter.
- Robust noise floor so empty or noise-only images yield zero cells.
- Marker-controlled watershed (h-maxima seeds) that separates touching cells by intensity valleys or shape necks.
- Shape metrics (perimeter, axes, orientation, eccentricity, circularity, solidity), integrated and
  background-corrected intensity, physical units, stable raster-order IDs, border filtering.
- Ground-truth synthetic generator (seeded, exact label images, overlap control, PSF blur, detector noise,
  illumination gradient) and IoU-based precision/recall/F1 evaluation.
- Self-describing JSON reports (schema version, software version, input SHA-256, full configuration),
  per-cell CSV, batch summary CSV, overlays, 16-bit label images, comparison figures, optional matplotlib plots.
- 161-test pytest suite (97 % branch coverage), ruff, mypy and GitHub Actions CI.
- Web UI: threshold/split/channel/border controls, CSV/JSON/PNG downloads, on-the-fly ground-truth scoring.

### Changed
- Thresholding defaults to Otsu with a noise floor instead of a fixed grey level of 50, which missed most
  dim cells (e.g. 4–17 of 60 synthetic ellipses were found by the old pipeline).
- The 5 × 5 morphological opening that erased thin rods was replaced by a 1 px opening.
- Brightness cut-offs are configurable fractions of full scale; defaults were re-calibrated on the bundled
  sample for the new segmentation.
- 16-bit images are analysed at full precision instead of being truncated to 8 bits.
- `requirements.txt` now installs the package; unused heavy dependencies (scikit-learn, matplotlib in core,
  duplicate `opencv-contrib-python`) were removed.

### Fixed
- Path traversal via uploaded filenames (uploads are now decoded in memory and never written to disk).
- Uploaded files left on disk when analysis failed.
- `/demo_analysis` crashed on colour images (`findContours` on a 3-channel array); replaced by
  `/api/generate` with `evaluate: true`.
- Shared mutable analyser state across concurrent web requests.
- Flask debug mode enabled on `0.0.0.0` when run directly.
- Synthetic generator was not reproducible (unseeded global RNGs) and produced inconsistent colour order.
- Web form allowed `max=50` cells with a default of 125.
