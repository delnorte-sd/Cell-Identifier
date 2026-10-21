# Interface design specification

Internal design notes for the Cell Identifier web workspace (version 1.1).

## Sources and the lessons taken from them

| Source | Lesson applied |
|---|---|
| napari viewer and image-layer docs | The canvas dominates the window. Results are **layers** stacked over the image (image → labels → outlines), each with its own visibility and opacity. A status bar reports cursor position, pixel value and the object under the cursor. Image pixels are drawn with nearest-neighbour interpolation, so zooming never invents detail. |
| Crameri, Shephard & Heron (2024), *Choosing Suitable Color Palettes for Accessible and Accurate Science Figures*, Current Protocols 4(8) | Colour must carry data, not decoration. Use colour-vision-deficiency-safe palettes (Okabe–Ito for categories), avoid rainbow maps and red/green pairs, keep the chrome neutral so the data colours stand out, and never alter the data when "enhancing" display. |
| Streamlit design docs | Not used (the stack is Flask with vanilla JS). The general idea still holds: frequent actions sit up front, and advanced settings go in disclosure sections. |

## Primary workflow

1. Open an image (button, drag and drop, or the bundled sample).
2. Analysis runs immediately with the defaults.
3. Inspect: zoom and pan, toggle layers, compare with the original, hover or click cells.
4. Adjust a setting. The results are then marked stale, and **Run analysis** (or Ctrl+Enter) re-runs it.
5. Export CSV, JSON, the overlay PNG or the figure.

The secondary workflow is **Synthetic validation**: generate an image with ground truth, analyse it, and read precision, recall and F1. The ground truth can be shown as its own layer.

## Layout (desktop ≥ 1180 px)

```
┌ header 52px: brand · workspace │ image name · status ───── mode switch · Open · Help ┐
├ controls 300px ┬ viewer (flex, dark canvas) ─────────────────────┬ results 360px ┤
│ 1 Input        │ toolbar: base · layers · opacity · compare · zoom │ summary        │
│ 2 Preprocess   │ canvas                                            │ histogram      │
│ 3 Detection    │                                                   │ cell table     │
│ 4 Filtering    │ status bar: W×H · dtype · zoom · x,y · value · ID  │ export         │
│ 5 Classes      │                                                   │                │
│ [Run analysis] │                                                   │                │
└────────────────┴───────────────────────────────────────────────────┴────────────────┘
```

* 760–1180 px: the results column moves below the viewer.
* Below 760 px: everything stacks, and the viewer is at least 60 % of the viewport height.

## Component hierarchy

* **Header:** wordmark, workspace label, current image and its dimensions, status pill (Ready / Analysing… / Stale / Error), mode segmented control, Open image, Help dialog.
* **Controls panel:** numbered `<details>` sections. Input and Detection are open by default; the rest are collapsed. Every setting has a visible label and units. The Run button is sticky at the bottom.
* **Viewer:** a `<canvas>` with a real view transform.
  * Base image: Original or Processed (the image segmentation actually saw).
  * Layers: Mask, Outlines, IDs, Ground truth (synthetic only).
  * Overlay opacity slider.
  * Compare mode: a draggable split, with the original on the left and the annotated view on the right.
  * Zoom in and out, Fit, 1:1.
* **Results panel:**
  * Six key numbers: count, coverage, mean area ± SD, mean intensity ± SD, threshold, and noise floor.
  * Brightness class counts with their legend swatches.
  * Area histogram drawn from the real cells.
  * Sortable cell table, with hover and selection synced to the viewer.
  * Export buttons.
  * In synthetic mode, an evaluation block.

## Visual tokens

| Token | Value | Use |
|---|---|---|
| `--bg` | `#F7F7F3` | app background |
| `--surface` | `#FFFFFF` | panels |
| `--text` | `#202522` | primary text (15.6:1 on bg) |
| `--text-2` | `#646A65` | secondary text (5.4:1 on bg; darkened from #737873 to pass WCAG AA) |
| `--border` | `#E1E3DC` | 1 px dividers |
| `--accent` | `#566D57` | primary button and focus (white text 5.6:1) |
| `--accent-weak` | `#E7EDE6` | selected states |
| `--viewer` | `#191C1A` | canvas background: fluorescence must sit on dark |
| `--danger` | `#9B3A2E` | errors |
| Data colours | Okabe–Ito sky blue / yellow / vermillion / orange | brightness classes, unchanged from the analysis figures |

* **Type:** system UI sans at 13 px base, with a scale of 11 / 12 / 13 / 15 / 18. Numbers use `font-variant-numeric: tabular-nums` and are right-aligned.
* **Spacing:** a 4 px grid (4, 8, 12, 16, 24).
* **Shape:** radius 6 px on controls; panels are flat and separated by borders, not shadows.
* **Motion:** none beyond 120 ms colour transitions and the busy indicator. Both respect `prefers-reduced-motion`.

## Interaction

* **Zoom:** the wheel zooms about the cursor; drag pans. Keys: `+`/`-` zoom, `F` fit, `1` = 100 %, `C` compare, `M` mask, `O` outlines.
* **Hover and selection:** hovering a cell highlights it and its table row. Clicking a cell or a row selects it, and double-clicking a row zooms to that cell.
* **Picking:** the server sends the label image encoded as 24-bit RGB, so the browser can read the exact cell ID under the cursor.
* **Stale state:** changing a setting after an analysis marks the results stale. They are never silently re-run.
* **Errors:** shown inline in the viewer and announced via an `aria-live` region. The previous results stay visible.

## Accessibility

* Landmarks: `header`, `nav`, `main`, and an `aside` for each side panel.
* Every input has a `<label>`.
* There is a visible `:focus-visible` ring.
* All controls are reachable by keyboard, and table rows are focusable.
* Status changes are announced through an `aria-live="polite"` region.
* Colour is never the only cue: brightness classes are also written out in the table.
* Contrast is at least 4.5:1 for text.
