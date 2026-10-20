# Methods, assumptions and limitations

This document describes exactly what Cell Identifier computes, so that results can be reported in a lab notebook, a wiki or a paper. Parameter names refer to `cell_identifier.config`.

## 1. Input

* PNG, JPEG, TIFF (first page) and BMP files are decoded in memory. 8-bit and 16-bit data are kept in their native type, and floating-point images are assumed to lie in `[0, 1]`.
* Colour images are converted to RGB order, and alpha channels are dropped.
* Images must be at least 8 × 8 pixels and at most 100 megapixels.
* Every result records the input file's SHA-256 hash.

**Full scale.** Intensity-type parameters are expressed as fractions of full scale: 255 for 8-bit, 65535 for 16-bit, and 1.0 for float. This is why one configuration serves both bit depths, and the test suite checks that 8-bit and 16-bit versions of the same image give identical segmentations.

## 2. Preprocessing (segmentation image only)

1. **Channel** (`preprocess.channel`): luminance `0.299 R + 0.587 G + 0.114 B` (the default, identical to OpenCV), a single channel, or the per-pixel maximum.
2. **Scaling** to a fraction of full scale.
3. **Denoising** (`denoise`): Gaussian, σ = `denoise_sigma` px (default 1), or a median filter.
4. **Background correction** (`background`, off by default):
   * `tophat` subtracts a grey-level opening with a disk of radius `background_radius`. This is an approximation of the rolling-ball algorithm.
   * `gaussian` subtracts a Gaussian blur with σ = `background_radius`.

   Either way, the radius must exceed the radius of the largest cell, or cells are partially removed.
5. **Normalisation** (`normalize`, off by default): linear stretch between two percentiles.

**Intensity measurements never use this image.** They use the raw channel from step 1 in native units.

## 3. Segmentation

1. **Threshold** (`threshold_method`):
   * `otsu` (default) or `triangle`: computed by scikit-image on a 1024-bin histogram.
   * `fixed`: `fixed_threshold`.
   * `adaptive`: box-filter local mean over `adaptive_block_size` px, plus `adaptive_offset`.
2. **Noise floor** (`min_snr`, default 3). σ is estimated as 1.4826 × MAD of `image − box_mean(image, adaptive_block_size)`; for Gaussian noise, this scaled MAD equals σ.
   * Global methods use `threshold = max(threshold, median(image) + min_snr·σ)`.
   * The adaptive method requires each pixel to exceed its local mean by at least `max(adaptive_offset, min_snr·σ)`.

   Because the local mean is subtracted first, linear illumination gradients do not inflate σ (verified in the tests). Densely packed bright cells bias σ slightly upwards, which makes the floor conservative.
3. **Morphology:**
   * Opening with a disk of radius `opening_radius` (default 1 px) removes isolated pixels.
   * Optional closing (`closing_radius`) bridges small gaps.
   * Enclosed holes are filled (`fill_holes`).
4. **Splitting touching cells** (`split_method`):
   * Seeds are the regional maxima of the morphological reconstruction of `f − h` under `f`, i.e. the h-maxima of `f`, using 8-connectivity.
   * A marker-controlled watershed (scikit-image) on `−f`, restricted to the foreground mask, assigns every foreground pixel to exactly one seed. No foreground pixel is lost.
   * `intensity` (default): `f` = the processed image and `h = intensity_h` (fraction of full scale). Boundaries follow intensity valleys.
   * `distance`: `f` = the Euclidean distance transform of the mask and `h = watershed_h` (pixels). Boundaries follow geometric necks.

## 4. Measurement and filtering

* **Objects kept:** those with `min_area ≤ area ≤ max_area`. With `exclude_border`, objects touching the image edge are dropped.
* **IDs:** cells are renumbered 1..N in raster order of their centroids (row first, then column).
* **Shape:** computed with `skimage.measure.regionprops`:
  * Area (pixel count) and perimeter (4-connected, corner-weighted estimate).
  * Equivalent diameter `√(4·area/π)`.
  * Major and minor axes and eccentricity of the ellipse with the same second central moments.
  * Solidity (area / convex-hull area). Scikit-image's convex hull includes pixel corners, so a perfect digital disk scores about 0.96.
  * Circularity `4π·area/perimeter²`, clipped at 1 because perimeter estimates of tiny objects can push it slightly above.
  * Orientation: angle of the major axis in degrees in `[0, 180)`, measured from +x towards +y. Image rows increase downwards, so on screen this is clockwise.
* **Intensity:** measured on the raw channel, in native units:
  * Mean, maximum and integrated (sum over the cell's pixels) intensity.
  * Background-corrected mean: the mean minus the median of all pixels outside the foreground mask.
* **Physical units:** with `pixel_size_um`, areas are multiplied by `pixel_size_um²` and lengths by `pixel_size_um`, and densities are reported per mm².
* **Summary statistics:** standard deviations are sample standard deviations (`ddof = 1`) and are reported as `null` for fewer than two cells.

## 5. Brightness classes

* **`absolute`** (default): a cell is *Low* if its mean intensity is `< low_cutoff × full scale`, *High* if it is `≥ high_cutoff × full scale`, and *Medium* otherwise. The defaults (142/255 and 156/255) are the tertiles of the bundled E. coli GFP sample under default settings. They are only meaningful for comparable acquisitions and **must be re-calibrated** for each microscope, objective, exposure and strain.
* **`quantile`:** cut-offs are the 1/3 and 2/3 quantiles of the cells in each image. Classes are relative to the image and cannot be compared across images.

## 6. Synthetic validation data

The generator (`cell_identifier.synthetic`) is isolated from the analysis code: the pipeline never imports it.

* **Cell shapes:** capsules (rods) or ellipses, with sizes, orientations and peak brightness drawn uniformly from configurable ranges by a seeded `numpy.random.Generator`.
* **Brightness profile:** each cell's brightness falls off quadratically from its centre line by `edge_falloff`, and overlapping cells add.
* **Image formation:** the field is `(background + Σ cells) × illumination`. It is blurred by a Gaussian point-spread function, then additive Gaussian detector noise is applied (after the blur, as in a camera). The result is clipped, tinted with the fluorophore colour and quantised to 8 bits.
* **Overlap rules:** when overlap is forbidden, cells must keep `min_separation` px apart. When it is allowed, at most `max_overlap_fraction` of a new cell may cover existing cells.
* **Ground truth:** an `int32` label image (the top-most cell owns shared pixels) plus every cell's generating parameters.

**Evaluation** (`cell_identifier.evaluation`) matches predicted and true objects when IoU > 0.5. This matching is necessarily one-to-one, so it is exact. It reports precision, recall, F1, mean matched IoU and count error.

### Reference benchmark (default analysis settings, 60 cells, 512 × 512 px)

| Synthetic scene | Mean F1 (seeds 0–4) |
|---|---|
| Rods, separated | 1.000 |
| Ellipses, separated | 1.000 |
| Rods, overlapping (≤ 30 % overlap) | 0.976 |
| Ellipses, overlapping (≤ 30 % overlap) | 0.965 |

To reproduce: `cell-identifier benchmark --count 5 --seed 0 [--shape ellipse] [--overlap]`.

## 7. Known limitations

* **Uniform touching cells.** Cells of uniform brightness that touch along their whole length have neither an intensity valley nor a geometric neck, so neither split method can separate them.
* **Clumps.** Heavily overlapping clumps (> 30 % overlap) are under-segmented. Filter them with `max_area` and `solidity` in downstream analysis.
* **Crowded fields.** The noise floor and the Otsu threshold assume that cells cover less than about half of the field of view. For confluent images, use `fixed` or `adaptive` thresholds with `min_snr = 0`.
* **Brightness cut-offs.** Absolute cut-offs carry no biological meaning until they are calibrated, for example against known expression levels or controls.
* **Synthetic realism.** Synthetic images are a validation tool, not a substitute for checking real images by eye. They do not model out-of-focus cells, debris, phase artefacts or photobleaching.
