/* Cell-Identifier workspace.
 *
 * A canvas viewer with a real view transform (zoom about the cursor, pan, fit,
 * 1:1), independently composited layers returned by /api/analyze (original,
 * processed, mask, outlines, ground truth), a draggable split comparison, and
 * exact hover picking from a 24-bit label-ID image. No framework, no build step.
 */
(() => {
    "use strict";

    const $ = (sel, root = document) => root.querySelector(sel);
    const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

    const CLASS_COLORS = { Low: "#56B4E9", Medium: "#F0E442", High: "#D55E00", Unclassified: "#E69F00" };
    const MIN_ZOOM = 0.05;
    const MAX_ZOOM = 40;

    const els = {
        app: $("#app"),
        form: $("#settings"),
        fileInput: $("#fileInput"),
        runBtn: $("#runBtn"),
        runLabel: $("#runLabel"),
        stage: $("#stage"),
        canvas: $("#canvas"),
        split: $("#split"),
        splitHandle: $("#splitHandle"),
        emptyState: $("#emptyState"),
        emptySynthetic: $("#emptySynthetic"),
        busy: $("#busy"),
        busyText: $("#busyText"),
        dropHint: $("#dropHint"),
        viewerError: $("#viewerError"),
        viewerErrorText: $("#viewerErrorText"),
        status: $("#status"),
        statusText: $("#statusText"),
        imageName: $("#imageName"),
        imageMeta: $("#imageMeta"),
        inputSummary: $("#inputSummary"),
        zoomReadout: $("#zoomReadout"),
        opacity: $("#opacity"),
        compareBtn: $("#compareBtn"),
        sb: { dims: $("#sbDims"), zoom: $("#sbZoom"), pos: $("#sbPos"), value: $("#sbValue"), cell: $("#sbCell") },
        resultsEmpty: $("#resultsEmpty"),
        resultsBody: $("#resultsBody"),
        summary: $("#summaryMetrics"),
        staleTag: $("#staleTag"),
        classBlock: $("#classBlock"),
        classList: $("#classList"),
        histogram: $("#histogram"),
        evalBlock: $("#evalBlock"),
        evalMetrics: $("#evalMetrics"),
        evalNote: $("#evalNote"),
        cellDetail: $("#cellDetail"),
        cellDetailId: $("#cellDetailId"),
        cellDetailList: $("#cellDetailList"),
        cellTable: $("#cellTable"),
        cellCount: $("#cellCount"),
        help: $("#helpDialog"),
    };
    const ctx = els.canvas.getContext("2d");

    const state = {
        mode: "analyze",
        source: null,            // {kind: "file", file} | {kind: "sample"}
        busy: false,
        stale: false,
        data: null,              // last successful payload
        images: {},              // layer name -> HTMLImageElement
        pick: null,              // Uint8ClampedArray of the 24-bit ID image
        pixels: null,            // Uint8ClampedArray of the original image (RGBA)
        width: 0,
        height: 0,
        view: { scale: 1, tx: 0, ty: 0, fitted: true },
        base: "original",
        layers: { outlines: true, mask: false, ids: false, truth: false },
        opacity: 0.9,
        compare: false,
        split: 0.5,
        hoverId: 0,
        selectedId: 0,
        cellsById: new Map(),
        sort: { key: "cell_id", dir: 1 },
        seed: null,
    };

    // ------------------------------------------------------------------ status

    function setStatus(kind, text) {
        els.status.dataset.state = kind;
        els.statusText.textContent = text;
    }

    function setBusy(on, text = "Analysing…") {
        state.busy = on;
        els.busy.hidden = !on;
        els.busyText.textContent = text;
        els.runBtn.disabled = on || !canRun();
        if (on) setStatus("busy", text);
    }

    function showError(message) {
        els.viewerErrorText.textContent = message;
        els.viewerError.hidden = false;
        setStatus("error", "Error");
    }

    function clearError() {
        els.viewerError.hidden = true;
    }

    function canRun() {
        return state.mode === "synthetic" || state.source !== null;
    }

    function markStale() {
        if (!state.data || state.busy) return;
        state.stale = true;
        els.staleTag.hidden = false;
        els.runBtn.classList.add("is-stale");
        setStatus("stale", "Settings changed — run again");
    }

    function clearStale() {
        state.stale = false;
        els.staleTag.hidden = true;
        els.runBtn.classList.remove("is-stale");
    }

    // ------------------------------------------------------------------ settings

    const DEFAULTS = new Map();
    $$("input, select", els.form).forEach((el) => {
        DEFAULTS.set(el.name, el.type === "checkbox" ? el.checked : el.value);
    });

    function syncOutputs() {
        $$("output[data-for]", els.form).forEach((out) => {
            const input = els.form.elements[out.dataset.for];
            if (input) out.textContent = `${input.value}${out.dataset.unit || ""}`;
        });
        $$("[data-show-if]", els.form).forEach((el) => {
            // Format: "name=value" or "name!=value".
            const [, name, op, value] = el.dataset.showIf.match(/^(\w+)(!?=)(.*)$/);
            const current = els.form.elements[name]?.value;
            el.hidden = op === "!=" ? current === value : current !== value;
        });
    }

    function analysisParams() {
        const f = els.form.elements;
        const params = {
            channel: f.channel.value,
            smoothing_sigma: f.smoothing_sigma.value,
            background: f.background.value,
            background_radius: f.background_radius.value,
            threshold_method: f.threshold_method.value,
            manual_threshold: f.manual_threshold.value,
            min_snr: f.min_snr.value,
            split_method: f.split_method.value,
            intensity_h: f.intensity_h.value,
            watershed_h: f.watershed_h.value,
            min_cell_area: f.min_cell_area.value,
            max_cell_area: f.max_cell_area.value,
            exclude_border: f.exclude_border.checked,
            classification: f.classification.value,
        };
        if (f.pixel_size_um.value) params.pixel_size_um = f.pixel_size_um.value;
        return params;
    }

    function validateSettings() {
        const f = els.form.elements;
        const min = Number(f.min_cell_area.value);
        const max = Number(f.max_cell_area.value);
        if (!(min >= 1)) return "Min area must be at least 1 px².";
        if (!(max >= min)) return "Max area must be greater than or equal to min area.";
        if (state.mode === "synthetic") {
            const n = Number(f.num_cells.value);
            if (!(n >= 1 && n <= 500)) return "Number of cells must be between 1 and 500.";
        }
        return null;
    }

    // ------------------------------------------------------------------ requests

    async function postJSON(url, body) {
        const response = await fetch(url, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        });
        return readResponse(response);
    }

    async function readResponse(response) {
        let data;
        try {
            data = await response.json();
        } catch {
            throw new Error(`server returned HTTP ${response.status}`);
        }
        if (!response.ok || !data.success) throw new Error(data.error || `HTTP ${response.status}`);
        return data;
    }

    async function run() {
        if (state.busy || !canRun()) return;
        const problem = validateSettings();
        if (problem) {
            showError(problem);
            return;
        }
        clearError();
        const params = analysisParams();
        try {
            let data;
            if (state.mode === "synthetic") {
                setBusy(true, "Generating & analysing…");
                const f = els.form.elements;
                // Pin a seed so re-running with new analysis settings reuses the same image.
                const seed = f.seed.value !== "" ? Number(f.seed.value) : Math.floor(Math.random() * 1e6);
                data = await postJSON("/api/generate", {
                    ...params,
                    num_cells: Number(f.num_cells.value),
                    cell_shape: f.cell_shape.value,
                    noise_level: Number(f.noise_level.value),
                    allow_overlap: f.allow_overlap.checked,
                    seed,
                    evaluate: true,
                });
                f.seed.value = String(seed);
            } else if (state.source.kind === "sample") {
                setBusy(true);
                data = await postJSON("/api/analyze", { sample: true, ...params });
            } else {
                setBusy(true);
                const body = new FormData();
                body.append("file", state.source.file);
                Object.entries(params).forEach(([k, v]) => body.append(k, String(v)));
                data = await readResponse(await fetch("/api/analyze", { method: "POST", body }));
            }
            await showResult(data);
            clearStale();
            setStatus("ready", `${data.summary.cell_count} cells detected`);
        } catch (err) {
            showError(err.message || String(err));
        } finally {
            setBusy(false);
        }
    }

    // ------------------------------------------------------------------ loading results

    function loadImage(src) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = () => resolve(img);
            img.onerror = () => reject(new Error("could not decode a result layer"));
            img.src = src;
        });
    }

    function readPixels(img) {
        const c = document.createElement("canvas");
        c.width = img.naturalWidth;
        c.height = img.naturalHeight;
        const g = c.getContext("2d", { willReadFrequently: true });
        g.drawImage(img, 0, 0);
        return g.getImageData(0, 0, c.width, c.height).data;
    }

    async function showResult(data) {
        const names = Object.keys(data.layers);
        const loaded = await Promise.all(names.map((n) => loadImage(data.layers[n])));
        const images = Object.fromEntries(names.map((n, i) => [n, loaded[i]]));
        const sameSize = state.width === data.image_info.width && state.height === data.image_info.height;

        state.data = data;
        state.images = images;
        state.width = data.image_info.width;
        state.height = data.image_info.height;
        state.pick = readPixels(images.ids);
        state.pixels = readPixels(images.original);
        state.cellsById = new Map(data.cells.map((c) => [c.cell_id, c]));
        const diameters = data.cells.map((c) => c.equivalent_diameter_px).sort((a, b) => a - b);
        state.medianDiameter = diameters.length ? diameters[Math.floor(diameters.length / 2)] : 0;
        state.hoverId = 0;
        if (!state.cellsById.has(state.selectedId)) state.selectedId = 0;
        if (!sameSize || state.view.fitted) fitView();

        els.app.dataset.state = "ready";
        $("#zoomDock").hidden = false;
        els.emptyState.hidden = true;
        els.emptySynthetic.hidden = true;
        updateHeader();
        renderResults();
        render();
    }

    function updateHeader() {
        const info = state.data?.image_info;
        if (!info) {
            els.imageName.textContent = state.source?.file?.name || (state.source?.kind === "sample" ? "sample.jpeg" : "No image");
            els.imageMeta.textContent = "";
            return;
        }
        els.imageName.textContent = info.name;
        els.imageName.title = info.name;
        const kind = info.channels === 1 ? "grey" : "RGB";
        els.imageMeta.textContent = `${info.width} × ${info.height} px · ${info.dtype} ${kind}`;
        els.sb.dims.textContent = `${info.width}×${info.height} ${info.dtype}`;
    }

    // ------------------------------------------------------------------ view transform

    function stageSize() {
        return { w: els.stage.clientWidth, h: els.stage.clientHeight };
    }

    function fitView() {
        if (!state.width) return;
        const { w, h } = stageSize();
        const pad = 24;
        const scale = Math.min((w - pad * 2) / state.width, (h - pad * 2) / state.height);
        const s = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, scale));
        state.view = { scale: s, tx: (w - state.width * s) / 2, ty: (h - state.height * s) / 2, fitted: true };
        render();
    }

    function zoomAt(factor, sx, sy) {
        if (!state.width) return;
        const v = state.view;
        const next = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, v.scale * factor));
        const k = next / v.scale;
        v.tx = sx - (sx - v.tx) * k;
        v.ty = sy - (sy - v.ty) * k;
        v.scale = next;
        v.fitted = false;
        render();
    }

    function zoomCenter(factor) {
        const { w, h } = stageSize();
        zoomAt(factor, w / 2, h / 2);
    }

    function zoomTo(scale) {
        zoomCenter(scale / state.view.scale);
    }

    function zoomToCell(id) {
        const cell = state.cellsById.get(id);
        if (!cell) return;
        const { w, h } = stageSize();
        const size = Math.max(cell.bbox_width, cell.bbox_height, 8);
        const scale = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, Math.min(w, h) / (size * 4)));
        state.view = {
            scale,
            tx: w / 2 - cell.centroid_x * scale,
            ty: h / 2 - cell.centroid_y * scale,
            fitted: false,
        };
        render();
    }

    function toImage(sx, sy) {
        const v = state.view;
        return { x: (sx - v.tx) / v.scale, y: (sy - v.ty) / v.scale };
    }

    // ------------------------------------------------------------------ rendering

    let frame = 0;
    function render() {
        if (frame) return;
        frame = requestAnimationFrame(draw);
    }

    function draw() {
        frame = 0;
        const dpr = window.devicePixelRatio || 1;
        const { w, h } = stageSize();
        if (els.canvas.width !== Math.round(w * dpr) || els.canvas.height !== Math.round(h * dpr)) {
            els.canvas.width = Math.round(w * dpr);
            els.canvas.height = Math.round(h * dpr);
        }
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, w, h);
        updateZoomReadout();
        if (!state.data) return;

        const v = state.view;
        const img = state.images;
        const splitX = state.compare ? Math.round(w * state.split) : 0;
        const base = state.base === "processed" ? img.processed : img.original;

        ctx.save();
        ctx.translate(v.tx, v.ty);
        ctx.scale(v.scale, v.scale);
        // Nearest-neighbour when magnifying, so pixels are shown, never invented.
        ctx.imageSmoothingEnabled = v.scale < 1;
        ctx.imageSmoothingQuality = "high";
        ctx.drawImage(base, 0, 0);
        ctx.restore();

        // Annotated side: overlays (everywhere, or right of the divider in compare mode).
        ctx.save();
        if (state.compare) {
            ctx.beginPath();
            ctx.rect(splitX, 0, w - splitX, h);
            ctx.clip();
        }
        ctx.translate(v.tx, v.ty);
        ctx.scale(v.scale, v.scale);
        ctx.imageSmoothingEnabled = false;
        if (state.layers.mask) {
            ctx.globalAlpha = state.opacity * 0.55;
            ctx.drawImage(img.mask, 0, 0);
        }
        if (state.layers.outlines) {
            ctx.globalAlpha = state.opacity;
            ctx.drawImage(img.outlines, 0, 0);
        }
        if (state.layers.truth && img.truth) {
            ctx.globalAlpha = state.opacity;
            ctx.drawImage(img.truth, 0, 0);
        }
        ctx.globalAlpha = 1;
        ctx.restore();

        // Original side in compare mode: always the untouched original.
        if (state.compare) {
            ctx.save();
            ctx.beginPath();
            ctx.rect(0, 0, splitX, h);
            ctx.clip();
            ctx.fillStyle = "#191C1A";
            ctx.fillRect(0, 0, splitX, h);
            ctx.translate(v.tx, v.ty);
            ctx.scale(v.scale, v.scale);
            ctx.imageSmoothingEnabled = v.scale < 1;
            ctx.drawImage(img.original, 0, 0);
            ctx.restore();
        }

        // Screen-space annotations (crisp at any zoom).
        if (state.layers.ids) drawIds(splitX, w, h);
        drawHighlight(state.hoverId, "rgba(255,255,255,0.9)", 1);
        drawHighlight(state.selectedId, "#E69F00", 2);

        if (state.compare) {
            els.splitHandle.style.left = `${splitX}px`;
        }
    }

    function drawIds(splitX, w, h) {
        const v = state.view;
        const cells = state.data.cells;
        // Skip labels when cells are too small on screen to label legibly.
        if (!cells.length || state.medianDiameter * v.scale < 9) return;
        ctx.save();
        ctx.font = "600 11px ui-sans-serif, system-ui, sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.lineWidth = 3;
        ctx.strokeStyle = "rgba(0,0,0,0.75)";
        ctx.fillStyle = "#fff";
        for (const c of cells) {
            const x = c.centroid_x * v.scale + v.tx;
            const y = c.centroid_y * v.scale + v.ty;
            if (x < splitX || x < -20 || y < -20 || x > w + 20 || y > h + 20) continue;
            ctx.strokeText(String(c.cell_id), x, y);
            ctx.fillText(String(c.cell_id), x, y);
        }
        ctx.restore();
    }

    function drawHighlight(id, color, width) {
        const c = state.cellsById.get(id);
        if (!c) return;
        const v = state.view;
        const pad = 3;
        const x = c.bbox_x * v.scale + v.tx - pad;
        const y = c.bbox_y * v.scale + v.ty - pad;
        ctx.save();
        ctx.strokeStyle = color;
        ctx.lineWidth = width;
        ctx.strokeRect(Math.round(x) + 0.5, Math.round(y) + 0.5,
            Math.round(c.bbox_width * v.scale + pad * 2), Math.round(c.bbox_height * v.scale + pad * 2));
        ctx.restore();
    }

    function updateZoomReadout() {
        const text = state.data ? `${Math.round(state.view.scale * 100)}%` : "–";
        els.zoomReadout.textContent = text;
        els.sb.zoom.textContent = state.data ? `zoom ${text}` : "";
    }

    // ------------------------------------------------------------------ results panel

    const fmt = (v, d = 1) => (v === null || v === undefined || Number.isNaN(v) ? "–" : Number(v).toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d }));
    const esc = (s) => String(s).replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[ch]);

    function metric(label, value, extra = "", lead = false) {
        return `<div class="${lead ? "metric-lead" : ""}"><dt>${esc(label)}</dt><dd>${value}${extra ? ` <small>${extra}</small>` : ""}</dd></div>`;
    }

    function renderResults() {
        const d = state.data;
        const s = d.summary;
        els.resultsEmpty.hidden = true;
        els.resultsBody.hidden = false;

        const physical = s.area_um2 && s.area_um2.mean !== null;
        els.summary.innerHTML = [
            metric("Cells", s.cell_count.toLocaleString(), s.border_cells ? `${s.border_cells} at edge` : "", true),
            metric("Coverage", fmt(s.coverage_fraction * 100, 1), "%", true),
            physical
                ? metric("Mean area", fmt(s.area_um2.mean, 2), `± ${fmt(s.area_um2.std, 2)} µm²`)
                : metric("Mean area", fmt(s.area_px.mean, 0), `± ${fmt(s.area_px.std, 0)} px²`),
            metric("Mean intensity", fmt(s.mean_intensity.mean, 1), `± ${fmt(s.mean_intensity.std, 1)}`),
            metric("Threshold", fmt(s.threshold_fraction * 255, 1), "/ 255"),
            metric("Noise floor", fmt(s.noise_floor_fraction * 255, 1), "/ 255"),
        ].join("");

        // Brightness classes: real counts, proportional bars, colour + text label.
        const counts = s.brightness_counts;
        const total = Object.values(counts).reduce((a, b) => a + b, 0);
        const method = d.report.config.classification.method;
        els.classBlock.hidden = method === "none";
        els.classList.innerHTML = ["Low", "Medium", "High"].map((name) => {
            const n = counts[name] || 0;
            const pct = total ? (n / total) * 100 : 0;
            return `<li><span class="sw" style="background:${CLASS_COLORS[name]}"></span><span>${name}</span>
                <span class="bar"><span style="width:${pct.toFixed(1)}%"></span></span><span class="num">${n}</span></li>`;
        }).join("");

        // Synthetic evaluation.
        if (d.evaluation) {
            const e = d.evaluation;
            els.evalBlock.hidden = false;
            els.evalMetrics.innerHTML = [
                metric("F1", fmt(e.f1, 3), "", true),
                metric("Precision", fmt(e.precision * 100, 1), "%", true),
                metric("Recall", fmt(e.recall * 100, 1), "%", true),
                metric("Detected / true", `${e.predicted_count} / ${e.true_count}`),
                metric("TP · FP · FN", `${e.true_positives} · ${e.false_positives} · ${e.false_negatives}`),
                metric("Mean IoU", fmt(e.mean_iou, 2)),
            ].join("");
            els.evalNote.textContent = `Matched at IoU > ${e.iou_threshold}. Seed ${d.seed}; ${d.requested} cells requested, ${d.cell_count} placed. Toggle “Ground truth” to see the true outlines.`;
        } else {
            els.evalBlock.hidden = true;
        }

        els.cellCount.textContent = d.cells.length.toLocaleString();
        renderHistogram();
        renderTable();
        renderCellDetail();
    }

    function renderHistogram() {
        const areas = state.data.cells.map((c) => c.area_px);
        if (!areas.length) {
            els.histogram.innerHTML = `<p class="hint">No cells detected.</p>`;
            return;
        }
        const W = 320, H = 96, padL = 26, padB = 18, padT = 4;
        const min = Math.min(...areas), max = Math.max(...areas);
        const bins = Math.min(24, Math.max(6, Math.ceil(Math.sqrt(areas.length))));
        const width = (max - min) / bins || 1;
        const counts = new Array(bins).fill(0);
        const binOf = (a) => Math.min(bins - 1, Math.floor((a - min) / width));
        areas.forEach((a) => counts[binOf(a)]++);
        const peak = Math.max(...counts);
        const sel = state.cellsById.get(state.selectedId);
        const selBin = sel ? binOf(sel.area_px) : -1;
        const bw = (W - padL) / bins;
        const bars = counts.map((n, i) => {
            const bh = peak ? (n / peak) * (H - padB - padT) : 0;
            return `<rect class="bar${i === selBin ? " is-selected" : ""}" x="${(padL + i * bw + 0.5).toFixed(1)}" y="${(H - padB - bh).toFixed(1)}" width="${Math.max(bw - 1, 1).toFixed(1)}" height="${bh.toFixed(1)}"><title>${Math.round(min + i * width)}–${Math.round(min + (i + 1) * width)} px²: ${n} cells</title></rect>`;
        }).join("");
        els.histogram.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Histogram of cell areas from ${min} to ${max} square pixels, peak ${peak} cells per bin">
            ${bars}
            <line class="axis" x1="${padL}" y1="${H - padB + 0.5}" x2="${W}" y2="${H - padB + 0.5}"/>
            <text x="${padL}" y="${H - 4}">${min}</text>
            <text x="${W}" y="${H - 4}" text-anchor="end">${max} px²</text>
            <text x="${padL - 4}" y="${padT + 8}" text-anchor="end">${peak}</text>
            <text x="${padL - 4}" y="${H - padB}" text-anchor="end">0</text>
        </svg>`;
    }

    function renderTable() {
        const { key, dir } = state.sort;
        const rows = [...state.data.cells].sort((a, b) => {
            const x = a[key], y = b[key];
            return (typeof x === "string" ? x.localeCompare(y) : x - y) * dir;
        });
        $$("th", els.cellTable).forEach((th) => {
            th.setAttribute("aria-sort", th.dataset.key === key ? (dir > 0 ? "ascending" : "descending") : "none");
        });
        $("tbody", els.cellTable).innerHTML = rows.map((c) => `
            <tr data-id="${c.cell_id}" tabindex="0" class="${c.cell_id === state.selectedId ? "is-selected" : ""}">
                <td>${c.cell_id}</td>
                <td>${c.area_px}</td>
                <td>${fmt(c.circularity, 2)}</td>
                <td>${fmt(c.mean_intensity, 1)}</td>
                <td><span class="sw" style="background:${CLASS_COLORS[c.brightness] || CLASS_COLORS.Unclassified}"></span>${esc(c.brightness)}</td>
            </tr>`).join("");
    }

    function renderCellDetail() {
        const c = state.cellsById.get(state.selectedId);
        els.cellDetail.hidden = !c;
        if (!c) return;
        els.cellDetailId.textContent = `#${c.cell_id}`;
        const um = c.area_um2 !== null && c.area_um2 !== undefined;
        const rows = [
            ["Centroid (x, y)", `${fmt(c.centroid_x, 1)}, ${fmt(c.centroid_y, 1)}`],
            ["Area", um ? `${c.area_px} px² · ${fmt(c.area_um2, 3)} µm²` : `${c.area_px} px²`],
            ["Perimeter", `${fmt(c.perimeter_px, 1)} px`],
            ["Major × minor axis", `${fmt(c.major_axis_px, 1)} × ${fmt(c.minor_axis_px, 1)} px`],
            ["Orientation", `${fmt(c.orientation_deg, 1)}°`],
            ["Aspect ratio", fmt(c.aspect_ratio, 2)],
            ["Circularity", fmt(c.circularity, 3)],
            ["Solidity", fmt(c.solidity, 3)],
            ["Mean intensity", fmt(c.mean_intensity, 1)],
            ["Background-corrected", fmt(c.mean_intensity_corrected, 1)],
            ["Integrated intensity", fmt(c.integrated_intensity, 0)],
            ["Brightness class", c.brightness],
            ["Touches edge", c.touches_border ? "yes" : "no"],
        ];
        els.cellDetailList.innerHTML = rows.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("");
    }

    function select(id, { scroll = true } = {}) {
        state.selectedId = id;
        $$("tbody tr", els.cellTable).forEach((tr) => {
            const on = Number(tr.dataset.id) === id;
            tr.classList.toggle("is-selected", on);
            if (on && scroll) tr.scrollIntoView({ block: "nearest" });
        });
        renderCellDetail();
        renderHistogram();
        render();
    }

    function hover(id) {
        if (id === state.hoverId) return;
        const prev = $(`tbody tr[data-id="${state.hoverId}"]`, els.cellTable);
        if (prev) prev.classList.remove("is-hover");
        state.hoverId = id;
        const row = $(`tbody tr[data-id="${id}"]`, els.cellTable);
        if (row) row.classList.add("is-hover");
        render();
    }

    // ------------------------------------------------------------------ input handling

    function openFile(file) {
        if (!file) return;
        if (state.mode !== "analyze") setMode("analyze");
        state.source = { kind: "file", file };
        state.view.fitted = true;
        state.selectedId = 0;
        els.inputSummary.textContent = `${file.name} · ${(file.size / 1024).toFixed(0)} KB`;
        els.imageName.textContent = file.name;
        els.imageMeta.textContent = "";
        run();
    }

    function useSample() {
        if (state.mode !== "analyze") setMode("analyze");
        state.source = { kind: "sample" };
        state.view.fitted = true;
        state.selectedId = 0;
        els.inputSummary.textContent = "Bundled sample · E. coli expressing GFP";
        run();
    }

    function setMode(mode) {
        state.mode = mode;
        els.app.dataset.mode = mode;
        $$(".mode-switch button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.mode === mode)));
        els.runLabel.textContent = mode === "synthetic" ? "Generate & analyze" : "Run analysis";
        // Each mode keeps its own result; switching starts clean to avoid mixing data.
        state.data = null;
        state.selectedId = 0;
        state.hoverId = 0;
        state.width = state.height = 0;
        clearStale();
        clearError();
        els.app.dataset.state = "empty";
        $("#zoomDock").hidden = true;
        els.resultsEmpty.hidden = false;
        els.resultsBody.hidden = true;
        els.emptyState.hidden = mode !== "analyze";
        els.emptySynthetic.hidden = mode !== "synthetic";
        if (mode === "synthetic") {
            state.layers.truth = true;
            $('[data-layer="truth"]').checked = true;
            els.imageName.textContent = "Synthetic image";
        } else {
            state.source = null;
            els.inputSummary.textContent = "No image selected";
            els.imageName.textContent = "No image";
        }
        els.imageMeta.textContent = "";
        els.sb.dims.textContent = "No image loaded";
        ["zoom", "pos", "value", "cell"].forEach((k) => { els.sb[k].textContent = ""; });
        els.runBtn.disabled = !canRun();
        setStatus("idle", "Ready");
        render();
    }

    function setBase(base) {
        state.base = base;
        $$("[data-base]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.base === base)));
        render();
    }

    function setLayer(name, on) {
        state.layers[name] = on;
        const box = $(`[data-layer="${name}"]`);
        if (box) box.checked = on;
        render();
    }

    function setCompare(on) {
        state.compare = on;
        els.compareBtn.setAttribute("aria-pressed", String(on));
        els.split.hidden = !on;
        render();
    }

    // ------------------------------------------------------------------ events

    els.form.addEventListener("input", () => {
        syncOutputs();
        markStale();
    });
    els.form.addEventListener("submit", (e) => e.preventDefault());

    els.runBtn.addEventListener("click", run);
    $("#resetBtn").addEventListener("click", () => {
        $$("input, select", els.form).forEach((el) => {
            if (!DEFAULTS.has(el.name)) return;
            if (el.type === "checkbox") el.checked = DEFAULTS.get(el.name);
            else el.value = DEFAULTS.get(el.name);
        });
        syncOutputs();
        markStale();
    });

    $("#openBtn").addEventListener("click", () => els.fileInput.click());
    els.fileInput.addEventListener("change", () => {
        openFile(els.fileInput.files[0]);
        els.fileInput.value = "";
    });
    document.addEventListener("click", (e) => {
        const action = e.target.closest("[data-action]")?.dataset.action;
        if (action === "open") els.fileInput.click();
        else if (action === "sample") useSample();
        else if (action === "run") run();
    });

    $$(".mode-switch button").forEach((b) => b.addEventListener("click", () => {
        if (b.dataset.mode !== state.mode) setMode(b.dataset.mode);
    }));
    $$("[data-base]").forEach((b) => b.addEventListener("click", () => setBase(b.dataset.base)));
    $$("[data-layer]").forEach((box) => box.addEventListener("change", () => setLayer(box.dataset.layer, box.checked)));
    els.opacity.addEventListener("input", () => {
        state.opacity = Number(els.opacity.value);
        render();
    });
    els.compareBtn.addEventListener("click", () => setCompare(!state.compare));
    $$("[data-zoom]").forEach((b) => b.addEventListener("click", () => {
        const z = b.dataset.zoom;
        if (z === "in") zoomCenter(1.25);
        else if (z === "out") zoomCenter(0.8);
        else if (z === "fit") fitView();
        else if (z === "one") zoomTo(1);
    }));
    $("#dismissError").addEventListener("click", () => {
        clearError();
        setStatus(state.data ? "ready" : "idle", state.data ? `${state.data.summary.cell_count} cells detected` : "Ready");
    });
    $("#helpBtn").addEventListener("click", () => els.help.showModal());
    $("#zoomToCell").addEventListener("click", () => zoomToCell(state.selectedId));

    // Table: sorting, hover, selection, keyboard.
    $$("th button", els.cellTable).forEach((btn) => btn.addEventListener("click", () => {
        const key = btn.parentElement.dataset.key;
        state.sort = { key, dir: state.sort.key === key ? -state.sort.dir : 1 };
        renderTable();
    }));
    const tbody = $("tbody", els.cellTable);
    tbody.addEventListener("click", (e) => {
        const tr = e.target.closest("tr[data-id]");
        if (tr) select(Number(tr.dataset.id), { scroll: false });
    });
    tbody.addEventListener("dblclick", (e) => {
        const tr = e.target.closest("tr[data-id]");
        if (tr) zoomToCell(Number(tr.dataset.id));
    });
    tbody.addEventListener("mouseover", (e) => {
        const tr = e.target.closest("tr[data-id]");
        if (tr) hover(Number(tr.dataset.id));
    });
    tbody.addEventListener("mouseleave", () => hover(0));
    tbody.addEventListener("keydown", (e) => {
        const tr = e.target.closest("tr[data-id]");
        if (!tr) return;
        if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            select(Number(tr.dataset.id), { scroll: false });
            zoomToCell(Number(tr.dataset.id));
        } else if (e.key === "ArrowDown" || e.key === "ArrowUp") {
            e.preventDefault();
            const next = e.key === "ArrowDown" ? tr.nextElementSibling : tr.previousElementSibling;
            if (next) {
                next.focus();
                select(Number(next.dataset.id), { scroll: false });
            }
        }
    });

    // Exports.
    function download(name, content, type) {
        const blob = content instanceof Blob ? content : new Blob([content], { type });
        const a = document.createElement("a");
        a.href = URL.createObjectURL(blob);
        a.download = name;
        document.body.appendChild(a);
        a.click();
        a.remove();
        setTimeout(() => URL.revokeObjectURL(a.href), 1000);
    }
    function stem() {
        const name = state.data?.image_info?.name || "cells";
        return name.replace(/\.[^.]+$/, "").replace(/[^\w.-]+/g, "_");
    }
    $$("[data-export]").forEach((b) => b.addEventListener("click", async () => {
        const d = state.data;
        if (!d) return;
        const kind = b.dataset.export;
        if (kind === "csv") download(`${stem()}_cells.csv`, d.csv, "text/csv");
        else if (kind === "json") download(`${stem()}_results.json`, JSON.stringify(d.report, null, 2), "application/json");
        else {
            const url = kind === "overlay" ? d.result_image : d.figure_image;
            download(`${stem()}_${kind}.png`, await (await fetch(url)).blob());
        }
    }));

    // Canvas: zoom, pan, hover picking, selection.
    els.canvas.addEventListener("wheel", (e) => {
        if (!state.data) return;
        e.preventDefault();
        const r = els.canvas.getBoundingClientRect();
        const factor = Math.exp(-Math.max(-100, Math.min(100, e.deltaY)) * 0.0025);
        zoomAt(factor, e.clientX - r.left, e.clientY - r.top);
    }, { passive: false });

    let drag = null;
    els.canvas.addEventListener("pointerdown", (e) => {
        if (!state.data || e.button !== 0) return;
        drag = { x: e.clientX, y: e.clientY, tx: state.view.tx, ty: state.view.ty, moved: false };
        els.canvas.setPointerCapture(e.pointerId);
    });
    els.canvas.addEventListener("pointermove", (e) => {
        if (!state.data) return;
        if (drag) {
            const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
            if (Math.abs(dx) + Math.abs(dy) > 3) {
                drag.moved = true;
                els.stage.classList.add("is-panning");
            }
            if (drag.moved) {
                state.view.tx = drag.tx + dx;
                state.view.ty = drag.ty + dy;
                state.view.fitted = false;
                render();
            }
        }
        updateCursorInfo(e);
    });
    els.canvas.addEventListener("pointerup", (e) => {
        if (drag && !drag.moved) {
            const id = idAt(e);
            select(id === state.selectedId ? 0 : id);
        }
        drag = null;
        els.stage.classList.remove("is-panning");
    });
    els.canvas.addEventListener("pointerleave", () => {
        if (!state.data) return;
        hover(0);
        els.sb.pos.textContent = "x – y –";
        els.sb.value.textContent = "–";
        els.sb.cell.textContent = "–";
    });
    els.canvas.addEventListener("dblclick", (e) => {
        const id = idAt(e);
        if (id) zoomToCell(id);
    });

    function pixelAt(e) {
        const r = els.canvas.getBoundingClientRect();
        const p = toImage(e.clientX - r.left, e.clientY - r.top);
        const x = Math.floor(p.x), y = Math.floor(p.y);
        if (x < 0 || y < 0 || x >= state.width || y >= state.height) return null;
        return { x, y, i: (y * state.width + x) * 4 };
    }

    function idAt(e) {
        const p = pixelAt(e);
        if (!p || !state.pick) return 0;
        return state.pick[p.i] + state.pick[p.i + 1] * 256 + state.pick[p.i + 2] * 65536;
    }

    function updateCursorInfo(e) {
        const p = pixelAt(e);
        if (!p) {
            hover(0);
            els.sb.pos.textContent = "x – y –";
            els.sb.value.textContent = "–";
            els.sb.cell.textContent = "–";
            return;
        }
        const id = state.pick[p.i] + state.pick[p.i + 1] * 256 + state.pick[p.i + 2] * 65536;
        hover(id);
        els.sb.pos.textContent = `x ${p.x} y ${p.y}`;
        const px = state.pixels;
        els.sb.value.textContent = state.data.image_info.channels === 1
            ? `value ${px[p.i]}`
            : `RGB ${px[p.i]} ${px[p.i + 1]} ${px[p.i + 2]}`;
        const cell = state.cellsById.get(id);
        els.sb.cell.textContent = cell ? `cell ${id} · ${cell.area_px} px² · ${cell.brightness}` : "background";
    }

    // Split handle dragging.
    els.splitHandle.addEventListener("pointerdown", (e) => {
        e.preventDefault();
        els.splitHandle.setPointerCapture(e.pointerId);
        const move = (ev) => {
            const r = els.stage.getBoundingClientRect();
            state.split = Math.max(0.02, Math.min(0.98, (ev.clientX - r.left) / r.width));
            render();
        };
        const up = () => {
            els.splitHandle.removeEventListener("pointermove", move);
            els.splitHandle.removeEventListener("pointerup", up);
        };
        els.splitHandle.addEventListener("pointermove", move);
        els.splitHandle.addEventListener("pointerup", up);
    });
    els.splitHandle.addEventListener("keydown", (e) => {
        if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
            e.preventDefault();
            state.split = Math.max(0.02, Math.min(0.98, state.split + (e.key === "ArrowLeft" ? -0.05 : 0.05)));
            render();
        }
    });

    // Drag and drop anywhere on the viewer.
    let dragDepth = 0;
    els.stage.addEventListener("dragenter", (e) => {
        if (![...e.dataTransfer.types].includes("Files")) return;
        e.preventDefault();
        dragDepth++;
        els.dropHint.hidden = false;
    });
    els.stage.addEventListener("dragover", (e) => e.preventDefault());
    els.stage.addEventListener("dragleave", () => {
        dragDepth = Math.max(0, dragDepth - 1);
        if (!dragDepth) els.dropHint.hidden = true;
    });
    els.stage.addEventListener("drop", (e) => {
        e.preventDefault();
        dragDepth = 0;
        els.dropHint.hidden = true;
        const file = e.dataTransfer.files[0];
        if (file) openFile(file);
    });

    // Keyboard shortcuts (ignored while typing in a field).
    document.addEventListener("keydown", (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
            e.preventDefault();
            run();
            return;
        }
        const t = e.target;
        if (t.closest("input, select, textarea, dialog") || e.ctrlKey || e.metaKey || e.altKey) return;
        const key = e.key.toLowerCase();
        if (key === "?") els.help.showModal();
        if (!state.data) return;
        const actions = {
            "+": () => zoomCenter(1.25),
            "=": () => zoomCenter(1.25),
            "-": () => zoomCenter(0.8),
            f: fitView,
            1: () => zoomTo(1),
            c: () => setCompare(!state.compare),
            o: () => setLayer("outlines", !state.layers.outlines),
            m: () => setLayer("mask", !state.layers.mask),
            i: () => setLayer("ids", !state.layers.ids),
            t: () => state.mode === "synthetic" && setLayer("truth", !state.layers.truth),
            p: () => setBase(state.base === "original" ? "processed" : "original"),
            escape: () => select(0),
        };
        if (actions[key]) {
            e.preventDefault();
            actions[key]();
        }
    });

    new ResizeObserver(() => {
        if (state.view.fitted) fitView();
        else render();
    }).observe(els.stage);

    syncOutputs();
    setStatus("idle", "Ready");
    render();
})();
