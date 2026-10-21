"""Train the learned segmentation engine and export it to ONNX.

Usage::

    python training/train.py --data datasets/prepared --out runs/unet --iters 12000

Training samples random 256 x 256 patches with heavy augmentation chosen to
bridge the gap between the 16-bit training data and the 8-bit RGB JPEG
micrographs typical of teaching and iGEM labs: scale, flips, rotations,
percentile-normalisation jitter, gamma, blur, Gaussian and Poisson-like noise,
8-bit quantisation, JPEG compression and out-of-focus haze (heavily blurred
blobs that must be classified as background). A pool of images from the
package's own synthetic generator, with exact masks, is mixed in so the model
also sees GFP-like rods on a dark field. The checkpoint with the best
validation score (mean of AP@0.5 and AP@0.75 on held-out images) is exported
to ONNX for inference with OpenCV.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from cell_identifier import SyntheticConfig, generate
from cell_identifier.evaluation import average_precision
from cell_identifier.learned import make_targets, normalize_input, postprocess
from model import UNet, count_parameters

PATCH = 256


def synthetic_pool(count: int, target_width: float, seed: int) -> list[dict[str, np.ndarray]]:
    """Synthetic training images with exact targets, rescaled to the working width."""
    rng = np.random.default_rng(seed)
    pool = []
    for i in range(count):
        config = SyntheticConfig(
            seed=seed * 1000 + i,
            num_cells=int(rng.integers(40, 160)),
            shape="pill" if rng.random() < 0.8 else "ellipse",
            allow_overlap=bool(rng.random() < 0.5),
            noise_sigma=float(rng.uniform(0.005, 0.05)),
            blur_sigma=float(rng.uniform(0.5, 1.8)),
            background_level=float(rng.uniform(0.0, 0.12)),
            illumination_gradient=float(rng.uniform(0.0, 0.4)),
        )
        synthetic = generate(config)
        image = synthetic.image[..., 1].astype(np.float32)  # green channel carries the signal
        widths = [c.width for c in synthetic.cells] or [target_width]
        factor = target_width / float(np.median(widths))
        size = (round(image.shape[1] * factor), round(image.shape[0] * factor))
        image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
        labels = cv2.resize(
            synthetic.labels.astype(np.float32), size, interpolation=cv2.INTER_NEAREST
        )
        pool.append(
            {
                "image": image,
                "targets": make_targets(labels.astype(np.int32)).astype(np.float16),
                "p_low": np.float32(np.percentile(image, 1)),
                "p_high": np.float32(np.percentile(image, 99.8)),
                "name": f"synthetic_{i}",
            }
        )
    return pool


def add_haze(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Add 1-4 smooth, out-of-focus fluorescent blobs (background, not cells)."""
    haze = np.zeros_like(x)
    h, w = x.shape
    for _ in range(int(rng.integers(1, 5))):
        radius = rng.uniform(12, 60)
        cy, cx = rng.uniform(0, h), rng.uniform(0, w)
        yy, xx = np.ogrid[:h, :w]
        haze += rng.uniform(0.05, 0.45) * np.exp(
            -((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * radius**2)
        )
    return x + haze.astype(np.float32)


def load_split(root: Path, split: str) -> list[dict[str, np.ndarray]]:
    items = []
    for path in sorted((root / split).glob("*.npz")):
        with np.load(path) as z:
            items.append({k: z[k] for k in z.files} | {"name": path.stem})
    return items


class PatchDataset(Dataset):
    """Infinite stream of augmented training patches."""

    def __init__(self, items: list[dict[str, np.ndarray]], length: int, seed: int = 0) -> None:
        self.items = items
        self.length = length
        self.seed = seed
        area = np.array([it["image"].size for it in items], dtype=np.float64)
        self.weights = (
            np.sqrt(area) / np.sqrt(area).sum()
        )  # favour big images, but not overwhelmingly

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        rng = np.random.default_rng((self.seed, index, int(time.time_ns() % 1_000_003)))
        it = self.items[rng.choice(len(self.items), p=self.weights)]
        image, targets = it["image"], it["targets"]

        # Scale augmentation: crop a window of PATCH / s pixels and resize it to PATCH.
        s = math.exp(rng.uniform(math.log(0.7), math.log(1.45)))
        win = max(16, round(PATCH / s))
        h, w = image.shape
        pad_y, pad_x = max(0, win - h), max(0, win - w)
        if pad_y or pad_x:
            image = np.pad(image, ((0, pad_y), (0, pad_x)), mode="reflect")
            targets = np.pad(targets, ((0, 0), (0, pad_y), (0, pad_x)), mode="constant")
            h, w = image.shape
        y0, x0 = rng.integers(0, h - win + 1), rng.integers(0, w - win + 1)
        crop = image[y0 : y0 + win, x0 : x0 + win]
        tcrop = targets[:, y0 : y0 + win, x0 : x0 + win].astype(np.float32)
        if win != PATCH:
            interp = cv2.INTER_AREA if win > PATCH else cv2.INTER_LINEAR
            crop = cv2.resize(crop, (PATCH, PATCH), interpolation=interp)
            tcrop = np.stack(
                [
                    cv2.resize(tcrop[0], (PATCH, PATCH), interpolation=cv2.INTER_NEAREST),
                    cv2.resize(tcrop[1], (PATCH, PATCH), interpolation=cv2.INTER_NEAREST),
                    cv2.resize(tcrop[2], (PATCH, PATCH), interpolation=cv2.INTER_LINEAR),
                ]
            )

        # Geometric augmentation.
        k = int(rng.integers(4))
        crop, tcrop = np.rot90(crop, k), np.rot90(tcrop, k, axes=(1, 2))
        if rng.random() < 0.5:
            crop, tcrop = crop[:, ::-1], tcrop[:, :, ::-1]

        # Photometric augmentation, on the percentile-normalised image.
        lo = float(it["p_low"]) * rng.uniform(0.8, 1.2)
        hi = float(it["p_high"]) * rng.uniform(0.85, 1.15)
        x = np.clip((crop - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)
        if rng.random() < 0.3:  # simulate a dimmer, low-SNR acquisition
            photons = rng.uniform(15, 150)  # expected photons at full brightness
            x = rng.poisson(x * photons) / photons
            x = normalize_input(x.astype(np.float32))
        if rng.random() < 0.35:
            x = add_haze(x, rng)
        x = np.clip(x, 0, 1) ** rng.uniform(0.6, 1.6)
        if rng.random() < 0.4:
            x = cv2.GaussianBlur(x.astype(np.float32), (0, 0), rng.uniform(0.3, 1.5))
        if rng.random() < 0.5:
            x = x + rng.normal(0, rng.uniform(0.005, 0.06), x.shape)
        x = x + rng.uniform(-0.05, 0.1)  # background offset
        x = np.clip(x, 0, 1)
        if rng.random() < 0.5:  # 8-bit capture, optionally JPEG-compressed
            q = np.rint(x * 255).astype(np.uint8)
            if rng.random() < 0.6:
                _, buf = cv2.imencode(
                    ".jpg", q, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(35, 96))]
                )
                q = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
            x = q.astype(np.float32) / 255
        x = normalize_input(x.astype(np.float32))  # exactly what inference does
        return (
            torch.from_numpy(np.ascontiguousarray(x)[None]).float(),
            torch.from_numpy(np.ascontiguousarray(tcrop)).float(),
        )


def dice_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    inter = (pred * target).sum(dim=(1, 2))
    return 1 - ((2 * inter + 1) / (pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2)) + 1)).mean()


def loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = pred.float().clamp(1e-6, 1 - 1e-6)
    fg, bd, dist = pred[:, 0], pred[:, 1], pred[:, 2]
    t_fg, t_bd, t_dist = target[:, 0], target[:, 1], target[:, 2]
    l_fg = F.binary_cross_entropy(fg, t_fg) + dice_loss(fg, t_fg)
    w_bd = 1 + 4 * t_bd  # boundaries are thin: up-weight them
    l_bd = (F.binary_cross_entropy(bd, t_bd, reduction="none") * w_bd).mean() + dice_loss(bd, t_bd)
    l_dist = F.mse_loss(dist, t_dist) * 4
    return l_fg + l_bd + l_dist


@torch.no_grad()
def predict_full(model: torch.nn.Module, image: np.ndarray, device: str) -> np.ndarray:
    x = normalize_input(image)
    h, w = x.shape
    ph, pw = (-h) % 16, (-w) % 16
    x = np.pad(x, ((0, ph), (0, pw)), mode="reflect")
    t = torch.from_numpy(x)[None, None].to(device)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
        out = model(t)
    return out[0, :, :h, :w].float().cpu().numpy()


def validate(
    model: torch.nn.Module, items: list[dict[str, np.ndarray]], device: str
) -> dict[str, float]:
    model.eval()
    ap50, ap75 = [], []
    for it in items:
        labels = postprocess(predict_full(model, it["image"], device))
        ap50.append(average_precision(labels, it["labels"], 0.5))
        ap75.append(average_precision(labels, it["labels"], 0.75))
    model.train()
    return {"ap50": float(np.mean(ap50)), "ap75": float(np.mean(ap75))}


def export_onnx(model: torch.nn.Module, path: Path) -> None:
    model = model.float().cpu().eval()
    dummy = torch.zeros(1, 1, 256, 256)
    torch.onnx.export(
        model,
        dummy,
        str(path),
        input_names=["image"],
        output_names=["maps"],
        dynamic_axes={"image": {2: "height", 3: "width"}, "maps": {2: "height", 3: "width"}},
        opset_version=17,
        dynamo=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("datasets/prepared"))
    parser.add_argument("--out", type=Path, default=Path("runs/unet"))
    parser.add_argument("--base", type=int, default=24)
    parser.add_argument("--iters", type=int, default=12000)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--val-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true", help="continue from <out>/last.pt")
    parser.add_argument("--synthetic", type=int, default=24, help="synthetic images to mix in")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)
    train_items, val_items = load_split(args.data, "train"), load_split(args.data, "val")
    for it in train_items:  # workers only need images and targets
        it.pop("labels", None)
    index = json.loads((args.data / "index.json").read_text(encoding="utf-8"))
    target_width = float(index.get("target_cell_width", 10.0))
    if args.synthetic:
        train_items += synthetic_pool(args.synthetic, target_width, args.seed)
    loader = DataLoader(
        PatchDataset(train_items, args.iters * args.batch, args.seed),
        batch_size=args.batch,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        pin_memory=device == "cuda",
    )
    model = UNet(base=args.base).to(device)
    n_params = count_parameters(model)
    print(f"device={device} params={n_params:,} train={len(train_items)} val={len(val_items)}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.iters, pct_start=0.05
    )
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    best, history, running, t0, start = -1.0, [], 0.0, time.time(), 0
    if args.resume and (args.out / "last.pt").is_file():
        state = torch.load(args.out / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        scaler.load_state_dict(state["scaler"])
        start, best, history = state["step"], state["best"], state["history"]
        print(f"resumed from step {start}")
    for step, (x, y) in enumerate(loader, start=start + 1):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
            pred = model(x)
        loss = loss_fn(pred, y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        running += loss.item()
        if step % args.val_every == 0 or step == args.iters:
            scores = validate(model, val_items, device)
            score = (scores["ap50"] + scores["ap75"]) / 2
            entry = {
                "step": step,
                "loss": running / args.val_every,
                **scores,
                "minutes": (time.time() - t0) / 60,
            }
            history.append(entry)
            running = 0.0
            flag = ""
            if score > best:
                best = score
                torch.save(model.state_dict(), args.out / "best.pt")
                flag = "  *best*"
            torch.save(
                {
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(),
                    "step": step,
                    "best": best,
                    "history": history,
                },
                args.out / "last.pt",
            )
            print(
                f"step {step:6d}  loss {entry['loss']:.4f}  val AP50 {scores['ap50']:.3f}  "
                f"AP75 {scores['ap75']:.3f}  {entry['minutes']:.1f} min{flag}",
                flush=True,
            )
        if step >= args.iters:
            break

    model.load_state_dict(torch.load(args.out / "best.pt", map_location=device))
    export_onnx(model, args.out / "cell_unet.onnx")
    (args.out / "cell_unet.json").write_text(
        json.dumps({"target_cell_width": target_width, "base_channels": args.base}, indent=1),
        encoding="utf-8",
    )
    (args.out / "history.json").write_text(
        json.dumps(
            {"args": {k: str(v) for k, v in vars(args).items()}, "history": history}, indent=1
        ),
        encoding="utf-8",
    )
    print(f"best val score {best:.4f}; exported {args.out / 'cell_unet.onnx'}")


if __name__ == "__main__":
    main()
