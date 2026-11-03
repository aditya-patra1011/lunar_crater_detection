"""Train the U-Net rim segmenter on the train split, keeping the best epoch on val.

Runs on CUDA when available, otherwise CPU. Val is used only for its loss, to pick the
epoch; detection metrics come later from the evaluation harness.

Usage::

    python -m lunar_crater.training.train_unet data/interim/tiles/v1 --out models/unet_v1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lunar_crater.evaluation.harness import load_split
from lunar_crater.models.unet import UNet, rim_loss
from lunar_crater.training.data import MIN_SPREAD_M, augment, load_examples


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_split_examples(tiles_dir: Path, split: str, thickness_px: float, limit: int | None):
    tiles, craters = load_split(tiles_dir, split)
    if limit:
        tiles = tiles.head(limit)
    return load_examples(tiles_dir, tiles, craters, thickness_px=thickness_px)


def batches(n: int, size: int, rng: np.random.Generator | None):
    order = rng.permutation(n) if rng is not None else np.arange(n)
    for start in range(0, n, size):
        yield order[start : start + size]


def to_tensors(images, targets, device):
    x = torch.from_numpy(np.ascontiguousarray(images)).unsqueeze(1).to(device)
    y = torch.from_numpy(np.ascontiguousarray(targets)).float().unsqueeze(1).to(device)
    return x, y


def run_epoch(model, images, targets, *, device, batch_size, optimizer=None, rng=None, scaler=None):
    """Mean loss over one pass; trains (with augmentation) when given an optimizer.

    On CUDA the forward pass runs in float16; pass the same ``scaler`` every epoch so
    gradient scaling keeps small gradients from underflowing.
    """
    training = optimizer is not None
    model.train(training)
    total, count = 0.0, 0
    cuda = device.type == "cuda"
    autocast = torch.autocast(device.type, enabled=cuda)
    scaler = scaler or torch.amp.GradScaler(device.type, enabled=cuda)
    for idx in batches(len(images), batch_size, rng if training else None):
        xb, yb = images[idx], targets[idx]
        if training:
            pairs = [augment(a, b, rng) for a, b in zip(xb, yb, strict=True)]
            xb, yb = np.stack([p[0] for p in pairs]), np.stack([p[1] for p in pairs])
        x, y = to_tensors(xb, yb, device)
        with torch.set_grad_enabled(training), autocast:
            loss = rim_loss(model(x).float(), y)
        if training:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        total += float(loss.detach()) * len(idx)
        count += len(idx)
    return total / max(count, 1)


def train(args: argparse.Namespace) -> pd.DataFrame:
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = pick_device(args.device)
    log(f"device: {device}")

    train_x, train_y = load_split_examples(args.tiles_dir, "train", args.thickness, args.limit)
    val_x, val_y = load_split_examples(args.tiles_dir, "val", args.thickness, args.limit)
    log(f"train {len(train_x)} tiles, val {len(val_x)} tiles, rim pixels {train_y.mean():.3%}")

    model = UNet(base=args.base, depth=args.depth).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    config = {
        "model": {"base": args.base, "depth": args.depth},
        "data": {
            "tiles_dir": str(args.tiles_dir),
            "thickness_px": args.thickness,
            "min_spread_m": MIN_SPREAD_M,
        },
        "train": {
            k: getattr(args, k)
            for k in ("epochs", "batch_size", "lr", "weight_decay", "seed", "limit")
        },
        "parameters": sum(p.numel() for p in model.parameters()),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    history, best = [], np.inf
    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        train_loss = run_epoch(
            model, train_x, train_y, device=device, batch_size=args.batch_size,
            optimizer=optimizer, rng=rng, scaler=scaler,
        )  # fmt: skip
        val_loss = run_epoch(model, val_x, val_y, device=device, batch_size=args.batch_size)
        schedule.step()
        seconds = time.perf_counter() - start
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "seconds": seconds}
        )
        if val_loss < best:
            best = val_loss
            torch.save(
                {"state_dict": model.state_dict(), "config": config, "epoch": epoch,
                 "val_loss": val_loss},
                args.out / "best.pt",
            )  # fmt: skip
        log(f"epoch {epoch}: train {train_loss:.4f} val {val_loss:.4f} ({seconds:.0f} s)")
        pd.DataFrame(history).to_csv(args.out / "history.csv", index=False)
    return pd.DataFrame(history)


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tiles_dir", type=Path, help="folder with tiles, tiles.csv, split.csv")
    parser.add_argument("--out", type=Path, required=True, help="folder for best.pt and logs")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--base", type=int, default=32, help="channels at full resolution")
    parser.add_argument("--depth", type=int, default=4, help="number of downsamplings")
    parser.add_argument("--thickness", type=float, default=2.0, help="rim width in pixels")
    parser.add_argument("--limit", type=int, help="use only the first N tiles of each split")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    train(parser.parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
