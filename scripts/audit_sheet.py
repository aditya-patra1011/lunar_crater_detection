"""Prepare, and later summarize, a manual check of detections with no catalog match.

Precision counts every unmatched detection as wrong, but some are real craters missing
from the catalog. This draws a random sample of them on the terrain for a person to
judge, and turns the verdicts into a corrected precision.

Usage::

    python scripts/audit_sheet.py prepare data/interim/tiles/v1 outputs/served_v1_test \\
        docs/audit --min-score 0.4362
    python scripts/audit_sheet.py summarize data/interim/tiles/v1 outputs/served_v1_test \\
        docs/audit --min-score 0.4362

``prepare`` writes ``audit.csv`` with an empty ``verdict`` column and pages of images.
Fill each verdict with ``crater``, ``not crater`` or ``unsure``, then run ``summarize``.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

from lunar_crater.data.tiles import TileSpec  # noqa: E402
from lunar_crater.evaluation.harness import load_split  # noqa: E402
from lunar_crater.postprocess.confidence import label_detections  # noqa: E402
from lunar_crater.postprocess.georef import to_moon_coords  # noqa: E402

VERDICTS = ("crater", "not crater", "unsure")
PER_PAGE = 12


def hillshade(dem: np.ndarray, scale_m: float) -> np.ndarray:
    dem = np.where(np.isfinite(dem), dem, np.nanmedian(dem))
    gy, gx = np.gradient(dem, scale_m)
    gx, gy = gx * 2.5, gy * 2.5
    shade = (0.6 * gx + 0.6 * gy + 0.53) / np.sqrt(gx**2 + gy**2 + 1)
    span = max(np.ptp(dem), 1.0)
    return np.clip(0.72 * np.clip(shade, 0, None) + 0.38 * (dem - dem.min()) / span, 0, 1)


def detections(tiles_dir: Path, run_dir: Path, min_score: float) -> pd.DataFrame:
    tiles, craters = load_split(tiles_dir, "test")
    preds = pd.read_csv(run_dir / "predictions.csv")
    rows = label_detections(preds[preds["score"] >= min_score], tiles, craters)
    parts = []
    for tile_id, group in rows.groupby("tile_id"):
        t = tiles.set_index("tile_id").loc[tile_id]
        spec = TileSpec(float(t["lat"]), float(t["lon"]), int(t["size_px"]), float(t["scale_m"]))
        parts.append(
            to_moon_coords(group.drop(columns="diam_km"), spec).assign(scale_m=spec.scale_m)
        )
    return pd.concat(parts, ignore_index=True)


def prepare(args) -> None:
    rows = detections(args.tiles_dir, args.run_dir, args.min_score)
    unmatched = rows[~rows["matched"]]
    sample = unmatched.sample(n=min(args.n, len(unmatched)), random_state=args.seed)
    sample = sample.sort_values("diam_km").reset_index(drop=True)
    sample.insert(0, "audit_id", [f"a{i + 1:03d}" for i in range(len(sample))])
    columns = ["audit_id", "tile_id", "x_px", "y_px", "r_px", "diam_km", "lat", "lon", "score"]
    sheet = sample[columns].round(4).assign(verdict="", notes="")
    args.out.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(args.out / "audit.csv", index=False)

    pages = math.ceil(len(sample) / PER_PAGE)
    for page in range(pages):
        chunk = sample.iloc[page * PER_PAGE : (page + 1) * PER_PAGE]
        fig, axes = plt.subplots(3, 4, figsize=(12, 9.6))
        for ax in axes.flat:
            ax.axis("off")
        for ax, d in zip(axes.flat, chunk.itertuples(), strict=False):
            with np.load(args.tiles_dir / "tiles" / f"{d.tile_id}.npz") as z:
                dem = z["dem"]
            half = max(3 * d.r_px, 24)
            x0, x1 = int(max(d.x_px - half, 0)), int(min(d.x_px + half, dem.shape[1] - 1))
            y0, y1 = int(max(d.y_px - half, 0)), int(min(d.y_px + half, dem.shape[0] - 1))
            ax.imshow(hillshade(dem, d.scale_m)[y0 : y1 + 1, x0 : x1 + 1], cmap="gray",
                      extent=(x0 - 0.5, x1 + 0.5, y1 + 0.5, y0 - 0.5))  # fmt: skip
            ax.add_patch(Circle((d.x_px, d.y_px), d.r_px, fill=False, color="#ff7aa8", lw=1.5,
                                ls="--"))  # fmt: skip
            ax.set_title(f"{d.audit_id}  {d.diam_km:.1f} km", fontsize=10)
        title = f"Detections with no catalog match, page {page + 1} of {pages}. "
        fig.suptitle(title + "Does the dashed circle mark a real crater?", fontsize=11)
        fig.tight_layout()
        fig.savefig(args.out / f"audit_page_{page + 1}.png", dpi=90)
        plt.close(fig)
    print(f"{len(sample)} of {len(unmatched)} unmatched detections written to {args.out}")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval for a share of k out of n."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return centre - half, centre + half


def summarize(args) -> None:
    sheet = pd.read_csv(args.out / "audit.csv", keep_default_na=False)
    verdicts = sheet["verdict"].str.strip().str.lower()
    unknown = sorted(set(verdicts) - set(VERDICTS))
    if unknown:
        raise SystemExit(f"fill every verdict with one of {VERDICTS}; found {unknown}")
    judged = verdicts[verdicts != "unsure"]
    real = int((judged == "crater").sum())
    low, high = wilson(real, len(judged))

    rows = detections(args.tiles_dir, args.run_dir, args.min_score)
    n, matched = len(rows), int(rows["matched"].sum())
    share = real / max(len(judged), 1)

    def corrected(s: float) -> float:
        return (matched + s * (n - matched)) / n

    print(f"verdicts: {real} crater, {len(judged) - real} not crater, "
          f"{int((verdicts == 'unsure').sum())} unsure")  # fmt: skip
    print(f"real craters among unmatched detections: {share:.0%} (95% {low:.0%} to {high:.0%})")
    print(f"precision against the catalog: {matched / n:.3f}")
    print(f"corrected precision: {corrected(share):.3f} "
          f"(95% {corrected(low):.3f} to {corrected(high):.3f})")  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=["prepare", "summarize"])
    parser.add_argument("tiles_dir", type=Path)
    parser.add_argument("run_dir", type=Path, help="harness run of the served model on test")
    parser.add_argument("out", type=Path, help="folder for audit.csv and the pages")
    parser.add_argument("--min-score", type=float, required=True)
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    prepare(args) if args.action == "prepare" else summarize(args)


if __name__ == "__main__":
    main()
