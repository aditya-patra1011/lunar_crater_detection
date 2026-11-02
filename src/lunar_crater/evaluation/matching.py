"""Match predicted craters to the catalog and score them by crater size.

The protocol is written up in docs/EVALUATION.md. In short: circles match when their
intersection-over-union is at least a threshold (0.5 by default), each crater matches
at most once (greedy by highest IoU), and craters are counted only when their centre
lies inside the tile, so rims cut by the tile edge are not penalised.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from lunar_crater.data.projection import MOON_RADIUS_M

DIAMETER_BINS_KM = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, math.inf)


def circle_iou(x1, y1, r1, x2, y2, r2) -> np.ndarray:
    """Intersection-over-union of circles; inputs broadcast against each other."""
    x1, y1, r1, x2, y2, r2 = np.broadcast_arrays(*map(np.asarray, (x1, y1, r1, x2, y2, r2)))
    d = np.hypot(x1 - x2, y1 - y2)
    small, big = np.minimum(r1, r2), np.maximum(r1, r2)

    with np.errstate(divide="ignore", invalid="ignore"):
        a1 = np.arccos(np.clip((d**2 + r1**2 - r2**2) / (2 * d * r1), -1, 1))
        a2 = np.arccos(np.clip((d**2 + r2**2 - r1**2) / (2 * d * r2), -1, 1))
        kite = (-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2)
        lens = r1**2 * a1 + r2**2 * a2 - 0.5 * np.sqrt(np.maximum(kite, 0))

    inter = np.where(d >= r1 + r2, 0.0, np.where(d <= big - small, np.pi * small**2, lens))
    union = np.pi * (r1**2 + r2**2) - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)


def match_circles(pred: np.ndarray, true: np.ndarray, iou_threshold: float = 0.5):
    """Greedily pair predictions with true craters, highest IoU first.

    ``pred`` and ``true`` are (n, 3) arrays of x, y, r in the same units. Returns
    (pred_index, true_index, iou) arrays of the accepted pairs.
    """
    pred, true = np.asarray(pred, dtype=float), np.asarray(true, dtype=float)
    empty = np.array([], dtype=int)
    if len(pred) == 0 or len(true) == 0:
        return empty, empty, np.array([])

    iou = circle_iou(*pred.T[:, :, None], *true.T[:, None, :])
    pi, ti = np.nonzero(iou >= iou_threshold)
    order = np.argsort(-iou[pi, ti], kind="stable")
    used_p, used_t, pairs = set(), set(), []
    for k in order:
        p, t = pi[k], ti[k]
        if p not in used_p and t not in used_t:
            used_p.add(p)
            used_t.add(t)
            pairs.append((p, t, iou[p, t]))
    if not pairs:
        return empty, empty, np.array([])
    p, t, v = zip(*pairs, strict=True)
    return np.array(p), np.array(t), np.array(v)


def radius_px_to_diam_km(r_px, scale_m: float) -> np.ndarray:
    """Invert the tile's r_px = R·sin(r/R)/scale to a surface diameter in km."""
    s = np.clip(np.asarray(r_px, dtype=float) * scale_m / MOON_RADIUS_M, 0, 1)
    return 2 * MOON_RADIUS_M * np.arcsin(s) / 1000.0


def evaluate_tile(
    pred: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    size_px: int,
    scale_m: float,
    iou_threshold: float = 0.5,
    min_diam_km: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score one tile.

    ``pred`` has columns x_px, y_px, r_px. ``truth`` is the tile's rows of craters.csv
    (x_px, y_px, r_px, diam_km, centre_in_tile). All of both take part in matching, but
    only craters centred in the tile and at least ``min_diam_km`` across are counted.

    Returns (truth_rows, pred_rows), each with columns ``diam_km`` and ``matched``.
    """
    p_idx, t_idx, _ = match_circles(
        pred[["x_px", "y_px", "r_px"]].to_numpy(),
        truth[["x_px", "y_px", "r_px"]].to_numpy(),
        iou_threshold,
    )
    t_matched = np.zeros(len(truth), dtype=bool)
    t_matched[t_idx] = True
    p_matched = np.zeros(len(pred), dtype=bool)
    p_matched[p_idx] = True

    truth_rows = pd.DataFrame({"diam_km": truth["diam_km"].to_numpy(), "matched": t_matched})
    t_keep = truth["centre_in_tile"].to_numpy(dtype=bool) & (truth_rows["diam_km"] >= min_diam_km)

    p_diam = radius_px_to_diam_km(pred["r_px"], scale_m)
    pred_rows = pd.DataFrame({"diam_km": p_diam, "matched": p_matched})
    x, y = pred["x_px"].to_numpy(), pred["y_px"].to_numpy()
    inside = (x >= -0.5) & (x < size_px - 0.5) & (y >= -0.5) & (y < size_px - 0.5)
    p_keep = inside & (p_diam >= min_diam_km)

    return truth_rows[t_keep].reset_index(drop=True), pred_rows[p_keep].reset_index(drop=True)


def summarize(
    truth_rows: pd.DataFrame, pred_rows: pd.DataFrame, bins=DIAMETER_BINS_KM
) -> pd.DataFrame:
    """Recall (binned by true diameter), precision (binned by predicted diameter) and F1.

    The last row, ``all``, pools every crater. Precision is a lower bound because the
    catalog is incomplete.
    """
    labels = [
        f"{lo:g}-{hi:g}" if math.isfinite(hi) else f">={lo:g}"
        for lo, hi in zip(bins[:-1], bins[1:], strict=True)
    ]

    def per_bin(rows: pd.DataFrame) -> pd.DataFrame:
        binned = pd.cut(rows["diam_km"], bins=list(bins), labels=labels, right=False)
        counts = rows.groupby(binned, observed=False)["matched"].agg(["size", "sum"])
        counts.loc["all"] = [len(rows), int(rows["matched"].sum())]
        return counts

    t, p = per_bin(truth_rows), per_bin(pred_rows)
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = t["sum"] / t["size"]
        precision = p["sum"] / p["size"]
        f1 = 2 * precision * recall / (precision + recall)
    out = pd.DataFrame(
        {
            "n_true": t["size"].astype(int),
            "recall": recall,
            "n_pred": p["size"].astype(int),
            "precision": precision,
            "f1": f1,
        }
    )
    out.index.name = "diam_km"
    return out
