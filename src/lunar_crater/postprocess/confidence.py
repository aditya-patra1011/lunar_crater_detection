"""Turn detection scores into the chance that a detection matches a catalog crater.

Scores rank detections but are not probabilities. This fits, on val detections, the
share that match a catalog crater as a rising function of score (isotonic regression),
separately for small and large craters, since their scores come from different
networks. The result is a lower bound on the chance a detection is a real crater,
because the catalog misses some.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from lunar_crater.evaluation.matching import match_circles, radius_px_to_diam_km

#: Craters at least this wide use the large-crater curve (the served model's size cut).
LARGE_FROM_KM = 6.0


def label_detections(preds: pd.DataFrame, tiles: pd.DataFrame, craters: pd.DataFrame):
    """Each counted detection with its diam_km and whether it matches a catalog crater.

    Uses the protocol in docs/EVALUATION.md: one-to-one matching at IoU 0.5, and only
    detections centred in the tile and at least 1 km across are kept.
    """
    cols = ["x_px", "y_px", "r_px"]
    by_tile = dict(tuple(craters.groupby("tile_id")))
    parts = []
    for t in tiles.itertuples():
        pred = preds[preds["tile_id"] == t.tile_id].reset_index(drop=True)
        if pred.empty:
            continue
        truth = by_tile.get(t.tile_id, craters.iloc[:0])
        p_idx, _, _ = match_circles(pred[cols].to_numpy(), truth[cols].to_numpy())
        matched = np.zeros(len(pred), dtype=bool)
        matched[p_idx] = True
        diam = radius_px_to_diam_km(pred["r_px"], t.scale_m)
        x, y = pred["x_px"].to_numpy(), pred["y_px"].to_numpy()
        edge = t.size_px - 0.5
        keep = (x >= -0.5) & (x < edge) & (y >= -0.5) & (y < edge) & (diam >= 1.0)
        parts.append(pred.assign(diam_km=diam, matched=matched)[keep])
    if not parts:
        return preds.iloc[:0].assign(diam_km=[], matched=[])
    return pd.concat(parts, ignore_index=True)


def isotonic(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rising step fit of y on x by pool-adjacent-violators; returns block x and y means."""
    order = np.argsort(x, kind="stable")
    x, y = np.asarray(x, float)[order], np.asarray(y, float)[order]
    means, weights, xs = [], [], []
    for xi, yi in zip(x, y, strict=True):
        means.append(yi)
        weights.append(1.0)
        xs.append([xi])
        while len(means) > 1 and means[-2] >= means[-1]:
            w = weights[-2] + weights[-1]
            means[-2] = (means[-2] * weights[-2] + means[-1] * weights[-1]) / w
            weights[-2] = w
            xs[-2].extend(xs.pop())
            means.pop()
            weights.pop()
    centres = np.array([np.median(block) for block in xs])
    return centres, np.array(means)


@dataclass
class Curve:
    """A rising piecewise-linear map from score to probability."""

    score: list[float] = field(default_factory=list)
    probability: list[float] = field(default_factory=list)

    def __call__(self, scores) -> np.ndarray:
        return np.interp(np.asarray(scores, float), self.score, self.probability)


@dataclass
class Confidence:
    """Probability curves for small and large detections, fitted on one split."""

    small: Curve
    large: Curve
    large_from_km: float = LARGE_FROM_KM
    fitted_on: str = ""
    n_detections: int = 0

    def __call__(self, scores, diam_km) -> np.ndarray:
        scores, diam_km = np.asarray(scores, float), np.asarray(diam_km, float)
        large = diam_km >= self.large_from_km
        return np.where(large, self.large(scores), self.small(scores))

    @classmethod
    def fit(cls, labelled: pd.DataFrame, *, fitted_on: str = "", knots: int = 40) -> Confidence:
        """Fit both curves on detections with score, diam_km and matched."""

        def curve(rows: pd.DataFrame) -> Curve:
            if rows.empty:
                return Curve([0.0, 1.0], [0.0, 0.0])
            x, y = isotonic(rows["score"].to_numpy(), rows["matched"].to_numpy(float))
            # Keep the curve small: at most ``knots`` points, spread over its range.
            keep = np.unique(np.linspace(0, len(x) - 1, min(knots, len(x))).round().astype(int))
            return Curve(x[keep].round(5).tolist(), y[keep].round(5).tolist())

        large = labelled["diam_km"] >= LARGE_FROM_KM
        return cls(
            small=curve(labelled[~large]),
            large=curve(labelled[large]),
            fitted_on=fitted_on,
            n_detections=len(labelled),
        )

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> Confidence:
        data = json.loads(Path(path).read_text())
        return cls(**{**data, "small": Curve(**data["small"]), "large": Curve(**data["large"])})


def reliability(probability: np.ndarray, matched: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Mean predicted probability against the matched share, in bins of equal count."""
    order = np.argsort(probability, kind="stable")
    rows = []
    for idx in np.array_split(order, bins):
        if len(idx):
            rows.append({"n": len(idx), "predicted": probability[idx].mean(),
                         "observed": matched[idx].mean()})  # fmt: skip
    return pd.DataFrame(rows)


def calibration_error(probability: np.ndarray, matched: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error: the count-weighted gap between predicted and observed."""
    table = reliability(probability, matched, bins)
    return float(np.average(np.abs(table["predicted"] - table["observed"]), weights=table["n"]))
