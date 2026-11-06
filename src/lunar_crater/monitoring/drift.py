"""Watch whether the tiles sent to the service still look like the ones it was checked on.

Each tile is reduced to a few numbers about its terrain (inputs) and about what the
model found on it (outputs). Recent tiles are compared with a reference set using the
population stability index (PSI) on reference deciles.

The common rule of thumb (watch above 0.1, shifted above 0.25) raises false alarms here,
because terrain differs a lot between regions of the Moon. So each feature's thresholds
can instead be learned from the reference itself: hold out a few regions at a time,
compare them with a reference built from the rest, and take the 95th and 99th
percentiles of the PSI that unseen but normal regions produce.
"""

from __future__ import annotations

import json
import threading
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

#: What each feature measures, for reports.
FEATURES = {
    "relief_m": "spread of heights in the tile (m)",
    "slope_deg": "median ground slope (degrees)",
    "missing": "share of missing heights",
    "n_craters": "craters found in the tile",
    "median_score": "median score of those craters",
}
INPUTS = ("relief_m", "slope_deg", "missing")
WATCH, DRIFT = 0.1, 0.25
#: Fewer recent tiles than this give too noisy a PSI to report. Thresholds are learned
#: on windows of this size, the noisiest the monitor reports on.
MIN_TILES = 100


def tile_features(dem: np.ndarray, scale_m: float, scores) -> dict:
    """The numbers the monitor tracks for one tile and its detections."""
    dem = np.asarray(dem, float)
    finite = np.isfinite(dem)
    filled = np.where(finite, dem, np.nanmedian(dem) if finite.any() else 0.0)
    gy, gx = np.gradient(filled, scale_m)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    scores = np.asarray(scores, float)
    return {
        "relief_m": float(np.nanstd(dem)) if finite.any() else 0.0,
        "slope_deg": float(np.median(slope)),
        "missing": float(1 - finite.mean()),
        "n_craters": float(len(scores)),
        "median_score": float(np.median(scores)) if len(scores) else 0.0,
    }


def psi(reference_share: np.ndarray, recent_share: np.ndarray, eps: float = 1e-4) -> float:
    """Population stability index between two sets of bin shares."""
    a = np.clip(np.asarray(recent_share, float), eps, None)
    r = np.clip(np.asarray(reference_share, float), eps, None)
    return float(np.sum((a - r) * np.log(a / r)))


def status(value: float, watch: float = WATCH, drift: float = DRIFT) -> str:
    return "drift" if value > drift else "watch" if value > watch else "stable"


@dataclass
class Reference:
    """Decile edges and bin shares of each feature over a reference set of tiles.

    ``watch`` and ``drift`` hold each feature's PSI thresholds; features missing from
    them use the rule of thumb.
    """

    edges: dict[str, list[float]]
    shares: dict[str, list[float]]
    n_tiles: int
    source: str = ""
    watch: dict[str, float] = field(default_factory=dict)
    drift: dict[str, float] = field(default_factory=dict)

    @classmethod
    def build(cls, features: pd.DataFrame, *, source: str = "", bins: int = 10) -> Reference:
        edges, shares = {}, {}
        for name in FEATURES:
            values = features[name].to_numpy(float)
            inner = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)[1:-1]))
            edges[name] = inner.round(6).tolist()
            shares[name] = bin_shares(values, inner).round(6).tolist()
        return cls(edges, shares, len(features), source)

    def compare(self, features: pd.DataFrame) -> dict:
        """PSI and status for each feature of ``features`` against this reference."""
        out = {}
        for name in FEATURES:
            values = features[name].to_numpy(float)
            recent = bin_shares(values, self.edges[name], prior=0.5)
            value = psi(self.shares[name], recent)
            watch, drift = self.thresholds(name)
            out[name] = {
                "psi": round(value, 4),
                "status": status(value, watch, drift),
                "watch_above": watch,
                "drift_above": drift,
                "recent_median": float(np.median(values)),
                "meaning": FEATURES[name],
            }
        return out

    def thresholds(self, name: str) -> tuple[float, float]:
        return self.watch.get(name, WATCH), self.drift.get(name, DRIFT)

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> Reference:
        return cls(**json.loads(Path(path).read_text()))


def bin_shares(values: np.ndarray, inner_edges, prior: float = 0.0) -> np.ndarray:
    """Share of ``values`` in each bin between the inner edges (open at both ends).

    ``prior`` adds that many tiles to every bin first, so a bin left empty by chance in a
    small sample does not count as a large shift.
    """
    counts = np.bincount(np.searchsorted(inner_edges, values, side="right"),
                         minlength=len(inner_edges) + 1)  # fmt: skip
    return (counts + prior) / (len(values) + prior * len(counts))


class DriftMonitor:
    """Keeps the most recent tiles' features and reports drift against a reference."""

    def __init__(self, reference: Reference | None, window: int = 200, min_tiles: int = MIN_TILES):
        self.reference = reference
        self.min_tiles = min_tiles
        self.recent: deque[dict] = deque(maxlen=window)
        self.seen = 0
        self._lock = threading.Lock()

    def record(self, features: dict) -> None:
        with self._lock:
            self.recent.append(features)
            self.seen += 1

    def report(self) -> dict:
        with self._lock:
            recent = pd.DataFrame(list(self.recent))
            seen = self.seen
        base = {"tiles_seen": seen, "window": self.recent.maxlen, "in_window": len(recent),
                "min_tiles": self.min_tiles}  # fmt: skip
        if self.reference is None:
            return {**base, "status": "off", "message": "No reference set is loaded."}
        if len(recent) < self.min_tiles:
            message = f"Needs {self.min_tiles} tiles to compare; has {len(recent)}."
            return {**base, "status": "waiting", "message": message}
        features = self.reference.compare(recent)
        order = ["stable", "watch", "drift"]
        worst = max((f["status"] for f in features.values()), key=order.index)
        inputs = max((features[n]["status"] for n in INPUTS), key=order.index)
        return {**base, "status": worst, "inputs_status": inputs, "features": features,
                "reference_tiles": self.reference.n_tiles}  # fmt: skip


def learn_thresholds(
    features: pd.DataFrame,
    regions: np.ndarray,
    *,
    held_out: int = 7,
    window: int = MIN_TILES,
    repeats: int = 300,
    seed: int = 0,
) -> tuple[dict[str, float], dict[str, float]]:
    """Per-feature watch and drift thresholds from normal variation between regions.

    Each repeat holds out ``held_out`` regions, builds a reference from the others, and
    compares ``window`` tiles drawn from the held-out regions. The 95th and 99th
    percentiles of those PSI values become the watch and drift thresholds.
    """
    rng = np.random.default_rng(seed)
    names = np.unique(regions)
    values = {name: [] for name in FEATURES}
    for _ in range(repeats):
        out = np.isin(regions, rng.choice(names, size=held_out, replace=False))
        reference = Reference.build(features[~out])
        recent = features[out].sample(n=window, replace=out.sum() < window, random_state=rng)
        for name, f in reference.compare(recent).items():
            values[name].append(f["psi"])
    watch = {n: round(float(np.percentile(v, 95)), 4) for n, v in values.items()}
    drift = {n: round(float(np.percentile(v, 99)), 4) for n, v in values.items()}
    # Never looser than the rule of thumb's watch line: a feature that never varies
    # (such as missing data) should still alarm when it does.
    watch = {n: max(v, WATCH) for n, v in watch.items()}
    drift = {n: max(v, DRIFT, watch[n]) for n, v in drift.items()}
    return watch, drift
