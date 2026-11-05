"""Export the trained networks to one ONNX file and check it against PyTorch.

Writes ``<out>.onnx``, which maps a normalized tile to rim probabilities, and
``<out>.json``, which holds the settings needed to turn those into craters.

Usage::

    python -m lunar_crater.serving.export --main models/unet_v1/best.pt \\
        --large models/ms_unet_v1/seed0/best.pt --out models/crater_detector
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import onnxruntime
import torch
from torch import nn

from lunar_crater.data.tiles import TileSpec
from lunar_crater.models.detector import load_model
from lunar_crater.postprocess.finder import RimCircleFinder

#: How far ONNX and PyTorch probabilities may differ before the export is refused.
MAX_DIFFERENCE = 1e-3


class RimNetwork(nn.Module):
    """Rim probabilities, (1, 2, h, w): main rims from one model, large rims from others."""

    def __init__(self, main: nn.Module, large: Sequence[nn.Module]):
        super().__init__()
        self.main = main
        self.large = nn.ModuleList(large)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        main = torch.sigmoid(self.main(x))[:, :1]
        large = [torch.sigmoid(model(x))[:, 1:2] for model in self.large]
        return torch.cat([main, torch.stack(large).mean(dim=0)], dim=1)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def export(
    main: str,
    large: Sequence[str],
    out: Path,
    finder: RimCircleFinder,
    *,
    min_score: float,
    spec: TileSpec,
) -> dict:
    """Write the ONNX file and settings; raises if ONNX and PyTorch disagree."""
    for path in large:
        if load_model(path).large_level == 0:
            raise ValueError(f"{path} has no large-rim output")
    network = RimNetwork(load_model(main), [load_model(path) for path in large]).eval()

    size = spec.size_px
    rng = np.random.default_rng(0)
    sample = torch.from_numpy(rng.normal(size=(1, 1, size, size)).astype(np.float32))
    out.parent.mkdir(parents=True, exist_ok=True)
    onnx_path = out.with_suffix(".onnx")
    torch.onnx.export(
        network,
        (sample,),
        str(onnx_path),
        input_names=["tile"],
        output_names=["rims"],
        dynamo=True,
        external_data=False,
    )

    session = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    other = rng.normal(size=(1, 1, size, size)).astype(np.float32)
    difference = 0.0
    for x in (sample.numpy(), other):
        with torch.no_grad():
            expected = network(torch.from_numpy(x)).numpy()
        difference = max(
            difference, float(np.abs(session.run(None, {"tile": x})[0] - expected).max())
        )
    if difference > MAX_DIFFERENCE:
        raise RuntimeError(f"ONNX and PyTorch differ by {difference:.2g}")

    settings = {
        "size_px": size,
        "scale_m": spec.scale_m,
        "min_score": min_score,
        "finder": dataclasses.asdict(finder),
        "sources": {"main": main, "large": list(large)},
        "onnx_sha256": file_sha256(onnx_path),
        "max_difference_from_pytorch": difference,
    }
    out.with_suffix(".json").write_text(json.dumps(settings, indent=2) + "\n")
    return settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--main", required=True, help="best.pt that finds small craters")
    parser.add_argument("--large", required=True, help="best.pt files joined by commas")
    parser.add_argument("--out", type=Path, required=True, help="output path, without suffix")
    parser.add_argument("--min-score", type=float, default=0.4349, help="chosen on val")
    parser.add_argument("--large-gain", type=float, default=1.25)
    parser.add_argument("--large-radius-scale", type=float, default=1.0318)
    parser.add_argument("--radius-scale", type=float, default=1.0723)
    args = parser.parse_args(argv)

    finder = RimCircleFinder(
        thickness_frac=0.2,
        nms_iou=0.1,
        size_exponent=0.5,
        radius_scale=args.radius_scale,
        large_source="head",
        large_from_km=6.0,
        large_gain=args.large_gain,
        large_radius_scale=args.large_radius_scale,
        large_min_score=0.03,
    )
    settings = export(
        args.main,
        args.large.split(","),
        args.out,
        finder,
        min_score=args.min_score,
        spec=TileSpec(0.0, 0.0),
    )
    print(json.dumps(settings, indent=2), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
