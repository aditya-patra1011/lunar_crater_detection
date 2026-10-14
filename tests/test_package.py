import importlib
from importlib.metadata import version

import pytest

import lunar_crater

SUBPACKAGES = [
    "data",
    "baselines",
    "models",
    "training",
    "postprocess",
    "evaluation",
    "serving",
    "monitoring",
]


def test_version_matches_installed_metadata():
    assert lunar_crater.__version__ == version("lunar-crater-detection")


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_imports(name):
    assert importlib.import_module(f"lunar_crater.{name}") is not None
