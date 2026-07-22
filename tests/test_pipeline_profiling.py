"""Regression tests for bounded pipeline/checkpoint profiling helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / "profiling" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def test_data_pipeline_profiler_validates_worker_prefetch_contract() -> None:
    module = _load_script("profile_data_pipeline.py")

    module.validate_loader_options(workers=0, prefetch_factor=None)
    module.validate_loader_options(workers=2, prefetch_factor=2)
    with pytest.raises(ValueError, match="workers > 0"):
        module.validate_loader_options(workers=0, prefetch_factor=2)
    with pytest.raises(ValueError, match="non-negative"):
        module.validate_loader_options(workers=-1, prefetch_factor=None)


def test_profiling_scripts_are_import_safe() -> None:
    data_module = _load_script("profile_data_pipeline.py")
    checkpoint_module = _load_script("profile_checkpoint_io.py")

    assert callable(data_module.profile_loader)
    assert callable(checkpoint_module.profile_checkpoint_io)
