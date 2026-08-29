"""Shared fixtures: bundled datasets and config."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from statarb import load_config, load_csv

DATA = Path(__file__).resolve().parents[2] / "data"


@pytest.fixture(scope="session")
def pair() -> dict[str, np.ndarray]:
    return load_csv(DATA / "pair_equity.csv")


@pytest.fixture(scope="session")
def fx() -> dict[str, np.ndarray]:
    return load_csv(DATA / "fx_triangle.csv")


@pytest.fixture(scope="session")
def walks() -> dict[str, np.ndarray]:
    return load_csv(DATA / "random_walks.csv")


@pytest.fixture(scope="session")
def config() -> dict:
    return load_config(DATA / "config.json")


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return DATA
