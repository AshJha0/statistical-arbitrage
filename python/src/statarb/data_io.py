"""Tiny CSV / JSON loaders so the package stays numpy-only."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_csv(path: str | Path) -> dict[str, np.ndarray]:
    """Load a simple headered numeric CSV into {column_name: float array}.

    Raises:
        ValueError: missing file, empty file, or ragged/non-numeric rows.
    """
    p = Path(path)
    if not p.is_file():
        raise ValueError(f"no such data file: {p}")
    lines = p.read_text().strip().splitlines()
    if len(lines) < 2:
        raise ValueError(f"empty data file: {p}")
    header = [h.strip() for h in lines[0].split(",")]
    ncol = len(header)
    data = np.empty((len(lines) - 1, ncol))
    for i, line in enumerate(lines[1:]):
        parts = line.split(",")
        if len(parts) != ncol:
            raise ValueError(f"ragged row {i + 2} in {p}")
        try:
            data[i] = [float(v) for v in parts]
        except ValueError as exc:
            raise ValueError(f"non-numeric value at row {i + 2} in {p}") from exc
    return {name: data[:, j].copy() for j, name in enumerate(header)}


def load_config(path: str | Path) -> dict:
    """Load the bundled config.json (flat dict of scalars)."""
    p = Path(path)
    if not p.is_file():
        raise ValueError(f"no such config file: {p}")
    with p.open() as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError("config.json must contain a JSON object")
    return cfg
