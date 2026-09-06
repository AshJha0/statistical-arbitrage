"""Tiny CSV / JSON loaders so the package stays numpy-only.

Validation (pinned across languages):

* the file must exist and hold a header plus at least one data row;
* every row must have exactly as many cells as the header (no ragged rows);
* every cell must parse as a FINITE number (``nan``/``inf`` are rejected,
  not silently propagated into the estimators);
* if a column named ``day`` exists it must be strictly increasing — an
  unsorted, duplicated or re-merged time index silently changes every lag,
  z-score and fill in the backtest, so it is refused at the boundary.  The
  models further assume the rows are EQUALLY SPACED consecutive
  observations (next-close execution, lag-1 differences, sqrt(252));
  gaps are the caller's responsibility (see API_SPEC section 10).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

#: Name of the optional time-index column that is validated on load.
DAY_COLUMN: str = "day"


def _validate_day_column(day: np.ndarray, path: Path) -> None:
    if day.shape[0] >= 2 and not np.all(np.diff(day) > 0.0):
        raise ValueError(
            f"day column must be strictly increasing (sorted, no duplicates) in {path}"
        )


def load_csv(path: str | Path) -> dict[str, np.ndarray]:
    """Load a simple headered numeric CSV into {column_name: float array}.

    Raises:
        ValueError: missing file, empty file, ragged/non-numeric/non-finite
            rows, or a ``day`` column that is not strictly increasing.
    """
    p = Path(path)
    if not p.is_file():
        raise ValueError(f"no such data file: {p}")
    lines = p.read_text().strip().splitlines()
    if len(lines) < 2:
        raise ValueError(f"empty data file: {p}")
    header = [h.strip() for h in lines[0].split(",")]
    ncol = len(header)
    if len(set(header)) != ncol:
        raise ValueError(f"duplicate column names in {p}")
    data = np.empty((len(lines) - 1, ncol))
    for i, line in enumerate(lines[1:]):
        parts = line.split(",")
        if len(parts) != ncol:
            raise ValueError(f"ragged row {i + 2} in {p}")
        try:
            row = [float(v) for v in parts]
        except ValueError as exc:
            raise ValueError(f"non-numeric value at row {i + 2} in {p}") from exc
        if not all(math.isfinite(v) for v in row):
            raise ValueError(f"non-finite value at row {i + 2} in {p}")
        data[i] = row
    out = {name: data[:, j].copy() for j, name in enumerate(header)}
    if DAY_COLUMN in out:
        _validate_day_column(out[DAY_COLUMN], p)
    return out


def load_config(path: str | Path) -> dict:
    """Load the bundled config.json (flat dict of finite numeric scalars).

    Raises:
        ValueError: missing file, non-object document, or any value that is
            not a finite number (JSON ``NaN``/``Infinity`` are rejected).
    """
    p = Path(path)
    if not p.is_file():
        raise ValueError(f"no such config file: {p}")
    with p.open() as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError("config.json must contain a JSON object")
    for k, v in cfg.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError(f"config key '{k}' is not a finite numeric scalar")
    return cfg
