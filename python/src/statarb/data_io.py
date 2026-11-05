"""Tiny CSV / JSON loaders so the package stays numpy-only.

Validation (pinned across languages):

* the file must exist and hold a header plus at least one data row;
* every row must have exactly as many cells as the header (no ragged rows);
* every cell must match the plain decimal/exponent grammar and be FINITE
  (``nan``, ``inf``, hex, ``_`` separators and type suffixes such as ``1f``
  are rejected, whatever the host language's number parser would allow);
* if a column named ``day`` exists its cells must be plain integers and it
  must be strictly increasing — an
  unsorted, duplicated or re-merged time index silently changes every lag,
  z-score and fill in the backtest, so it is refused at the boundary.  The
  models further assume the rows are EQUALLY SPACED consecutive
  observations (next-close execution, lag-1 differences, sqrt(252));
  gaps are the caller's responsibility (see API_SPEC section 10).
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np

#: Name of the optional time-index column that is validated on load.
DAY_COLUMN: str = "day"

#: Config keys that must be JSON integers (``60``, not ``60.0`` or ``6e1``).
INTEGER_KEYS: tuple[str, ...] = (
    "window", "adf_maxlag", "max_hold_days", "cooldown_days",
)

_NUMBER = re.compile(r"[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")
_INTEGER = re.compile(r"[+-]?[0-9]+")


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
        cells = [v.strip(" \t") for v in parts]
        for name, cell in zip(header, cells, strict=True):
            grammar = _INTEGER if name == DAY_COLUMN else _NUMBER
            if grammar.fullmatch(cell) is None:
                raise ValueError(f"non-numeric value at row {i + 2} in {p}")
        row = [float(v) for v in cells]
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
            not a finite number (JSON ``NaN``/``Infinity`` are rejected), an
            integer key holding a non-integer, or trailing content.
    """
    p = Path(path)
    if not p.is_file():
        raise ValueError(f"no such config file: {p}")
    cfg = json.loads(p.read_text())  # raises on anything after the object
    if not isinstance(cfg, dict):
        raise ValueError("config.json must contain a JSON object")
    for k, v in cfg.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"config key '{k}' is not a finite numeric scalar")
        if isinstance(v, float) and not math.isfinite(v):
            raise ValueError(f"config key '{k}' is not a finite numeric scalar")
        if isinstance(v, int) and abs(v) > 2**53:
            raise ValueError(f"config key '{k}' is out of range")
        if k in INTEGER_KEYS and not isinstance(v, int):
            raise ValueError(f"config key '{k}' must be an integer")
    return cfg
