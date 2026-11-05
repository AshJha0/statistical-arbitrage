"""Loaders: day-column monotonicity, non-finite cells, config validation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from statarb import load_config, load_csv


def _write(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / name
    p.write_text(text)
    return p


def test_csv_day_column_must_be_monotone(tmp_path: Path) -> None:
    """Proposed test 16 (MAJ-6)."""
    dup = _write(tmp_path, "dup.csv", "day,a\n0,1\n1,2\n2,3\n2,4\n3,5\n")
    with pytest.raises(ValueError, match="day column must be strictly increasing"):
        load_csv(dup)
    unsorted = _write(tmp_path, "unsorted.csv", "day,a\n0,1\n2,2\n1,3\n3,4\n")
    with pytest.raises(ValueError, match="day column must be strictly increasing"):
        load_csv(unsorted)
    # gaps are allowed (strictly increasing), sorted data loads
    gap = _write(tmp_path, "gap.csv", "day,a\n0,1\n1,2\n3,3\n4,4\n")
    d = load_csv(gap)
    assert np.array_equal(d["day"], [0, 1, 3, 4])
    # a file without a day column is not subject to the check
    nod = _write(tmp_path, "nod.csv", "b,a\n5,1\n2,2\n")
    assert load_csv(nod)["b"].tolist() == [5.0, 2.0]


def test_csv_rejects_nonfinite_ragged_and_nonnumeric(tmp_path: Path) -> None:
    for text, msg in (
        ("day,a\n0,1\n1,nan\n", "non-numeric"),
        ("day,a\n0,1\n1,inf\n", "non-numeric"),
        ("day,a\n0,1\n1,1e999\n", "non-finite"),
        ("day,a\n0,1\n1\n", "ragged"),
        ("day,a\n0,1\n1,abc\n", "non-numeric"),
        ("day,a\n", "empty"),
        ("day,day\n0,1\n", "duplicate column"),
    ):
        p = _write(tmp_path, "bad.csv", text)
        with pytest.raises(ValueError, match=msg):
            load_csv(p)
    with pytest.raises(ValueError, match="no such data file"):
        load_csv(tmp_path / "missing.csv")


def test_bundled_files_pass_validation(data_dir: Path) -> None:
    for name in ("pair_equity.csv", "fx_triangle.csv", "random_walks.csv"):
        d = load_csv(data_dir / name)
        assert np.array_equal(d["day"], np.arange(1500))


def test_config_rejects_nonfinite(tmp_path: Path) -> None:
    p = _write(tmp_path, "cfg.json", '{"cost_bps": NaN, "window": 60}')
    with pytest.raises(ValueError, match="finite"):
        load_config(p)
    p = _write(tmp_path, "cfg2.json", '{"cost_bps": true}')
    with pytest.raises(ValueError, match="finite"):
        load_config(p)
    p = _write(tmp_path, "cfg3.json", '[1, 2]')
    with pytest.raises(ValueError, match="object"):
        load_config(p)
    with pytest.raises(ValueError, match="no such config"):
        load_config(tmp_path / "nope.json")


@pytest.mark.parametrize("cell", ["0x10", "1_000", "1f", "1d", "1.5L", "1e", "e5",
                                  ".", "+", "--1", "1,5e", "Infinity", "NaN", "\u0663"])
def test_csv_plain_number_grammar_only(tmp_path: Path, cell: str) -> None:
    p = tmp_path / "bad.csv"
    p.write_text(f"a,b\n1,{cell}\n", encoding="utf8")
    with pytest.raises(ValueError):
        load_csv(p)


@pytest.mark.parametrize("cell,value", [("1", 1.0), ("-1.5", -1.5), ("+2.", 2.0),
                                        (".5", 0.5), ("1e3", 1000.0), ("2.5E-1", 0.25)])
def test_csv_accepts_plain_numbers(tmp_path: Path, cell: str, value: float) -> None:
    assert load_csv(_write(tmp_path, "ok.csv", f"a\n{cell}\n"))["a"][0] == value


@pytest.mark.parametrize("cell", ["1.0", "1.5", "1e0"])
def test_csv_day_cells_must_be_integers(tmp_path: Path, cell: str) -> None:
    with pytest.raises(ValueError, match="non-numeric"):
        load_csv(_write(tmp_path, "d.csv", f"day,a\n0,1\n{cell},2\n"))


@pytest.mark.parametrize("text", [
    '{"window": 60} trailing', '{"window": 60}{"a": 1}', '{"window": 60.0}',
    '{"window": 6e1}', '{"max_hold_days": 1.5}', '{"cooldown_days": 2.0}',
    '{"adf_maxlag": 1.0}', '{"a": 0x10}', '{"a": 1_000}', '{"a": 1f}',
    '{"a": 1e999}', '{"a": 1' + '0' * 400 + '}',
])
def test_config_strict(tmp_path: Path, text: str) -> None:
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, "c.json", text))
