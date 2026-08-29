"""Golden-value suite: every case in data/golden/golden.json is recomputed
from the bundled CSVs and must agree within its stated tolerance."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from statarb import (
    BacktestConfig,
    adf_test,
    backtest_pair,
    engle_granger,
    half_life,
    johansen_trace,
    kalman_hedge,
    load_config,
    load_csv,
    ols,
    rolling_zscore,
    spread,
)

DATA = Path(__file__).resolve().parents[2] / "data"

with (DATA / "golden" / "golden.json").open() as f:
    GOLDEN = json.load(f)["cases"]


def _compute(name: str, inputs: dict) -> float:
    cfg = load_config(DATA / "config.json")
    pair = load_csv(DATA / "pair_equity.csv")
    y, x = pair["stock_y"], pair["stock_x"]
    n_is = int(len(y) * cfg["insample_frac"])

    if name.startswith("ols_pair"):
        fit = ols(y, x, intercept=True)
        return {"ols_pair_alpha": fit.params[0],
                "ols_pair_beta": fit.params[1],
                "ols_pair_r2": fit.r2}[name]

    if name in ("adf_random_walk_tstat", "adf_random_walk_reject"):
        walks = load_csv(DATA / "random_walks.csv")
        res = adf_test(walks[inputs["column"]], maxlag=cfg["adf_maxlag"], nvars=1)
        return res.tstat if name.endswith("tstat") else float(res.reject(0.05))

    if name == "eg_spurious_coint":
        walks = load_csv(DATA / "random_walks.csv")
        eg = engle_granger(walks["walk_a"], walks["walk_b"],
                           maxlag=cfg["adf_maxlag"])
        return float(eg.cointegrated(0.05))

    eg = engle_granger(y[:n_is], x[:n_is], maxlag=cfg["adf_maxlag"])

    if name == "adf_resid_insample_tstat":
        return eg.adf.tstat
    if name == "eg_pair_insample_coint":
        return float(eg.cointegrated(0.05))
    if name == "halflife_insample_spread":
        return half_life(eg.resid)
    if name.startswith("johansen_pair"):
        jo = johansen_trace(np.column_stack([y[:n_is], x[:n_is]]))
        return {"johansen_pair_trace_r0": jo.trace_stats[0],
                "johansen_pair_trace_r1": jo.trace_stats[1],
                "johansen_pair_rank": float(jo.rank)}[name]
    if name.startswith("kalman_beta"):
        km = kalman_hedge(y, x, delta=cfg["kalman_delta"], r=cfg["kalman_r"])
        return km.beta[int(inputs["day"])]
    if name.startswith("zscore_day"):
        z = rolling_zscore(spread(y, x, eg.alpha, eg.beta), cfg["window"])
        return z[int(inputs["day"])]
    if name.startswith("backtest_"):
        bt_cfg = BacktestConfig(
            window=cfg["window"], entry_z=cfg["entry_z"], exit_z=cfg["exit_z"],
            stop_z=cfg["stop_z"], cost_bps=cfg["cost_bps"],
            slippage_bps=cfg["slippage_bps"], capital=cfg["capital"],
        )
        bt = backtest_pair(y, x, eg.alpha, eg.beta, bt_cfg)
        key = name.removeprefix("backtest_")
        return bt.metrics[key]
    raise AssertionError(f"unknown golden case: {name}")


@pytest.mark.parametrize("case", GOLDEN, ids=[c["name"] for c in GOLDEN])
def test_golden_case(case: dict) -> None:
    got = _compute(case["name"], case["inputs"])
    assert got == pytest.approx(case["expect"]["value"], abs=case["tol"])


def test_golden_schema_flat_scalars() -> None:
    """Golden file keeps the flat cross-language schema: scalar inputs and
    a single scalar expectation per case."""
    assert len(GOLDEN) >= 18
    names = [c["name"] for c in GOLDEN]
    assert len(names) == len(set(names))
    for c in GOLDEN:
        assert set(c) == {"name", "inputs", "expect", "tol"}
        for v in c["inputs"].values():
            assert isinstance(v, (int, float, str))
        assert set(c["expect"]) == {"value"}
        assert isinstance(c["expect"]["value"], (int, float))
        assert c["tol"] > 0.0
