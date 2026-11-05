"""Golden-value suite: every case in data/golden/golden.json is recomputed
from the bundled CSVs and must agree within its stated tolerance."""
from __future__ import annotations

import json
from functools import cache
from pathlib import Path

import numpy as np
import pytest

from statarb import (
    adf_test,
    backtest_pair,
    config_from_dict,
    engle_granger,
    engle_granger_both,
    half_life,
    johansen_trace,
    kalman_hedge,
    load_config,
    load_csv,
    ols,
    rolling_walk_forward,
    rolling_zscore,
    select_lag,
    spread,
    walk_forward,
)

DATA = Path(__file__).resolve().parents[2] / "data"

with (DATA / "golden" / "golden.json").open() as f:
    GOLDEN = json.load(f)["cases"]

#: Exact number of golden cases (kept in step with generate_data.py and the
#: C++/Rust/Java golden suites).
GOLDEN_COUNT = 928


SIZING = {"limits_beta": "beta_hedged", "limits_dollar": "equal_dollar",
          "plain_beta": "beta_hedged", "plain_dollar": "equal_dollar"}


def _scenario(cfg: dict, name: str):
    off = ({} if name.startswith("limits") else
           {"pnl_stop_frac": 0.0, "max_hold_days": 0, "cooldown_days": 0})
    return config_from_dict(cfg, sizing=SIZING[name], **off)


def _field(res, field: str) -> float:
    if field == "n_stops":
        return float(res.n_stops)
    if field == "equity_sum":
        return float(np.sum(res.equity))
    return res.metrics[field]


@cache
def _bt(name: str):
    cfg = load_config(DATA / "config.json")
    pair = load_csv(DATA / "pair_equity.csv")
    y, x = pair["stock_y"], pair["stock_x"]
    eg = engle_granger(y[:900], x[:900], maxlag=cfg["adf_maxlag"])
    return backtest_pair(y, x, eg.alpha, eg.beta, _scenario(cfg, name))


@cache
def _wf():
    cfg = load_config(DATA / "config.json")
    pair = load_csv(DATA / "pair_equity.csv")
    return walk_forward(pair["stock_y"], pair["stock_x"],
                        _scenario(cfg, "limits_beta"), cfg["insample_frac"])


@cache
def _rw(name: str, train: int, test: int):
    cfg = load_config(DATA / "config.json")
    pair = load_csv(DATA / "pair_equity.csv")
    return rolling_walk_forward(pair["stock_y"], pair["stock_x"],
                                _scenario(cfg, name), train, test)


def _compute_kind(inputs: dict) -> float:
    kind, field = inputs["kind"], inputs.get("field")
    if kind == "backtest":
        return _field(_bt(inputs["scenario"]), field)
    if kind == "backtest_equity":
        return _bt(inputs["scenario"]).equity[inputs["day"]]
    if kind == "backtest_trade":
        tr = _bt(inputs["scenario"]).trades[inputs["index"]]
        return {"entry": tr.entry_day, "exit": tr.exit_day,
                "direction": tr.direction, "pnl": tr.pnl,
                "stopped": float(tr.stopped)}[field]
    if kind == "walk_forward":
        wf = _wf()
        if inputs["segment"] == "fit":
            return {"alpha": wf.alpha, "beta": wf.beta, "split": wf.split}[field]
        return _field(wf.insample if inputs["segment"] == "insample" else wf.oos,
                      field)
    if kind == "rolling":
        rw = _rw(inputs["scenario"], inputs["train"], inputs["test"])
        return float(len(rw.blocks)) if field == "n_blocks" else _field(rw, field)
    if kind == "rolling_block":
        blk = _rw(inputs["scenario"], inputs["train"],
                  inputs["test"]).blocks[inputs["block"]]
        if field in ("start", "end", "alpha", "beta"):
            return getattr(blk, field)
        return float(blk.skipped) if field == "skipped" else blk.result.metrics[field]
    if kind == "margin":
        cfg = load_config(DATA / "config.json")
        paths = [_bt(n).zscore for n in SIZING]
        paths += [_wf().insample.zscore, _wf().oos.zscore]
        for n in ("limits_beta", "plain_beta"):
            paths += [b.result.zscore for b in _rw(n, 500, 100).blocks]
        got = min(float(np.nanmin(np.abs(np.abs(p) - cfg[k])))
                  for p in paths if not np.all(np.isnan(p)) for k in ("entry_z", "exit_z", "stop_z"))
        assert got > 1e-6
        return got
    raise AssertionError(f"unknown golden kind: {kind}")


def _compute_v12(name: str, inputs: dict, cfg: dict, y, x, eg) -> float | None:
    """Name-dispatched cases added in v1.2; None when the name is older."""
    if name.startswith(("johansen_fx", "adf_fx")):
        fx = load_csv(DATA / "fx_triangle.csv")
        Y = np.column_stack([np.log(fx["eurusd"]), np.log(fx["gbpusd"]),
                             np.log(fx["eurgbp"])])
        if name.startswith("adf_fx"):
            res = adf_test(ols(Y[:, 0], Y[:, 1:], intercept=True).resid,
                           maxlag=inputs["maxlag"], nvars=inputs["nvars"])
            return {"tstat": res.tstat, "crit1": res.crit["1%"],
                    "crit5": res.crit["5%"], "crit10": res.crit["10%"],
                    "pvalue": res.pvalue}[name.rsplit("_", 1)[1]]
        jo = johansen_trace(Y)
        tail = name.removeprefix("johansen_fx_")
        if tail == "rank":
            return float(jo.rank)
        if tail.startswith("trace_r"):
            return jo.trace_stats[int(tail[-1])]
        return jo.beta[int(tail[-1]), 0]
    if name.startswith(("kalman_alpha", "kalman_z")):
        km = kalman_hedge(y, x, delta=inputs["delta"], r=inputs["r"])
        path = km.alpha if name.startswith("kalman_alpha") else km.z
        return path[inputs["day"]]
    if name.startswith("adf_resid_lag"):
        res = adf_test(eg.resid, maxlag=inputs["maxlag"], nvars=inputs["nvars"])
        return {"tstat": res.tstat, "crit5": res.crit["5%"],
                "nobs": float(res.nobs)}[name.rsplit("_", 1)[1]]
    if name == "select_lag_resid":
        return float(select_lag(eg.resid, inputs["max_lag"], inputs["nvars"]))
    if name == "select_lag_walk":
        walks = load_csv(DATA / "random_walks.csv")
        return float(select_lag(walks[inputs["column"]], inputs["max_lag"],
                                inputs["nvars"]))
    if name.startswith("eg_both"):
        both = engle_granger_both(y[:900], x[:900], maxlag=cfg["adf_maxlag"])
        if name == "eg_both_insample_coint":
            return float(both.cointegrated(0.05))
        return both.x_on_y.adf.tstat
    if name == "halflife_growth_is_inf":
        return float(np.isinf(half_life(1.1 ** np.arange(inputs["n"]))))
    if name == "halflife_alternating_zero":
        return half_life(np.where(np.arange(inputs["n"]) % 2 == 0, 1.0, -1.0))
    return None


def _compute(name: str, inputs: dict) -> float:
    if "kind" in inputs:
        return _compute_kind(inputs)
    cfg = load_config(DATA / "config.json")
    pair = load_csv(DATA / "pair_equity.csv")
    y, x = pair["stock_y"], pair["stock_x"]
    n_is = int(len(y) * cfg["insample_frac"])

    if name.startswith("ols_pair"):
        fit = ols(y, x, intercept=True)
        return {"ols_pair_alpha": fit.params[0],
                "ols_pair_beta": fit.params[1],
                "ols_pair_r2": fit.r2}[name]

    if name.startswith("adf_random_walk"):
        walks = load_csv(DATA / "random_walks.csv")
        res = adf_test(walks[inputs["column"]], maxlag=cfg["adf_maxlag"], nvars=1)
        return {"adf_random_walk_tstat": res.tstat,
                "adf_random_walk_reject": float(res.reject(0.05)),
                "adf_random_walk_crit5": res.crit["5%"],
                "adf_random_walk_pvalue": res.pvalue}[name]

    if name == "eg_spurious_coint":
        walks = load_csv(DATA / "random_walks.csv")
        eg = engle_granger(walks["walk_a"], walks["walk_b"],
                           maxlag=cfg["adf_maxlag"])
        return float(eg.cointegrated(0.05))

    eg = engle_granger(y[:n_is], x[:n_is], maxlag=cfg["adf_maxlag"])

    got = _compute_v12(name, inputs, cfg, y, x, eg)
    if got is not None:
        return got
    if name == "adf_resid_insample_tstat":
        return eg.adf.tstat
    if name == "adf_resid_insample_crit5":
        return eg.adf.crit["5%"]
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
        return _bt("limits_beta").metrics[name.removeprefix("backtest_")]
    raise AssertionError(f"unknown golden case: {name}")


@pytest.mark.parametrize("case", GOLDEN, ids=[c["name"] for c in GOLDEN])
def test_golden_case(case: dict) -> None:
    got = _compute(case["name"], case["inputs"])
    assert got == pytest.approx(case["expect"]["value"], abs=case["tol"])


def test_golden_schema_flat_scalars() -> None:
    """Golden file keeps the flat cross-language schema: scalar inputs and
    a single scalar expectation per case."""
    assert len(GOLDEN) == GOLDEN_COUNT
    names = [c["name"] for c in GOLDEN]
    assert len(names) == len(set(names))
    for c in GOLDEN:
        assert set(c) == {"name", "inputs", "expect", "tol"}
        for v in c["inputs"].values():
            assert isinstance(v, (int, float, str))
        assert set(c["expect"]) == {"value"}
        assert isinstance(c["expect"]["value"], (int, float))
        assert c["tol"] > 0.0
