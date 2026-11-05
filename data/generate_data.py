"""Regenerate the bundled datasets and golden values (fixed seeds).

Writes (all deterministic):
    pair_equity.csv   1500 days, cointegrated pair, true beta = 1.7,
                      regime break at day 1050 (beta shifts to 1.1 and the
                      spread turns near-unit-root).
    fx_triangle.csv   EURUSD vs the synthetic triangle basket
                      EURGBP * GBPUSD (cointegration lives in logs).
    random_walks.csv  two INDEPENDENT random walks (spurious-regression
                      teaching case).
    config.json       signal / cost parameters shared by all languages.
    golden/golden.json  cross-language golden cases, computed by the
                      Python reference implementation FROM THE CSVs AS
                      WRITTEN (so file round-trip cannot break agreement)
                      and cross-validated against statsmodels first.

Run:  python3 generate_data.py   (from this directory)
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "python" / "src"))

from statarb import (  # noqa: E402
    config_from_dict,
    engle_granger_both,
    rolling_walk_forward,
    select_lag,
    walk_forward,
    BacktestConfig,
    adf_test,
    backtest_pair,
    critical_values,
    engle_granger,
    half_life,
    johansen_trace,
    kalman_hedge,
    load_csv,
    ols,
    rolling_zscore,
    spread,
)

FMT = "%.10f"


def write_csv(path: Path, columns: dict[str, np.ndarray]) -> None:
    names = list(columns)
    n = len(next(iter(columns.values())))
    with path.open("w", newline="\n") as f:
        f.write(",".join(names) + "\n")
        for i in range(n):
            f.write(
                ",".join(
                    str(int(columns[c][i])) if c == "day" else FMT % columns[c][i]
                    for c in names
                )
                + "\n"
            )


def gen_pair_equity() -> None:
    """Cointegrated equity pair: y = 12 + 1.7 x + OU(0.92) until day 1050;
    at the break the true beta ramps linearly from 1.7 to 1.1 over days
    1050..1080 (then stays at 1.1) and the spread innovation becomes
    near-unit-root (phi = 0.995) — the break regime."""
    rng = np.random.default_rng(42)
    T, T_break, T_ramp = 1500, 1050, 30
    x = 100.0 * np.exp(np.cumsum(rng.normal(2.0e-4, 0.01, T)))
    eps = rng.normal(0.0, 0.6, T)
    s = np.zeros(T)
    for t in range(1, T):
        phi = 0.92 if t < T_break else 0.995
        s[t] = phi * s[t - 1] + eps[t]
    a0, b1, b2 = 12.0, 1.7, 1.1
    b = np.full(T, b1)
    b[T_break : T_break + T_ramp + 1] = np.linspace(b1, b2, T_ramp + 1)
    b[T_break + T_ramp + 1 :] = b2
    y = a0 + b * x + s
    write_csv(
        HERE / "pair_equity.csv",
        {"day": np.arange(T), "stock_y": y, "stock_x": x},
    )


def gen_fx_triangle() -> None:
    """FX triangle: log(EURUSD) = log(EURGBP) + log(GBPUSD) + OU residual.
    basket = EURGBP * GBPUSD is the synthetic replicating basket."""
    rng = np.random.default_rng(7)
    T = 1500
    gbpusd = 1.27 * np.exp(np.cumsum(rng.normal(0.0, 0.004, T)))
    eurgbp = 0.86 * np.exp(np.cumsum(rng.normal(0.0, 0.003, T)))
    u = np.zeros(T)
    eps = rng.normal(0.0, 8.0e-4, T)
    for t in range(1, T):
        u[t] = 0.9 * u[t - 1] + eps[t]
    basket = eurgbp * gbpusd
    eurusd = basket * np.exp(u)
    write_csv(
        HERE / "fx_triangle.csv",
        {
            "day": np.arange(T),
            "eurusd": eurusd,
            "gbpusd": gbpusd,
            "eurgbp": eurgbp,
            "basket": basket,
        },
    )


def gen_random_walks() -> None:
    """Two independent geometric random walks — the spurious case."""
    rng = np.random.default_rng(100)
    T = 1500
    w1 = 50.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, T)))
    w2 = 80.0 * np.exp(np.cumsum(rng.normal(0.0, 0.012, T)))
    write_csv(
        HERE / "random_walks.csv", {"day": np.arange(T), "walk_a": w1, "walk_b": w2}
    )


CONFIG = {
    "window": 60,
    "entry_z": 2.0,
    "exit_z": 0.5,
    "stop_z": 4.0,
    "cost_bps": 1.0,
    "slippage_bps": 0.5,
    "capital": 1000000.0,
    "insample_frac": 0.6,
    "adf_maxlag": 1,
    "significance": 0.05,
    "kalman_delta": 0.0001,
    "kalman_r": 0.001,
    # Round risk limits chosen before looking at results; NOT tuned.
    "pnl_stop_frac": 0.02,
    "max_hold_days": 60,
    "cooldown_days": 20,
}

#: Backtest scenarios shared by the golden cases: (sizing, risk limits on).
SCENARIOS = {
    "limits_beta": ("beta_hedged", True),
    "limits_dollar": ("equal_dollar", True),
    "plain_beta": ("beta_hedged", False),
    "plain_dollar": ("equal_dollar", False),
}
EQUITY_DAYS = (0, 250, 500, 900, 1050, 1200, 1499)
METRIC_FIELDS = ("total_return", "ann_return", "sharpe", "max_drawdown",
                 "n_trades", "win_rate", "avg_holding_days", "turnover")
#: Pinned rolling walk-forward geometry (not tuned).
ROLL_TRAIN, ROLL_TEST = 500, 100


def scenario_config(name: str) -> BacktestConfig:
    sizing, limits = SCENARIOS[name]
    off = {} if limits else {"pnl_stop_frac": 0.0, "max_hold_days": 0,
                             "cooldown_days": 0}
    return config_from_dict(CONFIG, sizing=sizing, **off)


def cross_validate(pair: dict, walks: dict, fx: dict) -> None:
    """Assert our OLS/ADF agree with statsmodels before golden is written."""
    import statsmodels.api as sm
    from statsmodels.tsa.adfvalues import mackinnoncrit
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.tsa.vector_ar.vecm import coint_johansen

    y, x = pair["stock_y"], pair["stock_x"]
    n_is = int(math.floor(len(y) * CONFIG["insample_frac"] + 1e-9))

    ours = ols(y, x, intercept=True)
    sm_fit = sm.OLS(y, sm.add_constant(x)).fit()
    assert np.allclose(ours.params, sm_fit.params, rtol=0, atol=1e-9), "OLS params"
    assert abs(ours.r2 - sm_fit.rsquared) < 1e-10, "OLS R^2"
    assert np.allclose(ours.se, sm_fit.bse, rtol=0, atol=1e-9), "OLS SEs"

    resid = ols(y[:n_is], x[:n_is], intercept=True).resid
    ours_adf = adf_test(resid, maxlag=1, nvars=2)
    sm_t = adfuller(resid, maxlag=1, regression="c", autolag=None)[0]
    assert abs(ours_adf.tstat - sm_t) < 1e-8, f"ADF resid {ours_adf.tstat} vs {sm_t}"

    ours_rw = adf_test(walks["walk_a"], maxlag=1, nvars=1)
    sm_rw = adfuller(walks["walk_a"], maxlag=1, regression="c", autolag=None)
    assert abs(ours_rw.tstat - sm_rw[0]) < 1e-8, "ADF random walk"
    # finite-sample MacKinnon (2010) critical values at the regression nobs
    for key in ("1%", "5%", "10%"):
        assert abs(ours_rw.crit[key] - sm_rw[4][key]) < 1e-10, f"ADF cv {key}"
    for T in (20, 50, 898, 1498):
        for n in (1, 2, 3):
            assert np.allclose(critical_values(n, T), mackinnoncrit(n, "c", T),
                               atol=1e-12), f"cv N={n} T={T}"

    # Johansen: pinned Cholesky + Jacobi route vs an independent numpy
    # generalised-eigen solution and statsmodels' coint_johansen.
    for Y in (np.column_stack([y[:n_is], x[:n_is]]),
              np.column_stack([np.log(fx["eurusd"]), np.log(fx["gbpusd"]),
                               np.log(fx["eurgbp"])])):
        T, n = Y.shape
        dY = np.diff(Y, axis=0)
        Z0, Z1 = dY[1:], Y[1 : T - 1]
        ZD = np.column_stack([np.ones(T - 2), dY[:-1]])
        proj = ZD @ np.linalg.lstsq(ZD, np.column_stack([Z0, Z1]), rcond=None)[0]
        R = np.column_stack([Z0, Z1]) - proj
        R0, R1 = R[:, :n], R[:, n:]
        t_eff = T - 2
        S00, S01, S11 = R0.T @ R0 / t_eff, R0.T @ R1 / t_eff, R1.T @ R1 / t_eff
        M = S01.T @ np.linalg.solve(S00, S01)
        lam_ref = np.sort(np.real(np.linalg.eigvals(np.linalg.solve(S11, M))))[::-1]
        ours_jo = johansen_trace(Y)
        assert np.allclose(ours_jo.eigenvalues, lam_ref, atol=1e-8), "Johansen eig"
        sm_jo = coint_johansen(Y, det_order=0, k_ar_diff=1)
        assert np.allclose(ours_jo.trace_stats, sm_jo.lr1, atol=1e-6), "Johansen trace"
    for series in (resid, walks["walk_a"]):
        sm_lag = adfuller(series, maxlag=5, regression="c", autolag="BIC")[2]
        assert select_lag(series, 5, 1) == sm_lag, "select_lag vs adfuller BIC"
    print("statsmodels / numpy cross-validation: OK")


def build_golden() -> dict:
    pair = load_csv(HERE / "pair_equity.csv")
    walks = load_csv(HERE / "random_walks.csv")
    fx = load_csv(HERE / "fx_triangle.csv")
    cross_validate(pair, walks, fx)

    y, x = pair["stock_y"], pair["stock_x"]
    n_is = int(math.floor(len(y) * CONFIG["insample_frac"] + 1e-9))
    cases: list[dict] = []

    def case(name: str, inputs: dict, expect: dict, tol: float) -> None:
        cases.append({"name": name, "inputs": inputs, "expect": expect, "tol": tol})

    pair_in = {"data": "pair_equity.csv"}
    ins_in = {"data": "pair_equity.csv", "insample_days": n_is}

    fit = ols(y, x, intercept=True)
    case("ols_pair_alpha", pair_in, {"value": fit.params[0]}, 1e-10)
    case("ols_pair_beta", pair_in, {"value": fit.params[1]}, 1e-10)
    case("ols_pair_r2", pair_in, {"value": fit.r2}, 1e-10)

    eg = engle_granger(y[:n_is], x[:n_is], maxlag=CONFIG["adf_maxlag"])
    case("adf_resid_insample_tstat", ins_in, {"value": eg.adf.tstat}, 1e-8)
    case("adf_resid_insample_crit5", ins_in, {"value": eg.adf.crit["5%"]}, 1e-10)
    case("eg_pair_insample_coint", ins_in,
         {"value": 1.0 if eg.cointegrated(0.05) else 0.0}, 1e-9)

    rw = adf_test(walks["walk_a"], maxlag=CONFIG["adf_maxlag"], nvars=1)
    rw_in = {"data": "random_walks.csv", "column": "walk_a"}
    case("adf_random_walk_tstat", rw_in, {"value": rw.tstat}, 1e-8)
    case("adf_random_walk_reject", rw_in,
         {"value": 1.0 if rw.reject(0.05) else 0.0}, 1e-9)
    # MacKinnon (2010) finite-sample 5% critical value at T_eff = 1498 and
    # the interpolated p-value through the finite-sample knots.
    case("adf_random_walk_crit5", rw_in, {"value": rw.crit["5%"]}, 1e-10)
    case("adf_random_walk_pvalue", rw_in, {"value": rw.pvalue}, 1e-10)

    eg_sp = engle_granger(walks["walk_a"], walks["walk_b"],
                          maxlag=CONFIG["adf_maxlag"])
    case("eg_spurious_coint", {"data": "random_walks.csv"},
         {"value": 1.0 if eg_sp.cointegrated(0.05) else 0.0}, 1e-9)

    jo = johansen_trace(np.column_stack([y[:n_is], x[:n_is]]))
    case("johansen_pair_trace_r0", ins_in, {"value": jo.trace_stats[0]}, 1e-6)
    case("johansen_pair_trace_r1", ins_in, {"value": jo.trace_stats[1]}, 1e-6)
    case("johansen_pair_rank", ins_in, {"value": float(jo.rank)}, 1e-9)

    case("halflife_insample_spread", ins_in, {"value": half_life(eg.resid)}, 1e-8)

    km = kalman_hedge(y, x, delta=CONFIG["kalman_delta"], r=CONFIG["kalman_r"])
    kal_in = {"data": "pair_equity.csv", "delta": CONFIG["kalman_delta"],
              "r": CONFIG["kalman_r"]}
    case("kalman_beta_day500", {**kal_in, "day": 500}, {"value": km.beta[500]}, 1e-8)
    case("kalman_beta_day1200", {**kal_in, "day": 1200},
         {"value": km.beta[1200]}, 1e-8)
    case("kalman_beta_final", {**kal_in, "day": len(y) - 1},
         {"value": km.beta[-1]}, 1e-8)

    sp = spread(y, x, eg.alpha, eg.beta)
    z = rolling_zscore(sp, CONFIG["window"])
    z_in = {"data": "pair_equity.csv", "window": CONFIG["window"],
            "insample_days": n_is}
    case("zscore_day200", {**z_in, "day": 200}, {"value": z[200]}, 1e-10)
    case("zscore_day1100", {**z_in, "day": 1100}, {"value": z[1100]}, 1e-10)

    cfg = scenario_config("limits_beta")
    bt = backtest_pair(y, x, eg.alpha, eg.beta, cfg)
    bt_in = {"data": "pair_equity.csv", "config": "config.json",
             "insample_days": n_is}
    case("backtest_total_return", bt_in, {"value": bt.metrics["total_return"]}, 1e-8)
    case("backtest_sharpe", bt_in, {"value": bt.metrics["sharpe"]}, 1e-8)
    case("backtest_max_drawdown", bt_in, {"value": bt.metrics["max_drawdown"]}, 1e-8)
    case("backtest_n_trades", bt_in, {"value": bt.metrics["n_trades"]}, 1e-9)

    # ---- v1.2 cases.  Dispatch on inputs["kind"]; see API_SPEC section 11.
    def bt_field(res, field: str) -> float:
        if field == "n_stops":
            return float(res.n_stops)
        if field == "equity_sum":
            return float(np.sum(res.equity))
        return float(res.metrics[field])

    def bt_tol(field: str) -> float:
        if field in ("n_trades", "n_stops"):
            return 1e-9
        return 1e-3 if field == "equity_sum" else 1e-8

    fields = METRIC_FIELDS + ("n_stops", "equity_sum")
    z_paths = []
    for sc in SCENARIOS:
        res = backtest_pair(y, x, eg.alpha, eg.beta, scenario_config(sc))
        z_paths.append(res.zscore)
        base = {"data": "pair_equity.csv", "insample_days": n_is, "scenario": sc}
        for f in fields:
            case(f"bt_{sc}_{f}", {**base, "kind": "backtest", "field": f},
                 {"value": bt_field(res, f)}, bt_tol(f))
        for d in EQUITY_DAYS:
            case(f"bt_{sc}_equity_day{d}",
                 {**base, "kind": "backtest_equity", "day": d},
                 {"value": float(res.equity[d])}, 1e-6)
        for i, tr in enumerate(res.trades):
            vals = {"entry": float(tr.entry_day), "exit": float(tr.exit_day),
                    "direction": float(tr.direction), "pnl": float(tr.pnl),
                    "stopped": 1.0 if tr.stopped else 0.0}
            for f, v in vals.items():
                case(f"bt_{sc}_trade{i}_{f}",
                     {**base, "kind": "backtest_trade", "index": i, "field": f},
                     {"value": v}, 1e-6 if f == "pnl" else 1e-9)

    wf = walk_forward(y, x, cfg, CONFIG["insample_frac"])
    wf_in = {"data": "pair_equity.csv", "scenario": "limits_beta"}
    for seg, res in (("insample", wf.insample), ("oos", wf.oos)):
        z_paths.append(res.zscore)
        for f in fields:
            case(f"wf_{seg}_{f}",
                 {**wf_in, "kind": "walk_forward", "segment": seg, "field": f},
                 {"value": bt_field(res, f)}, bt_tol(f))
    for f, v in (("alpha", wf.alpha), ("beta", wf.beta), ("split", float(wf.split))):
        case(f"wf_fit_{f}", {**wf_in, "kind": "walk_forward", "segment": "fit",
                             "field": f}, {"value": v}, 1e-9)

    for sc in ("limits_beta", "plain_beta"):
        rw = rolling_walk_forward(y, x, scenario_config(sc), ROLL_TRAIN, ROLL_TEST)
        base = {"data": "pair_equity.csv", "scenario": sc,
                "train": ROLL_TRAIN, "test": ROLL_TEST}
        for f in fields:
            case(f"rwf_{sc}_{f}", {**base, "kind": "rolling", "field": f},
                 {"value": bt_field(rw, f)}, bt_tol(f))
        case(f"rwf_{sc}_n_blocks", {**base, "kind": "rolling", "field": "n_blocks"},
             {"value": float(len(rw.blocks))}, 1e-9)
        for k, blk in enumerate(rw.blocks):
            z_paths.append(blk.result.zscore)
            vals = {"start": float(blk.start), "end": float(blk.end),
                    "alpha": blk.alpha, "beta": blk.beta,
                    "skipped": 1.0 if blk.skipped else 0.0,
                    "total_return": blk.result.metrics["total_return"],
                    "sharpe": blk.result.metrics["sharpe"],
                    "n_trades": blk.result.metrics["n_trades"]}
            for f, v in vals.items():
                case(f"rwf_{sc}_block{k}_{f}",
                     {**base, "kind": "rolling_block", "block": k, "field": f},
                     {"value": v}, 1e-8)

    fx_y = np.column_stack([np.log(fx["eurusd"]), np.log(fx["gbpusd"]),
                            np.log(fx["eurgbp"])])
    j3 = johansen_trace(fx_y)
    fx_in = {"data": "fx_triangle.csv", "logs": "eurusd,gbpusd,eurgbp"}
    for r in range(3):
        case(f"johansen_fx_trace_r{r}", fx_in, {"value": j3.trace_stats[r]}, 1e-6)
        case(f"johansen_fx_vector{r}", fx_in, {"value": j3.beta[r, 0]}, 1e-6)
    case("johansen_fx_rank", fx_in, {"value": float(j3.rank)}, 1e-9)

    for d, tag in ((500, "day500"), (1200, "day1200"), (len(y) - 1, "final")):
        case(f"kalman_alpha_{tag}", {**kal_in, "day": d}, {"value": km.alpha[d]}, 1e-8)
        case(f"kalman_z_{tag}", {**kal_in, "day": d}, {"value": km.z[d]}, 1e-8)

    for lag in (0, 3):
        a = adf_test(eg.resid, maxlag=lag, nvars=2)
        lag_in = {**ins_in, "maxlag": lag, "nvars": 2}
        case(f"adf_resid_lag{lag}_tstat", lag_in, {"value": a.tstat}, 1e-8)
        case(f"adf_resid_lag{lag}_crit5", lag_in, {"value": a.crit["5%"]}, 1e-10)
        case(f"adf_resid_lag{lag}_nobs", lag_in, {"value": float(a.nobs)}, 1e-9)
    r3 = ols(fx_y[:, 0], fx_y[:, 1:], intercept=True).resid
    a3 = adf_test(r3, maxlag=CONFIG["adf_maxlag"], nvars=3)
    a3_in = {**fx_in, "maxlag": CONFIG["adf_maxlag"], "nvars": 3}
    case("adf_fx_nvars3_tstat", a3_in, {"value": a3.tstat}, 1e-8)
    case("adf_fx_nvars3_crit1", a3_in, {"value": a3.crit["1%"]}, 1e-10)
    case("adf_fx_nvars3_crit5", a3_in, {"value": a3.crit["5%"]}, 1e-10)
    case("adf_fx_nvars3_crit10", a3_in, {"value": a3.crit["10%"]}, 1e-10)
    case("adf_fx_nvars3_pvalue", a3_in, {"value": a3.pvalue}, 1e-10)

    case("select_lag_resid", {**ins_in, "max_lag": 5, "nvars": 2},
         {"value": float(select_lag(eg.resid, 5, 2))}, 1e-9)
    case("select_lag_walk", {**rw_in, "max_lag": 5, "nvars": 1},
         {"value": float(select_lag(walks["walk_a"], 5, 1))}, 1e-9)

    both = engle_granger_both(y[:n_is], x[:n_is], maxlag=CONFIG["adf_maxlag"])
    case("eg_both_insample_x_on_y_tstat", ins_in,
         {"value": both.x_on_y.adf.tstat}, 1e-8)
    case("eg_both_insample_coint", ins_in,
         {"value": 1.0 if both.cointegrated(0.05) else 0.0}, 1e-9)

    # Sentinels on closed-form series of 20 points (JSON has no inf, so the
    # +inf case is stored as the flag 1.0).
    growth = 1.1 ** np.arange(20)
    alternating = np.where(np.arange(20) % 2 == 0, 1.0, -1.0)
    assert math.isinf(half_life(growth)) and half_life(alternating) == 0.0
    case("halflife_growth_is_inf", {"series": "1.1^t", "n": 20}, {"value": 1.0}, 1e-9)
    case("halflife_alternating_zero", {"series": "(-1)^t", "n": 20}, {"value": 0.0}, 1e-9)

    # Smallest distance of any z from any threshold across every run above.
    # Ports assert it exceeds 1e-6: below that, rounding could flip a trade.
    margin = min(
        float(np.nanmin(np.abs(np.abs(zp) - th)))
        for zp in z_paths if not np.all(np.isnan(zp))
        for th in (CONFIG["entry_z"], CONFIG["exit_z"], CONFIG["stop_z"])
    )
    assert margin > 1e-6, f"threshold margin too small: {margin}"
    case("zscore_threshold_margin", {"kind": "margin"}, {"value": margin}, 1e-9)

    # sanity for the teaching narrative
    assert eg.cointegrated(0.05), "in-sample pair must be cointegrated"
    assert not eg_sp.cointegrated(0.05), "spurious pair must NOT be cointegrated"
    assert jo.rank >= 1, "Johansen must find rank >= 1 in-sample"
    assert bt.metrics["n_trades"] >= 3, "bundled backtest must trade"
    assert bt.n_stops >= 1, "break regime must trigger at least one stop"
    assert abs(km.beta[500] - 1.7) < 0.1, "Kalman should track beta ~1.7 pre-break"
    assert abs(km.beta[-1] - 1.1) < 0.1, "Kalman should adapt to beta ~1.1 post-break"
    return {"cases": cases}


def main() -> None:
    gen_pair_equity()
    gen_fx_triangle()
    gen_random_walks()
    with (HERE / "config.json").open("w", newline="\n") as f:
        json.dump(CONFIG, f, indent=2)
        f.write("\n")
    golden = build_golden()
    with (HERE / "golden" / "golden.json").open("w", newline="\n") as f:
        json.dump(golden, f, indent=2)
        f.write("\n")
    print(f"wrote {len(golden['cases'])} golden cases")


if __name__ == "__main__":
    main()
