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
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "python" / "src"))

from statarb import (  # noqa: E402
    BacktestConfig,
    adf_test,
    backtest_pair,
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
}


def cross_validate(pair: dict, walks: dict) -> None:
    """Assert our OLS/ADF agree with statsmodels before golden is written."""
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import adfuller

    y, x = pair["stock_y"], pair["stock_x"]
    n_is = int(len(y) * CONFIG["insample_frac"])

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
    sm_rw = adfuller(walks["walk_a"], maxlag=1, regression="c", autolag=None)[0]
    assert abs(ours_rw.tstat - sm_rw) < 1e-8, "ADF random walk"
    print("statsmodels cross-validation: OK")


def build_golden() -> dict:
    pair = load_csv(HERE / "pair_equity.csv")
    walks = load_csv(HERE / "random_walks.csv")
    cross_validate(pair, walks)

    y, x = pair["stock_y"], pair["stock_x"]
    n_is = int(len(y) * CONFIG["insample_frac"])
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
    case("eg_pair_insample_coint", ins_in,
         {"value": 1.0 if eg.cointegrated(0.05) else 0.0}, 1e-9)

    rw = adf_test(walks["walk_a"], maxlag=CONFIG["adf_maxlag"], nvars=1)
    rw_in = {"data": "random_walks.csv", "column": "walk_a"}
    case("adf_random_walk_tstat", rw_in, {"value": rw.tstat}, 1e-8)
    case("adf_random_walk_reject", rw_in,
         {"value": 1.0 if rw.reject(0.05) else 0.0}, 1e-9)

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

    cfg = BacktestConfig(
        window=CONFIG["window"], entry_z=CONFIG["entry_z"],
        exit_z=CONFIG["exit_z"], stop_z=CONFIG["stop_z"],
        cost_bps=CONFIG["cost_bps"], slippage_bps=CONFIG["slippage_bps"],
        capital=CONFIG["capital"],
    )
    bt = backtest_pair(y, x, eg.alpha, eg.beta, cfg)
    bt_in = {"data": "pair_equity.csv", "config": "config.json",
             "insample_days": n_is}
    case("backtest_total_return", bt_in, {"value": bt.metrics["total_return"]}, 1e-8)
    case("backtest_sharpe", bt_in, {"value": bt.metrics["sharpe"]}, 1e-8)
    case("backtest_max_drawdown", bt_in, {"value": bt.metrics["max_drawdown"]}, 1e-8)
    case("backtest_n_trades", bt_in, {"value": bt.metrics["n_trades"]}, 1e-9)

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
    for c in golden["cases"]:
        print(f"  {c['name']:28s} -> {c['expect']['value']:.10g}  (tol {c['tol']:g})")


if __name__ == "__main__":
    main()
