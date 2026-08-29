"""statarb end-to-end demo.

Sections:
  1. Cointegration diagnostics on the bundled equity pair (EG + Johansen).
  2. Kalman vs static OLS beta tracking through the day-1050 regime break.
  3. Full backtest report: in-sample vs walk-forward out-of-sample.
  4. Spurious-regression teaching case (independent random walks).

Run:  cd python && PYTHONPATH=src python3 demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from statarb import (
    BacktestConfig,
    BacktestResult,
    adf_test,
    engle_granger,
    half_life,
    johansen_trace,
    kalman_hedge,
    load_config,
    load_csv,
    ols,
    walk_forward,
)

DATA = Path(__file__).resolve().parents[1] / "data"


def rule(title: str) -> None:
    print(f"\n=== {title} " + "=" * max(0, 66 - len(title)))


def print_metrics(label: str, res: BacktestResult) -> None:
    m = res.metrics
    print(
        f"  {label:<12s} ret {m['total_return']*100:7.2f}%  "
        f"ann {m['ann_return']*100:7.2f}%  sharpe {m['sharpe']:6.2f}  "
        f"maxDD {m['max_drawdown']*100:5.2f}%  trades {int(m['n_trades']):3d}  "
        f"win {m['win_rate']*100:4.1f}%  hold {m['avg_holding_days']:5.1f}d  "
        f"turn {m['turnover']:5.1f}x  stops {res.n_stops}"
    )


def main() -> int:
    cfg = load_config(DATA / "config.json")
    pair = load_csv(DATA / "pair_equity.csv")
    fx = load_csv(DATA / "fx_triangle.csv")
    walks = load_csv(DATA / "random_walks.csv")
    y, x = pair["stock_y"], pair["stock_x"]
    n_is = int(len(y) * cfg["insample_frac"])
    bt_cfg = BacktestConfig(
        window=cfg["window"], entry_z=cfg["entry_z"], exit_z=cfg["exit_z"],
        stop_z=cfg["stop_z"], cost_bps=cfg["cost_bps"],
        slippage_bps=cfg["slippage_bps"], capital=cfg["capital"],
    )

    print("statarb demo — cointegration, Kalman hedging, pairs backtest")
    print(f"pair_equity.csv: {len(y)} days, in-sample = first {n_is} days "
          "(true beta 1.7, break at day 1050)")

    rule("1. Cointegration diagnostics (in-sample, days 0..%d)" % (n_is - 1))
    eg = engle_granger(y[:n_is], x[:n_is], maxlag=cfg["adf_maxlag"])
    print(f"  EG step 1  y = {eg.alpha:.4f} + {eg.beta:.4f} x   "
          f"(R^2 {eg.r2:.4f})")
    print(f"  EG step 2  ADF(resid) t = {eg.adf.tstat:.4f}  "
          f"crit 5% (2-var) = {eg.adf.crit['5%']:.2f}  "
          f"p ~ {eg.adf.pvalue:.4f}")
    print(f"  verdict    cointegrated at 5%: {eg.cointegrated(0.05)}")
    print(f"  half-life  {half_life(eg.resid):.2f} days")
    jo = johansen_trace(np.column_stack([y[:n_is], x[:n_is]]))
    for r in range(2):
        flag = ">" if jo.trace_stats[r] > jo.crit_95[r] else "<"
        print(f"  Johansen   trace(r={r}) = {jo.trace_stats[r]:8.3f} {flag} "
              f"cv95 {jo.crit_95[r]:6.2f}")
    print(f"  Johansen   rank = {jo.rank}, cointegrating vector "
          f"(1, {jo.beta[1, 0]:.4f})  [true beta 1.7]")
    jfx = johansen_trace(np.column_stack(
        [np.log(fx["eurusd"]), np.log(fx["gbpusd"]), np.log(fx["eurgbp"])]))
    print(f"  FX triangle (3 assets, logs): rank = {jfx.rank}, "
          f"leading vector ({jfx.beta[0,0]:.2f}, {jfx.beta[1,0]:.2f}, "
          f"{jfx.beta[2,0]:.2f})  [triangle: (1, -1, -1)]")

    rule("2. Kalman vs static OLS through the regime break")
    km = kalman_hedge(y, x, delta=cfg["kalman_delta"], r=cfg["kalman_r"])
    b_ols = float(ols(y, x, intercept=True).params[1])
    true_beta = np.full(len(y), 1.7)
    true_beta[1050:1081] = np.linspace(1.7, 1.1, 31)
    true_beta[1081:] = 1.1
    print("  day    true beta   kalman beta   static OLS beta")
    for d in (250, 500, 900, 1049, 1100, 1200, 1499):
        print(f"  {d:5d}   {true_beta[d]:8.3f}   {km.beta[d]:10.4f}   "
              f"{b_ols:12.4f}")
    sl = slice(300, None)
    te_k = float(np.sqrt(np.mean((km.beta[sl] - true_beta[sl]) ** 2)))
    te_o = float(np.sqrt(np.mean((b_ols - true_beta[sl]) ** 2)))
    print(f"  beta tracking RMSE (day 300+): kalman {te_k:.4f} vs "
          f"static OLS {te_o:.4f} -> kalman adapts, OLS cannot")

    rule("3. Pairs backtest — in-sample vs walk-forward out-of-sample")
    wf = walk_forward(y, x, bt_cfg, cfg["insample_frac"])
    print(f"  hedge estimated on days 0..{wf.split - 1}: "
          f"alpha {wf.alpha:.4f}, beta {wf.beta:.4f}")
    print_metrics("in-sample", wf.insample)
    print_metrics("OOS (40%)", wf.oos)
    print("  degradation is honest: the OOS window contains the day-1050")
    print("  break; stop-losses cap the damage of the broken relationship.")

    rule("4. Spurious-regression teaching case (independent random walks)")
    a1 = adf_test(walks["walk_a"], maxlag=cfg["adf_maxlag"], nvars=1)
    sp = engle_granger(walks["walk_a"], walks["walk_b"],
                       maxlag=cfg["adf_maxlag"])
    slope_t = float(ols(walks["walk_a"], walks["walk_b"]).tstats[1])
    print(f"  regressing walk_a on walk_b: slope t-stat = {slope_t:.1f} —")
    print("  'significant' by the usual +/-2 rule, yet the walks are independent.")
    print(f"  ADF(walk_a) t = {a1.tstat:.3f} (unit root not rejected) and")
    print(f"  ADF(resid) t = {sp.adf.tstat:.3f} vs 5% crit "
          f"{sp.adf.crit['5%']:.2f} -> cointegrated: {sp.cointegrated(0.05)}")
    print("  verdict: spurious relationship — NO backtest is run on this pair.")

    print("\ndemo complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
