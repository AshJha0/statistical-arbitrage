"""statarb end-to-end demo.

Sections:
  1. Cointegration diagnostics on the bundled equity pair (EG + Johansen).
  2. Kalman vs static OLS beta tracking through the day-1050 regime break.
  3. Backtest report: single holdout and rolling walk-forward, each with and
     without the risk limits, then how far the out-of-sample result can be
     trusted.
  4. Spurious-regression teaching case (independent random walks).

Run:  cd python && PYTHONPATH=src python3 demo.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from statarb import (
    BacktestResult,
    RollingWalkForwardResult,
    adf_test,
    engle_granger,
    half_life,
    johansen_trace,
    kalman_hedge,
    load_config,
    load_csv,
    ols,
    rolling_walk_forward,
    walk_forward,
)
from statarb.backtest import SPLIT_EPS, config_from_dict
from statarb.research import (
    block_bootstrap_sharpe_ci,
    cusum_break_test,
    probabilistic_sharpe,
)

DATA = Path(__file__).resolve().parents[1] / "data"


def rule(title: str) -> None:
    print(f"\n=== {title} " + "=" * max(0, 66 - len(title)))


def print_metrics(label: str, res: BacktestResult | RollingWalkForwardResult) -> None:
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
    n_is = int(math.floor(len(y) * cfg["insample_frac"] + SPLIT_EPS))
    bt_cfg = config_from_dict(cfg)
    no_limits = config_from_dict(cfg, pnl_stop_frac=0.0, max_hold_days=0,
                                 cooldown_days=0)

    print("statarb demo — cointegration, Kalman hedging, pairs backtest")
    print(f"pair_equity.csv: {len(y)} days, in-sample = first {n_is} days "
          "(true beta 1.7, break at day 1050)")

    rule(f"1. Cointegration diagnostics (in-sample, days 0..{n_is - 1})")
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
          f"static OLS {te_o:.4f}")
    print("  (beta tracking only: the library never trades the Kalman hedge, and a")
    print("   traded Kalman lost most of the pre-break edge in the review, one seed)")

    rule("3. Pairs backtest: holdout, rolling walk-forward, risk limits")
    print(f"  risk limits: P&L stop {cfg['pnl_stop_frac']:.0%} of capital, "
          f"max hold {cfg['max_hold_days']}d, cool-down {cfg['cooldown_days']}d")
    print("  (round numbers fixed in advance, not tuned on this data)")
    wf = walk_forward(y, x, bt_cfg, cfg["insample_frac"])
    wf0 = walk_forward(y, x, no_limits, cfg["insample_frac"])
    print(f"  single holdout, hedge from days 0..{wf.split - 1}: "
          f"alpha {wf.alpha:.4f}, beta {wf.beta:.4f}")
    print_metrics("in-sample", wf.insample)
    print_metrics("OOS limits", wf.oos)
    print_metrics("OOS none", wf0.oos)
    train, test = 500, 100
    rw = rolling_walk_forward(y, x, bt_cfg, train, test)
    rw0 = rolling_walk_forward(y, x, no_limits, train, test)
    print(f"  rolling walk-forward, refit every {test}d on the last {train}d "
          f"({len(rw.blocks)} blocks, days {rw.start_day}+):")
    print_metrics("roll limits", rw)
    print_metrics("roll none", rw0)
    returns = np.diff(rw.equity) / bt_cfg.capital
    ci = block_bootstrap_sharpe_ci(returns, seed=1)
    print(f"  rolling Sharpe {ci.sharpe:.2f}, 95% block-bootstrap interval "
          f"[{ci.lower:.2f}, {ci.upper:.2f}]; "
          f"P(true Sharpe > 0) = {probabilistic_sharpe(returns):.2f}")
    cusum = cusum_break_test(eg.resid[:n_is])
    full = cusum_break_test(y - (eg.alpha + eg.beta * x))
    print(f"  CUSUM on the spread: in-sample p = {cusum.pvalue:.3f}, full "
          f"sample p = {full.pvalue:.3f}, largest shift near day "
          f"{full.break_index}")
    print("  limits shrink the OOS loss but it is still a loss. The last three rolling blocks fit")
    print("  beta <= 0 and stayed flat, which flatters that result. The data is synthetic:")
    print("  none of this is evidence of edge.")

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
