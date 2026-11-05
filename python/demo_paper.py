"""Paper-trading demo on the bundled equity pair.  Nothing here is a real order.

The hedge is fitted on the first 60% of days and the pair is paper-traded on
the rest, the same split as ``walk_forward``.  The CSV has no volume, so ADV is
an assumption, and the fill-model parameters are assumed too: none of them are
calibrated to any market.  The prices are synthetic.

Run:  cd python && PYTHONPATH=src python demo_paper.py
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from statarb import load_config, load_csv, walk_forward
from statarb.backtest import config_from_dict
from statarb.trading import (
    ExecutionConfig,
    PaperBrokerConfig,
    PaperRunConfig,
    PaperRunResult,
    RiskLimits,
    run_paper,
)

DATA = Path(__file__).resolve().parents[1] / "data"
ADV_SHARES = 100_000.0  # assumed; the data has no volume


def sharpe(equity: np.ndarray) -> float:
    changes = np.diff(equity)
    sd = float(changes.std(ddof=1))
    return float(changes.mean()) / sd * math.sqrt(252.0) if sd > 0.0 else math.nan


def max_drawdown(equity: np.ndarray) -> float:
    return float(np.max(1.0 - equity / np.maximum.accumulate(equity)))


def traded_notional(result: PaperRunResult) -> float:
    return sum(r["qty"] * r["mid"] for r in result.audit.records if r["event"] == "fill")


def episodes(result: PaperRunResult, capital: float) -> dict[int, tuple[float, float]]:
    """Entry signal bar -> (equity change to the next entry, peak gross in $k)."""
    records = result.audit.records
    starts = [r["bar"] for r in records if r["event"] == "signal" and r["action"] == "enter"]
    out: dict[int, tuple[float, float]] = {}
    held: dict[str, int] = {}
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else result.equity.size - 1
        peak = 0.0
        marks: dict[str, float] = {}
        for r in records:
            if r["event"] == "fill" and start < r["bar"] <= end:
                held[r["symbol"]] = held.get(r["symbol"], 0) + (r["qty"] if r["side"] == "BUY" else -r["qty"])
                marks[r["symbol"]] = r["mid"]
                peak = max(peak, sum(abs(q) * marks.get(s, 0.0) for s, q in held.items()))
        out[start] = (float(result.equity[end] - result.equity[start]), peak / 1e3)
    return out


def report(title: str, result: PaperRunResult, capital: float) -> None:
    a = result.attribution
    print(f"\n{title}")
    print(f"  net P&L            {result.equity[-1] - capital:>12,.0f}   ({(result.equity[-1] / capital - 1):+.2%} of capital)")
    print(f"  spread P&L at mid  {a.spread:>12,.0f}")
    print(f"  half-spread        {-a.half_spread:>12,.0f}")
    print(f"  impact             {-a.impact:>12,.0f}")
    print(f"  commission         {-a.commission:>12,.0f}")
    print(f"  borrow             {-a.borrow:>12,.0f}")
    print(f"  Sharpe (daily x sqrt 252)  {sharpe(result.equity):.2f}    max drawdown {max_drawdown(result.equity):.2%}")
    print(f"  entries {result.entries}, outcomes {result.outcomes}, fills {result.fills}")
    print(f"  risk rejections {result.rejections or 'none'}")
    print(f"  kill switch {result.kill_reason or 'not tripped'}; reconciliation breaks {len(result.breaks)}")
    print(f"  largest leg gap on entry {result.max_entry_gap:.3f}; left open at end {result.final_positions or 'nothing'}")


def main() -> None:
    data = load_csv(DATA / "pair_equity.csv")
    y, x = data["stock_y"], data["stock_x"]
    rule = config_from_dict(load_config(DATA / "config.json"))
    wf = walk_forward(y, x, rule)
    y_oos, x_oos = y[wf.split:], x[wf.split:]
    capital = rule.capital
    print("PAPER TRADING ONLY: simulated fills on synthetic prices; ADV and costs are assumed.")
    print(f"hedge fit on days 0-{wf.split - 1}: alpha {wf.alpha:.3f}, beta {wf.beta:.3f}; traded on {y_oos.size} later days")
    print(f"\nBacktest out of sample (fractional shares, flat bps cost, instant fills): "
          f"net P&L {wf.oos.equity[-1] - capital:,.0f}, Sharpe {sharpe(wf.oos.equity):.2f}, "
          f"{len(wf.oos.trades)} trades")

    run = PaperRunConfig(adv_y=ADV_SHARES, adv_x=ADV_SHARES, seed=42)
    broker = PaperBrokerConfig(
        initial_cash=capital, half_spread_bps=1.0, spread_jitter=0.3, impact_bps=30.0, max_participation=0.02,
        liquidity_jitter=0.3, commission_bps=0.5, borrow_bps_annual=50.0, leg_latency_bars={"stock_x": 1},
    )
    limits = RiskLimits(
        max_gross=1.3 * capital, max_net=0.6 * capital, max_name_notional=0.9 * capital, max_order_adv_frac=0.05,
        price_band_frac=0.15, max_drawdown_frac=0.10, locates=frozenset({"stock_y", "stock_x"}),
    )
    execution = ExecutionConfig(max_imbalance_frac=0.5, timeout_bars=8)

    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "audit.jsonl"
        result = run_paper(y_oos, x_oos, wf.alpha, wf.beta, rule, run, broker, limits, execution, str(path))
        lines = path.read_text(encoding="utf-8").splitlines()
        report("Paper run (seed 42)", result, capital)
        print(f"  audit trail {len(lines)} records; last: {json.loads(lines[-1])['event']}")

    finals = [
        run_paper(y_oos, x_oos, wf.alpha, wf.beta, rule, replace(run, seed=s), broker, limits, execution).equity[-1] - capital
        for s in range(20)
    ]
    print(f"\nAcross 20 fill seeds, net P&L {min(finals):,.0f} to {max(finals):,.0f}. "
          "This range is fill noise only, not a confidence interval on the strategy.")

    # Attribution: add one feature at a time, from frictionless to the full run.
    def go(b: PaperBrokerConfig, lim: RiskLimits, r: PaperRunConfig, e: ExecutionConfig) -> PaperRunResult:
        return run_paper(y_oos, x_oos, wf.alpha, wf.beta, rule, r, b, lim, e)

    wide = RiskLimits(max_gross=1e12, max_net=1e12, max_name_notional=1e12, max_order_adv_frac=1.0, price_band_frac=1.0,
                      max_drawdown_frac=0.99, locates=limits.locates)
    deep = replace(run, adv_y=1e9, adv_x=1e9, wind_down_bars=1)
    free = PaperBrokerConfig(initial_cash=capital, half_spread_bps=0.0, impact_bps=0.0, max_participation=1.0)
    uncapped = replace(broker, max_participation=1.0)
    one_shot = ExecutionConfig(1.0, 8)
    ladder = [
        ("frictionless", go(free, wide, deep, one_shot)),
        ("+ fill costs", go(replace(uncapped, leg_latency_bars={}), wide, deep, one_shot)),
        ("+ stock_x latency", go(uncapped, wide, deep, one_shot)),
        ("+ leg pacing 0.5", go(uncapped, wide, deep, execution)),
        ("+ participation cap", go(broker, wide, replace(run, wind_down_bars=1), execution)),
        ("+ risk limits", go(broker, limits, replace(run, wind_down_bars=1), execution)),
        ("+ wind-down (full run)", result),
    ]
    base = traded_notional(ladder[0][1])
    print(f"\n{'step':24}{'net':>9}{'at mid':>9}{'costs':>8}{'entries':>8}{'rejected':>9}{'unwound':>8}{'traded':>8}")
    for name, r in ladder:
        a = r.attribution
        print(f"{name:24}{r.equity[-1] - capital:>9,.0f}{a.spread:>9,.0f}{a.costs + a.borrow:>8,.0f}{r.entries:>8}"
              f"{sum(r.rejections.values()):>9}{r.outcomes['UNWOUND']:>8}{traded_notional(r) / base:>8.0%}")

    print("\nPer trade, by entry signal bar (P&L to the next entry / peak gross in $k):")
    columns = [episodes(ladder[0][1], capital), episodes(result, capital)]
    for bar in sorted(columns[0]):
        cells = [f"{c[bar][0]:>9,.0f} /{c[bar][1]:>5.0f}" if bar in c else f"{'skipped':>16}" for c in columns]
        print(f"  bar {bar:>3} (day {bar + wf.split}):  frictionless {cells[0]}   full run {cells[1]}")

    print("\nPure delay, no costs, both legs late, nothing else changed:")
    for lag in range(5):
        delayed = replace(free, leg_latency_bars={"stock_y": lag, "stock_x": lag})
        r = go(delayed, wide, deep, one_shot)
        print(f"  +{lag} bars: net P&L {r.equity[-1] - capital:>9,.0f}")

    print("""
Why the full run is near zero when the frictionless run loses about $45k:
  Costs are not the reason: they make it worse, as the costs column shows. Risk checks and unwinds
  are not the reason either: nothing was rejected or unwound, and the kill switch did not trip.
  The whole gain is in P&L at mid, and it comes from filling later than the signal. Staged, delayed
  fills build the position two to three bars after the signal, the spread has usually moved further
  by then, so entries are better priced. The largest single change is the bar-475 trade: the better
  entry keeps it above the 2% P&L stop, so it no longer stops out near -$22k. A stop firing during
  the bar-150 entry also leaves that position part-built. The day-1073 break trade is NOT avoided.
  The last trade is skipped only by the wind-down. The pure-delay table reproduces the effect with
  no costs at all: on this sample the rule enters too early, and later fills do better. That is
  one path and about 9 trades, seen after the fact. It is not evidence of an execution edge, and
  delay has not been tested as a rule on unseen data.""")

    thin = run_paper(y_oos, x_oos, wf.alpha, wf.beta, rule, replace(run, adv_y=10_000.0, adv_x=10_000.0),
                     broker, limits, execution)
    report("Same run with ADV cut to 10,000 shares", thin, capital)


if __name__ == "__main__":
    main()
