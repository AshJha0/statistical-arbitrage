"""Backtest: accounting identity, dollar neutrality, stops, no-trade path,
walk-forward, config validation."""
from __future__ import annotations

import numpy as np
import pytest

from statarb import (
    BacktestConfig,
    backtest_pair,
    engle_granger,
    walk_forward,
)


def _cfg(config: dict) -> BacktestConfig:
    return BacktestConfig(
        window=config["window"],
        entry_z=config["entry_z"],
        exit_z=config["exit_z"],
        stop_z=config["stop_z"],
        cost_bps=config["cost_bps"],
        slippage_bps=config["slippage_bps"],
        capital=config["capital"],
    )


@pytest.fixture(scope="module")
def bundled_run(pair: dict, config: dict):
    y, x = pair["stock_y"], pair["stock_x"]
    n_is = int(len(y) * config["insample_frac"])
    eg = engle_granger(y[:n_is], x[:n_is], maxlag=config["adf_maxlag"])
    res = backtest_pair(y, x, eg.alpha, eg.beta, _cfg(config))
    return res, eg, y, x


def test_accounting_identity(bundled_run, config: dict) -> None:
    """E_{T-1} - capital == sum(pnl) - sum(costs), and per-day increments."""
    res, _, _, _ = bundled_run
    cap = config["capital"]
    assert res.equity[-1] - cap == pytest.approx(
        float(np.sum(res.daily_pnl) - np.sum(res.daily_cost)), abs=1e-10 * cap
    )
    prev = np.concatenate([[cap], res.equity[:-1]])
    assert np.allclose(
        res.equity - prev, res.daily_pnl - res.daily_cost, atol=1e-8
    )


def test_dollar_neutrality_at_entry(bundled_run, config: dict) -> None:
    """At each entry execution the two legs carry capital/2 notional each:
    the entry-day traded notional equals gross capital and costs equal
    capital * bps."""
    res, _, _, _ = bundled_run
    cap = config["capital"]
    bps = (config["cost_bps"] + config["slippage_bps"]) / 1e4
    assert len(res.trades) > 0
    for tr in res.trades:
        assert res.daily_cost[tr.entry_day] == pytest.approx(cap * bps, abs=1e-6)


def test_stop_loss_triggers_in_break_regime(bundled_run) -> None:
    res, _, _, _ = bundled_run
    stops = [t for t in res.trades if t.stopped]
    assert len(stops) >= 1
    # at least one stop fires inside the break regime (day > 1050)
    assert any(t.exit_day > 1050 for t in stops)


def test_trades_well_formed(bundled_run) -> None:
    res, _, _, _ = bundled_run
    m = res.metrics
    for tr in res.trades:
        assert tr.exit_day > tr.entry_day
        assert tr.direction in (-1, 1)
    assert m["n_trades"] == float(len(res.trades))
    assert 0.0 <= m["win_rate"] <= 1.0
    assert m["avg_holding_days"] > 0.0
    assert m["max_drawdown"] >= 0.0
    # each round trip trades ~2x capital (entry + exit legs)
    assert m["turnover"] >= 2.0 * len(res.trades) * 0.5


def test_no_trades_metrics_guarded(pair: dict, config: dict) -> None:
    """Unreachable entry threshold: flat equity, all metrics zero/finite."""
    y, x = pair["stock_y"][:400], pair["stock_x"][:400]
    cfg = BacktestConfig(window=60, entry_z=50.0, exit_z=0.5, stop_z=60.0,
                         capital=config["capital"])
    res = backtest_pair(y, x, 12.0, 1.7, cfg)
    m = res.metrics
    assert len(res.trades) == 0
    assert np.allclose(res.equity, config["capital"])
    for key in ("total_return", "ann_return", "sharpe", "max_drawdown",
                "win_rate", "avg_holding_days", "turnover"):
        assert m[key] == 0.0
    assert m["n_trades"] == 0.0


def test_next_day_execution_no_lookahead(pair: dict, config: dict) -> None:
    """No entry cost may fall on the same day its signal fired: the first
    |z| >= entry day must precede the first entry execution day."""
    res, eg, y, x = None, None, pair["stock_y"], pair["stock_x"]
    from statarb import rolling_zscore, spread
    eg = engle_granger(y[:900], x[:900])
    res = backtest_pair(y, x, eg.alpha, eg.beta, _cfg(config))
    z = rolling_zscore(spread(y, x, eg.alpha, eg.beta), config["window"])
    first_sig = next(t for t in range(len(z))
                     if np.isfinite(z[t]) and abs(z[t]) >= config["entry_z"])
    first_entry = min(tr.entry_day for tr in res.trades)
    assert first_entry == first_sig + 1


def test_walk_forward_split_and_degradation(pair: dict, config: dict) -> None:
    y, x = pair["stock_y"], pair["stock_x"]
    wf = walk_forward(y, x, _cfg(config), config["insample_frac"])
    assert wf.split == int(len(y) * config["insample_frac"])
    assert len(wf.insample.equity) == wf.split
    assert len(wf.oos.equity) == len(y) - wf.split
    # the OOS window contains the regime break: honest degradation
    assert wf.oos.metrics["sharpe"] < wf.insample.metrics["sharpe"]


def test_config_validation() -> None:
    with pytest.raises(ValueError, match="window"):
        BacktestConfig(window=1)
    with pytest.raises(ValueError, match="exit_z < entry_z < stop_z"):
        BacktestConfig(entry_z=2.0, exit_z=2.5, stop_z=4.0)
    with pytest.raises(ValueError, match="exit_z < entry_z < stop_z"):
        BacktestConfig(entry_z=2.0, exit_z=0.5, stop_z=1.5)
    with pytest.raises(ValueError, match="costs"):
        BacktestConfig(cost_bps=-1.0)
    with pytest.raises(ValueError, match="capital"):
        BacktestConfig(capital=0.0)


def test_backtest_input_errors(config: dict) -> None:
    cfg = _cfg(config)
    y = np.linspace(100.0, 110.0, 200)
    with pytest.raises(ValueError, match="equal length"):
        backtest_pair(y, y[:-1], 0.0, 1.0, cfg)
    with pytest.raises(ValueError, match="positive"):
        backtest_pair(y - 200.0, y, 0.0, 1.0, cfg)
    with pytest.raises(ValueError, match="larger than series"):
        backtest_pair(y[:30], y[:30] + 1.0, 0.0, 1.0, cfg)  # window 60 > 30
    with pytest.raises(ValueError, match="insample_frac"):
        walk_forward(y, y + 1.0, cfg, 1.5)
