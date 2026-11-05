"""Backtest: accounting identity, dollar neutrality, stops, no-trade path,
walk-forward, config validation."""
from __future__ import annotations

import math

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
    """At each entry execution gross notional equals capital (both sizing
    modes), so the entry-day cost equals capital * bps."""
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
    """Entry threshold that is reachable but never hit: flat equity, all
    metrics zero/finite."""
    y, x = pair["stock_y"][:400], pair["stock_x"][:400]
    cfg = BacktestConfig(window=60, entry_z=7.5, exit_z=0.5, stop_z=7.6,
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


# ---------------------------------------------------------------------------
# Synthetic-spread fixtures.  Every expected value below is derived
# independently of the backtest: z-scores are recomputed by `_zpath`
# (plain rolling mean / ddof-1 std) and closed forms are used where a
# window holds a single spike on a zero background:
#     z_spike = (W - 1) / sqrt(W),   z_next = -1 / sqrt(W).
# ---------------------------------------------------------------------------


def _zpath(s: np.ndarray, window: int) -> np.ndarray:
    z = np.full(len(s), np.nan)
    for t in range(window - 1, len(s)):
        w = s[t - window + 1 : t + 1]
        sd = w.std(ddof=1)
        if sd > 1e-12:
            z[t] = (s[t] - w.mean()) / sd
    return z


def test_end_of_data_pending_entry_discarded() -> None:
    """Proposed test 1 (CRIT-1): |z_{T-2}| in [entry, stop) while flat
    would execute at T-1 and be force-closed the same close.  The spec
    discards it: no trades, no turnover, equity flat at capital."""
    T = 80
    t = np.arange(T)
    x = 200.0 + 0.5 * np.sin(t)
    s = np.sin(0.7 * t)
    s[T - 2] = 6.0
    y = 10.0 + x + s
    cfg = BacktestConfig(window=20, entry_z=2.0, exit_z=0.5, stop_z=4.24,
                         cost_bps=1.0, slippage_bps=0.5)
    z = _zpath(s, 20)
    assert np.nanmax(np.abs(z[: T - 2])) < 2.0          # no earlier signal
    assert 2.0 <= z[T - 2] < 4.24                        # entry band at T-2
    res = backtest_pair(y, x, 10.0, 1.0, cfg)
    assert res.trades == []
    assert res.metrics["n_trades"] == 0.0
    assert res.metrics["turnover"] == 0.0
    assert res.equity[-1] == cfg.capital
    assert np.all(res.equity == cfg.capital)
    assert np.all(res.daily_cost == 0.0)


def test_zero_day_roundtrip_never_reported(pair: dict, config: dict) -> None:
    """Proposed test 2: over many truncations of the bundled pair no trade
    may ever have exit_day == entry_day (would have caught CRIT-1)."""
    y, x = pair["stock_y"], pair["stock_x"]
    eg = engle_granger(y[:900], x[:900], maxlag=config["adf_maxlag"])
    cfg = _cfg(config)
    lengths = list(range(70, 1501, 3)) + [1493, 1497, 1499, 1500]
    for T in lengths:
        res = backtest_pair(y[:T], x[:T], eg.alpha, eg.beta, cfg)
        for tr in res.trades:
            assert tr.exit_day > tr.entry_day, f"T={T}: {tr}"
            assert 0 <= tr.entry_day < T and tr.exit_day <= T - 1


def test_stop_band_while_flat_no_entry() -> None:
    """Proposed test 3: a single spike a on a zero background gives
    z = (W-1)/sqrt(W) = 6.166 >= stop_z while flat (no entry, pinned) and
    z = -1/sqrt(W) the next day (below entry): no trades, flat equity."""
    T, W, t0 = 120, 40, 60
    s = np.zeros(T)
    s[t0] = 7.0
    x = np.full(T, 50.0)
    y = 100.0 + s                       # alpha = 50, beta = 1 -> spread = s
    cfg = BacktestConfig(window=W, entry_z=2.0, exit_z=0.5, stop_z=4.0)
    z = _zpath(s, W)
    assert z[t0] == pytest.approx((W - 1) / math.sqrt(W), abs=1e-12)
    assert z[t0 + 1] == pytest.approx(-1.0 / math.sqrt(W), abs=1e-12)
    assert z[t0] >= cfg.stop_z
    assert np.all(np.isnan(z[: t0]))    # zero-variance windows: no signal
    res = backtest_pair(y, x, 50.0, 1.0, cfg)
    assert res.trades == []
    assert np.all(res.equity == cfg.capital)


def test_stop_exit_is_flagged_and_executes_next_day() -> None:
    """Proposed test 12: entry band at t0, in-position plateau, |z| >= stop
    at t1 = t0+3 -> exactly one trade, stopped, exit_day == t1 + 1, and the
    exit-day cost equals bps * exit notional recomputed from the legs."""
    T, W, t0 = 160, 40, 70
    t = np.arange(T)
    s = 0.2 * np.sin(0.7 * t)
    s[t0 : t0 + 3] = 0.5
    s[t0 + 3] = 5.0
    x = 200.0 + 0.5 * np.sin(t)
    y = 10.0 + x + s
    cfg = BacktestConfig(window=W, entry_z=2.0, exit_z=0.5, stop_z=4.0,
                         cost_bps=1.0, slippage_bps=0.5, sizing="equal_dollar")
    z = _zpath(s, W)
    assert np.nanmax(np.abs(z[:t0])) < cfg.entry_z
    assert cfg.entry_z <= z[t0] < cfg.stop_z
    assert all(cfg.exit_z < abs(v) < cfg.stop_z for v in z[t0 + 1 : t0 + 3])
    assert z[t0 + 3] >= cfg.stop_z
    assert np.nanmax(np.abs(z[t0 + 4 :])) < cfg.entry_z
    res = backtest_pair(y, x, 10.0, 1.0, cfg)
    assert len(res.trades) == 1
    tr = res.trades[0]
    assert (tr.entry_day, tr.exit_day, tr.direction, tr.stopped) == (t0 + 1, t0 + 4, -1, True)
    assert res.n_stops == 1
    bps = 1.5e-4
    cap = cfg.capital
    qy = -(cap / 2.0) / y[t0 + 1]
    qx = (cap / 2.0) / x[t0 + 1]
    assert res.daily_cost[t0 + 1] == pytest.approx(cap * bps, abs=1e-9)
    exit_notional = abs(qy) * y[t0 + 4] + abs(qx) * x[t0 + 4]
    assert res.daily_cost[t0 + 4] == pytest.approx(exit_notional * bps, abs=1e-9)
    assert res.daily_cost[t0 + 4] > 0.0
    assert res.metrics["avg_holding_days"] == 3.0


def test_reentry_after_exit_same_direction() -> None:
    """Proposed test 13: enter-short, exit, enter-short again with no
    cool-down: two trades with entry_day_2 == exit_day_1 + 1."""
    T, W, t0 = 160, 40, 70
    t = np.arange(T)
    s = 0.2 * np.sin(0.7 * t)
    s[t0], s[t0 + 1], s[t0 + 2], s[t0 + 3] = 0.5, 0.0, 0.5, 0.0
    x = 200.0 + 0.5 * np.sin(t)
    y = 10.0 + x + s
    cfg = BacktestConfig(window=W, entry_z=2.0, exit_z=0.5, stop_z=4.0)
    z = _zpath(s, W)
    assert cfg.entry_z <= z[t0] < cfg.stop_z
    assert abs(z[t0 + 1]) <= cfg.exit_z
    assert cfg.entry_z <= z[t0 + 2] < cfg.stop_z
    assert abs(z[t0 + 3]) <= cfg.exit_z
    assert np.nanmax(np.abs(np.concatenate([z[:t0], z[t0 + 4 :]]))) < cfg.entry_z
    res = backtest_pair(y, x, 10.0, 1.0, cfg)
    assert [(tr.entry_day, tr.exit_day, tr.direction, tr.stopped) for tr in res.trades] == [
        (t0 + 1, t0 + 2, -1, False),
        (t0 + 3, t0 + 4, -1, False),
    ]
    assert res.trades[1].entry_day == res.trades[0].exit_day + 1


def test_equity_nonpositive_metrics_defined() -> None:
    """Proposed test 14 (MIN-11): a 4x jump on the short leg loses 1.5x
    capital.  Expected: total_return = -1.5 exactly, ann_return pinned to
    -1.0, max_drawdown = 1.5, finite Sharpe, no NaN anywhere."""
    T, W, t0 = 45, 40, 40
    s = 0.2 * np.sin(0.7 * np.arange(T))
    s[t0:] = -0.5                        # long-spread entry band at t0
    x = np.full(T, 50.0)
    x[t0 + 2 :] = 200.0                  # short leg quadruples after entry
    y = 100.0 + s                        # alpha = 50, beta = 1
    cfg = BacktestConfig(window=W, entry_z=2.0, exit_z=0.5, stop_z=4.0,
                         cost_bps=0.0, slippage_bps=0.0, sizing="equal_dollar")
    res = backtest_pair(y, x, 50.0, 1.0, cfg)
    assert len(res.trades) == 1
    tr = res.trades[0]
    assert (tr.entry_day, tr.exit_day, tr.direction, tr.stopped) == (t0 + 1, t0 + 3, 1, True)
    # q_x = -(cap/2)/50 = -1e4 shares; x rises by 150 -> pnl = -1.5e6
    assert tr.pnl == pytest.approx(-1.5e6, abs=1e-6)
    assert res.equity[-1] == pytest.approx(-5.0e5, abs=1e-6)
    m = res.metrics
    assert m["total_return"] == pytest.approx(-1.5, abs=1e-12)
    assert m["ann_return"] == -1.0
    assert m["max_drawdown"] == pytest.approx(1.5, abs=1e-12)
    ret = -1.5
    mean = ret / T
    var = ((T - 1) * mean * mean + (ret - mean) ** 2) / (T - 1)
    assert m["sharpe"] == pytest.approx(mean / math.sqrt(var) * math.sqrt(252.0), abs=1e-10)
    assert m["turnover"] == pytest.approx(3.5, abs=1e-12)  # 1e6 in, 2.5e6 out
    assert all(math.isfinite(v) for v in m.values())


def test_backtest_config_rejects_nonfinite() -> None:
    """Proposed test 5 (MAJ-2): NaN/inf thresholds and costs are errors."""
    for kw in ({"cost_bps": math.nan}, {"slippage_bps": math.inf},
               {"entry_z": math.nan}, {"stop_z": math.nan},
               {"exit_z": -math.inf}, {"stop_z": math.inf},
               {"capital": math.nan}, {"cost_bps": -math.inf}):
        with pytest.raises(ValueError):
            BacktestConfig(**kw)
    with pytest.raises(ValueError, match="window"):
        BacktestConfig(window=2.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="window"):
        BacktestConfig(window=True)  # type: ignore[arg-type]


def test_walk_forward_length_mismatch_is_typed_error(pair: dict, config: dict) -> None:
    """Proposed test 7 (MAJ-5)."""
    with pytest.raises(ValueError, match="equal length"):
        walk_forward(pair["stock_y"][:1500], pair["stock_x"][:100], _cfg(config), 0.6)


def test_walk_forward_split_exact() -> None:
    """Proposed test 8 (MIN-10): floor(T * frac) is the intended integer
    (700 * 0.7 == 489.99999999999994 in binary floating point)."""
    cfg = BacktestConfig(window=5, entry_z=1.5, exit_z=0.5, stop_z=1.7)
    for T, frac, expected in ((700, 0.7, 490), (100, 0.29, 29), (1500, 0.6, 900)):
        t = np.arange(T)
        x = 50.0 + np.cos(0.2 * t)
        y = 100.0 + 2.0 * x + 0.1 * np.sin(0.3 * t)
        assert walk_forward(y, x, cfg, frac).split == expected


def test_compute_metrics_validation() -> None:
    from statarb import compute_metrics
    with pytest.raises(ValueError, match="empty"):
        compute_metrics(np.array([]), 1e6, [])
    with pytest.raises(ValueError, match="finite"):
        compute_metrics(np.array([1e6, math.nan]), 1e6, [])
    with pytest.raises(ValueError, match="capital"):
        compute_metrics(np.array([1e6]), 0.0, [])
    m = compute_metrics(np.array([1e6]), 1e6, [])
    assert m["sharpe"] == 0.0 and m["total_return"] == 0.0 and m["max_drawdown"] == 0.0
