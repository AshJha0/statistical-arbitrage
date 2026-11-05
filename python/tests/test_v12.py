"""v1.2 contract: sizing, risk limits, dead configs, pinned z sums, rolling
walk-forward, streaming API, lag selection and two-way Engle-Granger."""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from statarb import (
    BacktestConfig,
    KalmanStep,
    RollingZ,
    adf_test,
    backtest_pair,
    config_from_dict,
    engle_granger,
    engle_granger_both,
    kalman_hedge,
    rolling_walk_forward,
    rolling_zscore,
    select_lag,
    spread,
    walk_forward,
)

CAP = 1_000_000.0


def _scripted(s: np.ndarray, x: float = 50.0) -> tuple[np.ndarray, np.ndarray]:
    """Prices whose spread at alpha = 50, beta = 1 is exactly s."""
    xs = np.full(s.shape[0], x)
    return 50.0 + xs + s, xs


def _noise(T: int, seed: int = 3) -> tuple[np.ndarray, np.ndarray, float, float]:
    rng = np.random.default_rng(seed)
    x = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, T)))
    s = np.zeros(T)
    for t in range(1, T):
        s[t] = 0.9 * s[t - 1] + rng.normal(0.0, 1.0)
    return 20.0 + 1.5 * x + s + 40.0, x, 60.0, 1.5


def _bundled(pair: dict, config: dict):
    y, x = pair["stock_y"], pair["stock_x"]
    eg = engle_granger(y[:900], x[:900], maxlag=config["adf_maxlag"])
    return y, x, eg.alpha, eg.beta


# ---- item 1 and 2: beta and sizing ----------------------------------------

@pytest.mark.parametrize("beta", [0.0, -1.0, math.nan])
def test_backtest_rejects_non_positive_beta(beta: float) -> None:
    y, x, a, _ = _noise(100)
    with pytest.raises(ValueError, match="beta"):
        backtest_pair(y, x, a, beta, BacktestConfig())


def test_sizing_default_and_validation() -> None:
    assert BacktestConfig().sizing == "beta_hedged"
    with pytest.raises(ValueError, match="sizing"):
        BacktestConfig(sizing="kelly")


@pytest.mark.parametrize("sizing", ["beta_hedged", "equal_dollar"])
def test_gross_equals_capital_at_every_entry(pair, config, sizing) -> None:
    y, x, a, b = _bundled(pair, config)
    cfg = config_from_dict(config, sizing=sizing)
    res = backtest_pair(y, x, a, b, cfg)
    bps = (cfg.cost_bps + cfg.slippage_bps) / 1e4
    assert res.trades
    for tr in res.trades:
        assert res.daily_cost[tr.entry_day] == pytest.approx(CAP * bps, rel=1e-12)


def test_beta_hedged_leg_ratio_and_equal_dollar_neutral() -> None:
    """Day-after P&L reveals the quantities: a +1 move in x alone earns q_x."""
    s = np.zeros(120)
    s[:70] = 0.2 * np.sin(0.7 * np.arange(70))
    s[70:] = 2.0
    y, x = _scripted(s, 100.0)
    y, x = y.copy(), x.copy()
    beta = 1.0
    cfg = BacktestConfig(window=40, cost_bps=0.0, slippage_bps=0.0)
    entry = backtest_pair(y, x, 50.0, beta, cfg).trades[0].entry_day
    x[entry + 1:] += 1.0
    y[entry + 1:] += 1.0 * beta  # spread unchanged
    res = backtest_pair(y, x, 50.0, beta, cfg)
    assert res.trades[0].entry_day == entry
    # hedged at beta: moving along the hedge line earns nothing
    assert res.daily_pnl[entry + 1] == pytest.approx(0.0, abs=1e-8)

    y2 = y.copy()
    y2[entry + 1:] -= 1.0  # only x moved; short-spread position is long x
    res2 = backtest_pair(y2, x, 50.0, beta,
                         replace(cfg, sizing="equal_dollar"))
    qx = (CAP / 2.0) / x[entry]
    assert res2.trades[0].direction == -1
    assert res2.daily_pnl[entry + 1] == pytest.approx(qx * 1.0, rel=1e-12)


# ---- item 3: risk limits ---------------------------------------------------

@pytest.mark.parametrize("sizing", ["beta_hedged", "equal_dollar"])
@pytest.mark.parametrize("limits", [(0.0, 0, 0), (0.02, 60, 20), (0.005, 5, 3)])
def test_accounting_identity(pair, config, sizing, limits) -> None:
    y, x, a, b = _bundled(pair, config)
    cfg = config_from_dict(config, sizing=sizing, pnl_stop_frac=limits[0],
                           max_hold_days=limits[1], cooldown_days=limits[2])
    res = backtest_pair(y, x, a, b, cfg)
    lhs = res.equity[-1] - CAP
    assert lhs == pytest.approx(res.daily_pnl.sum() - res.daily_cost.sum(), abs=1e-6)
    assert lhs == pytest.approx(sum(t.pnl for t in res.trades), abs=1e-6)


@pytest.mark.parametrize("field,bad", [
    ("pnl_stop_frac", -0.1), ("pnl_stop_frac", math.nan), ("pnl_stop_frac", math.inf),
    ("max_hold_days", -1), ("max_hold_days", 1.5), ("max_hold_days", True),
    ("cooldown_days", -1), ("cooldown_days", 2.0),
])
def test_limit_validation(field: str, bad) -> None:
    with pytest.raises(ValueError):
        BacktestConfig(**{field: bad})


def _trade_then_loss(loss_day_value: float, T: int = 140) -> np.ndarray:
    """Short-spread entry signal on day 70, spread stays in the no-rule band
    and moves to ``loss_day_value`` on day 73."""
    s = 0.2 * np.sin(0.7 * np.arange(T))
    s[70:73] = 0.6
    s[73:] = loss_day_value
    return s


def test_pnl_stop_fires_and_beats_z_exit() -> None:
    """On the day the P&L stop and the normal z exit both match, the trade is
    flagged stopped: the P&L rule runs first."""
    s = 0.2 * np.sin(0.7 * np.arange(140))
    s[70] = 0.6                     # entry signal
    s[71] = 0.6                     # execute short
    s[72:] = 0.0
    y, x = _scripted(s)
    x = x.copy()
    x[72:] -= 5.0                   # hurts the long-x leg, spread rebuilt below
    y = 50.0 + x + s
    base = BacktestConfig(window=40, cost_bps=0.0, slippage_bps=0.0)
    z = rolling_zscore(spread(y, x, 50.0, 1.0), 40)
    assert abs(z[72]) <= base.exit_z
    plain = backtest_pair(y, x, 50.0, 1.0, base)
    assert plain.trades[0].stopped is False
    # hedged legs do not lose on a move along the hedge line; equal-dollar do
    cfg = replace(base, sizing="equal_dollar")
    unstopped = backtest_pair(y, x, 50.0, 1.0, cfg)
    loss = unstopped.daily_pnl[72]
    assert loss < 0.0
    stopped = backtest_pair(y, x, 50.0, 1.0,
                            replace(cfg, pnl_stop_frac=-loss / CAP / 2.0))
    tr = stopped.trades[0]
    assert tr.stopped is True and tr.exit_day == 73


def test_pnl_stop_uses_costs_so_far() -> None:
    """Entry cost alone (1.5 bps of capital) trips a stop tighter than it."""
    s = _trade_then_loss(0.6)
    y, x = _scripted(s)
    cfg = BacktestConfig(window=40, pnl_stop_frac=1.0e-4)
    res = backtest_pair(y, x, 50.0, 1.0, cfg)
    tr = res.trades[0]
    assert tr.stopped is True
    assert tr.exit_day == tr.entry_day + 1


def test_pnl_stop_boundary_is_inclusive() -> None:
    """P&L exactly -frac*capital triggers (<=)."""
    s = _trade_then_loss(0.6)
    y, x = _scripted(s)
    cfg = BacktestConfig(window=40, pnl_stop_frac=1.5e-4)  # = entry cost
    res = backtest_pair(y, x, 50.0, 1.0, cfg)
    assert res.daily_cost[res.trades[0].entry_day] == pytest.approx(150.0, rel=1e-12)
    # float product of the bps may land on either side by one ulp, so pin the
    # frac just above and below instead of asserting the exact tie.
    above = backtest_pair(y, x, 50.0, 1.0, replace(cfg, pnl_stop_frac=1.5001e-4))
    below = backtest_pair(y, x, 50.0, 1.0, replace(cfg, pnl_stop_frac=1.4999e-4))
    assert above.trades[0].stopped is False
    assert below.trades[0].stopped is True


def test_stop_priority_pnl_then_z_then_hold() -> None:
    """z stop outranks max-hold: same day, flagged stopped."""
    s = 0.2 * np.sin(0.7 * np.arange(160))
    s[70:73] = 0.5
    s[73] = 5.0
    y, x = _scripted(s)
    cfg = BacktestConfig(window=40, max_hold_days=2)
    z = rolling_zscore(s, 40)
    assert z[73] >= cfg.stop_z
    res = backtest_pair(y, x, 50.0, 1.0, cfg)
    tr = res.trades[0]
    assert tr.entry_day == 71 and tr.exit_day == 74 and tr.stopped is True


def test_max_hold_exits_unflagged_on_time() -> None:
    s = _trade_then_loss(0.6)
    y, x = _scripted(s)
    res = backtest_pair(y, x, 50.0, 1.0, BacktestConfig(window=40, max_hold_days=3))
    tr = res.trades[0]
    # signal when t - entry >= 3, execution one day later
    assert tr.exit_day - tr.entry_day == 4
    assert tr.stopped is False
    assert res.n_stops == 0


def test_limits_fire_on_nan_z() -> None:
    """A flat spread makes z NaN after the window clears; hold still exits."""
    W = 5
    s = np.array([0.0, 0.1, -0.1, 0.05, 0.0, 0.0, 0.3, 0.3] + [0.3] * 30)
    y, x = _scripted(s)
    z = rolling_zscore(s, W)
    cfg = BacktestConfig(window=W, entry_z=1.5, exit_z=0.1, stop_z=1.75,
                         max_hold_days=8)
    assert np.isnan(z[12])
    res = backtest_pair(y, x, 50.0, 1.0, cfg)
    assert res.trades[0].exit_day == 16  # entry 7, signal 15 on a NaN z
    assert res.trades[0].stopped is False


def _stop_then_signal(gap: int, cooldown: int) -> tuple[BacktestConfig, list]:
    """Stop executes on day e = 74; another entry signal fires on e + gap."""
    s = 0.2 * np.sin(0.7 * np.arange(240))
    s[70:73] = 0.5
    s[73] = 5.0
    e = 74
    s[e + gap] = 3.0
    y, x = _scripted(s)
    cfg = BacktestConfig(window=40, stop_z=5.0, cooldown_days=cooldown)
    cfg = replace(cfg, stop_z=4.0)
    return cfg, backtest_pair(y, x, 50.0, 1.0, cfg).trades


def test_cooldown_timing() -> None:
    gap = 10
    _, free = _stop_then_signal(gap, 0)
    assert free[0].stopped is True and free[0].exit_day == 74
    assert len(free) == 2 and free[1].entry_day == 74 + gap + 1
    # t = e + gap: blocked while t < e + cooldown, allowed at equality
    _, blocked = _stop_then_signal(gap, gap + 1)
    assert len(blocked) == 1
    _, allowed = _stop_then_signal(gap, gap)
    assert len(allowed) == 2 and allowed[1].entry_day == 74 + gap + 1


def test_cooldown_not_after_unstopped_exit() -> None:
    s = 0.2 * np.sin(0.7 * np.arange(160))
    s[70], s[71], s[72], s[73] = 0.5, 0.0, 0.5, 0.0
    y, x = _scripted(s)
    res = backtest_pair(y, x, 50.0, 1.0, BacktestConfig(window=40, cooldown_days=50))
    assert len(res.trades) == 2


def test_cooldown_property_on_bundled(pair, config) -> None:
    y, x, a, b = _bundled(pair, config)
    cfg = config_from_dict(config)
    res = backtest_pair(y, x, a, b, cfg)
    assert res.n_stops >= 1
    for prev, nxt in zip(res.trades, res.trades[1:], strict=False):
        if prev.stopped:  # signal day = entry - 1
            assert nxt.entry_day - 1 >= prev.exit_day + cfg.cooldown_days


def test_max_hold_property_on_bundled(pair, config) -> None:
    y, x, a, b = _bundled(pair, config)
    cfg = config_from_dict(config)
    for tr in backtest_pair(y, x, a, b, cfg).trades:
        assert tr.holding_days <= cfg.max_hold_days + 1


# ---- item 4: dead configs --------------------------------------------------

def test_dead_entry_rejected() -> None:
    with pytest.raises(ValueError, match="entry_z unreachable"):
        BacktestConfig(window=5, entry_z=2.0, stop_z=4.0)
    with pytest.raises(ValueError, match="entry_z unreachable"):
        BacktestConfig(window=4, entry_z=1.5, exit_z=0.5, stop_z=4.0,
                       pnl_stop_frac=0.1)  # (4-1)/2 == 1.5 exactly: still dead


def test_dead_stop_needs_pnl_stop() -> None:
    with pytest.raises(ValueError, match="stop_z unreachable"):
        BacktestConfig(window=20, entry_z=2.0, stop_z=8.0)
    with pytest.raises(ValueError, match="stop_z unreachable"):
        BacktestConfig(window=20, entry_z=2.0, stop_z=8.0, max_hold_days=10)
    BacktestConfig(window=20, entry_z=2.0, stop_z=8.0, pnl_stop_frac=0.01)


# ---- item 5: pinned summation ---------------------------------------------

def test_rolling_z_is_plain_sequential_sum() -> None:
    rng = np.random.default_rng(1)
    s = rng.normal(0.0, 1.0, 300) * 10.0 ** rng.integers(-6, 6, 300)
    W = 37
    z = rolling_zscore(s, W)
    v = s.tolist()
    for t in range(W - 1, 300):
        total = 0.0
        for e in v[t - W + 1 : t + 1]:
            total = total + e
        m = total / W
        ss = 0.0
        for e in v[t - W + 1 : t + 1]:
            ss = ss + (e - m) * (e - m)
        assert z[t] == (v[t] - m) / math.sqrt(ss / (W - 1))  # bitwise


# ---- item 6: rolling walk-forward -----------------------------------------

def _cfg_small() -> BacktestConfig:
    return BacktestConfig(window=30, entry_z=1.5, exit_z=0.3, stop_z=4.0,
                          pnl_stop_frac=0.02, max_hold_days=30, cooldown_days=5)


def test_rolling_geometry_and_flat_at_block_end() -> None:
    y, x, _, _ = _noise(1000)
    rw = rolling_walk_forward(y, x, _cfg_small(), 300, 150)
    assert [(b.start, b.end) for b in rw.blocks] == [
        (300, 450), (450, 600), (600, 750), (750, 900), (900, 1000)]
    assert rw.start_day == 300
    assert rw.equity.shape[0] == 700
    assert rw.trades
    for b in rw.blocks:
        assert b.result.equity.shape[0] == b.end - b.start
        for t in b.result.trades:
            assert b.start < t.entry_day <= t.exit_day <= b.end - 1
            assert t.exit_day > t.entry_day


def test_rolling_stitch_and_fit_window() -> None:
    y, x, _, _ = _noise(800)
    rw = rolling_walk_forward(y, x, _cfg_small(), 300, 100)
    pnl = sum(b.result.equity[-1] - CAP for b in rw.blocks)
    assert rw.equity[-1] - CAP == pytest.approx(pnl, abs=1e-6)
    assert rw.metrics["total_return"] == pytest.approx(pnl / CAP, abs=1e-12)
    assert rw.metrics["n_trades"] == sum(b.result.metrics["n_trades"] for b in rw.blocks)
    assert rw.metrics["turnover"] == pytest.approx(
        sum(b.result.metrics["turnover"] for b in rw.blocks))
    k = 2
    fit = engle_granger(y[k * 100 : k * 100 + 300], x[k * 100 : k * 100 + 300])
    assert rw.blocks[k].alpha == fit.alpha and rw.blocks[k].beta == fit.beta


def test_rolling_uses_no_future_data() -> None:
    """Perturbing prices from day c on leaves everything up to day c-1
    unchanged, except the forced close, which never moves before block end."""
    y, x, _, _ = _noise(900)
    cfg = _cfg_small()
    base = rolling_walk_forward(y, x, cfg, 300, 100)
    for c in (450, 600, 777):
        y2, x2 = y.copy(), x.copy()
        y2[c:] *= 1.0 + np.linspace(0.05, 0.5, 900 - c)
        x2[c:] *= 1.0 - np.linspace(0.02, 0.3, 900 - c)
        alt = rolling_walk_forward(y2, x2, cfg, 300, 100)
        n = c - base.start_day
        assert np.array_equal(base.equity[:n], alt.equity[:n])
        before = [b for b in base.blocks if b.start <= c]
        for b0, b1 in zip(before, alt.blocks, strict=False):
            assert (b0.alpha, b0.beta) == (b1.alpha, b1.beta)
        assert [t for t in base.trades if t.exit_day < c] == \
               [t for t in alt.trades if t.exit_day < c]


def test_rolling_first_block_entry_needs_block_signal() -> None:
    """A signal on the day before the block must not open a position."""
    y, x, _, _ = _noise(500)
    rw = rolling_walk_forward(y, x, _cfg_small(), 300, 200)
    assert all(t.entry_day >= 301 for t in rw.trades)


def test_rolling_block_matches_manual_backtest() -> None:
    y, x, _, _ = _noise(600, seed=8)
    cfg = replace(_cfg_small(), cooldown_days=0)
    rw = rolling_walk_forward(y, x, cfg, 300, 300)
    b = rw.blocks[0]
    z = rolling_zscore(spread(y[271:], x[271:], b.alpha, b.beta), cfg.window)
    assert np.array_equal(z[29:], b.result.zscore, equal_nan=True)


def test_rolling_negative_fit_block_stays_flat() -> None:
    T = 400
    t = np.arange(T)
    x = 100.0 + 0.1 * t + np.sin(t)
    y = 200.0 - 0.1 * t + np.cos(0.3 * t)   # falling against a rising x
    rw = rolling_walk_forward(y, x, _cfg_small(), 200, 100)
    assert all(b.skipped and b.beta < 0.0 for b in rw.blocks)
    assert rw.trades == []
    assert np.all(rw.equity == CAP)
    assert rw.metrics["turnover"] == 0.0


@pytest.mark.parametrize("train,test", [(10, 50), (300, 0), (300.0, 50), (300, 2.0),
                                        (1000, 50), (True, 50)])
def test_rolling_validation(train, test) -> None:
    y, x, _, _ = _noise(1000)
    with pytest.raises(ValueError):
        rolling_walk_forward(y, x, _cfg_small(), train, test)


def test_rolling_length_mismatch() -> None:
    y, x, _, _ = _noise(400)
    with pytest.raises(ValueError, match="equal length"):
        rolling_walk_forward(y, x[:-1], _cfg_small(), 100, 50)


def test_walk_forward_still_single_holdout(pair, config) -> None:
    cfg = config_from_dict(config)
    wf = walk_forward(pair["stock_y"], pair["stock_x"], cfg, 0.6)
    assert wf.split == 900 and "SINGLE HOLDOUT" in walk_forward.__doc__


# ---- item 7: streaming -----------------------------------------------------

@pytest.mark.parametrize("window", [2, 5, 60, 250])
def test_rolling_z_stream_matches_batch(pair, config, window) -> None:
    y, x, a, b = _bundled(pair, config)
    s = spread(y, x, a, b)
    batch = rolling_zscore(s, window)
    rz = RollingZ(window)
    got = np.array([rz.update(v) for v in s])
    assert np.array_equal(np.isnan(got), np.isnan(batch))
    ok = ~np.isnan(batch)
    assert np.max(np.abs(got[ok] - batch[ok])) < 1e-9


def test_rolling_z_stream_long_run_with_large_level() -> None:
    rng = np.random.default_rng(5)
    s = 1.0e4 + np.cumsum(rng.normal(0.0, 1.0, 50_000))
    batch = rolling_zscore(s, 60)
    rz = RollingZ(60)
    got = np.array([rz.update(v) for v in s])
    # Running moments drift slowly with the update count: 1e-9 holds at the
    # bundled length (test above), a looser bound after 50,000 updates.
    assert np.nanmax(np.abs(got - batch)) < 1e-7


def test_rolling_z_stream_zero_variance_and_validation() -> None:
    rz = RollingZ(3)
    assert all(math.isnan(rz.update(1.0)) for _ in range(5))
    for bad in (1, 1.5, True):
        with pytest.raises(ValueError):
            RollingZ(bad)
    with pytest.raises(ValueError):
        rz.update(math.nan)
    with pytest.raises(ValueError):
        rz.update(math.inf)
    assert math.isnan(rz.update(1.0))  # the rejected values left no trace


def test_kalman_step_matches_batch(pair, config) -> None:
    y, x = pair["stock_y"], pair["stock_x"]
    d, r = config["kalman_delta"], config["kalman_r"]
    km = kalman_hedge(y, x, delta=d, r=r)
    ks = KalmanStep(d, r)
    got = np.array([ks.update(a, b) for a, b in zip(x, y, strict=False)])
    assert np.max(np.abs(got[:, 0] - km.alpha)) < 1e-12
    assert np.max(np.abs(got[:, 1] - km.beta)) < 1e-12
    assert np.max(np.abs(got[:, 2] - km.z)) < 1e-12


def test_kalman_scalar_recursion_matches_matrix_joseph_form(pair, config) -> None:
    """Independent matrix form of the pinned recursion."""
    y, x = pair["stock_y"], pair["stock_x"]
    d, r = config["kalman_delta"], config["kalman_r"]
    s, P, Q, I2 = np.zeros(2), 1e3 * np.eye(2), d / (1 - d) * np.eye(2), np.eye(2)
    km = kalman_hedge(y, x, delta=d, r=r)
    for t in range(y.shape[0]):
        Pp = P + Q
        h = np.array([1.0, x[t]])
        S = h @ Pp @ h + r
        K = Pp @ h / S
        s = s + K * (y[t] - h @ s)
        A = I2 - np.outer(K, h)
        P = A @ Pp @ A.T + r * np.outer(K, K)
        P = 0.5 * (P + P.T)
        assert abs(s[1] - km.beta[t]) < 1e-9 and abs(s[0] - km.alpha[t]) < 1e-9
    assert np.allclose(P, km.final_cov, rtol=1e-8)


@pytest.mark.parametrize("delta,r", [(0.0, 1.0), (1.0, 1.0), (math.nan, 1.0),
                                     (0.1, 0.0), (0.1, math.inf), (0.1, math.nan)])
def test_kalman_step_validation(delta, r) -> None:
    with pytest.raises(ValueError):
        KalmanStep(delta, r)


def test_kalman_step_rejects_nonfinite() -> None:
    ks = KalmanStep()
    with pytest.raises(ValueError):
        ks.update(math.nan, 1.0)
    with pytest.raises(ValueError):
        ks.update(1.0, math.inf)


# ---- item 8: lag selection and two-way EG ---------------------------------

def _ar2(T: int = 800, seed: int = 2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = np.zeros(T)
    for t in range(2, T):
        v[t] = 0.3 * v[t - 1] + 0.5 * v[t - 2] + rng.normal()
    return v


def test_select_lag_finds_augmentation() -> None:
    # AR(2) levels = ADF regression with exactly one lagged difference
    assert select_lag(_ar2(), 6, 1) == 1
    rng = np.random.default_rng(4)
    assert select_lag(rng.normal(size=800), 6, 1) == 0


def test_select_lag_matches_direct_bic() -> None:
    from statarb.ols import ols
    v = _ar2(300, 9)
    m = 4
    dy = np.diff(v)
    lhs = dy[m:]
    n = lhs.shape[0]
    bics = []
    for p in range(m + 1):
        cols = [v[m:-1]] + [dy[m - i : -i] for i in range(1, p + 1)]
        e = ols(lhs, np.column_stack(cols), intercept=True).resid
        bics.append(n * math.log(float(e @ e) / n) + (2 + p) * math.log(n))
    assert select_lag(v, m, 1) == int(np.argmin(bics))


def test_select_lag_validation() -> None:
    v = _ar2(100)
    for args in ((-1, 1), (2, 4), (1.5, 1), (90, 1)):
        with pytest.raises(ValueError):
            select_lag(v, *args)
    assert select_lag(v, 0, 1) == 0
    adf_test(v, maxlag=select_lag(v, 3, 1))


def test_engle_granger_both_needs_both(pair, walks) -> None:
    y, x = pair["stock_y"][:900], pair["stock_x"][:900]
    both = engle_granger_both(y, x, 1)
    assert both.y_on_x.beta == engle_granger(y, x, 1).beta
    assert both.x_on_y.beta == engle_granger(x, y, 1).beta
    assert both.cointegrated(0.05) == (
        both.y_on_x.cointegrated(0.05) and both.x_on_y.cointegrated(0.05))
    assert not engle_granger_both(walks["walk_a"], walks["walk_b"], 1).cointegrated(0.05)
