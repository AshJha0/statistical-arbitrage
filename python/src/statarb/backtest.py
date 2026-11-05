"""Pairs backtest — state machine, risk limits and accounting (PINNED).

Timing convention (pinned, no look-ahead): the signal computed from the
day-t CLOSE is executed at the day t+1 CLOSE at day t+1 prices.  Daily
P&L on day t marks the position held since the day t-1 close:

    pnl_t  = q_y * (y_t - y_{t-1}) + q_x * (x_t - x_{t-1})
    cost_t = (cost_bps + slippage_bps) / 1e4 * traded_notional_t
    E_t    = E_{t-1} + pnl_t - cost_t,    E_{-1} := capital

so the accounting identity E_{T-1} - capital == sum(pnl) - sum(costs)
holds to rounding (tested at 1e-10).

State machine (states FLAT / LONG spread / SHORT spread, plus at most one
pending order):

* FLAT, signal day t:  entry_z <= z_t <  stop_z  -> pending SHORT entry
                      -stop_z  <  z_t <= -entry_z -> pending LONG entry
  (|z_t| >= stop_z while flat: no entry, pinned; NaN z: no action.)
* In position, signal day t, first match wins:
  (1) open trade P&L net of costs so far <= -pnl_stop_frac*capital -> STOP;
  (2) |z_t| >= stop_z -> STOP;  (3) t - entry_day >= max_hold_days -> exit
  (not a stop);  (4) |z_t| <= exit_z -> exit.  Rules (1) and (3) do not
  read z, so they still fire on a NaN z.  A limit of 0 is off.
* Cool-down: after a stopped exit executed on day e there is no entry
  signal while t < e + cooldown_days.
* Execution day t+1 close, dir = +1 LONG spread (long y / short x), -1 SHORT.
  sizing="beta_hedged" (default): u = capital / (y + beta*x), q_y = dir*u,
  q_x = -dir*beta*u, so the legs are in the hedge ratio and gross notional
  is exactly capital at entry (not dollar-neutral).
  sizing="equal_dollar": q_y = dir*(capital/2)/y, q_x = -dir*(capital/2)/x,
  dollar-neutral at entry.  Quantities are then held fixed.
* Costs are charged on traded notional |dq_y| * y + |dq_x| * x at the
  execution close, both entry and exit (= capital on entry).
* A position still open at the final day is force-closed at the final
  close (counted as a closed trade, not a stop).  A pending entry at the
  end of data is DISCARDED (never executed-and-force-closed the same day).

Metrics (pinned formulas, N = number of days):
  daily_ret_t   = (E_t - E_{t-1}) / capital          (return on gross capital)
  total_return  = (E_{N-1} - capital) / capital
  ann_return    = (1 + total_return)^(252/N) - 1     (-1.0 when total_return <= -1)
  sharpe        = mean(daily_ret) / std(daily_ret, ddof=1) * sqrt(252)
                  (0.0 when std == 0 or N < 2)
  max_drawdown  = max_t (peak_t - E_t) / peak_t, peak_t = running max E
                  (reported as a positive fraction)
  n_trades      = closed round trips;  win_rate = wins / n_trades (0 if none)
  avg_holding_days = mean(exit_day - entry_day) over closed trades (0 if none)
  turnover      = total traded notional / capital
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .ols import ols
from .signals import rolling_zscore, spread

_FLAT, _LONG, _SHORT = 0, 1, -1

#: Nudge added before flooring ``T * insample_frac`` in walk_forward so the
#: split is the intended integer despite binary-float products.  Pinned.
SPLIT_EPS: float = 1.0e-9

#: Accepted values of ``BacktestConfig.sizing``.
SIZING_MODES: tuple[str, str] = ("beta_hedged", "equal_dollar")


def max_abs_zscore(window: int) -> float:
    """Largest |z| a W-point sample can give its own latest point (ddof=1)."""
    return (window - 1) / math.sqrt(window)


@dataclass(frozen=True)
class BacktestConfig:
    """Signal / cost parameters for the pairs backtest.

    Attributes:
        window: Rolling z-score window W (>= 2).
        entry_z: Enter when |z| >= entry_z (and < stop_z).
        exit_z: Exit when |z| <= exit_z.
        stop_z: Stop-loss when |z| >= stop_z.
        cost_bps: Commission, basis points of traded notional.
        slippage_bps: Slippage, basis points of traded notional.
        capital: Gross capital; gross notional at entry.
        sizing: "beta_hedged" (default) or "equal_dollar".
        pnl_stop_frac: Stop when open trade P&L <= -frac * capital; 0 = off.
        max_hold_days: Exit when held this many days; 0 = off.
        cooldown_days: No entry signal for this many days after a stopped
            exit; 0 = off.
    """

    window: int = 60
    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float = 4.0
    cost_bps: float = 1.0
    slippage_bps: float = 0.5
    capital: float = 1_000_000.0
    sizing: str = "beta_hedged"
    pnl_stop_frac: float = 0.0
    max_hold_days: int = 0
    cooldown_days: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.window, bool) or not isinstance(
            self.window, (int, np.integer)
        ) or self.window < 2:
            raise ValueError("window must be an integer >= 2")
        for name in ("entry_z", "exit_z", "stop_z", "cost_bps", "slippage_bps"):
            v = getattr(self, name)
            if not (isinstance(v, (int, float, np.floating, np.integer))
                    and math.isfinite(v)):
                raise ValueError("thresholds and costs must be finite")
        if not (0.0 <= self.exit_z < self.entry_z < self.stop_z):
            raise ValueError("require 0 <= exit_z < entry_z < stop_z")
        if self.cost_bps < 0.0 or self.slippage_bps < 0.0:
            raise ValueError("costs must be >= 0")
        if not (isinstance(self.capital, (int, float, np.floating, np.integer))
                and self.capital > 0.0 and math.isfinite(self.capital)):
            raise ValueError("capital must be positive and finite")
        if self.sizing not in SIZING_MODES:
            raise ValueError("sizing must be 'beta_hedged' or 'equal_dollar'")
        v = self.pnl_stop_frac
        if not (isinstance(v, (int, float, np.floating, np.integer))
                and not isinstance(v, bool) and math.isfinite(v) and v >= 0.0):
            raise ValueError("pnl_stop_frac must be finite and >= 0")
        for name in ("max_hold_days", "cooldown_days"):
            d = getattr(self, name)
            if isinstance(d, bool) or not isinstance(d, (int, np.integer)) or d < 0:
                raise ValueError(f"{name} must be an integer >= 0")
        # Dead-config rejection: |z| can never reach (W-1)/sqrt(W).
        reach = max_abs_zscore(int(self.window))
        if reach <= self.entry_z:
            raise ValueError(
                f"entry_z unreachable: window {self.window} caps |z| at {reach:.4f}"
            )
        if reach <= self.stop_z and not self.pnl_stop_frac > 0.0:
            raise ValueError(
                f"stop_z unreachable: window {self.window} caps |z| at "
                f"{reach:.4f}; allowed only with pnl_stop_frac > 0"
            )


def config_from_dict(cfg: dict[str, Any], **overrides: Any) -> BacktestConfig:
    """Build a BacktestConfig from a load_config dict (absent limits = off)."""
    keys = ("window", "entry_z", "exit_z", "stop_z", "cost_bps", "slippage_bps",
            "capital", "pnl_stop_frac", "max_hold_days", "cooldown_days")
    kw = {k: cfg[k] for k in keys if k in cfg}
    kw.update(overrides)
    return BacktestConfig(**kw)


@dataclass
class Trade:
    """One closed round trip (days are execution-day indices)."""

    entry_day: int
    exit_day: int
    direction: int  # +1 long spread, -1 short spread
    pnl: float      # net of entry and exit costs
    stopped: bool

    @property
    def holding_days(self) -> int:
        return self.exit_day - self.entry_day


@dataclass
class BacktestResult:
    """Full backtest output: paths, trades, and pinned metrics."""

    equity: np.ndarray
    daily_pnl: np.ndarray
    daily_cost: np.ndarray
    zscore: np.ndarray
    trades: list[Trade] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def n_stops(self) -> int:
        return sum(1 for t in self.trades if t.stopped)


def compute_metrics(
    equity: np.ndarray, capital: float, trades: list[Trade]
) -> dict[str, float]:
    """Compute the pinned metric set from an equity curve and trade list.

    Raises:
        ValueError: empty equity curve, non-finite equity, or non-positive /
            non-finite capital.
    """
    equity = np.asarray(equity, dtype=float)
    n = equity.shape[0]
    if equity.ndim != 1 or n == 0:
        raise ValueError("equity curve must be a non-empty 1-D array")
    if not np.all(np.isfinite(equity)):
        raise ValueError("equity curve must be finite (no NaN/inf)")
    if not (isinstance(capital, (int, float, np.floating, np.integer))
            and capital > 0.0 and math.isfinite(capital)):
        raise ValueError("capital must be positive and finite")
    total_return = (float(equity[-1]) - capital) / capital
    # (1 + total)^(252/N) is undefined for equity <= 0: pin the total loss
    # of capital (and worse) to an annualised return of exactly -1.
    if total_return <= -1.0:
        ann_return = -1.0
    else:
        ann_return = (1.0 + total_return) ** (252.0 / n) - 1.0
    prev = np.concatenate([[capital], equity[:-1]])
    daily_ret = (equity - prev) / capital
    if n >= 2:
        sd = float(daily_ret.std(ddof=1))
        sharpe = float(daily_ret.mean()) / sd * math.sqrt(252.0) if sd > 0.0 else 0.0
    else:
        sharpe = 0.0
    peak = np.maximum.accumulate(equity)
    max_dd = float(np.max((peak - equity) / peak)) if n > 0 else 0.0
    n_trades = len(trades)
    wins = sum(1 for t in trades if t.pnl > 0.0)
    return {
        "total_return": total_return,
        "ann_return": ann_return,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "n_trades": float(n_trades),
        "win_rate": wins / n_trades if n_trades else 0.0,
        "avg_holding_days": (
            sum(t.holding_days for t in trades) / n_trades if n_trades else 0.0
        ),
    }


def backtest_pair(
    y: np.ndarray,
    x: np.ndarray,
    alpha: float,
    beta: float,
    config: BacktestConfig,
) -> BacktestResult:
    """Run the pinned pairs state machine over aligned price series.

    ``beta`` must be > 0: a pair needing a negative hedge is out of scope.

    Args:
        y, x: Close price series, equal length, strictly positive.
        alpha, beta: Hedge parameters defining the spread (estimated
            elsewhere — e.g. in-sample OLS or Engle-Granger).
        config: Validated signal/cost parameters.

    Raises:
        ValueError: length mismatch, non-finite/non-positive prices,
            beta <= 0 or NaN, or window larger than the series.
    """
    return _run_backtest(y, x, alpha, beta, config, 0)


def _run_backtest(
    y: np.ndarray,
    x: np.ndarray,
    alpha: float,
    beta: float,
    config: BacktestConfig,
    first_entry_signal: int,
) -> BacktestResult:
    """State machine; entry signals are ignored before ``first_entry_signal``."""
    if not beta > 0.0:
        raise ValueError("beta must be > 0: negative hedges are out of scope")
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if y.shape != x.shape or y.ndim != 1:
        raise ValueError("y and x must be 1-D arrays of equal length")
    if not (np.all(np.isfinite(y)) and np.all(np.isfinite(x))):
        raise ValueError("prices must be finite (no NaN/inf)")
    if np.any(y <= 0.0) or np.any(x <= 0.0):
        raise ValueError("prices must be strictly positive")

    s = spread(y, x, alpha, beta)
    z = rolling_zscore(s, config.window)
    bps = (config.cost_bps + config.slippage_bps) / 1.0e4
    cap = config.capital

    T = y.shape[0]
    equity = np.empty(T)
    daily_pnl = np.zeros(T)
    daily_cost = np.zeros(T)
    trades: list[Trade] = []

    state = _FLAT
    qy = qx = 0.0
    pending: tuple[str, int, bool] | None = None  # (action, dir, stop_flag)
    open_entry_day = -1
    open_dir = 0
    open_pnl = 0.0
    turnover_notional = 0.0
    eq = cap
    no_entry_before = first_entry_signal  # also moved by the cool-down

    for t in range(T):
        # 1) mark-to-market the position held since the day t-1 close
        pnl = qy * (y[t] - y[t - 1]) + qx * (x[t] - x[t - 1]) if t > 0 else 0.0
        if state != _FLAT:
            open_pnl += pnl
        cost = 0.0

        # 2) execute the order decided at the day t-1 close.  A pending
        #    ENTRY on the final day is discarded (pinned): it could only be
        #    force-closed at the same close, a phantom zero-day round trip.
        if pending is not None:
            action, direction, stop_flag = pending
            if action == "enter" and state == _FLAT and t < T - 1:
                if config.sizing == "beta_hedged":
                    u = cap / (y[t] + beta * x[t])
                    qy = direction * u
                    qx = -direction * beta * u
                else:
                    qy = direction * (cap / 2.0) / y[t]
                    qx = -direction * (cap / 2.0) / x[t]
                traded = abs(qy) * y[t] + abs(qx) * x[t]
                cost += traded * bps
                turnover_notional += traded
                state = _LONG if direction > 0 else _SHORT
                open_entry_day = t
                open_dir = direction
                open_pnl = -traded * bps
            elif action == "exit" and state != _FLAT:
                traded = abs(qy) * y[t] + abs(qx) * x[t]
                cost += traded * bps
                turnover_notional += traded
                open_pnl -= traded * bps
                trades.append(
                    Trade(open_entry_day, t, open_dir, open_pnl, stop_flag)
                )
                if stop_flag:
                    no_entry_before = max(no_entry_before,
                                          t + config.cooldown_days)
                qy = qx = 0.0
                state = _FLAT
            pending = None

        # 2b) force-close a position still open at the final close
        if t == T - 1 and state != _FLAT:
            traded = abs(qy) * y[t] + abs(qx) * x[t]
            cost += traded * bps
            turnover_notional += traded
            open_pnl -= traded * bps
            trades.append(Trade(open_entry_day, t, open_dir, open_pnl, False))
            qy = qx = 0.0
            state = _FLAT

        eq += pnl - cost
        equity[t] = eq
        daily_pnl[t] = pnl
        daily_cost[t] = cost

        # 3) generate the order to execute at the day t+1 close
        if t < T - 1:
            zt = z[t]
            z_ok = not math.isnan(zt)
            if state == _FLAT:
                if z_ok and t >= no_entry_before:
                    if config.entry_z <= zt < config.stop_z:
                        pending = ("enter", -1, False)
                    elif -config.stop_z < zt <= -config.entry_z:
                        pending = ("enter", +1, False)
            elif (config.pnl_stop_frac > 0.0
                  and open_pnl <= -config.pnl_stop_frac * cap):
                pending = ("exit", 0, True)
            elif z_ok and abs(zt) >= config.stop_z:
                pending = ("exit", 0, True)
            elif (config.max_hold_days > 0
                  and t - open_entry_day >= config.max_hold_days):
                pending = ("exit", 0, False)
            elif z_ok and abs(zt) <= config.exit_z:
                pending = ("exit", 0, False)

    metrics = compute_metrics(equity, cap, trades)
    metrics["turnover"] = turnover_notional / cap
    return BacktestResult(
        equity=equity,
        daily_pnl=daily_pnl,
        daily_cost=daily_cost,
        zscore=z,
        trades=trades,
        metrics=metrics,
    )


@dataclass(frozen=True)
class WalkForwardResult:
    """In-sample vs out-of-sample comparison with a single hedge estimate."""

    alpha: float
    beta: float
    insample: BacktestResult
    oos: BacktestResult
    split: int


def walk_forward(
    y: np.ndarray,
    x: np.ndarray,
    config: BacktestConfig,
    insample_frac: float = 0.6,
) -> WalkForwardResult:
    """SINGLE HOLDOUT (one fit, one split; the name is kept for
    compatibility; see rolling_walk_forward for the refitting version).

    Estimate the hedge on the first ``insample_frac`` of days by OLS,
    backtest that segment in-sample, then trade the remainder
    out-of-sample with the SAME (alpha, beta) — degradation on display.

    The out-of-sample z-score warms up inside the OOS segment only
    (no information leaks across the split).

    ``split = floor(T * insample_frac + 1e-9)`` (pinned): the tiny nudge
    removes binary-float artefacts such as ``700 * 0.7 == 489.99999999999994``
    so the split is the mathematically intended integer.

    Raises:
        ValueError: insample_frac outside (0, 1), length mismatch, or either
            segment shorter than the z-score window.
    """
    if not (0.0 < insample_frac < 1.0):
        raise ValueError("insample_frac must lie strictly inside (0, 1)")
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if y.shape != x.shape or y.ndim != 1:
        raise ValueError("y and x must be 1-D arrays of equal length")
    split = int(math.floor(y.shape[0] * insample_frac + SPLIT_EPS))
    if split < config.window or y.shape[0] - split < config.window:
        raise ValueError("each walk-forward segment must cover the z window")
    fit = ols(y[:split], x[:split], intercept=True)
    a, b = float(fit.params[0]), float(fit.params[1])
    ins = backtest_pair(y[:split], x[:split], a, b, config)
    oos = backtest_pair(y[split:], x[split:], a, b, config)
    return WalkForwardResult(alpha=a, beta=b, insample=ins, oos=oos, split=split)


@dataclass(frozen=True)
class RollingBlock:
    """One test block: days [start, end) traded with a fit from before start.

    ``result`` covers the block days only; its trades carry absolute days.
    ``skipped`` marks a block left flat because the fit gave beta <= 0.
    """

    start: int
    end: int
    alpha: float
    beta: float
    skipped: bool
    result: BacktestResult


@dataclass(frozen=True)
class RollingWalkForwardResult:
    """Stitched rolling walk-forward; ``equity[i]`` is day ``start_day + i``."""

    blocks: list[RollingBlock]
    start_day: int
    equity: np.ndarray
    trades: list[Trade]
    metrics: dict[str, float]

    @property
    def n_stops(self) -> int:
        return sum(1 for t in self.trades if t.stopped)


def _flat_block(n: int, cap: float) -> BacktestResult:
    equity = np.full(n, cap)
    metrics = compute_metrics(equity, cap, [])
    metrics["turnover"] = 0.0
    return BacktestResult(equity=equity, daily_pnl=np.zeros(n),
                          daily_cost=np.zeros(n), zscore=np.full(n, np.nan),
                          trades=[], metrics=metrics)


def rolling_walk_forward(
    y: np.ndarray,
    x: np.ndarray,
    config: BacktestConfig,
    train: int,
    test: int,
) -> RollingWalkForwardResult:
    """Refit on a rolling window and trade only the days after each fit.

    For k = 0, 1, ...: OLS on [k*test, k*test + train), trade days
    [start, end) with start = k*test + train, end = min(start + test, T).
    Each block is its own backtest on the W-1 days before it plus the block,
    so z on the first block day is valid and uses history only.  Entry
    signals start on the first block day, a position is flat at block end
    (forced close, not a stop) and cool-down does not carry between blocks.

    Raises:
        ValueError: length mismatch, non-integer train/test, test < 1,
            train < window, or train >= T.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if y.shape != x.shape or y.ndim != 1:
        raise ValueError("y and x must be 1-D arrays of equal length")
    for v in (train, test):
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
            raise ValueError("train and test must be integers")
    if test < 1:
        raise ValueError("test must be >= 1")
    if train < config.window:
        raise ValueError("train must cover the z window")
    n = y.shape[0]
    if train >= n:
        raise ValueError("train must be shorter than the series")

    cap = config.capital
    warm = config.window - 1
    blocks: list[RollingBlock] = []
    trades: list[Trade] = []
    pieces: list[np.ndarray] = []
    running = cap
    turnover = 0.0
    k = 0
    while k * test + train < n:
        fit_lo = k * test
        start = fit_lo + train
        end = min(start + test, n)
        fit = ols(y[fit_lo:start], x[fit_lo:start], intercept=True)
        a, b = float(fit.params[0]), float(fit.params[1])
        skipped = not b > 0.0
        if skipped:
            res = _flat_block(end - start, cap)
        else:
            lo = start - warm
            full = _run_backtest(y[lo:end], x[lo:end], a, b, config, warm)
            block_trades = [
                Trade(t.entry_day + lo, t.exit_day + lo, t.direction, t.pnl,
                      t.stopped)
                for t in full.trades
            ]
            equity = full.equity[warm:]
            metrics = compute_metrics(equity, cap, block_trades)
            metrics["turnover"] = full.metrics["turnover"]
            res = BacktestResult(
                equity=equity, daily_pnl=full.daily_pnl[warm:],
                daily_cost=full.daily_cost[warm:], zscore=full.zscore[warm:],
                trades=block_trades, metrics=metrics,
            )
        blocks.append(RollingBlock(start, end, a, b, skipped, res))
        trades.extend(res.trades)
        pieces.append(running + (res.equity - cap))
        running = float(pieces[-1][-1])
        turnover += res.metrics["turnover"]
        k += 1

    stitched = np.concatenate(pieces)
    metrics = compute_metrics(stitched, cap, trades)
    metrics["turnover"] = turnover
    return RollingWalkForwardResult(blocks=blocks, start_day=train,
                                    equity=stitched, trades=trades,
                                    metrics=metrics)
