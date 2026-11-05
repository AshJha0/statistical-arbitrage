"""Day loop: the streaming pairs rule through execution, risk and the ledger.

The rule matches ``statarb.backtest``: enter at |z| >= entry_z and below
stop_z, leave at |z| <= exit_z or on a stop, with the same P&L stop, maximum
hold and cool-down.  Its ``cost_bps`` and ``slippage_bps`` are ignored because
the broker's fill model prices the trades.  Differences from the backtest:

* shares are sized at the decision close and rounded down, and fills can take
  several bars;
* while an entry is incomplete only a stop acts on it: the remainder is
  cancelled and what filled is unwound; other signals wait;
* the cool-down runs from the bar after the stop decision, as in the backtest,
  however long the exit takes to fill;
* the book is wound down near the end of the data instead of force-closed at
  the last close, and may still hold shares if the wind-down did not finish.

Per bar: fills and borrow, booking, reconciliation, drawdown check, then the
signal.  A reconciliation break trips the kill switch.  No real order exists
anywhere in this loop.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from ..backtest import BacktestConfig
from ..signals import RollingZ
from .broker import PaperBroker, PaperBrokerConfig
from .execution import ExecutionConfig, Outcome, PairExecutor
from .ledger import Attribution, AuditLog, Break, Ledger
from .orders import Quote, round_shares
from .risk import RiskEngine, RiskLimits


class _Phase(Enum):
    FLAT = "FLAT"
    ENTERING = "ENTERING"
    HOLDING = "HOLDING"
    EXITING = "EXITING"


@dataclass(frozen=True)
class PaperRunConfig:
    """``adv_y`` and ``adv_x`` are assumed volumes; the CSV has none."""

    adv_y: float
    adv_x: float
    symbol_y: str = "stock_y"
    symbol_x: str = "stock_x"
    wind_down_bars: int = 15
    seed: int = 0


@dataclass
class PaperRunResult:
    equity: np.ndarray
    attribution: Attribution
    fills: int
    entries: int
    outcomes: dict[str, int]
    rejections: dict[str, int]
    breaks: list[Break]
    kill_reason: str
    final_positions: dict[str, int]
    max_entry_gap: float
    max_entry_excess: float
    audit: AuditLog = field(repr=False)
    ledger: Ledger = field(repr=False)


def run_paper(
    y: np.ndarray,
    x: np.ndarray,
    alpha: float,
    beta: float,
    rule: BacktestConfig,
    run: PaperRunConfig,
    broker_config: PaperBrokerConfig,
    limits: RiskLimits,
    execution: ExecutionConfig | None = None,
    audit_path: str | None = None,
) -> PaperRunResult:
    """Paper-trade the pair over ``y`` and ``x``."""
    if not beta > 0.0:
        raise ValueError("beta must be > 0: negative hedges are out of scope")
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if y.shape != x.shape or y.ndim != 1:
        raise ValueError("y and x must be 1-D arrays of equal length")

    audit = AuditLog(audit_path)
    risk = RiskEngine(limits, sink=audit.append)
    broker = PaperBroker(broker_config, seed=run.seed)
    ledger = Ledger(broker_config.initial_cash, broker_config.borrow_bps_annual, audit)
    sym_y, sym_x = run.symbol_y, run.symbol_x
    executor = PairExecutor(broker, risk, (sym_y, sym_x), execution)
    signal = RollingZ(rule.window)

    n = y.shape[0]
    equity = np.empty(n)
    breaks: list[Break] = []
    phase = _Phase.FLAT
    entry_bar = -1
    entry_equity = 0.0
    entries = 0
    no_entry_before = 0

    for t in range(n):
        py, px = float(y[t]), float(x[t])
        quotes = {
            sym_y: Quote(py, run.adv_y, float(y[t - 1]) if t else math.nan),
            sym_x: Quote(px, run.adv_x, float(x[t - 1]) if t else math.nan),
        }
        ledger.accrue_borrow(quotes)
        broker.on_bar(t, quotes)
        ledger.ingest(broker.fills())
        found = ledger.reconcile(broker, t)
        if found:
            breaks.extend(found)
            risk.trip(t, "RECONCILIATION_BREAK")
        equity[t] = ledger.equity({sym_y: py, sym_x: px})
        risk.on_equity(t, float(equity[t]))
        z = signal.update(py - alpha - beta * px)

        executor.step(t, quotes)
        if not executor.active:
            if phase is _Phase.ENTERING:
                if executor.last_outcome is Outcome.FILLED:
                    phase = _Phase.HOLDING
                else:
                    phase = _Phase.FLAT
            elif phase is _Phase.EXITING:
                phase = _Phase.FLAT

        if risk.killed or t >= n - 1 - run.wind_down_bars:
            if phase in (_Phase.ENTERING, _Phase.HOLDING):
                audit.append({"event": "flatten", "bar": t, "why": risk.kill_reason or "WIND_DOWN"})
                executor.flatten(t)
                executor.step(t, quotes)
                phase = _Phase.EXITING
            continue

        # NaN z fails every comparison below, so warm-up needs no special case
        if phase is _Phase.FLAT and rule.entry_z <= abs(z) < rule.stop_z and t >= no_entry_before:
            direction = -1 if z > 0.0 else 1
            if rule.sizing == "beta_hedged":
                unit = rule.capital / (py + beta * px)
                qty_y, qty_x = direction * unit, -direction * beta * unit
            else:
                qty_y = direction * (rule.capital / 2.0) / py
                qty_x = -direction * (rule.capital / 2.0) / px
            audit.append({"event": "signal", "bar": t, "action": "enter", "z": z, "direction": direction})
            executor.enter(t, {sym_y: round_shares(qty_y), sym_x: round_shares(qty_x)})
            executor.step(t, quotes)
            entries += 1
            phase, entry_bar, entry_equity = _Phase.ENTERING, t, float(equity[t])
        elif phase in (_Phase.ENTERING, _Phase.HOLDING):
            stopped = abs(z) >= rule.stop_z or (
                rule.pnl_stop_frac > 0.0 and equity[t] - entry_equity <= -rule.pnl_stop_frac * rule.capital
            )
            aged = rule.max_hold_days > 0 and t - entry_bar >= rule.max_hold_days
            normal_exit = phase is _Phase.HOLDING and (aged or abs(z) <= rule.exit_z)
            if stopped or normal_exit:
                audit.append({"event": "signal", "bar": t, "action": "exit", "z": z, "stop": bool(stopped)})
                executor.flatten(t)
                executor.step(t, quotes)
                phase = _Phase.EXITING
                if stopped:
                    no_entry_before = t + 1 + rule.cooldown_days

    rejections: dict[str, int] = {}
    for entry in risk.log:
        for reason in entry.get("reasons", []):
            rejections[reason] = rejections.get(reason, 0) + 1
    return PaperRunResult(
        equity=equity,
        attribution=ledger.attribution({sym_y: float(y[-1]), sym_x: float(x[-1])}),
        fills=len(broker.fills()),
        entries=entries,
        outcomes={k.value: v for k, v in executor.outcomes.items()},
        rejections=rejections,
        breaks=breaks,
        kill_reason=risk.kill_reason,
        final_positions=broker.positions(),
        max_entry_gap=executor.max_entry_gap,
        max_entry_excess=executor.max_entry_excess,
        audit=audit,
        ledger=ledger,
    )
