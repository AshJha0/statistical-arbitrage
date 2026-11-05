"""Broker seam and the simulated broker.

:class:`Broker` is the interface a real adapter could implement.  A live
adapter is an owner decision and is out of scope: nothing in this package
opens a network connection, holds credentials or can send a live order.
:class:`PaperBroker` is the only implementation.

Timing: an order submitted after the close of bar ``t`` is first eligible on
bar ``t + 1 + latency``, the same next-bar execution as the backtest.
"""
from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from .orders import Fill, Order, OrderStatus, Quote, Side, round_shares

TRADING_DAYS = 252.0


@runtime_checkable
class Broker(Protocol):
    """What the execution layer needs.  Returned orders are snapshots.

    A live adapter is an owner decision and is out of scope for this package.
    """

    def submit(self, order: Order) -> Order: ...

    def cancel(self, client_id: str) -> Order: ...

    def order(self, client_id: str) -> Order: ...

    def open_orders(self) -> list[Order]: ...

    def positions(self) -> dict[str, int]: ...

    def cash(self) -> float: ...

    def fills(self) -> list[Fill]: ...


def borrow_fee(qty: int, mid: float, bps_annual: float) -> float:
    """One bar of borrow fee on a position; zero unless it is short."""
    return max(-qty, 0) * mid * bps_annual / 1.0e4 / TRADING_DAYS


@dataclass(frozen=True)
class PaperBrokerConfig:
    """Fill model.

    Attributes:
        initial_cash: Starting cash.
        half_spread_bps: Half-spread paid on every share, in bps of mid.
        spread_jitter: Seeded relative noise on the half-spread, in [0, 1).
        impact_bps: Impact at 100% of ADV; the cost of a fill is
            ``impact_bps * sqrt(fill_qty / adv)`` in bps of mid.
        max_participation: Largest share of ADV one symbol can trade per bar.
        liquidity_jitter: Seeded relative cut to that cap, in [0, 1).
        commission_bps: Commission on traded notional.
        borrow_bps_annual: Fee on short notional, accrued per bar on 252 bars.
        leg_latency_bars: Extra bars of delay by symbol.
        borrow_unavailable: Symbols that cannot be newly shorted.
    """

    initial_cash: float = 1_000_000.0
    half_spread_bps: float = 1.0
    spread_jitter: float = 0.0
    impact_bps: float = 10.0
    max_participation: float = 0.05
    liquidity_jitter: float = 0.0
    commission_bps: float = 0.0
    borrow_bps_annual: float = 0.0
    leg_latency_bars: Mapping[str, int] = field(default_factory=dict)
    borrow_unavailable: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        for name in ("half_spread_bps", "impact_bps", "commission_bps", "borrow_bps_annual"):
            value = getattr(self, name)
            if not (math.isfinite(value) and value >= 0.0):
                raise ValueError(f"{name} must be finite and >= 0")
        if not 0.0 < self.max_participation <= 1.0:
            raise ValueError("max_participation must lie in (0, 1]")
        if not (0.0 <= self.spread_jitter < 1.0 and 0.0 <= self.liquidity_jitter < 1.0):
            raise ValueError("jitter must lie in [0, 1)")
        if any(v < 0 for v in self.leg_latency_bars.values()):
            raise ValueError("latency must be >= 0")


class PaperBroker:
    """Deterministic simulated broker: same seed and inputs, same fills."""

    def __init__(self, config: PaperBrokerConfig | None = None, seed: int = 0) -> None:
        self.config = config if config is not None else PaperBrokerConfig()
        self.borrow_unavailable: set[str] = set(self.config.borrow_unavailable)
        self._rng = np.random.default_rng(seed)
        self._orders: dict[str, Order] = {}
        self._eligible: dict[str, int] = {}
        self._positions: dict[str, int] = {}
        self._cash = float(self.config.initial_cash)
        self._fills: list[Fill] = []
        self._bar = -1

    def submit(self, order: Order) -> Order:
        """Accept an order.  Resending a client id is a no-op that returns the
        original; reusing the id for a different order raises ValueError."""
        known = self._orders.get(order.client_id)
        if known is not None:
            if (known.symbol, known.side, known.qty) != (order.symbol, order.side, order.qty):
                raise ValueError(f"client id {order.client_id} reused for a different order")
            return copy.copy(known)
        if order.status is not OrderStatus.NEW or order.filled_qty:
            raise ValueError("only a NEW, unfilled order can be submitted")
        accepted = copy.copy(order)
        if accepted.side is Side.SELL and accepted.symbol in self.borrow_unavailable:
            working = sum(o.remaining for o in self._orders.values()
                          if o.is_open and o.symbol == accepted.symbol and o.side is Side.SELL)
            if self._positions.get(accepted.symbol, 0) - working - accepted.qty < 0:
                accepted.reject("BORROW_UNAVAILABLE")
        self._orders[accepted.client_id] = accepted
        latency = self.config.leg_latency_bars.get(accepted.symbol, 0)
        self._eligible[accepted.client_id] = self._bar + 1 + latency
        return copy.copy(accepted)

    def cancel(self, client_id: str) -> Order:
        """Cancel an open order; on a finished order this changes nothing."""
        order = self._orders[client_id]
        if order.is_open:
            order.cancel()
        return copy.copy(order)

    def order(self, client_id: str) -> Order:
        return copy.copy(self._orders[client_id])

    def open_orders(self) -> list[Order]:
        return [copy.copy(o) for o in self._orders.values() if o.is_open]

    def positions(self) -> dict[str, int]:
        return {s: q for s, q in self._positions.items() if q != 0}

    def cash(self) -> float:
        return self._cash

    def fills(self) -> list[Fill]:
        return list(self._fills)

    def on_bar(self, bar: int, quotes: Mapping[str, Quote]) -> list[Fill]:
        """Advance one bar: accrue borrow on carried shorts, then match orders."""
        if bar <= self._bar:
            raise ValueError("bars must strictly increase")
        self._bar = bar
        cfg = self.config
        for symbol, qty in sorted(self._positions.items()):
            if qty < 0:
                self._cash -= borrow_fee(qty, quotes[symbol].mid, cfg.borrow_bps_annual)
        budget: dict[str, int] = {}
        for symbol in sorted(quotes):
            cut = 1.0 - cfg.liquidity_jitter * float(self._rng.random())
            budget[symbol] = round_shares(cfg.max_participation * quotes[symbol].adv * cut)
        new_fills: list[Fill] = []
        for order in self._orders.values():
            if not order.is_open or self._eligible[order.client_id] > bar or order.symbol not in quotes:
                continue
            qty = min(order.remaining, budget[order.symbol])
            if qty <= 0:
                continue
            budget[order.symbol] -= qty
            noise = float(self._rng.uniform(-1.0, 1.0))
            new_fills.append(self._execute(order, qty, quotes[order.symbol], noise, bar))
        return new_fills

    def _execute(self, order: Order, qty: int, quote: Quote, noise: float, bar: int) -> Fill:
        cfg = self.config
        half = cfg.half_spread_bps * (1.0 + cfg.spread_jitter * noise) / 1.0e4
        impact = cfg.impact_bps * math.sqrt(qty / quote.adv) / 1.0e4
        price = quote.mid * (1.0 + int(order.side) * (half + impact))
        fill = Fill(
            fill_id=f"{order.client_id}#{len(self._fills) + 1}",
            client_id=order.client_id,
            symbol=order.symbol,
            side=order.side,
            qty=qty,
            price=price,
            mid=quote.mid,
            bar=bar,
            half_spread_cost=qty * quote.mid * half,
            impact_cost=qty * quote.mid * impact,
            commission=qty * price * cfg.commission_bps / 1.0e4,
        )
        order.apply_fill(qty)
        self._fills.append(fill)
        self._positions[order.symbol] = self._positions.get(order.symbol, 0) + fill.signed_qty
        self._cash -= fill.signed_qty * price + fill.commission
        return fill
