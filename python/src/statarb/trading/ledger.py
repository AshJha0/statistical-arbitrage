"""Independent books: positions, P&L, reconciliation, audit trail, attribution.

The ledger rebuilds cash and positions from fills and its own borrow accrual,
so a reconciliation against the broker compares two separate calculations.

These identities hold at any mark (see the tests)::

    equity = cash + sum(qty * mark)
           = initial + realised + unrealised - commission - borrow
           = initial + spread_pnl - half_spread - impact - commission - borrow

``realised`` and ``unrealised`` are on fill prices, so they already contain the
half-spread and impact.  ``spread_pnl`` is the same trades valued at mid.
"""
from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from .broker import Broker, borrow_fee
from .orders import Fill, Quote


def _clean(value: Any) -> Any:
    """JSON has no NaN or infinity; they are written as null."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


class AuditLog:
    """Append-only JSONL.  Opens in append mode and has no way to rewrite.

    Records carry a running ``seq`` that continues across reopenings and the
    bar number rather than wall-clock time, so a seeded run replays byte for
    byte.  With no path the records are only kept in memory.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self._seq = 0
        if self.path is not None and self.path.exists():
            with self.path.open("r", encoding="utf-8") as handle:
                self._seq = sum(1 for _ in handle)
        self.records: list[dict[str, Any]] = []

    def append(self, record: Mapping[str, Any]) -> None:
        entry = {"seq": self._seq, **_clean(dict(record))}
        line = json.dumps(entry, sort_keys=True, allow_nan=False)
        if self.path is not None:
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(line + "\n")
        self._seq += 1
        self.records.append(entry)


class BreakKind(Enum):
    POSITION = "POSITION"
    CASH = "CASH"
    MISSING_IN_LEDGER = "MISSING_IN_LEDGER"
    MISSING_AT_BROKER = "MISSING_AT_BROKER"


@dataclass(frozen=True)
class Break:
    kind: BreakKind
    key: str
    ledger: float
    broker: float


@dataclass(frozen=True)
class Attribution:
    """P&L by source.  ``spread`` is the P&L at mid; the rest are costs >= 0."""

    spread: float
    half_spread: float
    impact: float
    commission: float
    borrow: float

    @property
    def costs(self) -> float:
        return self.half_spread + self.impact + self.commission

    @property
    def total(self) -> float:
        return self.spread - self.costs - self.borrow


class Ledger:
    """Average-cost position keeping from a stream of fills."""

    def __init__(
        self,
        initial_cash: float,
        borrow_bps_annual: float = 0.0,
        audit: AuditLog | None = None,
        cash_tolerance: float = 1.0e-6,
    ) -> None:
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.borrow_bps_annual = borrow_bps_annual
        self.audit = audit if audit is not None else AuditLog()
        self.cash_tolerance = cash_tolerance
        self.realised = 0.0
        self.half_spread = 0.0
        self.impact = 0.0
        self.commission = 0.0
        self.borrow = 0.0
        self._qty: dict[str, int] = {}
        self._avg: dict[str, float] = {}
        self._mid_basis = 0.0
        self._seen: set[str] = set()

    def positions(self) -> dict[str, int]:
        return {s: q for s, q in self._qty.items() if q != 0}

    def apply_fill(self, fill: Fill) -> None:
        """Book one fill.  Booking the same fill twice raises ValueError."""
        if fill.fill_id in self._seen:
            raise ValueError(f"fill {fill.fill_id} already booked")
        self._seen.add(fill.fill_id)
        held = self._qty.get(fill.symbol, 0)
        avg = self._avg.get(fill.symbol, 0.0)
        signed = fill.signed_qty
        if held == 0 or (held > 0) == (signed > 0):
            avg = (abs(held) * avg + fill.qty * fill.price) / (abs(held) + fill.qty)
        else:
            closing = min(fill.qty, abs(held))
            self.realised += closing * (fill.price - avg) * (1 if held > 0 else -1)
            if fill.qty > abs(held):
                avg = fill.price
            elif fill.qty == abs(held):
                avg = 0.0
        self._qty[fill.symbol] = held + signed
        self._avg[fill.symbol] = avg
        self.cash -= signed * fill.price + fill.commission
        self._mid_basis += signed * fill.mid
        self.half_spread += fill.half_spread_cost
        self.impact += fill.impact_cost
        self.commission += fill.commission
        self.audit.append({
            "event": "fill", "bar": fill.bar, "fill_id": fill.fill_id, "client_id": fill.client_id,
            "symbol": fill.symbol, "side": fill.side.name, "qty": fill.qty, "price": fill.price,
            "mid": fill.mid,
        })

    def ingest(self, fills: Iterable[Fill]) -> int:
        """Book fills not yet seen; safe to call with the full history."""
        new = [f for f in fills if f.fill_id not in self._seen]
        for fill in new:
            self.apply_fill(fill)
        return len(new)

    def accrue_borrow(self, quotes: Mapping[str, Quote]) -> float:
        """One bar of borrow fee on the shorts held now.  Call before booking
        that bar's fills: the fee is on the position carried into the bar."""
        total = 0.0
        for symbol, qty in sorted(self._qty.items()):
            if qty < 0:
                total += borrow_fee(qty, quotes[symbol].mid, self.borrow_bps_annual)
        self.cash -= total
        self.borrow += total
        return total

    def unrealised(self, marks: Mapping[str, float]) -> float:
        return sum(q * (marks[s] - self._avg[s]) for s, q in self._qty.items() if q != 0)

    def market_value(self, marks: Mapping[str, float]) -> float:
        return sum(q * marks[s] for s, q in self._qty.items() if q != 0)

    def equity(self, marks: Mapping[str, float]) -> float:
        return self.cash + self.market_value(marks)

    def attribution(self, marks: Mapping[str, float]) -> Attribution:
        return Attribution(self.market_value(marks) - self._mid_basis, self.half_spread, self.impact,
                           self.commission, self.borrow)

    def reconcile(self, broker: Broker, bar: int) -> list[Break]:
        """Compare with the broker; every break is returned and audited."""
        breaks: list[Break] = []
        ours, theirs = self.positions(), broker.positions()
        for symbol in sorted(set(ours) | set(theirs)):
            if ours.get(symbol, 0) != theirs.get(symbol, 0):
                breaks.append(Break(BreakKind.POSITION, symbol, ours.get(symbol, 0), theirs.get(symbol, 0)))
        if abs(self.cash - broker.cash()) > self.cash_tolerance:
            breaks.append(Break(BreakKind.CASH, "cash", self.cash, broker.cash()))
        ids = {f.fill_id for f in broker.fills()}
        for fill_id in sorted(ids - self._seen):
            breaks.append(Break(BreakKind.MISSING_IN_LEDGER, fill_id, 0.0, 1.0))
        for fill_id in sorted(self._seen - ids):
            breaks.append(Break(BreakKind.MISSING_AT_BROKER, fill_id, 1.0, 0.0))
        for item in breaks:
            self.audit.append({"event": "reconciliation_break", "bar": bar, "kind": item.kind.value,
                               "key": item.key, "ledger": item.ledger, "broker": item.broker})
        return breaks
