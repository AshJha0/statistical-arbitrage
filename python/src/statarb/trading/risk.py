"""Pre-trade checks and the post-trade drawdown kill switch.

Exposure is measured at the worst case: current position plus every open order
on the same side plus the new order, all at the current mid.  That makes the
limits hold however working orders end up filling.

Limits are tested at decision prices.  A later price move can take exposure
past a limit with no trade; from then on only orders that do not raise the
breached measure pass.

An order is *reducing* when, even if all open orders fill, it can only move
the position toward zero and cannot cross it.  Only position-increasing orders
are checked: a reducing order is always approved (and still logged), so risk
can always come off, under the kill switch, in a name restricted after entry,
or on bad market data.  An order that crosses zero is not reducing.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .orders import Order, Quote, Side


class RiskReason(Enum):
    KILL_SWITCH = "KILL_SWITCH"
    BAD_MARKET_DATA = "BAD_MARKET_DATA"
    RESTRICTED = "RESTRICTED"
    NO_LOCATE = "NO_LOCATE"
    PRICE_BAND = "PRICE_BAND"
    ORDER_SIZE_ADV = "ORDER_SIZE_ADV"
    MAX_NAME_NOTIONAL = "MAX_NAME_NOTIONAL"
    MAX_GROSS = "MAX_GROSS"
    MAX_NET = "MAX_NET"


@dataclass(frozen=True)
class RiskLimits:
    """Limits in dollars, except the fractions.

    Attributes:
        max_gross: Sum of absolute name notionals.
        max_net: Absolute signed sum of name notionals.
        max_name_notional: Largest absolute notional in one name.
        max_order_adv_frac: Largest order as a fraction of ADV.
        price_band_frac: Largest allowed |mid / prev_close - 1|.
        max_drawdown_frac: Peak-to-current equity loss that latches the switch.
        restricted: Symbols that may not trade at all.
        locates: Symbols that may be shorted; empty means no shorting.
    """

    max_gross: float
    max_net: float
    max_name_notional: float
    max_order_adv_frac: float = 0.05
    price_band_frac: float = 0.10
    max_drawdown_frac: float = 0.10
    restricted: frozenset[str] = frozenset()
    locates: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        for name in ("max_gross", "max_net", "max_name_notional", "max_order_adv_frac",
                     "price_band_frac", "max_drawdown_frac"):
            value = getattr(self, name)
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"{name} must be finite and > 0")


@dataclass(frozen=True)
class RiskDecision:
    client_id: str
    approved: bool
    reasons: tuple[RiskReason, ...]
    reducing: bool
    gross_after: float
    net_after: float
    name_after: float


def _good(value: float) -> bool:
    return math.isfinite(value) and value > 0.0


def _exposure(
    hi: Mapping[str, int], lo: Mapping[str, int], quotes: Mapping[str, Quote]
) -> tuple[float, float, float]:
    """Worst-case (gross, |net|, largest name) over the position ranges."""
    names = [max(abs(hi[s]), abs(lo[s])) * quotes[s].mid for s in hi]
    net_hi = sum(hi[s] * quotes[s].mid for s in hi)
    net_lo = sum(lo[s] * quotes[s].mid for s in hi)
    return float(sum(names)), max(abs(net_hi), abs(net_lo)), max(names, default=0.0)


class RiskEngine:
    """Limit checks plus a latching kill switch; logs every decision."""

    def __init__(self, limits: RiskLimits, sink: Callable[[dict[str, Any]], None] | None = None) -> None:
        self.limits = limits
        self.log: list[dict[str, Any]] = []
        self._sink = sink
        self._killed = False
        self._kill_reason = ""
        self._peak = -math.inf
        self._last_equity = math.nan

    @property
    def killed(self) -> bool:
        return self._killed

    @property
    def kill_reason(self) -> str:
        return self._kill_reason

    def _record(self, entry: dict[str, Any]) -> None:
        self.log.append(entry)
        if self._sink is not None:
            self._sink(entry)

    def check(
        self,
        order: Order,
        positions: Mapping[str, int],
        open_orders: Iterable[Order],
        quotes: Mapping[str, Quote],
        bar: int,
    ) -> RiskDecision:
        """Decide one order.  Returns every failed reason, not just the first."""
        working = list(open_orders)
        sym = order.symbol
        symbols = set(positions) | {o.symbol for o in working} | {sym}
        hi = {s: positions.get(s, 0) for s in symbols}
        lo = dict(hi)
        for o in working:
            if o.side is Side.BUY:
                hi[o.symbol] += o.remaining
            else:
                lo[o.symbol] -= o.remaining
        hi_after, lo_after = dict(hi), dict(lo)
        if order.side is Side.BUY:
            hi_after[sym] += order.qty
            reducing = hi_after[sym] <= 0
        else:
            lo_after[sym] -= order.qty
            reducing = lo_after[sym] >= 0

        reasons: list[RiskReason] = []
        if self._killed:
            reasons.append(RiskReason.KILL_SWITCH)
        if sym in self.limits.restricted:
            reasons.append(RiskReason.RESTRICTED)
        if order.side is Side.SELL and lo_after[sym] < 0 and sym not in self.limits.locates:
            reasons.append(RiskReason.NO_LOCATE)

        gross = net = name = math.nan
        quote = quotes.get(sym)
        marks_ok = all(s in quotes and _good(quotes[s].mid) for s in symbols)
        if quote is None or not (marks_ok and _good(quote.adv) and _good(quote.prev_close)):
            reasons.append(RiskReason.BAD_MARKET_DATA)
        else:
            if abs(quote.mid / quote.prev_close - 1.0) > self.limits.price_band_frac:
                reasons.append(RiskReason.PRICE_BAND)
            if order.qty > self.limits.max_order_adv_frac * quote.adv:
                reasons.append(RiskReason.ORDER_SIZE_ADV)
            gross0, net0, name0 = _exposure(hi, lo, quotes)
            gross, net, name = _exposure(hi_after, lo_after, quotes)
            if name > self.limits.max_name_notional and name > name0:
                reasons.append(RiskReason.MAX_NAME_NOTIONAL)
            if gross > self.limits.max_gross and gross > gross0:
                reasons.append(RiskReason.MAX_GROSS)
            if net > self.limits.max_net and net > net0:
                reasons.append(RiskReason.MAX_NET)

        if reducing:
            reasons = []
        decision = RiskDecision(order.client_id, not reasons, tuple(reasons), reducing, gross, net, name)
        self._record({
            "event": "risk_decision", "bar": bar, "client_id": order.client_id, "symbol": sym,
            "side": order.side.name, "qty": order.qty, "approved": decision.approved,
            "reasons": [r.value for r in reasons], "reducing": reducing,
            "gross_after": gross, "net_after": net, "name_after": name,
        })
        return decision

    def on_equity(self, bar: int, equity: float) -> bool:
        """Post-trade check; latches on drawdown or non-finite equity.  Returns ``killed``."""
        if not math.isfinite(equity):
            self.trip(bar, "NON_FINITE_EQUITY")
            return self._killed
        self._last_equity = equity
        self._peak = max(self._peak, equity)
        loss = self._peak - equity
        if loss > 0.0 and loss >= self.limits.max_drawdown_frac * abs(self._peak):
            self.trip(bar, "MAX_DRAWDOWN")
        return self._killed

    def trip(self, bar: int, reason: str) -> None:
        """Latch the switch.  Only :meth:`reset_kill_switch` clears it."""
        if self._killed:
            return
        self._killed = True
        self._kill_reason = reason
        self._record({"event": "kill_switch_tripped", "bar": bar, "reason": reason,
                      "equity": self._last_equity, "peak": self._peak})

    def reset_kill_switch(self, bar: int, operator: str, note: str) -> None:
        """Manual reset; the drawdown peak restarts from the last equity."""
        if not operator or not note:
            raise ValueError("a reset needs an operator and a note")
        self._record({"event": "kill_switch_reset", "bar": bar, "operator": operator, "note": note,
                      "was_killed": self._killed, "previous_reason": self._kill_reason})
        self._killed = False
        self._kill_reason = ""
        if math.isfinite(self._last_equity):
            self._peak = self._last_equity
