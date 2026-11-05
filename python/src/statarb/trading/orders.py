"""Orders, fills and the order state machine (paper trading only).

Legal transitions::

    NEW     -> PARTIAL | FILLED | CANCELLED | REJECTED
    PARTIAL -> PARTIAL | FILLED | CANCELLED
    FILLED, CANCELLED, REJECTED are terminal

Anything else raises :class:`IllegalTransition`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, IntEnum


class IllegalTransition(RuntimeError):
    """An order was asked to make a transition the state machine forbids."""


class OrderStatus(Enum):
    NEW = "NEW"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class Side(IntEnum):
    BUY = 1
    SELL = -1


LEGAL_TRANSITIONS: dict[OrderStatus, frozenset[OrderStatus]] = {
    OrderStatus.NEW: frozenset(
        {OrderStatus.PARTIAL, OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED}
    ),
    OrderStatus.PARTIAL: frozenset({OrderStatus.PARTIAL, OrderStatus.FILLED, OrderStatus.CANCELLED}),
    OrderStatus.FILLED: frozenset(),
    OrderStatus.CANCELLED: frozenset(),
    OrderStatus.REJECTED: frozenset(),
}


def round_shares(quantity: float) -> int:
    """Whole shares, truncated toward zero so a size is never rounded up."""
    if not math.isfinite(quantity):
        raise ValueError("quantity must be finite")
    return math.trunc(quantity)


@dataclass(frozen=True)
class Quote:
    """One symbol on one bar.  ``adv`` is average daily volume in shares."""

    mid: float
    adv: float
    prev_close: float = math.nan


@dataclass(frozen=True)
class Fill:
    """One execution.  Cost fields are positive dollars paid on this fill.

    ``price`` already contains the half-spread and the impact; ``commission``
    is charged to cash on top of it.
    """

    fill_id: str
    client_id: str
    symbol: str
    side: Side
    qty: int
    price: float
    mid: float
    bar: int
    half_spread_cost: float
    impact_cost: float
    commission: float

    @property
    def signed_qty(self) -> int:
        return int(self.side) * self.qty


@dataclass
class Order:
    """A market order for a whole number of shares, ``qty`` > 0."""

    client_id: str
    symbol: str
    side: Side
    qty: int
    status: OrderStatus = OrderStatus.NEW
    filled_qty: int = 0
    reject_reason: str = ""

    def __post_init__(self) -> None:
        if not self.client_id:
            raise ValueError("client_id must not be empty")
        if isinstance(self.qty, bool) or not isinstance(self.qty, int) or self.qty <= 0:
            raise ValueError("qty must be a positive whole number of shares")
        self.side = Side(self.side)

    @property
    def remaining(self) -> int:
        return self.qty - self.filled_qty

    @property
    def is_open(self) -> bool:
        return self.status in (OrderStatus.NEW, OrderStatus.PARTIAL)

    def transition(self, new: OrderStatus) -> None:
        if new not in LEGAL_TRANSITIONS[self.status]:
            raise IllegalTransition(f"{self.client_id}: {self.status.value} -> {new.value}")
        self.status = new

    def apply_fill(self, qty: int) -> None:
        """Record ``qty`` filled shares, moving to PARTIAL or FILLED."""
        if qty <= 0 or qty > self.remaining:
            raise ValueError(f"{self.client_id}: fill of {qty} with {self.remaining} remaining")
        self.transition(OrderStatus.FILLED if qty == self.remaining else OrderStatus.PARTIAL)
        self.filled_qty += qty

    def cancel(self) -> None:
        self.transition(OrderStatus.CANCELLED)

    def reject(self, reason: str) -> None:
        self.transition(OrderStatus.REJECTED)
        self.reject_reason = reason


class ClientOrderIdGenerator:
    """Deterministic ids ``prefix-00000001``, so a seeded run replays exactly."""

    def __init__(self, prefix: str = "ord") -> None:
        self.prefix = prefix
        self._n = 0

    def next(self) -> str:
        self._n += 1
        return f"{self.prefix}-{self._n:08d}"
