"""Two-leg execution with a bound on leg risk.

Each leg's progress is the share of its target quantity filled.  Child orders
are paced: on every step a leg may have at most ``ceil((m + b) * target)``
shares sent, where ``m`` is the progress of the slower leg and ``b`` is
``max_imbalance_frac``.  So a leading leg can never run more than ``b`` ahead
(plus one share of rounding), whatever the fill model does; the bound does not
rely on the other leg filling.  As the slow leg fills, the fast leg is released.

An entry that is not complete after ``timeout_bars``, or that has a child
rejected, is cancelled and unwound back to where it started.  The unwind is
paced the same way, so measured against the original target its imbalance
stays under ``2 * b`` plus rounding.  Exits and unwinds never give up: they
retry until done, and a rejected child is tried again on the next bar.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from .broker import Broker
from .orders import ClientOrderIdGenerator, Order, Quote, Side
from .risk import RiskEngine


class AttemptKind(Enum):
    ENTER = "ENTER"
    EXIT = "EXIT"
    UNWIND = "UNWIND"


class Outcome(Enum):
    FILLED = "FILLED"
    FLAT = "FLAT"
    UNWOUND = "UNWOUND"


_OUTCOME = {AttemptKind.ENTER: Outcome.FILLED, AttemptKind.EXIT: Outcome.FLAT, AttemptKind.UNWIND: Outcome.UNWOUND}


@dataclass(frozen=True)
class ExecutionConfig:
    max_imbalance_frac: float = 0.25
    timeout_bars: int = 10

    def __post_init__(self) -> None:
        if not 0.0 < self.max_imbalance_frac <= 1.0:
            raise ValueError("max_imbalance_frac must lie in (0, 1]")
        if self.timeout_bars < 1:
            raise ValueError("timeout_bars must be >= 1")


class PairExecutor:
    """Moves a pair of legs to a target position, one attempt at a time."""

    def __init__(
        self,
        broker: Broker,
        risk: RiskEngine,
        legs: tuple[str, str],
        config: ExecutionConfig | None = None,
        ids: ClientOrderIdGenerator | None = None,
    ) -> None:
        self.broker = broker
        self.risk = risk
        self.legs = legs
        self.config = config if config is not None else ExecutionConfig()
        self.ids = ids if ids is not None else ClientOrderIdGenerator("pair")
        self.last_outcome: Outcome | None = None
        self.outcomes: dict[Outcome, int] = dict.fromkeys(Outcome, 0)
        self.max_entry_excess = -math.inf  # largest entry gap minus its bound; <= 0 means held
        self.max_entry_gap = 0.0
        self._kind: AttemptKind | None = None
        self._origin: dict[str, int] = {}
        self._delta: dict[str, int] = {}
        self._children: dict[str, list[str]] = {leg: [] for leg in legs}
        self._start_bar = 0
        self._rejected = False

    @property
    def active(self) -> bool:
        return self._kind is not None

    def enter(self, bar: int, target: Mapping[str, int]) -> None:
        """Start moving to ``target``; unwinds on timeout or rejection."""
        if self.active:
            raise RuntimeError("an attempt is already running")
        self._begin(AttemptKind.ENTER, bar, target)

    def flatten(self, bar: int) -> None:
        """Cancel whatever is running and go flat; never gives up."""
        self._cancel_children()
        self._begin(AttemptKind.EXIT, bar, dict.fromkeys(self.legs, 0))

    def step(self, bar: int, quotes: Mapping[str, Quote]) -> None:
        """Run once per bar after fills: finish, abort, or send the next children."""
        if self._kind is None:
            return
        progress = self._progress()
        self._track_entry(progress)
        if self._finished(progress):
            return
        if self._kind is AttemptKind.ENTER and (
            self._rejected or bar - self._start_bar >= self.config.timeout_bars
        ):
            self._cancel_children()
            self._begin(AttemptKind.UNWIND, bar, self._origin)
            progress = self._progress()
            if self._finished(progress):
                return
        self._send(bar, quotes, progress)

    def _begin(self, kind: AttemptKind, bar: int, target: Mapping[str, int]) -> None:
        held = self.broker.positions()
        if kind is not AttemptKind.UNWIND:
            self._origin = {leg: held.get(leg, 0) for leg in self.legs}
        self._kind = kind
        self._start_bar = bar
        self._rejected = False
        self._delta = {leg: target[leg] - held.get(leg, 0) for leg in self.legs}
        self._children = {leg: [] for leg in self.legs}

    def _finished(self, progress: Mapping[str, float]) -> bool:
        if min(progress.values()) < 1.0:
            return False
        assert self._kind is not None
        self.last_outcome = _OUTCOME[self._kind]
        self.outcomes[self.last_outcome] += 1
        self._kind = None
        return True

    def _progress(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for leg in self.legs:
            filled = sum(self.broker.order(c).filled_qty for c in self._children[leg])
            result[leg] = filled / abs(self._delta[leg]) if self._delta[leg] else 1.0
        return result

    def _sent(self, leg: str) -> int:
        orders = [self.broker.order(c) for c in self._children[leg]]
        return sum(o.qty if o.is_open else o.filled_qty for o in orders)

    def _track_entry(self, progress: Mapping[str, float]) -> None:
        sizes = [abs(d) for d in self._delta.values()]
        if self._kind is AttemptKind.ENTER and min(sizes) > 0:
            gap = abs(progress[self.legs[0]] - progress[self.legs[1]])
            self.max_entry_gap = max(self.max_entry_gap, gap)
            bound = self.config.max_imbalance_frac + 1.0 / min(sizes)
            self.max_entry_excess = max(self.max_entry_excess, gap - bound)

    def _send(self, bar: int, quotes: Mapping[str, Quote], progress: Mapping[str, float]) -> None:
        slowest = min(progress.values())
        for leg in sorted(self.legs, key=lambda s: progress[s]):  # slow leg gets risk headroom first
            size = abs(self._delta[leg])
            allowed = min(size, math.ceil((slowest + self.config.max_imbalance_frac) * size))
            qty = allowed - self._sent(leg)
            if qty <= 0:
                continue
            side = Side.BUY if self._delta[leg] > 0 else Side.SELL
            order = Order(self.ids.next(), leg, side, qty)
            decision = self.risk.check(order, self.broker.positions(), self.broker.open_orders(), quotes, bar)
            if not decision.approved:
                self._rejected = True
                continue
            placed = self.broker.submit(order)
            self._children[leg].append(placed.client_id)
            if placed.reject_reason:
                self._rejected = True

    def _cancel_children(self) -> None:
        for children in self._children.values():
            for child in children:
                self.broker.cancel(child)
