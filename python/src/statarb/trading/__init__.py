"""Simulated execution, risk and broker layers around the pairs rule.

PAPER TRADING ONLY.  This package has no network code, holds no credentials
and has no live-order path.  :class:`~statarb.trading.broker.Broker` is the
seam a real adapter could implement later; writing one is an owner decision
and is out of scope here.

Python only: not ported to the other languages and not covered by goldens.
"""
from __future__ import annotations

from .broker import Broker, PaperBroker, PaperBrokerConfig
from .execution import ExecutionConfig, Outcome, PairExecutor
from .ledger import Attribution, AuditLog, Break, BreakKind, Ledger
from .orders import (
    ClientOrderIdGenerator,
    Fill,
    IllegalTransition,
    Order,
    OrderStatus,
    Quote,
    Side,
    round_shares,
)
from .risk import RiskDecision, RiskEngine, RiskLimits, RiskReason
from .runner import PaperRunConfig, PaperRunResult, run_paper

__all__ = [
    "Attribution",
    "AuditLog",
    "Break",
    "BreakKind",
    "Broker",
    "ClientOrderIdGenerator",
    "ExecutionConfig",
    "Fill",
    "IllegalTransition",
    "Ledger",
    "Order",
    "OrderStatus",
    "Outcome",
    "PairExecutor",
    "PaperBroker",
    "PaperBrokerConfig",
    "PaperRunConfig",
    "PaperRunResult",
    "Quote",
    "RiskDecision",
    "RiskEngine",
    "RiskLimits",
    "RiskReason",
    "Side",
    "round_shares",
    "run_paper",
]
