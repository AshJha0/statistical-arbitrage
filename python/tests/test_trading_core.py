"""Orders, broker, ledger and the no-network rule for the paper trading layer."""
from __future__ import annotations

import ast
import json
import math
from itertools import product
from pathlib import Path

import pytest

import statarb.trading as trading
from statarb.trading import (
    AuditLog,
    BreakKind,
    Broker,
    ClientOrderIdGenerator,
    Fill,
    IllegalTransition,
    Ledger,
    Order,
    OrderStatus,
    PaperBroker,
    PaperBrokerConfig,
    Quote,
    Side,
    round_shares,
)
from statarb.trading.orders import LEGAL_TRANSITIONS

Q = {"A": Quote(100.0, 10_000.0, 100.0), "B": Quote(50.0, 10_000.0, 50.0)}


def order_in(status: OrderStatus) -> Order:
    order = Order("c1", "A", Side.BUY, 10)
    if status is OrderStatus.PARTIAL:
        order.apply_fill(4)
    elif status is not OrderStatus.NEW:
        order.status = status
    return order


@pytest.mark.parametrize("start,end", list(product(OrderStatus, OrderStatus)))
def test_every_transition_pair(start: OrderStatus, end: OrderStatus) -> None:
    order = order_in(start)
    legal = {
        OrderStatus.NEW: {OrderStatus.PARTIAL, OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED},
        OrderStatus.PARTIAL: {OrderStatus.PARTIAL, OrderStatus.FILLED, OrderStatus.CANCELLED},
    }.get(start, set())
    assert LEGAL_TRANSITIONS[start] == legal
    if end in legal:
        order.transition(end)
        assert order.status is end
    else:
        with pytest.raises(IllegalTransition):
            order.transition(end)
        assert order.status is start


def test_fill_path_and_overfill() -> None:
    order = Order("c1", "A", Side.SELL, 10)
    order.apply_fill(4)
    assert (order.status, order.remaining) == (OrderStatus.PARTIAL, 6)
    with pytest.raises(ValueError):
        order.apply_fill(7)
    with pytest.raises(ValueError):
        order.apply_fill(0)
    order.apply_fill(6)
    assert order.status is OrderStatus.FILLED and not order.is_open
    with pytest.raises(ValueError):
        order.apply_fill(1)


@pytest.mark.parametrize("qty", [0, -1, 1.5, True])
def test_order_needs_whole_positive_shares(qty: object) -> None:
    with pytest.raises(ValueError):
        Order("c1", "A", Side.BUY, qty)  # type: ignore[arg-type]


def test_round_shares_truncates_toward_zero() -> None:
    assert (round_shares(9.99), round_shares(-9.99), round_shares(0.5)) == (9, -9, 0)
    with pytest.raises(ValueError):
        round_shares(math.nan)


def test_id_generator_is_deterministic() -> None:
    a, b = ClientOrderIdGenerator("x"), ClientOrderIdGenerator("x")
    assert [a.next(), a.next()] == [b.next(), b.next()] == ["x-00000001", "x-00000002"]


def test_submit_is_idempotent_by_client_id() -> None:
    broker = PaperBroker()
    broker.submit(Order("c1", "A", Side.BUY, 10))
    broker.submit(Order("c1", "A", Side.BUY, 10))
    assert len(broker.open_orders()) == 1
    broker.on_bar(0, Q)
    assert broker.positions() == {"A": 10}
    again = broker.submit(Order("c1", "A", Side.BUY, 10))
    assert again.status is OrderStatus.FILLED
    broker.on_bar(1, Q)
    assert broker.positions() == {"A": 10}
    with pytest.raises(ValueError):
        broker.submit(Order("c1", "A", Side.BUY, 11))


def test_returned_orders_are_snapshots() -> None:
    broker = PaperBroker()
    snapshot = broker.submit(Order("c1", "A", Side.BUY, 10))
    snapshot.filled_qty = 9
    assert broker.order("c1").filled_qty == 0


def test_fill_price_has_half_spread_and_sqrt_impact() -> None:
    cfg = PaperBrokerConfig(half_spread_bps=2.0, impact_bps=100.0, max_participation=1.0)
    broker = PaperBroker(cfg)
    broker.submit(Order("c1", "A", Side.BUY, 2_500))
    broker.submit(Order("c2", "B", Side.SELL, 2_500))
    buy, sell = broker.on_bar(0, Q)
    assert buy.price == pytest.approx(100.0 * (1 + (2.0 + 100.0 * 0.5) / 1e4))
    assert sell.price == pytest.approx(50.0 * (1 - (2.0 + 100.0 * 0.5) / 1e4))
    assert buy.half_spread_cost == pytest.approx(2_500 * 100.0 * 2e-4)
    assert buy.impact_cost == pytest.approx(2_500 * 100.0 * 50e-4)


def test_participation_cap_gives_whole_share_partials() -> None:
    broker = PaperBroker(PaperBrokerConfig(max_participation=0.01))
    quotes = {"A": Quote(100.0, 1_050.0, 100.0)}  # cap 10.5 shares a bar
    broker.submit(Order("c1", "A", Side.BUY, 25))
    assert [f.qty for f in broker.on_bar(0, quotes)] == [10]
    assert broker.order("c1").status is OrderStatus.PARTIAL
    assert [f.qty for f in broker.on_bar(1, quotes)] == [10]
    assert [f.qty for f in broker.on_bar(2, quotes)] == [5]
    assert broker.order("c1").status is OrderStatus.FILLED


def test_cap_is_shared_by_orders_in_one_name() -> None:
    broker = PaperBroker(PaperBrokerConfig(max_participation=0.01))
    broker.submit(Order("c1", "A", Side.BUY, 80))
    broker.submit(Order("c2", "A", Side.BUY, 80))
    assert sum(f.qty for f in broker.on_bar(0, Q)) == 100


def test_leg_latency_delays_only_that_leg() -> None:
    broker = PaperBroker(PaperBrokerConfig(leg_latency_bars={"B": 2}))
    broker.submit(Order("a", "A", Side.BUY, 1))
    broker.submit(Order("b", "B", Side.BUY, 1))
    assert [f.symbol for f in broker.on_bar(0, Q)] == ["A"]
    assert broker.on_bar(1, Q) == []
    assert [f.symbol for f in broker.on_bar(2, Q)] == ["B"]


def test_borrow_unavailable_rejects_shorts_but_not_sales_of_longs() -> None:
    broker = PaperBroker(PaperBrokerConfig(borrow_unavailable=frozenset({"A"})))
    assert broker.submit(Order("s0", "A", Side.SELL, 1)).reject_reason == "BORROW_UNAVAILABLE"
    assert broker.order("s0").status is OrderStatus.REJECTED
    broker.submit(Order("b", "A", Side.BUY, 10))
    broker.on_bar(0, Q)
    assert broker.submit(Order("s1", "A", Side.SELL, 6)).status is OrderStatus.NEW
    assert broker.submit(Order("s2", "A", Side.SELL, 6)).status is OrderStatus.REJECTED  # 6 already working


def test_cancel_stops_fills_and_is_harmless_when_finished() -> None:
    broker = PaperBroker()
    broker.submit(Order("c1", "A", Side.BUY, 10))
    assert broker.cancel("c1").status is OrderStatus.CANCELLED
    assert broker.on_bar(0, Q) == []
    assert broker.cancel("c1").status is OrderStatus.CANCELLED
    with pytest.raises(KeyError):
        broker.cancel("nope")


def test_same_seed_same_fills_other_seed_differs() -> None:
    def run(seed: int) -> list[Fill]:
        cfg = PaperBrokerConfig(spread_jitter=0.5, liquidity_jitter=0.5, max_participation=0.01)
        broker = PaperBroker(cfg, seed=seed)
        broker.submit(Order("c1", "A", Side.BUY, 500))
        for bar in range(10):
            broker.on_bar(bar, Q)
        return broker.fills()

    assert run(7) == run(7)
    assert run(7) != run(8)


def test_paper_broker_meets_the_protocol() -> None:
    assert isinstance(PaperBroker(), Broker)


def test_no_network_or_credentials_in_the_package() -> None:
    allowed = {"__future__", "collections", "copy", "dataclasses", "enum", "json", "math", "numpy",
               "pathlib", "typing"}
    root = Path(trading.__file__).parent
    for path in root.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                names = [node.module or ""]
            else:
                continue
            assert {n.split(".")[0] for n in names} <= allowed, path


def book(broker: PaperBroker, ledger: Ledger, bar: int, quotes: dict[str, Quote]) -> None:
    ledger.accrue_borrow(quotes)
    broker.on_bar(bar, quotes)
    ledger.ingest(broker.fills())


def traded_pair() -> tuple[PaperBroker, Ledger]:
    cfg = PaperBrokerConfig(commission_bps=1.0, borrow_bps_annual=300.0, spread_jitter=0.3)
    broker = PaperBroker(cfg, seed=3)
    ledger = Ledger(cfg.initial_cash, cfg.borrow_bps_annual)
    broker.submit(Order("o1", "A", Side.BUY, 100))
    broker.submit(Order("o2", "B", Side.SELL, 200))
    book(broker, ledger, 0, Q)
    broker.submit(Order("o3", "A", Side.SELL, 160))  # crosses through zero
    broker.submit(Order("o4", "B", Side.BUY, 50))
    moved = {"A": Quote(103.0, 10_000.0, 100.0), "B": Quote(49.0, 10_000.0, 50.0)}
    book(broker, ledger, 1, moved)
    return broker, ledger


def test_accounting_identities_and_clean_reconciliation() -> None:
    broker, ledger = traded_pair()
    assert ledger.reconcile(broker, 1) == []
    assert ledger.positions() == {"A": -60, "B": -150}
    marks = {"A": 97.0, "B": 52.5}
    equity = ledger.equity(marks)
    assert equity == pytest.approx(broker.cash() + sum(q * marks[s] for s, q in broker.positions().items()))
    assert equity == pytest.approx(
        ledger.initial_cash + ledger.realised + ledger.unrealised(marks) - ledger.commission - ledger.borrow
    )
    attribution = ledger.attribution(marks)
    assert equity == pytest.approx(ledger.initial_cash + attribution.total)
    assert ledger.borrow > 0.0 and attribution.costs > 0.0


def test_realised_pnl_average_cost_by_hand() -> None:
    def fill(i: int, side: Side, qty: int, price: float) -> Fill:
        return Fill(f"f{i}", f"c{i}", "A", side, qty, price, price, i, 0.0, 0.0, 0.0)

    ledger = Ledger(0.0)
    ledger.apply_fill(fill(1, Side.BUY, 10, 100.0))
    ledger.apply_fill(fill(2, Side.BUY, 10, 110.0))
    ledger.apply_fill(fill(3, Side.SELL, 5, 120.0))
    assert ledger.realised == pytest.approx(5 * 15.0)
    ledger.apply_fill(fill(4, Side.SELL, 20, 100.0))  # closes 15, opens short 5 at 100
    assert ledger.realised == pytest.approx(75.0 - 15 * 5.0)
    assert ledger.unrealised({"A": 90.0}) == pytest.approx(50.0)
    with pytest.raises(ValueError):
        ledger.apply_fill(fill(4, Side.SELL, 1, 100.0))


def test_ingest_is_safe_to_repeat() -> None:
    broker, ledger = traded_pair()
    assert ledger.ingest(broker.fills()) == 0
    assert ledger.reconcile(broker, 2) == []


def test_position_and_cash_breaks() -> None:
    broker, ledger = traded_pair()
    broker._positions["A"] += 1
    broker._cash += 0.01
    found = ledger.reconcile(broker, 2)
    assert {(b.kind, b.key) for b in found} == {(BreakKind.POSITION, "A"), (BreakKind.CASH, "cash")}
    position = next(b for b in found if b.kind is BreakKind.POSITION)
    assert (position.ledger, position.broker) == (-60, -59)
    assert sum(r["event"] == "reconciliation_break" for r in ledger.audit.records) == 2


def test_missing_fill_breaks_both_ways() -> None:
    broker, ledger = traded_pair()
    broker.submit(Order("o5", "A", Side.BUY, 1))
    broker.on_bar(2, Q)
    assert BreakKind.MISSING_IN_LEDGER in {b.kind for b in ledger.reconcile(broker, 2)}
    ledger.ingest(broker.fills())
    broker._fills.pop()
    assert BreakKind.MISSING_AT_BROKER in {b.kind for b in ledger.reconcile(broker, 3)}


def test_audit_log_appends_across_reopen_and_stays_valid_json(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path)
    log.append({"event": "a", "value": math.nan})
    first = path.read_text(encoding="utf-8")
    AuditLog(path).append({"event": "b"})
    text = path.read_text(encoding="utf-8")
    assert text.startswith(first)
    rows = [json.loads(line) for line in text.splitlines()]
    assert [r["seq"] for r in rows] == [0, 1]
    assert rows[0]["value"] is None
