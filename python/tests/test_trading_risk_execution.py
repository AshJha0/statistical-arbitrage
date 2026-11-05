"""Risk reasons, kill switch, leg-risk bounds, the day loop and the limit property."""
from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from statarb import BacktestConfig, backtest_pair, ols
from statarb.trading import (
    ExecutionConfig,
    Order,
    Outcome,
    PairExecutor,
    PaperBroker,
    PaperBrokerConfig,
    PaperRunConfig,
    Quote,
    RiskEngine,
    RiskLimits,
    RiskReason,
    Side,
    run_paper,
)

Q = {"A": Quote(100.0, 100_000.0, 100.0), "B": Quote(50.0, 100_000.0, 50.0)}
LIMITS = RiskLimits(
    max_gross=100_000.0, max_net=60_000.0, max_name_notional=50_000.0, max_order_adv_frac=0.01,
    price_band_frac=0.1, max_drawdown_frac=0.1, restricted=frozenset({"R"}), locates=frozenset({"A", "B"}),
)


def decide(order: Order, positions: dict[str, int] | None = None, working: tuple[Order, ...] = (),
           quotes: dict[str, Quote] | None = None, engine: RiskEngine | None = None) -> set[RiskReason]:
    engine = engine if engine is not None else RiskEngine(LIMITS)
    decision = engine.check(order, positions or {}, working, quotes if quotes is not None else Q, 0)
    assert decision.approved == (not decision.reasons)
    return set(decision.reasons)


def buy(symbol: str, qty: int, cid: str = "c") -> Order:
    return Order(cid, symbol, Side.BUY, qty)


def sell(symbol: str, qty: int, cid: str = "c") -> Order:
    return Order(cid, symbol, Side.SELL, qty)


def test_clean_order_passes() -> None:
    assert decide(buy("A", 100)) == set()


def test_max_name() -> None:
    assert decide(buy("A", 1), {"A": 500}) == {RiskReason.MAX_NAME_NOTIONAL}


def test_max_net() -> None:
    assert decide(buy("B", 800), {"A": 450, "B": 100}) == {RiskReason.MAX_NET}


def test_max_gross_without_net() -> None:
    limits = replace(LIMITS, max_gross=70_000.0)
    engine = RiskEngine(limits)
    assert decide(sell("B", 900), {"A": 300, "B": -100}, engine=engine) == {RiskReason.MAX_GROSS}


def test_order_size_vs_adv() -> None:
    assert decide(buy("B", 11), quotes={"B": Quote(50.0, 1_000.0, 50.0)}) == {RiskReason.ORDER_SIZE_ADV}


def test_restricted_list() -> None:
    quotes = {"R": Quote(10.0, 100_000.0, 10.0)}
    assert decide(buy("R", 1), quotes=quotes) == {RiskReason.RESTRICTED}
    assert RiskReason.RESTRICTED in decide(sell("R", 6), {"R": 5}, quotes=quotes)  # crosses zero


def test_locate_list() -> None:
    quotes = {"N": Quote(10.0, 100_000.0, 10.0)}
    assert decide(sell("N", 1), quotes=quotes) == {RiskReason.NO_LOCATE}
    assert decide(sell("N", 5), {"N": 5}, quotes=quotes) == set()
    assert decide(sell("N", 6), {"N": 5}, quotes=quotes) == {RiskReason.NO_LOCATE}


def test_price_band() -> None:
    assert decide(buy("A", 1), quotes={"A": Quote(111.0, 100_000.0, 100.0)}) == {RiskReason.PRICE_BAND}


def test_kill_switch_blocks_opening_but_not_reducing() -> None:
    engine = RiskEngine(LIMITS)
    engine.trip(0, "MANUAL")
    assert decide(buy("A", 1), engine=engine) == {RiskReason.KILL_SWITCH}
    assert decide(sell("A", 5), {"A": 5}, engine=engine) == set()
    assert decide(sell("A", 6), {"A": 5}, engine=engine) == {RiskReason.KILL_SWITCH}


@pytest.mark.parametrize("quotes", [
    {},
    {"A": Quote(math.nan, 100_000.0, 100.0)},
    {"A": Quote(100.0, math.nan, 100.0)},
    {"A": Quote(100.0, 0.0, 100.0)},
    {"A": Quote(100.0, 100_000.0)},
])
def test_bad_market_data_fails_closed(quotes: dict[str, Quote]) -> None:
    assert decide(buy("A", 1), quotes=quotes) == {RiskReason.BAD_MARKET_DATA}


def test_missing_mark_on_another_held_name_fails_closed() -> None:
    assert decide(buy("A", 1), {"B": 1}, quotes={"A": Q["A"]}) == {RiskReason.BAD_MARKET_DATA}


def test_working_orders_count_toward_limits() -> None:
    assert decide(buy("A", 1), working=(buy("A", 500, "w"),)) == {RiskReason.MAX_NAME_NOTIONAL}


def test_all_reasons_reported_together() -> None:
    reasons = decide(sell("N", 1), quotes={"N": Quote(10.0, 0.5, 5.0)})
    assert reasons == {RiskReason.NO_LOCATE, RiskReason.PRICE_BAND, RiskReason.ORDER_SIZE_ADV}


def test_order_that_shrinks_a_drifted_breach_passes() -> None:
    assert decide(sell("A", 100), {"A": 900}) == set()


def test_reducing_order_may_exceed_adv_size() -> None:
    assert decide(sell("B", 5_000), {"B": 5_000}) == set()


def test_reducing_one_leg_may_raise_net_but_is_still_allowed() -> None:
    decision = RiskEngine(LIMITS).check(buy("B", 900), {"A": 700, "B": -900}, (), Q, 0)
    assert decision.approved and decision.reducing and decision.net_after > LIMITS.max_net


def test_every_decision_logged_with_reasons() -> None:
    seen: list[dict[str, object]] = []
    engine = RiskEngine(LIMITS, sink=seen.append)
    decide(buy("A", 1), engine=engine)
    decide(buy("B", 11), quotes={"B": Quote(50.0, 1_000.0, 50.0)}, engine=engine)
    assert [e["approved"] for e in engine.log] == [True, False]
    assert engine.log[1]["reasons"] == ["ORDER_SIZE_ADV"]
    assert seen == engine.log


def test_drawdown_kill_latches_until_manual_reset() -> None:
    engine = RiskEngine(LIMITS)
    assert not engine.on_equity(0, 100.0)
    assert not engine.on_equity(1, 91.0)
    assert engine.on_equity(2, 90.0)
    assert engine.kill_reason == "MAX_DRAWDOWN"
    assert engine.on_equity(3, 500.0), "recovery must not clear the latch"
    with pytest.raises(ValueError):
        engine.reset_kill_switch(4, "", "note")
    assert engine.killed
    engine.on_equity(4, 90.0)
    engine.reset_kill_switch(5, "owner", "reviewed")
    assert not engine.killed
    assert not engine.on_equity(6, 90.0), "the peak restarts at the reset"
    assert engine.on_equity(7, 80.0)
    events = [e["event"] for e in engine.log]
    assert events == ["kill_switch_tripped", "kill_switch_reset", "kill_switch_tripped"]


def test_non_finite_equity_trips() -> None:
    engine = RiskEngine(LIMITS)
    assert engine.on_equity(0, math.nan)
    assert engine.kill_reason == "NON_FINITE_EQUITY"


WIDE = RiskLimits(max_gross=1e9, max_net=1e9, max_name_notional=1e9, max_order_adv_frac=1.0,
                  locates=frozenset({"A", "B"}))


def drive(executor: PairExecutor, broker: PaperBroker, quotes: dict[str, Quote], start: int, bars: int) -> int:
    bar = start
    for bar in range(start, start + bars):
        executor.step(bar, quotes)
        broker.on_bar(bar + 1, quotes)
        if not executor.active:
            break
    return bar + 1


def leg_quotes(adv_a: float, adv_b: float) -> dict[str, Quote]:
    return {"A": Quote(100.0, adv_a, 100.0), "B": Quote(50.0, adv_b, 50.0)}


def test_both_legs_complete() -> None:
    broker = PaperBroker(PaperBrokerConfig(max_participation=1.0))
    executor = PairExecutor(broker, RiskEngine(WIDE), ("A", "B"))
    executor.enter(0, {"A": 100, "B": -200})
    drive(executor, broker, leg_quotes(1e6, 1e6), 0, 20)
    executor.step(99, leg_quotes(1e6, 1e6))
    assert executor.last_outcome is Outcome.FILLED
    assert broker.positions() == {"A": 100, "B": -200}


@pytest.mark.parametrize("bound", [0.1, 0.25, 0.5])
@pytest.mark.parametrize("seed", range(5))
def test_leg_imbalance_bound_with_a_slow_leg(bound: float, seed: int) -> None:
    cfg = PaperBrokerConfig(max_participation=0.01, liquidity_jitter=0.9, leg_latency_bars={"B": 2})
    broker = PaperBroker(cfg, seed=seed)
    executor = PairExecutor(broker, RiskEngine(WIDE), ("A", "B"), ExecutionConfig(bound, timeout_bars=500))
    target = {"A": 1_000, "B": -1_000}
    quotes = leg_quotes(1e6, 2_000.0)  # A could fill in one bar; B trades about 10 shares a bar
    executor.enter(0, target)
    for bar in range(400):
        executor.step(bar, quotes)
        broker.on_bar(bar + 1, quotes)
        held = broker.positions()
        gap = abs(held.get("A", 0) / 1_000 + held.get("B", 0) / 1_000)
        assert gap <= bound + 1.0 / 1_000 + 1e-12
    assert executor.max_entry_excess <= 1e-12


def test_slow_leg_catching_up_releases_the_fast_leg() -> None:
    broker = PaperBroker(PaperBrokerConfig(max_participation=0.01))
    executor = PairExecutor(broker, RiskEngine(WIDE), ("A", "B"), ExecutionConfig(0.1, timeout_bars=500))
    quotes = leg_quotes(1e6, 10_000.0)
    executor.enter(0, {"A": 500, "B": -500})
    drive(executor, broker, quotes, 0, 200)
    executor.step(500, quotes)
    assert executor.last_outcome is Outcome.FILLED


def test_stuck_leg_times_out_and_unwinds_flat() -> None:
    broker = PaperBroker(PaperBrokerConfig(max_participation=0.05))
    executor = PairExecutor(broker, RiskEngine(WIDE), ("A", "B"), ExecutionConfig(0.2, timeout_bars=3))
    quotes = leg_quotes(1e6, 1.0)  # B never gets a whole share
    executor.enter(0, {"A": 1_000, "B": -1_000})
    drive(executor, broker, quotes, 0, 30)
    assert broker.positions() == {}
    assert broker.open_orders() == []
    assert executor.last_outcome is Outcome.UNWOUND
    assert any(f.side is Side.SELL for f in broker.fills())


def test_unwind_imbalance_stays_inside_twice_the_bound() -> None:
    broker = PaperBroker(PaperBrokerConfig(max_participation=0.01, liquidity_jitter=0.5), seed=1)
    bound = 0.2
    executor = PairExecutor(broker, RiskEngine(WIDE), ("A", "B"), ExecutionConfig(bound, timeout_bars=6))
    quotes = leg_quotes(1e6, 3_000.0)
    executor.enter(0, {"A": 1_000, "B": -1_000})
    for bar in range(600):
        executor.step(bar, quotes)
        broker.on_bar(bar + 1, quotes)
        held = broker.positions()
        assert abs(held.get("A", 0) + held.get("B", 0)) / 1_000 <= 2 * bound + 2.0 / 1_000
    assert broker.positions() == {}


def test_rejected_leg_aborts_the_entry() -> None:
    broker = PaperBroker(PaperBrokerConfig(borrow_unavailable=frozenset({"B"})))
    executor = PairExecutor(broker, RiskEngine(WIDE), ("A", "B"), ExecutionConfig(0.25, timeout_bars=50))
    quotes = leg_quotes(1e6, 1e6)
    executor.enter(0, {"A": 100, "B": -100})
    drive(executor, broker, quotes, 0, 30)
    assert broker.positions() == {}
    assert executor.last_outcome is Outcome.UNWOUND


def test_risk_rejected_entry_never_reaches_the_broker() -> None:
    broker = PaperBroker()
    limits = replace(WIDE, locates=frozenset())
    executor = PairExecutor(broker, RiskEngine(limits), ("A", "B"))
    quotes = leg_quotes(1e6, 1e6)
    executor.enter(0, {"A": -100, "B": -100})
    executor.step(0, quotes)
    assert broker.open_orders() == [] and broker.fills() == []
    executor.step(1, quotes)
    assert executor.last_outcome is Outcome.UNWOUND


def test_flatten_cancels_and_exits_under_kill_switch() -> None:
    broker = PaperBroker(PaperBrokerConfig(max_participation=1.0))
    risk = RiskEngine(WIDE)
    executor = PairExecutor(broker, risk, ("A", "B"), ExecutionConfig(1.0))
    quotes = leg_quotes(1e6, 1e6)
    executor.enter(0, {"A": 100, "B": -100})
    drive(executor, broker, quotes, 0, 5)
    executor.step(50, quotes)
    risk.trip(50, "MANUAL")
    executor.flatten(50)
    drive(executor, broker, quotes, 50, 20)
    assert broker.positions() == {}
    assert executor.last_outcome is Outcome.FLAT


def worst_case(broker: PaperBroker, quotes: dict[str, Quote]) -> tuple[float, float, float]:
    """Exposure of held positions, recomputed here without the risk module."""
    values = [q * quotes[s].mid for s, q in broker.positions().items()]
    return sum(abs(v) for v in values), abs(sum(values)), max((abs(v) for v in values), default=0.0)


@pytest.mark.parametrize("seed", range(20))
def test_property_limits_never_breached_at_fixed_prices(seed: int) -> None:
    rng = np.random.default_rng(seed)
    names = ["A", "B", "N", "R"]
    quotes = {s: Quote(float(rng.uniform(10, 200)), 50_000.0, 0.0) for s in names}
    quotes = {s: replace(q, prev_close=q.mid) for s, q in quotes.items()}
    limits = RiskLimits(max_gross=300_000.0, max_net=120_000.0, max_name_notional=150_000.0,
                        max_order_adv_frac=0.02, restricted=frozenset({"R"}), locates=frozenset({"A", "B"}))
    cfg = PaperBrokerConfig(max_participation=0.01, liquidity_jitter=0.8, spread_jitter=0.5)
    broker = PaperBroker(cfg, seed=seed)
    risk = RiskEngine(limits)
    approved = 0
    for bar in range(300):
        for i in range(int(rng.integers(0, 4))):
            side = Side.BUY if rng.random() < 0.5 else Side.SELL
            order = Order(f"o{bar}-{i}", names[int(rng.integers(0, 4))], side, int(rng.integers(1, 1_500)))
            decision = risk.check(order, broker.positions(), broker.open_orders(), quotes, bar)
            # A reduction is always approved and can raise net; gross and name cannot rise.
            # This flow leaves those out so the net bound below stays a strict check.
            if decision.approved and decision.net_after <= limits.max_net:
                approved += 1
                broker.submit(order)
                assert order.qty <= limits.max_order_adv_frac * quotes[order.symbol].adv or decision.reducing
        if rng.random() < 0.2 and (working := broker.open_orders()):
            broker.cancel(working[int(rng.integers(0, len(working)))].client_id)
        broker.on_bar(bar, quotes)
        gross, net, name = worst_case(broker, quotes)
        assert gross <= limits.max_gross
        assert net <= limits.max_net
        assert name <= limits.max_name_notional
        held = broker.positions()
        assert held.get("N", 0) >= 0
        assert "R" not in held
    assert approved > 50


@pytest.mark.parametrize("seed", range(10))
def test_property_approvals_respect_limits_when_prices_move(seed: int) -> None:
    rng = np.random.default_rng(100 + seed)
    limits = RiskLimits(max_gross=200_000.0, max_net=80_000.0, max_name_notional=100_000.0,
                        max_order_adv_frac=0.02, locates=frozenset({"A", "B"}))
    broker = PaperBroker(PaperBrokerConfig(max_participation=0.02, liquidity_jitter=0.5), seed=seed)
    risk = RiskEngine(limits)
    mids = {"A": 100.0, "B": 50.0}
    for bar in range(300):
        previous = dict(mids)
        mids = {s: m * math.exp(0.02 * float(rng.standard_normal())) for s, m in mids.items()}
        quotes = {s: Quote(mids[s], 50_000.0, previous[s]) for s in mids}
        broker.on_bar(bar, quotes)
        side = Side.BUY if rng.random() < 0.5 else Side.SELL
        order = Order(f"o{bar}", "AB"[int(rng.integers(0, 2))], side, int(rng.integers(1, 1_000)))
        before = risk.check(replace(order, client_id="probe", qty=1), broker.positions(), broker.open_orders(), quotes, bar)
        decision = risk.check(order, broker.positions(), broker.open_orders(), quotes, bar)
        if decision.approved:
            broker.submit(order)
            # At or under each limit, or a drifted breach that this order did not raise.
            assert decision.gross_after <= max(limits.max_gross, before.gross_after)
            assert decision.reducing or decision.net_after <= max(limits.max_net, before.net_after)
            assert decision.name_after <= max(limits.max_name_notional, before.name_after)


def mean_reverting(n: int = 500, seed: int = 11) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = 100.0 + np.cumsum(rng.normal(0.0, 0.3, n))
    noise = np.zeros(n)
    for t in range(1, n):
        noise[t] = 0.9 * noise[t - 1] + rng.normal(0.0, 1.0)
    return 10.0 + 1.5 * x + noise, x


RULE = BacktestConfig(window=30, capital=200_000.0)
RUN = PaperRunConfig(adv_y=50_000.0, adv_x=50_000.0, seed=5)
BROKER = PaperBrokerConfig(initial_cash=200_000.0, max_participation=0.01, liquidity_jitter=0.3, spread_jitter=0.3,
                           commission_bps=0.5, borrow_bps_annual=50.0)
RUN_LIMITS = RiskLimits(max_gross=260_000.0, max_net=120_000.0, max_name_notional=180_000.0,
                        locates=frozenset({"stock_y", "stock_x"}))


def test_day_loop_books_balance_and_end_flat() -> None:
    y, x = mean_reverting()
    result = run_paper(y, x, 10.0, 1.5, RULE, RUN, BROKER, RUN_LIMITS)
    assert result.entries > 3 and result.fills > 10
    assert result.breaks == [] and result.kill_reason == ""
    assert result.final_positions == {}
    assert result.equity[-1] == pytest.approx(BROKER.initial_cash + result.attribution.total)
    assert result.max_entry_excess <= 1e-12


def test_day_loop_is_deterministic_by_seed(tmp_path: Path) -> None:
    y, x = mean_reverting()
    paths = [tmp_path / "a.jsonl", tmp_path / "b.jsonl", tmp_path / "c.jsonl"]
    a = run_paper(y, x, 10.0, 1.5, RULE, RUN, BROKER, RUN_LIMITS, audit_path=str(paths[0]))
    b = run_paper(y, x, 10.0, 1.5, RULE, RUN, BROKER, RUN_LIMITS, audit_path=str(paths[1]))
    c = run_paper(y, x, 10.0, 1.5, RULE, replace(RUN, seed=6), BROKER, RUN_LIMITS, audit_path=str(paths[2]))
    assert np.array_equal(a.equity, b.equity)
    assert paths[0].read_bytes() == paths[1].read_bytes()
    assert not np.array_equal(a.equity, c.equity)


def test_day_loop_drawdown_kill_flattens_and_stops_trading() -> None:
    y, x = mean_reverting()
    tight = replace(RUN_LIMITS, max_drawdown_frac=0.001)
    result = run_paper(y, x, 10.0, 1.5, RULE, RUN, BROKER, tight)
    assert result.kill_reason == "MAX_DRAWDOWN"
    assert result.final_positions == {}
    trip = next(r for r in result.audit.records if r["event"] == "kill_switch_tripped")
    late = [r for r in result.audit.records if r["event"] == "signal" and r["bar"] > trip["bar"]]
    assert all(r["action"] != "enter" for r in late)


def test_day_loop_without_locates_never_goes_short() -> None:
    y, x = mean_reverting()
    result = run_paper(y, x, 10.0, 1.5, RULE, RUN, BROKER, replace(RUN_LIMITS, locates=frozenset()))
    assert result.rejections.get("NO_LOCATE", 0) > 0
    assert result.final_positions == {}
    assert result.outcomes["FILLED"] == 0


def test_frictionless_paper_run_tracks_the_backtest(pair: dict[str, np.ndarray]) -> None:
    """Parity check on the rule: same trades one bar after each signal, near-equal P&L."""
    y, x = pair["stock_y"], pair["stock_x"]
    fit = ols(y[:900], x[:900], intercept=True)
    alpha, beta = float(fit.params[0]), float(fit.params[1])
    rule = BacktestConfig(pnl_stop_frac=0.02, max_hold_days=60, cooldown_days=20, cost_bps=0.0, slippage_bps=0.0)
    expected = backtest_pair(y[900:], x[900:], alpha, beta, rule)
    broker = PaperBrokerConfig(initial_cash=rule.capital, half_spread_bps=0.0, impact_bps=0.0, max_participation=1.0)
    limits = RiskLimits(1e12, 1e12, 1e12, max_order_adv_frac=1.0, price_band_frac=1.0, max_drawdown_frac=0.99,
                        locates=frozenset({"stock_y", "stock_x"}))
    run = PaperRunConfig(adv_y=1e9, adv_x=1e9, wind_down_bars=1)
    result = run_paper(y[900:], x[900:], alpha, beta, rule, run, broker, limits, ExecutionConfig(1.0, 8))
    entries = [r["bar"] + 1 for r in result.audit.records if r["event"] == "signal" and r["action"] == "enter"]
    assert entries == [t.entry_day for t in expected.trades]
    assert result.equity[-1] == pytest.approx(expected.equity[-1], abs=0.005 * rule.capital)


@pytest.mark.parametrize("quotes", [
    {"A": Quote(200.0, 1.0, 100.0)},  # outside the band, tiny ADV
    {"A": Quote(math.nan, 100_000.0)},
    {},
])
def test_reducing_order_passes_every_check(quotes: dict[str, Quote]) -> None:
    limits = replace(LIMITS, restricted=frozenset({"A"}), locates=frozenset(), max_gross=1.0, max_net=1.0,
                     max_name_notional=1.0)
    engine = RiskEngine(limits)
    engine.trip(0, "MANUAL")
    assert decide(sell("A", 900), {"A": 900}, quotes=quotes, engine=engine) == set()
    assert decide(buy("A", 900), {"A": -900}, quotes=quotes, engine=engine) == set()
    assert engine.log[-1]["reducing"] is True
    assert RiskReason.KILL_SWITCH in decide(buy("A", 1), {"A": 900}, quotes=quotes, engine=engine)


def held_pair() -> tuple[PaperBroker, RiskEngine, PairExecutor, dict[str, Quote]]:
    broker = PaperBroker(PaperBrokerConfig(max_participation=1.0))
    risk = RiskEngine(WIDE)
    executor = PairExecutor(broker, risk, ("A", "B"), ExecutionConfig(1.0))
    quotes = leg_quotes(1e6, 1e6)
    executor.enter(0, {"A": 100, "B": -100})
    drive(executor, broker, quotes, 0, 5)
    executor.step(50, quotes)
    assert broker.positions() == {"A": 100, "B": -100}
    return broker, risk, executor, quotes


def test_exit_of_a_name_restricted_after_entry() -> None:
    broker, risk, executor, quotes = held_pair()
    risk.limits = replace(WIDE, restricted=frozenset({"A", "B"}), locates=frozenset())
    executor.flatten(50)
    drive(executor, broker, quotes, 50, 20)
    assert broker.positions() == {}


def test_exit_while_kill_switch_is_latched_and_limits_shrunk() -> None:
    broker, risk, executor, quotes = held_pair()
    risk.limits = replace(WIDE, max_gross=1.0, max_net=1.0, max_name_notional=1.0)
    risk.trip(50, "MANUAL")
    executor.flatten(50)
    drive(executor, broker, quotes, 50, 20)
    assert broker.positions() == {} and risk.killed


def test_stop_during_incomplete_entry_cancels_and_unwinds() -> None:
    n = 120
    x = np.full(n, 100.0)
    y = 100.0 + 0.05 * np.sin(np.arange(n))
    y[72:] += 3.0   # entry signal at bar 73, |z| between entry_z and stop_z
    y[75:] += 30.0  # the spread keeps running: stop signal while the entry is part-filled
    rule = BacktestConfig(window=30, capital=100_000.0, cooldown_days=500)
    cfg = PaperBrokerConfig(initial_cash=100_000.0, max_participation=0.01, half_spread_bps=0.0, impact_bps=0.0)
    limits = RiskLimits(1e9, 1e9, 1e9, max_order_adv_frac=1.0, price_band_frac=1.0, max_drawdown_frac=0.9,
                        locates=frozenset({"stock_y", "stock_x"}))
    run = PaperRunConfig(adv_y=2_000.0, adv_x=2_000.0, wind_down_bars=1)  # 20 shares a bar against about 500
    result = run_paper(y, x, 0.0, 1.0, rule, run, cfg, limits, ExecutionConfig(0.5, timeout_bars=100))
    signals = [r for r in result.audit.records if r["event"] == "signal"]
    assert [s["action"] for s in signals] == ["enter", "exit"] and signals[1]["stop"] is True
    assert signals[1]["bar"] == 75
    assert result.outcomes["FILLED"] == 0 and result.outcomes["FLAT"] == 1
    assert result.final_positions == {}
    built = [r for r in result.audit.records if r["event"] == "fill" and r["bar"] <= 75]
    assert 0 < sum(r["qty"] for r in built) < 500, "the entry was only part-filled"
