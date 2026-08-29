/// \file test_backtest.cpp
/// \brief Backtest tests: accounting identity, dollar neutrality at entry,
/// stop-losses in the break regime, no-trade guard, walk-forward split,
/// and configuration/price validation.
#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "statarb/backtest.hpp"
#include "statarb/data_io.hpp"
#include "statarb/engle_granger.hpp"

namespace {

struct Fixture {
    std::vector<double> y, x;
    double alpha = 0.0, beta = 0.0;
    statarb::BacktestConfig cfg{60, 2.0, 0.5, 4.0, 1.0, 0.5, 1.0e6};
    statarb::BacktestResult bt;

    Fixture() {
        const auto csv = statarb::load_csv("../data/pair_equity.csv");
        y = csv.at("stock_y");
        x = csv.at("stock_x");
        const std::vector<double> yi(y.begin(), y.begin() + 900);
        const std::vector<double> xi(x.begin(), x.begin() + 900);
        const statarb::EGResult eg = statarb::engle_granger(yi, xi, 1);
        alpha = eg.alpha;
        beta = eg.beta;
        bt = statarb::backtest_pair(y, x, alpha, beta, cfg);
    }
};

const Fixture& fx() {
    static const Fixture f;
    return f;
}

TEST(Backtest, AccountingIdentityHolds) {
    const statarb::BacktestResult& bt = fx().bt;
    long double pnl_sum = 0.0L, cost_sum = 0.0L;
    for (double v : bt.daily_pnl) pnl_sum += v;
    for (double v : bt.daily_cost) cost_sum += v;
    const double lhs = bt.equity.back() - fx().cfg.capital;
    const double rhs = static_cast<double>(pnl_sum - cost_sum);
    // Tested at 1e-10 relative to capital (pinned).
    EXPECT_NEAR(lhs, rhs, 1e-10 * fx().cfg.capital);
}

TEST(Backtest, DollarNeutralEqualDollarLegsAtEntry) {
    const statarb::BacktestResult& bt = fx().bt;
    ASSERT_FALSE(bt.trades.empty());
    const double bps = (fx().cfg.cost_bps + fx().cfg.slippage_bps) / 1.0e4;
    // Equal-dollar legs mean entry traded notional is EXACTLY capital
    // (capital/2 per leg), so the entry-day cost equals capital * bps.
    for (const statarb::Trade& t : bt.trades) {
        if (t.entry_day == t.exit_day) continue;  // final-day entry+force-close
        const double entry_cost =
            bt.daily_cost[static_cast<std::size_t>(t.entry_day)];
        EXPECT_NEAR(entry_cost, fx().cfg.capital * bps,
                    1e-9 * fx().cfg.capital * bps)
            << "entry day " << t.entry_day;
    }
}

TEST(Backtest, StopLossTriggersInBreakRegime) {
    // The relationship breaks at day 1050; the stop rule must fire.
    EXPECT_GT(fx().bt.n_stops(), 0);
    // Every stop belongs to a closed trade with entry before exit.
    for (const statarb::Trade& t : fx().bt.trades) {
        EXPECT_LE(t.entry_day, t.exit_day);
        EXPECT_TRUE(t.direction == 1 || t.direction == -1);
    }
    EXPECT_EQ(static_cast<double>(fx().bt.trades.size()),
              fx().bt.metrics.n_trades);
}

TEST(Backtest, NoTradeGuardAllMetricsZeroFlatEquity) {
    // y == x with (alpha, beta) = (0, 1): spread identically zero, the
    // zero-variance guard yields NaN z everywhere -> no trades at all.
    std::vector<double> p(200);
    for (int i = 0; i < 200; ++i) {
        p[static_cast<std::size_t>(i)] = 50.0 + std::sin(0.1 * i);
    }
    const statarb::BacktestConfig cfg(20, 2.0, 0.5, 4.0, 1.0, 0.5, 1.0e6);
    const statarb::BacktestResult bt = statarb::backtest_pair(p, p, 0.0, 1.0, cfg);
    EXPECT_TRUE(bt.trades.empty());
    const statarb::Metrics& m = bt.metrics;
    EXPECT_EQ(m.total_return, 0.0);
    EXPECT_EQ(m.ann_return, 0.0);
    EXPECT_EQ(m.sharpe, 0.0);
    EXPECT_EQ(m.max_drawdown, 0.0);
    EXPECT_EQ(m.n_trades, 0.0);
    EXPECT_EQ(m.win_rate, 0.0);
    EXPECT_EQ(m.avg_holding_days, 0.0);
    EXPECT_EQ(m.turnover, 0.0);
    for (double e : bt.equity) EXPECT_EQ(e, 1.0e6);  // flat at capital
}

TEST(Backtest, TurnoverCountsEntryAndExitNotional) {
    const statarb::BacktestResult& bt = fx().bt;
    // Each round trip trades ~capital at entry and ~capital +/- pnl drift
    // at exit, so turnover is close to 2 * n_trades.
    EXPECT_GT(bt.metrics.turnover, 1.5 * bt.metrics.n_trades);
    EXPECT_LT(bt.metrics.turnover, 2.5 * bt.metrics.n_trades);
}

TEST(Backtest, ConfigValidation) {
    using statarb::BacktestConfig;
    EXPECT_THROW(BacktestConfig(1, 2.0, 0.5, 4.0, 1.0, 0.5, 1e6),
                 std::invalid_argument);  // window < 2
    EXPECT_THROW(BacktestConfig(60, 2.0, 2.0, 4.0, 1.0, 0.5, 1e6),
                 std::invalid_argument);  // exit >= entry
    EXPECT_THROW(BacktestConfig(60, 2.0, 0.5, 2.0, 1.0, 0.5, 1e6),
                 std::invalid_argument);  // stop <= entry
    EXPECT_THROW(BacktestConfig(60, 2.0, -0.1, 4.0, 1.0, 0.5, 1e6),
                 std::invalid_argument);  // exit < 0
    EXPECT_THROW(BacktestConfig(60, 2.0, 0.5, 4.0, -1.0, 0.5, 1e6),
                 std::invalid_argument);  // negative cost
    EXPECT_THROW(BacktestConfig(60, 2.0, 0.5, 4.0, 1.0, 0.5, 0.0),
                 std::invalid_argument);  // capital <= 0
    EXPECT_THROW(BacktestConfig(60, 2.0, 0.5, 4.0, 1.0, 0.5,
                                std::numeric_limits<double>::infinity()),
                 std::invalid_argument);  // capital not finite
    EXPECT_NO_THROW(BacktestConfig(2, 1.0, 0.0, 2.0, 0.0, 0.0, 1.0));
}

TEST(Backtest, PriceValidation) {
    const statarb::BacktestConfig cfg(5, 2.0, 0.5, 4.0, 1.0, 0.5, 1e6);
    std::vector<double> good(50);
    for (int i = 0; i < 50; ++i) {
        good[static_cast<std::size_t>(i)] = 10.0 + std::sin(0.3 * i);
    }
    std::vector<double> shorter(good.begin(), good.end() - 1);
    EXPECT_THROW(statarb::backtest_pair(good, shorter, 0.0, 1.0, cfg),
                 std::invalid_argument);
    std::vector<double> neg = good;
    neg[10] = -1.0;
    EXPECT_THROW(statarb::backtest_pair(good, neg, 0.0, 1.0, cfg),
                 std::invalid_argument);
    std::vector<double> nan = good;
    nan[10] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(statarb::backtest_pair(nan, good, 0.0, 1.0, cfg),
                 std::invalid_argument);
    // Window larger than the series (via rolling_zscore).
    const statarb::BacktestConfig wide(60, 2.0, 0.5, 4.0, 1.0, 0.5, 1e6);
    EXPECT_THROW(statarb::backtest_pair(good, good, 0.0, 1.0, wide),
                 std::invalid_argument);
}

TEST(WalkForward, SplitAndSegmentsMatchSpec) {
    const statarb::WalkForwardResult wf =
        statarb::walk_forward(fx().y, fx().x, fx().cfg, 0.6);
    EXPECT_EQ(wf.split, 900u);  // floor(1500 * 0.6)
    EXPECT_EQ(wf.insample.equity.size(), 900u);
    EXPECT_EQ(wf.oos.equity.size(), 600u);
    // Same hedge in both segments; OOS z-score warms up inside OOS only.
    for (std::size_t t = 0; t < 59; ++t) {
        EXPECT_TRUE(std::isnan(wf.oos.zscore[t])) << "t=" << t;
    }
    EXPECT_TRUE(std::isfinite(wf.oos.zscore[59]));
    // Honest degradation on this dataset: OOS return below in-sample.
    EXPECT_LT(wf.oos.metrics.total_return, wf.insample.metrics.total_return);
}

TEST(WalkForward, Validation) {
    EXPECT_THROW(statarb::walk_forward(fx().y, fx().x, fx().cfg, 0.0),
                 std::invalid_argument);
    EXPECT_THROW(statarb::walk_forward(fx().y, fx().x, fx().cfg, 1.0),
                 std::invalid_argument);
    // Segment shorter than the window.
    std::vector<double> y(fx().y.begin(), fx().y.begin() + 100);
    std::vector<double> x(fx().x.begin(), fx().x.begin() + 100);
    EXPECT_THROW(statarb::walk_forward(y, x, fx().cfg, 0.5),
                 std::invalid_argument);
}

}  // namespace
