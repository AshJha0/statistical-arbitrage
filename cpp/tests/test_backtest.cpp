/// \file test_backtest.cpp
/// \brief Backtest tests: accounting identity, dollar neutrality at entry,
/// stop-losses in the break regime, no-trade guard, walk-forward split,
/// and configuration/price validation.
#include <gtest/gtest.h>

#include <algorithm>
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
    // Every stop belongs to a closed trade with entry strictly before exit.
    for (const statarb::Trade& t : fx().bt.trades) {
        EXPECT_LT(t.entry_day, t.exit_day);
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

TEST(Backtest, ConfigRejectsNonFinite) {
    // Proposed test 5 (MAJ-2): NaN/inf thresholds and costs are errors
    // (NaN compares false, so ordering/sign checks alone let it through).
    using statarb::BacktestConfig;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    EXPECT_THROW(BacktestConfig(60, 2.0, 0.5, 4.0, nan, 0.5, 1e6), std::invalid_argument);
    EXPECT_THROW(BacktestConfig(60, 2.0, 0.5, 4.0, 1.0, inf, 1e6), std::invalid_argument);
    EXPECT_THROW(BacktestConfig(60, nan, 0.5, 4.0, 1.0, 0.5, 1e6), std::invalid_argument);
    EXPECT_THROW(BacktestConfig(60, 2.0, 0.5, nan, 1.0, 0.5, 1e6), std::invalid_argument);
    EXPECT_THROW(BacktestConfig(60, 2.0, -inf, 4.0, 1.0, 0.5, 1e6), std::invalid_argument);
    EXPECT_THROW(BacktestConfig(60, 2.0, 0.5, inf, 1.0, 0.5, 1e6), std::invalid_argument);
    EXPECT_THROW(BacktestConfig(60, 2.0, 0.5, 4.0, 1.0, 0.5, nan), std::invalid_argument);
}

TEST(Backtest, ConfigRevalidatedAtRun) {
    // Proposed test 6 (MAJ-4): mutating the public members after
    // construction cannot bypass validation.
    statarb::BacktestConfig cfg(20, 2.0, 0.5, 4.0, 1.0, 0.5, 1e6);
    std::vector<double> p(100);
    for (int i = 0; i < 100; ++i) p[static_cast<std::size_t>(i)] = 50.0 + std::sin(0.1 * i);
    std::vector<double> q(100);
    for (int i = 0; i < 100; ++i) q[static_cast<std::size_t>(i)] = 30.0 + std::cos(0.3 * i);
    EXPECT_NO_THROW(statarb::backtest_pair(p, q, 0.0, 1.0, cfg));
    cfg.exit_z = 5.0;  // breaks exit_z < entry_z
    EXPECT_THROW(statarb::backtest_pair(p, q, 0.0, 1.0, cfg), std::invalid_argument);
    EXPECT_THROW(statarb::walk_forward(p, q, cfg, 0.5), std::invalid_argument);
    cfg.exit_z = 0.5;
    cfg.capital = 0.0;  // would divide by zero in the sizing
    EXPECT_THROW(statarb::backtest_pair(p, q, 0.0, 1.0, cfg), std::invalid_argument);
    EXPECT_THROW(cfg.validate(), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Synthetic-spread fixtures.  Every expected value is derived independently
// of the backtest: z-scores are recomputed by `zpath` (plain rolling mean /
// ddof-1 std) and closed forms are used where a window holds a single spike
// on a zero background: z_spike = (W-1)/sqrt(W), z_next = -1/sqrt(W).
// ---------------------------------------------------------------------------

std::vector<double> zpath(const std::vector<double>& s, std::size_t w) {
    std::vector<double> z(s.size(), std::numeric_limits<double>::quiet_NaN());
    for (std::size_t t = w - 1; t < s.size(); ++t) {
        double m = 0.0;
        for (std::size_t i = t + 1 - w; i <= t; ++i) m += s[i];
        m /= static_cast<double>(w);
        double ss = 0.0;
        for (std::size_t i = t + 1 - w; i <= t; ++i) ss += (s[i] - m) * (s[i] - m);
        const double sd = std::sqrt(ss / static_cast<double>(w - 1));
        if (sd > 1e-12) z[t] = (s[t] - m) / sd;
    }
    return z;
}

double max_abs_finite(const std::vector<double>& z, std::size_t lo, std::size_t hi) {
    double m = 0.0;
    for (std::size_t t = lo; t < hi; ++t) {
        if (std::isfinite(z[t])) m = std::max(m, std::fabs(z[t]));
    }
    return m;
}

TEST(Backtest, EndOfDataPendingEntryDiscarded) {
    // Proposed test 1 (CRIT-1): |z_{T-2}| in [entry, stop) while flat
    // would execute at T-1 and be force-closed the same close.  The spec
    // discards it: no trades, no turnover, equity flat at capital.
    const std::size_t T = 80;
    std::vector<double> x(T), s(T), y(T);
    for (std::size_t t = 0; t < T; ++t) {
        x[t] = 200.0 + 0.5 * std::sin(static_cast<double>(t));
        s[t] = std::sin(0.7 * static_cast<double>(t));
    }
    s[T - 2] = 6.0;
    for (std::size_t t = 0; t < T; ++t) y[t] = 10.0 + x[t] + s[t];
    const statarb::BacktestConfig cfg(20, 2.0, 0.5, 8.0, 1.0, 0.5, 1e6);
    const std::vector<double> z = zpath(s, 20);
    ASSERT_LT(max_abs_finite(z, 0, T - 2), 2.0);
    ASSERT_GE(z[T - 2], 2.0);
    ASSERT_LT(z[T - 2], 8.0);
    const statarb::BacktestResult bt = statarb::backtest_pair(y, x, 10.0, 1.0, cfg);
    EXPECT_TRUE(bt.trades.empty());
    EXPECT_EQ(bt.metrics.n_trades, 0.0);
    EXPECT_EQ(bt.metrics.turnover, 0.0);
    for (double e : bt.equity) EXPECT_EQ(e, 1e6);
    for (double c : bt.daily_cost) EXPECT_EQ(c, 0.0);
}

TEST(Backtest, ZeroDayRoundTripNeverReported) {
    // Proposed test 2: over every truncation of the bundled pair no trade
    // may ever have exit_day == entry_day (would have caught CRIT-1).
    for (std::size_t T = 70; T <= fx().y.size(); ++T) {
        const std::vector<double> y(fx().y.begin(), fx().y.begin() + static_cast<long>(T));
        const std::vector<double> x(fx().x.begin(), fx().x.begin() + static_cast<long>(T));
        const statarb::BacktestResult bt =
            statarb::backtest_pair(y, x, fx().alpha, fx().beta, fx().cfg);
        for (const statarb::Trade& t : bt.trades) {
            ASSERT_LT(t.entry_day, t.exit_day) << "T=" << T;
            ASSERT_LE(t.exit_day, static_cast<int>(T) - 1);
        }
    }
}

TEST(Backtest, StopBandWhileFlatNoEntry) {
    // Proposed test 3: a single spike on a zero background gives
    // z = (W-1)/sqrt(W) = 6.166 >= stop_z while flat (no entry, pinned)
    // and z = -1/sqrt(W) the next day: no trades, flat equity.
    const std::size_t T = 120, W = 40, t0 = 60;
    std::vector<double> s(T, 0.0), x(T, 50.0), y(T);
    s[t0] = 7.0;
    for (std::size_t t = 0; t < T; ++t) y[t] = 100.0 + s[t];  // alpha 50, beta 1
    const statarb::BacktestConfig cfg(static_cast<int>(W), 2.0, 0.5, 4.0, 1.0, 0.5, 1e6);
    const std::vector<double> z = zpath(s, W);
    EXPECT_NEAR(z[t0], (W - 1) / std::sqrt(static_cast<double>(W)), 1e-12);
    EXPECT_NEAR(z[t0 + 1], -1.0 / std::sqrt(static_cast<double>(W)), 1e-12);
    ASSERT_GE(z[t0], cfg.stop_z);
    for (std::size_t t = 0; t < t0; ++t) ASSERT_TRUE(std::isnan(z[t]));
    const statarb::BacktestResult bt = statarb::backtest_pair(y, x, 50.0, 1.0, cfg);
    EXPECT_TRUE(bt.trades.empty());
    for (double e : bt.equity) EXPECT_EQ(e, 1e6);
}

TEST(Backtest, StopExitIsFlaggedAndExecutesNextDay) {
    // Proposed test 12: entry band at t0, in-position plateau, |z| >= stop
    // at t1 = t0+3 -> one trade, stopped, exit_day == t1+1, exit-day cost
    // equals bps * exit notional recomputed from the pinned legs.
    const std::size_t T = 160, W = 40, t0 = 70;
    std::vector<double> s(T), x(T), y(T);
    for (std::size_t t = 0; t < T; ++t) {
        s[t] = 0.2 * std::sin(0.7 * static_cast<double>(t));
        x[t] = 200.0 + 0.5 * std::sin(static_cast<double>(t));
    }
    s[t0] = s[t0 + 1] = s[t0 + 2] = 0.5;
    s[t0 + 3] = 5.0;
    for (std::size_t t = 0; t < T; ++t) y[t] = 10.0 + x[t] + s[t];
    const statarb::BacktestConfig cfg(static_cast<int>(W), 2.0, 0.5, 4.0, 1.0, 0.5, 1e6);
    const std::vector<double> z = zpath(s, W);
    ASSERT_LT(max_abs_finite(z, 0, t0), cfg.entry_z);
    ASSERT_GE(z[t0], cfg.entry_z);
    ASSERT_LT(z[t0], cfg.stop_z);
    for (std::size_t t = t0 + 1; t < t0 + 3; ++t) {
        ASSERT_GT(std::fabs(z[t]), cfg.exit_z);
        ASSERT_LT(std::fabs(z[t]), cfg.stop_z);
    }
    ASSERT_GE(z[t0 + 3], cfg.stop_z);
    ASSERT_LT(max_abs_finite(z, t0 + 4, T), cfg.entry_z);
    const statarb::BacktestResult bt = statarb::backtest_pair(y, x, 10.0, 1.0, cfg);
    ASSERT_EQ(bt.trades.size(), 1u);
    const statarb::Trade& tr = bt.trades[0];
    EXPECT_EQ(tr.entry_day, static_cast<int>(t0 + 1));
    EXPECT_EQ(tr.exit_day, static_cast<int>(t0 + 4));
    EXPECT_EQ(tr.direction, -1);
    EXPECT_TRUE(tr.stopped);
    EXPECT_EQ(bt.n_stops(), 1);
    const double bps = 1.5e-4, cap = 1e6;
    const double qy = -(cap / 2.0) / y[t0 + 1];
    const double qx = (cap / 2.0) / x[t0 + 1];
    EXPECT_NEAR(bt.daily_cost[t0 + 1], cap * bps, 1e-9);
    const double exit_notional = std::fabs(qy) * y[t0 + 4] + std::fabs(qx) * x[t0 + 4];
    EXPECT_NEAR(bt.daily_cost[t0 + 4], exit_notional * bps, 1e-9);
    EXPECT_GT(bt.daily_cost[t0 + 4], 0.0);
    EXPECT_EQ(bt.metrics.avg_holding_days, 3.0);
}

TEST(Backtest, ReentryAfterExitSameDirection) {
    // Proposed test 13: enter-short, exit, enter-short again with no
    // cool-down: two trades with entry_day_2 == exit_day_1 + 1.
    const std::size_t T = 160, W = 40, t0 = 70;
    std::vector<double> s(T), x(T), y(T);
    for (std::size_t t = 0; t < T; ++t) {
        s[t] = 0.2 * std::sin(0.7 * static_cast<double>(t));
        x[t] = 200.0 + 0.5 * std::sin(static_cast<double>(t));
    }
    s[t0] = 0.5; s[t0 + 1] = 0.0; s[t0 + 2] = 0.5; s[t0 + 3] = 0.0;
    for (std::size_t t = 0; t < T; ++t) y[t] = 10.0 + x[t] + s[t];
    const statarb::BacktestConfig cfg(static_cast<int>(W), 2.0, 0.5, 4.0, 1.0, 0.5, 1e6);
    const std::vector<double> z = zpath(s, W);
    ASSERT_GE(z[t0], cfg.entry_z);
    ASSERT_LT(z[t0], cfg.stop_z);
    ASSERT_LE(std::fabs(z[t0 + 1]), cfg.exit_z);
    ASSERT_GE(z[t0 + 2], cfg.entry_z);
    ASSERT_LT(z[t0 + 2], cfg.stop_z);
    ASSERT_LE(std::fabs(z[t0 + 3]), cfg.exit_z);
    ASSERT_LT(max_abs_finite(z, 0, t0), cfg.entry_z);
    ASSERT_LT(max_abs_finite(z, t0 + 4, T), cfg.entry_z);
    const statarb::BacktestResult bt = statarb::backtest_pair(y, x, 10.0, 1.0, cfg);
    ASSERT_EQ(bt.trades.size(), 2u);
    EXPECT_EQ(bt.trades[0].entry_day, static_cast<int>(t0 + 1));
    EXPECT_EQ(bt.trades[0].exit_day, static_cast<int>(t0 + 2));
    EXPECT_EQ(bt.trades[0].direction, -1);
    EXPECT_FALSE(bt.trades[0].stopped);
    EXPECT_EQ(bt.trades[1].entry_day, static_cast<int>(t0 + 3));
    EXPECT_EQ(bt.trades[1].exit_day, static_cast<int>(t0 + 4));
    EXPECT_EQ(bt.trades[1].direction, -1);
    EXPECT_FALSE(bt.trades[1].stopped);
    EXPECT_EQ(bt.trades[1].entry_day, bt.trades[0].exit_day + 1);
}

TEST(Backtest, EquityNonPositiveMetricsDefined) {
    // Proposed test 14 (MIN-11): a 4x jump on the short leg loses 1.5x
    // capital: total_return = -1.5 exactly, ann_return pinned to -1.0,
    // max_drawdown = 1.5, finite Sharpe, no NaN anywhere.
    const std::size_t T = 45, W = 40, t0 = 40;
    std::vector<double> s(T), x(T, 50.0), y(T);
    for (std::size_t t = 0; t < T; ++t) s[t] = 0.2 * std::sin(0.7 * static_cast<double>(t));
    for (std::size_t t = t0; t < T; ++t) s[t] = -0.5;      // long-spread entry band
    for (std::size_t t = t0 + 2; t < T; ++t) x[t] = 200.0;  // short leg quadruples
    for (std::size_t t = 0; t < T; ++t) y[t] = 100.0 + s[t];  // alpha 50, beta 1
    const statarb::BacktestConfig cfg(static_cast<int>(W), 2.0, 0.5, 4.0, 0.0, 0.0, 1e6);
    const statarb::BacktestResult bt = statarb::backtest_pair(y, x, 50.0, 1.0, cfg);
    ASSERT_EQ(bt.trades.size(), 1u);
    EXPECT_EQ(bt.trades[0].entry_day, static_cast<int>(t0 + 1));
    EXPECT_EQ(bt.trades[0].exit_day, static_cast<int>(t0 + 3));
    EXPECT_EQ(bt.trades[0].direction, 1);
    EXPECT_TRUE(bt.trades[0].stopped);
    EXPECT_NEAR(bt.trades[0].pnl, -1.5e6, 1e-6);  // q_x = -1e4 shares * 150
    EXPECT_NEAR(bt.equity.back(), -5.0e5, 1e-6);
    const statarb::Metrics& m = bt.metrics;
    EXPECT_NEAR(m.total_return, -1.5, 1e-12);
    EXPECT_EQ(m.ann_return, -1.0);
    EXPECT_NEAR(m.max_drawdown, 1.5, 1e-12);
    const double ret = -1.5, n = static_cast<double>(T);
    const double mean = ret / n;
    const double var = ((n - 1.0) * mean * mean + (ret - mean) * (ret - mean)) / (n - 1.0);
    EXPECT_NEAR(m.sharpe, mean / std::sqrt(var) * std::sqrt(252.0), 1e-10);
    EXPECT_NEAR(m.turnover, 3.5, 1e-12);  // 1e6 in, 2.5e6 out
    EXPECT_TRUE(std::isfinite(m.win_rate) && std::isfinite(m.avg_holding_days));
}

TEST(Backtest, ComputeMetricsValidation) {
    EXPECT_THROW(statarb::compute_metrics({}, 1e6, {}), std::invalid_argument);
    EXPECT_THROW(statarb::compute_metrics({1e6, std::numeric_limits<double>::quiet_NaN()},
                                          1e6, {}),
                 std::invalid_argument);
    EXPECT_THROW(statarb::compute_metrics({1e6}, 0.0, {}), std::invalid_argument);
    const statarb::Metrics m = statarb::compute_metrics({1e6}, 1e6, {});
    EXPECT_EQ(m.sharpe, 0.0);
    EXPECT_EQ(m.total_return, 0.0);
    EXPECT_EQ(m.max_drawdown, 0.0);
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

TEST(WalkForward, LengthMismatchIsTypedError) {
    // Proposed test 7 (MAJ-5).
    const std::vector<double> x100(fx().x.begin(), fx().x.begin() + 100);
    EXPECT_THROW(statarb::walk_forward(fx().y, x100, fx().cfg, 0.6),
                 std::invalid_argument);
}

TEST(WalkForward, SplitExact) {
    // Proposed test 8 (MIN-10): floor(T * frac) is the intended integer
    // (700 * 0.7 == 489.99999999999994 in binary floating point).
    const statarb::BacktestConfig cfg(5, 50.0, 0.5, 60.0, 1.0, 0.5, 1e6);
    const struct { std::size_t T; double frac; std::size_t split; } cases[] = {
        {700, 0.7, 490}, {100, 0.29, 29}, {1500, 0.6, 900}};
    for (const auto& c : cases) {
        std::vector<double> y(c.T), x(c.T);
        for (std::size_t t = 0; t < c.T; ++t) {
            y[t] = 100.0 + std::sin(0.3 * static_cast<double>(t));
            x[t] = 50.0 + std::cos(0.2 * static_cast<double>(t));
        }
        EXPECT_EQ(statarb::walk_forward(y, x, cfg, c.frac).split, c.split) << c.T;
    }
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
