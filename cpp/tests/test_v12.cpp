/// \file test_v12.cpp
/// \brief v1.2 contract behaviour, mirroring python/tests/test_v12.py:
/// sizing, risk limits, dead configs, pinned z sums, rolling walk-forward,
/// streaming API, lag selection, two-way Engle-Granger and the strict loaders.
#include <gtest/gtest.h>

#include <unistd.h>

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "statarb/adf.hpp"
#include "statarb/backtest.hpp"
#include "statarb/data_io.hpp"
#include "statarb/engle_granger.hpp"
#include "statarb/kalman.hpp"
#include "statarb/ols.hpp"
#include "statarb/signals.hpp"

namespace {

using statarb::BacktestConfig;
using Vec = std::vector<double>;

constexpr double kCap = 1.0e6;
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();

struct Px {
    Vec y, x;
    double alpha = 50.0, beta = 1.0;
};

/// Prices whose spread at alpha = 50, beta = 1 is exactly s.
Px scripted(const Vec& s, double x = 50.0) {
    Px p;
    p.x.assign(s.size(), x);
    p.y.resize(s.size());
    for (std::size_t i = 0; i < s.size(); ++i) p.y[i] = 50.0 + x + s[i];
    return p;
}

Vec sine(std::size_t n) {
    Vec s(n);
    for (std::size_t i = 0; i < n; ++i) s[i] = 0.2 * std::sin(0.7 * static_cast<double>(i));
    return s;
}

BacktestConfig cfg_w(int window) {
    BacktestConfig c;
    c.window = window;
    return c;
}

Px noise(std::size_t n, unsigned seed = 3) {
    std::mt19937_64 gen(seed);
    std::normal_distribution<double> n1(0.0, 1.0);
    Px p;
    p.x.resize(n);
    p.y.resize(n);
    double lx = 0.0, s = 0.0;
    for (std::size_t t = 0; t < n; ++t) {
        lx += 0.01 * n1(gen);
        p.x[t] = 100.0 * std::exp(lx);
        s = t == 0 ? 0.0 : 0.9 * s + n1(gen);
        p.y[t] = 20.0 + 1.5 * p.x[t] + s + 40.0;
    }
    p.alpha = 60.0;
    p.beta = 1.5;
    return p;
}

struct Bundled {
    std::map<std::string, double> cfg;
    Vec y, x, walk_a, walk_b;
    double alpha = 0.0, beta = 0.0;
    Bundled() {
        cfg = statarb::load_config("../data/config.json");
        const auto pair = statarb::load_csv("../data/pair_equity.csv");
        y = pair.at("stock_y");
        x = pair.at("stock_x");
        const auto w = statarb::load_csv("../data/random_walks.csv");
        walk_a = w.at("walk_a");
        walk_b = w.at("walk_b");
        const Vec yi(y.begin(), y.begin() + 900), xi(x.begin(), x.begin() + 900);
        const auto eg = statarb::engle_granger(yi, xi, static_cast<int>(cfg.at("adf_maxlag")));
        alpha = eg.alpha;
        beta = eg.beta;
    }
};
const Bundled& bundled() {
    static const Bundled b;
    return b;
}

template <class F>
void expect_invalid(F&& f, const std::string& needle = "") {
    try {
        f();
        FAIL() << "expected std::invalid_argument";
    } catch (const std::invalid_argument& e) {
        if (!needle.empty()) {
            EXPECT_NE(std::string(e.what()).find(needle), std::string::npos) << e.what();
        }
    }
}

double sum(const Vec& v) {
    double s = 0.0;
    for (double e : v) s += e;
    return s;
}

// ---- items 1 and 2: beta and sizing ---------------------------------------

TEST(V12Sizing, RejectsNonPositiveBeta) {
    const Px p = noise(100);
    for (double beta : {0.0, -1.0, kNaN}) {
        expect_invalid([&] { statarb::backtest_pair(p.y, p.x, p.alpha, beta, BacktestConfig()); },
                       "beta");
    }
}

TEST(V12Sizing, DefaultAndValidation) {
    EXPECT_EQ(BacktestConfig().sizing, "beta_hedged");
    expect_invalid([] { BacktestConfig(60, 2.0, 0.5, 4.0, 1.0, 0.5, 1e6, "kelly"); }, "sizing");
}

TEST(V12Sizing, GrossEqualsCapitalAtEveryEntry) {
    const Bundled& b = bundled();
    for (const char* sizing : {"beta_hedged", "equal_dollar"}) {
        BacktestConfig c = statarb::config_from_map(b.cfg);
        c.sizing = sizing;
        const auto res = statarb::backtest_pair(b.y, b.x, b.alpha, b.beta, c);
        const double bps = (c.cost_bps + c.slippage_bps) / 1e4;
        ASSERT_FALSE(res.trades.empty());
        for (const auto& tr : res.trades) {
            EXPECT_NEAR(res.daily_cost[static_cast<std::size_t>(tr.entry_day)], kCap * bps,
                        kCap * bps * 1e-12);
        }
    }
}

TEST(V12Sizing, BetaHedgedLegRatioAndEqualDollarNeutral) {
    Vec s(120, 0.0);
    for (int i = 0; i < 70; ++i) s[static_cast<std::size_t>(i)] = 0.2 * std::sin(0.7 * i);
    for (std::size_t i = 70; i < 120; ++i) s[i] = 2.0;
    Px p = scripted(s, 100.0);
    BacktestConfig cfg = cfg_w(40);
    cfg.cost_bps = 0.0;
    cfg.slippage_bps = 0.0;
    const int entry = statarb::backtest_pair(p.y, p.x, 50.0, 1.0, cfg).trades.at(0).entry_day;
    for (std::size_t i = static_cast<std::size_t>(entry) + 1; i < p.x.size(); ++i) {
        p.x[i] += 1.0;
        p.y[i] += 1.0;  // spread unchanged
    }
    auto res = statarb::backtest_pair(p.y, p.x, 50.0, 1.0, cfg);
    ASSERT_EQ(res.trades.at(0).entry_day, entry);
    // hedged at beta: moving along the hedge line earns nothing
    EXPECT_NEAR(res.daily_pnl[static_cast<std::size_t>(entry) + 1], 0.0, 1e-8);

    Vec y2 = p.y;
    for (std::size_t i = static_cast<std::size_t>(entry) + 1; i < y2.size(); ++i) y2[i] -= 1.0;
    BacktestConfig ed = cfg;
    ed.sizing = "equal_dollar";
    const auto res2 = statarb::backtest_pair(y2, p.x, 50.0, 1.0, ed);
    const double qx = (kCap / 2.0) / p.x[static_cast<std::size_t>(entry)];
    EXPECT_EQ(res2.trades.at(0).direction, -1);
    EXPECT_NEAR(res2.daily_pnl[static_cast<std::size_t>(entry) + 1], qx * 1.0, qx * 1e-12);
}

// ---- item 3: risk limits ---------------------------------------------------

TEST(V12Limits, AccountingIdentity) {
    const Bundled& b = bundled();
    const double limits[3][3] = {{0.0, 0, 0}, {0.02, 60, 20}, {0.005, 5, 3}};
    for (const char* sizing : {"beta_hedged", "equal_dollar"}) {
        for (const auto& l : limits) {
            BacktestConfig c = statarb::config_from_map(b.cfg);
            c.sizing = sizing;
            c.pnl_stop_frac = l[0];
            c.max_hold_days = static_cast<int>(l[1]);
            c.cooldown_days = static_cast<int>(l[2]);
            const auto res = statarb::backtest_pair(b.y, b.x, b.alpha, b.beta, c);
            const double lhs = res.equity.back() - kCap;
            EXPECT_NEAR(lhs, sum(res.daily_pnl) - sum(res.daily_cost), 1e-6);
            double tp = 0.0;
            for (const auto& t : res.trades) tp += t.pnl;
            EXPECT_NEAR(lhs, tp, 1e-6);
        }
    }
}

TEST(V12Limits, Validation) {
    const auto mk = [](double pnl, int hold, int cool) {
        return BacktestConfig(60, 2.0, 0.5, 4.0, 1.0, 0.5, 1e6, "beta_hedged", pnl, hold, cool);
    };
    for (double bad : {-0.1, kNaN, kInf}) expect_invalid([&] { mk(bad, 0, 0); }, "pnl_stop_frac");
    expect_invalid([&] { mk(0.0, -1, 0); }, "max_hold_days");
    expect_invalid([&] { mk(0.0, 0, -1); }, "cooldown_days");
    // the run-time re-validation catches a mutation after construction
    BacktestConfig c;
    c.pnl_stop_frac = -1.0;
    const Px p = noise(200);
    expect_invalid([&] { statarb::backtest_pair(p.y, p.x, p.alpha, p.beta, c); });
}

/// Short-spread entry signal on day 70; spread then in the no-rule band and
/// moves to `loss_day_value` on day 73.
Vec trade_then_loss(double loss_day_value, std::size_t T = 140) {
    Vec s = sine(T);
    for (std::size_t i = 70; i < 73; ++i) s[i] = 0.6;
    for (std::size_t i = 73; i < T; ++i) s[i] = loss_day_value;
    return s;
}

TEST(V12Limits, PnlStopFiresAndBeatsZExit) {
    Vec s = sine(140);
    s[70] = 0.6;
    s[71] = 0.6;
    for (std::size_t i = 72; i < s.size(); ++i) s[i] = 0.0;
    Px p = scripted(s);
    for (std::size_t i = 72; i < p.x.size(); ++i) p.x[i] -= 5.0;
    for (std::size_t i = 0; i < p.x.size(); ++i) p.y[i] = 50.0 + p.x[i] + s[i];
    BacktestConfig base = cfg_w(40);
    base.cost_bps = 0.0;
    base.slippage_bps = 0.0;
    const Vec z = statarb::rolling_zscore(statarb::spread(p.y, p.x, 50.0, 1.0), 40);
    EXPECT_LE(std::fabs(z[72]), base.exit_z);
    const auto plain = statarb::backtest_pair(p.y, p.x, 50.0, 1.0, base);
    ASSERT_FALSE(plain.trades.empty());
    EXPECT_FALSE(plain.trades[0].stopped);
    BacktestConfig cfg = base;
    cfg.sizing = "equal_dollar";
    const auto unstopped = statarb::backtest_pair(p.y, p.x, 50.0, 1.0, cfg);
    const double loss = unstopped.daily_pnl[72];
    ASSERT_LT(loss, 0.0);
    cfg.pnl_stop_frac = -loss / kCap / 2.0;
    const auto stopped = statarb::backtest_pair(p.y, p.x, 50.0, 1.0, cfg);
    ASSERT_FALSE(stopped.trades.empty());
    EXPECT_TRUE(stopped.trades[0].stopped);
    EXPECT_EQ(stopped.trades[0].exit_day, 73);
}

TEST(V12Limits, PnlStopUsesCostsSoFar) {
    const Px p = scripted(trade_then_loss(0.6));
    BacktestConfig cfg = cfg_w(40);
    cfg.pnl_stop_frac = 1.0e-4;
    const auto res = statarb::backtest_pair(p.y, p.x, 50.0, 1.0, cfg);
    ASSERT_FALSE(res.trades.empty());
    EXPECT_TRUE(res.trades[0].stopped);
    EXPECT_EQ(res.trades[0].exit_day, res.trades[0].entry_day + 1);
}

TEST(V12Limits, PnlStopBoundaryIsInclusive) {
    const Px p = scripted(trade_then_loss(0.6));
    BacktestConfig cfg = cfg_w(40);
    cfg.pnl_stop_frac = 1.5e-4;  // = entry cost
    const auto res = statarb::backtest_pair(p.y, p.x, 50.0, 1.0, cfg);
    EXPECT_NEAR(res.daily_cost[static_cast<std::size_t>(res.trades.at(0).entry_day)], 150.0,
                150.0 * 1e-12);
    cfg.pnl_stop_frac = 1.5001e-4;
    EXPECT_FALSE(statarb::backtest_pair(p.y, p.x, 50.0, 1.0, cfg).trades.at(0).stopped);
    cfg.pnl_stop_frac = 1.4999e-4;
    EXPECT_TRUE(statarb::backtest_pair(p.y, p.x, 50.0, 1.0, cfg).trades.at(0).stopped);
}

TEST(V12Limits, StopPriorityZStopOutranksMaxHold) {
    Vec s = sine(160);
    for (std::size_t i = 70; i < 73; ++i) s[i] = 0.5;
    s[73] = 5.0;
    const Px p = scripted(s);
    BacktestConfig cfg = cfg_w(40);
    cfg.max_hold_days = 2;
    const Vec z = statarb::rolling_zscore(s, 40);
    EXPECT_GE(z[73], cfg.stop_z);
    const auto tr = statarb::backtest_pair(p.y, p.x, 50.0, 1.0, cfg).trades.at(0);
    EXPECT_EQ(tr.entry_day, 71);
    EXPECT_EQ(tr.exit_day, 74);
    EXPECT_TRUE(tr.stopped);
}

TEST(V12Limits, MaxHoldExitsUnflaggedOnTime) {
    const Px p = scripted(trade_then_loss(0.6));
    BacktestConfig cfg = cfg_w(40);
    cfg.max_hold_days = 3;
    const auto res = statarb::backtest_pair(p.y, p.x, 50.0, 1.0, cfg);
    const auto tr = res.trades.at(0);
    EXPECT_EQ(tr.exit_day - tr.entry_day, 4);  // signal at t - entry >= 3, executes a day later
    EXPECT_FALSE(tr.stopped);
    EXPECT_EQ(res.n_stops(), 0);
}

TEST(V12Limits, FireOnNaNZ) {
    Vec s = {0.0, 0.1, -0.1, 0.05, 0.0, 0.0, 0.3, 0.3};
    s.resize(38, 0.3);
    const Px p = scripted(s);
    const Vec z = statarb::rolling_zscore(s, 5);
    const BacktestConfig cfg(5, 1.5, 0.1, 1.75, 1.0, 0.5, 1e6, "beta_hedged", 0.0, 8, 0);
    EXPECT_TRUE(std::isnan(z[12]));
    const auto res = statarb::backtest_pair(p.y, p.x, 50.0, 1.0, cfg);
    ASSERT_FALSE(res.trades.empty());
    EXPECT_EQ(res.trades[0].exit_day, 16);  // entry 7, signal 15 on a NaN z
    EXPECT_FALSE(res.trades[0].stopped);
}

std::vector<statarb::Trade> stop_then_signal(int gap, int cooldown) {
    Vec s = sine(240);
    for (std::size_t i = 70; i < 73; ++i) s[i] = 0.5;
    s[73] = 5.0;
    s[static_cast<std::size_t>(74 + gap)] = 3.0;
    const Px p = scripted(s);
    BacktestConfig cfg = cfg_w(40);
    cfg.cooldown_days = cooldown;
    return statarb::backtest_pair(p.y, p.x, 50.0, 1.0, cfg).trades;
}

TEST(V12Limits, CooldownTiming) {
    const int gap = 10;
    const auto free_t = stop_then_signal(gap, 0);
    ASSERT_EQ(free_t.size(), 2u);
    EXPECT_TRUE(free_t[0].stopped);
    EXPECT_EQ(free_t[0].exit_day, 74);
    EXPECT_EQ(free_t[1].entry_day, 74 + gap + 1);
    // t = e + gap: blocked while t < e + cooldown, allowed at equality
    EXPECT_EQ(stop_then_signal(gap, gap + 1).size(), 1u);
    const auto allowed = stop_then_signal(gap, gap);
    ASSERT_EQ(allowed.size(), 2u);
    EXPECT_EQ(allowed[1].entry_day, 74 + gap + 1);
}

TEST(V12Limits, CooldownNotAfterUnstoppedExit) {
    Vec s = sine(160);
    s[70] = 0.5;
    s[71] = 0.0;
    s[72] = 0.5;
    s[73] = 0.0;
    const Px p = scripted(s);
    BacktestConfig cfg = cfg_w(40);
    cfg.cooldown_days = 50;
    EXPECT_EQ(statarb::backtest_pair(p.y, p.x, 50.0, 1.0, cfg).trades.size(), 2u);
}

TEST(V12Limits, CooldownAndMaxHoldPropertiesOnBundled) {
    const Bundled& b = bundled();
    const BacktestConfig c = statarb::config_from_map(b.cfg);
    const auto res = statarb::backtest_pair(b.y, b.x, b.alpha, b.beta, c);
    EXPECT_GE(res.n_stops(), 1);
    for (std::size_t i = 0; i + 1 < res.trades.size(); ++i) {
        const auto& prev = res.trades[i];
        const auto& nxt = res.trades[i + 1];
        if (prev.stopped) {
            EXPECT_GE(nxt.entry_day - 1, prev.exit_day + c.cooldown_days);
        }
    }
    for (const auto& tr : res.trades) EXPECT_LE(tr.holding_days(), c.max_hold_days + 1);
}

// ---- item 4: dead configs --------------------------------------------------

TEST(V12Dead, EntryRejected) {
    expect_invalid([] { BacktestConfig(5, 2.0, 0.5, 4.0); }, "entry_z unreachable");
    // (4-1)/2 == 1.5 exactly: still dead
    expect_invalid(
        [] { BacktestConfig(4, 1.5, 0.5, 4.0, 1.0, 0.5, 1e6, "beta_hedged", 0.1); },
        "entry_z unreachable");
}

TEST(V12Dead, StopNeedsPnlStop) {
    expect_invalid([] { BacktestConfig(20, 2.0, 0.5, 8.0); }, "stop_z unreachable");
    expect_invalid(
        [] { BacktestConfig(20, 2.0, 0.5, 8.0, 1.0, 0.5, 1e6, "beta_hedged", 0.0, 10); },
        "stop_z unreachable");
    BacktestConfig(20, 2.0, 0.5, 8.0, 1.0, 0.5, 1e6, "beta_hedged", 0.01);
}

// ---- item 5: pinned summation ----------------------------------------------

TEST(V12Sums, RollingZIsPlainSequentialSumBitwise) {
    std::mt19937_64 gen(1);
    std::normal_distribution<double> n1(0.0, 1.0);
    std::uniform_int_distribution<int> ex(-6, 5);
    Vec s(300);
    for (double& v : s) v = n1(gen) * std::pow(10.0, ex(gen));
    const int W = 37;
    const Vec z = statarb::rolling_zscore(s, W);
    for (std::size_t t = W - 1; t < 300; ++t) {
        double total = 0.0;
        for (std::size_t i = t - W + 1; i <= t; ++i) total = total + s[i];
        const double m = total / W;
        double ss = 0.0;
        for (std::size_t i = t - W + 1; i <= t; ++i) ss = ss + (s[i] - m) * (s[i] - m);
        const double expect = (s[t] - m) / std::sqrt(ss / (W - 1));
        ASSERT_EQ(z[t], expect) << "t=" << t;  // bitwise
    }
}

// ---- item 6: rolling walk-forward ------------------------------------------

BacktestConfig cfg_small() {
    return BacktestConfig(30, 1.5, 0.3, 4.0, 1.0, 0.5, 1e6, "beta_hedged", 0.02, 30, 5);
}

TEST(V12Rolling, GeometryAndFlatAtBlockEnd) {
    const Px p = noise(1000);
    const auto rw = statarb::rolling_walk_forward(p.y, p.x, cfg_small(), 300, 150);
    const int expect[5][2] = {{300, 450}, {450, 600}, {600, 750}, {750, 900}, {900, 1000}};
    ASSERT_EQ(rw.blocks.size(), 5u);
    for (std::size_t k = 0; k < 5; ++k) {
        EXPECT_EQ(rw.blocks[k].start, expect[k][0]);
        EXPECT_EQ(rw.blocks[k].end, expect[k][1]);
    }
    EXPECT_EQ(rw.start_day, 300);
    EXPECT_EQ(rw.equity.size(), 700u);
    EXPECT_FALSE(rw.trades.empty());
    for (const auto& b : rw.blocks) {
        EXPECT_EQ(b.result.equity.size(), static_cast<std::size_t>(b.end - b.start));
        for (const auto& t : b.result.trades) {
            EXPECT_GT(t.entry_day, b.start);
            EXPECT_LE(t.exit_day, b.end - 1);
            EXPECT_GT(t.exit_day, t.entry_day);
        }
    }
}

TEST(V12Rolling, StitchAndFitWindow) {
    const Px p = noise(800);
    const auto rw = statarb::rolling_walk_forward(p.y, p.x, cfg_small(), 300, 100);
    double pnl = 0.0, nt = 0.0, to = 0.0;
    for (const auto& b : rw.blocks) {
        pnl += b.result.equity.back() - kCap;
        nt += b.result.metrics.n_trades;
        to += b.result.metrics.turnover;
    }
    EXPECT_NEAR(rw.equity.back() - kCap, pnl, 1e-6);
    EXPECT_NEAR(rw.metrics.total_return, pnl / kCap, 1e-12);
    EXPECT_EQ(rw.metrics.n_trades, nt);
    EXPECT_NEAR(rw.metrics.turnover, to, 1e-12);
    const std::size_t k = 2;
    const Vec yf(p.y.begin() + 200, p.y.begin() + 500), xf(p.x.begin() + 200, p.x.begin() + 500);
    const auto fit = statarb::engle_granger(yf, xf);
    EXPECT_EQ(rw.blocks[k].alpha, fit.alpha);
    EXPECT_EQ(rw.blocks[k].beta, fit.beta);
}

TEST(V12Rolling, UsesNoFutureData) {
    const Px p = noise(900);
    const BacktestConfig cfg = cfg_small();
    const auto base = statarb::rolling_walk_forward(p.y, p.x, cfg, 300, 100);
    for (int c : {450, 600, 777}) {
        Vec y2 = p.y, x2 = p.x;
        const int m = 900 - c;
        for (int i = 0; i < m; ++i) {
            const double f = static_cast<double>(i) / (m - 1);
            y2[static_cast<std::size_t>(c + i)] *= 1.0 + (0.05 + f * (0.5 - 0.05));
            x2[static_cast<std::size_t>(c + i)] *= 1.0 - (0.02 + f * (0.3 - 0.02));
        }
        const auto alt = statarb::rolling_walk_forward(y2, x2, cfg, 300, 100);
        const std::size_t n = static_cast<std::size_t>(c - base.start_day);
        for (std::size_t i = 0; i < n; ++i) ASSERT_EQ(base.equity[i], alt.equity[i]);
        for (std::size_t k = 0; k < base.blocks.size() && k < alt.blocks.size(); ++k) {
            if (base.blocks[k].start > c) break;
            EXPECT_EQ(base.blocks[k].alpha, alt.blocks[k].alpha);
            EXPECT_EQ(base.blocks[k].beta, alt.blocks[k].beta);
        }
        std::vector<statarb::Trade> t0, t1;
        for (const auto& t : base.trades) {
            if (t.exit_day < c) t0.push_back(t);
        }
        for (const auto& t : alt.trades) {
            if (t.exit_day < c) t1.push_back(t);
        }
        ASSERT_EQ(t0.size(), t1.size());
        for (std::size_t i = 0; i < t0.size(); ++i) {
            EXPECT_EQ(t0[i].entry_day, t1[i].entry_day);
            EXPECT_EQ(t0[i].exit_day, t1[i].exit_day);
            EXPECT_EQ(t0[i].pnl, t1[i].pnl);
        }
    }
}

TEST(V12Rolling, FirstBlockEntryNeedsBlockSignal) {
    const Px p = noise(500);
    const auto rw = statarb::rolling_walk_forward(p.y, p.x, cfg_small(), 300, 200);
    for (const auto& t : rw.trades) EXPECT_GE(t.entry_day, 301);
}

TEST(V12Rolling, BlockZMatchesManualWarmup) {
    const Px p = noise(600, 8);
    BacktestConfig cfg = cfg_small();
    cfg.cooldown_days = 0;
    const auto rw = statarb::rolling_walk_forward(p.y, p.x, cfg, 300, 300);
    const auto& b = rw.blocks.at(0);
    const Vec yb(p.y.begin() + 271, p.y.end()), xb(p.x.begin() + 271, p.x.end());
    const Vec z = statarb::rolling_zscore(statarb::spread(yb, xb, b.alpha, b.beta), cfg.window);
    ASSERT_EQ(z.size() - 29, b.result.zscore.size());
    for (std::size_t i = 0; i < b.result.zscore.size(); ++i) {
        const double a = z[29 + i], c = b.result.zscore[i];
        ASSERT_TRUE((std::isnan(a) && std::isnan(c)) || a == c) << i;
    }
}

TEST(V12Rolling, NegativeFitBlockStaysFlat) {
    const std::size_t T = 400;
    Vec x(T), y(T);
    for (std::size_t t = 0; t < T; ++t) {
        const double d = static_cast<double>(t);
        x[t] = 100.0 + 0.1 * d + std::sin(d);
        y[t] = 200.0 - 0.1 * d + std::cos(0.3 * d);
    }
    const auto rw = statarb::rolling_walk_forward(y, x, cfg_small(), 200, 100);
    for (const auto& b : rw.blocks) {
        EXPECT_TRUE(b.skipped);
        EXPECT_LT(b.beta, 0.0);
    }
    EXPECT_TRUE(rw.trades.empty());
    for (double e : rw.equity) EXPECT_EQ(e, kCap);
    EXPECT_EQ(rw.metrics.turnover, 0.0);
}

TEST(V12Rolling, Validation) {
    const Px p = noise(1000);
    for (const auto& tt : {std::pair<int, int>{10, 50}, {300, 0}, {1000, 50}, {300, -5}}) {
        expect_invalid([&] {
            statarb::rolling_walk_forward(p.y, p.x, cfg_small(), tt.first, tt.second);
        });
    }
    const Px q = noise(400);
    Vec shorter(q.x.begin(), q.x.end() - 1);
    expect_invalid([&] { statarb::rolling_walk_forward(q.y, shorter, cfg_small(), 100, 50); },
                   "equal length");
}

TEST(V12Rolling, WalkForwardStillSingleHoldout) {
    const Bundled& b = bundled();
    const auto wf = statarb::walk_forward(b.y, b.x, statarb::config_from_map(b.cfg), 0.6);
    EXPECT_EQ(wf.split, 900u);
}

// ---- item 7: streaming -----------------------------------------------------

TEST(V12Stream, RollingZMatchesBatch) {
    const Bundled& b = bundled();
    const Vec s = statarb::spread(b.y, b.x, b.alpha, b.beta);
    for (int window : {2, 5, 60, 250}) {
        const Vec batch = statarb::rolling_zscore(s, window);
        statarb::RollingZ rz(window);
        double worst = 0.0;
        for (std::size_t t = 0; t < s.size(); ++t) {
            const double got = rz.update(s[t]);
            ASSERT_EQ(std::isnan(got), std::isnan(batch[t])) << "w=" << window << " t=" << t;
            if (!std::isnan(got)) worst = std::max(worst, std::fabs(got - batch[t]));
        }
        EXPECT_LT(worst, 1e-9) << "window " << window;
    }
}

TEST(V12Stream, RollingZLongRunWithLargeLevel) {
    std::mt19937_64 gen(5);
    std::normal_distribution<double> n1(0.0, 1.0);
    Vec s(50000);
    double level = 1.0e4;
    for (double& v : s) {
        level += n1(gen);
        v = level;
    }
    const Vec batch = statarb::rolling_zscore(s, 60);
    statarb::RollingZ rz(60);
    double worst = 0.0;
    for (std::size_t t = 0; t < s.size(); ++t) {
        const double got = rz.update(s[t]);
        if (!std::isnan(batch[t])) worst = std::max(worst, std::fabs(got - batch[t]));
    }
    EXPECT_LT(worst, 1e-7);
}

TEST(V12Stream, RollingZZeroVarianceAndValidation) {
    statarb::RollingZ rz(3);
    for (int i = 0; i < 5; ++i) EXPECT_TRUE(std::isnan(rz.update(1.0)));
    expect_invalid([] { statarb::RollingZ(1); });
    expect_invalid([] { statarb::RollingZ(0); });
    expect_invalid([&] { rz.update(kNaN); });
    expect_invalid([&] { rz.update(kInf); });
    EXPECT_TRUE(std::isnan(rz.update(1.0)));  // the rejected values left no trace
}

TEST(V12Stream, KalmanStepMatchesBatch) {
    const Bundled& b = bundled();
    const double d = b.cfg.at("kalman_delta"), r = b.cfg.at("kalman_r");
    const auto km = statarb::kalman_hedge(b.y, b.x, d, r);
    statarb::KalmanStep ks(d, r);
    double worst = 0.0;
    for (std::size_t t = 0; t < b.y.size(); ++t) {
        const auto o = ks.update(b.x[t], b.y[t]);
        worst = std::max({worst, std::fabs(o.alpha - km.alpha[t]), std::fabs(o.beta - km.beta[t]),
                          std::fabs(o.z - km.z[t])});
    }
    EXPECT_LT(worst, 1e-12);
}

TEST(V12Stream, KalmanScalarRecursionMatchesMatrixJosephForm) {
    const Bundled& b = bundled();
    const double d = b.cfg.at("kalman_delta"), r = b.cfg.at("kalman_r");
    const double q = d / (1.0 - d);
    const auto km = statarb::kalman_hedge(b.y, b.x, d, r);
    double s[2] = {0, 0};
    double P[2][2] = {{1e3, 0}, {0, 1e3}};
    for (std::size_t t = 0; t < b.y.size(); ++t) {
        double Pp[2][2] = {{P[0][0] + q, P[0][1]}, {P[1][0], P[1][1] + q}};
        const double h[2] = {1.0, b.x[t]};
        double Ph[2] = {Pp[0][0] * h[0] + Pp[0][1] * h[1], Pp[1][0] * h[0] + Pp[1][1] * h[1]};
        const double S = h[0] * Ph[0] + h[1] * Ph[1] + r;
        const double K[2] = {Ph[0] / S, Ph[1] / S};
        const double e = b.y[t] - (h[0] * s[0] + h[1] * s[1]);
        s[0] += K[0] * e;
        s[1] += K[1] * e;
        const double A[2][2] = {{1 - K[0] * h[0], -K[0] * h[1]}, {-K[1] * h[0], 1 - K[1] * h[1]}};
        double AP[2][2], N[2][2];
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j) AP[i][j] = A[i][0] * Pp[0][j] + A[i][1] * Pp[1][j];
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                N[i][j] = AP[i][0] * A[j][0] + AP[i][1] * A[j][1] + r * K[i] * K[j];
        const double off = 0.5 * (N[0][1] + N[1][0]);
        P[0][0] = N[0][0];
        P[0][1] = P[1][0] = off;
        P[1][1] = N[1][1];
        ASSERT_LT(std::fabs(s[1] - km.beta[t]), 1e-9);
        ASSERT_LT(std::fabs(s[0] - km.alpha[t]), 1e-9);
    }
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 2; ++j)
            EXPECT_NEAR(P[i][j], km.final_cov(i, j), 1e-8 * std::fabs(P[i][j]) + 1e-14);
}

TEST(V12Stream, KalmanStepValidation) {
    const double bad[][2] = {{0.0, 1.0}, {1.0, 1.0}, {kNaN, 1.0}, {0.1, 0.0}, {0.1, kInf}, {0.1, kNaN}};
    for (const auto& p : bad) expect_invalid([&] { statarb::KalmanStep(p[0], p[1]); });
    statarb::KalmanStep ks;
    expect_invalid([&] { ks.update(kNaN, 1.0); });
    expect_invalid([&] { ks.update(1.0, kInf); });
    // state unchanged after a rejected update
    statarb::KalmanStep fresh;
    const auto a = ks.update(10.0, 20.0);
    const auto c = fresh.update(10.0, 20.0);
    EXPECT_EQ(a.alpha, c.alpha);
    EXPECT_EQ(a.beta, c.beta);
}

// ---- item 8: lag selection and two-way EG -----------------------------------

Vec ar2(std::size_t T = 800, unsigned seed = 2) {
    std::mt19937_64 gen(seed);
    std::normal_distribution<double> n1(0.0, 1.0);
    Vec v(T, 0.0);
    for (std::size_t t = 2; t < T; ++t) v[t] = 0.3 * v[t - 1] + 0.5 * v[t - 2] + n1(gen);
    return v;
}

TEST(V12Lag, SelectLagFindsAugmentation) {
    EXPECT_EQ(statarb::select_lag(ar2(), 6, 1), 1);
    std::mt19937_64 gen(4);
    std::normal_distribution<double> n1(0.0, 1.0);
    Vec w(800);
    for (double& e : w) e = n1(gen);
    EXPECT_EQ(statarb::select_lag(w, 6, 1), 0);
}

TEST(V12Lag, SelectLagMatchesDirectBic) {
    const Vec v = ar2(300, 9);
    const std::size_t m = 4;
    Vec dy(v.size() - 1);
    for (std::size_t t = 1; t < v.size(); ++t) dy[t - 1] = v[t] - v[t - 1];
    const std::size_t n = dy.size() - m;
    Vec lhs(dy.begin() + static_cast<long>(m), dy.end());
    double best = kInf;
    int best_p = 0;
    for (std::size_t p = 0; p <= m; ++p) {
        statarb::Matrix x(n, 1 + p);
        for (std::size_t i = 0; i < n; ++i) {
            x(i, 0) = v[m + i];
            for (std::size_t l = 1; l <= p; ++l) x(i, l) = dy[m - l + i];
        }
        const auto fit = statarb::ols(lhs, x, true);
        double rss = 0.0;
        for (double e : fit.resid) rss += e * e;
        const double bic = static_cast<double>(n) * std::log(rss / static_cast<double>(n)) +
                           static_cast<double>(2 + p) * std::log(static_cast<double>(n));
        if (bic < best) {
            best = bic;
            best_p = static_cast<int>(p);
        }
    }
    EXPECT_EQ(statarb::select_lag(v, static_cast<int>(m), 1), best_p);
}

TEST(V12Lag, SelectLagValidation) {
    const Vec v = ar2(100);
    expect_invalid([&] { statarb::select_lag(v, -1, 1); });
    expect_invalid([&] { statarb::select_lag(v, 2, 4); });
    expect_invalid([&] { statarb::select_lag(v, 90, 1); });
    EXPECT_EQ(statarb::select_lag(v, 0, 1), 0);
    statarb::adf_test(v, statarb::select_lag(v, 3, 1), 1);
}

TEST(V12Lag, EngleGrangerBothNeedsBoth) {
    const Bundled& b = bundled();
    const Vec y(b.y.begin(), b.y.begin() + 900), x(b.x.begin(), b.x.begin() + 900);
    const auto both = statarb::engle_granger_both(y, x, 1);
    EXPECT_EQ(both.y_on_x.beta, statarb::engle_granger(y, x, 1).beta);
    EXPECT_EQ(both.x_on_y.beta, statarb::engle_granger(x, y, 1).beta);
    EXPECT_EQ(both.cointegrated(0.05),
              both.y_on_x.cointegrated(0.05) && both.x_on_y.cointegrated(0.05));
    EXPECT_FALSE(statarb::engle_granger_both(b.walk_a, b.walk_b, 1).cointegrated(0.05));
}

// ---- item 9: strict loaders --------------------------------------------------

struct TempFile {
    std::string path;
    explicit TempFile(const std::string& text) {
        char name[] = "/tmp/statarb_v12_XXXXXX";
        const int fd = mkstemp(name);
        if (fd >= 0) close(fd);
        path = name;
        std::ofstream(path, std::ios::binary) << text;
    }
    ~TempFile() { std::remove(path.c_str()); }
};

bool config_rejected(const std::string& text) {
    const TempFile f(text);
    try {
        statarb::load_config(f.path);
        return false;
    } catch (const std::invalid_argument&) {
        return true;
    }
}

bool csv_rejected(const std::string& text) {
    const TempFile f(text);
    try {
        statarb::load_csv(f.path);
        return false;
    } catch (const std::invalid_argument&) {
        return true;
    }
}

TEST(V12Loaders, ConfigRejectsTrailingContent) {
    EXPECT_FALSE(config_rejected("{\"window\": 60}"));
    EXPECT_FALSE(config_rejected("  {\"window\": 60}\n\n"));
    EXPECT_FALSE(config_rejected("{}"));
    EXPECT_TRUE(config_rejected("{\"window\": 60} x"));
    EXPECT_TRUE(config_rejected("{\"window\": 60}{\"a\": 1}"));
    EXPECT_TRUE(config_rejected("{} {}"));
    EXPECT_TRUE(config_rejected("{}x"));
    EXPECT_TRUE(config_rejected("{\"a\": 1,}"));
    EXPECT_TRUE(config_rejected(""));
    EXPECT_TRUE(config_rejected("{\"a\": 1"));
}

TEST(V12Loaders, ConfigNumberGrammar) {
    for (const char* v : {"0x10", "1_0", "1f", "+1", "01", ".5", "1.", "nan", "NaN", "Infinity",
                          "-Infinity", "1e999", "true", "\"1\"", "[1]", "1e", "--1", ""}) {
        EXPECT_TRUE(config_rejected(std::string("{\"a\": ") + v + "}")) << v;
    }
    for (const char* v : {"0", "-0", "1", "-1.5", "0.5", "1e3", "1E-3", "2.5e+2"}) {
        EXPECT_FALSE(config_rejected(std::string("{\"a\": ") + v + "}")) << v;
    }
}

TEST(V12Loaders, ConfigIntegerKeys) {
    for (const char* k : {"window", "adf_maxlag", "max_hold_days", "cooldown_days"}) {
        const std::string key = std::string("\"") + k + "\"";
        EXPECT_FALSE(config_rejected("{" + key + ": 60}")) << k;
        EXPECT_TRUE(config_rejected("{" + key + ": 60.0}")) << k;
        EXPECT_TRUE(config_rejected("{" + key + ": 6e1}")) << k;
        EXPECT_TRUE(config_rejected("{" + key + ": 2.5}")) << k;
    }
    EXPECT_FALSE(config_rejected("{\"capital\": 1e6}"));
}

TEST(V12Loaders, ConfigIntegerMagnitude) {
    EXPECT_FALSE(config_rejected("{\"a\": 9007199254740992}"));
    EXPECT_FALSE(config_rejected("{\"a\": -9007199254740992}"));
    EXPECT_TRUE(config_rejected("{\"a\": 9007199254740993}"));
    EXPECT_TRUE(config_rejected("{\"a\": 12345678901234567890}"));
}

TEST(V12Loaders, ConfigIsLocaleIndependent) {
    const char* names[] = {"de_DE.UTF-8", "fr_FR.UTF-8", "de_DE", "fr_FR"};
    const char* old = std::setlocale(LC_ALL, nullptr);
    const std::string saved = old ? old : "C";
    for (const char* n : names) {
        if (!std::setlocale(LC_ALL, n)) continue;
        const TempFile f("{\"a\": 0.5, \"b\": 1.25e2}");
        const auto cfg = statarb::load_config(f.path);
        const TempFile c("x\n1.5\n2.25\n");
        const auto d = statarb::load_csv(c.path);
        std::setlocale(LC_ALL, saved.c_str());
        EXPECT_EQ(cfg.at("a"), 0.5);
        EXPECT_EQ(cfg.at("b"), 125.0);
        EXPECT_EQ(d.at("x")[0], 1.5);
        EXPECT_EQ(d.at("x")[1], 2.25);
        return;
    }
    std::setlocale(LC_ALL, saved.c_str());
}

TEST(V12Loaders, CsvNumberGrammar) {
    for (const char* v : {"0x10", "1_0", "1f", "1d", "1L", "nan", "inf", "Infinity", "-inf",
                          "1e999", "1e", "e5", "+", "-", ".", "1,5", "1 2", "0b1", "\xD9\xA1"}) {
        EXPECT_TRUE(csv_rejected(std::string("a\n") + v + "\n")) << v;
    }
    const auto ok = [](const char* v, double expect) {
        const TempFile f(std::string("a\n") + v + "\n");
        const auto d = statarb::load_csv(f.path);
        EXPECT_EQ(d.at("a")[0], expect) << v;
    };
    ok("5", 5.0);
    ok("+5", 5.0);
    ok("-5.", -5.0);
    ok(".5", 0.5);
    ok("1e3", 1000.0);
    ok("  2.5 \t", 2.5);
    ok("1e-999", 0.0);  // underflow is finite, as in Python's float()
}

TEST(V12Loaders, CsvDayColumnMustBeInteger) {
    EXPECT_FALSE(csv_rejected("day,a\n1,1.5\n2,2.5\n"));
    EXPECT_TRUE(csv_rejected("day,a\n1.0,1.5\n2.0,2.5\n"));
    EXPECT_TRUE(csv_rejected("day,a\n1e0,1.5\n2e0,2.5\n"));
    EXPECT_TRUE(csv_rejected("day,a\n0x1,1.5\n0x2,2.5\n"));
}

}  // namespace
