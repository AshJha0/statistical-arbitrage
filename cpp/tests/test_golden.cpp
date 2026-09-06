/// \file test_golden.cpp
/// \brief Golden-value suite: every case in ../data/golden/golden.json is
/// recomputed from the bundled CSVs and must agree within its tolerance.
/// Tests run with the working directory set to cpp/ (see CMakeLists.txt).
#include <gtest/gtest.h>

#include <cmath>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "mini_json.hpp"
#include "statarb/adf.hpp"
#include "statarb/backtest.hpp"
#include "statarb/data_io.hpp"
#include "statarb/engle_granger.hpp"
#include "statarb/johansen.hpp"
#include "statarb/kalman.hpp"
#include "statarb/ols.hpp"
#include "statarb/signals.hpp"

namespace {

const std::string kDataDir = "../data";

/// Exact number of golden cases (kept in step with generate_data.py).
constexpr std::size_t kGoldenCount = 24;

struct SharedData {
    std::map<std::string, double> cfg;
    std::vector<double> y, x;              // pair_equity columns
    std::vector<double> walk_a, walk_b;    // random_walks columns
    int n_is = 0;
    statarb::EGResult eg_insample;

    SharedData() {
        cfg = statarb::load_config(kDataDir + "/config.json");
        const auto pair = statarb::load_csv(kDataDir + "/pair_equity.csv");
        y = pair.at("stock_y");
        x = pair.at("stock_x");
        const auto walks = statarb::load_csv(kDataDir + "/random_walks.csv");
        walk_a = walks.at("walk_a");
        walk_b = walks.at("walk_b");
        n_is = static_cast<int>(static_cast<double>(y.size()) *
                                cfg.at("insample_frac"));
        const std::vector<double> yi(y.begin(), y.begin() + n_is);
        const std::vector<double> xi(x.begin(), x.begin() + n_is);
        eg_insample = statarb::engle_granger(
            yi, xi, static_cast<int>(cfg.at("adf_maxlag")));
    }
};

const SharedData& shared() {
    static const SharedData data;
    return data;
}

/// Recompute one golden case exactly as python/tests/test_golden.py does.
double compute_case(const std::string& name, const mini_json::Value& inputs) {
    const SharedData& d = shared();
    const int maxlag = static_cast<int>(d.cfg.at("adf_maxlag"));

    if (name.rfind("ols_pair", 0) == 0) {
        const statarb::OLSResult fit = statarb::ols(d.y, d.x, true);
        if (name == "ols_pair_alpha") return fit.params[0];
        if (name == "ols_pair_beta") return fit.params[1];
        if (name == "ols_pair_r2") return fit.r2;
    }
    if (name.rfind("adf_random_walk", 0) == 0) {
        const std::vector<double>& col =
            inputs.at("column").str == "walk_a" ? d.walk_a : d.walk_b;
        const statarb::ADFResult res = statarb::adf_test(col, maxlag, 1);
        if (name == "adf_random_walk_tstat") return res.tstat;
        if (name == "adf_random_walk_reject") return res.reject(0.05) ? 1.0 : 0.0;
        if (name == "adf_random_walk_crit5") return res.crit.cv5;
        if (name == "adf_random_walk_pvalue") return res.pvalue;
    }
    if (name == "eg_spurious_coint") {
        const statarb::EGResult eg =
            statarb::engle_granger(d.walk_a, d.walk_b, maxlag);
        return eg.cointegrated(0.05) ? 1.0 : 0.0;
    }
    if (name == "adf_resid_insample_tstat") return d.eg_insample.adf.tstat;
    if (name == "adf_resid_insample_crit5") return d.eg_insample.adf.crit.cv5;
    if (name == "eg_pair_insample_coint") {
        return d.eg_insample.cointegrated(0.05) ? 1.0 : 0.0;
    }
    if (name == "halflife_insample_spread") {
        return statarb::half_life(d.eg_insample.resid);
    }
    if (name.rfind("johansen_pair", 0) == 0) {
        statarb::Matrix lv(static_cast<std::size_t>(d.n_is), 2);
        for (int i = 0; i < d.n_is; ++i) {
            lv(static_cast<std::size_t>(i), 0) = d.y[static_cast<std::size_t>(i)];
            lv(static_cast<std::size_t>(i), 1) = d.x[static_cast<std::size_t>(i)];
        }
        const statarb::JohansenResult jo = statarb::johansen_trace(lv);
        if (name == "johansen_pair_trace_r0") return jo.trace_stats[0];
        if (name == "johansen_pair_trace_r1") return jo.trace_stats[1];
        if (name == "johansen_pair_rank") return static_cast<double>(jo.rank);
    }
    if (name.rfind("kalman_beta", 0) == 0) {
        const statarb::KalmanResult km = statarb::kalman_hedge(
            d.y, d.x, d.cfg.at("kalman_delta"), d.cfg.at("kalman_r"));
        return km.beta[static_cast<std::size_t>(inputs.at("day").number)];
    }
    if (name.rfind("zscore_day", 0) == 0) {
        const std::vector<double> z = statarb::rolling_zscore(
            statarb::spread(d.y, d.x, d.eg_insample.alpha, d.eg_insample.beta),
            static_cast<int>(d.cfg.at("window")));
        return z[static_cast<std::size_t>(inputs.at("day").number)];
    }
    if (name.rfind("backtest_", 0) == 0) {
        const statarb::BacktestConfig bt_cfg(
            static_cast<int>(d.cfg.at("window")), d.cfg.at("entry_z"),
            d.cfg.at("exit_z"), d.cfg.at("stop_z"), d.cfg.at("cost_bps"),
            d.cfg.at("slippage_bps"), d.cfg.at("capital"));
        const statarb::BacktestResult bt = statarb::backtest_pair(
            d.y, d.x, d.eg_insample.alpha, d.eg_insample.beta, bt_cfg);
        if (name == "backtest_total_return") return bt.metrics.total_return;
        if (name == "backtest_sharpe") return bt.metrics.sharpe;
        if (name == "backtest_max_drawdown") return bt.metrics.max_drawdown;
        if (name == "backtest_n_trades") return bt.metrics.n_trades;
    }
    throw std::runtime_error("unknown golden case: " + name);
}

TEST(Golden, AllCasesWithinTolerance) {
    const mini_json::Value doc =
        mini_json::parse_file(kDataDir + "/golden/golden.json");
    const mini_json::Value& cases = doc.at("cases");
    ASSERT_EQ(cases.arr.size(), kGoldenCount);
    for (const mini_json::Value& c : cases.arr) {
        const std::string name = c.at("name").str;
        SCOPED_TRACE("golden case: " + name);
        const double expect = c.at("expect").at("value").number;
        const double tol = c.at("tol").number;
        const double got = compute_case(name, c.at("inputs"));
        EXPECT_NEAR(got, expect, tol);
    }
}

TEST(Golden, SchemaFlatScalars) {
    const mini_json::Value doc =
        mini_json::parse_file(kDataDir + "/golden/golden.json");
    const mini_json::Value& cases = doc.at("cases");
    for (const mini_json::Value& c : cases.arr) {
        ASSERT_EQ(c.obj.size(), 4u);
        ASSERT_TRUE(c.has("name") && c.has("inputs") && c.has("expect") &&
                    c.has("tol"));
        for (const auto& kv : c.at("inputs").obj) {
            const mini_json::Value::Type t = kv.second.type;
            ASSERT_TRUE(t == mini_json::Value::Type::kNumber ||
                        t == mini_json::Value::Type::kString);
        }
        ASSERT_EQ(c.at("expect").obj.size(), 1u);
        ASSERT_GT(c.at("tol").number, 0.0);
    }
}

TEST(Golden, IntegerCasesExact) {
    // Trade count and Johansen rank are integers: exact match required.
    const mini_json::Value doc =
        mini_json::parse_file(kDataDir + "/golden/golden.json");
    for (const mini_json::Value& c : doc.at("cases").arr) {
        const std::string name = c.at("name").str;
        if (name == "backtest_n_trades" || name == "johansen_pair_rank") {
            SCOPED_TRACE(name);
            EXPECT_EQ(compute_case(name, c.at("inputs")),
                      c.at("expect").at("value").number);
        }
    }
}

}  // namespace
