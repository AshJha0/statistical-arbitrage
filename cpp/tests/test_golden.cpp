/// \file test_golden.cpp
/// \brief Golden-value suite: EVERY case in ../data/golden/golden.json is
/// recomputed from the bundled CSVs and must agree within its tolerance.
/// Cases carrying `inputs.kind` are dispatched by kind (no enumerated
/// subset); the older name-keyed cases are dispatched by name exactly as
/// python/tests/test_golden.py does.  Tests run with the working directory
/// set to cpp/ (see CMakeLists.txt).
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <set>
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
constexpr std::size_t kGoldenCount = 928;

bool starts_with(const std::string& s, const std::string& p) {
    return s.compare(0, p.size(), p) == 0;
}

struct SharedData {
    std::map<std::string, double> cfg;
    std::vector<double> y, x;              // pair_equity columns
    std::vector<double> walk_a, walk_b;    // random_walks columns
    int n_is = 0;
    statarb::EGResult eg_insample;
    std::map<std::string, statarb::BacktestResult> bt;
    std::map<std::string, statarb::RollingWalkForwardResult> rw;
    std::unique_ptr<statarb::WalkForwardResult> wf;

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

    /// Scenario names: {limits,plain}_{beta,dollar}.
    statarb::BacktestConfig scenario(const std::string& name) const {
        std::map<std::string, double> c = cfg;
        if (!starts_with(name, "limits")) {
            c["pnl_stop_frac"] = 0.0;
            c["max_hold_days"] = 0.0;
            c["cooldown_days"] = 0.0;
        }
        statarb::BacktestConfig out = statarb::config_from_map(c);
        if (name == "limits_beta" || name == "plain_beta") out.sizing = "beta_hedged";
        else if (name == "limits_dollar" || name == "plain_dollar") out.sizing = "equal_dollar";
        else throw std::runtime_error("unknown scenario: " + name);
        return out;
    }

    const statarb::BacktestResult& backtest(const std::string& name) {
        auto it = bt.find(name);
        if (it == bt.end()) {
            it = bt.emplace(name, statarb::backtest_pair(y, x, eg_insample.alpha,
                                                         eg_insample.beta, scenario(name)))
                     .first;
        }
        return it->second;
    }

    const statarb::WalkForwardResult& walk() {
        if (!wf) {
            wf = std::make_unique<statarb::WalkForwardResult>(statarb::walk_forward(
                y, x, scenario("limits_beta"), cfg.at("insample_frac")));
        }
        return *wf;
    }

    const statarb::RollingWalkForwardResult& rolling(const std::string& name, int train,
                                                     int test) {
        const std::string key = name + "/" + std::to_string(train) + "/" + std::to_string(test);
        auto it = rw.find(key);
        if (it == rw.end()) {
            it = rw.emplace(key, statarb::rolling_walk_forward(y, x, scenario(name), train, test))
                     .first;
        }
        return it->second;
    }
};

SharedData& shared() {
    static SharedData data;
    return data;
}

double metric(const statarb::Metrics& m, const std::string& field) {
    if (field == "total_return") return m.total_return;
    if (field == "ann_return") return m.ann_return;
    if (field == "sharpe") return m.sharpe;
    if (field == "max_drawdown") return m.max_drawdown;
    if (field == "n_trades") return m.n_trades;
    if (field == "win_rate") return m.win_rate;
    if (field == "avg_holding_days") return m.avg_holding_days;
    if (field == "turnover") return m.turnover;
    throw std::runtime_error("unknown metric field: " + field);
}

double sum_of(const std::vector<double>& v) {
    double s = 0.0;
    for (double e : v) s += e;
    return s;
}

double field_of(const statarb::BacktestResult& r, const std::string& field) {
    if (field == "n_stops") return static_cast<double>(r.n_stops());
    if (field == "equity_sum") return sum_of(r.equity);
    return metric(r.metrics, field);
}

double field_of(const statarb::RollingWalkForwardResult& r, const std::string& field) {
    if (field == "n_stops") return static_cast<double>(r.n_stops());
    if (field == "equity_sum") return sum_of(r.equity);
    return metric(r.metrics, field);
}

double min_margin(SharedData& d) {
    const std::vector<std::string> thresholds = {"entry_z", "exit_z", "stop_z"};
    std::vector<const std::vector<double>*> paths;
    for (const char* n : {"limits_beta", "limits_dollar", "plain_beta", "plain_dollar"}) {
        paths.push_back(&d.backtest(n).zscore);
    }
    paths.push_back(&d.walk().insample.zscore);
    paths.push_back(&d.walk().oos.zscore);
    for (const char* n : {"limits_beta", "plain_beta"}) {
        for (const auto& b : d.rolling(n, 500, 100).blocks) paths.push_back(&b.result.zscore);
    }
    double best = std::numeric_limits<double>::infinity();
    for (const auto* p : paths) {
        for (const std::string& k : thresholds) {
            for (double z : *p) {
                if (std::isnan(z)) continue;
                best = std::min(best, std::fabs(std::fabs(z) - d.cfg.at(k)));
            }
        }
    }
    return best;
}

int as_int(const mini_json::Value& v) { return static_cast<int>(v.number); }

double compute_kind(const std::string& kind, const mini_json::Value& in) {
    SharedData& d = shared();
    const std::string field = in.has("field") ? in.at("field").str : "";
    if (kind == "backtest") return field_of(d.backtest(in.at("scenario").str), field);
    if (kind == "backtest_equity") {
        return d.backtest(in.at("scenario").str).equity.at(static_cast<std::size_t>(as_int(in.at("day"))));
    }
    if (kind == "backtest_trade") {
        const auto& tr = d.backtest(in.at("scenario").str)
                             .trades.at(static_cast<std::size_t>(as_int(in.at("index"))));
        if (field == "entry") return tr.entry_day;
        if (field == "exit") return tr.exit_day;
        if (field == "direction") return tr.direction;
        if (field == "pnl") return tr.pnl;
        if (field == "stopped") return tr.stopped ? 1.0 : 0.0;
        throw std::runtime_error("unknown trade field: " + field);
    }
    if (kind == "walk_forward") {
        const auto& wf = d.walk();
        const std::string seg = in.at("segment").str;
        if (seg == "fit") {
            if (field == "alpha") return wf.alpha;
            if (field == "beta") return wf.beta;
            if (field == "split") return static_cast<double>(wf.split);
            throw std::runtime_error("unknown fit field: " + field);
        }
        return field_of(seg == "insample" ? wf.insample : wf.oos, field);
    }
    if (kind == "rolling") {
        const auto& r = d.rolling(in.at("scenario").str, as_int(in.at("train")), as_int(in.at("test")));
        return field == "n_blocks" ? static_cast<double>(r.blocks.size()) : field_of(r, field);
    }
    if (kind == "rolling_block") {
        const auto& r = d.rolling(in.at("scenario").str, as_int(in.at("train")), as_int(in.at("test")));
        const auto& b = r.blocks.at(static_cast<std::size_t>(as_int(in.at("block"))));
        if (field == "start") return b.start;
        if (field == "end") return b.end;
        if (field == "alpha") return b.alpha;
        if (field == "beta") return b.beta;
        if (field == "skipped") return b.skipped ? 1.0 : 0.0;
        return metric(b.result.metrics, field);
    }
    if (kind == "margin") {
        const double got = min_margin(d);
        EXPECT_GT(got, 1e-6);
        return got;
    }
    throw std::runtime_error("unknown golden kind: " + kind);
}

std::string last_token(const std::string& name) { return name.substr(name.rfind('_') + 1); }

/// Name-dispatched cases added in v1.2; false when the name is older.
bool compute_v12(const std::string& name, const mini_json::Value& in, double& out) {
    SharedData& d = shared();
    if (starts_with(name, "johansen_fx") || starts_with(name, "adf_fx")) {
        const auto fx = statarb::load_csv(kDataDir + "/fx_triangle.csv");
        const std::vector<double>& a = fx.at("eurusd");
        const std::vector<double>& b = fx.at("gbpusd");
        const std::vector<double>& c = fx.at("eurgbp");
        const std::size_t n = a.size();
        statarb::Matrix ylog(n, 3);
        for (std::size_t i = 0; i < n; ++i) {
            ylog(i, 0) = std::log(a[i]);
            ylog(i, 1) = std::log(b[i]);
            ylog(i, 2) = std::log(c[i]);
        }
        if (starts_with(name, "adf_fx")) {
            std::vector<double> y0(n);
            statarb::Matrix rest(n, 2);
            for (std::size_t i = 0; i < n; ++i) {
                y0[i] = ylog(i, 0);
                rest(i, 0) = ylog(i, 1);
                rest(i, 1) = ylog(i, 2);
            }
            const statarb::OLSResult fit = statarb::ols(y0, rest, true);
            const statarb::ADFResult res = statarb::adf_test(
                fit.resid, as_int(in.at("maxlag")), as_int(in.at("nvars")));
            const std::string f = last_token(name);
            if (f == "tstat") out = res.tstat;
            else if (f == "crit1") out = res.crit.cv1;
            else if (f == "crit5") out = res.crit.cv5;
            else if (f == "crit10") out = res.crit.cv10;
            else if (f == "pvalue") out = res.pvalue;
            else throw std::runtime_error("unknown adf_fx field: " + name);
            return true;
        }
        const statarb::JohansenResult jo = statarb::johansen_trace(ylog);
        const std::string tail = name.substr(std::string("johansen_fx_").size());
        const std::size_t idx = static_cast<std::size_t>(tail.back() - '0');
        if (tail == "rank") out = static_cast<double>(jo.rank);
        else if (starts_with(tail, "trace_r")) out = jo.trace_stats.at(idx);
        else out = jo.beta(idx, 0);
        return true;
    }
    if (starts_with(name, "kalman_alpha") || starts_with(name, "kalman_z")) {
        const statarb::KalmanResult km =
            statarb::kalman_hedge(d.y, d.x, in.at("delta").number, in.at("r").number);
        const auto& path = starts_with(name, "kalman_alpha") ? km.alpha : km.z;
        out = path.at(static_cast<std::size_t>(as_int(in.at("day"))));
        return true;
    }
    if (starts_with(name, "adf_resid_lag")) {
        const statarb::ADFResult res = statarb::adf_test(
            d.eg_insample.resid, as_int(in.at("maxlag")), as_int(in.at("nvars")));
        const std::string f = last_token(name);
        if (f == "tstat") out = res.tstat;
        else if (f == "crit5") out = res.crit.cv5;
        else if (f == "nobs") out = static_cast<double>(res.nobs);
        else throw std::runtime_error("unknown adf_resid_lag field: " + name);
        return true;
    }
    if (name == "select_lag_resid") {
        out = statarb::select_lag(d.eg_insample.resid, as_int(in.at("max_lag")),
                                  as_int(in.at("nvars")));
        return true;
    }
    if (name == "select_lag_walk") {
        const auto walks = statarb::load_csv(kDataDir + "/random_walks.csv");
        out = statarb::select_lag(walks.at(in.at("column").str), as_int(in.at("max_lag")),
                                  as_int(in.at("nvars")));
        return true;
    }
    if (starts_with(name, "eg_both")) {
        const std::vector<double> yi(d.y.begin(), d.y.begin() + 900);
        const std::vector<double> xi(d.x.begin(), d.x.begin() + 900);
        const statarb::EGBothResult both = statarb::engle_granger_both(
            yi, xi, static_cast<int>(d.cfg.at("adf_maxlag")));
        out = name == "eg_both_insample_coint" ? (both.cointegrated(0.05) ? 1.0 : 0.0)
                                               : both.x_on_y.adf.tstat;
        return true;
    }
    if (name == "halflife_growth_is_inf") {
        std::vector<double> v(static_cast<std::size_t>(as_int(in.at("n"))));
        for (std::size_t i = 0; i < v.size(); ++i) v[i] = std::pow(1.1, static_cast<double>(i));
        out = std::isinf(statarb::half_life(v)) ? 1.0 : 0.0;
        return true;
    }
    if (name == "halflife_alternating_zero") {
        std::vector<double> v(static_cast<std::size_t>(as_int(in.at("n"))));
        for (std::size_t i = 0; i < v.size(); ++i) v[i] = i % 2 == 0 ? 1.0 : -1.0;
        out = statarb::half_life(v);
        return true;
    }
    return false;
}

/// Recompute one golden case exactly as python/tests/test_golden.py does.
double compute_case(const std::string& name, const mini_json::Value& inputs) {
    if (inputs.has("kind")) return compute_kind(inputs.at("kind").str, inputs);

    SharedData& d = shared();
    const int maxlag = static_cast<int>(d.cfg.at("adf_maxlag"));

    if (starts_with(name, "ols_pair")) {
        const statarb::OLSResult fit = statarb::ols(d.y, d.x, true);
        if (name == "ols_pair_alpha") return fit.params[0];
        if (name == "ols_pair_beta") return fit.params[1];
        if (name == "ols_pair_r2") return fit.r2;
    }
    if (starts_with(name, "adf_random_walk")) {
        const std::vector<double>& col =
            inputs.at("column").str == "walk_a" ? d.walk_a : d.walk_b;
        const statarb::ADFResult res = statarb::adf_test(col, maxlag, 1);
        if (name == "adf_random_walk_tstat") return res.tstat;
        if (name == "adf_random_walk_reject") return res.reject(0.05) ? 1.0 : 0.0;
        if (name == "adf_random_walk_crit5") return res.crit.cv5;
        if (name == "adf_random_walk_pvalue") return res.pvalue;
    }
    if (name == "eg_spurious_coint") {
        const statarb::EGResult eg = statarb::engle_granger(d.walk_a, d.walk_b, maxlag);
        return eg.cointegrated(0.05) ? 1.0 : 0.0;
    }
    double v12 = 0.0;
    if (compute_v12(name, inputs, v12)) return v12;
    if (name == "adf_resid_insample_tstat") return d.eg_insample.adf.tstat;
    if (name == "adf_resid_insample_crit5") return d.eg_insample.adf.crit.cv5;
    if (name == "eg_pair_insample_coint") {
        return d.eg_insample.cointegrated(0.05) ? 1.0 : 0.0;
    }
    if (name == "halflife_insample_spread") {
        return statarb::half_life(d.eg_insample.resid);
    }
    if (starts_with(name, "johansen_pair")) {
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
    if (starts_with(name, "kalman_beta")) {
        const statarb::KalmanResult km = statarb::kalman_hedge(
            d.y, d.x, d.cfg.at("kalman_delta"), d.cfg.at("kalman_r"));
        return km.beta[static_cast<std::size_t>(inputs.at("day").number)];
    }
    if (starts_with(name, "zscore_day")) {
        const std::vector<double> z = statarb::rolling_zscore(
            statarb::spread(d.y, d.x, d.eg_insample.alpha, d.eg_insample.beta),
            static_cast<int>(d.cfg.at("window")));
        return z[static_cast<std::size_t>(inputs.at("day").number)];
    }
    if (starts_with(name, "backtest_")) {
        return metric(d.backtest("limits_beta").metrics, name.substr(std::string("backtest_").size()));
    }
    throw std::runtime_error("unknown golden case: " + name);
}

TEST(Golden, AllCasesWithinTolerance) {
    const mini_json::Value doc =
        mini_json::parse_file(kDataDir + "/golden/golden.json");
    const mini_json::Value& cases = doc.at("cases");
    ASSERT_EQ(cases.arr.size(), kGoldenCount);
    std::size_t checked = 0;
    for (const mini_json::Value& c : cases.arr) {
        const std::string name = c.at("name").str;
        SCOPED_TRACE("golden case: " + name);
        const double expect = c.at("expect").at("value").number;
        const double tol = c.at("tol").number;
        const double got = compute_case(name, c.at("inputs"));
        EXPECT_NEAR(got, expect, tol);
        ++checked;
    }
    EXPECT_EQ(checked, kGoldenCount);
}

TEST(Golden, SchemaFlatScalars) {
    const mini_json::Value doc =
        mini_json::parse_file(kDataDir + "/golden/golden.json");
    const mini_json::Value& cases = doc.at("cases");
    ASSERT_EQ(cases.arr.size(), kGoldenCount);
    std::set<std::string> names;
    for (const mini_json::Value& c : cases.arr) {
        ASSERT_TRUE(names.insert(c.at("name").str).second) << "duplicate name";
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
    // Counts, ranks, days, directions and flags are integers: exact match.
    const mini_json::Value doc =
        mini_json::parse_file(kDataDir + "/golden/golden.json");
    std::size_t n = 0;
    for (const mini_json::Value& c : doc.at("cases").arr) {
        const std::string name = c.at("name").str;
        const mini_json::Value& in = c.at("inputs");
        const std::string field = in.has("field") ? in.at("field").str : "";
        const bool integral =
            name == "backtest_n_trades" || name == "johansen_pair_rank" ||
            field == "n_trades" || field == "n_stops" || field == "n_blocks" ||
            field == "entry" || field == "exit" || field == "direction" ||
            field == "stopped" || field == "start" || field == "end" ||
            field == "skipped" || field == "split" || name == "select_lag_resid" ||
            name == "select_lag_walk";
        if (!integral) continue;
        SCOPED_TRACE(name);
        EXPECT_EQ(compute_case(name, in), c.at("expect").at("value").number);
        ++n;
    }
    EXPECT_GT(n, 100u);
}

TEST(Golden, EveryKindIsDispatched) {
    // Guard against a silently skipped group: count the kinds in the file.
    const mini_json::Value doc =
        mini_json::parse_file(kDataDir + "/golden/golden.json");
    std::map<std::string, int> kinds;
    for (const mini_json::Value& c : doc.at("cases").arr) {
        const mini_json::Value& in = c.at("inputs");
        ++kinds[in.has("kind") ? in.at("kind").str : "(by name)"];
    }
    for (const char* k : {"backtest", "backtest_equity", "backtest_trade", "walk_forward",
                          "rolling", "rolling_block", "margin"}) {
        EXPECT_GT(kinds[k], 0) << k;
    }
}

}  // namespace
