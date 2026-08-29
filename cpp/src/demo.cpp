/// \file demo.cpp
/// \brief statarb end-to-end demo (mirror of python/demo.py).
///
/// Sections:
///   1. Cointegration diagnostics on the bundled equity pair (EG + Johansen).
///   2. Kalman vs static OLS beta tracking through the day-1050 regime break.
///   3. Full backtest report: in-sample vs walk-forward out-of-sample.
///   4. Spurious-regression teaching case (independent random walks).
///
/// Run from cpp/:  ./build/statarb_demo  [data_dir]   (default ../data)
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "statarb/adf.hpp"
#include "statarb/backtest.hpp"
#include "statarb/data_io.hpp"
#include "statarb/engle_granger.hpp"
#include "statarb/johansen.hpp"
#include "statarb/kalman.hpp"
#include "statarb/ols.hpp"
#include "statarb/signals.hpp"

namespace {

void rule(const std::string& title) {
    std::printf("\n=== %s ", title.c_str());
    const int dashes = 66 - static_cast<int>(title.size());
    for (int i = 0; i < dashes; ++i) std::printf("=");
    std::printf("\n");
}

void print_metrics(const char* label, const statarb::BacktestResult& res) {
    const statarb::Metrics& m = res.metrics;
    std::printf(
        "  %-12s ret %7.2f%%  ann %7.2f%%  sharpe %6.2f  maxDD %5.2f%%  "
        "trades %3d  win %4.1f%%  hold %5.1fd  turn %5.1fx  stops %d\n",
        label, m.total_return * 100.0, m.ann_return * 100.0, m.sharpe,
        m.max_drawdown * 100.0, static_cast<int>(m.n_trades),
        m.win_rate * 100.0, m.avg_holding_days, m.turnover, res.n_stops());
}

}  // namespace

int main(int argc, char** argv) {
    const std::string data_dir = argc > 1 ? argv[1] : "../data";
    const auto cfg = statarb::load_config(data_dir + "/config.json");
    const auto pair = statarb::load_csv(data_dir + "/pair_equity.csv");
    const auto fx = statarb::load_csv(data_dir + "/fx_triangle.csv");
    const auto walks = statarb::load_csv(data_dir + "/random_walks.csv");
    const std::vector<double>& y = pair.at("stock_y");
    const std::vector<double>& x = pair.at("stock_x");
    const std::size_t n_days = y.size();
    const int n_is = static_cast<int>(static_cast<double>(n_days) *
                                      cfg.at("insample_frac"));
    const int maxlag = static_cast<int>(cfg.at("adf_maxlag"));
    const statarb::BacktestConfig bt_cfg(
        static_cast<int>(cfg.at("window")), cfg.at("entry_z"), cfg.at("exit_z"),
        cfg.at("stop_z"), cfg.at("cost_bps"), cfg.at("slippage_bps"),
        cfg.at("capital"));

    std::printf("statarb demo — cointegration, Kalman hedging, pairs backtest\n");
    std::printf("pair_equity.csv: %zu days, in-sample = first %d days "
                "(true beta 1.7, break at day 1050)\n", n_days, n_is);

    rule("1. Cointegration diagnostics (in-sample, days 0.." +
         std::to_string(n_is - 1) + ")");
    const std::vector<double> y_in(y.begin(), y.begin() + n_is);
    const std::vector<double> x_in(x.begin(), x.begin() + n_is);
    const statarb::EGResult eg = statarb::engle_granger(y_in, x_in, maxlag);
    std::printf("  EG step 1  y = %.4f + %.4f x   (R^2 %.4f)\n", eg.alpha,
                eg.beta, eg.r2);
    std::printf("  EG step 2  ADF(resid) t = %.4f  crit 5%% (2-var) = %.2f  "
                "p ~ %.4f\n", eg.adf.tstat, eg.adf.crit.cv5, eg.adf.pvalue);
    std::printf("  verdict    cointegrated at 5%%: %s\n",
                eg.cointegrated(0.05) ? "true" : "false");
    std::printf("  half-life  %.2f days\n", statarb::half_life(eg.resid));

    statarb::Matrix lv(static_cast<std::size_t>(n_is), 2);
    for (int i = 0; i < n_is; ++i) {
        lv(static_cast<std::size_t>(i), 0) = y[static_cast<std::size_t>(i)];
        lv(static_cast<std::size_t>(i), 1) = x[static_cast<std::size_t>(i)];
    }
    const statarb::JohansenResult jo = statarb::johansen_trace(lv);
    for (std::size_t r = 0; r < 2; ++r) {
        std::printf("  Johansen   trace(r=%zu) = %8.3f %c cv95 %6.2f\n", r,
                    jo.trace_stats[r],
                    jo.trace_stats[r] > jo.crit_95[r] ? '>' : '<',
                    jo.crit_95[r]);
    }
    std::printf("  Johansen   rank = %d, cointegrating vector (1, %.4f)  "
                "[true beta 1.7]\n", jo.rank, jo.beta(1, 0));

    const std::vector<double>& eurusd = fx.at("eurusd");
    const std::vector<double>& gbpusd = fx.at("gbpusd");
    const std::vector<double>& eurgbp = fx.at("eurgbp");
    statarb::Matrix fx_logs(eurusd.size(), 3);
    for (std::size_t i = 0; i < eurusd.size(); ++i) {
        fx_logs(i, 0) = std::log(eurusd[i]);
        fx_logs(i, 1) = std::log(gbpusd[i]);
        fx_logs(i, 2) = std::log(eurgbp[i]);
    }
    const statarb::JohansenResult jfx = statarb::johansen_trace(fx_logs);
    std::printf("  FX triangle (3 assets, logs): rank = %d, leading vector "
                "(%.2f, %.2f, %.2f)  [triangle: (1, -1, -1)]\n", jfx.rank,
                jfx.beta(0, 0), jfx.beta(1, 0), jfx.beta(2, 0));

    rule("2. Kalman vs static OLS through the regime break");
    const statarb::KalmanResult km =
        statarb::kalman_hedge(y, x, cfg.at("kalman_delta"), cfg.at("kalman_r"));
    const double b_ols = statarb::ols(y, x, true).params[1];
    std::vector<double> true_beta(n_days, 1.7);
    for (std::size_t d = 1050; d <= 1080 && d < n_days; ++d) {
        true_beta[d] = 1.7 + (1.1 - 1.7) * static_cast<double>(d - 1050) / 30.0;
    }
    for (std::size_t d = 1081; d < n_days; ++d) true_beta[d] = 1.1;
    std::printf("  day    true beta   kalman beta   static OLS beta\n");
    for (std::size_t d : {std::size_t{250}, std::size_t{500}, std::size_t{900},
                          std::size_t{1049}, std::size_t{1100},
                          std::size_t{1200}, std::size_t{1499}}) {
        std::printf("  %5zu   %8.3f   %10.4f   %12.4f\n", d, true_beta[d],
                    km.beta[d], b_ols);
    }
    long double se_k = 0.0L, se_o = 0.0L;
    const std::size_t start = 300;
    for (std::size_t d = start; d < n_days; ++d) {
        const long double dk = km.beta[d] - true_beta[d];
        const long double dob = b_ols - true_beta[d];
        se_k += dk * dk;
        se_o += dob * dob;
    }
    const double denom = static_cast<double>(n_days - start);
    std::printf("  beta tracking RMSE (day 300+): kalman %.4f vs static OLS "
                "%.4f -> kalman adapts, OLS cannot\n",
                std::sqrt(static_cast<double>(se_k) / denom),
                std::sqrt(static_cast<double>(se_o) / denom));

    rule("3. Pairs backtest — in-sample vs walk-forward out-of-sample");
    const statarb::WalkForwardResult wf =
        statarb::walk_forward(y, x, bt_cfg, cfg.at("insample_frac"));
    std::printf("  hedge estimated on days 0..%zu: alpha %.4f, beta %.4f\n",
                wf.split - 1, wf.alpha, wf.beta);
    print_metrics("in-sample", wf.insample);
    print_metrics("OOS (40%)", wf.oos);
    std::printf("  degradation is honest: the OOS window contains the day-1050\n");
    std::printf("  break; stop-losses cap the damage of the broken relationship.\n");

    rule("4. Spurious-regression teaching case (independent random walks)");
    const std::vector<double>& wa = walks.at("walk_a");
    const std::vector<double>& wb = walks.at("walk_b");
    const statarb::ADFResult a1 = statarb::adf_test(wa, maxlag, 1);
    const statarb::EGResult sp = statarb::engle_granger(wa, wb, maxlag);
    const double slope_t = statarb::ols(wa, wb, true).tstats[1];
    std::printf("  regressing walk_a on walk_b: slope t-stat = %.1f —\n", slope_t);
    std::printf("  'significant' by the usual +/-2 rule, yet the walks are "
                "independent.\n");
    std::printf("  ADF(walk_a) t = %.3f (unit root not rejected) and\n", a1.tstat);
    std::printf("  ADF(resid) t = %.3f vs 5%% crit %.2f -> cointegrated: %s\n",
                sp.adf.tstat, sp.adf.crit.cv5,
                sp.cointegrated(0.05) ? "true" : "false");
    std::printf("  verdict: spurious relationship — NO backtest is run on this "
                "pair.\n");

    std::printf("\ndemo complete.\n");
    return 0;
}
