#include "statarb/backtest.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "statarb/linalg.hpp"
#include "statarb/ols.hpp"
#include "statarb/signals.hpp"

namespace statarb {

namespace {
constexpr int kFlat = 0;
constexpr int kLong = 1;
constexpr int kShort = -1;

enum class Action { kNone, kEnter, kExit };
}  // namespace

BacktestConfig::BacktestConfig(int window_, double entry_z_, double exit_z_,
                               double stop_z_, double cost_bps_,
                               double slippage_bps_, double capital_)
    : window(window_), entry_z(entry_z_), exit_z(exit_z_), stop_z(stop_z_),
      cost_bps(cost_bps_), slippage_bps(slippage_bps_), capital(capital_) {
    validate();
}

void BacktestConfig::validate() const {
    if (window < 2) {
        throw std::invalid_argument("window must be an integer >= 2");
    }
    if (!(std::isfinite(entry_z) && std::isfinite(exit_z) && std::isfinite(stop_z) &&
          std::isfinite(cost_bps) && std::isfinite(slippage_bps))) {
        throw std::invalid_argument("thresholds and costs must be finite");
    }
    if (!(0.0 <= exit_z && exit_z < entry_z && entry_z < stop_z)) {
        throw std::invalid_argument("require 0 <= exit_z < entry_z < stop_z");
    }
    if (cost_bps < 0.0 || slippage_bps < 0.0) {
        throw std::invalid_argument("costs must be >= 0");
    }
    if (!(capital > 0.0) || !std::isfinite(capital)) {
        throw std::invalid_argument("capital must be positive and finite");
    }
}

int BacktestResult::n_stops() const {
    int n = 0;
    for (const Trade& t : trades) {
        if (t.stopped) ++n;
    }
    return n;
}

Metrics compute_metrics(const std::vector<double>& equity, double capital,
                        const std::vector<Trade>& trades) {
    Metrics m;
    const std::size_t n = equity.size();
    if (n == 0) {
        throw std::invalid_argument("equity curve must be a non-empty 1-D array");
    }
    if (!all_finite(equity)) {
        throw std::invalid_argument("equity curve must be finite (no NaN/inf)");
    }
    if (!(capital > 0.0) || !std::isfinite(capital)) {
        throw std::invalid_argument("capital must be positive and finite");
    }
    m.total_return = (equity.back() - capital) / capital;
    // (1 + total)^(252/N) is undefined for equity <= 0: pin the total loss
    // of capital (and worse) to an annualised return of exactly -1.
    m.ann_return =
        m.total_return <= -1.0
            ? -1.0
            : std::pow(1.0 + m.total_return, 252.0 / static_cast<double>(n)) - 1.0;

    // daily_ret_t = (E_t - E_{t-1}) / capital with E_{-1} = capital.
    std::vector<double> daily_ret(n);
    for (std::size_t t = 0; t < n; ++t) {
        const double prev = t == 0 ? capital : equity[t - 1];
        daily_ret[t] = (equity[t] - prev) / capital;
    }
    if (n >= 2) {
        const double mean = compensated_sum(daily_ret) / static_cast<double>(n);
        NeumaierSum acc;
        for (double v : daily_ret) acc.add((v - mean) * (v - mean));
        const double sd = std::sqrt(acc.value() / static_cast<double>(n - 1));
        m.sharpe = sd > 0.0 ? mean / sd * std::sqrt(252.0) : 0.0;
    } else {
        m.sharpe = 0.0;
    }

    double peak = equity[0];
    double max_dd = 0.0;
    for (std::size_t t = 0; t < n; ++t) {
        peak = std::max(peak, equity[t]);
        max_dd = std::max(max_dd, (peak - equity[t]) / peak);
    }
    m.max_drawdown = max_dd;

    const std::size_t n_trades = trades.size();
    m.n_trades = static_cast<double>(n_trades);
    if (n_trades > 0) {
        std::size_t wins = 0;
        long holding = 0;
        for (const Trade& t : trades) {
            if (t.pnl > 0.0) ++wins;
            holding += t.holding_days();
        }
        m.win_rate = static_cast<double>(wins) / static_cast<double>(n_trades);
        m.avg_holding_days =
            static_cast<double>(holding) / static_cast<double>(n_trades);
    }
    return m;
}

BacktestResult backtest_pair(const std::vector<double>& y,
                             const std::vector<double>& x,
                             double alpha, double beta,
                             const BacktestConfig& config) {
    config.validate();  // public members may have been mutated since construction
    if (y.size() != x.size() || y.empty()) {
        throw std::invalid_argument("y and x must be 1-D arrays of equal length");
    }
    if (!all_finite(y) || !all_finite(x)) {
        throw std::invalid_argument("prices must be finite (no NaN/inf)");
    }
    for (std::size_t t = 0; t < y.size(); ++t) {
        if (y[t] <= 0.0 || x[t] <= 0.0) {
            throw std::invalid_argument("prices must be strictly positive");
        }
    }

    const std::vector<double> s = spread(y, x, alpha, beta);
    const std::vector<double> z = rolling_zscore(s, config.window);
    const double bps = (config.cost_bps + config.slippage_bps) / 1.0e4;
    const double cap = config.capital;

    const std::size_t t_len = y.size();
    BacktestResult res;
    res.equity.resize(t_len);
    res.daily_pnl.assign(t_len, 0.0);
    res.daily_cost.assign(t_len, 0.0);
    res.zscore = z;

    int state = kFlat;
    double qy = 0.0, qx = 0.0;
    Action pending = Action::kNone;
    int pend_dir = 0;
    bool pend_stop = false;
    int open_entry_day = -1;
    int open_dir = 0;
    double open_pnl = 0.0;
    double turnover_notional = 0.0;
    double eq = cap;

    for (std::size_t t = 0; t < t_len; ++t) {
        // 1) mark-to-market the position held since the day t-1 close
        const double pnl =
            t > 0 ? qy * (y[t] - y[t - 1]) + qx * (x[t] - x[t - 1]) : 0.0;
        if (state != kFlat) open_pnl += pnl;
        double cost = 0.0;

        // 2) execute the order decided at the day t-1 close.  A pending
        //    ENTRY on the final day is discarded (pinned): it could only be
        //    force-closed at the same close, a phantom zero-day round trip.
        if (pending != Action::kNone) {
            if (pending == Action::kEnter && state == kFlat && t + 1 < t_len) {
                qy = pend_dir * (cap / 2.0) / y[t];
                qx = -pend_dir * (cap / 2.0) / x[t];
                const double traded = std::fabs(qy) * y[t] + std::fabs(qx) * x[t];
                cost += traded * bps;
                turnover_notional += traded;
                state = pend_dir > 0 ? kLong : kShort;
                open_entry_day = static_cast<int>(t);
                open_dir = pend_dir;
                open_pnl = -traded * bps;
            } else if (pending == Action::kExit && state != kFlat) {
                const double traded = std::fabs(qy) * y[t] + std::fabs(qx) * x[t];
                cost += traded * bps;
                turnover_notional += traded;
                open_pnl -= traded * bps;
                res.trades.push_back(Trade{open_entry_day, static_cast<int>(t),
                                           open_dir, open_pnl, pend_stop});
                qy = qx = 0.0;
                state = kFlat;
            }
            pending = Action::kNone;
        }

        // 2b) force-close a position still open at the final close
        if (t == t_len - 1 && state != kFlat) {
            const double traded = std::fabs(qy) * y[t] + std::fabs(qx) * x[t];
            cost += traded * bps;
            turnover_notional += traded;
            open_pnl -= traded * bps;
            res.trades.push_back(Trade{open_entry_day, static_cast<int>(t),
                                       open_dir, open_pnl, false});
            qy = qx = 0.0;
            state = kFlat;
        }

        eq += pnl - cost;
        res.equity[t] = eq;
        res.daily_pnl[t] = pnl;
        res.daily_cost[t] = cost;

        // 3) generate the order to execute at the day t+1 close
        if (t < t_len - 1) {
            const double zt = z[t];
            if (!std::isnan(zt)) {
                if (state == kFlat) {
                    if (config.entry_z <= zt && zt < config.stop_z) {
                        pending = Action::kEnter;
                        pend_dir = -1;
                        pend_stop = false;
                    } else if (-config.stop_z < zt && zt <= -config.entry_z) {
                        pending = Action::kEnter;
                        pend_dir = +1;
                        pend_stop = false;
                    }
                } else {
                    if (std::fabs(zt) >= config.stop_z) {
                        pending = Action::kExit;
                        pend_dir = 0;
                        pend_stop = true;
                    } else if (std::fabs(zt) <= config.exit_z) {
                        pending = Action::kExit;
                        pend_dir = 0;
                        pend_stop = false;
                    }
                }
            }
        }
    }

    res.metrics = compute_metrics(res.equity, cap, res.trades);
    res.metrics.turnover = turnover_notional / cap;
    return res;
}

WalkForwardResult walk_forward(const std::vector<double>& y,
                               const std::vector<double>& x,
                               const BacktestConfig& config,
                               double insample_frac) {
    if (!(insample_frac > 0.0 && insample_frac < 1.0)) {
        throw std::invalid_argument("insample_frac must lie strictly inside (0, 1)");
    }
    if (y.size() != x.size()) {
        throw std::invalid_argument("y and x must be 1-D arrays of equal length");
    }
    config.validate();
    const std::size_t split = static_cast<std::size_t>(std::floor(
        static_cast<double>(y.size()) * insample_frac + kSplitEps));
    const std::size_t w = static_cast<std::size_t>(std::max(config.window, 0));
    if (split < w || y.size() - split < w) {
        throw std::invalid_argument("each walk-forward segment must cover the z window");
    }
    const std::vector<double> y_in(y.begin(), y.begin() + static_cast<long>(split));
    const std::vector<double> x_in(x.begin(), x.begin() + static_cast<long>(split));
    const std::vector<double> y_out(y.begin() + static_cast<long>(split), y.end());
    const std::vector<double> x_out(x.begin() + static_cast<long>(split), x.end());

    const OLSResult fit = ols(y_in, x_in, /*intercept=*/true);
    WalkForwardResult res;
    res.alpha = fit.params[0];
    res.beta = fit.params[1];
    res.split = split;
    res.insample = backtest_pair(y_in, x_in, res.alpha, res.beta, config);
    res.oos = backtest_pair(y_out, x_out, res.alpha, res.beta, config);
    return res;
}

}  // namespace statarb
