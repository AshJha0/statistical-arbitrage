/// \file backtest.hpp
/// \brief Dollar-neutral pairs backtest — state machine and accounting
///        (PINNED).
///
/// Timing convention (pinned, no look-ahead): the signal computed from
/// the day-t CLOSE is executed at the day t+1 CLOSE at day t+1 prices.
/// Daily P&L on day t marks the position held since the day t-1 close:
///
///     pnl_t  = q_y (y_t - y_{t-1}) + q_x (x_t - x_{t-1})
///     cost_t = (cost_bps + slippage_bps)/1e4 * traded_notional_t
///     E_t    = E_{t-1} + pnl_t - cost_t,    E_{-1} := capital
///
/// so E_{T-1} - capital == sum(pnl) - sum(costs) to rounding (tested).
///
/// State machine (FLAT / LONG spread / SHORT spread, at most one pending
/// order; NaN z: no action):
///  - FLAT:  entry_z <= z_t <  stop_z  -> pending SHORT entry;
///          -stop_z  <  z_t <= -entry_z -> pending LONG entry;
///           |z_t| >= stop_z while flat -> NO entry (pinned).
///  - In position: |z_t| >= stop_z -> pending STOP exit (flagged);
///    else |z_t| <= exit_z -> pending normal exit.
///  - Entry legs are EQUAL-DOLLAR (pinned): q_y = dir*(capital/2)/y,
///    q_x = -dir*(capital/2)/x — exactly dollar-neutral at entry; the
///    hedge ratio enters through the spread/signal only.  Quantities are
///    held fixed for the life of the trade (no daily rebalance).
///  - Costs are charged on traded notional at each execution close.
///  - A position still open at the final day is force-closed at the final
///    close (a normal closed trade); a pending entry at the end of data
///    is DISCARDED (never executed-and-force-closed at the same close).
///  - Metrics: ann_return = (1+total)^(252/N) - 1, pinned to -1.0 when the
///    equity curve ends at or below zero (total_return <= -1).
#pragma once

#include <cstddef>
#include <vector>

namespace statarb {

/// Nudge added before flooring T * insample_frac in walk_forward.  Pinned.
inline constexpr double kSplitEps = 1.0e-9;

/// Signal / cost parameters for the pairs backtest (validated on
/// construction AND re-validated by backtest_pair / walk_forward, so
/// mutating the public members afterwards cannot bypass the checks).
struct BacktestConfig {
    int window;           ///< Rolling z-score window W (>= 2).
    double entry_z;       ///< Enter when |z| >= entry_z (and < stop_z).
    double exit_z;        ///< Exit when |z| <= exit_z.
    double stop_z;        ///< Stop-loss when |z| >= stop_z.
    double cost_bps;      ///< Commission, bps of traded notional.
    double slippage_bps;  ///< Slippage, bps of traded notional.
    double capital;       ///< Gross capital; each leg is capital/2 at entry.

    /// Construct and validate.
    /// \throws std::invalid_argument when window < 2, any threshold or cost
    ///         is NaN/inf, the ordering 0 <= exit_z < entry_z < stop_z is
    ///         violated, costs are negative, or capital is not positive
    ///         and finite.
    explicit BacktestConfig(int window_ = 60, double entry_z_ = 2.0,
                            double exit_z_ = 0.5, double stop_z_ = 4.0,
                            double cost_bps_ = 1.0, double slippage_bps_ = 0.5,
                            double capital_ = 1.0e6);

    /// Re-check the pinned parameter domain (the constructor's checks).
    /// \throws std::invalid_argument on any violation.
    void validate() const;
};

/// One closed round trip (days are execution-day indices).
struct Trade {
    int entry_day = 0;
    int exit_day = 0;
    int direction = 0;     ///< +1 long spread, -1 short spread.
    double pnl = 0.0;      ///< Net of entry and exit costs.
    bool stopped = false;  ///< True when closed by the stop rule.

    int holding_days() const { return exit_day - entry_day; }
};

/// Pinned metric set (N = number of days).
struct Metrics {
    double total_return = 0.0;      ///< (E_{N-1} - capital) / capital.
    double ann_return = 0.0;        ///< (1+total_return)^(252/N) - 1.
    double sharpe = 0.0;            ///< mean/std(daily ret, ddof=1)*sqrt(252).
    double max_drawdown = 0.0;      ///< max (peak - E)/peak, positive fraction.
    double n_trades = 0.0;          ///< Closed round trips.
    double win_rate = 0.0;          ///< wins / n_trades (0 if none).
    double avg_holding_days = 0.0;  ///< mean(exit - entry) (0 if none).
    double turnover = 0.0;          ///< Total traded notional / capital.
};

/// Full backtest output: paths, trades, and pinned metrics.
struct BacktestResult {
    std::vector<double> equity;      ///< Equity curve E_t.
    std::vector<double> daily_pnl;   ///< Mark-to-market P&L per day.
    std::vector<double> daily_cost;  ///< Execution costs per day.
    std::vector<double> zscore;      ///< Signal z-score path (NaN warm-up).
    std::vector<Trade> trades;       ///< Closed round trips in order.
    Metrics metrics;                 ///< Pinned metric set.

    /// Number of stop-loss exits among the closed trades.
    int n_stops() const;
};

/// Compute the pinned metric set from an equity curve and trade list
/// (turnover is filled in by backtest_pair, not here).
/// \throws std::invalid_argument on an empty or non-finite equity curve
///         or a non-positive / non-finite capital.
Metrics compute_metrics(const std::vector<double>& equity, double capital,
                        const std::vector<Trade>& trades);

/// Run the pinned pairs state machine over aligned price series.
///
/// \param y,x Close price series, equal length, strictly positive.
/// \param alpha,beta Hedge parameters defining the spread (estimated
///        elsewhere — e.g. in-sample OLS or Engle-Granger).
/// \param config Validated signal/cost parameters.
/// \throws std::invalid_argument on an invalid config (re-validated),
///         length mismatch, non-finite or non-positive prices, or window
///         larger than the series (via rolling_zscore).
BacktestResult backtest_pair(const std::vector<double>& y,
                             const std::vector<double>& x,
                             double alpha, double beta,
                             const BacktestConfig& config);

/// In-sample vs out-of-sample comparison with a single hedge estimate.
struct WalkForwardResult {
    double alpha = 0.0;       ///< In-sample OLS intercept.
    double beta = 0.0;        ///< In-sample OLS hedge ratio.
    BacktestResult insample;  ///< Backtest on days 0..split-1.
    BacktestResult oos;       ///< Backtest on days split..T-1.
    std::size_t split = 0;    ///< First out-of-sample day index.
};

/// Estimate the hedge on the first `insample_frac` of days by OLS,
/// backtest that segment in-sample, then trade the remainder
/// out-of-sample with the SAME (alpha, beta).  The OOS z-score warms up
/// inside the OOS segment only — no information crosses the split.
///
/// `split = floor(T * insample_frac + kSplitEps)` (pinned): the nudge
/// removes binary-float artefacts such as 700 * 0.7 == 489.99999999999994.
///
/// \throws std::invalid_argument when insample_frac lies outside (0, 1),
///         the series differ in length, the config is invalid, or either
///         segment is shorter than the z-score window.
WalkForwardResult walk_forward(const std::vector<double>& y,
                               const std::vector<double>& x,
                               const BacktestConfig& config,
                               double insample_frac = 0.6);

}  // namespace statarb
