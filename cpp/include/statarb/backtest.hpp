/// \file backtest.hpp
/// \brief Pairs backtest — state machine, sizing, risk limits and accounting
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
///  - In position, first match wins: (1) open trade P&L net of costs so far
///    <= -pnl_stop_frac*capital -> STOP; (2) |z_t| >= stop_z -> STOP;
///    (3) t - entry_day >= max_hold_days -> exit (not a stop); (4)
///    |z_t| <= exit_z -> exit.  Rules (1) and (3) do not read z, so they fire
///    on a NaN z.  A limit of 0 is off.  Cool-down: after a stopped exit
///    executed on day e there is no entry signal while t < e + cooldown_days.
///  - Sizing "beta_hedged" (default): u = capital/(y + beta*x) at execution,
///    q_y = dir*u, q_x = -dir*beta*u (gross = capital at entry, not
///    dollar-neutral).  "equal_dollar": q_y = dir*(capital/2)/y,
///    q_x = -dir*(capital/2)/x.  Quantities are held fixed for the life of
///    the trade (no daily rebalance).  beta <= 0 (or NaN) is an error.
///  - Costs are charged on traded notional at each execution close.
///  - A position still open at the final day is force-closed at the final
///    close (a normal closed trade); a pending entry at the end of data
///    is DISCARDED (never executed-and-force-closed at the same close).
///  - Metrics: ann_return = (1+total)^(252/N) - 1, pinned to -1.0 when the
///    equity curve ends at or below zero (total_return <= -1).
#pragma once

#include <cstddef>
#include <map>
#include <string>
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
    double capital;       ///< Gross capital; gross notional at entry.
    std::string sizing;   ///< "beta_hedged" (default) or "equal_dollar".
    double pnl_stop_frac; ///< Stop when open trade P&L <= -frac*capital; 0 = off.
    int max_hold_days;    ///< Exit when held this many days; 0 = off.
    int cooldown_days;    ///< No entry signal for this many days after a stopped exit; 0 = off.

    /// Construct and validate.
    /// \throws std::invalid_argument when window < 2, any threshold or cost
    ///         is NaN/inf, the ordering 0 <= exit_z < entry_z < stop_z is
    ///         violated, costs are negative, capital is not positive and
    ///         finite, sizing is unknown, a risk limit is negative or
    ///         non-finite, entry_z is unreachable ((W-1)/sqrt(W) <= entry_z),
    ///         or stop_z is unreachable with pnl_stop_frac == 0.
    explicit BacktestConfig(int window_ = 60, double entry_z_ = 2.0,
                            double exit_z_ = 0.5, double stop_z_ = 4.0,
                            double cost_bps_ = 1.0, double slippage_bps_ = 0.5,
                            double capital_ = 1.0e6,
                            std::string sizing_ = "beta_hedged",
                            double pnl_stop_frac_ = 0.0, int max_hold_days_ = 0,
                            int cooldown_days_ = 0);

    /// Re-check the pinned parameter domain (the constructor's checks).
    /// \throws std::invalid_argument on any violation.
    void validate() const;
};

/// Build a config from a load_config map (absent limits = off, sizing default).
/// \throws std::invalid_argument for a non-integral window / day field.
BacktestConfig config_from_map(const std::map<std::string, double>& cfg);

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

/// SINGLE HOLDOUT (one fit, one split; the name is kept for compatibility;
/// see rolling_walk_forward for the refitting version).
///
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

/// One test block of rolling_walk_forward: days [start, end) traded with a fit
/// from before start.  `result` covers the block days only; its trades carry
/// absolute days.  `skipped` marks a block left flat because the fit gave
/// beta <= 0 (or NaN).
struct RollingBlock {
    int start = 0;
    int end = 0;
    double alpha = 0.0;
    double beta = 0.0;
    bool skipped = false;
    BacktestResult result;
};

/// Stitched rolling walk-forward; equity[i] is day start_day + i.
struct RollingWalkForwardResult {
    std::vector<RollingBlock> blocks;
    int start_day = 0;
    std::vector<double> equity;
    std::vector<Trade> trades;
    Metrics metrics;

    int n_stops() const;
};

/// Refit on a rolling window and trade only the days after each fit.
///
/// For k = 0, 1, ...: OLS on [k*test, k*test + train), trade days
/// [start, end) with start = k*test + train, end = min(start + test, T).
/// Each block is its own backtest on the W-1 days before it plus the block,
/// so z on the first block day is valid and uses history only.  Entry
/// signals start on the first block day, a position is flat at block end
/// (forced close, not a stop) and cool-down does not carry between blocks.
/// A block with beta <= 0 stays flat (skipped).  Equity is stitched by
/// adding each block's P&L to the running equity.
///
/// \throws std::invalid_argument on length mismatch, test < 1,
///         train < window, train >= T, or an invalid config.
RollingWalkForwardResult rolling_walk_forward(const std::vector<double>& y,
                                              const std::vector<double>& x,
                                              const BacktestConfig& config,
                                              int train, int test);

}  // namespace statarb

