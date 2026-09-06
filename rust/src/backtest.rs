//! Dollar-neutral pairs backtest — state machine and accounting (PINNED).
//!
//! Timing convention (pinned, no look-ahead): the signal computed from
//! the day-t CLOSE is executed at the day t+1 CLOSE at day t+1 prices.
//! Daily P&L on day t marks the position held since the day t-1 close:
//!
//! ```text
//! pnl_t  = q_y * (y_t - y_{t-1}) + q_x * (x_t - x_{t-1})
//! cost_t = (cost_bps + slippage_bps) / 1e4 * traded_notional_t
//! E_t    = E_{t-1} + pnl_t - cost_t,    E_{-1} := capital
//! ```
//!
//! so the accounting identity `E_{T-1} - capital == sum(pnl) - sum(costs)`
//! holds to rounding (tested at 1e-10 relative to capital).
//!
//! State machine (states FLAT / LONG spread / SHORT spread, plus at most
//! one pending order):
//!
//! * FLAT, signal day t: `entry_z <= z_t < stop_z` -> pending SHORT
//!   entry; `-stop_z < z_t <= -entry_z` -> pending LONG entry
//!   (`|z_t| >= stop_z` while flat: no entry, pinned; NaN z: no action).
//! * In position, signal day t: `|z_t| >= stop_z` -> pending STOP exit;
//!   else `|z_t| <= exit_z` -> pending normal exit.
//! * Execution day t+1 close. Entry legs are EQUAL-DOLLAR (pinned):
//!   `q_y = dir * (capital/2) / y`, `q_x = -dir * (capital/2) / x` with
//!   `dir = +1` for LONG spread (long y / short x), `-1` for SHORT. The
//!   hedge ratio enters through the spread/signal; sizing is equal-dollar
//!   so the position is exactly dollar-neutral at entry. Quantities are
//!   held fixed for the life of the trade (no daily rebalance).
//! * Costs are charged on traded notional `|dq_y| y + |dq_x| x` at the
//!   execution close, both entry and exit (= capital per side here).
//! * A position still open at the final day is force-closed at the final
//!   close (counted as a closed trade, not a stop). A pending entry at
//!   the end of data is DISCARDED (never executed-and-force-closed at the
//!   same close).
//! * Metrics: `ann_return = (1+total)^(252/N) - 1`, pinned to `-1.0` when
//!   the equity curve ends at or below zero (`total_return <= -1`).

use crate::error::{Result, StatarbError};
use crate::ols::ols;
use crate::signals::{rolling_zscore, spread};

/// Nudge added before flooring `T * insample_frac` in [`walk_forward`] so
/// the split is the intended integer despite binary-float products
/// (`700.0 * 0.7 == 489.99999999999994`). Pinned.
pub const SPLIT_EPS: f64 = 1.0e-9;

/// Signal / cost parameters for the pairs backtest.
///
/// Construct via [`BacktestConfig::new`], which enforces the pinned
/// validation (`window >= 2`, `0 <= exit_z < entry_z < stop_z`, costs
/// `>= 0`, `capital > 0` finite); [`backtest_pair`] re-validates.
#[derive(Debug, Clone)]
pub struct BacktestConfig {
    /// Rolling z-score window W (>= 2).
    pub window: usize,
    /// Enter when `|z| >= entry_z` (and `< stop_z`).
    pub entry_z: f64,
    /// Exit when `|z| <= exit_z`.
    pub exit_z: f64,
    /// Stop-loss when `|z| >= stop_z`.
    pub stop_z: f64,
    /// Commission, basis points of traded notional.
    pub cost_bps: f64,
    /// Slippage, basis points of traded notional.
    pub slippage_bps: f64,
    /// Gross capital; each leg is sized to `capital/2` at entry.
    pub capital: f64,
}

impl BacktestConfig {
    /// Validated constructor (pinned parameter domain).
    ///
    /// # Errors
    ///
    /// [`StatarbError::InvalidParam`] on `window < 2`, NaN/inf thresholds
    /// or costs, threshold ordering violation, negative costs, or
    /// non-positive/non-finite capital.
    pub fn new(
        window: usize,
        entry_z: f64,
        exit_z: f64,
        stop_z: f64,
        cost_bps: f64,
        slippage_bps: f64,
        capital: f64,
    ) -> Result<Self> {
        let cfg = BacktestConfig {
            window,
            entry_z,
            exit_z,
            stop_z,
            cost_bps,
            slippage_bps,
            capital,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    /// Re-check the pinned parameter domain (used by [`backtest_pair`]).
    pub fn validate(&self) -> Result<()> {
        if self.window < 2 {
            return Err(StatarbError::InvalidParam(
                "window must be an integer >= 2".to_string(),
            ));
        }
        if ![self.entry_z, self.exit_z, self.stop_z, self.cost_bps, self.slippage_bps]
            .iter()
            .all(|v| v.is_finite())
        {
            return Err(StatarbError::InvalidParam(
                "thresholds and costs must be finite".to_string(),
            ));
        }
        if !(0.0 <= self.exit_z && self.exit_z < self.entry_z && self.entry_z < self.stop_z) {
            return Err(StatarbError::InvalidParam(
                "require 0 <= exit_z < entry_z < stop_z".to_string(),
            ));
        }
        if self.cost_bps < 0.0 || self.slippage_bps < 0.0 {
            return Err(StatarbError::InvalidParam(
                "costs must be >= 0".to_string(),
            ));
        }
        if !(self.capital > 0.0 && self.capital.is_finite()) {
            return Err(StatarbError::InvalidParam(
                "capital must be positive and finite".to_string(),
            ));
        }
        Ok(())
    }
}

/// One closed round trip (days are execution-day indices).
#[derive(Debug, Clone)]
pub struct Trade {
    /// Day index of the entry execution.
    pub entry_day: usize,
    /// Day index of the exit execution.
    pub exit_day: usize,
    /// +1 long spread, -1 short spread.
    pub direction: i32,
    /// Net trade P&L (accrued daily pnl minus entry and exit costs).
    pub pnl: f64,
    /// True when the exit was a stop-loss.
    pub stopped: bool,
}

impl Trade {
    /// Holding period in days (`exit_day - entry_day`).
    pub fn holding_days(&self) -> usize {
        self.exit_day - self.entry_day
    }
}

/// Pinned performance metrics (N = number of days):
///
/// ```text
/// daily_ret_t      = (E_t - E_{t-1}) / capital
/// total_return     = (E_{N-1} - capital) / capital
/// ann_return       = (1 + total_return)^(252/N) - 1
/// sharpe           = mean(daily_ret) / std(daily_ret, ddof=1) * sqrt(252)
///                    (0.0 when std == 0 or N < 2)
/// max_drawdown     = max_t (peak_t - E_t) / peak_t   (positive fraction)
/// win_rate         = wins / n_trades  (win: net trade pnl > 0; 0 if none)
/// avg_holding_days = mean(exit_day - entry_day)      (0 if no trades)
/// turnover         = total traded notional / capital
/// ```
#[derive(Debug, Clone, Default)]
pub struct Metrics {
    /// `(E_{N-1} - capital) / capital`.
    pub total_return: f64,
    /// `(1 + total_return)^(252/N) - 1`.
    pub ann_return: f64,
    /// Annualised daily Sharpe ratio (rf = 0).
    pub sharpe: f64,
    /// Maximum peak-to-trough drawdown, positive fraction.
    pub max_drawdown: f64,
    /// Closed round trips (force-close counts; discarded pendings do not).
    pub n_trades: usize,
    /// Fraction of closed trades with positive net P&L.
    pub win_rate: f64,
    /// Mean holding period of closed trades, days.
    pub avg_holding_days: f64,
    /// Total traded notional over gross capital.
    pub turnover: f64,
}

/// Full backtest output: paths, trades, and pinned metrics.
#[derive(Debug, Clone)]
pub struct BacktestResult {
    /// Equity curve `E_t`, length T.
    pub equity: Vec<f64>,
    /// Daily mark-to-market P&L.
    pub daily_pnl: Vec<f64>,
    /// Daily execution costs.
    pub daily_cost: Vec<f64>,
    /// The rolling z-score signal used.
    pub zscore: Vec<f64>,
    /// Closed round trips in chronological order.
    pub trades: Vec<Trade>,
    /// Pinned metric set.
    pub metrics: Metrics,
}

impl BacktestResult {
    /// Number of closed trades that exited via stop-loss.
    pub fn n_stops(&self) -> usize {
        self.trades.iter().filter(|t| t.stopped).count()
    }
}

/// Order decided at the previous close, to execute at the current close.
#[derive(Debug, Clone, Copy)]
enum Pending {
    /// Enter with direction +1 (long spread) or -1 (short spread).
    Enter(i32),
    /// Exit; the flag marks a stop-loss exit.
    Exit(bool),
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum State {
    Flat,
    Long,
    Short,
}

/// Compute the pinned metric set from an equity curve and trade list
/// (`turnover` is filled in by [`backtest_pair`], not here).
///
/// # Errors
///
/// [`StatarbError::InvalidParam`] on an empty equity curve or a
/// non-positive/non-finite capital; [`StatarbError::NonFinite`] on a
/// non-finite equity value.
pub fn compute_metrics(equity: &[f64], capital: f64, trades: &[Trade]) -> Result<Metrics> {
    let n = equity.len();
    if n == 0 {
        return Err(StatarbError::InvalidParam(
            "equity curve must be a non-empty 1-D array".to_string(),
        ));
    }
    if !equity.iter().all(|v| v.is_finite()) {
        return Err(StatarbError::NonFinite(
            "equity curve must be finite (no NaN/inf)".to_string(),
        ));
    }
    if !(capital > 0.0 && capital.is_finite()) {
        return Err(StatarbError::InvalidParam(
            "capital must be positive and finite".to_string(),
        ));
    }
    let total_return = (equity[n - 1] - capital) / capital;
    // (1 + total)^(252/N) is undefined for equity <= 0: pin the total loss
    // of capital (and worse) to an annualised return of exactly -1.
    let ann_return = if total_return <= -1.0 {
        -1.0
    } else {
        (1.0 + total_return).powf(252.0 / n as f64) - 1.0
    };
    let daily_ret: Vec<f64> = (0..n)
        .map(|t| {
            let prev = if t == 0 { capital } else { equity[t - 1] };
            (equity[t] - prev) / capital
        })
        .collect();
    let sharpe = if n >= 2 {
        let mean = daily_ret.iter().sum::<f64>() / n as f64;
        let var = daily_ret.iter().map(|r| (r - mean) * (r - mean)).sum::<f64>() / (n - 1) as f64;
        let sd = var.sqrt();
        if sd > 0.0 {
            mean / sd * 252.0_f64.sqrt()
        } else {
            0.0
        }
    } else {
        0.0
    };
    let mut peak = f64::NEG_INFINITY;
    let mut max_dd = f64::NEG_INFINITY;
    for &e in equity {
        peak = peak.max(e);
        max_dd = max_dd.max((peak - e) / peak);
    }
    let n_trades = trades.len();
    let wins = trades.iter().filter(|t| t.pnl > 0.0).count();
    Ok(Metrics {
        total_return,
        ann_return,
        sharpe,
        max_drawdown: max_dd,
        n_trades,
        win_rate: if n_trades > 0 {
            wins as f64 / n_trades as f64
        } else {
            0.0
        },
        avg_holding_days: if n_trades > 0 {
            trades.iter().map(|t| t.holding_days()).sum::<usize>() as f64 / n_trades as f64
        } else {
            0.0
        },
        turnover: 0.0, // filled by the caller
    })
}

/// Run the pinned pairs state machine over aligned price series.
///
/// * `y`, `x` — close price series, equal length, strictly positive.
/// * `alpha`, `beta` — hedge parameters defining the spread (estimated
///   elsewhere — e.g. in-sample OLS or Engle-Granger).
/// * `config` — validated signal/cost parameters.
///
/// # Errors
///
/// Length mismatch, non-finite/non-positive prices, invalid config, or
/// window larger than the series (via the rolling z-score).
pub fn backtest_pair(
    y: &[f64],
    x: &[f64],
    alpha: f64,
    beta: f64,
    config: &BacktestConfig,
) -> Result<BacktestResult> {
    config.validate()?;
    if y.len() != x.len() {
        return Err(StatarbError::LengthMismatch(
            "y and x must be 1-D arrays of equal length".to_string(),
        ));
    }
    if !y.iter().chain(x.iter()).all(|v| v.is_finite()) {
        return Err(StatarbError::NonFinite(
            "prices must be finite (no NaN/inf)".to_string(),
        ));
    }
    if y.iter().chain(x.iter()).any(|&v| v <= 0.0) {
        return Err(StatarbError::InvalidParam(
            "prices must be strictly positive".to_string(),
        ));
    }

    let s = spread(y, x, alpha, beta)?;
    let z = rolling_zscore(&s, config.window)?;
    let bps = (config.cost_bps + config.slippage_bps) / 1.0e4;
    let cap = config.capital;

    let t_len = y.len();
    let mut equity = vec![0.0; t_len];
    let mut daily_pnl = vec![0.0; t_len];
    let mut daily_cost = vec![0.0; t_len];
    let mut trades: Vec<Trade> = Vec::new();

    let mut state = State::Flat;
    let (mut qy, mut qx) = (0.0_f64, 0.0_f64);
    let mut pending: Option<Pending> = None;
    let mut open_entry_day = 0usize;
    let mut open_dir = 0i32;
    let mut open_pnl = 0.0_f64;
    let mut turnover_notional = 0.0_f64;
    let mut eq = cap;

    for t in 0..t_len {
        // 1) mark-to-market the position held since the day t-1 close
        let pnl = if t > 0 {
            qy * (y[t] - y[t - 1]) + qx * (x[t] - x[t - 1])
        } else {
            0.0
        };
        if state != State::Flat {
            open_pnl += pnl;
        }
        let mut cost = 0.0;

        // 2) execute the order decided at the day t-1 close.  A pending
        //    ENTRY on the final day is discarded (pinned): it could only be
        //    force-closed at the same close, a phantom zero-day round trip.
        if let Some(order) = pending {
            match order {
                Pending::Enter(direction) if state == State::Flat && t + 1 < t_len => {
                    qy = direction as f64 * (cap / 2.0) / y[t];
                    qx = -(direction as f64) * (cap / 2.0) / x[t];
                    let traded = qy.abs() * y[t] + qx.abs() * x[t];
                    cost += traded * bps;
                    turnover_notional += traded;
                    state = if direction > 0 { State::Long } else { State::Short };
                    open_entry_day = t;
                    open_dir = direction;
                    open_pnl = -traded * bps;
                }
                Pending::Exit(stop_flag) if state != State::Flat => {
                    let traded = qy.abs() * y[t] + qx.abs() * x[t];
                    cost += traded * bps;
                    turnover_notional += traded;
                    open_pnl -= traded * bps;
                    trades.push(Trade {
                        entry_day: open_entry_day,
                        exit_day: t,
                        direction: open_dir,
                        pnl: open_pnl,
                        stopped: stop_flag,
                    });
                    qy = 0.0;
                    qx = 0.0;
                    state = State::Flat;
                }
                _ => {}
            }
            pending = None;
        }

        // 2b) force-close a position still open at the final close
        if t == t_len - 1 && state != State::Flat {
            let traded = qy.abs() * y[t] + qx.abs() * x[t];
            cost += traded * bps;
            turnover_notional += traded;
            open_pnl -= traded * bps;
            trades.push(Trade {
                entry_day: open_entry_day,
                exit_day: t,
                direction: open_dir,
                pnl: open_pnl,
                stopped: false,
            });
            qy = 0.0;
            qx = 0.0;
            state = State::Flat;
        }

        eq += pnl - cost;
        equity[t] = eq;
        daily_pnl[t] = pnl;
        daily_cost[t] = cost;

        // 3) generate the order to execute at the day t+1 close
        if t < t_len - 1 {
            let zt = z[t];
            if !zt.is_nan() {
                match state {
                    State::Flat => {
                        if config.entry_z <= zt && zt < config.stop_z {
                            pending = Some(Pending::Enter(-1));
                        } else if -config.stop_z < zt && zt <= -config.entry_z {
                            pending = Some(Pending::Enter(1));
                        }
                    }
                    _ => {
                        if zt.abs() >= config.stop_z {
                            pending = Some(Pending::Exit(true));
                        } else if zt.abs() <= config.exit_z {
                            pending = Some(Pending::Exit(false));
                        }
                    }
                }
            }
        }
    }

    let mut metrics = compute_metrics(&equity, cap, &trades)?;
    metrics.turnover = turnover_notional / cap;
    Ok(BacktestResult {
        equity,
        daily_pnl,
        daily_cost,
        zscore: z,
        trades,
        metrics,
    })
}

/// In-sample vs out-of-sample comparison with a single hedge estimate.
#[derive(Debug, Clone)]
pub struct WalkForwardResult {
    /// In-sample OLS intercept.
    pub alpha: f64,
    /// In-sample OLS hedge ratio.
    pub beta: f64,
    /// Backtest on days `0..split-1`.
    pub insample: BacktestResult,
    /// Backtest on days `split..T-1` with the SAME hedge parameters.
    pub oos: BacktestResult,
    /// First out-of-sample day index, `floor(T * insample_frac)`.
    pub split: usize,
}

/// Estimate the hedge on the first `insample_frac` of days by OLS,
/// backtest that segment in-sample, then trade the remainder
/// out-of-sample with the SAME `(alpha, beta)` — degradation on display.
///
/// The out-of-sample z-score warms up inside the OOS segment only — no
/// information crosses the split.
///
/// `split = floor(T * insample_frac + SPLIT_EPS)` (pinned).
///
/// # Errors
///
/// `insample_frac` outside (0, 1), a length mismatch between `y` and `x`,
/// an invalid config, or either segment shorter than the z-score window.
pub fn walk_forward(
    y: &[f64],
    x: &[f64],
    config: &BacktestConfig,
    insample_frac: f64,
) -> Result<WalkForwardResult> {
    if !(insample_frac > 0.0 && insample_frac < 1.0) {
        return Err(StatarbError::InvalidParam(
            "insample_frac must lie strictly inside (0, 1)".to_string(),
        ));
    }
    if y.len() != x.len() {
        return Err(StatarbError::LengthMismatch(
            "y and x must be 1-D arrays of equal length".to_string(),
        ));
    }
    config.validate()?;
    let split = (y.len() as f64 * insample_frac + SPLIT_EPS).floor() as usize;
    if split < config.window || y.len() - split < config.window {
        return Err(StatarbError::TooShort(
            "each walk-forward segment must cover the z window".to_string(),
        ));
    }
    let fit = ols(&y[..split], &x[..split], true)?;
    let (a, b) = (fit.params[0], fit.params[1]);
    let insample = backtest_pair(&y[..split], &x[..split], a, b, config)?;
    let oos = backtest_pair(&y[split..], &x[split..], a, b, config)?;
    Ok(WalkForwardResult {
        alpha: a,
        beta: b,
        insample,
        oos,
        split,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_validation() {
        assert!(BacktestConfig::new(60, 2.0, 0.5, 4.0, 1.0, 0.5, 1e6).is_ok());
        assert!(BacktestConfig::new(1, 2.0, 0.5, 4.0, 1.0, 0.5, 1e6).is_err());
        assert!(BacktestConfig::new(60, 2.0, 2.5, 4.0, 1.0, 0.5, 1e6).is_err());
        assert!(BacktestConfig::new(60, 2.0, 0.5, 1.5, 1.0, 0.5, 1e6).is_err());
        assert!(BacktestConfig::new(60, 2.0, 0.5, 4.0, -1.0, 0.5, 1e6).is_err());
        assert!(BacktestConfig::new(60, 2.0, 0.5, 4.0, 1.0, 0.5, 0.0).is_err());
    }

    #[test]
    fn nonpositive_prices_rejected() {
        let cfg = BacktestConfig::new(2, 2.0, 0.5, 4.0, 0.0, 0.0, 1e6).unwrap();
        let y = vec![1.0, -1.0, 1.0, 1.0];
        let x = vec![1.0, 1.0, 1.0, 1.0];
        assert!(matches!(
            backtest_pair(&y, &x, 0.0, 1.0, &cfg),
            Err(StatarbError::InvalidParam(_))
        ));
    }
}
