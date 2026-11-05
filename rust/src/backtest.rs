//! Pairs backtest — state machine, risk limits and accounting (PINNED).
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
//! * FLAT, signal day t (and `t >= e + cooldown_days` for the latest
//!   stopped exit executed on day e): `entry_z <= z_t < stop_z` -> pending
//!   SHORT entry; `-stop_z < z_t <= -entry_z` -> pending LONG entry
//!   (`|z_t| >= stop_z` while flat: no entry, pinned; NaN z: no action).
//! * In position, signal day t, first match wins: (1) open trade P&L net
//!   of costs so far `<= -pnl_stop_frac * capital` -> STOP; (2)
//!   `|z_t| >= stop_z` -> STOP; (3) `t - entry_day >= max_hold_days` ->
//!   exit (not a stop); (4) `|z_t| <= exit_z` -> exit. Rules 1 and 3 do not
//!   read z, so they still fire on a NaN z. A limit of 0 is off.
//! * Execution day t+1 close, `dir = +1` for LONG spread (long y / short
//!   x), `-1` for SHORT. [`Sizing::BetaHedged`] (default):
//!   `u = capital / (y + beta*x)`, `q_y = dir*u`, `q_x = -dir*beta*u`
//!   (gross notional is capital at entry, legs in the hedge ratio, not
//!   dollar-neutral). [`Sizing::EqualDollar`]: `q_y = dir*(capital/2)/y`,
//!   `q_x = -dir*(capital/2)/x` (dollar-neutral at entry). Quantities are
//!   held fixed for the life of the trade (no daily rebalance).
//! * Costs are charged on traded notional `|dq_y| y + |dq_x| x` at the
//!   execution close, both entry and exit.
//! * A position still open at the final day is force-closed at the final
//!   close (counted as a closed trade, not a stop). A pending entry at
//!   the end of data is DISCARDED (never executed-and-force-closed at the
//!   same close).
//! * Metrics: `ann_return = (1+total)^(252/N) - 1`, pinned to `-1.0` when
//!   the equity curve ends at or below zero (`total_return <= -1`).
//!
//! The risk limits in `data/config.json` (0.02, 60, 20) are round numbers
//! fixed in advance; they were not tuned on this data.

use std::collections::HashMap;

use crate::error::{Result, StatarbError};
use crate::ols::ols;
use crate::signals::{rolling_zscore, spread};

/// Nudge added before flooring `T * insample_frac` in [`walk_forward`] so
/// the split is the intended integer despite binary-float products
/// (`700.0 * 0.7 == 489.99999999999994`). Pinned.
pub const SPLIT_EPS: f64 = 1.0e-9;

/// Largest |z| a W-point sample can give its own latest point (ddof = 1):
/// `(W-1)/sqrt(W)`.
pub fn max_abs_zscore(window: usize) -> f64 {
    (window as f64 - 1.0) / (window as f64).sqrt()
}

/// Position sizing rule (config `sizing`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Sizing {
    /// `u = capital / (y + beta*x)`, `q_y = dir*u`, `q_x = -dir*beta*u`.
    #[default]
    BetaHedged,
    /// `q_y = dir*(capital/2)/y`, `q_x = -dir*(capital/2)/x`.
    EqualDollar,
}

impl Sizing {
    /// Parse the config name `"beta_hedged"` or `"equal_dollar"`.
    ///
    /// # Errors
    ///
    /// Any other name.
    pub fn from_name(name: &str) -> Result<Self> {
        match name {
            "beta_hedged" => Ok(Sizing::BetaHedged),
            "equal_dollar" => Ok(Sizing::EqualDollar),
            _ => Err(StatarbError::InvalidParam(
                "sizing must be 'beta_hedged' or 'equal_dollar'".to_string(),
            )),
        }
    }

    /// The config name of this rule.
    pub fn name(self) -> &'static str {
        match self {
            Sizing::BetaHedged => "beta_hedged",
            Sizing::EqualDollar => "equal_dollar",
        }
    }
}

/// Signal / cost / risk parameters for the pairs backtest.
///
/// Construct via [`BacktestConfig::new`] (risk limits off, default sizing)
/// or [`BacktestConfig::from_map`], or start from [`Default`] and set
/// fields. [`backtest_pair`], [`walk_forward`] and
/// [`rolling_walk_forward`] re-validate, so mutating a field cannot bypass
/// [`BacktestConfig::validate`].
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
    /// Gross capital: gross notional at every entry.
    pub capital: f64,
    /// Position sizing rule (default [`Sizing::BetaHedged`]).
    pub sizing: Sizing,
    /// Stop when open trade P&L (net of costs so far) is at or below
    /// `-pnl_stop_frac * capital`; 0 = off.
    pub pnl_stop_frac: f64,
    /// Exit when held this many days; 0 = off.
    pub max_hold_days: usize,
    /// No entry signal for this many days after a stopped exit; 0 = off.
    pub cooldown_days: usize,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        BacktestConfig {
            window: 60,
            entry_z: 2.0,
            exit_z: 0.5,
            stop_z: 4.0,
            cost_bps: 1.0,
            slippage_bps: 0.5,
            capital: 1_000_000.0,
            sizing: Sizing::BetaHedged,
            pnl_stop_frac: 0.0,
            max_hold_days: 0,
            cooldown_days: 0,
        }
    }
}

/// A config count (days / window) given as a float: finite integer >= 0.
fn as_count(name: &str, v: f64) -> Result<usize> {
    if v.is_finite() && v >= 0.0 && v.fract() == 0.0 && v <= 9_007_199_254_740_992.0 {
        Ok(v as usize)
    } else {
        Err(StatarbError::InvalidParam(format!(
            "{name} must be an integer >= 0"
        )))
    }
}

impl BacktestConfig {
    /// Validated constructor for the seven signal / cost parameters; risk
    /// limits are off and sizing is the default. Set the other fields on
    /// the returned value (they are public) and call [`Self::validate`], or
    /// use [`Self::from_map`].
    ///
    /// # Errors
    ///
    /// See [`Self::validate`].
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
            ..BacktestConfig::default()
        };
        cfg.validate()?;
        Ok(cfg)
    }

    /// Build a config from a [`crate::load_config`] map; absent keys keep
    /// the defaults (so absent risk limits are off). Reads `window`,
    /// `entry_z`, `exit_z`, `stop_z`, `cost_bps`, `slippage_bps`,
    /// `capital`, `pnl_stop_frac`, `max_hold_days`, `cooldown_days`.
    ///
    /// # Errors
    ///
    /// A count key that is not a non-negative integer, or anything
    /// [`Self::validate`] rejects.
    pub fn from_map(cfg: &HashMap<String, f64>) -> Result<Self> {
        let mut out = BacktestConfig::default();
        let f = |k: &str, dst: &mut f64| {
            if let Some(&v) = cfg.get(k) {
                *dst = v;
            }
        };
        f("entry_z", &mut out.entry_z);
        f("exit_z", &mut out.exit_z);
        f("stop_z", &mut out.stop_z);
        f("cost_bps", &mut out.cost_bps);
        f("slippage_bps", &mut out.slippage_bps);
        f("capital", &mut out.capital);
        f("pnl_stop_frac", &mut out.pnl_stop_frac);
        if let Some(&v) = cfg.get("window") {
            out.window = as_count("window", v)?;
        }
        if let Some(&v) = cfg.get("max_hold_days") {
            out.max_hold_days = as_count("max_hold_days", v)?;
        }
        if let Some(&v) = cfg.get("cooldown_days") {
            out.cooldown_days = as_count("cooldown_days", v)?;
        }
        out.validate()?;
        Ok(out)
    }

    /// Re-check the pinned parameter domain (used by every entry point).
    ///
    /// # Errors
    ///
    /// [`StatarbError::InvalidParam`] on `window < 2`, non-finite
    /// thresholds / costs / `pnl_stop_frac`, a violation of
    /// `0 <= exit_z < entry_z < stop_z`, negative costs or
    /// `pnl_stop_frac`, non-positive / non-finite capital, an unreachable
    /// `entry_z` (`(W-1)/sqrt(W) <= entry_z`) or an unreachable `stop_z`
    /// without `pnl_stop_frac > 0`.
    pub fn validate(&self) -> Result<()> {
        if self.window < 2 {
            return Err(StatarbError::InvalidParam(
                "window must be an integer >= 2".to_string(),
            ));
        }
        if ![
            self.entry_z,
            self.exit_z,
            self.stop_z,
            self.cost_bps,
            self.slippage_bps,
        ]
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
            return Err(StatarbError::InvalidParam("costs must be >= 0".to_string()));
        }
        if !(self.capital > 0.0 && self.capital.is_finite()) {
            return Err(StatarbError::InvalidParam(
                "capital must be positive and finite".to_string(),
            ));
        }
        if !(self.pnl_stop_frac.is_finite() && self.pnl_stop_frac >= 0.0) {
            return Err(StatarbError::InvalidParam(
                "pnl_stop_frac must be finite and >= 0".to_string(),
            ));
        }
        // Dead-config rejection: |z| can never reach (W-1)/sqrt(W).
        let reach = max_abs_zscore(self.window);
        if reach <= self.entry_z {
            return Err(StatarbError::InvalidParam(format!(
                "entry_z unreachable: window {} caps |z| at {reach:.4}",
                self.window
            )));
        }
        if reach <= self.stop_z && self.pnl_stop_frac <= 0.0 {
            return Err(StatarbError::InvalidParam(format!(
                "stop_z unreachable: window {} caps |z| at {reach:.4}; \
                 allowed only with pnl_stop_frac > 0",
                self.window
            )));
        }
        Ok(())
    }
}

/// One closed round trip (days are execution-day indices).
#[derive(Debug, Clone, PartialEq)]
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
        let var = daily_ret
            .iter()
            .map(|r| (r - mean) * (r - mean))
            .sum::<f64>()
            / (n - 1) as f64;
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
///   elsewhere — e.g. in-sample OLS or Engle-Granger). `beta` must be
///   `> 0`: a pair needing a negative hedge is out of scope.
/// * `config` — validated signal / cost / risk parameters.
///
/// # Errors
///
/// Length mismatch, non-finite/non-positive prices, invalid config,
/// `beta <= 0` or NaN, or window larger than the series (via the rolling
/// z-score).
pub fn backtest_pair(
    y: &[f64],
    x: &[f64],
    alpha: f64,
    beta: f64,
    config: &BacktestConfig,
) -> Result<BacktestResult> {
    run_backtest(y, x, alpha, beta, config, 0)
}

/// State machine; entry signals are ignored before `first_entry_signal`.
fn run_backtest(
    y: &[f64],
    x: &[f64],
    alpha: f64,
    beta: f64,
    config: &BacktestConfig,
    first_entry_signal: usize,
) -> Result<BacktestResult> {
    config.validate()?;
    if beta.is_nan() || beta <= 0.0 {
        return Err(StatarbError::InvalidParam(
            "beta must be > 0: negative hedges are out of scope".to_string(),
        ));
    }
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
    let mut no_entry_before = first_entry_signal; // also moved by the cool-down

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
                    let d = direction as f64;
                    match config.sizing {
                        Sizing::BetaHedged => {
                            let u = cap / (y[t] + beta * x[t]);
                            qy = d * u;
                            qx = -d * beta * u;
                        }
                        Sizing::EqualDollar => {
                            qy = d * (cap / 2.0) / y[t];
                            qx = -d * (cap / 2.0) / x[t];
                        }
                    }
                    let traded = qy.abs() * y[t] + qx.abs() * x[t];
                    cost += traded * bps;
                    turnover_notional += traded;
                    state = if direction > 0 {
                        State::Long
                    } else {
                        State::Short
                    };
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
                    if stop_flag {
                        no_entry_before = no_entry_before.max(t + config.cooldown_days);
                    }
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
            let z_ok = !zt.is_nan();
            if state == State::Flat {
                if z_ok && t >= no_entry_before {
                    if config.entry_z <= zt && zt < config.stop_z {
                        pending = Some(Pending::Enter(-1));
                    } else if -config.stop_z < zt && zt <= -config.entry_z {
                        pending = Some(Pending::Enter(1));
                    }
                }
            } else {
                // first match wins: P&L stop, z stop, max hold, z exit
                // (the P&L and hold rules do not read z)
                let pnl_stop =
                    config.pnl_stop_frac > 0.0 && open_pnl <= -config.pnl_stop_frac * cap;
                let held = config.max_hold_days > 0 && t - open_entry_day >= config.max_hold_days;
                if pnl_stop || (z_ok && zt.abs() >= config.stop_z) {
                    pending = Some(Pending::Exit(true));
                } else if held || (z_ok && zt.abs() <= config.exit_z) {
                    pending = Some(Pending::Exit(false));
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

/// SINGLE HOLDOUT (one fit, one split; the name is kept for compatibility,
/// see [`rolling_walk_forward`] for the refitting version).
///
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
/// an invalid config, either segment shorter than the z-score window, or a
/// fitted `beta <= 0`.
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

/// One rolling test block: days `[start, end)` traded with a fit from
/// before `start`. `result` covers the block days only; its trades carry
/// absolute day indices. `skipped` marks a block left flat because the fit
/// gave `beta <= 0`.
#[derive(Debug, Clone)]
pub struct RollingBlock {
    /// First traded day.
    pub start: usize,
    /// One past the last traded day.
    pub end: usize,
    /// Intercept fitted on `[start - train, start)`.
    pub alpha: f64,
    /// Hedge ratio fitted on `[start - train, start)`.
    pub beta: f64,
    /// True when the block stayed flat because `beta <= 0`.
    pub skipped: bool,
    /// Backtest trimmed to the block days.
    pub result: BacktestResult,
}

/// Stitched rolling walk-forward; `equity[i]` is day `start_day + i`.
#[derive(Debug, Clone)]
pub struct RollingWalkForwardResult {
    /// The test blocks in order.
    pub blocks: Vec<RollingBlock>,
    /// First stitched day (`train`).
    pub start_day: usize,
    /// Stitched equity: each block's P&L added to the running equity.
    pub equity: Vec<f64>,
    /// All block trades (absolute days).
    pub trades: Vec<Trade>,
    /// Metrics on the stitched curve; `turnover` is the sum over blocks.
    pub metrics: Metrics,
}

impl RollingWalkForwardResult {
    /// Number of closed trades that exited via stop.
    pub fn n_stops(&self) -> usize {
        self.trades.iter().filter(|t| t.stopped).count()
    }
}

fn flat_block(n: usize, cap: f64) -> Result<BacktestResult> {
    let equity = vec![cap; n];
    let metrics = compute_metrics(&equity, cap, &[])?;
    Ok(BacktestResult {
        equity,
        daily_pnl: vec![0.0; n],
        daily_cost: vec![0.0; n],
        zscore: vec![f64::NAN; n],
        trades: Vec::new(),
        metrics,
    })
}

/// Refit on a rolling window and trade only the days after each fit.
///
/// For `k = 0, 1, ..` while `k*test + train < T`: OLS on
/// `[k*test, k*test + train)`, trade days `[start, end)` with
/// `start = k*test + train`, `end = min(start + test, T)`. Each block is
/// its own backtest on the `W-1` days before it plus the block, so z on the
/// first block day is valid and uses history only. Entry signals start on
/// the first block day, a position is flat at block end (forced close, not
/// a stop) and cool-down does not carry between blocks. A block whose fit
/// gives `beta <= 0` stays flat with `skipped = true`.
///
/// # Errors
///
/// Length mismatch, invalid config, `test < 1`, `train < window`,
/// `train >= T`, or an OLS / backtest error inside a block.
pub fn rolling_walk_forward(
    y: &[f64],
    x: &[f64],
    config: &BacktestConfig,
    train: usize,
    test: usize,
) -> Result<RollingWalkForwardResult> {
    if y.len() != x.len() {
        return Err(StatarbError::LengthMismatch(
            "y and x must be 1-D arrays of equal length".to_string(),
        ));
    }
    config.validate()?;
    if test < 1 {
        return Err(StatarbError::InvalidParam("test must be >= 1".to_string()));
    }
    if train < config.window {
        return Err(StatarbError::InvalidParam(
            "train must cover the z window".to_string(),
        ));
    }
    let n = y.len();
    if train >= n {
        return Err(StatarbError::InvalidParam(
            "train must be shorter than the series".to_string(),
        ));
    }

    let cap = config.capital;
    let warm = config.window - 1;
    let mut blocks: Vec<RollingBlock> = Vec::new();
    let mut trades: Vec<Trade> = Vec::new();
    let mut stitched: Vec<f64> = Vec::with_capacity(n - train);
    let mut running = cap;
    let mut turnover = 0.0_f64;
    let mut k = 0usize;
    while k * test + train < n {
        let fit_lo = k * test;
        let start = fit_lo + train;
        let end = (start + test).min(n);
        let fit = ols(&y[fit_lo..start], &x[fit_lo..start], true)?;
        let (a, b) = (fit.params[0], fit.params[1]);
        let skipped = b.is_nan() || b <= 0.0;
        let res = if skipped {
            flat_block(end - start, cap)?
        } else {
            let lo = start - warm;
            let full = run_backtest(&y[lo..end], &x[lo..end], a, b, config, warm)?;
            let block_trades: Vec<Trade> = full
                .trades
                .iter()
                .map(|t| Trade {
                    entry_day: t.entry_day + lo,
                    exit_day: t.exit_day + lo,
                    direction: t.direction,
                    pnl: t.pnl,
                    stopped: t.stopped,
                })
                .collect();
            let equity = full.equity[warm..].to_vec();
            let mut metrics = compute_metrics(&equity, cap, &block_trades)?;
            metrics.turnover = full.metrics.turnover;
            BacktestResult {
                equity,
                daily_pnl: full.daily_pnl[warm..].to_vec(),
                daily_cost: full.daily_cost[warm..].to_vec(),
                zscore: full.zscore[warm..].to_vec(),
                trades: block_trades,
                metrics,
            }
        };
        trades.extend(res.trades.iter().cloned());
        for &e in &res.equity {
            stitched.push(running + (e - cap));
        }
        running = *stitched.last().unwrap_or(&running);
        turnover += res.metrics.turnover;
        blocks.push(RollingBlock {
            start,
            end,
            alpha: a,
            beta: b,
            skipped,
            result: res,
        });
        k += 1;
    }

    let mut metrics = compute_metrics(&stitched, cap, &trades)?;
    metrics.turnover = turnover;
    Ok(RollingWalkForwardResult {
        blocks,
        start_day: train,
        equity: stitched,
        trades,
        metrics,
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
        let cfg = BacktestConfig::new(2, 0.5, 0.2, 0.6, 0.0, 0.0, 1e6).unwrap();
        let y = vec![1.0, -1.0, 1.0, 1.0];
        let x = vec![1.0, 1.0, 1.0, 1.0];
        assert!(matches!(
            backtest_pair(&y, &x, 0.0, 1.0, &cfg),
            Err(StatarbError::InvalidParam(_))
        ));
    }

    #[test]
    fn sizing_names_round_trip() {
        assert_eq!(
            Sizing::from_name("beta_hedged").unwrap(),
            Sizing::BetaHedged
        );
        assert_eq!(
            Sizing::from_name("equal_dollar").unwrap().name(),
            "equal_dollar"
        );
        assert!(Sizing::from_name("kelly").is_err());
    }
}
