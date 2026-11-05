//! v1.2 contract (API_SPEC sections 2, 4, 6-8, 10): beta check, sizing,
//! risk limits, dead configs, pinned z sums, rolling walk-forward,
//! streaming API, lag selection, two-way Engle-Granger and strict loaders.
//! Mirrors python/tests/test_v12.py and the v1.2 loader tests.

#![allow(clippy::needless_range_loop)] // small fixed matrices: index form mirrors the maths

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use statarb::{
    adf_test, backtest_pair, engle_granger, engle_granger_both, kalman_hedge, load_config,
    load_csv, rolling_walk_forward, rolling_zscore, select_lag, spread, walk_forward,
    BacktestConfig, KalmanStep, RollingZ, Sizing, StatarbError, Trade,
};

const CAP: f64 = 1_000_000.0;

fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../data")
}

/// Deterministic xorshift64* normal generator (no RNG crate needed).
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let v = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D);
        ((v >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        let (u1, u2) = (self.uniform(), self.uniform());
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// Prices whose spread at alpha = 50, beta = 1 is exactly `s`.
fn scripted(s: &[f64], x0: f64) -> (Vec<f64>, Vec<f64>) {
    let xs = vec![x0; s.len()];
    let y = s.iter().map(|v| 50.0 + x0 + v).collect();
    (y, xs)
}

fn sine(n: usize, amp: f64) -> Vec<f64> {
    (0..n).map(|t| amp * (0.7 * t as f64).sin()).collect()
}

/// (y, x, alpha, beta): trending x, AR(0.9) spread, true hedge (60, 1.5).
fn noise(t_len: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = Rng::new(seed);
    let mut lvl = 0.0;
    let x: Vec<f64> = (0..t_len)
        .map(|_| {
            lvl += 0.01 * rng.normal();
            100.0 * lvl.exp()
        })
        .collect();
    let mut s = vec![0.0; t_len];
    for t in 1..t_len {
        s[t] = 0.9 * s[t - 1] + rng.normal();
    }
    let y = (0..t_len)
        .map(|t| 20.0 + 1.5 * x[t] + s[t] + 40.0)
        .collect();
    (y, x)
}

fn cfg(window: usize) -> BacktestConfig {
    BacktestConfig {
        window,
        ..BacktestConfig::default()
    }
}

/// (y, x, alpha, beta, config map) with the hedge from EG on days 0..900.
fn bundled() -> (Vec<f64>, Vec<f64>, f64, f64, HashMap<String, f64>) {
    let config = load_config(&data_dir().join("config.json")).unwrap();
    let pair = load_csv(&data_dir().join("pair_equity.csv")).unwrap();
    let (y, x) = (pair["stock_y"].clone(), pair["stock_x"].clone());
    let eg = engle_granger(&y[..900], &x[..900], config["adf_maxlag"] as usize).unwrap();
    (y, x, eg.alpha, eg.beta, config)
}

fn tup(t: &Trade) -> (usize, usize, i32, bool) {
    (t.entry_day, t.exit_day, t.direction, t.stopped)
}

// ---- items 1 and 2: beta and sizing ---------------------------------------

#[test]
fn backtest_rejects_non_positive_beta() {
    let (y, x) = noise(100, 3);
    for beta in [0.0, -1.0, f64::NAN] {
        let r = backtest_pair(&y, &x, 60.0, beta, &BacktestConfig::default());
        assert!(
            matches!(&r, Err(StatarbError::InvalidParam(m)) if m.contains("beta")),
            "{beta}"
        );
    }
}

#[test]
fn sizing_default_and_names() {
    assert_eq!(BacktestConfig::default().sizing, Sizing::BetaHedged);
    assert!(Sizing::from_name("kelly").is_err());
    assert_eq!(
        Sizing::from_name("equal_dollar").unwrap(),
        Sizing::EqualDollar
    );
}

#[test]
fn gross_equals_capital_at_every_entry() {
    let (y, x, a, b, config) = bundled();
    for sizing in [Sizing::BetaHedged, Sizing::EqualDollar] {
        let mut c = BacktestConfig::from_map(&config).unwrap();
        c.sizing = sizing;
        let res = backtest_pair(&y, &x, a, b, &c).unwrap();
        let bps = (c.cost_bps + c.slippage_bps) / 1e4;
        assert!(!res.trades.is_empty());
        for tr in &res.trades {
            let got = res.daily_cost[tr.entry_day];
            assert!((got - CAP * bps).abs() <= 1e-12 * CAP * bps, "{sizing:?}");
        }
    }
}

#[test]
fn beta_hedged_leg_ratio_and_equal_dollar_neutral() {
    // a day-after P&L reveals the quantities
    let mut s = vec![0.0; 120];
    for (t, v) in s.iter_mut().enumerate().take(70) {
        *v = 0.2 * (0.7 * t as f64).sin();
    }
    for v in s.iter_mut().skip(70) {
        *v = 2.0;
    }
    let (mut y, mut x) = scripted(&s, 100.0);
    let beta = 1.0;
    let mut c = cfg(40);
    c.cost_bps = 0.0;
    c.slippage_bps = 0.0;
    let entry = backtest_pair(&y, &x, 50.0, beta, &c).unwrap().trades[0].entry_day;
    for t in entry + 1..120 {
        x[t] += 1.0;
        y[t] += beta; // spread unchanged
    }
    let res = backtest_pair(&y, &x, 50.0, beta, &c).unwrap();
    assert_eq!(res.trades[0].entry_day, entry);
    // hedged at beta: moving along the hedge line earns nothing
    assert!(res.daily_pnl[entry + 1].abs() < 1e-8);

    let mut y2 = y.clone();
    for v in y2.iter_mut().skip(entry + 1) {
        *v -= 1.0; // only x moved; the short-spread position is long x
    }
    c.sizing = Sizing::EqualDollar;
    let res2 = backtest_pair(&y2, &x, 50.0, beta, &c).unwrap();
    let qx = (CAP / 2.0) / x[entry];
    assert_eq!(res2.trades[0].direction, -1);
    assert!((res2.daily_pnl[entry + 1] - qx).abs() <= 1e-12 * qx);
}

// ---- item 3: risk limits ---------------------------------------------------

#[test]
fn accounting_identity_all_sizings_and_limits() {
    let (y, x, a, b, config) = bundled();
    for sizing in [Sizing::BetaHedged, Sizing::EqualDollar] {
        for (frac, hold, cool) in [(0.0, 0, 0), (0.02, 60, 20), (0.005, 5, 3)] {
            let mut c = BacktestConfig::from_map(&config).unwrap();
            c.sizing = sizing;
            c.pnl_stop_frac = frac;
            c.max_hold_days = hold;
            c.cooldown_days = cool;
            let res = backtest_pair(&y, &x, a, b, &c).unwrap();
            let lhs = res.equity[res.equity.len() - 1] - CAP;
            let rhs = res.daily_pnl.iter().sum::<f64>() - res.daily_cost.iter().sum::<f64>();
            assert!((lhs - rhs).abs() < 1e-6, "{sizing:?} {frac} {hold} {cool}");
            let tsum: f64 = res.trades.iter().map(|t| t.pnl).sum();
            assert!((lhs - tsum).abs() < 1e-6, "{sizing:?} {frac} {hold} {cool}");
        }
    }
}

#[test]
fn limit_validation() {
    for bad in [-0.1, f64::NAN, f64::INFINITY] {
        let c = BacktestConfig {
            pnl_stop_frac: bad,
            ..BacktestConfig::default()
        };
        assert!(
            matches!(c.validate(), Err(StatarbError::InvalidParam(_))),
            "{bad}"
        );
    }
    // counts come from a float map: negative and fractional are refused
    for (k, v) in [
        ("max_hold_days", -1.0),
        ("max_hold_days", 1.5),
        ("cooldown_days", -1.0),
        ("cooldown_days", 2.5),
    ] {
        let m = HashMap::from([(k.to_string(), v)]);
        assert!(
            matches!(
                BacktestConfig::from_map(&m),
                Err(StatarbError::InvalidParam(_))
            ),
            "{k} {v}"
        );
    }
    // limits are re-validated at run time
    let (y, x) = noise(100, 3);
    let bad = BacktestConfig {
        pnl_stop_frac: f64::NAN,
        ..BacktestConfig::default()
    };
    assert!(backtest_pair(&y, &x, 60.0, 1.5, &bad).is_err());
}

/// Short-spread entry signal on day 70; the spread then holds 0.6.
fn trade_then_hold(t_len: usize) -> Vec<f64> {
    let mut s = sine(t_len, 0.2);
    for v in s.iter_mut().skip(70) {
        *v = 0.6;
    }
    s
}

#[test]
fn pnl_stop_fires_and_beats_z_exit() {
    // On the day the P&L stop and the normal z exit both match, the trade
    // is flagged stopped: the P&L rule runs first.
    let mut s = sine(140, 0.2);
    s[70] = 0.6;
    s[71] = 0.6;
    for v in s.iter_mut().skip(72) {
        *v = 0.0;
    }
    let (_, mut x) = scripted(&s, 50.0);
    for v in x.iter_mut().skip(72) {
        *v -= 5.0; // hurts the long-x leg of the short-spread position
    }
    let y: Vec<f64> = (0..140).map(|t| 50.0 + x[t] + s[t]).collect();
    let mut base = cfg(40);
    base.cost_bps = 0.0;
    base.slippage_bps = 0.0;
    let z = rolling_zscore(&spread(&y, &x, 50.0, 1.0).unwrap(), 40).unwrap();
    assert!(z[72].abs() <= base.exit_z);
    let plain = backtest_pair(&y, &x, 50.0, 1.0, &base).unwrap();
    assert!(!plain.trades[0].stopped);
    // hedged legs do not lose on a move along the hedge line; equal-dollar do
    base.sizing = Sizing::EqualDollar;
    let unstopped = backtest_pair(&y, &x, 50.0, 1.0, &base).unwrap();
    let loss = unstopped.daily_pnl[72];
    assert!(loss < 0.0);
    base.pnl_stop_frac = -loss / CAP / 2.0;
    let stopped = backtest_pair(&y, &x, 50.0, 1.0, &base).unwrap();
    let tr = &stopped.trades[0];
    assert!(tr.stopped && tr.exit_day == 73, "{tr:?}");
}

#[test]
fn pnl_stop_uses_costs_so_far() {
    // the entry cost alone (1.5 bps of capital) trips a tighter stop
    let (y, x) = scripted(&trade_then_hold(140), 50.0);
    let mut c = cfg(40);
    c.pnl_stop_frac = 1.0e-4;
    let res = backtest_pair(&y, &x, 50.0, 1.0, &c).unwrap();
    let tr = &res.trades[0];
    assert!(tr.stopped);
    assert_eq!(tr.exit_day, tr.entry_day + 1);
}

#[test]
fn pnl_stop_boundary_is_inclusive() {
    let (y, x) = scripted(&trade_then_hold(140), 50.0);
    let mut c = cfg(40);
    c.pnl_stop_frac = 1.5e-4; // = entry cost
    let res = backtest_pair(&y, &x, 50.0, 1.0, &c).unwrap();
    let entry_cost = res.daily_cost[res.trades[0].entry_day];
    assert!((entry_cost - 150.0).abs() <= 1e-12 * 150.0);
    // one ulp may fall either side of the exact tie, so bracket it
    c.pnl_stop_frac = 1.5001e-4;
    assert!(!backtest_pair(&y, &x, 50.0, 1.0, &c).unwrap().trades[0].stopped);
    c.pnl_stop_frac = 1.4999e-4;
    assert!(backtest_pair(&y, &x, 50.0, 1.0, &c).unwrap().trades[0].stopped);
}

#[test]
fn stop_priority_z_stop_outranks_max_hold() {
    let mut s = sine(160, 0.2);
    for v in &mut s[70..73] {
        *v = 0.5;
    }
    s[73] = 5.0;
    let (y, x) = scripted(&s, 50.0);
    let mut c = cfg(40);
    c.max_hold_days = 2;
    let z = rolling_zscore(&s, 40).unwrap();
    assert!(z[73] >= c.stop_z);
    let tr = &backtest_pair(&y, &x, 50.0, 1.0, &c).unwrap().trades[0];
    assert_eq!(tup(tr), (71, 74, -1, true));
}

#[test]
fn max_hold_exits_unflagged_on_time() {
    let (y, x) = scripted(&trade_then_hold(140), 50.0);
    let mut c = cfg(40);
    c.max_hold_days = 3;
    let res = backtest_pair(&y, &x, 50.0, 1.0, &c).unwrap();
    let tr = &res.trades[0];
    // signal when t - entry >= 3, execution one day later
    assert_eq!(tr.exit_day - tr.entry_day, 4);
    assert!(!tr.stopped);
    assert_eq!(res.n_stops(), 0);
}

#[test]
fn limits_fire_on_nan_z() {
    // a flat spread makes z NaN once the window clears; max hold still exits
    let w = 5;
    let mut s = vec![0.0, 0.1, -0.1, 0.05, 0.0, 0.0, 0.3, 0.3];
    s.extend(vec![0.3; 30]);
    let (y, x) = scripted(&s, 50.0);
    let z = rolling_zscore(&s, w).unwrap();
    let mut c = cfg(w);
    c.entry_z = 1.5;
    c.exit_z = 0.1;
    c.stop_z = 1.75;
    c.max_hold_days = 8;
    assert!(z[12].is_nan());
    let tr = &backtest_pair(&y, &x, 50.0, 1.0, &c).unwrap().trades[0];
    assert_eq!((tr.entry_day, tr.exit_day, tr.stopped), (7, 16, false));
}

fn stop_then_signal(gap: usize, cooldown: usize) -> Vec<Trade> {
    // the stop executes on day e = 74; another entry signal fires on e + gap
    let mut s = sine(240, 0.2);
    for v in &mut s[70..73] {
        *v = 0.5;
    }
    s[73] = 5.0;
    s[74 + gap] = 3.0;
    let (y, x) = scripted(&s, 50.0);
    let mut c = cfg(40);
    c.cooldown_days = cooldown;
    backtest_pair(&y, &x, 50.0, 1.0, &c).unwrap().trades
}

#[test]
fn cooldown_timing() {
    let gap = 10;
    let free = stop_then_signal(gap, 0);
    assert!(free[0].stopped && free[0].exit_day == 74);
    assert_eq!(free.len(), 2);
    assert_eq!(free[1].entry_day, 74 + gap + 1);
    // t = e + gap: blocked while t < e + cooldown, allowed at equality
    assert_eq!(stop_then_signal(gap, gap + 1).len(), 1);
    let allowed = stop_then_signal(gap, gap);
    assert_eq!(allowed.len(), 2);
    assert_eq!(allowed[1].entry_day, 74 + gap + 1);
}

#[test]
fn cooldown_not_after_unstopped_exit() {
    let mut s = sine(160, 0.2);
    s[70] = 0.5;
    s[71] = 0.0;
    s[72] = 0.5;
    s[73] = 0.0;
    let (y, x) = scripted(&s, 50.0);
    let mut c = cfg(40);
    c.cooldown_days = 50;
    assert_eq!(
        backtest_pair(&y, &x, 50.0, 1.0, &c).unwrap().trades.len(),
        2
    );
}

#[test]
fn cooldown_and_max_hold_properties_on_bundled() {
    let (y, x, a, b, config) = bundled();
    let c = BacktestConfig::from_map(&config).unwrap();
    assert_eq!(
        (c.pnl_stop_frac, c.max_hold_days, c.cooldown_days),
        (0.02, 60, 20)
    );
    let res = backtest_pair(&y, &x, a, b, &c).unwrap();
    assert!(res.n_stops() >= 1);
    for w in res.trades.windows(2) {
        if w[0].stopped {
            // signal day = entry - 1
            assert!(w[1].entry_day > w[0].exit_day + c.cooldown_days);
        }
    }
    for tr in &res.trades {
        assert!(tr.holding_days() <= c.max_hold_days + 1);
    }
}

// ---- item 4: dead configs --------------------------------------------------

fn msg(r: statarb::Result<()>) -> String {
    r.unwrap_err().to_string()
}

#[test]
fn dead_entry_rejected() {
    let c = BacktestConfig {
        window: 5,
        ..BacktestConfig::default()
    };
    assert!(msg(c.validate()).contains("entry_z unreachable"));
    // (4-1)/2 == 1.5 exactly: still dead, even with a P&L stop
    let c = BacktestConfig {
        window: 4,
        entry_z: 1.5,
        exit_z: 0.5,
        stop_z: 4.0,
        pnl_stop_frac: 0.1,
        ..BacktestConfig::default()
    };
    assert!(msg(c.validate()).contains("entry_z unreachable"));
}

#[test]
fn dead_stop_needs_pnl_stop() {
    let dead = BacktestConfig {
        window: 20,
        stop_z: 8.0,
        ..BacktestConfig::default()
    };
    assert!(msg(dead.validate()).contains("stop_z unreachable"));
    let hold = BacktestConfig {
        max_hold_days: 10,
        ..dead.clone()
    };
    assert!(msg(hold.validate()).contains("stop_z unreachable"));
    let ok = BacktestConfig {
        pnl_stop_frac: 0.01,
        ..dead
    };
    ok.validate().unwrap();
}

// ---- item 5: pinned summation ---------------------------------------------

#[test]
fn rolling_z_is_plain_sequential_sum() {
    let mut rng = Rng::new(1);
    let s: Vec<f64> = (0..300)
        .map(|_| rng.normal() * 10f64.powi((rng.uniform() * 12.0) as i32 - 6))
        .collect();
    let w = 37;
    let z = rolling_zscore(&s, w).unwrap();
    for t in w - 1..300 {
        let mut total = 0.0;
        for e in &s[t + 1 - w..=t] {
            total += e;
        }
        let m = total / w as f64;
        let mut ss = 0.0;
        for e in &s[t + 1 - w..=t] {
            ss += (e - m) * (e - m);
        }
        let expect = (s[t] - m) / (ss / (w - 1) as f64).sqrt();
        assert_eq!(z[t].to_bits(), expect.to_bits(), "t={t}"); // bitwise
    }
}

// ---- item 6: rolling walk-forward -----------------------------------------

fn cfg_small() -> BacktestConfig {
    BacktestConfig {
        window: 30,
        entry_z: 1.5,
        exit_z: 0.3,
        stop_z: 4.0,
        pnl_stop_frac: 0.02,
        max_hold_days: 30,
        cooldown_days: 5,
        ..BacktestConfig::default()
    }
}

#[test]
fn rolling_geometry_and_flat_at_block_end() {
    let (y, x) = noise(1000, 3);
    let rw = rolling_walk_forward(&y, &x, &cfg_small(), 300, 150).unwrap();
    let bounds: Vec<(usize, usize)> = rw.blocks.iter().map(|b| (b.start, b.end)).collect();
    assert_eq!(
        bounds,
        vec![(300, 450), (450, 600), (600, 750), (750, 900), (900, 1000)]
    );
    assert_eq!(rw.start_day, 300);
    assert_eq!(rw.equity.len(), 700);
    assert!(!rw.trades.is_empty());
    for b in &rw.blocks {
        assert_eq!(b.result.equity.len(), b.end - b.start);
        for t in &b.result.trades {
            assert!(b.start < t.entry_day && t.entry_day <= t.exit_day && t.exit_day < b.end);
            assert!(t.exit_day > t.entry_day);
        }
    }
}

#[test]
fn rolling_stitch_and_fit_window() {
    let (y, x) = noise(800, 3);
    let rw = rolling_walk_forward(&y, &x, &cfg_small(), 300, 100).unwrap();
    let pnl: f64 = rw
        .blocks
        .iter()
        .map(|b| b.result.equity.last().unwrap() - CAP)
        .sum();
    assert!((rw.equity.last().unwrap() - CAP - pnl).abs() < 1e-6);
    assert!((rw.metrics.total_return - pnl / CAP).abs() < 1e-12);
    let n_trades: usize = rw.blocks.iter().map(|b| b.result.metrics.n_trades).sum();
    assert_eq!(rw.metrics.n_trades, n_trades);
    let turn: f64 = rw.blocks.iter().map(|b| b.result.metrics.turnover).sum();
    assert!((rw.metrics.turnover - turn).abs() < 1e-12);
    let k = 2;
    let fit = engle_granger(&y[k * 100..k * 100 + 300], &x[k * 100..k * 100 + 300], 1).unwrap();
    assert_eq!(rw.blocks[k].alpha, fit.alpha);
    assert_eq!(rw.blocks[k].beta, fit.beta);
}

#[test]
fn rolling_uses_no_future_data() {
    // Perturbing prices from day c on leaves everything before day c
    // unchanged (the forced close never moves before block end).
    let (y, x) = noise(900, 3);
    let c = cfg_small();
    let base = rolling_walk_forward(&y, &x, &c, 300, 100).unwrap();
    for cut in [450usize, 600, 777] {
        let (mut y2, mut x2) = (y.clone(), x.clone());
        let m = 900 - cut;
        for i in 0..m {
            let f = i as f64 / (m - 1) as f64;
            y2[cut + i] *= 1.0 + 0.05 + 0.45 * f;
            x2[cut + i] *= 1.0 - (0.02 + 0.28 * f);
        }
        let alt = rolling_walk_forward(&y2, &x2, &c, 300, 100).unwrap();
        let n = cut - base.start_day;
        assert_eq!(base.equity[..n], alt.equity[..n]);
        for (b0, b1) in base
            .blocks
            .iter()
            .filter(|b| b.start <= cut)
            .zip(&alt.blocks)
        {
            assert_eq!((b0.alpha, b0.beta), (b1.alpha, b1.beta));
        }
        let before = |r: &statarb::RollingWalkForwardResult| -> Vec<Trade> {
            r.trades
                .iter()
                .filter(|t| t.exit_day < cut)
                .cloned()
                .collect()
        };
        assert_eq!(before(&base), before(&alt));
    }
}

#[test]
fn rolling_first_block_entry_needs_block_signal() {
    let (y, x) = noise(500, 3);
    let rw = rolling_walk_forward(&y, &x, &cfg_small(), 300, 200).unwrap();
    assert!(rw.trades.iter().all(|t| t.entry_day >= 301));
}

#[test]
fn rolling_block_matches_manual_backtest() {
    let (y, x) = noise(600, 8);
    let c = BacktestConfig {
        cooldown_days: 0,
        ..cfg_small()
    };
    let rw = rolling_walk_forward(&y, &x, &c, 300, 300).unwrap();
    let b = &rw.blocks[0];
    let z = rolling_zscore(
        &spread(&y[271..], &x[271..], b.alpha, b.beta).unwrap(),
        c.window,
    )
    .unwrap();
    let got = &b.result.zscore;
    assert_eq!(z[29..].len(), got.len());
    for (a, g) in z[29..].iter().zip(got) {
        assert!(a.to_bits() == g.to_bits() || (a.is_nan() && g.is_nan()));
    }
}

#[test]
fn rolling_negative_fit_block_stays_flat() {
    let t_len = 400;
    let x: Vec<f64> = (0..t_len)
        .map(|t| 100.0 + 0.1 * t as f64 + (t as f64).sin())
        .collect();
    let y: Vec<f64> = (0..t_len)
        .map(|t| 200.0 - 0.1 * t as f64 + (0.3 * t as f64).cos())
        .collect();
    let rw = rolling_walk_forward(&y, &x, &cfg_small(), 200, 100).unwrap();
    assert!(rw.blocks.iter().all(|b| b.skipped && b.beta < 0.0));
    assert!(rw
        .blocks
        .iter()
        .all(|b| b.result.zscore.iter().all(|v| v.is_nan())));
    assert!(rw.trades.is_empty());
    assert!(rw.equity.iter().all(|&e| e == CAP));
    assert_eq!(rw.metrics.turnover, 0.0);
}

#[test]
fn rolling_validation() {
    let (y, x) = noise(1000, 3);
    let c = cfg_small();
    for (train, test) in [(10usize, 50usize), (300, 0), (1000, 50), (2000, 50)] {
        assert!(
            matches!(
                rolling_walk_forward(&y, &x, &c, train, test),
                Err(StatarbError::InvalidParam(_))
            ),
            "{train} {test}"
        );
    }
    let (y, x) = noise(400, 3);
    assert!(matches!(
        rolling_walk_forward(&y, &x[..399], &c, 100, 50),
        Err(StatarbError::LengthMismatch(_))
    ));
}

#[test]
fn walk_forward_still_single_holdout() {
    let (y, x, _, _, config) = bundled();
    let c = BacktestConfig::from_map(&config).unwrap();
    assert_eq!(walk_forward(&y, &x, &c, 0.6).unwrap().split, 900);
}

// ---- item 7: streaming -----------------------------------------------------

#[test]
fn rolling_z_stream_matches_batch() {
    let (y, x, a, b, _) = bundled();
    let s = spread(&y, &x, a, b).unwrap();
    for window in [2usize, 5, 60, 250] {
        let batch = rolling_zscore(&s, window).unwrap();
        let mut rz = RollingZ::new(window).unwrap();
        let mut worst = 0.0_f64;
        for (t, &v) in s.iter().enumerate() {
            let got = rz.update(v).unwrap();
            assert_eq!(got.is_nan(), batch[t].is_nan(), "w={window} t={t}");
            if !got.is_nan() {
                worst = worst.max((got - batch[t]).abs());
            }
        }
        assert!(worst < 1e-9, "w={window}: {worst:e}");
    }
}

#[test]
fn rolling_z_stream_long_run_with_large_level() {
    let mut rng = Rng::new(5);
    let mut level = 1.0e4;
    let s: Vec<f64> = (0..50_000)
        .map(|_| {
            level += rng.normal();
            level
        })
        .collect();
    let batch = rolling_zscore(&s, 60).unwrap();
    let mut rz = RollingZ::new(60).unwrap();
    let mut worst = 0.0_f64;
    for (t, &v) in s.iter().enumerate() {
        let got = rz.update(v).unwrap();
        if !batch[t].is_nan() && !got.is_nan() {
            worst = worst.max((got - batch[t]).abs());
        }
    }
    // running moments drift with the update count: 1e-9 holds at the
    // bundled length (test above), a looser bound after 50,000 updates
    assert!(worst < 1e-7, "{worst:e}");
}

#[test]
fn rolling_z_stream_zero_variance_and_validation() {
    let mut rz = RollingZ::new(3).unwrap();
    for _ in 0..5 {
        assert!(rz.update(1.0).unwrap().is_nan());
    }
    for bad in [0usize, 1] {
        assert!(matches!(
            RollingZ::new(bad),
            Err(StatarbError::InvalidParam(_))
        ));
    }
    assert!(matches!(
        rz.update(f64::NAN),
        Err(StatarbError::NonFinite(_))
    ));
    assert!(rz.update(f64::INFINITY).is_err());
    assert!(rz.update(1.0).unwrap().is_nan()); // the rejected values left no trace
}

#[test]
fn kalman_step_matches_batch() {
    let (y, x, _, _, config) = bundled();
    let (d, r) = (config["kalman_delta"], config["kalman_r"]);
    let km = kalman_hedge(&y, &x, d, r).unwrap();
    let mut ks = KalmanStep::new(d, r).unwrap();
    for t in 0..y.len() {
        let (a, b, z) = ks.update(x[t], y[t]).unwrap();
        assert!((a - km.alpha[t]).abs() < 1e-12 && (b - km.beta[t]).abs() < 1e-12);
        assert!((z - km.z[t]).abs() < 1e-12);
    }
    assert_eq!(ks.covariance(), km.final_cov);
}

#[test]
fn kalman_scalar_recursion_matches_matrix_joseph_form() {
    let (y, x, _, _, config) = bundled();
    let (d, r) = (config["kalman_delta"], config["kalman_r"]);
    let km = kalman_hedge(&y, &x, d, r).unwrap();
    let q = d / (1.0 - d);
    let mut s = [0.0_f64; 2];
    let mut p = [[1e3, 0.0], [0.0, 1e3]];
    for t in 0..y.len() {
        let pp = [[p[0][0] + q, p[0][1]], [p[1][0], p[1][1] + q]];
        let h = [1.0, x[t]];
        let ph = [
            pp[0][0] * h[0] + pp[0][1] * h[1],
            pp[1][0] * h[0] + pp[1][1] * h[1],
        ];
        let sv = h[0] * ph[0] + h[1] * ph[1] + r;
        let k = [ph[0] / sv, ph[1] / sv];
        let e = y[t] - (h[0] * s[0] + h[1] * s[1]);
        s = [s[0] + k[0] * e, s[1] + k[1] * e];
        let a = [
            [1.0 - k[0] * h[0], -k[0] * h[1]],
            [-k[1] * h[0], 1.0 - k[1] * h[1]],
        ];
        let mut n = [[0.0_f64; 2]; 2];
        for i in 0..2 {
            for j in 0..2 {
                let mut v = r * k[i] * k[j];
                for m in 0..2 {
                    for l in 0..2 {
                        v += a[i][m] * pp[m][l] * a[j][l];
                    }
                }
                n[i][j] = v;
            }
        }
        let off = 0.5 * (n[0][1] + n[1][0]);
        p = [[n[0][0], off], [off, n[1][1]]];
        assert!((s[1] - km.beta[t]).abs() < 1e-9 && (s[0] - km.alpha[t]).abs() < 1e-9);
    }
    for i in 0..2 {
        for j in 0..2 {
            assert!(
                (p[i][j] - km.final_cov[i][j]).abs() <= 1e-8 * km.final_cov[i][j].abs().max(1e-12)
            );
        }
    }
}

#[test]
fn kalman_step_validation() {
    for (d, r) in [
        (0.0, 1.0),
        (1.0, 1.0),
        (f64::NAN, 1.0),
        (0.1, 0.0),
        (0.1, f64::INFINITY),
        (0.1, f64::NAN),
    ] {
        assert!(KalmanStep::new(d, r).is_err(), "{d} {r}");
    }
    let mut ks = KalmanStep::new(1e-4, 1e-3).unwrap();
    assert!(ks.update(f64::NAN, 1.0).is_err());
    assert!(ks.update(1.0, f64::INFINITY).is_err());
    // the rejected updates left the prior untouched
    assert_eq!(ks.covariance(), [[1e3, 0.0], [0.0, 1e3]]);
}

// ---- item 8: lag selection and two-way EG ---------------------------------

fn ar2(t_len: usize, seed: u64) -> Vec<f64> {
    let mut rng = Rng::new(seed);
    let mut v = vec![0.0; t_len];
    for t in 2..t_len {
        v[t] = 0.3 * v[t - 1] + 0.5 * v[t - 2] + rng.normal();
    }
    v
}

#[test]
fn select_lag_finds_augmentation() {
    // AR(2) levels = the ADF regression with exactly one lagged difference
    assert_eq!(select_lag(&ar2(800, 2), 6, 1).unwrap(), 1);
    let mut rng = Rng::new(4);
    let white: Vec<f64> = (0..800).map(|_| rng.normal()).collect();
    assert_eq!(select_lag(&white, 6, 1).unwrap(), 0);
}

#[test]
fn select_lag_matches_direct_bic() {
    let v = ar2(300, 9);
    let m = 4usize;
    let dy: Vec<f64> = v.windows(2).map(|w| w[1] - w[0]).collect();
    let lhs = &dy[m..];
    let n = lhs.len() as f64;
    let mut best = (0usize, f64::INFINITY);
    for p in 0..=m {
        let mut cols = vec![v[m..v.len() - 1].to_vec()];
        for i in 1..=p {
            cols.push(dy[m - i..dy.len() - i].to_vec());
        }
        let e = statarb::ols_multi(lhs, &cols, true).unwrap().resid;
        let rss: f64 = e.iter().map(|v| v * v).sum();
        let bic = n * (rss / n).ln() + (2 + p) as f64 * n.ln();
        if bic < best.1 {
            best = (p, bic);
        }
    }
    assert_eq!(select_lag(&v, m, 1).unwrap(), best.0);
}

#[test]
fn select_lag_validation() {
    let v = ar2(100, 2);
    assert!(select_lag(&v, 2, 4).is_err()); // nvars
    assert!(select_lag(&v, 90, 1).is_err()); // too short
    assert!(select_lag(&v, 1001, 1).is_err()); // above MAX_LAG
    assert_eq!(select_lag(&v, 0, 1).unwrap(), 0);
    let lag = select_lag(&v, 3, 1).unwrap();
    adf_test(&v, lag, 1).unwrap();
}

#[test]
fn engle_granger_both_needs_both() {
    let (y, x, _, _, _) = bundled();
    let (y, x) = (&y[..900], &x[..900]);
    let both = engle_granger_both(y, x, 1).unwrap();
    assert_eq!(both.y_on_x.beta, engle_granger(y, x, 1).unwrap().beta);
    assert_eq!(both.x_on_y.beta, engle_granger(x, y, 1).unwrap().beta);
    assert_eq!(
        both.cointegrated(0.05).unwrap(),
        both.y_on_x.cointegrated(0.05).unwrap() && both.x_on_y.cointegrated(0.05).unwrap()
    );
    assert!(both.cointegrated(0.07).is_err());
    let walks = load_csv(&data_dir().join("random_walks.csv")).unwrap();
    let sp = engle_granger_both(&walks["walk_a"], &walks["walk_b"], 1).unwrap();
    assert!(!sp.cointegrated(0.05).unwrap());
}

// ---- item 9: strict loaders ------------------------------------------------

fn temp(name: &str, text: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("statarb_v12_{}_{name}", std::process::id()));
    std::fs::write(&p, text).unwrap();
    p
}

#[test]
fn csv_number_grammar_is_strict() {
    for (i, bad) in [
        "0x10", "1_000", "1f", "1d", "1L", "Infinity", "nan", "+", ".", "1e", "1e+", "--1", "1 2",
        "١٢",
    ]
    .iter()
    .enumerate()
    {
        let p = temp(&format!("g{i}.csv"), &format!("day,a\n0,1\n1,{bad}\n"));
        assert!(matches!(load_csv(&p), Err(StatarbError::Data(_))), "{bad}");
        let _ = std::fs::remove_file(p);
    }
    // overflow is finite-checked after the grammar
    let p = temp("ovf.csv", "day,a\n0,1\n1,1e999\n");
    assert!(matches!(load_csv(&p), Err(StatarbError::Data(m)) if m.contains("non-finite")));
    // accepted forms, with spaces and tabs trimmed
    let p = temp("ok.csv", "day,a\n0, +1.5 \n1,\t.5e1\n2,3.\n3,-2E-1\n");
    assert_eq!(load_csv(&p).unwrap()["a"], vec![1.5, 5.0, 3.0, -0.2]);
}

#[test]
fn csv_day_column_must_be_integer() {
    for bad in ["1.0", "1e0", "+", "0x1"] {
        let p = temp("day.csv", &format!("day,a\n0,1\n{bad},2\n"));
        assert!(matches!(load_csv(&p), Err(StatarbError::Data(_))), "{bad}");
    }
    let p = temp("dayok.csv", "day,a\n+0,1\n1,2\n");
    assert_eq!(load_csv(&p).unwrap()["day"], vec![0.0, 1.0]);
}

fn config_err(name: &str, text: &str) -> bool {
    matches!(load_config(&temp(name, text)), Err(StatarbError::Data(_)))
}

#[test]
fn config_rejects_trailing_content_and_non_integers() {
    assert!(config_err("t1.json", "{\"window\": 60} {\"a\": 1}"));
    assert!(config_err("t2.json", "{\"window\": 60} x"));
    assert!(config_err("t3.json", "{\"window\": 60.0}"));
    assert!(config_err("t4.json", "{\"window\": 6e1}"));
    assert!(config_err("t5.json", "{\"max_hold_days\": 1.5}"));
    assert!(config_err("t6.json", "{\"cooldown_days\": 2.0}"));
    assert!(config_err("t7.json", "{\"adf_maxlag\": 1e0}"));
    assert!(config_err("t8.json", "{\"a\": 9007199254740993}")); // > 2^53, any key
    assert!(config_err("t9.json", "{\"a\": 100000000000000000000}")); // beyond i64
    assert!(config_err("t10.json", "{\"a\": 1e999}"));
    assert!(config_err("t11.json", "{\"a\": Infinity}"));
    let cfg = load_config(&temp(
        "ok.json",
        "{\"window\": 60, \"a\": 1.5, \"b\": 9007199254740992, \"c\": 2.0}",
    ))
    .unwrap();
    assert_eq!((cfg["window"], cfg["a"], cfg["c"]), (60.0, 1.5, 2.0));
    assert_eq!(cfg["b"], 9007199254740992.0);
}

#[test]
fn bundled_config_has_round_risk_limits() {
    let config = load_config(&data_dir().join("config.json")).unwrap();
    assert_eq!(config["pnl_stop_frac"], 0.02);
    assert_eq!(config["max_hold_days"], 60.0);
    assert_eq!(config["cooldown_days"], 20.0);
}
