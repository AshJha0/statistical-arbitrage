//! Integration tests for the pairs backtest: accounting identity, dollar
//! neutrality, stop-losses in the break regime, no-trade guard,
//! walk-forward behaviour, and validation errors.

use std::path::{Path, PathBuf};

use statarb::{
    backtest_pair, engle_granger, load_config, load_csv, walk_forward, BacktestConfig,
    BacktestResult, Sizing, StatarbError,
};

fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../data")
}

fn bundled() -> (Vec<f64>, Vec<f64>, BacktestConfig, f64, f64) {
    let cfg = load_config(&data_dir().join("config.json")).unwrap();
    let pair = load_csv(&data_dir().join("pair_equity.csv")).unwrap();
    let y = pair["stock_y"].clone();
    let x = pair["stock_x"].clone();
    let n_is = (y.len() as f64 * cfg["insample_frac"]) as usize;
    let eg = engle_granger(&y[..n_is], &x[..n_is], cfg["adf_maxlag"] as usize).unwrap();
    let bt_cfg = BacktestConfig::from_map(&cfg).unwrap();
    (y, x, bt_cfg, eg.alpha, eg.beta)
}

fn bundled_run() -> (BacktestResult, BacktestConfig) {
    let (y, x, cfg, a, b) = bundled();
    let res = backtest_pair(&y, &x, a, b, &cfg).unwrap();
    (res, cfg)
}

#[test]
fn accounting_identity_holds() {
    // E_{T-1} - capital == sum(pnl) - sum(costs) at 1e-10 relative to capital
    let (res, cfg) = bundled_run();
    let cap = cfg.capital;
    let sum_pnl: f64 = res.daily_pnl.iter().sum();
    let sum_cost: f64 = res.daily_cost.iter().sum();
    let last = *res.equity.last().unwrap();
    assert!(
        ((last - cap) - (sum_pnl - sum_cost)).abs() <= 1e-10 * cap,
        "accounting identity violated"
    );
    // per-day increments match pnl - cost
    for t in 0..res.equity.len() {
        let prev = if t == 0 { cap } else { res.equity[t - 1] };
        let inc = res.equity[t] - prev;
        assert!((inc - (res.daily_pnl[t] - res.daily_cost[t])).abs() < 1e-8);
    }
}

#[test]
fn dollar_neutral_at_entry() {
    // equal_dollar: each entry trades exactly capital of notional (capital/2
    // per leg), so the entry-day cost equals capital * total bps
    let (y, x, mut cfg, a, b) = bundled();
    cfg.sizing = Sizing::EqualDollar;
    let res = backtest_pair(&y, &x, a, b, &cfg).unwrap();
    let cap = cfg.capital;
    let bps = (cfg.cost_bps + cfg.slippage_bps) / 1e4;
    assert!(!res.trades.is_empty());
    for tr in &res.trades {
        assert!((res.daily_cost[tr.entry_day] - cap * bps).abs() < 1e-6);
        // recompute the pinned legs: net dollar position is exactly zero
        let d = tr.direction as f64;
        let qy = d * (cap / 2.0) / y[tr.entry_day];
        let qx = -d * (cap / 2.0) / x[tr.entry_day];
        assert!((qy * y[tr.entry_day] + qx * x[tr.entry_day]).abs() < 1e-9);
    }
}

#[test]
fn stop_loss_triggers_in_break_regime() {
    let (res, _) = bundled_run();
    let stops: Vec<_> = res.trades.iter().filter(|t| t.stopped).collect();
    assert!(!stops.is_empty(), "break regime must trigger stops");
    assert!(
        stops.iter().any(|t| t.exit_day > 1050),
        "at least one stop fires after the day-1050 break"
    );
}

#[test]
fn trades_well_formed_and_metrics_consistent() {
    let (res, _) = bundled_run();
    let m = &res.metrics;
    for tr in &res.trades {
        assert!(tr.exit_day > tr.entry_day);
        assert!(tr.direction == 1 || tr.direction == -1);
    }
    assert_eq!(m.n_trades, res.trades.len());
    assert!(m.win_rate >= 0.0 && m.win_rate <= 1.0);
    assert!(m.avg_holding_days > 0.0);
    assert!(m.max_drawdown >= 0.0);
    // each round trip trades roughly 2x capital (entry + exit legs)
    let per_trade = m.turnover / m.n_trades as f64;
    assert!(
        per_trade > 1.5 && per_trade < 2.5,
        "turnover/trade {per_trade}"
    );
    assert!(res.n_stops() <= m.n_trades);
}

#[test]
fn no_trades_all_metrics_zero() {
    // a spread that never leaves the exit band: no trades, flat equity
    // (an unreachable entry_z is a dead config and is rejected instead)
    let t_len = 200usize;
    let x = vec![50.0; t_len];
    let y: Vec<f64> = (0..t_len)
        .map(|t| 100.0 + 0.01 * (0.7 * t as f64).sin())
        .collect();
    let cfg = BacktestConfig::new(60, 7.0, 0.5, 7.5, 1.0, 0.5, 1e6).unwrap();
    let (a, b) = (50.0, 1.0);
    let res = backtest_pair(&y, &x, a, b, &cfg).unwrap();
    assert!(res.trades.is_empty());
    let m = &res.metrics;
    assert_eq!(m.total_return, 0.0);
    assert_eq!(m.ann_return, 0.0);
    assert_eq!(m.sharpe, 0.0);
    assert_eq!(m.max_drawdown, 0.0);
    assert_eq!(m.n_trades, 0);
    assert_eq!(m.win_rate, 0.0);
    assert_eq!(m.avg_holding_days, 0.0);
    assert_eq!(m.turnover, 0.0);
    assert!(res.equity.iter().all(|&e| e == 1e6));
}

#[test]
fn walk_forward_split_and_oos_warmup() {
    let (y, x, cfg, _, _) = bundled();
    let wf = walk_forward(&y, &x, &cfg, 0.6).unwrap();
    assert_eq!(wf.split, 900);
    assert_eq!(wf.insample.equity.len(), 900);
    assert_eq!(wf.oos.equity.len(), 600);
    // OOS z-score warms up inside the OOS segment only: NaN for the
    // first window-1 days, so no information crosses the split
    assert!(wf.oos.zscore[..cfg.window - 1].iter().all(|v| v.is_nan()));
    assert!(wf.oos.zscore[cfg.window - 1].is_finite());
    // honest degradation on the bundled data: OOS sharpe below in-sample
    assert!(wf.oos.metrics.sharpe < wf.insample.metrics.sharpe);
}

#[test]
fn walk_forward_validation() {
    let (y, x, cfg, _, _) = bundled();
    assert!(matches!(
        walk_forward(&y, &x, &cfg, 0.0),
        Err(StatarbError::InvalidParam(_))
    ));
    assert!(matches!(
        walk_forward(&y, &x, &cfg, 1.0),
        Err(StatarbError::InvalidParam(_))
    ));
    // OOS segment shorter than the window
    assert!(matches!(
        walk_forward(&y, &x, &cfg, 0.99),
        Err(StatarbError::TooShort(_))
    ));
}

#[test]
fn backtest_input_validation() {
    let (y, x, cfg, a, b) = bundled();
    assert!(matches!(
        backtest_pair(&y[..100], &x[..99], a, b, &cfg),
        Err(StatarbError::LengthMismatch(_))
    ));
    // window larger than the series
    assert!(matches!(
        backtest_pair(&y[..30], &x[..30], a, b, &cfg),
        Err(StatarbError::InvalidParam(_))
    ));
    let mut bad = y.clone();
    bad[10] = f64::NAN;
    assert!(matches!(
        backtest_pair(&bad, &x, a, b, &cfg),
        Err(StatarbError::NonFinite(_))
    ));
}

// ---------------------------------------------------------------------------
// Synthetic-spread fixtures.  Every expected value is derived independently
// of the backtest: z-scores are recomputed by `zpath` (plain rolling mean /
// ddof-1 std) and closed forms are used where a window holds a single spike
// on a zero background: z_spike = (W-1)/sqrt(W), z_next = -1/sqrt(W).
// ---------------------------------------------------------------------------

fn zpath(s: &[f64], w: usize) -> Vec<f64> {
    let mut z = vec![f64::NAN; s.len()];
    for t in (w - 1)..s.len() {
        let win = &s[t + 1 - w..=t];
        let m = win.iter().sum::<f64>() / w as f64;
        let ss: f64 = win.iter().map(|v| (v - m) * (v - m)).sum();
        let sd = (ss / (w - 1) as f64).sqrt();
        if sd > 1e-12 {
            z[t] = (s[t] - m) / sd;
        }
    }
    z
}

fn max_abs_finite(z: &[f64]) -> f64 {
    z.iter()
        .filter(|v| v.is_finite())
        .fold(0.0, |m, v| m.max(v.abs()))
}

#[test]
fn end_of_data_pending_entry_discarded() {
    // proposed test 1 (CRIT-1): |z_{T-2}| in [entry, stop) while flat would
    // execute at T-1 and be force-closed the same close; the spec discards it
    let t_len = 80usize;
    let x: Vec<f64> = (0..t_len).map(|t| 200.0 + 0.5 * (t as f64).sin()).collect();
    let mut s: Vec<f64> = (0..t_len).map(|t| (0.7 * t as f64).sin()).collect();
    s[t_len - 2] = 3.0;
    let y: Vec<f64> = (0..t_len).map(|t| 10.0 + x[t] + s[t]).collect();
    let cfg = BacktestConfig::new(20, 2.0, 0.5, 4.0, 1.0, 0.5, 1e6).unwrap();
    let z = zpath(&s, 20);
    assert!(max_abs_finite(&z[..t_len - 2]) < 2.0);
    assert!(z[t_len - 2] >= 2.0 && z[t_len - 2] < 4.0);
    let res = backtest_pair(&y, &x, 10.0, 1.0, &cfg).unwrap();
    assert!(res.trades.is_empty());
    assert_eq!(res.metrics.n_trades, 0);
    assert_eq!(res.metrics.turnover, 0.0);
    assert!(res.equity.iter().all(|&e| e == 1e6));
    assert!(res.daily_cost.iter().all(|&c| c == 0.0));
}

#[test]
fn zero_day_roundtrip_never_reported() {
    // proposed test 2: over every truncation of the bundled pair no trade may
    // ever have exit_day == entry_day (would have caught CRIT-1)
    let (y, x, cfg, a, b) = bundled();
    for t_len in 70..=y.len() {
        let res = backtest_pair(&y[..t_len], &x[..t_len], a, b, &cfg).unwrap();
        for tr in &res.trades {
            assert!(tr.exit_day > tr.entry_day, "T={t_len}: {tr:?}");
            assert!(tr.exit_day < t_len);
        }
    }
}

#[test]
fn stop_band_while_flat_no_entry() {
    // proposed test 3: a single spike on a zero background gives
    // z = (W-1)/sqrt(W) = 6.166 >= stop_z while flat (no entry, pinned) and
    // z = -1/sqrt(W) the next day: no trades, flat equity
    let (t_len, w, t0) = (120usize, 40usize, 60usize);
    let mut s = vec![0.0; t_len];
    s[t0] = 7.0;
    let x = vec![50.0; t_len];
    let y: Vec<f64> = s.iter().map(|v| 100.0 + v).collect(); // alpha 50, beta 1
    let cfg = BacktestConfig::new(w, 2.0, 0.5, 4.0, 1.0, 0.5, 1e6).unwrap();
    let z = zpath(&s, w);
    assert!((z[t0] - (w as f64 - 1.0) / (w as f64).sqrt()).abs() < 1e-12);
    assert!((z[t0 + 1] + 1.0 / (w as f64).sqrt()).abs() < 1e-12);
    assert!(z[t0] >= cfg.stop_z);
    assert!(z[..t0].iter().all(|v| v.is_nan()));
    let res = backtest_pair(&y, &x, 50.0, 1.0, &cfg).unwrap();
    assert!(res.trades.is_empty());
    assert!(res.equity.iter().all(|&e| e == 1e6));
}

#[test]
fn stop_exit_is_flagged_and_executes_next_day() {
    // proposed test 12: entry band at t0, in-position plateau, |z| >= stop at
    // t1 = t0+3 -> one trade, stopped, exit_day == t1+1, exit-day cost equals
    // bps * exit notional recomputed from the pinned legs
    let (t_len, w, t0) = (160usize, 40usize, 70usize);
    let mut s: Vec<f64> = (0..t_len).map(|t| 0.2 * (0.7 * t as f64).sin()).collect();
    let x: Vec<f64> = (0..t_len).map(|t| 200.0 + 0.5 * (t as f64).sin()).collect();
    s[t0] = 0.5;
    s[t0 + 1] = 0.5;
    s[t0 + 2] = 0.5;
    s[t0 + 3] = 5.0;
    let y: Vec<f64> = (0..t_len).map(|t| 10.0 + x[t] + s[t]).collect();
    let mut cfg = BacktestConfig::new(w, 2.0, 0.5, 4.0, 1.0, 0.5, 1e6).unwrap();
    cfg.sizing = Sizing::EqualDollar;
    let z = zpath(&s, w);
    assert!(max_abs_finite(&z[..t0]) < cfg.entry_z);
    assert!(z[t0] >= cfg.entry_z && z[t0] < cfg.stop_z);
    for &v in &z[t0 + 1..t0 + 3] {
        assert!(v.abs() > cfg.exit_z && v.abs() < cfg.stop_z);
    }
    assert!(z[t0 + 3] >= cfg.stop_z);
    assert!(max_abs_finite(&z[t0 + 4..]) < cfg.entry_z);
    let res = backtest_pair(&y, &x, 10.0, 1.0, &cfg).unwrap();
    assert_eq!(res.trades.len(), 1);
    let tr = &res.trades[0];
    assert_eq!(
        (tr.entry_day, tr.exit_day, tr.direction, tr.stopped),
        (t0 + 1, t0 + 4, -1, true)
    );
    assert_eq!(res.n_stops(), 1);
    let (bps, cap) = (1.5e-4, 1e6);
    let qy = -(cap / 2.0) / y[t0 + 1];
    let qx = (cap / 2.0) / x[t0 + 1];
    assert!((res.daily_cost[t0 + 1] - cap * bps).abs() < 1e-9);
    let exit_notional = qy.abs() * y[t0 + 4] + qx.abs() * x[t0 + 4];
    assert!((res.daily_cost[t0 + 4] - exit_notional * bps).abs() < 1e-9);
    assert!(res.daily_cost[t0 + 4] > 0.0);
    assert_eq!(res.metrics.avg_holding_days, 3.0);
}

#[test]
fn reentry_after_exit_same_direction() {
    // proposed test 13: enter-short, exit, enter-short again with no
    // cool-down: two trades with entry_day_2 == exit_day_1 + 1
    let (t_len, w, t0) = (160usize, 40usize, 70usize);
    let mut s: Vec<f64> = (0..t_len).map(|t| 0.2 * (0.7 * t as f64).sin()).collect();
    let x: Vec<f64> = (0..t_len).map(|t| 200.0 + 0.5 * (t as f64).sin()).collect();
    s[t0] = 0.5;
    s[t0 + 1] = 0.0;
    s[t0 + 2] = 0.5;
    s[t0 + 3] = 0.0;
    let y: Vec<f64> = (0..t_len).map(|t| 10.0 + x[t] + s[t]).collect();
    let cfg = BacktestConfig::new(w, 2.0, 0.5, 4.0, 1.0, 0.5, 1e6).unwrap();
    let z = zpath(&s, w);
    assert!(z[t0] >= cfg.entry_z && z[t0] < cfg.stop_z);
    assert!(z[t0 + 1].abs() <= cfg.exit_z);
    assert!(z[t0 + 2] >= cfg.entry_z && z[t0 + 2] < cfg.stop_z);
    assert!(z[t0 + 3].abs() <= cfg.exit_z);
    assert!(max_abs_finite(&z[..t0]) < cfg.entry_z);
    assert!(max_abs_finite(&z[t0 + 4..]) < cfg.entry_z);
    let res = backtest_pair(&y, &x, 10.0, 1.0, &cfg).unwrap();
    let got: Vec<(usize, usize, i32, bool)> = res
        .trades
        .iter()
        .map(|t| (t.entry_day, t.exit_day, t.direction, t.stopped))
        .collect();
    assert_eq!(
        got,
        vec![(t0 + 1, t0 + 2, -1, false), (t0 + 3, t0 + 4, -1, false)]
    );
    assert_eq!(res.trades[1].entry_day, res.trades[0].exit_day + 1);
}

#[test]
fn equity_nonpositive_metrics_defined() {
    // proposed test 14 (MIN-11): a 4x jump on the short leg loses 1.5x
    // capital: total_return = -1.5 exactly, ann_return pinned to -1.0,
    // max_drawdown = 1.5, finite Sharpe, no NaN anywhere
    let (t_len, w, t0) = (45usize, 40usize, 40usize);
    let mut s: Vec<f64> = (0..t_len).map(|t| 0.2 * (0.7 * t as f64).sin()).collect();
    for v in s.iter_mut().skip(t0) {
        *v = -0.5; // long-spread entry band
    }
    let mut x = vec![50.0; t_len];
    for v in x.iter_mut().skip(t0 + 2) {
        *v = 200.0; // short leg quadruples after entry
    }
    let y: Vec<f64> = s.iter().map(|v| 100.0 + v).collect(); // alpha 50, beta 1
    let mut cfg = BacktestConfig::new(w, 2.0, 0.5, 4.0, 0.0, 0.0, 1e6).unwrap();
    cfg.sizing = Sizing::EqualDollar;
    let res = backtest_pair(&y, &x, 50.0, 1.0, &cfg).unwrap();
    assert_eq!(res.trades.len(), 1);
    let tr = &res.trades[0];
    assert_eq!(
        (tr.entry_day, tr.exit_day, tr.direction, tr.stopped),
        (t0 + 1, t0 + 3, 1, true)
    );
    assert!((tr.pnl + 1.5e6).abs() < 1e-6); // q_x = -1e4 shares * 150
    assert!((res.equity[t_len - 1] + 5.0e5).abs() < 1e-6);
    let m = &res.metrics;
    assert!((m.total_return + 1.5).abs() < 1e-12);
    assert_eq!(m.ann_return, -1.0);
    assert!((m.max_drawdown - 1.5).abs() < 1e-12);
    let (ret, n) = (-1.5_f64, t_len as f64);
    let mean = ret / n;
    let var = ((n - 1.0) * mean * mean + (ret - mean) * (ret - mean)) / (n - 1.0);
    assert!((m.sharpe - mean / var.sqrt() * 252.0_f64.sqrt()).abs() < 1e-10);
    assert!((m.turnover - 3.5).abs() < 1e-12); // 1e6 in, 2.5e6 out
    assert!(m.win_rate.is_finite() && m.avg_holding_days.is_finite());
}

#[test]
fn backtest_config_rejects_nonfinite() {
    // proposed test 5 (MAJ-2)
    let bad = [
        BacktestConfig::new(60, 2.0, 0.5, 4.0, f64::NAN, 0.5, 1e6),
        BacktestConfig::new(60, 2.0, 0.5, 4.0, 1.0, f64::INFINITY, 1e6),
        BacktestConfig::new(60, f64::NAN, 0.5, 4.0, 1.0, 0.5, 1e6),
        BacktestConfig::new(60, 2.0, 0.5, f64::NAN, 1.0, 0.5, 1e6),
        BacktestConfig::new(60, 2.0, f64::NEG_INFINITY, 4.0, 1.0, 0.5, 1e6),
        BacktestConfig::new(60, 2.0, 0.5, f64::INFINITY, 1.0, 0.5, 1e6),
        BacktestConfig::new(60, 2.0, 0.5, 4.0, 1.0, 0.5, f64::NAN),
    ];
    for b in bad {
        assert!(matches!(b, Err(StatarbError::InvalidParam(_))));
    }
}

#[test]
fn config_revalidated_at_run() {
    // proposed test 6 (MAJ-4): a struct literal that bypasses `new` is
    // re-validated by backtest_pair / walk_forward
    let cfg = BacktestConfig {
        window: 20,
        entry_z: 2.0,
        exit_z: 5.0, // breaks exit_z < entry_z
        stop_z: 4.0,
        cost_bps: 1.0,
        slippage_bps: 0.5,
        capital: 1e6,
        ..BacktestConfig::default()
    };
    let p: Vec<f64> = (0..100).map(|i| 50.0 + (0.1 * i as f64).sin()).collect();
    let q: Vec<f64> = (0..100).map(|i| 30.0 + (0.3 * i as f64).cos()).collect();
    assert!(matches!(
        backtest_pair(&p, &q, 0.0, 1.0, &cfg),
        Err(StatarbError::InvalidParam(_))
    ));
    assert!(matches!(
        walk_forward(&p, &q, &cfg, 0.5),
        Err(StatarbError::InvalidParam(_))
    ));
    let zero_cap = BacktestConfig {
        exit_z: 0.5,
        capital: 0.0,
        ..cfg.clone()
    };
    assert!(matches!(
        backtest_pair(&p, &q, 0.0, 1.0, &zero_cap),
        Err(StatarbError::InvalidParam(_))
    ));
}

#[test]
fn walk_forward_length_mismatch_is_typed_error() {
    // proposed test 7 (MAJ-5): previously an index-out-of-range panic
    let (y, x, cfg, _, _) = bundled();
    assert!(matches!(
        walk_forward(&y[..1500], &x[..100], &cfg, 0.6),
        Err(StatarbError::LengthMismatch(_))
    ));
}

#[test]
fn walk_forward_split_exact() {
    // proposed test 8 (MIN-10): floor(T * frac) is the intended integer
    // (700 * 0.7 == 489.99999999999994 in binary floating point)
    let cfg = BacktestConfig::new(5, 1.5, 0.5, 1.7, 1.0, 0.5, 1e6).unwrap();
    for (t_len, frac, expected) in [(700usize, 0.7, 490usize), (100, 0.29, 29), (1500, 0.6, 900)] {
        let y: Vec<f64> = (0..t_len).map(|t| 100.0 + (0.3 * t as f64).sin()).collect();
        let x: Vec<f64> = (0..t_len).map(|t| 50.0 + (0.2 * t as f64).cos()).collect();
        assert_eq!(
            walk_forward(&y, &x, &cfg, frac).unwrap().split,
            expected,
            "T={t_len}"
        );
    }
}

#[test]
fn compute_metrics_validation() {
    use statarb::compute_metrics;
    assert!(matches!(
        compute_metrics(&[], 1e6, &[]),
        Err(StatarbError::InvalidParam(_))
    ));
    assert!(matches!(
        compute_metrics(&[1e6, f64::NAN], 1e6, &[]),
        Err(StatarbError::NonFinite(_))
    ));
    assert!(matches!(
        compute_metrics(&[1e6], 0.0, &[]),
        Err(StatarbError::InvalidParam(_))
    ));
    let m = compute_metrics(&[1e6], 1e6, &[]).unwrap();
    assert_eq!((m.sharpe, m.total_return, m.max_drawdown), (0.0, 0.0, 0.0));
}
