//! Integration tests for the pairs backtest: accounting identity, dollar
//! neutrality, stop-losses in the break regime, no-trade guard,
//! walk-forward behaviour, and validation errors.

use std::path::{Path, PathBuf};

use statarb::{
    backtest_pair, engle_granger, load_config, load_csv, walk_forward, BacktestConfig,
    BacktestResult, StatarbError,
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
    let bt_cfg = BacktestConfig::new(
        cfg["window"] as usize,
        cfg["entry_z"],
        cfg["exit_z"],
        cfg["stop_z"],
        cfg["cost_bps"],
        cfg["slippage_bps"],
        cfg["capital"],
    )
    .unwrap();
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
    // each entry trades exactly capital of notional (capital/2 per leg),
    // so the entry-day cost equals capital * total bps
    let (res, cfg) = bundled_run();
    let cap = cfg.capital;
    let bps = (cfg.cost_bps + cfg.slippage_bps) / 1e4;
    assert!(!res.trades.is_empty());
    let (y, x, _, _, _) = bundled();
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
    assert!(per_trade > 1.5 && per_trade < 2.5, "turnover/trade {per_trade}");
    assert!(res.n_stops() <= m.n_trades);
}

#[test]
fn no_trades_all_metrics_zero() {
    // entry threshold far beyond any achievable z: no trades, flat equity
    let (y, x, _, a, b) = bundled();
    let cfg = BacktestConfig::new(60, 50.0, 0.5, 51.0, 1.0, 0.5, 1e6).unwrap();
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
