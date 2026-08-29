//! statarb end-to-end demo.
//!
//! Sections:
//!   1. Cointegration diagnostics on the bundled equity pair (EG + Johansen).
//!   2. Kalman vs static OLS beta tracking through the day-1050 regime break.
//!   3. Full backtest report: in-sample vs walk-forward out-of-sample.
//!   4. Spurious-regression teaching case (independent random walks).
//!
//! Run: `cd rust && cargo run --bin demo`

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use statarb::{
    adf_test, engle_granger, half_life, johansen_trace, kalman_hedge, load_config, load_csv,
    ols, walk_forward, BacktestConfig, BacktestResult, Result,
};

fn rule(title: &str) {
    let pad = 66_usize.saturating_sub(title.len());
    println!("\n=== {title} {}", "=".repeat(pad));
}

fn print_metrics(label: &str, res: &BacktestResult) {
    let m = &res.metrics;
    println!(
        "  {label:<12} ret {:7.2}%  ann {:7.2}%  sharpe {:6.2}  maxDD {:5.2}%  \
         trades {:3}  win {:4.1}%  hold {:5.1}d  turn {:5.1}x  stops {}",
        m.total_return * 100.0,
        m.ann_return * 100.0,
        m.sharpe,
        m.max_drawdown * 100.0,
        m.n_trades,
        m.win_rate * 100.0,
        m.avg_holding_days,
        m.turnover,
        res.n_stops()
    );
}

fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../data")
}

fn run() -> Result<()> {
    let data = data_dir();
    let cfg = load_config(&data.join("config.json"))?;
    let pair = load_csv(&data.join("pair_equity.csv"))?;
    let fx = load_csv(&data.join("fx_triangle.csv"))?;
    let walks = load_csv(&data.join("random_walks.csv"))?;
    let y = &pair["stock_y"];
    let x = &pair["stock_x"];
    let n_is = (y.len() as f64 * cfg["insample_frac"]) as usize;
    let maxlag = cfg["adf_maxlag"] as usize;
    let bt_cfg = BacktestConfig::new(
        cfg["window"] as usize,
        cfg["entry_z"],
        cfg["exit_z"],
        cfg["stop_z"],
        cfg["cost_bps"],
        cfg["slippage_bps"],
        cfg["capital"],
    )?;

    println!("statarb demo — cointegration, Kalman hedging, pairs backtest");
    println!(
        "pair_equity.csv: {} days, in-sample = first {} days (true beta 1.7, break at day 1050)",
        y.len(),
        n_is
    );

    rule(&format!("1. Cointegration diagnostics (in-sample, days 0..{})", n_is - 1));
    let eg = engle_granger(&y[..n_is], &x[..n_is], maxlag)?;
    println!(
        "  EG step 1  y = {:.4} + {:.4} x   (R^2 {:.4})",
        eg.alpha, eg.beta, eg.r2
    );
    println!(
        "  EG step 2  ADF(resid) t = {:.4}  crit 5% (2-var) = {:.2}  p ~ {:.4}",
        eg.adf.tstat, eg.adf.crit[1], eg.adf.pvalue
    );
    println!("  verdict    cointegrated at 5%: {}", eg.cointegrated(0.05)?);
    println!("  half-life  {:.2} days", half_life(&eg.resid)?);
    let levels: Vec<Vec<f64>> = (0..n_is).map(|t| vec![y[t], x[t]]).collect();
    let jo = johansen_trace(&levels, 1)?;
    for r in 0..2 {
        let flag = if jo.trace_stats[r] > jo.crit_95[r] { ">" } else { "<" };
        println!(
            "  Johansen   trace(r={r}) = {:8.3} {flag} cv95 {:6.2}",
            jo.trace_stats[r], jo.crit_95[r]
        );
    }
    println!(
        "  Johansen   rank = {}, cointegrating vector (1, {:.4})  [true beta 1.7]",
        jo.rank, jo.beta[1][0]
    );
    let fx_levels: Vec<Vec<f64>> = (0..fx["eurusd"].len())
        .map(|t| {
            vec![
                fx["eurusd"][t].ln(),
                fx["gbpusd"][t].ln(),
                fx["eurgbp"][t].ln(),
            ]
        })
        .collect();
    let jfx = johansen_trace(&fx_levels, 1)?;
    println!(
        "  FX triangle (3 assets, logs): rank = {}, leading vector ({:.2}, {:.2}, {:.2})  \
         [triangle: (1, -1, -1)]",
        jfx.rank, jfx.beta[0][0], jfx.beta[1][0], jfx.beta[2][0]
    );

    rule("2. Kalman vs static OLS through the regime break");
    let km = kalman_hedge(y, x, cfg["kalman_delta"], cfg["kalman_r"])?;
    let b_ols = ols(y, x, true)?.params[1];
    let mut true_beta = vec![1.7; y.len()];
    for (i, tb) in true_beta.iter_mut().enumerate().take(1081).skip(1050) {
        *tb = 1.7 + (1.1 - 1.7) * (i - 1050) as f64 / 30.0;
    }
    for tb in true_beta.iter_mut().skip(1081) {
        *tb = 1.1;
    }
    println!("  day    true beta   kalman beta   static OLS beta");
    for &d in &[250usize, 500, 900, 1049, 1100, 1200, 1499] {
        println!(
            "  {d:5}   {:8.3}   {:10.4}   {:12.4}",
            true_beta[d], km.beta[d], b_ols
        );
    }
    let from = 300;
    let n_tail = (y.len() - from) as f64;
    let te_k = (true_beta[from..]
        .iter()
        .zip(&km.beta[from..])
        .map(|(tb, kb)| (kb - tb) * (kb - tb))
        .sum::<f64>()
        / n_tail)
        .sqrt();
    let te_o = (true_beta[from..]
        .iter()
        .map(|tb| (b_ols - tb) * (b_ols - tb))
        .sum::<f64>()
        / n_tail)
        .sqrt();
    println!(
        "  beta tracking RMSE (day 300+): kalman {te_k:.4} vs static OLS {te_o:.4} \
         -> kalman adapts, OLS cannot"
    );

    rule("3. Pairs backtest — in-sample vs walk-forward out-of-sample");
    let wf = walk_forward(y, x, &bt_cfg, cfg["insample_frac"])?;
    println!(
        "  hedge estimated on days 0..{}: alpha {:.4}, beta {:.4}",
        wf.split - 1,
        wf.alpha,
        wf.beta
    );
    print_metrics("in-sample", &wf.insample);
    print_metrics("OOS (40%)", &wf.oos);
    println!("  degradation is honest: the OOS window contains the day-1050");
    println!("  break; stop-losses cap the damage of the broken relationship.");

    rule("4. Spurious-regression teaching case (independent random walks)");
    let wa = &walks["walk_a"];
    let wb = &walks["walk_b"];
    let a1 = adf_test(wa, maxlag, 1)?;
    let sp = engle_granger(wa, wb, maxlag)?;
    let slope_t = ols(wa, wb, true)?.tstats[1];
    println!("  regressing walk_a on walk_b: slope t-stat = {slope_t:.1} —");
    println!("  'significant' by the usual +/-2 rule, yet the walks are independent.");
    println!("  ADF(walk_a) t = {:.3} (unit root not rejected) and", a1.tstat);
    println!(
        "  ADF(resid) t = {:.3} vs 5% crit {:.2} -> cointegrated: {}",
        sp.adf.tstat,
        sp.adf.crit[1],
        sp.cointegrated(0.05)?
    );
    println!("  verdict: spurious relationship — NO backtest is run on this pair.");

    println!("\ndemo complete.");
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("demo failed: {e}");
            ExitCode::FAILURE
        }
    }
}
