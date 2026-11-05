//! Micro-benchmark (std only): per-update latency of `RollingZ::update` and
//! `KalmanStep::update`, plus throughput of batch `rolling_zscore` and
//! `backtest_pair` on the bundled pair.
//!
//! Timer overhead (`Instant::now`) is comparable to one update, so latency is
//! measured over batches of 64 updates; the per-update figures are batch
//! averages (batch time / 64), not single-call latencies. Absolute numbers
//! depend on the CPU and are not a tick-to-trade claim.
//!
//! Run: `cd rust && cargo run --release --bin bench`

use std::hint::black_box;
use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use statarb::{
    backtest_pair, load_config, load_csv, rolling_zscore, spread, BacktestConfig, KalmanStep,
    Result, RollingZ,
};

const BATCH: usize = 64;
const WARMUP_BATCHES: usize = 2_000;
const MEASURED_BATCHES: usize = 50_000;

fn pct(sorted: &[f64], q: f64) -> f64 {
    let idx = ((sorted.len() as f64 - 1.0) * q).round() as usize;
    sorted[idx]
}

fn report(name: &str, mut ns: Vec<f64>) {
    ns.sort_by(|a, b| a.total_cmp(b));
    println!(
        "  {name:<18} p50 {:8.1}  p99 {:8.1}  p99.9 {:8.1}  max {:8.1}",
        pct(&ns, 0.5),
        pct(&ns, 0.99),
        pct(&ns, 0.999),
        ns[ns.len() - 1]
    );
}

/// Run `batches` batches of `BATCH` updates; returns ns/update per batch.
fn run_batches(batches: usize, mut one: impl FnMut(usize)) -> Vec<f64> {
    let mut out = Vec::with_capacity(batches);
    let mut i = 0;
    for _ in 0..batches {
        let t0 = Instant::now();
        for _ in 0..BATCH {
            one(i);
            i += 1;
        }
        out.push(t0.elapsed().as_nanos() as f64 / BATCH as f64);
    }
    out
}

fn throughput(name: &str, unit: &str, items: usize, reps: usize, mut f: impl FnMut()) {
    for _ in 0..(reps / 10).max(1) {
        f();
    }
    let t0 = Instant::now();
    for _ in 0..reps {
        f();
    }
    let secs = t0.elapsed().as_secs_f64();
    let per_call_us = secs / reps as f64 * 1e6;
    let rate = items as f64 * reps as f64 / secs / 1e6;
    println!("  {name:<18} {per_call_us:9.1} us/call  {rate:8.2} M {unit}/s");
}

fn run() -> Result<()> {
    let data = Path::new(env!("CARGO_MANIFEST_DIR")).join("../data");
    let cfg = load_config(&data.join("config.json"))?;
    let pair = load_csv(&data.join("pair_equity.csv"))?;
    let (y, x) = (&pair["stock_y"], &pair["stock_x"]);
    let n = y.len();
    let bt = BacktestConfig::from_map(&cfg)?;
    let sp = spread(y, x, 0.0, 1.7)?;

    println!("statarb bench (std only); latencies are ns per update, batch averages over {BATCH} updates");
    println!("  {MEASURED_BATCHES} measured batches after {WARMUP_BATCHES} warm-up batches; data: {n} bars");

    let mut rz = RollingZ::new(bt.window)?;
    run_batches(WARMUP_BATCHES, |i| {
        black_box(rz.update(black_box(sp[i % n])).unwrap());
    });
    let ns = run_batches(MEASURED_BATCHES, |i| {
        black_box(rz.update(black_box(sp[i % n])).unwrap());
    });
    report("RollingZ::update", ns);

    let mut k = KalmanStep::new(1e-4, 1e-3)?;
    run_batches(WARMUP_BATCHES, |i| {
        black_box(k.update(black_box(x[i % n]), black_box(y[i % n])).unwrap());
    });
    let ns = run_batches(MEASURED_BATCHES, |i| {
        black_box(k.update(black_box(x[i % n]), black_box(y[i % n])).unwrap());
    });
    report("KalmanStep::update", ns);

    println!("throughput (bundled pair, {n} bars per call):");
    throughput("rolling_zscore", "bars", n, 2_000, || {
        black_box(rolling_zscore(black_box(&sp), bt.window).unwrap());
    });
    throughput("backtest_pair", "bars", n, 500, || {
        black_box(backtest_pair(black_box(y), black_box(x), 0.0, 1.7, &bt).unwrap());
    });
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("bench error: {e}");
            ExitCode::FAILURE
        }
    }
}
