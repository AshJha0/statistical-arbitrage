//! Golden-value suite: every case in ../data/golden/golden.json is
//! recomputed from the bundled CSVs and must agree within its stated
//! tolerance (booleans encoded 0.0/1.0; integer expectations exact).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use statarb::{
    adf_test, backtest_pair, engle_granger, half_life, johansen_trace, kalman_hedge,
    load_config, load_csv, ols, rolling_zscore, spread, BacktestConfig,
};

#[derive(Deserialize)]
struct GoldenFile {
    cases: Vec<GoldenCase>,
}

#[derive(Deserialize)]
struct GoldenCase {
    name: String,
    inputs: HashMap<String, serde_json::Value>,
    expect: Expect,
    tol: f64,
}

#[derive(Deserialize)]
struct Expect {
    value: f64,
}

/// Exact number of golden cases (kept in step with generate_data.py).
const GOLDEN_COUNT: usize = 24;

fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../data")
}

fn load_golden() -> Vec<GoldenCase> {
    let text = std::fs::read_to_string(data_dir().join("golden/golden.json"))
        .expect("golden.json must exist at ../data/golden/golden.json");
    let parsed: GoldenFile = serde_json::from_str(&text).expect("golden.json must parse");
    parsed.cases
}

/// Recompute one golden case, mirroring the Python golden test exactly.
fn compute(name: &str, inputs: &HashMap<String, serde_json::Value>) -> f64 {
    let data = data_dir();
    let cfg = load_config(&data.join("config.json")).unwrap();
    let pair = load_csv(&data.join("pair_equity.csv")).unwrap();
    let y = &pair["stock_y"];
    let x = &pair["stock_x"];
    let n_is = (y.len() as f64 * cfg["insample_frac"] + statarb::SPLIT_EPS).floor() as usize;
    let maxlag = cfg["adf_maxlag"] as usize;

    if let Some(rest) = name.strip_prefix("ols_pair_") {
        let fit = ols(y, x, true).unwrap();
        return match rest {
            "alpha" => fit.params[0],
            "beta" => fit.params[1],
            "r2" => fit.r2,
            other => panic!("unknown ols golden case: {other}"),
        };
    }
    if let Some(rest) = name.strip_prefix("adf_random_walk_") {
        let walks = load_csv(&data.join("random_walks.csv")).unwrap();
        let column = inputs["column"].as_str().unwrap();
        let res = adf_test(&walks[column], maxlag, 1).unwrap();
        return match rest {
            "tstat" => res.tstat,
            "reject" => res.reject(0.05).unwrap() as i32 as f64,
            "crit5" => res.crit[1],
            "pvalue" => res.pvalue,
            other => panic!("unknown adf golden case: {other}"),
        };
    }
    if name == "eg_spurious_coint" {
        let walks = load_csv(&data.join("random_walks.csv")).unwrap();
        let eg = engle_granger(&walks["walk_a"], &walks["walk_b"], maxlag).unwrap();
        return eg.cointegrated(0.05).unwrap() as i32 as f64;
    }

    let eg = engle_granger(&y[..n_is], &x[..n_is], maxlag).unwrap();
    match name {
        "adf_resid_insample_tstat" => eg.adf.tstat,
        "adf_resid_insample_crit5" => eg.adf.crit[1],
        "eg_pair_insample_coint" => eg.cointegrated(0.05).unwrap() as i32 as f64,
        "halflife_insample_spread" => half_life(&eg.resid).unwrap(),
        "johansen_pair_trace_r0" | "johansen_pair_trace_r1" | "johansen_pair_rank" => {
            let levels: Vec<Vec<f64>> = (0..n_is).map(|t| vec![y[t], x[t]]).collect();
            let jo = johansen_trace(&levels, 1).unwrap();
            match name {
                "johansen_pair_trace_r0" => jo.trace_stats[0],
                "johansen_pair_trace_r1" => jo.trace_stats[1],
                _ => jo.rank as f64,
            }
        }
        "kalman_beta_day500" | "kalman_beta_day1200" | "kalman_beta_final" => {
            let km = kalman_hedge(y, x, cfg["kalman_delta"], cfg["kalman_r"]).unwrap();
            let day = inputs["day"].as_f64().unwrap() as usize;
            km.beta[day]
        }
        "zscore_day200" | "zscore_day1100" => {
            let s = spread(y, x, eg.alpha, eg.beta).unwrap();
            let z = rolling_zscore(&s, cfg["window"] as usize).unwrap();
            let day = inputs["day"].as_f64().unwrap() as usize;
            z[day]
        }
        "backtest_total_return" | "backtest_sharpe" | "backtest_max_drawdown"
        | "backtest_n_trades" => {
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
            let bt = backtest_pair(y, x, eg.alpha, eg.beta, &bt_cfg).unwrap();
            match name {
                "backtest_total_return" => bt.metrics.total_return,
                "backtest_sharpe" => bt.metrics.sharpe,
                "backtest_max_drawdown" => bt.metrics.max_drawdown,
                _ => bt.metrics.n_trades as f64,
            }
        }
        other => panic!("unknown golden case: {other}"),
    }
}

#[test]
fn golden_cases_all_match() {
    let cases = load_golden();
    assert_eq!(cases.len(), GOLDEN_COUNT, "golden case count must be exact");
    let mut failures = Vec::new();
    for case in &cases {
        let got = compute(&case.name, &case.inputs);
        let diff = (got - case.expect.value).abs();
        if !(diff <= case.tol) {
            failures.push(format!(
                "{}: got {got:.15e}, expected {:.15e}, |diff| {diff:.3e} > tol {:.1e}",
                case.name, case.expect.value, case.tol
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "golden mismatches:\n{}",
        failures.join("\n")
    );
}

#[test]
fn golden_integer_cases_exact() {
    // trade count and Johansen rank are integers and must match exactly
    let cases = load_golden();
    for case in &cases {
        if case.name == "backtest_n_trades" || case.name == "johansen_pair_rank" {
            let got = compute(&case.name, &case.inputs);
            assert_eq!(got, case.expect.value, "{} must match exactly", case.name);
        }
    }
}

#[test]
fn golden_schema_flat_scalars() {
    // names unique; every case carries one scalar expectation and a
    // positive tolerance (the flat cross-language schema)
    let cases = load_golden();
    let mut names: Vec<&str> = cases.iter().map(|c| c.name.as_str()).collect();
    names.sort_unstable();
    let before = names.len();
    names.dedup();
    assert_eq!(before, names.len(), "golden case names must be unique");
    for case in &cases {
        assert!(case.tol > 0.0);
        assert!(case.expect.value.is_finite());
    }
}
