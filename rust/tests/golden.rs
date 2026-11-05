//! Golden-value suite: EVERY case in ../data/golden/golden.json is
//! recomputed from the bundled CSVs and must agree within its stated
//! tolerance (booleans encoded 0.0/1.0; integer expectations exact).
//!
//! The test loops over the whole file and dispatches on `inputs.kind`
//! (or on the case name when there is no kind); there is no enumerated
//! subset. A case that cannot be dispatched, or that disagrees, is a
//! failure, and any failure fails the test.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use statarb::{
    adf_test, backtest_pair, engle_granger, engle_granger_both, half_life, johansen_trace,
    kalman_hedge, load_config, load_csv, ols, ols_multi, rolling_walk_forward, rolling_zscore,
    select_lag, spread, walk_forward, BacktestConfig, BacktestResult, EgResult,
    RollingWalkForwardResult, Sizing, WalkForwardResult,
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

/// Exact number of golden cases (kept in step with generate_data.py and
/// the Python, C++ and Java suites).
const GOLDEN_COUNT: usize = 928;

/// Ports assert only that the stored margin exceeds this bound.
const MARGIN_BOUND: f64 = 1.0e-6;

const SCENARIOS: [&str; 4] = ["limits_beta", "limits_dollar", "plain_beta", "plain_dollar"];

type Res<T> = std::result::Result<T, String>;

fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../data")
}

fn load_golden() -> Vec<GoldenCase> {
    let text = std::fs::read_to_string(data_dir().join("golden/golden.json"))
        .expect("golden.json must exist at ../data/golden/golden.json");
    let parsed: GoldenFile = serde_json::from_str(&text).expect("golden.json must parse");
    parsed.cases
}

fn e<E: std::fmt::Display>(err: E) -> String {
    err.to_string()
}

/// Everything shared between cases, computed once.
struct Ctx {
    cfg: HashMap<String, f64>,
    y: Vec<f64>,
    x: Vec<f64>,
    n_is: usize,
    eg: EgResult,
    bt: HashMap<&'static str, BacktestResult>,
    wf: WalkForwardResult,
    rw: HashMap<(String, usize, usize), RollingWalkForwardResult>,
}

fn scenario_config(cfg: &HashMap<String, f64>, name: &str) -> Res<BacktestConfig> {
    let mut c = BacktestConfig::from_map(cfg).map_err(e)?;
    c.sizing = if name.ends_with("dollar") {
        Sizing::EqualDollar
    } else {
        Sizing::BetaHedged
    };
    if name.starts_with("plain") {
        c.pnl_stop_frac = 0.0;
        c.max_hold_days = 0;
        c.cooldown_days = 0;
    }
    c.validate().map_err(e)?;
    Ok(c)
}

impl Ctx {
    fn new() -> Res<Ctx> {
        let data = data_dir();
        let cfg = load_config(&data.join("config.json")).map_err(e)?;
        let pair = load_csv(&data.join("pair_equity.csv")).map_err(e)?;
        let (y, x) = (pair["stock_y"].clone(), pair["stock_x"].clone());
        let n_is = (y.len() as f64 * cfg["insample_frac"] + statarb::SPLIT_EPS).floor() as usize;
        let eg = engle_granger(&y[..n_is], &x[..n_is], cfg["adf_maxlag"] as usize).map_err(e)?;
        let mut bt = HashMap::new();
        for name in SCENARIOS {
            let c = scenario_config(&cfg, name)?;
            bt.insert(
                name,
                backtest_pair(&y, &x, eg.alpha, eg.beta, &c).map_err(e)?,
            );
        }
        let c = scenario_config(&cfg, "limits_beta")?;
        let wf = walk_forward(&y, &x, &c, cfg["insample_frac"]).map_err(e)?;
        Ok(Ctx {
            cfg,
            y,
            x,
            n_is,
            eg,
            bt,
            wf,
            rw: HashMap::new(),
        })
    }

    fn rolling(&mut self, name: &str, train: usize, test: usize) -> Res<&RollingWalkForwardResult> {
        let key = (name.to_string(), train, test);
        if !self.rw.contains_key(&key) {
            let c = scenario_config(&self.cfg, name)?;
            let r = rolling_walk_forward(&self.y, &self.x, &c, train, test).map_err(e)?;
            self.rw.insert(key.clone(), r);
        }
        Ok(&self.rw[&key])
    }
}

fn s<'a>(inputs: &'a HashMap<String, serde_json::Value>, key: &str) -> Res<&'a str> {
    inputs
        .get(key)
        .and_then(|v| v.as_str())
        .ok_or_else(|| format!("missing string input '{key}'"))
}

fn n(inputs: &HashMap<String, serde_json::Value>, key: &str) -> Res<f64> {
    inputs
        .get(key)
        .and_then(|v| v.as_f64())
        .ok_or_else(|| format!("missing numeric input '{key}'"))
}

fn u(inputs: &HashMap<String, serde_json::Value>, key: &str) -> Res<usize> {
    let v = n(inputs, key)?;
    if v >= 0.0 && v.fract() == 0.0 {
        Ok(v as usize)
    } else {
        Err(format!("input '{key}' is not a non-negative integer"))
    }
}

/// A metric field of a backtest result (`n_stops`, `equity_sum` included).
fn field(res: &BacktestResult, f: &str) -> Res<f64> {
    let m = &res.metrics;
    Ok(match f {
        "total_return" => m.total_return,
        "ann_return" => m.ann_return,
        "sharpe" => m.sharpe,
        "max_drawdown" => m.max_drawdown,
        "n_trades" => m.n_trades as f64,
        "win_rate" => m.win_rate,
        "avg_holding_days" => m.avg_holding_days,
        "turnover" => m.turnover,
        "n_stops" => res.n_stops() as f64,
        "equity_sum" => {
            let mut sum = 0.0_f64;
            for &v in &res.equity {
                sum += v;
            }
            sum
        }
        other => return Err(format!("unknown metric field '{other}'")),
    })
}

fn scenario<'a>(
    ctx: &'a Ctx,
    inputs: &HashMap<String, serde_json::Value>,
) -> Res<&'a BacktestResult> {
    let name = s(inputs, "scenario")?;
    ctx.bt
        .get(name)
        .ok_or_else(|| format!("unknown scenario '{name}'"))
}

fn margin_of(paths: &[&Vec<f64>], cfg: &HashMap<String, f64>) -> f64 {
    let mut best = f64::INFINITY;
    for p in paths {
        for &z in p.iter().filter(|v| !v.is_nan()) {
            for k in ["entry_z", "exit_z", "stop_z"] {
                best = best.min((z.abs() - cfg[k]).abs());
            }
        }
    }
    best
}

fn compute_kind(
    ctx: &mut Ctx,
    kind: &str,
    inputs: &HashMap<String, serde_json::Value>,
) -> Res<f64> {
    match kind {
        "backtest" => field(scenario(ctx, inputs)?, s(inputs, "field")?),
        "backtest_equity" => {
            let day = u(inputs, "day")?;
            scenario(ctx, inputs)?
                .equity
                .get(day)
                .copied()
                .ok_or_else(|| "day out of range".to_string())
        }
        "backtest_trade" => {
            let idx = u(inputs, "index")?;
            let tr = scenario(ctx, inputs)?
                .trades
                .get(idx)
                .ok_or_else(|| format!("trade index {idx} out of range"))?;
            Ok(match s(inputs, "field")? {
                "entry" => tr.entry_day as f64,
                "exit" => tr.exit_day as f64,
                "direction" => tr.direction as f64,
                "pnl" => tr.pnl,
                "stopped" => tr.stopped as i32 as f64,
                other => return Err(format!("unknown trade field '{other}'")),
            })
        }
        "walk_forward" => {
            let f = s(inputs, "field")?;
            match s(inputs, "segment")? {
                "fit" => Ok(match f {
                    "alpha" => ctx.wf.alpha,
                    "beta" => ctx.wf.beta,
                    "split" => ctx.wf.split as f64,
                    other => return Err(format!("unknown fit field '{other}'")),
                }),
                "insample" => field(&ctx.wf.insample, f),
                "oos" => field(&ctx.wf.oos, f),
                other => Err(format!("unknown segment '{other}'")),
            }
        }
        "rolling" => {
            let (name, train, test) = (
                s(inputs, "scenario")?,
                u(inputs, "train")?,
                u(inputs, "test")?,
            );
            let f = s(inputs, "field")?;
            let rw = ctx.rolling(name, train, test)?;
            if f == "n_blocks" {
                return Ok(rw.blocks.len() as f64);
            }
            let m = &rw.metrics;
            Ok(match f {
                "total_return" => m.total_return,
                "ann_return" => m.ann_return,
                "sharpe" => m.sharpe,
                "max_drawdown" => m.max_drawdown,
                "n_trades" => m.n_trades as f64,
                "win_rate" => m.win_rate,
                "avg_holding_days" => m.avg_holding_days,
                "turnover" => m.turnover,
                "n_stops" => rw.n_stops() as f64,
                "equity_sum" => {
                    let mut sum = 0.0_f64;
                    for &v in &rw.equity {
                        sum += v;
                    }
                    sum
                }
                other => return Err(format!("unknown rolling field '{other}'")),
            })
        }
        "rolling_block" => {
            let (name, train, test) = (
                s(inputs, "scenario")?,
                u(inputs, "train")?,
                u(inputs, "test")?,
            );
            let idx = u(inputs, "block")?;
            let f = s(inputs, "field")?;
            let rw = ctx.rolling(name, train, test)?;
            let b = rw
                .blocks
                .get(idx)
                .ok_or_else(|| format!("block {idx} out of range"))?;
            Ok(match f {
                "start" => b.start as f64,
                "end" => b.end as f64,
                "alpha" => b.alpha,
                "beta" => b.beta,
                "skipped" => b.skipped as i32 as f64,
                "total_return" => b.result.metrics.total_return,
                "sharpe" => b.result.metrics.sharpe,
                "n_trades" => b.result.metrics.n_trades as f64,
                other => return Err(format!("unknown block field '{other}'")),
            })
        }
        "margin" => {
            // the rolling cache needs &mut: fill it before borrowing paths
            for name in ["limits_beta", "plain_beta"] {
                ctx.rolling(name, 500, 100)?;
            }
            let mut paths: Vec<&Vec<f64>> = SCENARIOS.iter().map(|k| &ctx.bt[k].zscore).collect();
            paths.push(&ctx.wf.insample.zscore);
            paths.push(&ctx.wf.oos.zscore);
            for name in ["limits_beta", "plain_beta"] {
                let rw = &ctx.rw[&(name.to_string(), 500, 100)];
                paths.extend(rw.blocks.iter().map(|b| &b.result.zscore));
            }
            let got = margin_of(&paths, &ctx.cfg);
            if got.is_nan() || got <= MARGIN_BOUND {
                return Err(format!("recomputed margin {got:e} <= {MARGIN_BOUND:e}"));
            }
            Ok(got)
        }
        other => Err(format!("unknown golden kind '{other}'")),
    }
}

fn fx_logs() -> Res<Vec<Vec<f64>>> {
    let fx = load_csv(&data_dir().join("fx_triangle.csv")).map_err(e)?;
    Ok(["eurusd", "gbpusd", "eurgbp"]
        .iter()
        .map(|k| fx[*k].iter().map(|v| v.ln()).collect())
        .collect())
}

/// Name-dispatched cases (no `kind`).
fn compute_named(ctx: &Ctx, name: &str, inputs: &HashMap<String, serde_json::Value>) -> Res<f64> {
    let data = data_dir();
    let (y, x) = (&ctx.y, &ctx.x);
    let maxlag = ctx.cfg["adf_maxlag"] as usize;

    if let Some(rest) = name.strip_prefix("ols_pair_") {
        let fit = ols(y, x, true).map_err(e)?;
        return match rest {
            "alpha" => Ok(fit.params[0]),
            "beta" => Ok(fit.params[1]),
            "r2" => Ok(fit.r2),
            other => Err(format!("unknown ols case '{other}'")),
        };
    }
    if let Some(rest) = name.strip_prefix("adf_random_walk_") {
        let walks = load_csv(&data.join("random_walks.csv")).map_err(e)?;
        let res = adf_test(&walks[s(inputs, "column")?], maxlag, 1).map_err(e)?;
        return match rest {
            "tstat" => Ok(res.tstat),
            "reject" => Ok(res.reject(0.05).map_err(e)? as i32 as f64),
            "crit5" => Ok(res.crit[1]),
            "pvalue" => Ok(res.pvalue),
            other => Err(format!("unknown adf case '{other}'")),
        };
    }
    if name == "eg_spurious_coint" {
        let walks = load_csv(&data.join("random_walks.csv")).map_err(e)?;
        let eg = engle_granger(&walks["walk_a"], &walks["walk_b"], maxlag).map_err(e)?;
        return Ok(eg.cointegrated(0.05).map_err(e)? as i32 as f64);
    }
    if name == "select_lag_walk" {
        let walks = load_csv(&data.join("random_walks.csv")).map_err(e)?;
        let lag = select_lag(
            &walks[s(inputs, "column")?],
            u(inputs, "max_lag")?,
            u(inputs, "nvars")?,
        )
        .map_err(e)?;
        return Ok(lag as f64);
    }
    if name.starts_with("adf_fx") || name.starts_with("johansen_fx") {
        let logs = fx_logs()?;
        if name.starts_with("adf_fx") {
            let fit = ols_multi(&logs[0], &logs[1..], true).map_err(e)?;
            let res = adf_test(&fit.resid, u(inputs, "maxlag")?, u(inputs, "nvars")?).map_err(e)?;
            return match name.rsplit('_').next().unwrap_or("") {
                "tstat" => Ok(res.tstat),
                "crit1" => Ok(res.crit[0]),
                "crit5" => Ok(res.crit[1]),
                "crit10" => Ok(res.crit[2]),
                "pvalue" => Ok(res.pvalue),
                other => Err(format!("unknown adf_fx case '{other}'")),
            };
        }
        let levels: Vec<Vec<f64>> = (0..logs[0].len())
            .map(|t| vec![logs[0][t], logs[1][t], logs[2][t]])
            .collect();
        let jo = johansen_trace(&levels, 1).map_err(e)?;
        let tail = name.trim_start_matches("johansen_fx_");
        if tail == "rank" {
            return Ok(jo.rank as f64);
        }
        let idx = tail
            .chars()
            .last()
            .and_then(|c| c.to_digit(10))
            .ok_or("bad johansen_fx index")? as usize;
        return if tail.starts_with("trace_r") {
            jo.trace_stats
                .get(idx)
                .copied()
                .ok_or_else(|| "trace index".to_string())
        } else if tail.starts_with("vector") {
            Ok(jo.beta[idx][0])
        } else {
            Err(format!("unknown johansen_fx case '{tail}'"))
        };
    }
    if name.starts_with("kalman_") {
        let km = kalman_hedge(y, x, n(inputs, "delta")?, n(inputs, "r")?).map_err(e)?;
        let day = u(inputs, "day")?;
        let path = if name.starts_with("kalman_alpha") {
            &km.alpha
        } else if name.starts_with("kalman_beta") {
            &km.beta
        } else if name.starts_with("kalman_z") {
            &km.z
        } else {
            return Err(format!("unknown kalman case '{name}'"));
        };
        return path
            .get(day)
            .copied()
            .ok_or_else(|| "day out of range".to_string());
    }
    if name.starts_with("adf_resid_lag") {
        let res = adf_test(&ctx.eg.resid, u(inputs, "maxlag")?, u(inputs, "nvars")?).map_err(e)?;
        return match name.rsplit('_').next().unwrap_or("") {
            "tstat" => Ok(res.tstat),
            "crit5" => Ok(res.crit[1]),
            "nobs" => Ok(res.nobs as f64),
            other => Err(format!("unknown adf_resid_lag case '{other}'")),
        };
    }
    if name == "select_lag_resid" {
        let lag =
            select_lag(&ctx.eg.resid, u(inputs, "max_lag")?, u(inputs, "nvars")?).map_err(e)?;
        return Ok(lag as f64);
    }
    if name.starts_with("eg_both") {
        let both = engle_granger_both(&y[..ctx.n_is], &x[..ctx.n_is], maxlag).map_err(e)?;
        return if name == "eg_both_insample_coint" {
            Ok(both.cointegrated(0.05).map_err(e)? as i32 as f64)
        } else if name == "eg_both_insample_x_on_y_tstat" {
            Ok(both.x_on_y.adf.tstat)
        } else {
            Err(format!("unknown eg_both case '{name}'"))
        };
    }
    if name == "halflife_growth_is_inf" {
        let series: Vec<f64> = (0..u(inputs, "n")?)
            .map(|t| 1.1_f64.powf(t as f64))
            .collect();
        return Ok(half_life(&series).map_err(e)?.is_infinite() as i32 as f64);
    }
    if name == "halflife_alternating_zero" {
        let series: Vec<f64> = (0..u(inputs, "n")?)
            .map(|t| if t % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        return half_life(&series).map_err(e);
    }
    match name {
        "adf_resid_insample_tstat" => Ok(ctx.eg.adf.tstat),
        "adf_resid_insample_crit5" => Ok(ctx.eg.adf.crit[1]),
        "eg_pair_insample_coint" => Ok(ctx.eg.cointegrated(0.05).map_err(e)? as i32 as f64),
        "halflife_insample_spread" => half_life(&ctx.eg.resid).map_err(e),
        "johansen_pair_trace_r0" | "johansen_pair_trace_r1" | "johansen_pair_rank" => {
            let levels: Vec<Vec<f64>> = (0..ctx.n_is).map(|t| vec![y[t], x[t]]).collect();
            let jo = johansen_trace(&levels, 1).map_err(e)?;
            Ok(match name {
                "johansen_pair_trace_r0" => jo.trace_stats[0],
                "johansen_pair_trace_r1" => jo.trace_stats[1],
                _ => jo.rank as f64,
            })
        }
        "zscore_day200" | "zscore_day1100" => {
            let sp = spread(y, x, ctx.eg.alpha, ctx.eg.beta).map_err(e)?;
            let z = rolling_zscore(&sp, ctx.cfg["window"] as usize).map_err(e)?;
            z.get(u(inputs, "day")?)
                .copied()
                .ok_or_else(|| "day out of range".to_string())
        }
        "backtest_total_return"
        | "backtest_sharpe"
        | "backtest_max_drawdown"
        | "backtest_n_trades" => {
            field(&ctx.bt["limits_beta"], name.trim_start_matches("backtest_"))
        }
        other => Err(format!("unknown golden case '{other}'")),
    }
}

fn compute(ctx: &mut Ctx, case: &GoldenCase) -> Res<f64> {
    match case.inputs.get("kind") {
        Some(k) => compute_kind(
            ctx,
            k.as_str().ok_or("kind must be a string")?,
            &case.inputs,
        ),
        None => compute_named(ctx, &case.name, &case.inputs),
    }
}

/// Integer-valued expectations must match exactly (counts, ranks, lags,
/// days, flags).
fn is_exact(case: &GoldenCase) -> bool {
    let f = case
        .inputs
        .get("field")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let by_field = matches!(
        f,
        "n_trades"
            | "n_stops"
            | "n_blocks"
            | "start"
            | "end"
            | "skipped"
            | "split"
            | "entry"
            | "exit"
            | "direction"
            | "stopped"
    );
    let by_name = ["_n_trades", "_rank", "_nobs", "_coint", "_reject"]
        .iter()
        .any(|t| case.name.ends_with(t))
        || case.name.starts_with("select_lag");
    by_field || by_name
}

#[test]
fn golden_all_cases_match() {
    let cases = load_golden();
    assert_eq!(cases.len(), GOLDEN_COUNT, "golden case count must be exact");
    let mut ctx = Ctx::new().expect("context must build");
    let mut failures = Vec::new();
    let mut checked = 0usize;
    for case in &cases {
        checked += 1;
        match compute(&mut ctx, case) {
            Err(msg) => failures.push(format!("{}: {msg}", case.name)),
            Ok(got) => {
                let diff = (got - case.expect.value).abs();
                let tol = if is_exact(case) { 0.0 } else { case.tol };
                if diff.is_nan() || diff > tol {
                    failures.push(format!(
                        "{}: got {got:.15e}, expected {:.15e}, |diff| {diff:.3e} > tol {tol:.1e}",
                        case.name, case.expect.value
                    ));
                }
            }
        }
    }
    assert_eq!(checked, GOLDEN_COUNT, "every case must be checked");
    assert!(
        failures.is_empty(),
        "{} golden mismatches:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn golden_margin_exceeds_bound() {
    // the stored threshold margin must leave room above the rounding level
    let cases = load_golden();
    let margin: Vec<&GoldenCase> = cases
        .iter()
        .filter(|c| c.name == "zscore_threshold_margin")
        .collect();
    assert_eq!(margin.len(), 1);
    assert!(
        margin[0].expect.value > MARGIN_BOUND,
        "stored margin {}",
        margin[0].expect.value
    );
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
        for v in case.inputs.values() {
            assert!(
                v.is_number() || v.is_string(),
                "inputs must be scalars: {}",
                case.name
            );
        }
    }
}
