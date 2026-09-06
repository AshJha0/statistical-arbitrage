//! Integration tests for the time-series toolkit: OLS closed forms,
//! ADF verdicts on bundled fixed data, Johansen properties, Kalman
//! limits and PSD covariance, and error paths.

use std::path::{Path, PathBuf};

use statarb::{
    adf_test, engle_granger, half_life, interpolate_pvalue, johansen_trace, kalman_hedge,
    load_csv, ols, ols_multi, rolling_zscore, spread, StatarbError,
};

fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../data")
}

fn pair() -> (Vec<f64>, Vec<f64>) {
    let d = load_csv(&data_dir().join("pair_equity.csv")).unwrap();
    (d["stock_y"].clone(), d["stock_x"].clone())
}

fn walks() -> (Vec<f64>, Vec<f64>) {
    let d = load_csv(&data_dir().join("random_walks.csv")).unwrap();
    (d["walk_a"].clone(), d["walk_b"].clone())
}

// ---------------------------------------------------------------- OLS

#[test]
fn ols_matches_two_var_closed_form() {
    // slope = cov(x, y) / var(x), intercept = mean(y) - b mean(x)
    let (y, x) = pair();
    let n = y.len() as f64;
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let cov = x
        .iter()
        .zip(&y)
        .map(|(a, b)| (a - mx) * (b - my))
        .sum::<f64>()
        / (n - 1.0);
    let var = x.iter().map(|a| (a - mx) * (a - mx)).sum::<f64>() / (n - 1.0);
    let fit = ols(&y, &x, true).unwrap();
    let b_cf = cov / var;
    let a_cf = my - b_cf * mx;
    assert!((fit.params[1] - b_cf).abs() < 1e-10);
    assert!((fit.params[0] - a_cf).abs() < 1e-8);
    assert_eq!(fit.nobs, y.len());
    assert_eq!(fit.nparams, 2);
}

#[test]
fn ols_property_grid_recovers_coefficients() {
    // property-style loop: noiseless y = a + b x recovers (a, b) exactly
    let x: Vec<f64> = (0..60).map(|t| 1.0 + 8.0 * t as f64 / 59.0).collect();
    for a in [-5.0, 0.0, 2.5] {
        for b in [-2.0, 0.5, 10.0] {
            let y: Vec<f64> = x.iter().map(|v| a + b * v).collect();
            let fit = ols(&y, &x, true).unwrap();
            assert!((fit.params[0] - a).abs() < 1e-9, "a={a} b={b}");
            assert!((fit.params[1] - b).abs() < 1e-9, "a={a} b={b}");
        }
    }
}

#[test]
fn ols_multivariate_and_r2_bounds() {
    let (y, x) = pair();
    let trend: Vec<f64> = (0..500).map(|t| t as f64).collect();
    let fit = ols_multi(&y[..500], &[x[..500].to_vec(), trend], true).unwrap();
    assert_eq!(fit.params.len(), 3);
    assert_eq!(fit.se.len(), 3);
    assert!(fit.r2 > 0.0 && fit.r2 < 1.0);
    assert!(fit.sigma2 > 0.0);
    // residuals of an intercept model sum to ~0
    let rsum: f64 = fit.resid.iter().sum();
    assert!(rsum.abs() < 1e-6);
}

#[test]
fn ols_error_paths() {
    let y: Vec<f64> = (0..20).map(|v| v as f64).collect();
    let x19: Vec<f64> = (0..19).map(|v| v as f64).collect();
    assert!(matches!(
        ols(&y, &x19, true),
        Err(StatarbError::LengthMismatch(_))
    ));
    let mut bad = y.clone();
    bad[3] = f64::NAN;
    assert!(matches!(
        ols(&bad, &y, true),
        Err(StatarbError::NonFinite(_))
    ));
    // n <= k
    assert!(matches!(
        ols(&[1.0, 2.0], &[3.0, 4.0], true),
        Err(StatarbError::TooShort(_))
    ));
    // constant regressor next to the intercept -> collinear
    assert!(matches!(
        ols(&y, &vec![7.0; 20], true),
        Err(StatarbError::Singular(_))
    ));
}

// ---------------------------------------------------------------- ADF

#[test]
fn adf_white_noise_strongly_rejects() {
    // the first difference of a random walk is white noise (bundled,
    // deterministic — no runtime RNG)
    let (wa, _) = walks();
    let noise: Vec<f64> = wa.windows(2).map(|w| w[1] - w[0]).collect();
    let res = adf_test(&noise, 1, 1).unwrap();
    assert!(res.tstat < -10.0, "white noise tstat = {}", res.tstat);
    assert!(res.reject(0.01).unwrap());
    assert!(res.reject(0.05).unwrap());
    assert_eq!(res.pvalue, 1.0e-4); // clamped at the pinned floor
}

#[test]
fn adf_random_walk_does_not_reject() {
    let (wa, _) = walks();
    let res = adf_test(&wa, 1, 1).unwrap();
    assert!(!res.reject(0.01).unwrap());
    assert!(!res.reject(0.05).unwrap());
    assert!(!res.reject(0.10).unwrap());
    assert!(res.pvalue > 0.10);
    assert_eq!(res.nobs, wa.len() - 2);
    assert_eq!(res.maxlag, 1);
}

#[test]
fn adf_pvalue_monotone_in_tstat() {
    // property-style: the interpolated p-value is non-decreasing in t
    let mut last = 0.0;
    for i in 0..80 {
        let t = -6.0 + 0.1 * i as f64;
        let p = interpolate_pvalue(t, 2, Some(898)).unwrap();
        assert!(p >= last, "p-value must be non-decreasing at t={t}");
        assert!((1.0e-4..=0.9999).contains(&p));
        last = p;
    }
}

#[test]
fn adf_reject_validates_alpha() {
    let (wa, _) = walks();
    let res = adf_test(&wa, 1, 1).unwrap();
    assert!(matches!(
        res.reject(0.025),
        Err(StatarbError::InvalidParam(_))
    ));
}

#[test]
fn adf_bundled_random_walk_uses_finite_sample_table() {
    // MacKinnon (2010) 5% value at T_eff = 1498, hand-evaluated
    let (wa, _) = walks();
    let res = adf_test(&wa, 1, 1).unwrap();
    assert_eq!(res.nobs, 1498);
    let t = 1498.0_f64;
    let e5 = -2.86154 - 2.8903 / t - 4.234 / (t * t) - 40.040 / (t * t * t);
    assert!((res.crit[1] - e5).abs() < 1e-12);
    assert!(res.crit[1] < -2.86154);
    assert_eq!(res.crit, statarb::critical_values(1, Some(1498)).unwrap());
}

#[test]
fn adf_lag_zero_supported() {
    let (wa, _) = walks();
    let res = adf_test(&wa, 0, 1).unwrap();
    assert_eq!(res.nobs, wa.len() - 1);
    assert!(res.tstat.is_finite());
}

// ------------------------------------------------------------- signals

#[test]
fn zscore_warmup_length_grid() {
    // property-style: exactly window-1 leading NaNs for every window
    let s: Vec<f64> = (0..40).map(|t| (t as f64 * 0.7).sin()).collect();
    for window in [2usize, 5, 10, 39, 40] {
        let z = rolling_zscore(&s, window).unwrap();
        let nan_count = z.iter().take_while(|v| v.is_nan()).count();
        assert_eq!(nan_count, window - 1, "window {window}");
        assert!(z[window - 1..].iter().all(|v| v.is_finite()));
    }
}

#[test]
fn zscore_window_errors() {
    let s = vec![1.0, 2.0, 3.0];
    assert!(matches!(
        rolling_zscore(&s, 1),
        Err(StatarbError::InvalidParam(_))
    ));
    assert!(matches!(
        rolling_zscore(&s, 4),
        Err(StatarbError::InvalidParam(_))
    ));
    assert!(matches!(
        rolling_zscore(&[1.0, f64::INFINITY, 2.0], 2),
        Err(StatarbError::NonFinite(_))
    ));
}

#[test]
fn spread_and_half_life_errors() {
    assert!(matches!(
        spread(&[1.0, 2.0], &[1.0], 0.0, 1.0),
        Err(StatarbError::LengthMismatch(_))
    ));
    assert!(matches!(
        spread(&[1.0, 2.0], &[1.0, 2.0], f64::NAN, 1.0),
        Err(StatarbError::NonFinite(_))
    ));
    assert!(matches!(
        half_life(&[1.0; 11]),
        Err(StatarbError::TooShort(_))
    ));
    assert!(matches!(
        half_life(&[2.0; 30]),
        Err(StatarbError::ConstantSeries(_))
    ));
}

#[test]
fn half_life_of_insample_spread_is_reasonable() {
    let (y, x) = pair();
    let eg = engle_granger(&y[..900], &x[..900], 1).unwrap();
    let hl = half_life(&eg.resid).unwrap();
    // OU spread with phi = 0.92 -> theoretical half-life ~ ln2/ln(1/0.92) ~ 8.3 days
    assert!(hl > 4.0 && hl < 16.0, "half-life {hl}");
}

// ------------------------------------------------------------ Johansen

#[test]
fn johansen_eigenvalue_ordering_and_range() {
    let (y, x) = pair();
    let levels: Vec<Vec<f64>> = (0..900).map(|t| vec![y[t], x[t]]).collect();
    let jo = johansen_trace(&levels, 1).unwrap();
    assert_eq!(jo.eigenvalues.len(), 2);
    assert!(jo.eigenvalues[0] >= jo.eigenvalues[1]);
    assert!(jo.eigenvalues[1] >= 0.0);
    assert!(jo.eigenvalues[0] < 1.0);
    assert_eq!(jo.t_eff, 898);
    // leading cointegrating vector normalised to first component +1
    assert!((jo.beta[0][0] - 1.0).abs() < 1e-12);
    // trace stats decrease with r and rank is 1 on the cointegrated pair
    assert!(jo.trace_stats[0] > jo.trace_stats[1]);
    assert_eq!(jo.rank, 1);
    assert_eq!(jo.crit_95, vec![15.41, 3.76]);
}

#[test]
fn johansen_fx_triangle_rank() {
    let d = load_csv(&data_dir().join("fx_triangle.csv")).unwrap();
    let n = d["eurusd"].len();
    let levels: Vec<Vec<f64>> = (0..n)
        .map(|t| {
            vec![
                d["eurusd"][t].ln(),
                d["gbpusd"][t].ln(),
                d["eurgbp"][t].ln(),
            ]
        })
        .collect();
    let jo = johansen_trace(&levels, 1).unwrap();
    assert_eq!(jo.eigenvalues.len(), 3);
    assert!(jo.rank >= 1, "FX triangle must show cointegration");
}

#[test]
fn johansen_collinear_levels_rejected() {
    // proposed test 11: exactly collinear level columns are refused
    let (y, x) = pair();
    let two: Vec<Vec<f64>> = (0..200).map(|t| vec![y[t], 2.0 * y[t] + 1.0]).collect();
    let three: Vec<Vec<f64>> = (0..200).map(|t| vec![y[t], x[t], y[t] + x[t]]).collect();
    for lv in [&two, &three] {
        match johansen_trace(lv, 1) {
            Err(StatarbError::Singular(m)) => assert!(m.contains("singular"), "{m}"),
            other => panic!("expected Singular, got {other:?}"),
        }
    }
}

#[test]
fn linalg_helpers_reject_nonfinite() {
    let nan_m = vec![vec![1.0, f64::NAN], vec![f64::NAN, 1.0]];
    assert!(matches!(
        statarb::jacobi_eigh(&nan_m),
        Err(StatarbError::NonFinite(_))
    ));
    assert!(matches!(
        statarb::cholesky_lower(&nan_m),
        Err(StatarbError::NonFinite(_))
    ));
    assert!(matches!(
        statarb::cholesky_lower(&[vec![1.0, 2.0, 3.0]]),
        Err(StatarbError::InvalidParam(_))
    ));
}

#[test]
fn johansen_error_paths() {
    let (y, x) = pair();
    let two: Vec<Vec<f64>> = (0..100).map(|t| vec![y[t], x[t]]).collect();
    // wrong lag
    assert!(matches!(
        johansen_trace(&two, 2),
        Err(StatarbError::InvalidParam(_))
    ));
    // n outside {2,3}
    let four: Vec<Vec<f64>> = (0..100).map(|t| vec![y[t], x[t], y[t] * 2.0, x[t] * 2.0]).collect();
    assert!(matches!(
        johansen_trace(&four, 1),
        Err(StatarbError::InvalidParam(_))
    ));
    // too short
    assert!(matches!(
        johansen_trace(&two[..10], 1),
        Err(StatarbError::TooShort(_))
    ));
    // constant column
    let constant: Vec<Vec<f64>> = (0..100).map(|t| vec![y[t], 5.0]).collect();
    assert!(matches!(
        johansen_trace(&constant, 1),
        Err(StatarbError::ConstantSeries(_))
    ));
}

// -------------------------------------------------------------- Kalman

#[test]
fn kalman_delta_to_zero_approaches_static_ols() {
    // on the pre-break segment the relationship is stable: with a tiny
    // delta the filter converges toward the static OLS beta (loose band)
    let (y, x) = pair();
    let seg = 900;
    let km = kalman_hedge(&y[..seg], &x[..seg], 1e-16, 1e-3).unwrap();
    let b_ols = ols(&y[..seg], &x[..seg], true).unwrap().params[1];
    assert!(
        (km.beta[seg - 1] - b_ols).abs() < 0.05,
        "kalman {} vs ols {b_ols}",
        km.beta[seg - 1]
    );
}

#[test]
fn kalman_joseph_covariance_stays_psd() {
    // 2x2 closed form: PSD iff trace >= 0 and det >= 0 (allow rounding)
    let (y, x) = pair();
    let km = kalman_hedge(&y, &x, 1e-4, 1e-3).unwrap();
    let p = km.final_cov;
    let trace = p[0][0] + p[1][1];
    let det = p[0][0] * p[1][1] - p[0][1] * p[1][0];
    assert!((p[0][1] - p[1][0]).abs() < 1e-15, "P must stay symmetric");
    assert!(trace >= 0.0);
    assert!(det >= -1e-18, "det = {det}");
    // closed-form eigenvalues of the symmetric 2x2
    let mid = 0.5 * trace;
    let rad = (mid * mid - det).max(0.0).sqrt();
    assert!(mid - rad >= -1e-12, "smallest eigenvalue must be >= 0");
    assert!(km.s_var.iter().all(|&v| v > 0.0));
}

#[test]
fn kalman_rejects_nonfinite_params() {
    // proposed test 4 (MAJ-3): r in {NaN, +inf}, delta NaN/inf -> error
    // (previously r = NaN produced an all-NaN beta path silently)
    let (y, x) = pair();
    let (y, x) = (&y[..50], &x[..50]);
    for r in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0, -1e-3] {
        assert!(matches!(
            kalman_hedge(y, x, 1e-4, r),
            Err(StatarbError::InvalidParam(_))
        ));
    }
    for d in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(matches!(
            kalman_hedge(y, x, d, 1e-3),
            Err(StatarbError::InvalidParam(_))
        ));
    }
    let km = kalman_hedge(y, x, 1e-4, 1e-3).unwrap();
    assert!(km.beta.iter().chain(km.z.iter()).all(|v| v.is_finite()));
}

#[test]
fn kalman_tracks_break_better_than_ols() {
    let (y, x) = pair();
    let km = kalman_hedge(&y, &x, 1e-4, 1e-3).unwrap();
    // after the break settles the filtered beta hugs the new true 1.1
    assert!((km.beta[1200] - 1.1).abs() < 0.05);
    // while well before the break it hugs the old true 1.7
    assert!((km.beta[900] - 1.7).abs() < 0.05);
}
