//! # statarb — statistical arbitrage toolkit
//!
//! Rust port of the `statarb` reference package (repository
//! `statistical-arbitrage`):
//! cointegration diagnostics, a Kalman-filter dynamic hedge, and a
//! pairs backtest. Every algorithm follows the pinned
//! cross-language contract in `API_SPEC.md`; agreement with the Python
//! reference is enforced by the golden suite in
//! `../data/golden/golden.json`.
//!
//! Components:
//!
//! * [`ols`] / [`ols_multi`] — OLS by the pinned normal-equations LU
//!   solve, with plain standard errors and a collinearity guard.
//! * [`adf_test`] — augmented Dickey-Fuller test with fixed lag and the
//!   embedded MacKinnon (2010) finite-sample critical-value response
//!   surfaces (1/2/3 variables).
//! * [`engle_granger`] — two-step cointegration test (hedge OLS + ADF on
//!   the residuals against the 2-variable table).
//! * [`johansen_trace`] — Johansen-lite VECM trace test for 2-3 assets
//!   (native Cholesky + cyclic Jacobi eigen-decomposition).
//! * [`kalman_hedge`] — random-walk state-space `(alpha_t, beta_t)`
//!   filter with the Joseph-form covariance update.
//! * [`spread`] / [`rolling_zscore`] / [`half_life`] — signal helpers.
//! * [`backtest_pair`] / [`walk_forward`] — the pinned day-t signal /
//!   day t+1 close execution state machine with equal-dollar legs.
//!
//! All fallible entry points return [`Result`] with a descriptive
//! [`StatarbError`]; nothing panics on bad input, and the crate is fully
//! deterministic (no RNG at runtime).

pub mod adf;
pub mod backtest;
pub mod data_io;
pub mod engle_granger;
pub mod error;
pub mod johansen;
pub mod kalman;
pub mod linalg;
pub mod ols;
pub mod signals;

pub use adf::{
    adf_test, critical_values, interpolate_pvalue, response_surface, select_lag, AdfResult,
    MAX_LAG, MIN_NOBS,
};
pub use backtest::{
    backtest_pair, compute_metrics, max_abs_zscore, rolling_walk_forward, walk_forward,
    BacktestConfig, BacktestResult, Metrics, RollingBlock, RollingWalkForwardResult, Sizing, Trade,
    WalkForwardResult, SPLIT_EPS,
};
pub use data_io::{load_config, load_csv};
pub use engle_granger::{engle_granger, engle_granger_both, EgBothResult, EgResult};
pub use error::{Result, StatarbError};
pub use johansen::{johansen_trace, trace_crit_95, JohansenResult};
pub use kalman::{kalman_hedge, KalmanResult, KalmanStep};
pub use linalg::{cholesky_lower, jacobi_eigh};
pub use ols::{ols, ols_multi, OlsResult};
pub use signals::{half_life, rolling_zscore, spread, RollingZ};
