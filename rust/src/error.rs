//! Error type for the `statarb` crate.
//!
//! Every fallible public function returns [`Result`] with a
//! [`StatarbError`] variant carrying a descriptive message — no panics on
//! bad input (mirrors the Python package's `ValueError` behaviour).

use std::fmt;

/// Crate-wide result alias.
pub type Result<T> = std::result::Result<T, StatarbError>;

/// Validation / numerical error raised by the statarb toolkit.
///
/// The variants group the pinned error conditions of the API spec
/// (section 9); the payload is a human-readable message matching the
/// Python reference wording where practical.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StatarbError {
    /// Input series have different lengths.
    LengthMismatch(String),
    /// An input contains NaN or +/-inf.
    NonFinite(String),
    /// A series is constant where variation is required (OLS response,
    /// ADF input, half-life spread, Johansen level column).
    ConstantSeries(String),
    /// (Near-)singular or collinear design: `cond(X'X) > 1e12`, a
    /// singular solve, or a non-positive-definite Cholesky pivot.
    Singular(String),
    /// Too few observations for the requested estimation.
    TooShort(String),
    /// A parameter is outside its pinned domain (delta, r, window,
    /// z-thresholds, nvars, maxlag_diff, alpha, insample_frac, ...).
    InvalidParam(String),
    /// Data file / CSV / JSON problem.
    Data(String),
}

impl fmt::Display for StatarbError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StatarbError::LengthMismatch(m)
            | StatarbError::NonFinite(m)
            | StatarbError::ConstantSeries(m)
            | StatarbError::Singular(m)
            | StatarbError::TooShort(m)
            | StatarbError::InvalidParam(m)
            | StatarbError::Data(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for StatarbError {}
