/// \file signals.hpp
/// \brief Spread construction, rolling z-score, and mean-reversion half-life.
#pragma once

#include <cstddef>
#include <vector>

namespace statarb {

/// Rolling standard deviations at or below this are treated as
/// zero-variance windows: the z-score is NaN (no signal).  Pinned.
inline constexpr double kZscoreStdFloor = 1.0e-12;

/// Hedge spread s_t = y_t - (alpha + beta * x_t).
/// \throws std::invalid_argument on length mismatch or non-finite input.
std::vector<double> spread(const std::vector<double>& y,
                           const std::vector<double>& x,
                           double alpha, double beta);

/// Rolling z-score with the current observation included (pinned):
///
///     z_t = (s_t - mean(s_{t-W+1..t})) / std(s_{t-W+1..t}, ddof=1)
///
/// The first W-1 entries are NaN (insufficient history).  A window whose
/// sample standard deviation is <= kZscoreStdFloor yields NaN
/// (zero-variance guard) rather than +/-inf.
///
/// \throws std::invalid_argument when window < 2, window > len(s), or
///         the input is non-finite.
std::vector<double> rolling_zscore(const std::vector<double>& s, int window);

/// Half-life of mean reversion from an AR(1) fit on the spread.
///
/// Regression (pinned):  ds_t = c + rho * s_{t-1} + e_t  (OLS, intercept);
/// half_life = -ln(2) / ln(1 + rho).
///
/// Guards (pinned): rho >= 0 (no reversion) returns +inf; 1 + rho <= 0
/// returns the sentinel 0.0.  Both sentinels mean "the AR(1) fit is not
/// interpretable as a half-life — do not trade this spread" (0.0 covers
/// both the oscillating -2 < rho < -1 and the explosive rho <= -2 cases
/// and deliberately does not distinguish them); a filter such as
/// 0 < hl < 30 excludes both.
///
/// \throws std::invalid_argument with fewer than 12 observations, a
///         constant series, or non-finite input.
double half_life(const std::vector<double>& s);

/// Streaming rolling z-score: ring buffer, O(1) per update, no allocation in
/// update().  Same definition and NaN rules as rolling_zscore, but with running
/// moments (Welford add, then a sliding replace), so it agrees to about 1e-9
/// and is NOT bitwise pinned; the batch function stays the trading reference.
class RollingZ {
public:
    /// \throws std::invalid_argument when window < 2.
    explicit RollingZ(int window);
    /// Add one observation; returns z, or NaN while warming up / zero variance.
    /// \throws std::invalid_argument on a non-finite value (state unchanged).
    double update(double s);
    int window() const { return window_; }

private:
    int window_;
    std::vector<double> buf_;
    std::size_t head_ = 0;
    int count_ = 0;
    double mean_ = 0.0;
    double m2_ = 0.0;
};

}  // namespace statarb
