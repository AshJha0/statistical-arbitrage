/// \file kalman.hpp
/// \brief Kalman-filter dynamic hedge ratio with Joseph-form covariance
///        update.
///
/// Model (PINNED):
///
///     state       s_t = (alpha_t, beta_t)', s_t = s_{t-1} + w_t
///     observation y_t = H_t s_t + eps_t,    H_t = [1, x_t]
///     Q = (delta / (1 - delta)) * I_2       (RiskMetrics-style scaling)
///     R = r                                 (scalar observation variance)
///     s_0 = (0, 0)',  P_0 = 1e3 * I_2       (pinned prior)
///
/// Recursion per observation t = 0..T-1 (the very first observation is
/// already filtered against the prior):
///
///     P_pred = P + Q
///     e_t    = y_t - H_t s
///     S_t    = H_t P_pred H_t' + r
///     K_t    = P_pred H_t' / S_t
///     s      = s + K_t e_t
///     P      = (I - K H) P_pred (I - K H)' + K r K'   (JOSEPH form)
///     P      = (P + P')/2
///     z_t    = e_t / sqrt(S_t)
///
/// The Joseph covariance update is pinned: algebraically equal to
/// (I - K H) P_pred but keeps P symmetric PSD under rounding.
/// delta -> 0 recovers recursive least squares (static OLS in the limit).
#pragma once

#include <vector>

#include "statarb/linalg.hpp"

namespace statarb {

/// Pinned prior covariance scale P_0 = kP0Scale * I.
inline constexpr double kP0Scale = 1.0e3;

/// Filtered dynamic hedge output.
struct KalmanResult {
    std::vector<double> alpha;       ///< Filtered alpha_t path, length T.
    std::vector<double> beta;        ///< Filtered beta_t path, length T.
    std::vector<double> z;           ///< Innovation z-scores e_t / sqrt(S_t).
    std::vector<double> innovation;  ///< Raw innovations e_t.
    std::vector<double> s_var;       ///< Innovation variances S_t.
    Matrix final_cov;                ///< Final 2x2 posterior covariance.
};

/// Filter the dynamic hedge state (alpha_t, beta_t) through (y, x).
///
/// \param y Dependent price series.
/// \param x Hedge instrument price series, same length.
/// \param delta State-noise discount in the OPEN interval (0, 1);
///        Q = delta/(1-delta) * I.
/// \param r Observation noise variance, > 0.
/// \throws std::invalid_argument on length mismatch, fewer than 2
///         observations, non-finite input, delta outside (0, 1), or
///         r <= 0.
KalmanResult kalman_hedge(const std::vector<double>& y,
                          const std::vector<double>& x,
                          double delta = 1.0e-4, double r = 1.0e-3);

}  // namespace statarb
