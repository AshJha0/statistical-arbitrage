/// \file test_kalman.cpp
/// \brief Kalman dynamic hedge tests: RLS limit (delta -> 0), Joseph-form
/// PSD covariance (2x2 closed-form eigenvalues), break adaptation,
/// validation paths.
#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "statarb/data_io.hpp"
#include "statarb/kalman.hpp"
#include "statarb/ols.hpp"

namespace {

struct PairData {
    std::vector<double> y, x;
};

const PairData& pair() {
    static const PairData d = [] {
        const auto csv = statarb::load_csv("../data/pair_equity.csv");
        return PairData{csv.at("stock_y"), csv.at("stock_x")};
    }();
    return d;
}

TEST(Kalman, DeltaToZeroConvergesTowardStaticOls) {
    // On a stable synthetic relationship, delta -> 0 recovers recursive
    // least squares, so the final filtered beta approaches the static
    // OLS beta (loose band per spec).
    const int n = 600;
    std::vector<double> x(n), y(n);
    for (int i = 0; i < n; ++i) {
        x[static_cast<std::size_t>(i)] =
            20.0 + 2.0 * std::sin(0.05 * i) + 0.5 * std::cos(1.3 * i);
        y[static_cast<std::size_t>(i)] =
            1.0 + 1.7 * x[static_cast<std::size_t>(i)] + 0.3 * std::sin(2.1 * i);
    }
    const double b_ols = statarb::ols(y, x, true).params[1];
    const statarb::KalmanResult km = statarb::kalman_hedge(y, x, 1e-10, 1e-3);
    EXPECT_NEAR(km.beta.back(), b_ols, 0.05);
    EXPECT_NEAR(km.alpha.back() + km.beta.back() * 20.0, 1.0 + 1.7 * 20.0, 1.0);
}

TEST(Kalman, JosephFormKeepsCovariancePsd) {
    const statarb::KalmanResult km =
        statarb::kalman_hedge(pair().y, pair().x, 1e-4, 1e-3);
    // 2x2 closed-form eigenvalues of the final posterior covariance:
    // mean +/- sqrt(((p00-p11)/2)^2 + p01^2) — both must be >= 0 and the
    // matrix exactly symmetric (Joseph form + symmetrisation).
    const double p00 = km.final_cov(0, 0);
    const double p01 = km.final_cov(0, 1);
    const double p10 = km.final_cov(1, 0);
    const double p11 = km.final_cov(1, 1);
    EXPECT_DOUBLE_EQ(p01, p10);
    const double mean = 0.5 * (p00 + p11);
    const double disc = std::sqrt(0.25 * (p00 - p11) * (p00 - p11) + p01 * p01);
    EXPECT_GE(mean - disc, -1e-15);  // smallest eigenvalue PSD to rounding
    EXPECT_GE(mean + disc, 0.0);
    // Innovation variances are strictly positive throughout.
    for (double s : km.s_var) EXPECT_GT(s, 0.0);
}

TEST(Kalman, AdaptsThroughRegimeBreak) {
    const statarb::KalmanResult km =
        statarb::kalman_hedge(pair().y, pair().x, 1e-4, 1e-3);
    // Pre-break the filter tracks ~1.7, post-break ~1.1 (true betas).
    EXPECT_NEAR(km.beta[900], 1.7, 0.15);
    EXPECT_NEAR(km.beta[1400], 1.1, 0.15);
    // Output paths have full length and finite values.
    ASSERT_EQ(km.beta.size(), pair().y.size());
    for (std::size_t t = 0; t < km.z.size(); ++t) {
        EXPECT_TRUE(std::isfinite(km.z[t])) << "t=" << t;
    }
}

TEST(Kalman, ValidationErrors) {
    const std::vector<double> y{1.0, 2.0, 3.0};
    const std::vector<double> x{1.0, 2.0, 3.0};
    const std::vector<double> x2{1.0, 2.0};
    EXPECT_THROW(statarb::kalman_hedge(y, x2, 1e-4, 1e-3), std::invalid_argument);
    EXPECT_THROW(statarb::kalman_hedge({1.0}, {1.0}, 1e-4, 1e-3),
                 std::invalid_argument);
    EXPECT_THROW(statarb::kalman_hedge(y, x, 0.0, 1e-3), std::invalid_argument);
    EXPECT_THROW(statarb::kalman_hedge(y, x, 1.0, 1e-3), std::invalid_argument);
    EXPECT_THROW(statarb::kalman_hedge(y, x, -0.1, 1e-3), std::invalid_argument);
    EXPECT_THROW(statarb::kalman_hedge(y, x, 1e-4, 0.0), std::invalid_argument);
    EXPECT_THROW(statarb::kalman_hedge(y, x, 1e-4, -1.0), std::invalid_argument);
    std::vector<double> bad{1.0, std::numeric_limits<double>::quiet_NaN(), 3.0};
    EXPECT_THROW(statarb::kalman_hedge(bad, x, 1e-4, 1e-3), std::invalid_argument);
}

TEST(Kalman, RejectsNonFiniteParams) {
    // Proposed test 4 (MAJ-3): r in {NaN, +inf}, delta NaN/inf -> error on
    // the first 50 bundled rows; a valid call yields finite paths.
    const std::vector<double> y(pair().y.begin(), pair().y.begin() + 50);
    const std::vector<double> x(pair().x.begin(), pair().x.begin() + 50);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    for (double r : {nan, inf, -inf, 0.0, -1e-3}) {
        EXPECT_THROW(statarb::kalman_hedge(y, x, 1e-4, r), std::invalid_argument);
    }
    for (double d : {nan, inf, -inf}) {
        EXPECT_THROW(statarb::kalman_hedge(y, x, d, 1e-3), std::invalid_argument);
    }
    const statarb::KalmanResult km = statarb::kalman_hedge(y, x, 1e-4, 1e-3);
    for (std::size_t t = 0; t < km.beta.size(); ++t) {
        EXPECT_TRUE(std::isfinite(km.beta[t]) && std::isfinite(km.z[t]));
    }
}

}  // namespace
