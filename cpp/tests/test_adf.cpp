/// \file test_adf.cpp
/// \brief ADF tests: verdicts on the bundled fixed data (white noise via
/// first differences of a random walk vs the walk itself), p-value
/// interpolation properties, and validation paths.
#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "statarb/adf.hpp"
#include "statarb/data_io.hpp"

namespace {

std::vector<double> walk_a() {
    static const std::vector<double> w =
        statarb::load_csv("../data/random_walks.csv").at("walk_a");
    return w;
}

TEST(Adf, WhiteNoiseStronglyRejectsUnitRoot) {
    // First differences of a random walk ARE the bundled white noise:
    // deterministic fixed data, no runtime RNG.
    const std::vector<double> w = walk_a();
    std::vector<double> noise(w.size() - 1);
    for (std::size_t t = 1; t < w.size(); ++t) noise[t - 1] = w[t] - w[t - 1];
    const statarb::ADFResult res = statarb::adf_test(noise, 1, 1);
    EXPECT_TRUE(res.reject(0.01));
    EXPECT_TRUE(res.reject(0.05));
    EXPECT_LT(res.tstat, -10.0);  // deeply stationary
    EXPECT_LE(res.pvalue, 0.01);
}

TEST(Adf, RandomWalkFailsToReject) {
    const statarb::ADFResult res = statarb::adf_test(walk_a(), 1, 1);
    EXPECT_FALSE(res.reject(0.01));
    EXPECT_FALSE(res.reject(0.05));
    EXPECT_FALSE(res.reject(0.10));
    EXPECT_GT(res.pvalue, 0.10);
    EXPECT_EQ(res.maxlag, 1);
    EXPECT_EQ(res.nobs, walk_a().size() - 2);  // T - 1 - p
}

TEST(Adf, CriticalValueTablesArePinned) {
    // Asymptotic b_inf column of MacKinnon (2010) Table 1, no-trend case.
    const statarb::AdfCriticalRow r1 = statarb::adf_critical_values(1);
    EXPECT_DOUBLE_EQ(r1.cv1, -3.43035);
    EXPECT_DOUBLE_EQ(r1.cv5, -2.86154);
    EXPECT_DOUBLE_EQ(r1.cv10, -2.56677);
    const statarb::AdfCriticalRow r2 = statarb::adf_critical_values(2);
    EXPECT_DOUBLE_EQ(r2.cv1, -3.89644);
    EXPECT_DOUBLE_EQ(r2.cv5, -3.33613);
    EXPECT_DOUBLE_EQ(r2.cv10, -3.04445);
    const statarb::AdfCriticalRow r3 = statarb::adf_critical_values(3);
    EXPECT_DOUBLE_EQ(r3.cv1, -4.29374);
    EXPECT_DOUBLE_EQ(r3.cv5, -3.74066);
    EXPECT_DOUBLE_EQ(r3.cv10, -3.45218);
    // Response-surface rows pinned verbatim (spot checks).
    const statarb::AdfSurfaceRow* s1 = statarb::adf_response_surface(1);
    EXPECT_DOUBLE_EQ(s1[1].b1, -2.8903);
    EXPECT_DOUBLE_EQ(s1[1].b2, -4.234);
    EXPECT_DOUBLE_EQ(s1[1].b3, -40.040);
    const statarb::AdfSurfaceRow* s3 = statarb::adf_response_surface(3);
    EXPECT_DOUBLE_EQ(s3[0].b3, 47.433);
    EXPECT_THROW(statarb::adf_response_surface(0), std::invalid_argument);
}

TEST(Adf, FiniteSampleCriticalValuesFollowResponseSurface) {
    // cv(T) = b_inf + b1/T + b2/T^2 + b3/T^3, hand-evaluated for the
    // pinned N=1 5% row at T = 50 and T = 20, and the N=2 5% row at
    // T = 898 (the in-sample EG regression size).
    const double e50 = -2.86154 - 2.8903 / 50.0 - 4.234 / 2500.0 - 40.040 / 125000.0;
    EXPECT_NEAR(statarb::adf_critical_values(1, 50).cv5, e50, 1e-12);
    const double e20 = -2.86154 - 2.8903 / 20.0 - 4.234 / 400.0 - 40.040 / 8000.0;
    EXPECT_NEAR(statarb::adf_critical_values(1, 20).cv5, e20, 1e-12);
    const double e898 = -3.33613 - 6.1101 / 898.0 - 6.823 / (898.0 * 898.0);
    EXPECT_NEAR(statarb::adf_critical_values(2, 898).cv5, e898, 1e-12);
    // Finite-sample values are more negative than b_inf and converge to it.
    for (int n = 1; n <= 3; ++n) {
        const statarb::AdfCriticalRow inf = statarb::adf_critical_values(n);
        const statarb::AdfCriticalRow c20 = statarb::adf_critical_values(n, 20);
        const statarb::AdfCriticalRow big = statarb::adf_critical_values(n, 10000000);
        EXPECT_LT(c20.cv1, inf.cv1);
        EXPECT_LT(c20.cv5, inf.cv5);
        EXPECT_LT(c20.cv10, inf.cv10);
        EXPECT_NEAR(big.cv5, inf.cv5, 1e-5);
    }
    EXPECT_THROW(statarb::adf_critical_values(1, 19), std::invalid_argument);
    EXPECT_THROW(statarb::adf_critical_values(4, 100), std::invalid_argument);
    // Finite-sample knots reproduce the levels exactly.
    EXPECT_NEAR(statarb::interpolate_pvalue(statarb::adf_critical_values(2, 898).cv5, 2, 898),
                0.05, 1e-12);
}

TEST(Adf, SmallSampleGuard) {
    // T_eff < 20 is refused; at T_eff = 20 the verdict uses cv(20).
    std::vector<double> s(40);
    for (int t = 0; t < 40; ++t) {
        s[static_cast<std::size_t>(t)] = std::pow(0.9, t) + 0.05 * std::sin(2.1 * t);
    }
    const std::vector<double> s21(s.begin(), s.begin() + 21);  // T_eff = 19
    EXPECT_THROW(statarb::adf_test(s21, 1, 1), std::invalid_argument);
    const std::vector<double> s15(s.begin(), s.begin() + 15);  // T_eff = 13
    EXPECT_THROW(statarb::adf_test(s15, 1, 1), std::invalid_argument);
    const std::vector<double> s22(s.begin(), s.begin() + 22);  // T_eff = 20
    const statarb::ADFResult res = statarb::adf_test(s22, 1, 1);
    EXPECT_EQ(res.nobs, 20u);
    const double e20 = -2.86154 - 2.8903 / 20.0 - 4.234 / 400.0 - 40.040 / 8000.0;
    EXPECT_NEAR(res.crit.cv5, e20, 1e-12);
    EXPECT_EQ(res.reject(0.05), res.tstat < res.crit.cv5);
}

TEST(Adf, BundledRandomWalkUsesFiniteSampleTable) {
    const statarb::ADFResult res = statarb::adf_test(walk_a(), 1, 1);
    ASSERT_EQ(res.nobs, 1498u);
    const double T = 1498.0;
    const double e5 = -2.86154 - 2.8903 / T - 4.234 / (T * T) - 40.040 / (T * T * T);
    EXPECT_NEAR(res.crit.cv5, e5, 1e-12);
    EXPECT_LT(res.crit.cv5, -2.86154);
}

TEST(Adf, MaxLagUpperBoundAndNaNTstat) {
    EXPECT_THROW(statarb::adf_test(walk_a(), statarb::kAdfMaxLag + 1, 1),
                 std::invalid_argument);
    // Below the cap but longer than the series: "too short", not overflow.
    EXPECT_THROW(statarb::adf_test(walk_a(), statarb::kAdfMaxLag, 1),
                 std::invalid_argument);
    EXPECT_THROW(statarb::adf_test(walk_a(), 2147483647, 1), std::invalid_argument);
    EXPECT_THROW(statarb::interpolate_pvalue(std::numeric_limits<double>::quiet_NaN(), 1),
                 std::invalid_argument);
    // +/-inf t-stats (exact-fit regressions) clamp rather than throw.
    EXPECT_DOUBLE_EQ(statarb::interpolate_pvalue(-std::numeric_limits<double>::infinity(), 1),
                     statarb::kPValueMin);
    EXPECT_DOUBLE_EQ(statarb::interpolate_pvalue(std::numeric_limits<double>::infinity(), 1),
                     statarb::kPValueMax);
}

TEST(Adf, PValueInterpolationProperties) {
    // Knots are reproduced exactly (asymptotic table).
    EXPECT_NEAR(statarb::interpolate_pvalue(-3.43035, 1), 0.01, 1e-12);
    EXPECT_NEAR(statarb::interpolate_pvalue(-2.86154, 1), 0.05, 1e-12);
    EXPECT_NEAR(statarb::interpolate_pvalue(-2.56677, 1), 0.10, 1e-12);
    // Monotone non-decreasing in the t-stat over a grid (property check).
    double prev = 0.0;
    for (double t = -8.0; t <= 4.0; t += 0.25) {
        const double p = statarb::interpolate_pvalue(t, 1);
        EXPECT_GE(p, prev) << "t=" << t;
        EXPECT_GE(p, statarb::kPValueMin);
        EXPECT_LE(p, statarb::kPValueMax);
        prev = p;
    }
    // Clamps at the extremes.
    EXPECT_DOUBLE_EQ(statarb::interpolate_pvalue(-50.0, 1), statarb::kPValueMin);
    EXPECT_DOUBLE_EQ(statarb::interpolate_pvalue(50.0, 1), statarb::kPValueMax);
}

TEST(Adf, ValidationErrors) {
    const std::vector<double> w = walk_a();
    EXPECT_THROW(statarb::adf_test(w, -1, 1), std::invalid_argument);
    EXPECT_THROW(statarb::adf_test(w, 1, 0), std::invalid_argument);
    EXPECT_THROW(statarb::adf_test(w, 1, 4), std::invalid_argument);

    const std::vector<double> constant(50, 1.0);
    EXPECT_THROW(statarb::adf_test(constant, 1, 1), std::invalid_argument);

    std::vector<double> shorty(10);
    for (int i = 0; i < 10; ++i) shorty[static_cast<std::size_t>(i)] = i * 1.1;
    EXPECT_THROW(statarb::adf_test(shorty, 1, 1), std::invalid_argument);

    std::vector<double> bad = w;
    bad[5] = std::numeric_limits<double>::infinity();
    EXPECT_THROW(statarb::adf_test(bad, 1, 1), std::invalid_argument);

    // Invalid alpha in reject().
    const statarb::ADFResult res = statarb::adf_test(w, 1, 1);
    EXPECT_THROW(res.reject(0.025), std::invalid_argument);
}

TEST(Adf, LagZeroUsesAllRows) {
    const std::vector<double> w = walk_a();
    const statarb::ADFResult res = statarb::adf_test(w, 0, 1);
    EXPECT_EQ(res.nobs, w.size() - 1);
    EXPECT_TRUE(std::isfinite(res.tstat));
}

}  // namespace
