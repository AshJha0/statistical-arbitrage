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
    const statarb::AdfCriticalRow& r1 = statarb::adf_critical_values(1);
    EXPECT_DOUBLE_EQ(r1.cv1, -3.43);
    EXPECT_DOUBLE_EQ(r1.cv5, -2.86);
    EXPECT_DOUBLE_EQ(r1.cv10, -2.57);
    const statarb::AdfCriticalRow& r2 = statarb::adf_critical_values(2);
    EXPECT_DOUBLE_EQ(r2.cv5, -3.34);
    const statarb::AdfCriticalRow& r3 = statarb::adf_critical_values(3);
    EXPECT_DOUBLE_EQ(r3.cv1, -4.29);
}

TEST(Adf, PValueInterpolationProperties) {
    // Knots are reproduced exactly.
    EXPECT_NEAR(statarb::interpolate_pvalue(-3.43, 1), 0.01, 1e-12);
    EXPECT_NEAR(statarb::interpolate_pvalue(-2.86, 1), 0.05, 1e-12);
    EXPECT_NEAR(statarb::interpolate_pvalue(-2.57, 1), 0.10, 1e-12);
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
