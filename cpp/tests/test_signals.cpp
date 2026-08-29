/// \file test_signals.cpp
/// \brief Rolling z-score, spread and half-life tests, including the
/// zero-variance guard and every validation path.
#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "statarb/signals.hpp"

namespace {

TEST(Zscore, HandComputedWindowThree) {
    const std::vector<double> s{1.0, 2.0, 4.0, 4.0, 1.0};
    const std::vector<double> z = statarb::rolling_zscore(s, 3);
    EXPECT_TRUE(std::isnan(z[0]));
    EXPECT_TRUE(std::isnan(z[1]));
    {
        // window {1,2,4}: mean 7/3, sample sd sqrt(7/3)
        const double m = 7.0 / 3.0;
        const double sd = std::sqrt(((1 - m) * (1 - m) + (2 - m) * (2 - m) +
                                     (4 - m) * (4 - m)) / 2.0);
        EXPECT_NEAR(z[2], (4.0 - m) / sd, 1e-12);
    }
    {
        // window {2,4,4}: mean 10/3
        const double m = 10.0 / 3.0;
        const double sd = std::sqrt(((2 - m) * (2 - m) + (4 - m) * (4 - m) +
                                     (4 - m) * (4 - m)) / 2.0);
        EXPECT_NEAR(z[3], (4.0 - m) / sd, 1e-12);
    }
}

TEST(Zscore, ZeroVarianceWindowGuardYieldsNaN) {
    const std::vector<double> s{5.0, 5.0, 5.0, 5.0, 6.0};
    const std::vector<double> z = statarb::rolling_zscore(s, 3);
    EXPECT_TRUE(std::isnan(z[2]));  // {5,5,5}: sd = 0 -> NaN, never inf
    EXPECT_TRUE(std::isnan(z[3]));
    EXPECT_TRUE(std::isfinite(z[4]));  // {5,5,6} has variance
}

TEST(Zscore, WindowEqualToLengthHasSingleValue) {
    const std::vector<double> s{1.0, 2.0, 3.0, 4.0};
    const std::vector<double> z = statarb::rolling_zscore(s, 4);
    EXPECT_TRUE(std::isnan(z[0]));
    EXPECT_TRUE(std::isnan(z[1]));
    EXPECT_TRUE(std::isnan(z[2]));
    // mean 2.5, sd sqrt(5/3)
    EXPECT_NEAR(z[3], (4.0 - 2.5) / std::sqrt(5.0 / 3.0), 1e-12);
}

TEST(Zscore, ValidationErrors) {
    const std::vector<double> s{1.0, 2.0, 3.0};
    EXPECT_THROW(statarb::rolling_zscore(s, 1), std::invalid_argument);
    EXPECT_THROW(statarb::rolling_zscore(s, 4), std::invalid_argument);
    std::vector<double> bad{1.0, std::numeric_limits<double>::quiet_NaN(), 3.0};
    EXPECT_THROW(statarb::rolling_zscore(bad, 2), std::invalid_argument);
}

TEST(Spread, DefinitionAndErrors) {
    const std::vector<double> y{10.0, 11.0, 12.0};
    const std::vector<double> x{4.0, 5.0, 6.0};
    const std::vector<double> s = statarb::spread(y, x, 1.0, 2.0);
    EXPECT_NEAR(s[0], 10.0 - (1.0 + 2.0 * 4.0), 1e-15);
    EXPECT_NEAR(s[2], 12.0 - (1.0 + 2.0 * 6.0), 1e-15);
    const std::vector<double> x2{4.0, 5.0};
    EXPECT_THROW(statarb::spread(y, x2, 0.0, 1.0), std::invalid_argument);
    EXPECT_THROW(
        statarb::spread(y, x, std::numeric_limits<double>::quiet_NaN(), 1.0),
        std::invalid_argument);
}

TEST(HalfLife, RecoversAr1Decay) {
    // Exact AR(1) with phi = 0.9 and a deterministic tiny perturbation:
    // rho = phi - 1 = -0.1, half-life -ln2/ln(0.9) ~ 6.58 days.
    const int n = 400;
    std::vector<double> s(n);
    s[0] = 5.0;
    for (int t = 1; t < n; ++t) {
        s[static_cast<std::size_t>(t)] =
            0.9 * s[static_cast<std::size_t>(t - 1)] + 0.01 * std::sin(0.9 * t);
    }
    const double hl = statarb::half_life(s);
    EXPECT_NEAR(hl, -std::log(2.0) / std::log(0.9), 0.5);
}

TEST(HalfLife, GuardsAndErrors) {
    // Explosive series: ds_t = s_{t-1} exactly -> rho = 1 >= 0 -> +inf.
    std::vector<double> expl(15);
    for (int t = 0; t < 15; ++t) expl[static_cast<std::size_t>(t)] = std::pow(2.0, t);
    EXPECT_TRUE(std::isinf(statarb::half_life(expl)));

    // Alternating series: ds_t = -2 s_{t-1} -> rho = -2, 1+rho <= 0 -> 0.
    std::vector<double> alt(20);
    for (int t = 0; t < 20; ++t) alt[static_cast<std::size_t>(t)] = t % 2 == 0 ? 1.0 : -1.0;
    EXPECT_DOUBLE_EQ(statarb::half_life(alt), 0.0);

    EXPECT_THROW(statarb::half_life(std::vector<double>(5, 1.0)),
                 std::invalid_argument);  // too short
    EXPECT_THROW(statarb::half_life(std::vector<double>(20, 1.0)),
                 std::invalid_argument);  // constant
    std::vector<double> bad(20, 1.0);
    for (int t = 0; t < 20; ++t) bad[static_cast<std::size_t>(t)] = t;
    bad[7] = std::numeric_limits<double>::infinity();
    EXPECT_THROW(statarb::half_life(bad), std::invalid_argument);
}

}  // namespace
