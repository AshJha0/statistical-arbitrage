/// \file test_ols.cpp
/// \brief OLS unit tests: closed-form 2-variable solution, R^2, SEs,
/// property-style grid, and every validation path.
#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "statarb/linalg.hpp"
#include "statarb/ols.hpp"

namespace {

/// Deterministic pseudo-noise (no RNG anywhere): bounded, zero-ish mean.
double wiggle(int i) { return std::sin(0.7 * i) + 0.25 * std::cos(2.3 * i); }

TEST(Ols, MatchesClosedFormTwoVariableSolution) {
    const int n = 200;
    std::vector<double> x(n), y(n);
    for (int i = 0; i < n; ++i) {
        x[i] = 10.0 + 0.05 * i + wiggle(i);
        y[i] = 3.0 + 1.7 * x[i] + 0.5 * wiggle(i + 7);
    }
    const statarb::OLSResult fit = statarb::ols(y, x, true);

    // Closed form: b = cov(x,y)/var(x), a = mean(y) - b mean(x).
    double mx = 0.0, my = 0.0;
    for (int i = 0; i < n; ++i) {
        mx += x[i];
        my += y[i];
    }
    mx /= n;
    my /= n;
    double sxy = 0.0, sxx = 0.0;
    for (int i = 0; i < n; ++i) {
        sxy += (x[i] - mx) * (y[i] - my);
        sxx += (x[i] - mx) * (x[i] - mx);
    }
    const double b = sxy / sxx;
    const double a = my - b * mx;
    EXPECT_NEAR(fit.params[1], b, 1e-10);
    EXPECT_NEAR(fit.params[0], a, 1e-9);
    EXPECT_EQ(fit.nobs, static_cast<std::size_t>(n));
    EXPECT_EQ(fit.nparams, 2u);
    // t-stats are params / se by definition.
    EXPECT_NEAR(fit.tstats[1], fit.params[1] / fit.se[1], 1e-12);
}

TEST(Ols, PropertyGridRecoversExactLinearLaw) {
    // Property-style check: exact linear data over a grid of (a, b) is
    // recovered to near machine precision with R^2 = 1.
    const int n = 50;
    std::vector<double> x(n);
    for (int i = 0; i < n; ++i) x[i] = 1.0 + 0.1 * i + 0.01 * wiggle(i);
    for (double a : {-5.0, 0.0, 2.5}) {
        for (double b : {-2.0, 0.5, 3.0}) {
            std::vector<double> y(n);
            for (int i = 0; i < n; ++i) y[i] = a + b * x[i];
            const statarb::OLSResult fit = statarb::ols(y, x, true);
            EXPECT_NEAR(fit.params[0], a, 1e-8) << "a=" << a << " b=" << b;
            EXPECT_NEAR(fit.params[1], b, 1e-9) << "a=" << a << " b=" << b;
            EXPECT_NEAR(fit.r2, 1.0, 1e-12);
        }
    }
}

TEST(Ols, ResidualsSumToZeroWithIntercept) {
    const int n = 120;
    std::vector<double> x(n), y(n);
    for (int i = 0; i < n; ++i) {
        x[i] = 5.0 + 0.02 * i + wiggle(i);
        y[i] = 1.0 - 0.4 * x[i] + wiggle(i + 3);
    }
    const statarb::OLSResult fit = statarb::ols(y, x, true);
    double s = 0.0;
    for (double r : fit.resid) s += r;
    EXPECT_NEAR(s, 0.0, 1e-8);
    EXPECT_GE(fit.r2, 0.0);
    EXPECT_LE(fit.r2, 1.0);
    EXPECT_GT(fit.sigma2, 0.0);
}

TEST(Ols, NoInterceptUsesUncenteredTss) {
    const int n = 60;
    std::vector<double> x(n), y(n);
    for (int i = 0; i < n; ++i) {
        x[i] = 1.0 + 0.1 * i;
        y[i] = 2.0 * x[i] + 0.1 * wiggle(i);
    }
    const statarb::OLSResult fit = statarb::ols(y, x, false);
    EXPECT_EQ(fit.nparams, 1u);
    EXPECT_NEAR(fit.params[0], 2.0, 0.01);
    // Uncentered R^2 for a strong through-origin fit is close to 1.
    EXPECT_GT(fit.r2, 0.999);
}

TEST(Ols, ValidationErrors) {
    const std::vector<double> y{1.0, 2.0, 3.0, 4.0};
    const std::vector<double> x{1.0, 2.0, 3.0};
    EXPECT_THROW(statarb::ols(y, x, true), std::invalid_argument);

    std::vector<double> bad{1.0, 2.0, std::numeric_limits<double>::quiet_NaN(), 4.0};
    const std::vector<double> x4{1.0, 2.0, 3.0, 4.0};
    EXPECT_THROW(statarb::ols(bad, x4, true), std::invalid_argument);
    EXPECT_THROW(statarb::ols(x4, bad, true), std::invalid_argument);

    // Constant y with intercept: TSS = 0 -> R^2 undefined.
    const std::vector<double> yc(10, 5.0);
    std::vector<double> xv(10);
    for (int i = 0; i < 10; ++i) xv[static_cast<std::size_t>(i)] = i;
    EXPECT_THROW(statarb::ols(yc, xv, true), std::invalid_argument);

    // Constant regressor next to an intercept -> collinear.
    const std::vector<double> xc(10, 3.0);
    std::vector<double> yv(10);
    for (int i = 0; i < 10; ++i) yv[static_cast<std::size_t>(i)] = i + 0.5;
    EXPECT_THROW(statarb::ols(yv, xc, true), std::invalid_argument);

    // Perfect collinearity between two regressors.
    statarb::Matrix xm(10, 2);
    for (std::size_t i = 0; i < 10; ++i) {
        xm(i, 0) = static_cast<double>(i) + 1.0;
        xm(i, 1) = 2.0 * (static_cast<double>(i) + 1.0);
    }
    EXPECT_THROW(statarb::ols(yv, xm, true), std::invalid_argument);

    // Too few observations: n <= k.
    const std::vector<double> y2{1.0, 2.0};
    const std::vector<double> x2{1.0, 2.0};
    EXPECT_THROW(statarb::ols(y2, x2, true), std::invalid_argument);
}

TEST(Linalg, SolveAndInverseRoundTrip) {
    statarb::Matrix a(3, 3);
    const double vals[3][3] = {{4.0, 1.0, 0.5}, {1.0, 3.0, 0.2}, {0.5, 0.2, 2.0}};
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) a(i, j) = vals[i][j];
    }
    const statarb::Matrix inv = statarb::inverse(a);
    const statarb::Matrix prod = statarb::matmul(a, inv);
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            EXPECT_NEAR(prod(i, j), i == j ? 1.0 : 0.0, 1e-12);
        }
    }
    EXPECT_THROW(statarb::solve(statarb::Matrix(2, 2, 0.0),
                                statarb::Matrix::identity(2)),
                 std::invalid_argument);
}

}  // namespace
