/// \file test_johansen.cpp
/// \brief Johansen-lite tests: eigenvalue ordering, rank verdicts on the
/// bundled pair and FX triangle, native Cholesky/Jacobi, validation.
#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "statarb/data_io.hpp"
#include "statarb/johansen.hpp"
#include "statarb/linalg.hpp"

namespace {

statarb::Matrix pair_levels(std::size_t n_rows) {
    const auto pair = statarb::load_csv("../data/pair_equity.csv");
    const std::vector<double>& y = pair.at("stock_y");
    const std::vector<double>& x = pair.at("stock_x");
    statarb::Matrix lv(n_rows, 2);
    for (std::size_t i = 0; i < n_rows; ++i) {
        lv(i, 0) = y[i];
        lv(i, 1) = x[i];
    }
    return lv;
}

TEST(Johansen, PairEigenvalueOrderingAndRank) {
    const statarb::JohansenResult jo = statarb::johansen_trace(pair_levels(900));
    ASSERT_EQ(jo.eigenvalues.size(), 2u);
    EXPECT_GE(jo.eigenvalues[0], jo.eigenvalues[1]);  // lambda1 >= lambda2
    EXPECT_GE(jo.eigenvalues[1], 0.0);                // >= 0
    EXPECT_LT(jo.eigenvalues[0], 1.0);
    EXPECT_EQ(jo.rank, 1);
    EXPECT_EQ(jo.t_eff, 898u);
    // Cointegrating vector is normalised to first component +1 and its
    // second component is close to -beta (true beta 1.7 in-sample).
    EXPECT_DOUBLE_EQ(jo.beta(0, 0), 1.0);
    EXPECT_NEAR(jo.beta(1, 0), -1.7, 0.35);
    // trace(0) exceeds its 95% cv, trace(1) does not (rank 1 verdict).
    EXPECT_GT(jo.trace_stats[0], jo.crit_95[0]);
    EXPECT_LT(jo.trace_stats[1], jo.crit_95[1]);
}

TEST(Johansen, FxTriangleThreeAssetLogs) {
    const auto fx = statarb::load_csv("../data/fx_triangle.csv");
    const std::vector<double>& eurusd = fx.at("eurusd");
    const std::vector<double>& gbpusd = fx.at("gbpusd");
    const std::vector<double>& eurgbp = fx.at("eurgbp");
    statarb::Matrix lv(eurusd.size(), 3);
    for (std::size_t i = 0; i < eurusd.size(); ++i) {
        lv(i, 0) = std::log(eurusd[i]);
        lv(i, 1) = std::log(gbpusd[i]);
        lv(i, 2) = std::log(eurgbp[i]);
    }
    const statarb::JohansenResult jo = statarb::johansen_trace(lv);
    ASSERT_EQ(jo.eigenvalues.size(), 3u);
    EXPECT_GE(jo.eigenvalues[0], jo.eigenvalues[1]);
    EXPECT_GE(jo.eigenvalues[1], jo.eigenvalues[2]);
    EXPECT_GE(jo.eigenvalues[2], 0.0);
    EXPECT_GE(jo.rank, 1);  // the triangle relation cointegrates the logs
}

TEST(Johansen, TraceCriticalValuesPinned) {
    EXPECT_DOUBLE_EQ(statarb::trace_crit_95(1), 3.76);
    EXPECT_DOUBLE_EQ(statarb::trace_crit_95(2), 15.41);
    EXPECT_DOUBLE_EQ(statarb::trace_crit_95(3), 29.68);
    EXPECT_THROW(statarb::trace_crit_95(4), std::invalid_argument);
}

TEST(Johansen, ValidationErrors) {
    // n outside {2,3}
    statarb::Matrix one_col(30, 1, 1.0);
    EXPECT_THROW(statarb::johansen_trace(one_col), std::invalid_argument);
    statarb::Matrix four(30, 4, 1.0);
    EXPECT_THROW(statarb::johansen_trace(four), std::invalid_argument);
    // maxlag_diff != 1
    EXPECT_THROW(statarb::johansen_trace(pair_levels(100), 2),
                 std::invalid_argument);
    // too short
    EXPECT_THROW(statarb::johansen_trace(pair_levels(19)), std::invalid_argument);
    // constant column
    statarb::Matrix cc(30, 2);
    for (std::size_t i = 0; i < 30; ++i) {
        cc(i, 0) = static_cast<double>(i) + std::sin(0.5 * static_cast<double>(i));
        cc(i, 1) = 7.0;
    }
    EXPECT_THROW(statarb::johansen_trace(cc), std::invalid_argument);
    // non-finite
    statarb::Matrix nf = pair_levels(50);
    nf(10, 1) = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(statarb::johansen_trace(nf), std::invalid_argument);
}

TEST(Johansen, CollinearLevelsRejected) {
    // Proposed test 11: exactly collinear level columns are refused with a
    // "singular" error rather than producing garbage eigenvalues.
    const auto pair = statarb::load_csv("../data/pair_equity.csv");
    const std::vector<double>& y = pair.at("stock_y");
    const std::vector<double>& x = pair.at("stock_x");
    statarb::Matrix two(200, 2), three(200, 3);
    for (std::size_t i = 0; i < 200; ++i) {
        two(i, 0) = y[i];
        two(i, 1) = 2.0 * y[i] + 1.0;
        three(i, 0) = y[i];
        three(i, 1) = x[i];
        three(i, 2) = y[i] + x[i];
    }
    for (const statarb::Matrix* lv : {&two, &three}) {
        try {
            statarb::johansen_trace(*lv);
            FAIL() << "collinear levels must throw";
        } catch (const std::invalid_argument& e) {
            EXPECT_NE(std::string(e.what()).find("singular"), std::string::npos) << e.what();
        }
    }
}

TEST(Linalg, HelpersRejectNonFinite) {
    statarb::Matrix nan_m(2, 2, 1.0);
    nan_m(0, 1) = nan_m(1, 0) = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(statarb::jacobi_eigh(nan_m), std::invalid_argument);
    EXPECT_THROW(statarb::cholesky_lower(nan_m), std::invalid_argument);
    EXPECT_THROW(statarb::cholesky_lower(statarb::Matrix(2, 3, 1.0)), std::invalid_argument);
}

TEST(Linalg, CholeskyFactorsSpdMatrix) {
    statarb::Matrix a(3, 3);
    const double vals[3][3] = {{4.0, 2.0, 0.6}, {2.0, 5.0, 1.0}, {0.6, 1.0, 3.0}};
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) a(i, j) = vals[i][j];
    }
    const statarb::Matrix l = statarb::cholesky_lower(a);
    const statarb::Matrix llt = statarb::matmul(l, statarb::transpose(l));
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            EXPECT_NEAR(llt(i, j), a(i, j), 1e-12);
        }
        for (std::size_t j = i + 1; j < 3; ++j) EXPECT_DOUBLE_EQ(l(i, j), 0.0);
    }
    // Not positive definite -> pivot error.
    statarb::Matrix npd(2, 2);
    npd(0, 0) = 1.0;
    npd(0, 1) = 2.0;
    npd(1, 0) = 2.0;
    npd(1, 1) = 1.0;
    EXPECT_THROW(statarb::cholesky_lower(npd), std::invalid_argument);
}

TEST(Linalg, JacobiMatchesTwoByTwoClosedForm) {
    statarb::Matrix a(2, 2);
    a(0, 0) = 3.0;
    a(0, 1) = 1.5;
    a(1, 0) = 1.5;
    a(1, 1) = 2.0;
    const statarb::EigenResult eig = statarb::jacobi_eigh(a);
    // Closed form: mean +/- sqrt(((a-d)/2)^2 + b^2).
    const double mean = 2.5;
    const double disc = std::sqrt(0.25 + 1.5 * 1.5);
    EXPECT_NEAR(eig.values[0], mean + disc, 1e-13);
    EXPECT_NEAR(eig.values[1], mean - disc, 1e-13);
    // Eigenvector columns are orthonormal and satisfy A v = lambda v.
    for (std::size_t j = 0; j < 2; ++j) {
        const double v0 = eig.vectors(0, j);
        const double v1 = eig.vectors(1, j);
        EXPECT_NEAR(v0 * v0 + v1 * v1, 1.0, 1e-13);
        EXPECT_NEAR(a(0, 0) * v0 + a(0, 1) * v1, eig.values[j] * v0, 1e-12);
        EXPECT_NEAR(a(1, 0) * v0 + a(1, 1) * v1, eig.values[j] * v1, 1e-12);
    }
    // Non-symmetric input is rejected.
    statarb::Matrix ns(2, 2);
    ns(0, 0) = 1.0;
    ns(0, 1) = 2.0;
    ns(1, 0) = -2.0;
    ns(1, 1) = 1.0;
    EXPECT_THROW(statarb::jacobi_eigh(ns), std::invalid_argument);
}

}  // namespace
