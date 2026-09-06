#include "statarb/ols.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace statarb {

OLSResult ols(const std::vector<double>& y, const Matrix& x, bool intercept) {
    const std::size_t n = y.size();
    if (x.rows() != n) {
        throw std::invalid_argument("series length mismatch: len(y)=" +
                                    std::to_string(n) + ", len(x)=" +
                                    std::to_string(x.rows()));
    }
    if (!all_finite(y) || !all_finite(x)) {
        throw std::invalid_argument("inputs must be finite (no NaN/inf)");
    }

    Matrix design(n, x.cols() + (intercept ? 1 : 0));
    const std::size_t offset = intercept ? 1 : 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (intercept) design(i, 0) = 1.0;
        for (std::size_t j = 0; j < x.cols(); ++j) design(i, j + offset) = x(i, j);
    }
    const std::size_t k = design.cols();
    if (n <= k) {
        throw std::invalid_argument("need more observations than parameters: n=" +
                                    std::to_string(n) + ", k=" + std::to_string(k));
    }

    const Matrix xt = transpose(design);
    const Matrix xtx = matmul(xt, design);
    if (sym_cond(xtx) > kCondLimit) {
        throw std::invalid_argument(
            "design matrix is (near-)singular: constant or collinear regressors");
    }
    const std::vector<double> xty = matvec(xt, y);
    OLSResult res;
    res.params = solve_vec(xtx, xty);
    res.resid.resize(n);
    const std::vector<double> fitted = matvec(design, res.params);
    for (std::size_t i = 0; i < n; ++i) res.resid[i] = y[i] - fitted[i];
    const double rss = dot(res.resid, res.resid);
    res.sigma2 = rss / static_cast<double>(n - k);
    // cov(b) = sigma2 * inv(X'X); only the diagonal is needed for SEs.
    const Matrix xtx_inv = inverse(xtx);
    res.se.resize(k);
    res.tstats.resize(k);
    for (std::size_t j = 0; j < k; ++j) {
        res.se[j] = std::sqrt(res.sigma2 * xtx_inv(j, j));
        res.tstats[j] = res.params[j] / res.se[j];
    }

    double tss;
    if (intercept) {
        const double mean = compensated_sum(y) / static_cast<double>(n);
        NeumaierSum acc;
        for (double v : y) acc.add((v - mean) * (v - mean));
        tss = acc.value();
    } else {
        tss = dot(y, y);
    }
    if (tss <= 0.0) {
        throw std::invalid_argument("y is constant: R^2 undefined");
    }
    res.r2 = 1.0 - rss / tss;
    res.nobs = n;
    res.nparams = k;
    return res;
}

OLSResult ols(const std::vector<double>& y, const std::vector<double>& x,
              bool intercept) {
    Matrix xm(x.size(), 1);
    for (std::size_t i = 0; i < x.size(); ++i) xm(i, 0) = x[i];
    return ols(y, xm, intercept);
}

}  // namespace statarb
