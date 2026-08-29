#include "statarb/johansen.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace statarb {

double trace_crit_95(int n_minus_r) {
    // Osterwald-Lenum (1992) 95% trace critical values, model with an
    // unrestricted constant.  PINNED verbatim.
    switch (n_minus_r) {
        case 1: return 3.76;
        case 2: return 15.41;
        case 3: return 29.68;
        default:
            throw std::invalid_argument("trace critical value defined for n-r in {1,2,3}");
    }
}

namespace {

/// Residuals of each column of z on the common design d (normal
/// equations, one solve per block).  Throws when cond(d'd) > 1e12.
Matrix residualise(const Matrix& z, const Matrix& d) {
    const Matrix dt = transpose(d);
    const Matrix dtd = matmul(dt, d);
    if (sym_cond(dtd) > 1.0e12) {
        throw std::invalid_argument("partialled-out design is (near-)singular");
    }
    const Matrix coef = solve(dtd, matmul(dt, z));
    const Matrix fitted = matmul(d, coef);
    Matrix r(z.rows(), z.cols());
    for (std::size_t i = 0; i < z.rows(); ++i) {
        for (std::size_t j = 0; j < z.cols(); ++j) {
            r(i, j) = z(i, j) - fitted(i, j);
        }
    }
    return r;
}

/// S_ij = a' b / t_eff.
Matrix product_moment(const Matrix& a, const Matrix& b, std::size_t t_eff) {
    Matrix s = matmul(transpose(a), b);
    Matrix out(s.rows(), s.cols());
    for (std::size_t i = 0; i < s.rows(); ++i) {
        for (std::size_t j = 0; j < s.cols(); ++j) {
            out(i, j) = s(i, j) / static_cast<double>(t_eff);
        }
    }
    return out;
}

}  // namespace

JohansenResult johansen_trace(const Matrix& levels, int maxlag_diff) {
    const std::size_t t_raw = levels.rows();
    const std::size_t n = levels.cols();
    if (n != 2 && n != 3) {
        throw std::invalid_argument("johansen_trace supports n = 2 or 3 assets only");
    }
    if (maxlag_diff != 1) {
        throw std::invalid_argument("only maxlag_diff = 1 is implemented (pinned)");
    }
    if (t_raw < 20) {
        throw std::invalid_argument("need at least 20 observations");
    }
    if (!all_finite(levels)) {
        throw std::invalid_argument("inputs must be finite (no NaN/inf)");
    }
    for (std::size_t j = 0; j < n; ++j) {
        double mn = levels(0, j), mx = levels(0, j);
        for (std::size_t i = 1; i < t_raw; ++i) {
            mn = std::min(mn, levels(i, j));
            mx = std::max(mx, levels(i, j));
        }
        if (mx - mn == 0.0) {
            throw std::invalid_argument("constant column in levels");
        }
    }

    const std::size_t t_eff = t_raw - 2;
    // dY has T-1 rows; usable regression rows are t = 2..T-1.
    Matrix dy(t_raw - 1, n);
    for (std::size_t i = 1; i < t_raw; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            dy(i - 1, j) = levels(i, j) - levels(i - 1, j);
        }
    }
    Matrix z0(t_eff, n);       // dY_t
    Matrix z1(t_eff, n);       // Y_{t-1}
    Matrix zd(t_eff, n + 1);   // [1, dY_{t-1}]
    for (std::size_t i = 0; i < t_eff; ++i) {
        zd(i, 0) = 1.0;
        for (std::size_t j = 0; j < n; ++j) {
            z0(i, j) = dy(i + 1, j);
            z1(i, j) = levels(i + 1, j);
            zd(i, j + 1) = dy(i, j);
        }
    }

    const Matrix r0 = residualise(z0, zd);
    const Matrix r1 = residualise(z1, zd);
    const Matrix s00 = product_moment(r0, r0, t_eff);
    const Matrix s01 = product_moment(r0, r1, t_eff);
    const Matrix s10 = transpose(s01);
    const Matrix s11 = product_moment(r1, r1, t_eff);

    const Matrix l = cholesky_lower(s11);
    const Matrix m = matmul(s10, solve(s00, s01));  // S10 S00^{-1} S01
    const Matrix l_inv = solve(l, Matrix::identity(n));
    Matrix a = matmul(matmul(l_inv, m), transpose(l_inv));
    for (std::size_t i = 0; i < n; ++i) {  // kill rounding asymmetry
        for (std::size_t j = 0; j < n; ++j) {
            const double sym = 0.5 * (a(i, j) + a(j, i));
            a(i, j) = sym;
            a(j, i) = sym;
        }
    }
    const EigenResult eig = jacobi_eigh(a);

    JohansenResult res;
    res.t_eff = t_eff;
    res.eigenvalues.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        res.eigenvalues[i] =
            std::min(std::max(eig.values[i], 0.0), 1.0 - 1.0e-12);
    }
    // Cointegrating vectors beta_i = L^{-T} v_i, first component -> +1.
    res.beta = matmul(transpose(l_inv), eig.vectors);
    for (std::size_t i = 0; i < n; ++i) {
        const double head = res.beta(0, i);
        if (std::fabs(head) > 1.0e-12) {
            for (std::size_t k = 0; k < n; ++k) res.beta(k, i) /= head;
        }
    }
    res.trace_stats.resize(n);
    res.crit_95.resize(n);
    for (std::size_t r = 0; r < n; ++r) {
        long double acc = 0.0L;
        for (std::size_t i = r; i < n; ++i) {
            acc += std::log(1.0 - res.eigenvalues[i]);
        }
        res.trace_stats[r] =
            -static_cast<double>(t_eff) * static_cast<double>(acc);
        res.crit_95[r] = trace_crit_95(static_cast<int>(n - r));
    }
    res.rank = static_cast<int>(n);
    for (std::size_t r = 0; r < n; ++r) {
        if (res.trace_stats[r] < res.crit_95[r]) {
            res.rank = static_cast<int>(r);
            break;
        }
    }
    return res;
}

}  // namespace statarb
