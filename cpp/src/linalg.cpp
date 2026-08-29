#include "statarb/linalg.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace statarb {

Matrix Matrix::identity(std::size_t n) {
    Matrix m(n, n, 0.0);
    for (std::size_t i = 0; i < n; ++i) m(i, i) = 1.0;
    return m;
}

bool all_finite(const std::vector<double>& v) {
    for (double x : v) {
        if (!std::isfinite(x)) return false;
    }
    return true;
}

bool all_finite(const Matrix& m) { return all_finite(m.data()); }

double dot(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("dot: length mismatch");
    }
    long double acc = 0.0L;
    for (std::size_t i = 0; i < a.size(); ++i) {
        acc += static_cast<long double>(a[i]) * static_cast<long double>(b[i]);
    }
    return static_cast<double>(acc);
}

Matrix matmul(const Matrix& a, const Matrix& b) {
    if (a.cols() != b.rows()) {
        throw std::invalid_argument("matmul: inner dimension mismatch");
    }
    Matrix c(a.rows(), b.cols());
    for (std::size_t i = 0; i < a.rows(); ++i) {
        for (std::size_t j = 0; j < b.cols(); ++j) {
            long double acc = 0.0L;
            for (std::size_t k = 0; k < a.cols(); ++k) {
                acc += static_cast<long double>(a(i, k)) *
                       static_cast<long double>(b(k, j));
            }
            c(i, j) = static_cast<double>(acc);
        }
    }
    return c;
}

Matrix transpose(const Matrix& a) {
    Matrix t(a.cols(), a.rows());
    for (std::size_t i = 0; i < a.rows(); ++i) {
        for (std::size_t j = 0; j < a.cols(); ++j) t(j, i) = a(i, j);
    }
    return t;
}

std::vector<double> matvec(const Matrix& a, const std::vector<double>& v) {
    if (a.cols() != v.size()) {
        throw std::invalid_argument("matvec: dimension mismatch");
    }
    std::vector<double> out(a.rows());
    for (std::size_t i = 0; i < a.rows(); ++i) {
        long double acc = 0.0L;
        for (std::size_t j = 0; j < a.cols(); ++j) {
            acc += static_cast<long double>(a(i, j)) *
                   static_cast<long double>(v[j]);
        }
        out[i] = static_cast<double>(acc);
    }
    return out;
}

Matrix solve(Matrix a, Matrix b) {
    const std::size_t n = a.rows();
    if (a.cols() != n) throw std::invalid_argument("solve: A must be square");
    if (b.rows() != n) throw std::invalid_argument("solve: rows(B) != rows(A)");
    // LU with partial pivoting, eliminating in place; B carries the RHS.
    for (std::size_t col = 0; col < n; ++col) {
        std::size_t piv = col;
        double best = std::fabs(a(col, col));
        for (std::size_t r = col + 1; r < n; ++r) {
            const double v = std::fabs(a(r, col));
            if (v > best) {
                best = v;
                piv = r;
            }
        }
        if (best == 0.0) {
            throw std::invalid_argument("solve: matrix is singular");
        }
        if (piv != col) {
            for (std::size_t j = 0; j < n; ++j) std::swap(a(col, j), a(piv, j));
            for (std::size_t j = 0; j < b.cols(); ++j) std::swap(b(col, j), b(piv, j));
        }
        for (std::size_t r = col + 1; r < n; ++r) {
            const double f = a(r, col) / a(col, col);
            if (f == 0.0) continue;
            for (std::size_t j = col; j < n; ++j) a(r, j) -= f * a(col, j);
            for (std::size_t j = 0; j < b.cols(); ++j) b(r, j) -= f * b(col, j);
        }
    }
    // Back substitution.
    Matrix x(n, b.cols());
    for (std::size_t jc = 0; jc < b.cols(); ++jc) {
        for (std::size_t ri = n; ri-- > 0;) {
            long double acc = b(ri, jc);
            for (std::size_t j = ri + 1; j < n; ++j) {
                acc -= static_cast<long double>(a(ri, j)) *
                       static_cast<long double>(x(j, jc));
            }
            x(ri, jc) = static_cast<double>(acc / a(ri, ri));
        }
    }
    return x;
}

std::vector<double> solve_vec(const Matrix& a, const std::vector<double>& b) {
    Matrix rhs(b.size(), 1);
    for (std::size_t i = 0; i < b.size(); ++i) rhs(i, 0) = b[i];
    Matrix x = solve(a, rhs);
    std::vector<double> out(b.size());
    for (std::size_t i = 0; i < b.size(); ++i) out[i] = x(i, 0);
    return out;
}

Matrix inverse(const Matrix& a) {
    if (a.rows() != a.cols()) {
        throw std::invalid_argument("inverse: matrix must be square");
    }
    return solve(a, Matrix::identity(a.rows()));
}

Matrix cholesky_lower(const Matrix& a) {
    const std::size_t n = a.rows();
    if (a.cols() != n) {
        throw std::invalid_argument("cholesky_lower: matrix must be square");
    }
    Matrix l(n, n, 0.0);
    for (std::size_t j = 0; j < n; ++j) {
        long double acc = 0.0L;
        for (std::size_t k = 0; k < j; ++k) {
            acc += static_cast<long double>(l(j, k)) *
                   static_cast<long double>(l(j, k));
        }
        const double d = a(j, j) - static_cast<double>(acc);
        if (d <= 0.0) {
            throw std::invalid_argument(
                "matrix is not positive definite (Cholesky pivot <= 0)");
        }
        l(j, j) = std::sqrt(d);
        for (std::size_t i = j + 1; i < n; ++i) {
            long double s = 0.0L;
            for (std::size_t k = 0; k < j; ++k) {
                s += static_cast<long double>(l(i, k)) *
                     static_cast<long double>(l(j, k));
            }
            l(i, j) = (a(i, j) - static_cast<double>(s)) / l(j, j);
        }
    }
    return l;
}

namespace {

/// Off-diagonal Frobenius norm sqrt(2 * sum_{i>j} a_ij^2), the pinned
/// Jacobi convergence measure.
double off_diagonal_norm(const Matrix& a) {
    long double acc = 0.0L;
    for (std::size_t i = 1; i < a.rows(); ++i) {
        for (std::size_t j = 0; j < i; ++j) {
            acc += static_cast<long double>(a(i, j)) *
                   static_cast<long double>(a(i, j));
        }
    }
    return std::sqrt(static_cast<double>(acc) * 2.0);
}

}  // namespace

EigenResult jacobi_eigh(const Matrix& a) {
    const std::size_t n = a.rows();
    if (a.cols() != n) {
        throw std::invalid_argument("jacobi_eigh requires a symmetric square matrix");
    }
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            if (std::fabs(a(i, j) - a(j, i)) > 1.0e-10) {
                throw std::invalid_argument(
                    "jacobi_eigh requires a symmetric square matrix");
            }
        }
    }
    Matrix m = a;
    Matrix v = Matrix::identity(n);
    for (int sweep = 0; sweep < kJacobiMaxSweeps; ++sweep) {
        if (off_diagonal_norm(m) < kJacobiTol) break;
        for (std::size_t p = 0; p + 1 < n; ++p) {
            for (std::size_t q = p + 1; q < n; ++q) {
                const double apq = m(p, q);
                if (std::fabs(apq) <= kJacobiTol) continue;
                const double theta = (m(q, q) - m(p, p)) / (2.0 * apq);
                double t;
                if (theta == 0.0) {
                    t = 1.0;
                } else {
                    const double sgn = theta > 0.0 ? 1.0 : -1.0;
                    t = sgn / (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
                }
                const double c = 1.0 / std::sqrt(t * t + 1.0);
                const double s = t * c;
                // A := J' A J with J = I except J_pp=J_qq=c, J_pq=s, J_qp=-s.
                for (std::size_t k = 0; k < n; ++k) {  // column update A J
                    const double akp = m(k, p);
                    const double akq = m(k, q);
                    m(k, p) = c * akp - s * akq;
                    m(k, q) = s * akp + c * akq;
                }
                for (std::size_t k = 0; k < n; ++k) {  // row update J' (A J)
                    const double apk = m(p, k);
                    const double aqk = m(q, k);
                    m(p, k) = c * apk - s * aqk;
                    m(q, k) = s * apk + c * aqk;
                }
                for (std::size_t k = 0; k < n; ++k) {  // V := V J
                    const double vkp = v(k, p);
                    const double vkq = v(k, q);
                    v(k, p) = c * vkp - s * vkq;
                    v(k, q) = s * vkp + c * vkq;
                }
            }
        }
    }
    std::vector<double> vals(n);
    for (std::size_t i = 0; i < n; ++i) vals[i] = m(i, i);
    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::stable_sort(order.begin(), order.end(),
                     [&vals](std::size_t i, std::size_t j) { return vals[i] > vals[j]; });
    EigenResult res;
    res.values.resize(n);
    res.vectors = Matrix(n, n);
    for (std::size_t j = 0; j < n; ++j) {
        res.values[j] = vals[order[j]];
        for (std::size_t i = 0; i < n; ++i) res.vectors(i, j) = v(i, order[j]);
    }
    return res;
}

double sym_cond(const Matrix& a) {
    const EigenResult eig = jacobi_eigh(a);
    double lo = std::numeric_limits<double>::infinity();
    double hi = 0.0;
    for (double lam : eig.values) {
        const double m = std::fabs(lam);
        lo = std::min(lo, m);
        hi = std::max(hi, m);
    }
    if (lo == 0.0) return std::numeric_limits<double>::infinity();
    return hi / lo;
}

}  // namespace statarb
