/// \file linalg.hpp
/// \brief Minimal dense linear algebra for the statarb toolkit.
///
/// Everything here is deliberately small and self-contained: the design
/// matrices in this project are tiny (k <= 4), so a hand-rolled LU solve,
/// column Cholesky and cyclic Jacobi eigensolver (both PINNED by
/// API_SPEC.md) are all that is needed.  Dot products accumulate in
/// `long double` so cross-language golden agreement at 1e-10 survives
/// 1500-row sums.
#pragma once

#include <cstddef>
#include <vector>

namespace statarb {

/// Dense row-major matrix of doubles (small sizes only).
class Matrix {
public:
    Matrix() = default;

    /// Construct a rows x cols matrix filled with `fill`.
    Matrix(std::size_t rows, std::size_t cols, double fill = 0.0)
        : rows_(rows), cols_(cols), data_(rows * cols, fill) {}

    /// n x n identity matrix.
    static Matrix identity(std::size_t n);

    std::size_t rows() const { return rows_; }
    std::size_t cols() const { return cols_; }

    double& operator()(std::size_t i, std::size_t j) { return data_[i * cols_ + j]; }
    double operator()(std::size_t i, std::size_t j) const { return data_[i * cols_ + j]; }

    /// Read-only access to the flat row-major buffer.
    const std::vector<double>& data() const { return data_; }

private:
    std::size_t rows_ = 0;
    std::size_t cols_ = 0;
    std::vector<double> data_;
};

/// True when every element of `v` is finite (no NaN/inf).
bool all_finite(const std::vector<double>& v);

/// True when every element of `m` is finite (no NaN/inf).
bool all_finite(const Matrix& m);

/// Dot product with long-double accumulation.
double dot(const std::vector<double>& a, const std::vector<double>& b);

/// Matrix product a * b (long-double accumulation per entry).
Matrix matmul(const Matrix& a, const Matrix& b);

/// Transpose of `a`.
Matrix transpose(const Matrix& a);

/// Matrix-vector product a * v.
std::vector<double> matvec(const Matrix& a, const std::vector<double>& v);

/// Solve A X = B by LU decomposition with partial pivoting (PINNED OLS
/// solve path — no QR, no SVD).  Throws std::invalid_argument when A is
/// singular to working precision.
Matrix solve(Matrix a, Matrix b);

/// Solve A x = b for a single right-hand side.
std::vector<double> solve_vec(const Matrix& a, const std::vector<double>& b);

/// Inverse of a small square matrix via the LU solve against I.
Matrix inverse(const Matrix& a);

/// Lower-triangular Cholesky factor of a symmetric positive-definite
/// matrix (standard column algorithm, PINNED for Johansen).  Throws
/// std::invalid_argument when a pivot is <= 0 (not positive definite).
Matrix cholesky_lower(const Matrix& a);

/// Eigen-decomposition result of a symmetric matrix.
struct EigenResult {
    std::vector<double> values;  ///< Eigenvalues, sorted descending.
    Matrix vectors;              ///< Matching eigenvector columns.
};

/// Eigen-decomposition of a symmetric matrix by the cyclic Jacobi
/// rotation algorithm (PINNED — no library eigensolver): sweep the
/// upper triangle in row-major order; for |a_pq| > 1e-14 rotate with
/// theta = (a_qq - a_pp)/(2 a_pq), t = sign(theta)/(|theta| +
/// sqrt(theta^2 + 1)) (t = 1 when theta = 0), c = 1/sqrt(t^2+1),
/// s = t c; stop when the off-diagonal Frobenius norm < 1e-14 or after
/// 100 sweeps.  Non-convergence is NOT an error: the best iterate is
/// returned (matrices here are 2x2/3x3 and converge).
///
/// Throws std::invalid_argument unless `a` is square and symmetric.
EigenResult jacobi_eigh(const Matrix& a);

/// 2-norm condition number of a symmetric matrix: ratio of largest to
/// smallest absolute eigenvalue (+inf when the smallest is zero).
/// Matches numpy.linalg.cond on the symmetric PSD matrices used here.
double sym_cond(const Matrix& a);

/// Jacobi convergence threshold on the off-diagonal Frobenius norm.  Pinned.
inline constexpr double kJacobiTol = 1.0e-14;

/// Maximum number of Jacobi sweeps.  Pinned.
inline constexpr int kJacobiMaxSweeps = 100;

}  // namespace statarb
