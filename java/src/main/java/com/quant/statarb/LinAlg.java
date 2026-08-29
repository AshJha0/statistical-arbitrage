package com.quant.statarb;

/**
 * Small dense linear-algebra kernel shared by the statarb package.
 *
 * <p>All matrices are row-major {@code double[rows][cols]}. The matrices in
 * this project are tiny (at most 4x4 systems built from cross products of
 * long series), so the pinned algorithms are simple and explicit:
 *
 * <ul>
 *   <li>linear solves via LU decomposition with partial pivoting (the
 *       cross-language contract pins "normal equations with an LU solve" —
 *       no QR, no SVD);</li>
 *   <li>cross products ({@code X'X}, {@code X'y}) accumulated with Kahan
 *       compensated summation so that sums over 1500-day series agree with
 *       the numpy reference implementation well inside the golden
 *       tolerances;</li>
 *   <li>symmetric eigenvalues via the pinned cyclic Jacobi rotation
 *       (see {@link #jacobiEigh}), also used for the 2-norm condition
 *       number of the symmetric matrices we guard.</li>
 * </ul>
 */
public final class LinAlg {

    /** Jacobi convergence threshold on the off-diagonal Frobenius norm. Pinned. */
    public static final double JACOBI_TOL = 1.0e-14;

    /** Maximum number of cyclic Jacobi sweeps. Pinned. */
    public static final int JACOBI_MAX_SWEEPS = 100;

    private LinAlg() {
    }

    /** Eigen-decomposition of a symmetric matrix: values descending, vectors as columns. */
    public record Eigen(double[] values, double[][] vectors) {
    }

    /**
     * Dot product with Kahan compensated summation.
     *
     * @param a first vector
     * @param b second vector, same length
     * @return sum of elementwise products, compensated
     */
    public static double dot(double[] a, double[] b) {
        if (a.length != b.length) {
            throw new IllegalArgumentException("dot: length mismatch");
        }
        double sum = 0.0;
        double comp = 0.0;
        for (int i = 0; i < a.length; i++) {
            double term = a[i] * b[i] - comp;
            double next = sum + term;
            comp = (next - sum) - term;
            sum = next;
        }
        return sum;
    }

    /** Compensated sum of a vector's entries. */
    public static double sum(double[] a) {
        double s = 0.0;
        double comp = 0.0;
        for (double v : a) {
            double term = v - comp;
            double next = s + term;
            comp = (next - s) - term;
            s = next;
        }
        return s;
    }

    /** Compensated arithmetic mean. */
    public static double mean(double[] a) {
        return sum(a) / a.length;
    }

    /** {@code X' X} (k x k) with compensated column dot products. */
    public static double[][] crossProduct(double[][] x) {
        int n = x.length;
        int k = x[0].length;
        double[][] cols = columns(x, n, k);
        double[][] out = new double[k][k];
        for (int i = 0; i < k; i++) {
            for (int j = i; j < k; j++) {
                double v = dot(cols[i], cols[j]);
                out[i][j] = v;
                out[j][i] = v;
            }
        }
        return out;
    }

    /** {@code X' y} (length k) with compensated dot products. */
    public static double[] crossProduct(double[][] x, double[] y) {
        int n = x.length;
        int k = x[0].length;
        double[][] cols = columns(x, n, k);
        double[] out = new double[k];
        for (int i = 0; i < k; i++) {
            out[i] = dot(cols[i], y);
        }
        return out;
    }

    /** {@code A' B} for tall matrices with compensated dot products. */
    public static double[][] crossProduct(double[][] a, double[][] b) {
        int n = a.length;
        int ka = a[0].length;
        int kb = b[0].length;
        double[][] colsA = columns(a, n, ka);
        double[][] colsB = columns(b, n, kb);
        double[][] out = new double[ka][kb];
        for (int i = 0; i < ka; i++) {
            for (int j = 0; j < kb; j++) {
                out[i][j] = dot(colsA[i], colsB[j]);
            }
        }
        return out;
    }

    private static double[][] columns(double[][] x, int n, int k) {
        double[][] cols = new double[k][n];
        for (int t = 0; t < n; t++) {
            double[] row = x[t];
            for (int j = 0; j < k; j++) {
                cols[j][t] = row[j];
            }
        }
        return cols;
    }

    /** Plain matrix product {@code A B} (small matrices, no compensation needed). */
    public static double[][] matmul(double[][] a, double[][] b) {
        int n = a.length;
        int m = b[0].length;
        int inner = b.length;
        double[][] out = new double[n][m];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                double s = 0.0;
                for (int p = 0; p < inner; p++) {
                    s += a[i][p] * b[p][j];
                }
                out[i][j] = s;
            }
        }
        return out;
    }

    /** Matrix transpose. */
    public static double[][] transpose(double[][] a) {
        int n = a.length;
        int m = a[0].length;
        double[][] out = new double[m][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                out[j][i] = a[i][j];
            }
        }
        return out;
    }

    /** n x n identity matrix. */
    public static double[][] identity(int n) {
        double[][] out = new double[n][n];
        for (int i = 0; i < n; i++) {
            out[i][i] = 1.0;
        }
        return out;
    }

    /**
     * Solve {@code A x = b} by LU decomposition with partial pivoting (pinned).
     *
     * @throws IllegalArgumentException when a pivot is exactly zero (singular)
     */
    public static double[] solve(double[][] a, double[] b) {
        double[][] rhs = new double[b.length][1];
        for (int i = 0; i < b.length; i++) {
            rhs[i][0] = b[i];
        }
        double[][] sol = solveMatrix(a, rhs);
        double[] out = new double[b.length];
        for (int i = 0; i < b.length; i++) {
            out[i] = sol[i][0];
        }
        return out;
    }

    /**
     * Solve {@code A X = B} (multiple right-hand sides) by LU with partial
     * pivoting. Inputs are copied; the arguments are not modified.
     *
     * @throws IllegalArgumentException when the matrix is singular
     */
    public static double[][] solveMatrix(double[][] a, double[][] b) {
        int n = a.length;
        int m = b[0].length;
        double[][] lu = new double[n][];
        double[][] rhs = new double[n][];
        for (int i = 0; i < n; i++) {
            lu[i] = a[i].clone();
            rhs[i] = b[i].clone();
        }
        for (int col = 0; col < n; col++) {
            int piv = col;
            double best = Math.abs(lu[col][col]);
            for (int r = col + 1; r < n; r++) {
                double v = Math.abs(lu[r][col]);
                if (v > best) {
                    best = v;
                    piv = r;
                }
            }
            if (best == 0.0) {
                throw new IllegalArgumentException("singular matrix in LU solve");
            }
            if (piv != col) {
                double[] tmp = lu[piv];
                lu[piv] = lu[col];
                lu[col] = tmp;
                tmp = rhs[piv];
                rhs[piv] = rhs[col];
                rhs[col] = tmp;
            }
            for (int r = col + 1; r < n; r++) {
                double f = lu[r][col] / lu[col][col];
                if (f != 0.0) {
                    for (int c = col; c < n; c++) {
                        lu[r][c] -= f * lu[col][c];
                    }
                    for (int c = 0; c < m; c++) {
                        rhs[r][c] -= f * rhs[col][c];
                    }
                }
            }
        }
        for (int col = n - 1; col >= 0; col--) {
            for (int c = 0; c < m; c++) {
                double s = rhs[col][c];
                for (int p = col + 1; p < n; p++) {
                    s -= lu[col][p] * rhs[p][c];
                }
                rhs[col][c] = s / lu[col][col];
            }
        }
        return rhs;
    }

    /** Matrix inverse via {@link #solveMatrix} against the identity. */
    public static double[][] inverse(double[][] a) {
        return solveMatrix(a, identity(a.length));
    }

    /**
     * 2-norm condition number of a symmetric matrix: ratio of the largest to
     * the smallest absolute eigenvalue (equal to the singular-value ratio for
     * symmetric matrices, matching {@code numpy.linalg.cond}).
     *
     * @return the condition number, {@code +inf} when the smallest eigenvalue
     *     magnitude is zero
     */
    public static double condSymmetric(double[][] a) {
        double[] vals = jacobiEigh(a).values();
        double hi = 0.0;
        double lo = Double.POSITIVE_INFINITY;
        for (double v : vals) {
            double av = Math.abs(v);
            hi = Math.max(hi, av);
            lo = Math.min(lo, av);
        }
        return lo == 0.0 ? Double.POSITIVE_INFINITY : hi / lo;
    }

    /**
     * Lower-triangular Cholesky factor of a small SPD matrix (pinned
     * column-by-column algorithm).
     *
     * @throws IllegalArgumentException when a pivot is not strictly positive
     */
    public static double[][] choleskyLower(double[][] a) {
        int n = a.length;
        double[][] l = new double[n][n];
        for (int j = 0; j < n; j++) {
            double d = a[j][j];
            for (int p = 0; p < j; p++) {
                d -= l[j][p] * l[j][p];
            }
            if (d <= 0.0) {
                throw new IllegalArgumentException(
                        "matrix is not positive definite (Cholesky pivot <= 0)");
            }
            l[j][j] = Math.sqrt(d);
            for (int i = j + 1; i < n; i++) {
                double s = a[i][j];
                for (int p = 0; p < j; p++) {
                    s -= l[i][p] * l[j][p];
                }
                l[i][j] = s / l[j][j];
            }
        }
        return l;
    }

    /**
     * Eigen-decomposition of a symmetric matrix by cyclic Jacobi rotations
     * (pinned algorithm — no library eigensolver).
     *
     * <p>Sweeps the upper triangle in row-major order; for each off-diagonal
     * element with {@code |a_pq| > JACOBI_TOL} applies the classical rotation
     * {@code theta = (a_qq - a_pp) / (2 a_pq)},
     * {@code t = sign(theta) / (|theta| + sqrt(theta^2 + 1))} (t = 1 when
     * theta = 0), {@code c = 1/sqrt(t^2+1)}, {@code s = t c}. Stops when the
     * off-diagonal Frobenius norm falls below {@link #JACOBI_TOL} or after
     * {@link #JACOBI_MAX_SWEEPS} sweeps (non-convergence is not an error: the
     * best iterate is returned).
     *
     * @param a symmetric square matrix (checked to {@code atol=1e-10})
     * @return eigenvalues sorted descending with matching eigenvector columns
     * @throws IllegalArgumentException when the matrix is not symmetric
     */
    public static Eigen jacobiEigh(double[][] a) {
        int n = a.length;
        double[][] m = new double[n][];
        for (int i = 0; i < n; i++) {
            if (a[i].length != n) {
                throw new IllegalArgumentException("jacobiEigh requires a square matrix");
            }
            m[i] = a[i].clone();
        }
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (Math.abs(m[i][j] - m[j][i]) > 1e-10) {
                    throw new IllegalArgumentException(
                            "jacobiEigh requires a symmetric square matrix");
                }
            }
        }
        double[][] v = identity(n);
        for (int sweep = 0; sweep < JACOBI_MAX_SWEEPS; sweep++) {
            double off = 0.0;
            for (int i = 1; i < n; i++) {
                for (int j = 0; j < i; j++) {
                    off += m[i][j] * m[i][j];
                }
            }
            off = Math.sqrt(off * 2.0);
            if (off < JACOBI_TOL) {
                break;
            }
            for (int p = 0; p < n - 1; p++) {
                for (int q = p + 1; q < n; q++) {
                    double apq = m[p][q];
                    if (Math.abs(apq) <= JACOBI_TOL) {
                        continue;
                    }
                    double theta = (m[q][q] - m[p][p]) / (2.0 * apq);
                    double t = theta == 0.0
                            ? 1.0
                            : Math.signum(theta) / (Math.abs(theta) + Math.sqrt(theta * theta + 1.0));
                    double c = 1.0 / Math.sqrt(t * t + 1.0);
                    double s = t * c;
                    // A := J' A J and V := V J with the (p, q) plane rotation
                    // J[p][p]=c, J[q][q]=c, J[p][q]=s, J[q][p]=-s, done as
                    // full small-matrix products to mirror the reference.
                    double[][] j = identity(n);
                    j[p][p] = c;
                    j[q][q] = c;
                    j[p][q] = s;
                    j[q][p] = -s;
                    m = matmul(matmul(transpose(j), m), j);
                    v = matmul(v, j);
                }
            }
        }
        double[] vals = new double[n];
        for (int i = 0; i < n; i++) {
            vals[i] = m[i][i];
        }
        Integer[] order = new Integer[n];
        for (int i = 0; i < n; i++) {
            order[i] = i;
        }
        java.util.Arrays.sort(order, (i, j) -> Double.compare(vals[j], vals[i]));
        double[] sortedVals = new double[n];
        double[][] sortedVecs = new double[n][n];
        for (int c = 0; c < n; c++) {
            sortedVals[c] = vals[order[c]];
            for (int r = 0; r < n; r++) {
                sortedVecs[r][c] = v[r][order[c]];
            }
        }
        return new Eigen(sortedVals, sortedVecs);
    }

    /** True when every entry of {@code a} is finite. */
    public static boolean allFinite(double[] a) {
        for (double v : a) {
            if (!Double.isFinite(v)) {
                return false;
            }
        }
        return true;
    }

    /** True when every entry of the matrix is finite. */
    public static boolean allFinite(double[][] a) {
        for (double[] row : a) {
            if (!allFinite(row)) {
                return false;
            }
        }
        return true;
    }
}
