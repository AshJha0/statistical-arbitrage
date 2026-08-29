package com.quant.statarb;

/**
 * Johansen-lite trace test for 2- and 3-asset systems.
 *
 * <p>Canonical correlation setup (PINNED). VECM with unrestricted constant
 * and k = 1 lagged difference:
 *
 * <pre>
 *     dY_t = c + Gamma * dY_{t-1} + Pi * Y_{t-1} + e_t
 * </pre>
 *
 * <p>Steps (T raw observations, n = 2 or 3 assets, T_eff = T - 2 usable rows,
 * t = 2..T-1 zero-indexed):
 *
 * <ol>
 *   <li>Z0 = dY_t, Z1 = Y_{t-1}, ZD = [1, dY_{t-1}] (partialled-out block).</li>
 *   <li>R0 = residuals of Z0 on ZD; R1 = residuals of Z1 on ZD (multivariate
 *       OLS via normal equations, one solve per system).</li>
 *   <li>Product-moment matrices S_ij = R_i' R_j / T_eff.</li>
 *   <li>Eigenvalue problem {@code |lambda S11 - S10 inv(S00) S01| = 0}, made
 *       SYMMETRIC via the Cholesky factor S11 = L L':
 *       {@code A = inv(L) S10 inv(S00) S01 inv(L)'}, decomposed by the native
 *       cyclic Jacobi rotation ({@link LinAlg#jacobiEigh} — pinned, no library
 *       eigensolver). Eigenvalues sorted descending; cointegrating vectors
 *       {@code beta_i = inv(L)' v_i}, normalised so the first component is +1.</li>
 *   <li>Trace statistics {@code trace(r) = -T_eff * sum_{i>=r} ln(1 - lambda_i)},
 *       compared against the embedded Osterwald-Lenum (1992) 95% trace table
 *       for the unrestricted-constant case. The estimated rank is the smallest
 *       r with trace(r) &lt; cv(n - r), else n.</li>
 * </ol>
 *
 * <p>Honest limitations: this is a fixed-lag (k=1), fixed-deterministic-case
 * implementation with only the 95% column of the critical-value table; no
 * small-sample correction, no maximum-eigenvalue variant, no p-values. It is
 * meant for 2-3 asset pairs-trading diagnostics, not general VECM work.
 */
public final class Johansen {

    /**
     * Osterwald-Lenum (1992) 95% critical values for the TRACE statistic,
     * model with unrestricted constant, indexed by {@code n - r}:
     * 1 -&gt; 3.76, 2 -&gt; 15.41, 3 -&gt; 29.68. PINNED.
     */
    public static final double[] TRACE_CRIT_95 = {Double.NaN, 3.76, 15.41, 29.68};

    private Johansen() {
    }

    /**
     * Johansen-lite trace-test result.
     *
     * @param eigenvalues canonical correlations lambda_i, descending, in [0, 1)
     * @param traceStats  trace(r) for r = 0..n-1
     * @param crit95      matching Osterwald-Lenum 95% critical values
     * @param rank        estimated cointegration rank at 95%
     * @param beta        (n, n) matrix; column i is the cointegrating vector
     *                    for lambda_i, normalised to first component +1
     * @param tEff        effective sample size T - 2
     */
    public record Result(double[] eigenvalues, double[] traceStats, double[] crit95,
                         int rank, double[][] beta, int tEff) {
    }

    /** Residuals of each column of z on the common design d (normal equations). */
    private static double[][] residualise(double[][] z, double[][] d) {
        double[][] dtd = LinAlg.crossProduct(d);
        if (LinAlg.condSymmetric(dtd) > 1.0e12) {
            throw new IllegalArgumentException("partialled-out design is (near-)singular");
        }
        double[][] coef = LinAlg.solveMatrix(dtd, LinAlg.crossProduct(d, z));
        int n = z.length;
        int k = z[0].length;
        int p = d[0].length;
        double[][] out = new double[n][k];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) {
                double fit = 0.0;
                for (int c = 0; c < p; c++) {
                    fit += d[i][c] * coef[c][j];
                }
                out[i][j] = z[i][j] - fit;
            }
        }
        return out;
    }

    /**
     * Run the Johansen-lite trace test on a (T, n) matrix of levels.
     *
     * @param levels      price levels, T rows by n columns with n in {2, 3},
     *                    T &ge; 20
     * @param maxlagDiff  number of lagged differences k; only k=1 is
     *                    implemented (pinned) — any other value throws
     * @throws IllegalArgumentException on wrong shape, n not in {2,3},
     *     T &lt; 20, non-finite values, constant column, or maxlagDiff != 1
     */
    public static Result traceTest(double[][] levels, int maxlagDiff) {
        int t = levels.length;
        if (t == 0) {
            throw new IllegalArgumentException("levels must be a 2-D (T, n) array");
        }
        int n = levels[0].length;
        for (double[] row : levels) {
            if (row.length != n) {
                throw new IllegalArgumentException("levels must be a 2-D (T, n) array");
            }
        }
        if (n != 2 && n != 3) {
            throw new IllegalArgumentException("johansen trace test supports n = 2 or 3 assets only");
        }
        if (maxlagDiff != 1) {
            throw new IllegalArgumentException("only maxlag_diff = 1 is implemented (pinned)");
        }
        if (t < 20) {
            throw new IllegalArgumentException("need at least 20 observations");
        }
        if (!LinAlg.allFinite(levels)) {
            throw new IllegalArgumentException("inputs must be finite (no NaN/inf)");
        }
        for (int j = 0; j < n; j++) {
            double lo = levels[0][j];
            double hi = levels[0][j];
            for (double[] row : levels) {
                lo = Math.min(lo, row[j]);
                hi = Math.max(hi, row[j]);
            }
            if (hi - lo == 0.0) {
                throw new IllegalArgumentException("constant column in levels");
            }
        }

        int tEff = t - 2;
        double[][] dy = new double[t - 1][n];
        for (int r = 0; r < t - 1; r++) {
            for (int j = 0; j < n; j++) {
                dy[r][j] = levels[r + 1][j] - levels[r][j];
            }
        }
        double[][] z0 = new double[tEff][n]; // dY_t, t = 2..T-1
        double[][] z1 = new double[tEff][n]; // Y_{t-1}
        double[][] zd = new double[tEff][n + 1]; // [1, dY_{t-1}]
        for (int r = 0; r < tEff; r++) {
            System.arraycopy(dy[r + 1], 0, z0[r], 0, n);
            System.arraycopy(levels[r + 1], 0, z1[r], 0, n);
            zd[r][0] = 1.0;
            System.arraycopy(dy[r], 0, zd[r], 1, n);
        }

        double[][] r0 = residualise(z0, zd);
        double[][] r1 = residualise(z1, zd);
        double[][] s00 = scale(LinAlg.crossProduct(r0, r0), 1.0 / tEff);
        double[][] s01 = scale(LinAlg.crossProduct(r0, r1), 1.0 / tEff);
        double[][] s10 = LinAlg.transpose(s01);
        double[][] s11 = scale(LinAlg.crossProduct(r1, r1), 1.0 / tEff);

        double[][] chol = LinAlg.choleskyLower(s11);
        double[][] m = LinAlg.matmul(s10, LinAlg.solveMatrix(s00, s01)); // S10 S00^{-1} S01
        double[][] linv = LinAlg.solveMatrix(chol, LinAlg.identity(n));  // L^{-1}
        double[][] a = LinAlg.matmul(LinAlg.matmul(linv, m), LinAlg.transpose(linv));
        for (int i = 0; i < n; i++) { // kill rounding asymmetry
            for (int j = 0; j < n; j++) {
                double v = 0.5 * (a[i][j] + a[j][i]);
                a[i][j] = v;
                a[j][i] = v;
            }
        }
        LinAlg.Eigen eig = LinAlg.jacobiEigh(a);
        double[] lam = eig.values().clone();
        for (int i = 0; i < n; i++) {
            lam[i] = Math.min(Math.max(lam[i], 0.0), 1.0 - 1.0e-12);
        }
        double[][] beta = LinAlg.matmul(LinAlg.transpose(linv), eig.vectors());
        for (int i = 0; i < n; i++) {
            if (Math.abs(beta[0][i]) > 1.0e-12) {
                double head = beta[0][i];
                for (int r = 0; r < n; r++) {
                    beta[r][i] /= head;
                }
            }
        }

        double[] trace = new double[n];
        double[] crit = new double[n];
        for (int r = 0; r < n; r++) {
            double s = 0.0;
            for (int i = r; i < n; i++) {
                s += Math.log(1.0 - lam[i]);
            }
            trace[r] = -tEff * s;
            crit[r] = TRACE_CRIT_95[n - r];
        }
        int rank = n;
        for (int r = 0; r < n; r++) {
            if (trace[r] < crit[r]) {
                rank = r;
                break;
            }
        }
        return new Result(lam, trace, crit, rank, beta, tEff);
    }

    /** Convenience overload with the pinned single lagged difference. */
    public static Result traceTest(double[][] levels) {
        return traceTest(levels, 1);
    }

    private static double[][] scale(double[][] a, double f) {
        for (double[] row : a) {
            for (int j = 0; j < row.length; j++) {
                row[j] *= f;
            }
        }
        return a;
    }
}
