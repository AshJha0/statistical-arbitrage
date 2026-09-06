package com.quant.statarb;

/**
 * Ordinary least squares — pinned solution method.
 *
 * <p>The solution method is PINNED for cross-language reproducibility:
 *
 * <pre>
 *     b      = solve(X'X, X'y)          (normal equations, LU solve)
 *     sigma2 = RSS / (n - k)
 *     cov(b) = sigma2 * inv(X'X)
 * </pre>
 *
 * No QR, no SVD. The design matrices in this project are tiny (k &le; 4) and
 * well-conditioned, so the normal equations lose no meaningful precision.
 * A conditioning guard ({@code cond(X'X) > 1e12}) rejects (near-)collinear
 * designs with an {@link IllegalArgumentException} rather than returning
 * garbage.
 */
public final class Ols {

    /** Condition-number threshold on X'X above which the design is collinear. Pinned. */
    public static final double COND_LIMIT = 1.0e12;

    private Ols() {
    }

    /**
     * Result of an OLS fit.
     *
     * @param params  coefficient vector, intercept first when requested
     * @param se      plain (non-robust) standard errors, same order as params
     * @param tstats  {@code params / se} elementwise
     * @param resid   residual vector {@code y - X b} (length nobs)
     * @param r2      coefficient of determination (centered TSS with an
     *                intercept, uncentered without)
     * @param sigma2  residual variance estimate {@code RSS / (nobs - nparams)}
     * @param nobs    number of observations
     * @param nparams number of estimated coefficients (including intercept)
     */
    public record Result(double[] params, double[] se, double[] tstats, double[] resid,
                         double r2, double sigma2, int nobs, int nparams) {
    }

    /**
     * Fit {@code y = a + b x + e} on a single regressor.
     *
     * @see #fit(double[], double[][], boolean)
     */
    public static Result fit(double[] y, double[] x, boolean intercept) {
        double[][] xm = new double[x.length][1];
        for (int i = 0; i < x.length; i++) {
            xm[i][0] = x[i];
        }
        return fit(y, xm, intercept);
    }

    /**
     * Fit {@code y = X b + e} by ordinary least squares (normal equations).
     *
     * @param y         response, length n
     * @param x         regressors, n rows; the intercept column is prepended
     *                  automatically when {@code intercept} — do not include one
     * @param intercept include a constant term
     * @return result with the intercept coefficient first (when requested)
     * @throws IllegalArgumentException on length mismatch, non-finite input,
     *     too few observations (n &le; k), constant response with intercept
     *     (undefined R^2), or (near-)collinear design (cond(X'X) &gt; 1e12 —
     *     this also catches a constant regressor when an intercept is present)
     */
    public static Result fit(double[] y, double[][] x, boolean intercept) {
        int n = y.length;
        if (x.length != n) {
            throw new IllegalArgumentException(
                    "series length mismatch: len(y)=" + n + ", len(x)=" + x.length);
        }
        if (n == 0) {
            throw new IllegalArgumentException("need more observations than parameters: n=0");
        }
        if (!(LinAlg.allFinite(y) && LinAlg.allFinite(x))) {
            throw new IllegalArgumentException("inputs must be finite (no NaN/inf)");
        }
        int m = x[0].length;
        for (double[] row : x) {
            if (row.length != m) {
                throw new IllegalArgumentException("regressor rows must have equal length");
            }
        }
        double[][] design;
        if (intercept) {
            design = new double[n][m + 1];
            for (int i = 0; i < n; i++) {
                design[i][0] = 1.0;
                System.arraycopy(x[i], 0, design[i], 1, m);
            }
        } else {
            design = x;
        }
        int k = design[0].length;
        if (n <= k) {
            throw new IllegalArgumentException(
                    "need more observations than parameters: n=" + n + ", k=" + k);
        }

        double[][] xtx = LinAlg.crossProduct(design);
        if (LinAlg.condSymmetric(xtx) > COND_LIMIT) {
            throw new IllegalArgumentException(
                    "design matrix is (near-)singular: constant or collinear regressors");
        }
        double[] xty = LinAlg.crossProduct(design, y);
        double[] params = LinAlg.solve(xtx, xty);

        double[] resid = new double[n];
        for (int i = 0; i < n; i++) {
            double fitVal = 0.0;
            for (int j = 0; j < k; j++) {
                fitVal += design[i][j] * params[j];
            }
            resid[i] = y[i] - fitVal;
        }
        double rss = LinAlg.dot(resid, resid);
        double sigma2 = rss / (n - k);
        double[][] cov = LinAlg.inverse(xtx);
        double[] se = new double[k];
        double[] tstats = new double[k];
        for (int j = 0; j < k; j++) {
            se[j] = Math.sqrt(sigma2 * cov[j][j]);
            tstats[j] = params[j] / se[j];
        }

        double tss;
        if (intercept) {
            double ybar = LinAlg.mean(y);
            double[] dev = new double[n];
            for (int i = 0; i < n; i++) {
                dev[i] = y[i] - ybar;
            }
            tss = LinAlg.dot(dev, dev);
        } else {
            tss = LinAlg.dot(y, y);
        }
        if (tss <= 0.0) {
            throw new IllegalArgumentException("y is constant: R^2 undefined");
        }
        double r2 = 1.0 - rss / tss;
        return new Result(params, se, tstats, resid, r2, sigma2, n, k);
    }
}
