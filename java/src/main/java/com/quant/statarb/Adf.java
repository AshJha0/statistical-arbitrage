package com.quant.statarb;

/**
 * Augmented Dickey-Fuller test with embedded critical-value tables.
 *
 * <p>Regression form (PINNED, constant / no-trend case):
 *
 * <pre>
 *     dy_t = c + rho * y_{t-1} + sum_{i=1..p} phi_i * dy_{t-i} + e_t
 * </pre>
 *
 * with a FIXED augmentation lag {@code p = maxlag} (default 1; no automatic
 * lag selection). The test statistic is the plain OLS t-statistic on rho.
 * With T raw observations the regression uses the T - 1 - p rows for which
 * all lagged differences exist.
 *
 * <p>Critical values are MacKinnon-style ASYMPTOTIC values, constant no-trend
 * case, embedded verbatim for the 1-variable ADF test and for the
 * Engle-Granger residual test with 2 and 3 variables. The p-value is a
 * documented approximation: piecewise-linear interpolation of the
 * significance level through the three (critical value, level) knots, with
 * linear extrapolation using the edge-segment slopes, clamped to
 * [1e-4, 0.9999]. The exact MacKinnon response-surface regression is
 * deliberately NOT implemented.
 */
public final class Adf {

    /**
     * MacKinnon-style asymptotic critical values, constant no-trend case,
     * indexed {@code [nvars - 1][level]} with levels (1%, 5%, 10%).
     * nvars 1 = plain ADF; 2, 3 = Engle-Granger residual test. PINNED.
     */
    public static final double[][] CRITICAL_VALUES = {
        {-3.43, -2.86, -2.57},
        {-3.90, -3.34, -3.04},
        {-4.29, -3.74, -3.45},
    };

    /** Clamp bounds for the interpolated p-value. Pinned. */
    public static final double PVALUE_MIN = 1.0e-4;
    /** Upper clamp bound for the interpolated p-value. Pinned. */
    public static final double PVALUE_MAX = 0.9999;

    private Adf() {
    }

    /**
     * Result of an ADF unit-root test.
     *
     * @param tstat  OLS t-statistic on rho
     * @param pvalue interpolated approximate p-value (see class docs)
     * @param crit1  1% critical value used
     * @param crit5  5% critical value used
     * @param crit10 10% critical value used
     * @param maxlag augmentation lag p used (fixed, not selected)
     * @param nobs   effective regression observations (T - 1 - p)
     * @param nvars  table selector (1 = ADF, 2/3 = EG residual case)
     */
    public record Result(double tstat, double pvalue, double crit1, double crit5,
                         double crit10, int maxlag, int nobs, int nvars) {

        /**
         * Reject the unit-root null at {@code alpha} in {0.01, 0.05, 0.10}.
         *
         * <p>Verdict is by direct table comparison (tstat &lt; critical value),
         * NOT via the interpolated p-value.
         *
         * @throws IllegalArgumentException for any other alpha
         */
        public boolean reject(double alpha) {
            double cv;
            if (alpha == 0.01) {
                cv = crit1;
            } else if (alpha == 0.05) {
                cv = crit5;
            } else if (alpha == 0.10) {
                cv = crit10;
            } else {
                throw new IllegalArgumentException("alpha must be one of 0.01, 0.05, 0.10");
            }
            return tstat < cv;
        }
    }

    /**
     * Approximate p-value by piecewise-linear interpolation.
     *
     * <p>Knots are (cv_1%, 0.01), (cv_5%, 0.05), (cv_10%, 0.10); outside the
     * knot range the edge segment's slope extrapolates linearly; the result
     * is clamped to [{@link #PVALUE_MIN}, {@link #PVALUE_MAX}].
     *
     * @param tstat the ADF t-statistic
     * @param nvars critical-value table selector in {1, 2, 3}
     */
    public static double interpolatePvalue(double tstat, int nvars) {
        if (nvars < 1 || nvars > 3) {
            throw new IllegalArgumentException("nvars must be 1, 2 or 3");
        }
        double cv1 = CRITICAL_VALUES[nvars - 1][0];
        double cv5 = CRITICAL_VALUES[nvars - 1][1];
        double cv10 = CRITICAL_VALUES[nvars - 1][2];
        double p;
        if (tstat <= cv5) {
            double slope = (0.05 - 0.01) / (cv5 - cv1);
            p = 0.01 + slope * (tstat - cv1);
        } else {
            double slope = (0.10 - 0.05) / (cv10 - cv5);
            p = 0.05 + slope * (tstat - cv5);
        }
        return Math.min(Math.max(p, PVALUE_MIN), PVALUE_MAX);
    }

    /**
     * Run the ADF regression and return the unit-root test result.
     *
     * @param y      series to test (levels for nvars=1; cointegrating
     *               residuals for nvars=2 or 3), finite
     * @param maxlag fixed augmentation lag p &ge; 0 (default in this project: 1)
     * @param nvars  critical-value table selector: 1 for a plain ADF test,
     *               2 or 3 for the Engle-Granger residual test
     * @throws IllegalArgumentException on negative maxlag, unknown nvars,
     *     constant series, non-finite input, or too few observations
     *     (effective nobs must be &ge; nparams + 10)
     */
    public static Result test(double[] y, int maxlag, int nvars) {
        if (maxlag < 0) {
            throw new IllegalArgumentException("maxlag must be an integer >= 0");
        }
        if (nvars < 1 || nvars > 3) {
            throw new IllegalArgumentException("nvars must be 1, 2 or 3");
        }
        if (!LinAlg.allFinite(y)) {
            throw new IllegalArgumentException("inputs must be finite (no NaN/inf)");
        }
        int tLen = y.length;
        int p = maxlag;
        int nparams = 2 + p; // c, rho, phi_1..phi_p
        int eff = tLen - 1 - p;
        if (isConstant(y)) {
            throw new IllegalArgumentException("series is constant: ADF test undefined");
        }
        if (eff < nparams + 10) {
            throw new IllegalArgumentException(
                    "series too short for ADF with maxlag=" + p + ": effective nobs " + eff);
        }

        double[] dy = new double[tLen - 1];
        for (int t = 0; t < tLen - 1; t++) {
            dy[t] = y[t + 1] - y[t];
        }
        // usable rows t = p+1 .. T-1 of the level series
        double[] lhs = new double[eff];
        double[][] design = new double[eff][1 + p];
        for (int r = 0; r < eff; r++) {
            lhs[r] = dy[p + r];
            design[r][0] = y[p + r]; // y_{t-1}
            for (int i = 1; i <= p; i++) {
                design[r][i] = dy[p + r - i]; // dy_{t-i}
            }
        }
        Ols.Result fit = Ols.fit(lhs, design, true);
        double tstat = fit.tstats()[1]; // rho is first regressor after intercept
        return new Result(
                tstat,
                interpolatePvalue(tstat, nvars),
                CRITICAL_VALUES[nvars - 1][0],
                CRITICAL_VALUES[nvars - 1][1],
                CRITICAL_VALUES[nvars - 1][2],
                p, eff, nvars);
    }

    /** Convenience overload: plain ADF (nvars = 1). */
    public static Result test(double[] y, int maxlag) {
        return test(y, maxlag, 1);
    }

    static boolean isConstant(double[] y) {
        double lo = y[0];
        double hi = y[0];
        for (double v : y) {
            lo = Math.min(lo, v);
            hi = Math.max(hi, v);
        }
        return hi - lo == 0.0;
    }
}
