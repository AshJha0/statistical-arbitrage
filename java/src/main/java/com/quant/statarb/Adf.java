package com.quant.statarb;

/**
 * Augmented Dickey-Fuller test with MacKinnon (2010) critical values.
 *
 * <p>Regression form (PINNED, constant / no-trend case):
 *
 * <pre>
 *     dy_t = c + rho * y_{t-1} + sum_{i=1..p} phi_i * dy_{t-i} + e_t
 * </pre>
 *
 * with a FIXED augmentation lag {@code p = maxlag} (default 1; no automatic
 * lag selection). The test statistic is the plain OLS t-statistic on rho.
 * With T raw observations the regression uses the {@code T_eff = T - 1 - p}
 * rows for which all lagged differences exist.
 *
 * <p>Critical values (PINNED) are the MacKinnon (2010) finite-sample response
 * surfaces for the constant / no-trend case,
 * {@code cv(T_eff) = b_inf + b1/T_eff + b2/T_eff^2 + b3/T_eff^3}, embedded
 * verbatim for the 1-variable ADF test and for the Engle-Granger residual test
 * with 2 and 3 variables, where {@code T_eff} is the number of observations in
 * the test regression (as in statsmodels' {@code adfuller}). The surfaces were
 * fitted on samples of 20 or more, so {@code T_eff >= 20} is enforced.
 *
 * <p>The p-value is a documented approximation: piecewise-linear interpolation
 * of the significance level through the three (critical value, level) knots
 * at {@code T_eff}, with linear extrapolation using the edge-segment slopes,
 * clamped to [1e-4, 0.9999]. The exact MacKinnon (1994) p-value response
 * surface is deliberately NOT implemented; the verdict never uses the p-value.
 */
public final class Adf {

    /**
     * MacKinnon (2010) response-surface coefficients, constant / no-trend
     * case (Table 1 of "Critical Values for Cointegration Tests", Queen's
     * Economics Dept. WP 1227), indexed {@code [nvars - 1][level][k]} with
     * levels (1%, 5%, 10%) and k = (b_inf, b1, b2, b3). nvars 1 = plain ADF;
     * 2, 3 = Engle-Granger residual test. PINNED.
     */
    public static final double[][][] RESPONSE_SURFACE = {
        {
            {-3.43035, -6.5393, -16.786, -79.433},
            {-2.86154, -2.8903, -4.234, -40.040},
            {-2.56677, -1.5384, -2.809, 0.0},
        },
        {
            {-3.89644, -10.9519, -33.527, 0.0},
            {-3.33613, -6.1101, -6.823, 0.0},
            {-3.04445, -4.2412, -2.720, 0.0},
        },
        {
            {-4.29374, -14.4354, -33.195, 47.433},
            {-3.74066, -8.5632, -10.852, 27.982},
            {-3.45218, -6.2143, -3.718, 0.0},
        },
    };

    /**
     * Asymptotic (T to infinity) critical values {@code b_inf}, indexed
     * {@code [nvars - 1][level]} with levels (1%, 5%, 10%). {@link #test}
     * always uses the finite-sample values from {@link #criticalValues(int, int)}.
     */
    public static final double[][] CRITICAL_VALUES = {
        {-3.43035, -2.86154, -2.56677},
        {-3.89644, -3.33613, -3.04445},
        {-4.29374, -3.74066, -3.45218},
    };

    /** Smallest effective sample the response surfaces are trusted for. Pinned. */
    public static final int MIN_NOBS = 20;

    /** Largest augmentation lag accepted (documented upper bound). Pinned. */
    public static final int MAX_LAG = 1000;

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
     * @param crit1  finite-sample 1% critical value used for the verdict
     * @param crit5  finite-sample 5% critical value used for the verdict
     * @param crit10 finite-sample 10% critical value used for the verdict
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

    private static void checkNvars(int nvars) {
        if (nvars < 1 || nvars > 3) {
            throw new IllegalArgumentException("nvars must be 1, 2 or 3");
        }
    }

    /**
     * Asymptotic critical values (1%, 5%, 10%) for {@code nvars} variables.
     *
     * @throws IllegalArgumentException unless nvars is 1, 2 or 3
     */
    public static double[] criticalValues(int nvars) {
        checkNvars(nvars);
        return CRITICAL_VALUES[nvars - 1].clone();
    }

    /**
     * Finite-sample MacKinnon (2010) critical values (1%, 5%, 10%) at the
     * effective regression sample size {@code nobs}.
     *
     * @throws IllegalArgumentException unless nvars is 1, 2 or 3 and
     *     {@code nobs >= MIN_NOBS}
     */
    public static double[] criticalValues(int nvars, int nobs) {
        checkNvars(nvars);
        if (nobs < MIN_NOBS) {
            throw new IllegalArgumentException(
                    "critical values need at least " + MIN_NOBS + " effective observations");
        }
        double inv = 1.0 / nobs;
        double[] out = new double[3];
        for (int i = 0; i < 3; i++) {
            double[] r = RESPONSE_SURFACE[nvars - 1][i];
            // Horner form, identical to the Python reference.
            out[i] = r[0] + inv * (r[1] + inv * (r[2] + inv * r[3]));
        }
        return out;
    }

    private static double interpolateThrough(double tstat, double[] cv) {
        if (Double.isNaN(tstat)) {
            throw new IllegalArgumentException("tstat must not be NaN");
        }
        double cv1 = cv[0];
        double cv5 = cv[1];
        double cv10 = cv[2];
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
     * Approximate p-value by piecewise-linear interpolation through the
     * asymptotic knots (cv_1%, 0.01), (cv_5%, 0.05), (cv_10%, 0.10); outside
     * the knot range the edge segment's slope extrapolates linearly; the
     * result is clamped to [{@link #PVALUE_MIN}, {@link #PVALUE_MAX}].
     * A NaN t-stat is an error; +/-inf (exact-fit regressions) clamps.
     *
     * @param tstat the ADF t-statistic
     * @param nvars critical-value table selector in {1, 2, 3}
     */
    public static double interpolatePvalue(double tstat, int nvars) {
        return interpolateThrough(tstat, criticalValues(nvars));
    }

    /** Same, through the finite-sample knots at effective sample size {@code nobs}. */
    public static double interpolatePvalue(double tstat, int nvars, int nobs) {
        return interpolateThrough(tstat, criticalValues(nvars, nobs));
    }

    /**
     * Run the ADF regression and return the unit-root test result.
     *
     * @param y      series to test (levels for nvars=1; cointegrating
     *               residuals for nvars=2 or 3), finite
     * @param maxlag fixed augmentation lag p in [0, {@link #MAX_LAG}]
     *               (default in this project: 1)
     * @param nvars  critical-value table selector: 1 for a plain ADF test,
     *               2 or 3 for the Engle-Granger residual test
     * @throws IllegalArgumentException on negative or oversized maxlag,
     *     unknown nvars, constant series, non-finite input, or too few
     *     observations (effective nobs must be &ge; max(nparams + 10, MIN_NOBS))
     */
    public static Result test(double[] y, int maxlag, int nvars) {
        if (maxlag < 0) {
            throw new IllegalArgumentException("maxlag must be an integer >= 0");
        }
        if (maxlag > MAX_LAG) {
            throw new IllegalArgumentException("maxlag must be <= " + MAX_LAG);
        }
        checkNvars(nvars);
        if (!LinAlg.allFinite(y)) {
            throw new IllegalArgumentException("inputs must be finite (no NaN/inf)");
        }
        int tLen = y.length;
        int p = maxlag;
        // long arithmetic: no overflow for any admissible maxlag
        long nparams = 2L + p; // c, rho, phi_1..phi_p
        long eff = (long) tLen - 1 - p;
        if (tLen > 0 && isConstant(y)) {
            throw new IllegalArgumentException("series is constant: ADF test undefined");
        }
        if (eff < Math.max(nparams + 10, MIN_NOBS)) {
            throw new IllegalArgumentException(
                    "series too short for ADF with maxlag=" + p + ": effective nobs " + eff);
        }
        int rows = (int) eff;

        double[] dy = new double[tLen - 1];
        for (int t = 0; t < tLen - 1; t++) {
            dy[t] = y[t + 1] - y[t];
        }
        // usable rows t = p+1 .. T-1 of the level series
        double[] lhs = new double[rows];
        double[][] design = new double[rows][1 + p];
        for (int r = 0; r < rows; r++) {
            lhs[r] = dy[p + r];
            design[r][0] = y[p + r]; // y_{t-1}
            for (int i = 1; i <= p; i++) {
                design[r][i] = dy[p + r - i]; // dy_{t-i}
            }
        }
        Ols.Result fit = Ols.fit(lhs, design, true);
        double tstat = fit.tstats()[1]; // rho is first regressor after intercept
        double[] cv = criticalValues(nvars, rows);
        return new Result(tstat, interpolatePvalue(tstat, nvars, rows),
                cv[0], cv[1], cv[2], p, rows, nvars);
    }

    /** Convenience overload: plain ADF (nvars = 1). */
    public static Result test(double[] y, int maxlag) {
        return test(y, maxlag, 1);
    }

    /**
     * Lag in {@code 0..maxLag} with the smallest BIC, for use with {@link #test}.
     *
     * <p>Every candidate is fitted on the SAME rows {@code t = maxLag+1 .. T-1}
     * so the BICs are comparable: {@code BIC(p) = n ln(RSS_p / n) + (2+p) ln n}
     * with {@code n = T - 1 - maxLag}. A tie goes to the smaller lag.
     *
     * @param y      level series
     * @param maxLag largest candidate lag
     * @param nvars  1, 2 or 3 (validation only; it does not affect the result)
     * @return the selected lag
     * @throws IllegalArgumentException with the validation of
     *     {@code test(y, maxLag, nvars)}
     */
    public static int selectLag(double[] y, int maxLag, int nvars) {
        test(y, maxLag, nvars); // shared validation
        int m = maxLag;
        int tLen = y.length;
        int n = tLen - 1 - m;
        double[] dy = new double[tLen - 1];
        for (int t = 0; t < tLen - 1; t++) {
            dy[t] = y[t + 1] - y[t];
        }
        double[] lhs = new double[n];
        for (int r = 0; r < n; r++) {
            lhs[r] = dy[m + r];
        }
        int bestLag = 0;
        double bestBic = Double.POSITIVE_INFINITY;
        for (int p = 0; p <= m; p++) {
            double[][] design = new double[n][1 + p];
            for (int r = 0; r < n; r++) {
                design[r][0] = y[m + r];
                for (int i = 1; i <= p; i++) {
                    design[r][i] = dy[m + r - i];
                }
            }
            double[] e = Ols.fit(lhs, design, true).resid();
            double rss = LinAlg.dot(e, e);
            double fitTerm = rss > 0.0 ? n * Math.log(rss / n) : Double.NEGATIVE_INFINITY;
            double bic = fitTerm + (2 + p) * Math.log(n);
            if (bic < bestBic || p == 0) {
                bestLag = p;
                bestBic = bic;
            }
        }
        return bestLag;
    }

    /** True when every entry equals the first (false for an empty array). */
    static boolean isConstant(double[] y) {
        if (y.length == 0) {
            return false;
        }
        double lo = y[0];
        double hi = y[0];
        for (double v : y) {
            lo = Math.min(lo, v);
            hi = Math.max(hi, v);
        }
        return hi - lo == 0.0;
    }
}
