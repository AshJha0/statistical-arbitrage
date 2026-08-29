package com.quant.statarb;

/**
 * Spread construction, rolling z-score, and mean-reversion half-life.
 */
public final class Signals {

    /**
     * Rolling standard deviations at or below this are treated as
     * zero-variance windows: the z-score is NaN (no signal). Pinned.
     */
    public static final double ZSCORE_STD_FLOOR = 1.0e-12;

    private Signals() {
    }

    /**
     * Hedge spread {@code s_t = y_t - (alpha + beta * x_t)}.
     *
     * @throws IllegalArgumentException on length mismatch or non-finite input
     */
    public static double[] spread(double[] y, double[] x, double alpha, double beta) {
        if (y.length != x.length) {
            throw new IllegalArgumentException("y and x must be 1-D arrays of equal length");
        }
        if (!(LinAlg.allFinite(y) && LinAlg.allFinite(x))) {
            throw new IllegalArgumentException("inputs must be finite (no NaN/inf)");
        }
        if (!(Double.isFinite(alpha) && Double.isFinite(beta))) {
            throw new IllegalArgumentException("alpha and beta must be finite");
        }
        double[] s = new double[y.length];
        for (int t = 0; t < y.length; t++) {
            s[t] = y[t] - (alpha + beta * x[t]);
        }
        return s;
    }

    /**
     * Rolling z-score with the current observation included (pinned).
     *
     * <pre>
     *     z_t = (s_t - mean(s_{t-W+1..t})) / std(s_{t-W+1..t}, ddof=1)
     * </pre>
     *
     * <p>The first W-1 entries are NaN (insufficient history). A window whose
     * sample standard deviation is &le; {@link #ZSCORE_STD_FLOOR} yields NaN
     * (zero-variance guard) rather than +/-inf.
     *
     * @throws IllegalArgumentException when window &lt; 2, window &gt; length,
     *     or the input is non-finite
     */
    public static double[] rollingZscore(double[] s, int window) {
        if (window < 2) {
            throw new IllegalArgumentException("window must be an integer >= 2");
        }
        if (window > s.length) {
            throw new IllegalArgumentException(
                    "window (" + window + ") larger than series length (" + s.length + ")");
        }
        if (!LinAlg.allFinite(s)) {
            throw new IllegalArgumentException("inputs must be finite (no NaN/inf)");
        }
        int n = s.length;
        double[] z = new double[n];
        java.util.Arrays.fill(z, Double.NaN);
        for (int t = window - 1; t < n; t++) {
            int lo = t - window + 1;
            double m = 0.0;
            for (int i = lo; i <= t; i++) {
                m += s[i];
            }
            m /= window;
            double ss = 0.0;
            for (int i = lo; i <= t; i++) {
                double d = s[i] - m;
                ss += d * d;
            }
            double sd = Math.sqrt(ss / (window - 1));
            if (sd > ZSCORE_STD_FLOOR) {
                z[t] = (s[t] - m) / sd;
            }
        }
        return z;
    }

    /**
     * Half-life of mean reversion from an AR(1) fit on the spread.
     *
     * <p>Regression (pinned): {@code ds_t = c + rho * s_{t-1} + e_t} (OLS with
     * intercept); then {@code half_life = -ln(2) / ln(1 + rho)}.
     *
     * <p>Guards (pinned): {@code rho >= 0} (no reversion) returns +inf;
     * {@code 1 + rho <= 0} (over-shooting flip) returns 0.0.
     *
     * @throws IllegalArgumentException on fewer than 12 observations,
     *     constant series, or non-finite input
     */
    public static double halfLife(double[] s) {
        if (s.length < 12) {
            throw new IllegalArgumentException("need at least 12 observations for half-life");
        }
        if (!LinAlg.allFinite(s)) {
            throw new IllegalArgumentException("inputs must be finite (no NaN/inf)");
        }
        if (Adf.isConstant(s)) {
            throw new IllegalArgumentException("series is constant: half-life undefined");
        }
        int n = s.length;
        double[] ds = new double[n - 1];
        double[] lag = new double[n - 1];
        for (int t = 0; t < n - 1; t++) {
            ds[t] = s[t + 1] - s[t];
            lag[t] = s[t];
        }
        Ols.Result fit = Ols.fit(ds, lag, true);
        double rho = fit.params()[1];
        if (rho >= 0.0) {
            return Double.POSITIVE_INFINITY;
        }
        if (1.0 + rho <= 0.0) {
            return 0.0;
        }
        return -Math.log(2.0) / Math.log(1.0 + rho);
    }
}
