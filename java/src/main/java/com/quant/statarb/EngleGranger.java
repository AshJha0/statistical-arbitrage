package com.quant.statarb;

/**
 * Engle-Granger two-step cointegration test.
 *
 * <p>Step 1: static OLS hedge regression {@code y_t = a + b x_t + e_t}
 * (normal equations, see {@link Ols}).
 * Step 2: ADF test (constant, fixed maxlag) on the residuals e_t, judged
 * against the Engle-Granger 2-variable critical-value table
 * ({@link Adf#CRITICAL_VALUES} with {@code nvars=2}) — the residuals are
 * estimated, so the plain Dickey-Fuller table would over-reject.
 */
public final class EngleGranger {

    private EngleGranger() {
    }

    /**
     * Engle-Granger two-step result.
     *
     * @param alpha intercept of the hedge regression
     * @param beta  hedge ratio b
     * @param resid cointegrating residuals {@code y - a - b x}
     * @param r2    R^2 of the hedge regression
     * @param adf   ADF result on the residuals (nvars=2 table)
     */
    public record Result(double alpha, double beta, double[] resid, double r2,
                         Adf.Result adf) {

        /** True when the residual ADF rejects a unit root at {@code alpha}. */
        public boolean cointegrated(double alpha) {
            return adf.reject(alpha);
        }
    }

    /**
     * Run the Engle-Granger two-step test of cointegration of y on x.
     *
     * @param y      dependent price series
     * @param x      regressor price series, same length
     * @param maxlag fixed ADF augmentation lag (this project's default: 1)
     * @throws IllegalArgumentException propagated from OLS/ADF validation
     *     (length mismatch, constant series, non-finite values, short series)
     */
    public static Result test(double[] y, double[] x, int maxlag) {
        Ols.Result fit = Ols.fit(y, x, true);
        Adf.Result adf = Adf.test(fit.resid(), maxlag, 2);
        return new Result(fit.params()[0], fit.params()[1], fit.resid(), fit.r2(), adf);
    }
}
