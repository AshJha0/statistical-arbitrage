package com.quant.statarb;

/**
 * Kalman-filter dynamic hedge ratio with Joseph-form covariance update.
 *
 * <p>Model (PINNED):
 *
 * <pre>
 *     state       s_t = (alpha_t, beta_t)',  s_t = s_{t-1} + w_t  (random walk)
 *     observation y_t = H_t s_t + eps_t,     H_t = [1, x_t]
 *     Q = (delta / (1 - delta)) * I_2        (RiskMetrics-style scaling)
 *     R = r                                  (scalar observation variance)
 *     s_0 = (0, 0)',  P_0 = 1e3 * I_2        (diffuse-ish prior)
 * </pre>
 *
 * <p>Recursion per observation t = 0..T-1 (the very first observation is
 * already filtered against the prior):
 *
 * <pre>
 *     predict:  s_pred = s_prev,             P_pred = P_prev + Q
 *     innovate: e_t = y_t - H_t s_pred,      S_t = H_t P_pred H_t' + r
 *     gain:     K_t = P_pred H_t' / S_t
 *     update:   s_t = s_pred + K_t e_t
 *               P_t = (I - K_t H_t) P_pred (I - K_t H_t)' + K_t r K_t'
 * </pre>
 *
 * <p>The covariance update is the JOSEPH form (pinned): algebraically
 * identical to {@code (I - K H) P_pred} but remains symmetric positive
 * semi-definite under rounding, which the plain form does not guarantee.
 * {@code delta -> 0} recovers recursive least squares (static OLS in the
 * limit). Innovation z-scores {@code z_t = e_t / sqrt(S_t)} provide an
 * alternative trading signal to the rolling z-score.
 */
public final class Kalman {

    /** Pinned prior covariance scale: P_0 = P0_SCALE * I. */
    public static final double P0_SCALE = 1.0e3;

    private Kalman() {
    }

    /**
     * Filtered dynamic hedge output.
     *
     * @param alpha      filtered alpha_t path, length T
     * @param beta       filtered beta_t path, length T
     * @param z          innovation z-scores e_t / sqrt(S_t), length T
     * @param innovation raw innovations e_t
     * @param sVar       innovation variances S_t
     * @param finalCov   final 2x2 posterior covariance P_{T-1}
     */
    public record Result(double[] alpha, double[] beta, double[] z,
                         double[] innovation, double[] sVar, double[][] finalCov) {
    }

    /**
     * Filter the dynamic hedge state (alpha_t, beta_t) through (y, x).
     *
     * @param y     dependent price series
     * @param x     hedge instrument price series, same length
     * @param delta state-noise discount in the OPEN interval (0, 1);
     *              {@code Q = delta/(1-delta) * I}
     * @param r     observation noise variance, &gt; 0
     * @throws IllegalArgumentException on length mismatch, fewer than 2
     *     observations, non-finite input, delta outside the open interval
     *     (0, 1), or r not a finite number &gt; 0 (NaN and +inf rejected)
     */
    public static Result hedge(double[] y, double[] x, double delta, double r) {
        if (y.length != x.length) {
            throw new IllegalArgumentException(
                    "series length mismatch: len(y)=" + y.length + ", len(x)=" + x.length);
        }
        if (y.length < 2) {
            throw new IllegalArgumentException("need at least 2 observations");
        }
        if (!(LinAlg.allFinite(y) && LinAlg.allFinite(x))) {
            throw new IllegalArgumentException("inputs must be finite (no NaN/inf)");
        }
        checkParams(delta, r);

        int n = y.length;
        KalmanStep step = new KalmanStep(delta, r);
        double[] alpha = new double[n];
        double[] beta = new double[n];
        double[] z = new double[n];
        double[] innov = new double[n];
        double[] svar = new double[n];

        // One recursion for batch and streaming, so the two cannot diverge.
        for (int t = 0; t < n; t++) {
            z[t] = step.update(x[t], y[t]);
            alpha[t] = step.alpha();
            beta[t] = step.beta();
            innov[t] = step.innovation();
            svar[t] = step.sVar();
        }

        double[][] finalCov = step.covariance();
        return new Result(alpha, beta, z, innov, svar, finalCov);
    }

    /** Shared delta / r validation (negated comparisons so NaN fails). */
    static void checkParams(double delta, double r) {
        if (!(delta > 0.0 && delta < 1.0)) {
            throw new IllegalArgumentException("delta must lie strictly inside (0, 1)");
        }
        if (!(r > 0.0 && Double.isFinite(r))) {
            throw new IllegalArgumentException("observation variance r must be > 0 and finite");
        }
    }
}
