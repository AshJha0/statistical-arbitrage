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
     *     observations, non-finite input, delta outside (0, 1), or r &le; 0
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
        if (!(delta > 0.0 && delta < 1.0)) {
            throw new IllegalArgumentException("delta must lie strictly inside (0, 1)");
        }
        if (r <= 0.0) {
            throw new IllegalArgumentException("observation variance r must be > 0");
        }

        int n = y.length;
        double q = delta / (1.0 - delta);
        double s0 = 0.0;
        double s1 = 0.0;
        // P as explicit 2x2 entries; symmetric by construction
        double p00 = P0_SCALE;
        double p01 = 0.0;
        double p10 = 0.0;
        double p11 = P0_SCALE;

        double[] alpha = new double[n];
        double[] beta = new double[n];
        double[] z = new double[n];
        double[] innov = new double[n];
        double[] svar = new double[n];

        for (int t = 0; t < n; t++) {
            // predict: P_pred = P + Q (state prediction = s)
            double a00 = p00 + q;
            double a01 = p01;
            double a10 = p10;
            double a11 = p11 + q;
            double h1 = x[t]; // H = [1, h1]
            double e = y[t] - (s0 + h1 * s1);
            // S = H P_pred H' + r, computed as (H @ P_pred) @ H' like numpy
            double hp0 = a00 + h1 * a10;
            double hp1 = a01 + h1 * a11;
            double s = hp0 + hp1 * h1 + r;
            double k0 = (a00 + a01 * h1) / s; // K = P_pred H' / S
            double k1 = (a10 + a11 * h1) / s;
            s0 += k0 * e;
            s1 += k1 * e;
            // Joseph form: P = (I - K H) P_pred (I - K H)' + r K K'
            double i00 = 1.0 - k0;
            double i01 = -k0 * h1;
            double i10 = -k1;
            double i11 = 1.0 - k1 * h1;
            double b00 = i00 * a00 + i01 * a10;
            double b01 = i00 * a01 + i01 * a11;
            double b10 = i10 * a00 + i11 * a10;
            double b11 = i10 * a01 + i11 * a11;
            p00 = b00 * i00 + b01 * i01 + r * k0 * k0;
            p01 = b00 * i10 + b01 * i11 + r * k0 * k1;
            p10 = b10 * i00 + b11 * i01 + r * k1 * k0;
            p11 = b10 * i10 + b11 * i11 + r * k1 * k1;
            double sym = 0.5 * (p01 + p10); // symmetrise against rounding
            p01 = sym;
            p10 = sym;

            alpha[t] = s0;
            beta[t] = s1;
            innov[t] = e;
            svar[t] = s;
            z[t] = e / Math.sqrt(s);
        }

        double[][] finalCov = {{p00, p01}, {p10, p11}};
        return new Result(alpha, beta, z, innov, svar, finalCov);
    }
}
