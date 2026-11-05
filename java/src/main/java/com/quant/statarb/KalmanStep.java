package com.quant.statarb;

/**
 * Streaming form of {@link Kalman#hedge}: scalar state, O(1) per update and no
 * allocation in {@link #update}.
 *
 * <p>Same model and prior as the batch filter. {@link Kalman#hedge} loops over
 * this class, so the two agree exactly; the scalar order of {@code update}
 * is pinned (API_SPEC section 6) because {@code P_0 = 1e3 I} makes early
 * updates sensitive to rounding.
 *
 * <p>Use {@code update(x, y)} (note: x first), then read {@link #alpha()},
 * {@link #beta()} and the returned z.
 */
public final class KalmanStep {

    private final double q;
    private final double r;
    private double a;
    private double b;
    private double p00 = Kalman.P0_SCALE;
    private double p01;
    private double p11 = Kalman.P0_SCALE;
    private double innovation;
    private double sVar;

    /**
     * Creates a filter in the pinned prior state.
     *
     * @param delta state-noise discount in the OPEN interval (0, 1)
     * @param r     observation variance, finite and &gt; 0
     * @throws IllegalArgumentException on invalid delta or r (NaN rejected)
     */
    public KalmanStep(double delta, double r) {
        Kalman.checkParams(delta, r);
        this.q = delta / (1.0 - delta);
        this.r = r;
    }

    /**
     * Filter one observation and return the innovation z-score.
     *
     * @param x hedge instrument price
     * @param y dependent price
     * @return z = e / sqrt(S)
     * @throws IllegalArgumentException on non-finite x or y (state unchanged)
     */
    public double update(double x, double y) {
        if (!(Double.isFinite(x) && Double.isFinite(y))) {
            throw new IllegalArgumentException("inputs must be finite (no NaN/inf)");
        }
        double pp00 = p00 + q;
        double pp01 = p01;
        double pp11 = p11 + q;
        double e = y - (a + x * b);
        double ph0 = pp00 + pp01 * x;
        double ph1 = pp01 + pp11 * x;
        double s = ph0 + x * ph1 + r;
        double k0 = ph0 / s;
        double k1 = ph1 / s;
        a += k0 * e;
        b += k1 * e;
        double a00 = 1.0 - k0;
        double a01 = -k0 * x;
        double a10 = -k1;
        double a11 = 1.0 - k1 * x;
        double b00 = a00 * pp00 + a01 * pp01;
        double b01 = a00 * pp01 + a01 * pp11;
        double b10 = a10 * pp00 + a11 * pp01;
        double b11 = a10 * pp01 + a11 * pp11;
        double n00 = b00 * a00 + b01 * a01 + r * (k0 * k0);
        double n01 = b00 * a10 + b01 * a11 + r * (k0 * k1);
        double n10 = b10 * a00 + b11 * a01 + r * (k1 * k0);
        double n11 = b10 * a10 + b11 * a11 + r * (k1 * k1);
        p00 = n00;
        p01 = 0.5 * (n01 + n10);
        p11 = n11;
        innovation = e;
        sVar = s;
        return e / Math.sqrt(s);
    }

    /** Filtered alpha after the latest update. */
    public double alpha() {
        return a;
    }

    /** Filtered beta after the latest update. */
    public double beta() {
        return b;
    }

    /** Raw innovation e_t of the latest update. */
    public double innovation() {
        return innovation;
    }

    /** Innovation variance S_t of the latest update. */
    public double sVar() {
        return sVar;
    }

    /** Current 2x2 posterior covariance (allocates a new array). */
    public double[][] covariance() {
        return new double[][] {{p00, p01}, {p01, p11}};
    }
}
