package com.quant.statarb;

/**
 * Streaming rolling z-score: ring buffer and running moments, O(1) per update,
 * no allocation in {@link #update}.
 *
 * <p>Same definition and NaN rules as {@link Signals#rollingZscore}, but with
 * running moments (Welford while filling, then a sliding replace), so it agrees
 * with the batch function to about 1e-9 and is NOT bitwise equal. The batch
 * function stays the trading reference.
 */
public final class RollingZ {

    private final int window;
    private final double[] buf;
    private int head;
    private int count;
    private double mean;
    private double m2;

    /**
     * Creates an empty window.
     *
     * @param window window length, an integer &ge; 2
     * @throws IllegalArgumentException when window &lt; 2
     */
    public RollingZ(int window) {
        if (window < 2) {
            throw new IllegalArgumentException("window must be an integer >= 2");
        }
        this.window = window;
        this.buf = new double[window];
    }

    /**
     * Add one observation.
     *
     * @param s the new observation
     * @return z, or NaN during warm-up and on zero variance
     * @throws IllegalArgumentException on a non-finite observation (state unchanged)
     */
    public double update(double s) {
        if (!Double.isFinite(s)) {
            throw new IllegalArgumentException("inputs must be finite (no NaN/inf)");
        }
        if (count < window) {
            count++;
            double d = s - mean;
            mean += d / count;
            m2 += d * (s - mean);
        } else {
            double old = buf[head];
            double oldMean = mean;
            mean = oldMean + (s - old) / window;
            m2 += (s - old) * (s - mean + old - oldMean);
            if (m2 < 0.0) {
                m2 = 0.0;
            }
        }
        buf[head] = s;
        head = (head + 1) % window;
        if (count < window) {
            return Double.NaN;
        }
        double sd = Math.sqrt(m2 / (window - 1));
        if (sd > Signals.ZSCORE_STD_FLOOR) {
            return (s - mean) / sd;
        }
        return Double.NaN;
    }
}
