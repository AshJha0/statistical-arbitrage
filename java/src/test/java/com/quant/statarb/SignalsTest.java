package com.quant.statarb;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import java.util.Random;
import org.junit.Test;

/** Rolling z-score and spread: windowing, guards, error paths. */
public class SignalsTest {

    @Test
    public void zscoreMatchesManualWindow() {
        Random rng = new Random(10);
        double[] s = new double[200];
        for (int i = 0; i < 200; i++) {
            s[i] = rng.nextGaussian();
        }
        int w = 30;
        double[] z = Signals.rollingZscore(s, w);
        for (int t = 0; t < w - 1; t++) {
            assertTrue("head must be NaN", Double.isNaN(z[t]));
        }
        for (int t : new int[] {w - 1, 100, 199}) {
            double m = 0.0;
            for (int i = t - w + 1; i <= t; i++) {
                m += s[i];
            }
            m /= w;
            double ss = 0.0;
            for (int i = t - w + 1; i <= t; i++) {
                ss += (s[i] - m) * (s[i] - m);
            }
            double expected = (s[t] - m) / Math.sqrt(ss / (w - 1));
            assertEquals(expected, z[t], 1e-12);
        }
    }

    @Test
    public void zeroVarianceWindowGuard() {
        double[] s = new double[53];
        java.util.Arrays.fill(s, 2.0);
        s[51] = 3.0;
        s[52] = 4.0;
        double[] z = Signals.rollingZscore(s, 10);
        assertTrue("flat window -> NaN, not inf", Double.isNaN(z[49]));
        assertTrue(Double.isFinite(z[52]));
    }

    @Test
    public void windowErrors() {
        double[] s = new double[20];
        for (int i = 0; i < 20; i++) {
            s[i] = i;
        }
        assertThrows(IllegalArgumentException.class, () -> Signals.rollingZscore(s, 21));
        assertThrows(IllegalArgumentException.class, () -> Signals.rollingZscore(s, 1));
        double[] withNan = new double[20];
        for (int i = 0; i < 20; i += 2) {
            withNan[i + 1] = Double.NaN;
        }
        assertThrows(IllegalArgumentException.class, () -> Signals.rollingZscore(withNan, 4));
    }

    @Test
    public void spreadDefinitionAndErrors() {
        double[] y = TestData.slice(TestData.pair().get("stock_y"), 0, 50);
        double[] x = TestData.slice(TestData.pair().get("stock_x"), 0, 50);
        double[] s = Signals.spread(y, x, 12.0, 1.7);
        for (int t = 0; t < 50; t++) {
            assertEquals(y[t] - 12.0 - 1.7 * x[t], s[t], 1e-14);
        }
        assertThrows(IllegalArgumentException.class,
                () -> Signals.spread(y, TestData.slice(x, 0, 49), 0.0, 1.0));
        assertThrows(IllegalArgumentException.class,
                () -> Signals.spread(y, x, Double.NaN, 1.0));
    }
}
