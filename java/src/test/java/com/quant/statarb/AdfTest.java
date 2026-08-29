package com.quant.statarb;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import java.util.Random;
import org.junit.Test;

/** ADF verdicts on bundled data, p-value interpolation, half-life, errors. */
public class AdfTest {

    @Test
    public void whiteNoiseStronglyRejects() {
        Random rng = new Random(2);
        double[] wn = new double[800];
        for (int i = 0; i < 800; i++) {
            wn[i] = rng.nextGaussian();
        }
        Adf.Result res = Adf.test(wn, 1);
        assertTrue("t should be beyond the 1% value", res.tstat() < Adf.CRITICAL_VALUES[0][0]);
        assertTrue(res.reject(0.01));
        assertTrue(res.reject(0.05));
        assertTrue(res.reject(0.10));
        assertTrue(res.pvalue() <= 0.01 + 1e-12);
    }

    @Test
    public void randomWalkDoesNotReject() {
        Adf.Result res = Adf.test(TestData.walks().get("walk_a"), 1);
        assertFalse(res.reject(0.05));
        assertTrue(res.pvalue() > 0.10);
    }

    @Test
    public void effectiveNobs() {
        Random rng = new Random(4);
        double[] y = new double[100];
        for (int i = 0; i < 100; i++) {
            y[i] = rng.nextGaussian();
        }
        for (int lag : new int[] {0, 1, 4}) {
            assertEquals(100 - 1 - lag, Adf.test(y, lag).nobs());
        }
    }

    @Test
    public void pvalueInterpolationPinnedPoints() {
        for (int nvars = 1; nvars <= 3; nvars++) {
            double[] cv = Adf.CRITICAL_VALUES[nvars - 1];
            assertEquals(0.01, Adf.interpolatePvalue(cv[0], nvars), 1e-12);
            assertEquals(0.05, Adf.interpolatePvalue(cv[1], nvars), 1e-12);
            assertEquals(0.10, Adf.interpolatePvalue(cv[2], nvars), 1e-12);
        }
        // monotone non-decreasing in the t-stat, and clamped at both ends
        double prev = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < 60; i++) {
            double t = -8.0 + 12.0 * i / 59.0;
            double p = Adf.interpolatePvalue(t, 1);
            assertTrue(p >= prev - 1e-15);
            prev = p;
        }
        assertEquals(1e-4, Adf.interpolatePvalue(-50.0, 1), 1e-15);
        assertEquals(0.9999, Adf.interpolatePvalue(50.0, 1), 1e-15);
    }

    @Test
    public void criticalValueTablePinned() {
        assertEquals(-3.43, Adf.CRITICAL_VALUES[0][0], 0.0);
        assertEquals(-2.86, Adf.CRITICAL_VALUES[0][1], 0.0);
        assertEquals(-2.57, Adf.CRITICAL_VALUES[0][2], 0.0);
        assertEquals(-3.90, Adf.CRITICAL_VALUES[1][0], 0.0);
        assertEquals(-3.34, Adf.CRITICAL_VALUES[1][1], 0.0);
        assertEquals(-3.04, Adf.CRITICAL_VALUES[1][2], 0.0);
        assertEquals(-4.29, Adf.CRITICAL_VALUES[2][0], 0.0);
        assertEquals(-3.74, Adf.CRITICAL_VALUES[2][1], 0.0);
        assertEquals(-3.45, Adf.CRITICAL_VALUES[2][2], 0.0);
    }

    @Test
    public void errorPaths() {
        Random rng = new Random(5);
        double[] good = new double[100];
        for (int i = 0; i < 100; i++) {
            good[i] = rng.nextGaussian();
        }
        double[] constant = new double[100];
        java.util.Arrays.fill(constant, 1.0);
        assertThrows(IllegalArgumentException.class, () -> Adf.test(constant, 1));
        assertThrows(IllegalArgumentException.class, () -> Adf.test(good, -1));
        assertThrows(IllegalArgumentException.class, () -> Adf.test(good, 1, 4));
        assertThrows(IllegalArgumentException.class, () -> Adf.test(good, 1, 0));
        assertThrows(IllegalArgumentException.class,
                () -> Adf.test(java.util.Arrays.copyOf(good, 10), 1));
        double[] withInf = good.clone();
        withInf[50] = Double.POSITIVE_INFINITY;
        assertThrows(IllegalArgumentException.class, () -> Adf.test(withInf, 1));
        Adf.Result ok = Adf.test(good, 1);
        assertThrows(IllegalArgumentException.class, () -> ok.reject(0.20));
    }

    @Test
    public void halfLifeOnKnownAr1() {
        // AR(1) with phi = 0.9 has half-life -ln2/ln(0.9) ~ 6.58 days
        Random rng = new Random(6);
        int n = 20_000;
        double[] s = new double[n];
        for (int t = 1; t < n; t++) {
            s[t] = 0.9 * s[t - 1] + rng.nextGaussian();
        }
        double hl = Signals.halfLife(s);
        double expected = -Math.log(2.0) / Math.log(0.9);
        assertEquals(expected, hl, 0.15 * expected);
    }

    @Test
    public void halfLifeGuards() {
        // random walk: rho-hat ~ 0 -> +inf guard or huge-but-finite; > 100 either way
        assertTrue(Signals.halfLife(TestData.walks().get("walk_a")) > 100.0);
        // rho > 0 (explosive trend) -> +inf
        double[] explosive = new double[50];
        for (int t = 0; t < 50; t++) {
            explosive[t] = Math.pow(1.05, t);
        }
        assertEquals(Double.POSITIVE_INFINITY, Signals.halfLife(explosive), 0.0);
        // 1 + rho <= 0 (overshooting flip) -> 0.0
        double[] flip = new double[40];
        for (int t = 0; t < 40; t++) {
            flip[t] = Math.pow(-1.2, t) * 1e-3;
        }
        assertEquals(0.0, Signals.halfLife(flip), 0.0);
        double[] constant = new double[50];
        java.util.Arrays.fill(constant, 2.0);
        assertThrows(IllegalArgumentException.class, () -> Signals.halfLife(constant));
        assertThrows(IllegalArgumentException.class,
                () -> Signals.halfLife(new double[] {1.0, 2.0, 3.0, 4.0, 5.0}));
    }
}
