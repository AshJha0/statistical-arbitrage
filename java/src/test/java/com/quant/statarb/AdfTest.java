package com.quant.statarb;

import static org.junit.Assert.assertArrayEquals;
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
        // asymptotic b_inf column of MacKinnon (2010) Table 1, no-trend case
        assertEquals(-3.43035, Adf.CRITICAL_VALUES[0][0], 0.0);
        assertEquals(-2.86154, Adf.CRITICAL_VALUES[0][1], 0.0);
        assertEquals(-2.56677, Adf.CRITICAL_VALUES[0][2], 0.0);
        assertEquals(-3.89644, Adf.CRITICAL_VALUES[1][0], 0.0);
        assertEquals(-3.33613, Adf.CRITICAL_VALUES[1][1], 0.0);
        assertEquals(-3.04445, Adf.CRITICAL_VALUES[1][2], 0.0);
        assertEquals(-4.29374, Adf.CRITICAL_VALUES[2][0], 0.0);
        assertEquals(-3.74066, Adf.CRITICAL_VALUES[2][1], 0.0);
        assertEquals(-3.45218, Adf.CRITICAL_VALUES[2][2], 0.0);
        // response-surface rows pinned verbatim (spot checks)
        assertArrayEquals(new double[] {-2.86154, -2.8903, -4.234, -40.040},
                Adf.RESPONSE_SURFACE[0][1], 0.0);
        assertEquals(47.433, Adf.RESPONSE_SURFACE[2][0][3], 0.0);
        for (int n = 1; n <= 3; n++) {
            assertArrayEquals(Adf.CRITICAL_VALUES[n - 1], Adf.criticalValues(n), 0.0);
        }
        assertThrows(IllegalArgumentException.class, () -> Adf.criticalValues(0));
    }

    @Test
    public void finiteSampleCriticalValuesFollowResponseSurface() {
        // cv(T) = b_inf + b1/T + b2/T^2 + b3/T^3, hand-evaluated for the
        // pinned N=1 5% row at T = 50 and T = 20, and the N=2 5% row at T = 898
        double e50 = -2.86154 - 2.8903 / 50.0 - 4.234 / 2500.0 - 40.040 / 125000.0;
        assertEquals(e50, Adf.criticalValues(1, 50)[1], 1e-12);
        double e20 = -2.86154 - 2.8903 / 20.0 - 4.234 / 400.0 - 40.040 / 8000.0;
        assertEquals(e20, Adf.criticalValues(1, 20)[1], 1e-12);
        double e898 = -3.33613 - 6.1101 / 898.0 - 6.823 / (898.0 * 898.0);
        assertEquals(e898, Adf.criticalValues(2, 898)[1], 1e-12);
        for (int n = 1; n <= 3; n++) {
            double[] inf = Adf.criticalValues(n);
            double[] c20 = Adf.criticalValues(n, 20);
            double[] big = Adf.criticalValues(n, 10_000_000);
            for (int i = 0; i < 3; i++) {
                assertTrue(c20[i] < inf[i]);
                assertEquals(inf[i], big[i], 1e-5);
            }
        }
        assertThrows(IllegalArgumentException.class, () -> Adf.criticalValues(1, 19));
        assertThrows(IllegalArgumentException.class, () -> Adf.criticalValues(4, 100));
        // finite-sample knots reproduce the levels exactly
        assertEquals(0.05, Adf.interpolatePvalue(Adf.criticalValues(2, 898)[1], 2, 898), 1e-12);
    }

    @Test
    public void smallSampleGuard() {
        // T_eff < 20 is refused; at T_eff = 20 the verdict uses cv(20)
        double[] s = new double[40];
        for (int t = 0; t < 40; t++) {
            s[t] = Math.pow(0.9, t) + 0.05 * Math.sin(2.1 * t);
        }
        double[] s21 = java.util.Arrays.copyOf(s, 21); // T_eff = 19
        assertThrows(IllegalArgumentException.class, () -> Adf.test(s21, 1));
        double[] s15 = java.util.Arrays.copyOf(s, 15); // T_eff = 13 (old guard passed)
        assertThrows(IllegalArgumentException.class, () -> Adf.test(s15, 1));
        Adf.Result res = Adf.test(java.util.Arrays.copyOf(s, 22), 1); // T_eff = 20
        assertEquals(20, res.nobs());
        double e20 = -2.86154 - 2.8903 / 20.0 - 4.234 / 400.0 - 40.040 / 8000.0;
        assertEquals(e20, res.crit5(), 1e-12);
        assertEquals(res.tstat() < res.crit5(), res.reject(0.05));
    }

    @Test
    public void bundledRandomWalkUsesFiniteSampleTable() {
        Adf.Result res = Adf.test(TestData.walks().get("walk_a"), 1);
        assertEquals(1498, res.nobs());
        double t = 1498.0;
        double e5 = -2.86154 - 2.8903 / t - 4.234 / (t * t) - 40.040 / (t * t * t);
        assertEquals(e5, res.crit5(), 1e-12);
        assertTrue(res.crit5() < -2.86154);
    }

    @Test
    public void maxLagUpperBoundAndNaNTstat() {
        double[] w = TestData.walks().get("walk_a");
        assertThrows(IllegalArgumentException.class, () -> Adf.test(w, Adf.MAX_LAG + 1));
        assertThrows(IllegalArgumentException.class, () -> Adf.test(w, Adf.MAX_LAG)); // too short
        assertThrows(IllegalArgumentException.class, () -> Adf.test(w, Integer.MAX_VALUE));
        assertThrows(IllegalArgumentException.class, () -> Adf.interpolatePvalue(Double.NaN, 1));
        assertThrows(IllegalArgumentException.class, () -> Adf.interpolatePvalue(-3.0, 4));
        // +/-inf t-stats (exact-fit regressions) clamp rather than throw
        assertEquals(Adf.PVALUE_MIN, Adf.interpolatePvalue(Double.NEGATIVE_INFINITY, 1), 0.0);
        assertEquals(Adf.PVALUE_MAX, Adf.interpolatePvalue(Double.POSITIVE_INFINITY, 1), 0.0);
    }

    @Test
    public void emptyArraysAreTypedErrors() {
        // MIN-12: previously ArrayIndexOutOfBoundsException
        assertThrows(IllegalArgumentException.class, () -> Adf.test(new double[0], 1));
        assertThrows(IllegalArgumentException.class,
                () -> Ols.fit(new double[0], new double[0], true));
        assertThrows(IllegalArgumentException.class,
                () -> Ols.fit(new double[0], new double[0][0], true));
        assertThrows(IllegalArgumentException.class, () -> Signals.halfLife(new double[0]));
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
