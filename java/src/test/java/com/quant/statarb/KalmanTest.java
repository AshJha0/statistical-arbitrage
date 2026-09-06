package com.quant.statarb;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import java.util.Random;
import org.junit.Test;

/** Kalman dynamic hedge: RLS limit, PSD covariance, adaptation, errors. */
public class KalmanTest {

    @Test
    public void deltaToZeroConvergesToOls() {
        // with delta -> 0 the filter is recursive least squares and the final
        // beta approaches the static OLS beta (loose band; P0 is finite)
        int nIs = TestData.insampleDays();
        double[] y = TestData.slice(TestData.pair().get("stock_y"), 0, nIs);
        double[] x = TestData.slice(TestData.pair().get("stock_x"), 0, nIs);
        double betaOls = Ols.fit(y, x, true).params()[1];
        Kalman.Result km = Kalman.hedge(y, x, 1e-16, 1e-3);
        assertEquals(betaOls, km.beta()[nIs - 1], 0.05);
    }

    @Test
    public void josephFormCovarianceStaysPsd() {
        // 2x2 closed-form eigen check: PSD iff trace >= 0 and det >= 0;
        // checked on the final posterior including a stressed run (tiny r)
        for (double r : new double[] {1e-3, 1e-8}) {
            Kalman.Result km = Kalman.hedge(TestData.pair().get("stock_y"),
                    TestData.pair().get("stock_x"), 1e-4, r);
            double[][] p = km.finalCov();
            double tr = p[0][0] + p[1][1];
            double det = p[0][0] * p[1][1] - p[0][1] * p[1][0];
            assertEquals(p[0][1], p[1][0], 1e-15); // symmetric
            assertTrue(tr >= -1e-15);
            assertTrue(det >= -1e-12 * Math.max(tr * tr, 1.0));
            // closed-form 2x2 eigenvalues both >= 0 (up to rounding)
            double disc = Math.sqrt(Math.max(tr * tr / 4.0 - det, 0.0));
            assertTrue(tr / 2.0 - disc >= -1e-10);
        }
    }

    @Test
    public void tracksBetaThroughBreak() {
        double[] y = TestData.pair().get("stock_y");
        double[] x = TestData.pair().get("stock_x");
        Kalman.Result km = Kalman.hedge(y, x, TestData.config().get("kalman_delta"),
                TestData.config().get("kalman_r"));
        assertEquals(1.7, km.beta()[500], 0.1);          // pre-break
        assertEquals(1.1, km.beta()[y.length - 1], 0.1); // post-break
        double betaOls = Ols.fit(y, x, true).params()[1];
        // Kalman tracking error vs the piecewise-true beta beats static OLS
        double[] trueBeta = new double[y.length];
        java.util.Arrays.fill(trueBeta, 1.7);
        for (int t = 1050; t <= 1080; t++) {
            trueBeta[t] = 1.7 + (1.1 - 1.7) * (t - 1050) / 30.0;
        }
        for (int t = 1081; t < y.length; t++) {
            trueBeta[t] = 1.1;
        }
        double teKalman = 0.0;
        double teOls = 0.0;
        for (int t = 300; t < y.length; t++) {
            teKalman += (km.beta()[t] - trueBeta[t]) * (km.beta()[t] - trueBeta[t]);
            teOls += (betaOls - trueBeta[t]) * (betaOls - trueBeta[t]);
        }
        assertTrue(Math.sqrt(teKalman) < Math.sqrt(teOls));
    }

    @Test
    public void innovationZscoresFiniteAndStandardised() {
        Random rng = new Random(9);
        int n = 600;
        double[] x = new double[n];
        double[] y = new double[n];
        double level = 100.0;
        for (int t = 0; t < n; t++) {
            level += rng.nextGaussian();
            x[t] = level;
            y[t] = 5.0 + 1.5 * x[t] + 0.5 * rng.nextGaussian();
        }
        Kalman.Result km = Kalman.hedge(y, x, 1e-4, 0.25);
        assertTrue(LinAlg.allFinite(km.z()));
        // after burn-in the innovations should be roughly standardised
        double[] tail = TestData.slice(km.z(), 100, n);
        double m = LinAlg.mean(tail);
        double ss = 0.0;
        for (double v : tail) {
            ss += (v - m) * (v - m);
        }
        double sd = Math.sqrt(ss / tail.length);
        assertTrue(sd > 0.2 && sd < 5.0);
        for (int t = 0; t < n; t++) {
            assertEquals(km.innovation()[t] / Math.sqrt(km.sVar()[t]), km.z()[t], 1e-12);
        }
    }

    @Test
    public void errorPaths() {
        double[] y = new double[10];
        double[] x = new double[10];
        for (int i = 0; i < 10; i++) {
            y[i] = i;
            x[i] = i + 1.0;
        }
        for (double badDelta : new double[] {0.0, 1.0, -0.1, 1.5}) {
            assertThrows(IllegalArgumentException.class,
                    () -> Kalman.hedge(y, x, badDelta, 1e-3));
        }
        assertThrows(IllegalArgumentException.class, () -> Kalman.hedge(y, x, 1e-4, 0.0));
        assertThrows(IllegalArgumentException.class,
                () -> Kalman.hedge(y, TestData.slice(x, 0, 9), 1e-4, 1e-3));
        assertThrows(IllegalArgumentException.class,
                () -> Kalman.hedge(new double[] {1.0}, new double[] {2.0}, 1e-4, 1e-3));
        assertThrows(IllegalArgumentException.class,
                () -> Kalman.hedge(new double[] {1.0, Double.NaN, 3.0},
                        new double[] {0.0, 1.0, 2.0}, 1e-4, 1e-3));
    }

    @Test
    public void rejectsNonFiniteParams() {
        // proposed test 4 (MAJ-3): r in {NaN, +inf}, delta NaN/inf -> error on
        // the first 50 bundled rows (previously r = NaN gave an all-NaN path)
        double[] y = TestData.slice(TestData.pair().get("stock_y"), 0, 50);
        double[] x = TestData.slice(TestData.pair().get("stock_x"), 0, 50);
        for (double r : new double[] {Double.NaN, Double.POSITIVE_INFINITY,
                Double.NEGATIVE_INFINITY, 0.0, -1e-3}) {
            assertThrows(IllegalArgumentException.class, () -> Kalman.hedge(y, x, 1e-4, r));
        }
        for (double d : new double[] {Double.NaN, Double.POSITIVE_INFINITY,
                Double.NEGATIVE_INFINITY}) {
            assertThrows(IllegalArgumentException.class, () -> Kalman.hedge(y, x, d, 1e-3));
        }
        Kalman.Result km = Kalman.hedge(y, x, 1e-4, 1e-3);
        assertTrue(LinAlg.allFinite(km.beta()) && LinAlg.allFinite(km.z()));
    }
}
