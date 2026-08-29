package com.quant.statarb;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import java.util.Random;
import org.junit.Test;

/** OLS: closed-form checks, exact-fit recovery, error paths. */
public class OlsTest {

    /** Deterministic pseudo-random series for property checks. */
    private static double[] gaussian(long seed, int n, double mu, double sigma) {
        Random rng = new Random(seed);
        double[] out = new double[n];
        for (int i = 0; i < n; i++) {
            out[i] = mu + sigma * rng.nextGaussian();
        }
        return out;
    }

    @Test
    public void twoVarClosedForm() {
        double[] x = gaussian(1, 400, 0.0, 2.0);
        double[] noise = gaussian(2, 400, 0.0, 0.1);
        double[] y = new double[400];
        for (int i = 0; i < 400; i++) {
            y[i] = 3.0 + 0.5 * x[i] + noise[i];
        }
        Ols.Result fit = Ols.fit(y, x, true);
        // slope = cov(x, y) / var(x), intercept = ybar - slope * xbar
        double xbar = LinAlg.mean(x);
        double ybar = LinAlg.mean(y);
        double sxy = 0.0;
        double sxx = 0.0;
        double syy = 0.0;
        for (int i = 0; i < 400; i++) {
            sxy += (x[i] - xbar) * (y[i] - ybar);
            sxx += (x[i] - xbar) * (x[i] - xbar);
            syy += (y[i] - ybar) * (y[i] - ybar);
        }
        double slope = sxy / sxx;
        assertEquals(slope, fit.params()[1], 1e-12);
        assertEquals(ybar - slope * xbar, fit.params()[0], 1e-12);
        double corr2 = (sxy * sxy) / (sxx * syy);
        assertEquals(corr2, fit.r2(), 1e-12);
        assertEquals(400, fit.nobs());
        assertEquals(2, fit.nparams());
    }

    @Test
    public void exactFitAndNoIntercept() {
        double[] x = new double[29];
        double[] y = new double[29];
        for (int i = 0; i < 29; i++) {
            x[i] = i + 1.0;
            y[i] = 2.0 * x[i];
        }
        Ols.Result fit = Ols.fit(y, x, false);
        assertEquals(2.0, fit.params()[0], 1e-12);
        assertEquals(1.0, fit.r2(), 1e-12);
        for (double r : fit.resid()) {
            assertTrue(Math.abs(r) < 1e-10);
        }
    }

    @Test
    public void propertyGridRecoversCoefficients() {
        // noiseless y = a + b x recovers (a, b) over a coefficient grid
        double[] x = new double[60];
        for (int i = 0; i < 60; i++) {
            x[i] = 1.0 + 8.0 * i / 59.0;
        }
        for (double a : new double[] {-5.0, 0.0, 2.5}) {
            for (double b : new double[] {-2.0, 0.5, 10.0}) {
                double[] y = new double[60];
                for (int i = 0; i < 60; i++) {
                    y[i] = a + b * x[i];
                }
                Ols.Result fit = Ols.fit(y, x, true);
                assertEquals(a, fit.params()[0], 1e-9);
                assertEquals(b, fit.params()[1], 1e-9);
            }
        }
    }

    @Test
    public void tstatsAreParamsOverSe() {
        double[] x = gaussian(3, 200, 5.0, 1.0);
        double[] noise = gaussian(4, 200, 0.0, 0.3);
        double[] y = new double[200];
        for (int i = 0; i < 200; i++) {
            y[i] = 1.0 + 2.0 * x[i] + noise[i];
        }
        Ols.Result fit = Ols.fit(y, x, true);
        for (int j = 0; j < 2; j++) {
            assertEquals(fit.params()[j] / fit.se()[j], fit.tstats()[j], 1e-12);
            assertTrue(fit.se()[j] > 0.0);
        }
        assertTrue(fit.sigma2() > 0.0);
    }

    @Test
    public void errorPaths() {
        double[] y = new double[20];
        for (int i = 0; i < 20; i++) {
            y[i] = i;
        }
        assertThrows(IllegalArgumentException.class,
                () -> Ols.fit(y, new double[19], true));
        assertThrows(IllegalArgumentException.class,
                () -> Ols.fit(new double[] {1.0, Double.NaN, 2.0, 3.0},
                        new double[] {0.0, 1.0, 2.0, 3.0}, true));
        double[] constY = new double[20];
        java.util.Arrays.fill(constY, 3.0);
        assertThrows(IllegalArgumentException.class, () -> Ols.fit(constY, y, true));
        double[] constX = new double[20];
        java.util.Arrays.fill(constX, 7.0);
        assertThrows(IllegalArgumentException.class, () -> Ols.fit(y, constX, true));
        assertThrows(IllegalArgumentException.class,
                () -> Ols.fit(new double[] {1.0, 2.0}, new double[] {1.0, 2.0}, true));
    }

    @Test
    public void nearPerfectCollinearityRaises() {
        double[] x1 = gaussian(5, 200, 0.0, 1.0);
        double[][] x = new double[200][2];
        double[] y = new double[200];
        Random rng = new Random(6);
        for (int i = 0; i < 200; i++) {
            x[i][0] = x1[i];
            x[i][1] = x1[i] * (1.0 + 1e-14) + 1e-13;
            y[i] = x1[i] + rng.nextGaussian();
        }
        assertThrows(IllegalArgumentException.class, () -> Ols.fit(y, x, true));
    }
}
