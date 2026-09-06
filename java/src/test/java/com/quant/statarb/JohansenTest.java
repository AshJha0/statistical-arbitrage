package com.quant.statarb;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import java.util.Random;
import org.junit.Test;

/** Johansen-lite: eigen ordering, rank verdicts, Jacobi, Cholesky, errors. */
public class JohansenTest {

    private static double[][] columnStack(double[]... cols) {
        int n = cols[0].length;
        double[][] out = new double[n][cols.length];
        for (int t = 0; t < n; t++) {
            for (int j = 0; j < cols.length; j++) {
                out[t][j] = cols[j][t];
            }
        }
        return out;
    }

    @Test
    public void pairTraceOrderingAndRank() {
        int nIs = TestData.insampleDays();
        double[][] lv = columnStack(
                TestData.slice(TestData.pair().get("stock_y"), 0, nIs),
                TestData.slice(TestData.pair().get("stock_x"), 0, nIs));
        Johansen.Result res = Johansen.traceTest(lv);
        double[] lam = res.eigenvalues();
        assertTrue(lam[0] >= lam[1]);
        assertTrue(lam[1] >= 0.0);
        assertTrue(lam[0] < 1.0);
        // trace stats are nested sums of positive terms: trace(0) > trace(1)
        assertTrue(res.traceStats()[0] > res.traceStats()[1]);
        assertTrue(res.traceStats()[1] >= 0.0);
        assertEquals(1, res.rank()); // cointegrated pair, not stationary in levels
        // cointegrating vector recovers roughly (1, -beta_true) pre-break
        assertEquals(1.0, res.beta()[0][0], 1e-12);
        assertEquals(1.7, -res.beta()[1][0], 0.25);
        assertEquals(nIs - 2, res.tEff());
    }

    @Test
    public void spuriousPairRankZero() {
        double[][] lv = columnStack(TestData.walks().get("walk_a"),
                TestData.walks().get("walk_b"));
        assertEquals(0, Johansen.traceTest(lv).rank());
    }

    @Test
    public void fxTriangleThreeAsset() {
        // log EURUSD - log GBPUSD - log EURGBP is stationary: rank >= 1 and
        // the leading vector has signs (+, -, -) after normalisation
        double[] eurusd = TestData.fx().get("eurusd");
        double[] gbpusd = TestData.fx().get("gbpusd");
        double[] eurgbp = TestData.fx().get("eurgbp");
        int n = eurusd.length;
        double[][] lv = new double[n][3];
        for (int t = 0; t < n; t++) {
            lv[t][0] = Math.log(eurusd[t]);
            lv[t][1] = Math.log(gbpusd[t]);
            lv[t][2] = Math.log(eurgbp[t]);
        }
        Johansen.Result res = Johansen.traceTest(lv);
        assertTrue(res.rank() >= 1);
        double[] lam = res.eigenvalues();
        assertTrue(lam[0] >= lam[1]);
        assertTrue(lam[1] >= lam[2]);
        assertTrue(lam[2] >= 0.0);
        assertEquals(1.0, res.beta()[0][0], 1e-12);
        assertEquals(-1.0, res.beta()[1][0], 0.15);
        assertEquals(-1.0, res.beta()[2][0], 0.15);
    }

    @Test
    public void jacobiMatchesClosedForm2x2() {
        // eigenvalues of [[2,1],[1,3]] are (5 +/- sqrt(5)) / 2
        double[][] a = {{2.0, 1.0}, {1.0, 3.0}};
        LinAlg.Eigen eig = LinAlg.jacobiEigh(a);
        assertEquals((5.0 + Math.sqrt(5.0)) / 2.0, eig.values()[0], 1e-12);
        assertEquals((5.0 - Math.sqrt(5.0)) / 2.0, eig.values()[1], 1e-12);
        // eigenvector property A v = lambda v
        for (int i = 0; i < 2; i++) {
            for (int r = 0; r < 2; r++) {
                double av = a[r][0] * eig.vectors()[0][i] + a[r][1] * eig.vectors()[1][i];
                assertEquals(eig.values()[i] * eig.vectors()[r][i], av, 1e-10);
            }
        }
    }

    @Test
    public void jacobiOnRandomSymmetricPsd() {
        Random rng = new Random(8);
        for (int n = 2; n <= 3; n++) {
            for (int rep = 0; rep < 5; rep++) {
                double[][] m = new double[n][n];
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < n; j++) {
                        m[i][j] = rng.nextGaussian();
                    }
                }
                double[][] a = LinAlg.matmul(m, LinAlg.transpose(m)); // symmetric PSD
                LinAlg.Eigen eig = LinAlg.jacobiEigh(a);
                for (int i = 0; i < n; i++) {
                    if (i > 0) {
                        assertTrue(eig.values()[i - 1] >= eig.values()[i]);
                    }
                    assertTrue(eig.values()[i] >= -1e-10);
                    for (int r = 0; r < n; r++) {
                        double av = 0.0;
                        for (int c = 0; c < n; c++) {
                            av += a[r][c] * eig.vectors()[c][i];
                        }
                        assertEquals(eig.values()[i] * eig.vectors()[r][i], av, 1e-8);
                    }
                }
            }
        }
    }

    @Test
    public void choleskyNative() {
        double[][] a = {{4.0, 2.0, 0.6}, {2.0, 5.0, 1.0}, {0.6, 1.0, 3.0}};
        double[][] l = LinAlg.choleskyLower(a);
        double[][] llt = LinAlg.matmul(l, LinAlg.transpose(l));
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(a[i][j], llt[i][j], 1e-12);
                if (j > i) {
                    assertEquals(0.0, l[i][j], 0.0); // strictly lower triangular
                }
            }
        }
        assertThrows(IllegalArgumentException.class,
                () -> LinAlg.choleskyLower(new double[][] {{1.0, 2.0}, {2.0, 1.0}}));
    }

    @Test
    public void traceCriticalValuesPinned() {
        assertEquals(3.76, Johansen.TRACE_CRIT_95[1], 0.0);
        assertEquals(15.41, Johansen.TRACE_CRIT_95[2], 0.0);
        assertEquals(29.68, Johansen.TRACE_CRIT_95[3], 0.0);
    }

    @Test
    public void errorPaths() {
        double[] y = TestData.slice(TestData.pair().get("stock_y"), 0, 100);
        double[] x = TestData.slice(TestData.pair().get("stock_x"), 0, 100);
        assertThrows(IllegalArgumentException.class,
                () -> Johansen.traceTest(columnStack(y, x, y, x)));
        assertThrows(IllegalArgumentException.class,
                () -> Johansen.traceTest(columnStack(
                        TestData.slice(y, 0, 10), TestData.slice(x, 0, 10))));
        double[] constant = new double[100];
        java.util.Arrays.fill(constant, 5.0);
        assertThrows(IllegalArgumentException.class,
                () -> Johansen.traceTest(columnStack(y, constant)));
        assertThrows(IllegalArgumentException.class,
                () -> Johansen.traceTest(columnStack(y, x), 2));
        double[] bad = x.clone();
        bad[3] = Double.NaN;
        assertThrows(IllegalArgumentException.class,
                () -> Johansen.traceTest(columnStack(y, bad)));
    }

    @Test
    public void collinearLevelsRejected() {
        // proposed test 11: exactly collinear level columns are refused with a
        // "singular" error rather than producing garbage eigenvalues
        double[] y = TestData.slice(TestData.pair().get("stock_y"), 0, 200);
        double[] x = TestData.slice(TestData.pair().get("stock_x"), 0, 200);
        double[] y2 = new double[200];
        double[] yx = new double[200];
        for (int t = 0; t < 200; t++) {
            y2[t] = 2.0 * y[t] + 1.0;
            yx[t] = y[t] + x[t];
        }
        for (double[][] lv : new double[][][] {columnStack(y, y2), columnStack(y, x, yx)}) {
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                    () -> Johansen.traceTest(lv));
            assertTrue(e.getMessage(), e.getMessage().contains("singular"));
        }
    }

    @Test
    public void linalgHelpersRejectNonFinite() {
        double[][] nanM = {{1.0, Double.NaN}, {Double.NaN, 1.0}};
        assertThrows(IllegalArgumentException.class, () -> LinAlg.jacobiEigh(nanM));
        assertThrows(IllegalArgumentException.class, () -> LinAlg.choleskyLower(nanM));
        assertThrows(IllegalArgumentException.class,
                () -> LinAlg.choleskyLower(new double[][] {{1.0, 2.0, 3.0}}));
    }
}
