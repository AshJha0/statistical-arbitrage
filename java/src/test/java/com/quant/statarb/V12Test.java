package com.quant.statarb;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import java.util.Arrays;
import java.util.List;
import java.util.Random;
import org.junit.Test;

/**
 * v1.2 contract (mirrors python/tests/test_v12.py): beta check, sizing, risk
 * limits, dead configs, pinned z sums, rolling walk-forward, streaming API,
 * lag selection and two-way Engle-Granger.
 */
public class V12Test {

    private static final double CAP = 1_000_000.0;

    private static Backtest.Config cfg(int window) {
        return new Backtest.Config(window, 2.0, 0.5, 4.0, 1.0, 0.5, CAP);
    }

    private static Backtest.Config noCost(int window) {
        return new Backtest.Config(window, 2.0, 0.5, 4.0, 0.0, 0.0, CAP);
    }

    private static double[] sine(int n) {
        double[] s = new double[n];
        for (int t = 0; t < n; t++) {
            s[t] = 0.2 * Math.sin(0.7 * t);
        }
        return s;
    }

    private static double[] constant(int n, double v) {
        double[] a = new double[n];
        Arrays.fill(a, v);
        return a;
    }

    /** Prices whose spread at alpha = 50, beta = 1 is exactly s; returns {y, x}. */
    private static double[][] scripted(double[] s, double x0) {
        double[] x = constant(s.length, x0);
        double[] y = new double[s.length];
        for (int t = 0; t < s.length; t++) {
            y[t] = 50.0 + x[t] + s[t];
        }
        return new double[][] {y, x};
    }

    private static double[][] scripted(double[] s) {
        return scripted(s, 50.0);
    }

    /** Random-walk x and an AR(0.9) spread: y = 20 + 1.5 x + s + 40; returns {y, x}. */
    private static double[][] noise(int n, long seed) {
        Random rng = new Random(seed);
        double[] x = new double[n];
        double[] y = new double[n];
        double lx = 0.0;
        double s = 0.0;
        for (int t = 0; t < n; t++) {
            lx += 0.01 * rng.nextGaussian();
            x[t] = 100.0 * Math.exp(lx);
            s = t == 0 ? 0.0 : 0.9 * s + rng.nextGaussian();
            y[t] = 20.0 + 1.5 * x[t] + s + 40.0;
        }
        return new double[][] {y, x};
    }

    private static EngleGranger.Result bundledEg() {
        return EngleGranger.test(TestData.slice(TestData.pair().get("stock_y"), 0, 900),
                TestData.slice(TestData.pair().get("stock_x"), 0, 900),
                TestData.config().get("adf_maxlag").intValue());
    }

    private static Backtest.Result bundled(Backtest.Config c) {
        EngleGranger.Result eg = bundledEg();
        return Backtest.run(TestData.pair().get("stock_y"), TestData.pair().get("stock_x"),
                eg.alpha(), eg.beta(), c);
    }

    // ---- items 1 and 2: beta and sizing ------------------------------------

    @Test
    public void backtestRejectsNonPositiveBeta() {
        double[][] p = noise(100, 3);
        for (double beta : new double[] {0.0, -1.0, Double.NaN}) {
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                    () -> Backtest.run(p[0], p[1], 60.0, beta, cfg(60)));
            assertTrue(e.getMessage(), e.getMessage().contains("beta"));
        }
    }

    @Test
    public void sizingDefaultAndValidation() {
        assertEquals(Backtest.BETA_HEDGED, cfg(60).sizing());
        assertThrows(IllegalArgumentException.class, () -> cfg(60).withSizing("kelly"));
        assertThrows(IllegalArgumentException.class, () -> cfg(60).withSizing(null));
    }

    @Test
    public void grossEqualsCapitalAtEveryEntry() {
        for (String sizing : new String[] {Backtest.BETA_HEDGED, Backtest.EQUAL_DOLLAR}) {
            Backtest.Config c = TestData.btConfig().withSizing(sizing);
            Backtest.Result res = bundled(c);
            double bps = (c.costBps() + c.slippageBps()) / 1e4;
            assertFalse(res.trades().isEmpty());
            for (Backtest.Trade tr : res.trades()) {
                assertEquals(CAP * bps, res.dailyCost()[tr.entryDay()], CAP * bps * 1e-12);
            }
        }
    }

    @Test
    public void betaHedgedLegRatioAndEqualDollarNeutral() {
        double[] s = new double[120];
        for (int t = 0; t < 70; t++) {
            s[t] = 0.2 * Math.sin(0.7 * t);
        }
        Arrays.fill(s, 70, 120, 2.0);
        double[][] p = scripted(s, 100.0);
        double[] y = p[0];
        double[] x = p[1];
        double beta = 1.0;
        Backtest.Config c = noCost(40);
        int entry = Backtest.run(y, x, 50.0, beta, c).trades().get(0).entryDay();
        for (int t = entry + 1; t < x.length; t++) {
            x[t] += 1.0;
            y[t] += beta; // spread unchanged
        }
        Backtest.Result res = Backtest.run(y, x, 50.0, beta, c);
        assertEquals(entry, res.trades().get(0).entryDay());
        // hedged at beta: moving along the hedge line earns nothing
        assertEquals(0.0, res.dailyPnl()[entry + 1], 1e-8);

        double[] y2 = y.clone();
        for (int t = entry + 1; t < y2.length; t++) {
            y2[t] -= 1.0; // only x moved; the short-spread position is long x
        }
        Backtest.Result res2 = Backtest.run(y2, x, 50.0, beta,
                c.withSizing(Backtest.EQUAL_DOLLAR));
        double qx = (CAP / 2.0) / x[entry];
        assertEquals(-1, res2.trades().get(0).direction());
        assertEquals(qx, res2.dailyPnl()[entry + 1], qx * 1e-12);
    }

    // ---- item 3: risk limits -----------------------------------------------

    @Test
    public void accountingIdentityAcrossSizingAndLimits() {
        double[][] limits = {{0.0, 0, 0}, {0.02, 60, 20}, {0.005, 5, 3}};
        for (String sizing : new String[] {Backtest.BETA_HEDGED, Backtest.EQUAL_DOLLAR}) {
            for (double[] l : limits) {
                Backtest.Config c = TestData.btConfig().withSizing(sizing)
                        .withLimits(l[0], (int) l[1], (int) l[2]);
                Backtest.Result res = bundled(c);
                double pnl = 0.0;
                double cost = 0.0;
                for (int t = 0; t < res.equity().length; t++) {
                    pnl += res.dailyPnl()[t];
                    cost += res.dailyCost()[t];
                }
                double lhs = res.equity()[res.equity().length - 1] - CAP;
                assertEquals(pnl - cost, lhs, 1e-6);
                double tp = 0.0;
                for (Backtest.Trade tr : res.trades()) {
                    tp += tr.pnl();
                }
                assertEquals(tp, lhs, 1e-6);
            }
        }
    }

    @Test
    public void limitValidation() {
        Backtest.Config c = cfg(60);
        for (double bad : new double[] {-0.1, Double.NaN, Double.POSITIVE_INFINITY}) {
            assertThrows(IllegalArgumentException.class, () -> c.withLimits(bad, 0, 0));
        }
        assertThrows(IllegalArgumentException.class, () -> c.withLimits(0.0, -1, 0));
        assertThrows(IllegalArgumentException.class, () -> c.withLimits(0.0, 0, -1));
    }

    private static double[] tradeThenLoss(double lossValue) {
        double[] s = sine(140);
        Arrays.fill(s, 70, 73, 0.6);
        Arrays.fill(s, 73, 140, lossValue);
        return s;
    }

    @Test
    public void pnlStopFiresAndBeatsZExit() {
        double[] s = sine(140);
        s[70] = 0.6; // entry signal
        s[71] = 0.6; // execute short
        Arrays.fill(s, 72, 140, 0.0);
        double[] x = constant(140, 50.0);
        for (int t = 72; t < 140; t++) {
            x[t] -= 5.0; // hurts the long-x leg
        }
        double[] y = new double[140];
        for (int t = 0; t < 140; t++) {
            y[t] = 50.0 + x[t] + s[t];
        }
        Backtest.Config base = noCost(40);
        double[] z = Signals.rollingZscore(Signals.spread(y, x, 50.0, 1.0), 40);
        assertTrue(Math.abs(z[72]) <= base.exitZ());
        assertFalse(Backtest.run(y, x, 50.0, 1.0, base).trades().get(0).stopped());
        // hedged legs do not lose on a move along the hedge line; equal-dollar do
        Backtest.Config eq = base.withSizing(Backtest.EQUAL_DOLLAR);
        double loss = Backtest.run(y, x, 50.0, 1.0, eq).dailyPnl()[72];
        assertTrue(loss < 0.0);
        Backtest.Trade tr = Backtest.run(y, x, 50.0, 1.0,
                eq.withLimits(-loss / CAP / 2.0, 0, 0)).trades().get(0);
        assertTrue(tr.stopped());
        assertEquals(73, tr.exitDay());
    }

    @Test
    public void pnlStopUsesCostsSoFar() {
        double[][] p = scripted(tradeThenLoss(0.6));
        Backtest.Trade tr = Backtest.run(p[0], p[1], 50.0, 1.0,
                cfg(40).withLimits(1.0e-4, 0, 0)).trades().get(0);
        assertTrue(tr.stopped());
        assertEquals(tr.entryDay() + 1, tr.exitDay());
    }

    @Test
    public void pnlStopBoundaryIsInclusive() {
        double[][] p = scripted(tradeThenLoss(0.6));
        Backtest.Config c = cfg(40).withLimits(1.5e-4, 0, 0);
        Backtest.Result res = Backtest.run(p[0], p[1], 50.0, 1.0, c);
        assertEquals(150.0, res.dailyCost()[res.trades().get(0).entryDay()], 150.0 * 1e-12);
        // the float product may land on either side by one ulp: pin just above and below
        assertFalse(Backtest.run(p[0], p[1], 50.0, 1.0, c.withLimits(1.5001e-4, 0, 0))
                .trades().get(0).stopped());
        assertTrue(Backtest.run(p[0], p[1], 50.0, 1.0, c.withLimits(1.4999e-4, 0, 0))
                .trades().get(0).stopped());
    }

    @Test
    public void stopPriorityZThenHold() {
        double[] s = sine(160);
        Arrays.fill(s, 70, 73, 0.5);
        s[73] = 5.0;
        double[][] p = scripted(s);
        Backtest.Config c = cfg(40).withLimits(0.0, 2, 0);
        assertTrue(Signals.rollingZscore(s, 40)[73] >= c.stopZ());
        Backtest.Trade tr = Backtest.run(p[0], p[1], 50.0, 1.0, c).trades().get(0);
        assertEquals(71, tr.entryDay());
        assertEquals(74, tr.exitDay());
        assertTrue(tr.stopped());
    }

    @Test
    public void maxHoldExitsUnflaggedOnTime() {
        double[][] p = scripted(tradeThenLoss(0.6));
        Backtest.Result res = Backtest.run(p[0], p[1], 50.0, 1.0,
                cfg(40).withLimits(0.0, 3, 0));
        Backtest.Trade tr = res.trades().get(0);
        assertEquals(4, tr.exitDay() - tr.entryDay()); // signal at t - entry >= 3, run next day
        assertFalse(tr.stopped());
        assertEquals(0, res.nStops());
    }

    @Test
    public void limitsFireOnNanZ() {
        int w = 5;
        double[] s = new double[38];
        Arrays.fill(s, 0.3);
        double[] head = {0.0, 0.1, -0.1, 0.05, 0.0, 0.0, 0.3, 0.3};
        System.arraycopy(head, 0, s, 0, head.length);
        double[][] p = scripted(s);
        assertTrue(Double.isNaN(Signals.rollingZscore(s, w)[12]));
        Backtest.Config c = new Backtest.Config(w, 1.5, 0.1, 1.75, 1.0, 0.5, CAP)
                .withLimits(0.0, 8, 0);
        Backtest.Trade tr = Backtest.run(p[0], p[1], 50.0, 1.0, c).trades().get(0);
        assertEquals(16, tr.exitDay()); // entry 7, signal 15 on a NaN z
        assertFalse(tr.stopped());
    }

    private static List<Backtest.Trade> stopThenSignal(int gap, int cooldown) {
        double[] s = sine(240);
        Arrays.fill(s, 70, 73, 0.5);
        s[73] = 5.0;
        int e = 74;
        s[e + gap] = 3.0;
        double[][] p = scripted(s);
        return Backtest.run(p[0], p[1], 50.0, 1.0, cfg(40).withLimits(0.0, 0, cooldown))
                .trades();
    }

    @Test
    public void cooldownTiming() {
        int gap = 10;
        List<Backtest.Trade> free = stopThenSignal(gap, 0);
        assertTrue(free.get(0).stopped());
        assertEquals(74, free.get(0).exitDay());
        assertEquals(2, free.size());
        assertEquals(74 + gap + 1, free.get(1).entryDay());
        // t = e + gap: blocked while t < e + cooldown, allowed at equality
        assertEquals(1, stopThenSignal(gap, gap + 1).size());
        List<Backtest.Trade> allowed = stopThenSignal(gap, gap);
        assertEquals(2, allowed.size());
        assertEquals(74 + gap + 1, allowed.get(1).entryDay());
    }

    @Test
    public void cooldownNotAfterUnstoppedExit() {
        double[] s = sine(160);
        s[70] = 0.5;
        s[71] = 0.0;
        s[72] = 0.5;
        s[73] = 0.0;
        double[][] p = scripted(s);
        assertEquals(2, Backtest.run(p[0], p[1], 50.0, 1.0,
                cfg(40).withLimits(0.0, 0, 50)).trades().size());
    }

    @Test
    public void cooldownAndHoldPropertiesOnBundled() {
        Backtest.Config c = TestData.btConfig();
        Backtest.Result res = bundled(c);
        assertTrue(res.nStops() >= 1);
        List<Backtest.Trade> tr = res.trades();
        for (int i = 0; i + 1 < tr.size(); i++) {
            if (tr.get(i).stopped()) { // signal day = entry - 1
                assertTrue(tr.get(i + 1).entryDay() - 1 >= tr.get(i).exitDay() + c.cooldownDays());
            }
        }
        for (Backtest.Trade t : tr) {
            assertTrue(t.holdingDays() <= c.maxHoldDays() + 1);
        }
    }

    // ---- item 4: dead configs ----------------------------------------------

    private static void assertDead(String needle, Runnable r) {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, r::run);
        assertTrue(e.getMessage(), e.getMessage().contains(needle));
    }

    @Test
    public void deadEntryRejected() {
        assertDead("entry_z unreachable", () -> new Backtest.Config(5, 2.0, 0.5, 4.0, 1, 0.5, CAP));
        // (4-1)/2 == 1.5 exactly: still dead, even with a P&L stop
        assertDead("entry_z unreachable", () -> new Backtest.Config(4, 1.5, 0.5, 4.0, 1, 0.5, CAP,
                Backtest.BETA_HEDGED, 0.1, 0, 0));
    }

    @Test
    public void deadStopNeedsPnlStop() {
        assertDead("stop_z unreachable", () -> new Backtest.Config(20, 2.0, 0.5, 8.0, 1, 0.5, CAP));
        assertDead("stop_z unreachable", () -> new Backtest.Config(20, 2.0, 0.5, 8.0, 1, 0.5, CAP,
                Backtest.BETA_HEDGED, 0.0, 10, 0));
        new Backtest.Config(20, 2.0, 0.5, 8.0, 1, 0.5, CAP, Backtest.BETA_HEDGED, 0.01, 0, 0);
    }

    // ---- item 5: pinned summation ------------------------------------------

    @Test
    public void rollingZIsPlainSequentialSum() {
        Random rng = new Random(1);
        int n = 300;
        int w = 37;
        double[] s = new double[n];
        for (int t = 0; t < n; t++) {
            s[t] = rng.nextGaussian() * Math.pow(10.0, rng.nextInt(12) - 6);
        }
        double[] z = Signals.rollingZscore(s, w);
        for (int t = w - 1; t < n; t++) {
            double total = 0.0;
            for (int i = t - w + 1; i <= t; i++) {
                total = total + s[i];
            }
            double m = total / w;
            double ss = 0.0;
            for (int i = t - w + 1; i <= t; i++) {
                ss = ss + (s[i] - m) * (s[i] - m);
            }
            assertEquals(Double.doubleToLongBits((s[t] - m) / Math.sqrt(ss / (w - 1))),
                    Double.doubleToLongBits(z[t]));
        }
    }

    // ---- item 6: rolling walk-forward --------------------------------------

    private static Backtest.Config small() {
        return new Backtest.Config(30, 1.5, 0.3, 4.0, 1.0, 0.5, CAP).withLimits(0.02, 30, 5);
    }

    @Test
    public void rollingGeometryAndFlatAtBlockEnd() {
        double[][] p = noise(1000, 3);
        Backtest.RollingResult rw = Backtest.rollingWalkForward(p[0], p[1], small(), 300, 150);
        int[][] expect = {{300, 450}, {450, 600}, {600, 750}, {750, 900}, {900, 1000}};
        assertEquals(expect.length, rw.blocks().size());
        for (int i = 0; i < expect.length; i++) {
            assertEquals(expect[i][0], rw.blocks().get(i).start());
            assertEquals(expect[i][1], rw.blocks().get(i).end());
        }
        assertEquals(300, rw.startDay());
        assertEquals(700, rw.equity().length);
        assertFalse(rw.trades().isEmpty());
        for (Backtest.RollingBlock b : rw.blocks()) {
            assertEquals(b.end() - b.start(), b.result().equity().length);
            for (Backtest.Trade t : b.result().trades()) {
                assertTrue(b.start() < t.entryDay() && t.entryDay() < t.exitDay()
                        && t.exitDay() <= b.end() - 1);
            }
        }
    }

    @Test
    public void rollingStitchAndFitWindow() {
        double[][] p = noise(800, 3);
        Backtest.RollingResult rw = Backtest.rollingWalkForward(p[0], p[1], small(), 300, 100);
        double pnl = 0.0;
        double turnover = 0.0;
        double n = 0.0;
        for (Backtest.RollingBlock b : rw.blocks()) {
            double[] e = b.result().equity();
            pnl += e[e.length - 1] - CAP;
            turnover += b.result().metrics().get("turnover");
            n += b.result().metrics().get("n_trades");
        }
        assertEquals(pnl, rw.equity()[rw.equity().length - 1] - CAP, 1e-6);
        assertEquals(pnl / CAP, rw.metrics().get("total_return"), 1e-12);
        assertEquals(n, rw.metrics().get("n_trades"), 0.0);
        assertEquals(turnover, rw.metrics().get("turnover"), 1e-9);
        int k = 2;
        EngleGranger.Result fit = EngleGranger.test(
                TestData.slice(p[0], k * 100, k * 100 + 300),
                TestData.slice(p[1], k * 100, k * 100 + 300), 1);
        assertEquals(fit.alpha(), rw.blocks().get(k).alpha(), 0.0);
        assertEquals(fit.beta(), rw.blocks().get(k).beta(), 0.0);
    }

    @Test
    public void rollingUsesNoFutureData() {
        double[][] p = noise(900, 3);
        Backtest.Config c = small();
        Backtest.RollingResult base = Backtest.rollingWalkForward(p[0], p[1], c, 300, 100);
        for (int cut : new int[] {450, 600, 777}) {
            double[] y2 = p[0].clone();
            double[] x2 = p[1].clone();
            int m = 900 - cut;
            for (int i = 0; i < m; i++) {
                double f = m == 1 ? 0.0 : (double) i / (m - 1);
                y2[cut + i] *= 1.0 + (0.05 + f * 0.45);
                x2[cut + i] *= 1.0 - (0.02 + f * 0.28);
            }
            Backtest.RollingResult alt = Backtest.rollingWalkForward(y2, x2, c, 300, 100);
            int n = cut - base.startDay();
            assertArrayEquals(Arrays.copyOf(base.equity(), n), Arrays.copyOf(alt.equity(), n), 0.0);
            for (int i = 0; i < base.blocks().size(); i++) {
                Backtest.RollingBlock b0 = base.blocks().get(i);
                if (b0.start() <= cut) {
                    assertEquals(b0.alpha(), alt.blocks().get(i).alpha(), 0.0);
                    assertEquals(b0.beta(), alt.blocks().get(i).beta(), 0.0);
                }
            }
            assertEquals(base.trades().stream().filter(t -> t.exitDay() < cut).toList(),
                    alt.trades().stream().filter(t -> t.exitDay() < cut).toList());
        }
    }

    @Test
    public void rollingFirstBlockEntryNeedsBlockSignal() {
        double[][] p = noise(500, 3);
        Backtest.RollingResult rw = Backtest.rollingWalkForward(p[0], p[1], small(), 300, 200);
        for (Backtest.Trade t : rw.trades()) {
            assertTrue(t.entryDay() >= 301);
        }
    }

    @Test
    public void rollingBlockMatchesManualZ() {
        double[][] p = noise(600, 8);
        Backtest.Config c = small().withLimits(0.02, 30, 0);
        Backtest.RollingResult rw = Backtest.rollingWalkForward(p[0], p[1], c, 300, 300);
        Backtest.RollingBlock b = rw.blocks().get(0);
        double[] z = Signals.rollingZscore(Signals.spread(TestData.slice(p[0], 271, 600),
                TestData.slice(p[1], 271, 600), b.alpha(), b.beta()), c.window());
        double[] got = b.result().zscore();
        assertEquals(300, got.length);
        for (int i = 0; i < got.length; i++) {
            assertEquals(Double.doubleToLongBits(z[29 + i]), Double.doubleToLongBits(got[i]));
        }
    }

    @Test
    public void rollingNegativeFitBlockStaysFlat() {
        int n = 400;
        double[] x = new double[n];
        double[] y = new double[n];
        for (int t = 0; t < n; t++) {
            x[t] = 100.0 + 0.1 * t + Math.sin(t);
            y[t] = 200.0 - 0.1 * t + Math.cos(0.3 * t); // falling against a rising x
        }
        Backtest.RollingResult rw = Backtest.rollingWalkForward(y, x, small(), 200, 100);
        for (Backtest.RollingBlock b : rw.blocks()) {
            assertTrue(b.skipped() && b.beta() < 0.0);
            for (double z : b.result().zscore()) {
                assertTrue(Double.isNaN(z));
            }
        }
        assertTrue(rw.trades().isEmpty());
        for (double e : rw.equity()) {
            assertEquals(CAP, e, 0.0);
        }
        assertEquals(0.0, rw.metrics().get("turnover"), 0.0);
    }

    @Test
    public void rollingValidation() {
        double[][] p = noise(1000, 3);
        int[][] bad = {{10, 50}, {300, 0}, {1000, 50}, {1200, 50}};
        for (int[] tt : bad) {
            assertThrows(IllegalArgumentException.class,
                    () -> Backtest.rollingWalkForward(p[0], p[1], small(), tt[0], tt[1]));
        }
        double[][] q = noise(400, 3);
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> Backtest.rollingWalkForward(q[0], TestData.slice(q[1], 0, 399), small(),
                        100, 50));
        assertTrue(e.getMessage().contains("equal length"));
    }

    @Test
    public void walkForwardStillSingleHoldout() {
        Backtest.WalkForwardResult wf = Backtest.walkForward(TestData.pair().get("stock_y"),
                TestData.pair().get("stock_x"), TestData.btConfig(), 0.6);
        assertEquals(900, wf.split());
    }

    // ---- item 7: streaming ---------------------------------------------------

    @Test
    public void rollingZStreamMatchesBatch() {
        EngleGranger.Result eg = bundledEg();
        double[] s = Signals.spread(TestData.pair().get("stock_y"),
                TestData.pair().get("stock_x"), eg.alpha(), eg.beta());
        for (int window : new int[] {2, 5, 60, 250}) {
            double[] batch = Signals.rollingZscore(s, window);
            RollingZ rz = new RollingZ(window);
            for (int t = 0; t < s.length; t++) {
                double got = rz.update(s[t]);
                assertEquals(Double.isNaN(batch[t]), Double.isNaN(got));
                if (!Double.isNaN(batch[t])) {
                    assertEquals("w=" + window + " t=" + t, batch[t], got, 1e-9);
                }
            }
        }
    }

    @Test
    public void rollingZStreamLongRunWithLargeLevel() {
        Random rng = new Random(5);
        int n = 50_000;
        double[] s = new double[n];
        double level = 1.0e4;
        for (int t = 0; t < n; t++) {
            level += rng.nextGaussian();
            s[t] = level;
        }
        double[] batch = Signals.rollingZscore(s, 60);
        RollingZ rz = new RollingZ(60);
        double worst = 0.0;
        for (int t = 0; t < n; t++) {
            double got = rz.update(s[t]);
            if (!Double.isNaN(batch[t])) {
                worst = Math.max(worst, Math.abs(got - batch[t]));
            }
        }
        // running moments drift slowly with the update count; 1e-9 holds at the
        // bundled length (test above), a looser bound after 50,000 updates
        assertTrue("worst " + worst, worst < 1e-7);
    }

    @Test
    public void rollingZStreamZeroVarianceAndValidation() {
        RollingZ rz = new RollingZ(3);
        for (int i = 0; i < 5; i++) {
            assertTrue(Double.isNaN(rz.update(1.0)));
        }
        assertThrows(IllegalArgumentException.class, () -> new RollingZ(1));
        assertThrows(IllegalArgumentException.class, () -> new RollingZ(0));
        assertThrows(IllegalArgumentException.class, () -> rz.update(Double.NaN));
        assertThrows(IllegalArgumentException.class, () -> rz.update(Double.POSITIVE_INFINITY));
        assertTrue(Double.isNaN(rz.update(1.0))); // the rejected values left no trace
    }

    @Test
    public void kalmanStepMatchesBatch() {
        double[] y = TestData.pair().get("stock_y");
        double[] x = TestData.pair().get("stock_x");
        double d = TestData.config().get("kalman_delta");
        double r = TestData.config().get("kalman_r");
        Kalman.Result km = Kalman.hedge(y, x, d, r);
        KalmanStep ks = new KalmanStep(d, r);
        for (int t = 0; t < y.length; t++) {
            double z = ks.update(x[t], y[t]);
            assertEquals(km.alpha()[t], ks.alpha(), 1e-12);
            assertEquals(km.beta()[t], ks.beta(), 1e-12);
            assertEquals(km.z()[t], z, 1e-12);
        }
    }

    @Test
    public void kalmanScalarRecursionMatchesMatrixJosephForm() {
        double[] y = TestData.pair().get("stock_y");
        double[] x = TestData.pair().get("stock_x");
        double d = TestData.config().get("kalman_delta");
        double r = TestData.config().get("kalman_r");
        double q = d / (1 - d);
        double[] s = {0.0, 0.0};
        double[][] pm = {{1e3, 0.0}, {0.0, 1e3}};
        Kalman.Result km = Kalman.hedge(y, x, d, r);
        for (int t = 0; t < y.length; t++) {
            double[][] pp = {{pm[0][0] + q, pm[0][1]}, {pm[1][0], pm[1][1] + q}};
            double[] h = {1.0, x[t]};
            double[] ph = {pp[0][0] * h[0] + pp[0][1] * h[1], pp[1][0] * h[0] + pp[1][1] * h[1]};
            double sv = h[0] * ph[0] + h[1] * ph[1] + r;
            double[] k = {ph[0] / sv, ph[1] / sv};
            double e = y[t] - (h[0] * s[0] + h[1] * s[1]);
            s[0] += k[0] * e;
            s[1] += k[1] * e;
            double[][] a = {{1 - k[0] * h[0], -k[0] * h[1]}, {-k[1] * h[0], 1 - k[1] * h[1]}};
            double[][] ap = LinAlg.matmul(a, pp);
            double[][] n = LinAlg.matmul(ap, LinAlg.transpose(a));
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++) {
                    n[i][j] += r * k[i] * k[j];
                }
            }
            double off = 0.5 * (n[0][1] + n[1][0]);
            n[0][1] = off;
            n[1][0] = off;
            pm = n;
            assertEquals(km.beta()[t], s[1], 1e-9);
            assertEquals(km.alpha()[t], s[0], 1e-9);
        }
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                assertEquals(km.finalCov()[i][j], pm[i][j],
                        1e-8 * Math.max(1.0, Math.abs(pm[i][j])));
            }
        }
    }

    @Test
    public void kalmanStepValidation() {
        double[][] bad = {{0.0, 1.0}, {1.0, 1.0}, {Double.NaN, 1.0}, {0.1, 0.0},
            {0.1, Double.POSITIVE_INFINITY}, {0.1, Double.NaN}};
        for (double[] b : bad) {
            assertThrows(IllegalArgumentException.class, () -> new KalmanStep(b[0], b[1]));
        }
    }

    @Test
    public void kalmanStepRejectsNonFiniteAndKeepsState() {
        KalmanStep ks = new KalmanStep(1e-4, 1e-3);
        KalmanStep ref = new KalmanStep(1e-4, 1e-3);
        assertEquals(ref.update(50.0, 100.0), ks.update(50.0, 100.0), 0.0);
        assertThrows(IllegalArgumentException.class, () -> ks.update(Double.NaN, 1.0));
        assertThrows(IllegalArgumentException.class, () -> ks.update(1.0, Double.POSITIVE_INFINITY));
        assertEquals(ref.update(51.0, 101.0), ks.update(51.0, 101.0), 0.0);
        assertEquals(ref.beta(), ks.beta(), 0.0);
    }

    // ---- item 8: lag selection and two-way EG -------------------------------

    private static double[] ar2(int n, long seed) {
        Random rng = new Random(seed);
        double[] v = new double[n];
        for (int t = 2; t < n; t++) {
            v[t] = 0.3 * v[t - 1] + 0.5 * v[t - 2] + rng.nextGaussian();
        }
        return v;
    }

    @Test
    public void selectLagFindsAugmentation() {
        // AR(2) levels = ADF regression with exactly one lagged difference
        assertEquals(1, Adf.selectLag(ar2(800, 2), 6, 1));
        Random rng = new Random(4);
        double[] w = new double[800];
        for (int t = 0; t < w.length; t++) {
            w[t] = rng.nextGaussian();
        }
        assertEquals(0, Adf.selectLag(w, 6, 1));
    }

    @Test
    public void selectLagMatchesDirectBic() {
        double[] v = ar2(300, 9);
        int m = 4;
        int n = v.length - 1 - m;
        double[] dy = new double[v.length - 1];
        for (int t = 0; t < dy.length; t++) {
            dy[t] = v[t + 1] - v[t];
        }
        double[] lhs = Arrays.copyOfRange(dy, m, dy.length);
        int best = 0;
        double bestBic = Double.POSITIVE_INFINITY;
        for (int p = 0; p <= m; p++) {
            double[][] design = new double[n][1 + p];
            for (int r = 0; r < n; r++) {
                design[r][0] = v[m + r];
                for (int i = 1; i <= p; i++) {
                    design[r][i] = dy[m + r - i];
                }
            }
            double[] e = Ols.fit(lhs, design, true).resid();
            double bic = n * Math.log(LinAlg.dot(e, e) / n) + (2 + p) * Math.log(n);
            if (bic < bestBic) {
                bestBic = bic;
                best = p;
            }
        }
        assertEquals(best, Adf.selectLag(v, m, 1));
    }

    @Test
    public void selectLagValidation() {
        double[] v = ar2(100, 2);
        int[][] bad = {{-1, 1}, {2, 4}, {90, 1}, {1001, 1}};
        for (int[] b : bad) {
            assertThrows(IllegalArgumentException.class, () -> Adf.selectLag(v, b[0], b[1]));
        }
        assertEquals(0, Adf.selectLag(v, 0, 1));
        Adf.test(v, Adf.selectLag(v, 3, 1), 1);
    }

    @Test
    public void engleGrangerBothNeedsBoth() {
        double[] y = TestData.slice(TestData.pair().get("stock_y"), 0, 900);
        double[] x = TestData.slice(TestData.pair().get("stock_x"), 0, 900);
        EngleGranger.BothResult both = EngleGranger.testBoth(y, x, 1);
        assertEquals(EngleGranger.test(y, x, 1).beta(), both.yOnX().beta(), 0.0);
        assertEquals(EngleGranger.test(x, y, 1).beta(), both.xOnY().beta(), 0.0);
        assertEquals(both.yOnX().cointegrated(0.05) && both.xOnY().cointegrated(0.05),
                both.cointegrated(0.05));
        assertFalse(EngleGranger.testBoth(TestData.walks().get("walk_a"),
                TestData.walks().get("walk_b"), 1).cointegrated(0.05));
    }
}
