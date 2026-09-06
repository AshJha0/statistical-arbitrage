package com.quant.statarb;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import java.util.Map;
import org.junit.Test;

/**
 * Backtest: accounting identity, dollar neutrality, stops, no-trade path,
 * next-day execution, walk-forward, config validation.
 */
public class BacktestTest {

    private static Backtest.Result bundled;
    private static EngleGranger.Result bundledEg;

    private static synchronized Backtest.Result bundledRun() {
        if (bundled == null) {
            int nIs = TestData.insampleDays();
            bundledEg = EngleGranger.test(
                    TestData.slice(TestData.pair().get("stock_y"), 0, nIs),
                    TestData.slice(TestData.pair().get("stock_x"), 0, nIs),
                    TestData.config().get("adf_maxlag").intValue());
            bundled = Backtest.run(TestData.pair().get("stock_y"),
                    TestData.pair().get("stock_x"), bundledEg.alpha(), bundledEg.beta(),
                    TestData.btConfig());
        }
        return bundled;
    }

    @Test
    public void accountingIdentity() {
        // E_{T-1} - capital == sum(pnl) - sum(costs), and per-day increments
        Backtest.Result res = bundledRun();
        double cap = TestData.config().get("capital");
        double sumPnl = LinAlg.sum(res.dailyPnl());
        double sumCost = LinAlg.sum(res.dailyCost());
        int n = res.equity().length;
        assertEquals(sumPnl - sumCost, res.equity()[n - 1] - cap, 1e-10 * cap);
        double prev = cap;
        for (int t = 0; t < n; t++) {
            assertEquals(res.dailyPnl()[t] - res.dailyCost()[t],
                    res.equity()[t] - prev, 1e-8);
            prev = res.equity()[t];
        }
    }

    @Test
    public void dollarNeutralityAtEntry() {
        // at each entry execution the two legs carry capital/2 notional each:
        // the entry-day traded notional equals gross capital and costs equal
        // capital * bps
        Backtest.Result res = bundledRun();
        double cap = TestData.config().get("capital");
        double bps = (TestData.config().get("cost_bps")
                + TestData.config().get("slippage_bps")) / 1e4;
        assertTrue(res.trades().size() > 0);
        for (Backtest.Trade tr : res.trades()) {
            assertEquals(cap * bps, res.dailyCost()[tr.entryDay()], 1e-6);
        }
    }

    @Test
    public void stopLossTriggersInBreakRegime() {
        Backtest.Result res = bundledRun();
        assertTrue(res.nStops() >= 1);
        boolean stopAfterBreak = false;
        for (Backtest.Trade t : res.trades()) {
            if (t.stopped() && t.exitDay() > 1050) {
                stopAfterBreak = true;
            }
        }
        assertTrue("at least one stop fires inside the break regime", stopAfterBreak);
    }

    @Test
    public void tradesWellFormed() {
        Backtest.Result res = bundledRun();
        Map<String, Double> m = res.metrics();
        for (Backtest.Trade tr : res.trades()) {
            assertTrue(tr.exitDay() > tr.entryDay());
            assertTrue(tr.direction() == 1 || tr.direction() == -1);
            assertEquals(tr.exitDay() - tr.entryDay(), tr.holdingDays());
        }
        assertEquals(res.trades().size(), m.get("n_trades"), 0.0);
        assertTrue(m.get("win_rate") >= 0.0 && m.get("win_rate") <= 1.0);
        assertTrue(m.get("avg_holding_days") > 0.0);
        assertTrue(m.get("max_drawdown") >= 0.0);
        // each round trip trades ~2x capital (entry + exit legs)
        assertTrue(m.get("turnover") >= res.trades().size());
    }

    @Test
    public void noTradesMetricsGuarded() {
        // unreachable entry threshold: flat equity, all metrics zero/finite
        double cap = TestData.config().get("capital");
        double[] y = TestData.slice(TestData.pair().get("stock_y"), 0, 400);
        double[] x = TestData.slice(TestData.pair().get("stock_x"), 0, 400);
        Backtest.Config cfg = new Backtest.Config(60, 50.0, 0.5, 60.0, 1.0, 0.5, cap);
        Backtest.Result res = Backtest.run(y, x, 12.0, 1.7, cfg);
        assertEquals(0, res.trades().size());
        for (double e : res.equity()) {
            assertEquals(cap, e, 0.0);
        }
        for (String key : new String[] {"total_return", "ann_return", "sharpe",
                "max_drawdown", "n_trades", "win_rate", "avg_holding_days", "turnover"}) {
            assertEquals("metric " + key, 0.0, res.metrics().get(key), 0.0);
        }
    }

    @Test
    public void nextDayExecutionNoLookahead() {
        // no entry cost may fall on the same day its signal fired: the first
        // |z| >= entry day must precede the first entry execution day
        Backtest.Result res = bundledRun();
        double entryZ = TestData.config().get("entry_z");
        double[] z = res.zscore();
        int firstSig = -1;
        for (int t = 0; t < z.length; t++) {
            if (Double.isFinite(z[t]) && Math.abs(z[t]) >= entryZ) {
                firstSig = t;
                break;
            }
        }
        int firstEntry = Integer.MAX_VALUE;
        for (Backtest.Trade tr : res.trades()) {
            firstEntry = Math.min(firstEntry, tr.entryDay());
        }
        assertEquals(firstSig + 1, firstEntry);
    }

    @Test
    public void walkForwardSplitAndDegradation() {
        double[] y = TestData.pair().get("stock_y");
        double[] x = TestData.pair().get("stock_x");
        double frac = TestData.config().get("insample_frac");
        Backtest.WalkForwardResult wf =
                Backtest.walkForward(y, x, TestData.btConfig(), frac);
        assertEquals((int) (y.length * frac), wf.split());
        assertEquals(wf.split(), wf.insample().equity().length);
        assertEquals(y.length - wf.split(), wf.oos().equity().length);
        // the OOS window contains the regime break: honest degradation
        assertTrue(wf.oos().metrics().get("sharpe") < wf.insample().metrics().get("sharpe"));
    }

    @Test
    public void configValidation() {
        assertThrows(IllegalArgumentException.class,
                () -> new Backtest.Config(1, 2.0, 0.5, 4.0, 1.0, 0.5, 1e6));
        assertThrows(IllegalArgumentException.class,
                () -> new Backtest.Config(60, 2.0, 2.5, 4.0, 1.0, 0.5, 1e6));
        assertThrows(IllegalArgumentException.class,
                () -> new Backtest.Config(60, 2.0, 0.5, 1.5, 1.0, 0.5, 1e6));
        assertThrows(IllegalArgumentException.class,
                () -> new Backtest.Config(60, 2.0, 0.5, 4.0, -1.0, 0.5, 1e6));
        assertThrows(IllegalArgumentException.class,
                () -> new Backtest.Config(60, 2.0, 0.5, 4.0, 1.0, 0.5, 0.0));
        assertThrows(IllegalArgumentException.class,
                () -> new Backtest.Config(60, 2.0, 0.5, 4.0, 1.0, 0.5,
                        Double.POSITIVE_INFINITY));
    }

    @Test
    public void backtestInputErrors() {
        Backtest.Config cfg = TestData.btConfig();
        double[] y = new double[200];
        for (int i = 0; i < 200; i++) {
            y[i] = 100.0 + 10.0 * i / 199.0;
        }
        assertThrows(IllegalArgumentException.class,
                () -> Backtest.run(y, TestData.slice(y, 0, 199), 0.0, 1.0, cfg));
        double[] neg = new double[200];
        for (int i = 0; i < 200; i++) {
            neg[i] = y[i] - 200.0;
        }
        assertThrows(IllegalArgumentException.class,
                () -> Backtest.run(neg, y, 0.0, 1.0, cfg));
        double[] shortY = TestData.slice(y, 0, 30);
        double[] shortX = new double[30];
        for (int i = 0; i < 30; i++) {
            shortX[i] = shortY[i] + 1.0;
        }
        assertThrows(IllegalArgumentException.class,
                () -> Backtest.run(shortY, shortX, 0.0, 1.0, cfg)); // window 60 > 30
        double[] x2 = new double[200];
        for (int i = 0; i < 200; i++) {
            x2[i] = y[i] + 1.0;
        }
        assertThrows(IllegalArgumentException.class,
                () -> Backtest.walkForward(y, x2, cfg, 1.5));
    }

    // -----------------------------------------------------------------------
    // Synthetic-spread fixtures.  Every expected value is derived independently
    // of the backtest: z-scores are recomputed by zpath (plain rolling mean /
    // ddof-1 std) and closed forms are used where a window holds a single
    // spike on a zero background: z_spike = (W-1)/sqrt(W), z_next = -1/sqrt(W).
    // -----------------------------------------------------------------------

    private static double[] zpath(double[] s, int w) {
        double[] z = new double[s.length];
        java.util.Arrays.fill(z, Double.NaN);
        for (int t = w - 1; t < s.length; t++) {
            double m = 0.0;
            for (int i = t + 1 - w; i <= t; i++) {
                m += s[i];
            }
            m /= w;
            double ss = 0.0;
            for (int i = t + 1 - w; i <= t; i++) {
                ss += (s[i] - m) * (s[i] - m);
            }
            double sd = Math.sqrt(ss / (w - 1));
            if (sd > 1e-12) {
                z[t] = (s[t] - m) / sd;
            }
        }
        return z;
    }

    private static double maxAbsFinite(double[] z, int lo, int hi) {
        double m = 0.0;
        for (int t = lo; t < hi; t++) {
            if (Double.isFinite(z[t])) {
                m = Math.max(m, Math.abs(z[t]));
            }
        }
        return m;
    }

    @Test
    public void endOfDataPendingEntryDiscarded() {
        // proposed test 1 (CRIT-1): |z_{T-2}| in [entry, stop) while flat would
        // execute at T-1 and be force-closed the same close; the spec discards it
        int n = 80;
        double[] x = new double[n];
        double[] s = new double[n];
        double[] y = new double[n];
        for (int t = 0; t < n; t++) {
            x[t] = 200.0 + 0.5 * Math.sin(t);
            s[t] = Math.sin(0.7 * t);
        }
        s[n - 2] = 6.0;
        for (int t = 0; t < n; t++) {
            y[t] = 10.0 + x[t] + s[t];
        }
        Backtest.Config cfg = new Backtest.Config(20, 2.0, 0.5, 8.0, 1.0, 0.5, 1e6);
        double[] z = zpath(s, 20);
        assertTrue(maxAbsFinite(z, 0, n - 2) < 2.0);
        assertTrue(z[n - 2] >= 2.0 && z[n - 2] < 8.0);
        Backtest.Result res = Backtest.run(y, x, 10.0, 1.0, cfg);
        assertTrue(res.trades().isEmpty());
        assertEquals(0.0, res.metrics().get("n_trades"), 0.0);
        assertEquals(0.0, res.metrics().get("turnover"), 0.0);
        for (int t = 0; t < n; t++) {
            assertEquals(1e6, res.equity()[t], 0.0);
            assertEquals(0.0, res.dailyCost()[t], 0.0);
        }
    }

    @Test
    public void zeroDayRoundTripNeverReported() {
        // proposed test 2: over every truncation of the bundled pair no trade
        // may ever have exitDay == entryDay (would have caught CRIT-1)
        bundledRun();
        double[] y = TestData.pair().get("stock_y");
        double[] x = TestData.pair().get("stock_x");
        for (int n = 70; n <= y.length; n++) {
            Backtest.Result res = Backtest.run(TestData.slice(y, 0, n), TestData.slice(x, 0, n),
                    bundledEg.alpha(), bundledEg.beta(), TestData.btConfig());
            for (Backtest.Trade tr : res.trades()) {
                assertTrue("T=" + n + ": " + tr, tr.exitDay() > tr.entryDay());
                assertTrue(tr.exitDay() <= n - 1);
            }
        }
    }

    @Test
    public void stopBandWhileFlatNoEntry() {
        // proposed test 3: a single spike on a zero background gives
        // z = (W-1)/sqrt(W) = 6.166 >= stopZ while flat (no entry, pinned) and
        // z = -1/sqrt(W) the next day: no trades, flat equity
        int n = 120;
        int w = 40;
        int t0 = 60;
        double[] s = new double[n];
        s[t0] = 7.0;
        double[] x = new double[n];
        java.util.Arrays.fill(x, 50.0);
        double[] y = new double[n];
        for (int t = 0; t < n; t++) {
            y[t] = 100.0 + s[t]; // alpha 50, beta 1 -> spread = s
        }
        Backtest.Config cfg = new Backtest.Config(w, 2.0, 0.5, 4.0, 1.0, 0.5, 1e6);
        double[] z = zpath(s, w);
        assertEquals((w - 1) / Math.sqrt(w), z[t0], 1e-12);
        assertEquals(-1.0 / Math.sqrt(w), z[t0 + 1], 1e-12);
        assertTrue(z[t0] >= cfg.stopZ());
        for (int t = 0; t < t0; t++) {
            assertTrue(Double.isNaN(z[t]));
        }
        Backtest.Result res = Backtest.run(y, x, 50.0, 1.0, cfg);
        assertTrue(res.trades().isEmpty());
        for (double e : res.equity()) {
            assertEquals(1e6, e, 0.0);
        }
    }

    @Test
    public void stopExitIsFlaggedAndExecutesNextDay() {
        // proposed test 12: entry band at t0, in-position plateau, |z| >= stop
        // at t1 = t0+3 -> one trade, stopped, exitDay == t1+1, exit-day cost
        // equals bps * exit notional recomputed from the pinned legs
        int n = 160;
        int w = 40;
        int t0 = 70;
        double[] s = new double[n];
        double[] x = new double[n];
        double[] y = new double[n];
        for (int t = 0; t < n; t++) {
            s[t] = 0.2 * Math.sin(0.7 * t);
            x[t] = 200.0 + 0.5 * Math.sin(t);
        }
        s[t0] = 0.5;
        s[t0 + 1] = 0.5;
        s[t0 + 2] = 0.5;
        s[t0 + 3] = 5.0;
        for (int t = 0; t < n; t++) {
            y[t] = 10.0 + x[t] + s[t];
        }
        Backtest.Config cfg = new Backtest.Config(w, 2.0, 0.5, 4.0, 1.0, 0.5, 1e6);
        double[] z = zpath(s, w);
        assertTrue(maxAbsFinite(z, 0, t0) < cfg.entryZ());
        assertTrue(z[t0] >= cfg.entryZ() && z[t0] < cfg.stopZ());
        for (int t = t0 + 1; t < t0 + 3; t++) {
            assertTrue(Math.abs(z[t]) > cfg.exitZ() && Math.abs(z[t]) < cfg.stopZ());
        }
        assertTrue(z[t0 + 3] >= cfg.stopZ());
        assertTrue(maxAbsFinite(z, t0 + 4, n) < cfg.entryZ());
        Backtest.Result res = Backtest.run(y, x, 10.0, 1.0, cfg);
        assertEquals(1, res.trades().size());
        Backtest.Trade tr = res.trades().get(0);
        assertEquals(t0 + 1, tr.entryDay());
        assertEquals(t0 + 4, tr.exitDay());
        assertEquals(-1, tr.direction());
        assertTrue(tr.stopped());
        assertEquals(1, res.nStops());
        double bps = 1.5e-4;
        double cap = 1e6;
        double qy = -(cap / 2.0) / y[t0 + 1];
        double qx = (cap / 2.0) / x[t0 + 1];
        assertEquals(cap * bps, res.dailyCost()[t0 + 1], 1e-9);
        double exitNotional = Math.abs(qy) * y[t0 + 4] + Math.abs(qx) * x[t0 + 4];
        assertEquals(exitNotional * bps, res.dailyCost()[t0 + 4], 1e-9);
        assertTrue(res.dailyCost()[t0 + 4] > 0.0);
        assertEquals(3.0, res.metrics().get("avg_holding_days"), 0.0);
    }

    @Test
    public void reentryAfterExitSameDirection() {
        // proposed test 13: enter-short, exit, enter-short again with no
        // cool-down: two trades with entryDay_2 == exitDay_1 + 1
        int n = 160;
        int w = 40;
        int t0 = 70;
        double[] s = new double[n];
        double[] x = new double[n];
        double[] y = new double[n];
        for (int t = 0; t < n; t++) {
            s[t] = 0.2 * Math.sin(0.7 * t);
            x[t] = 200.0 + 0.5 * Math.sin(t);
        }
        s[t0] = 0.5;
        s[t0 + 1] = 0.0;
        s[t0 + 2] = 0.5;
        s[t0 + 3] = 0.0;
        for (int t = 0; t < n; t++) {
            y[t] = 10.0 + x[t] + s[t];
        }
        Backtest.Config cfg = new Backtest.Config(w, 2.0, 0.5, 4.0, 1.0, 0.5, 1e6);
        double[] z = zpath(s, w);
        assertTrue(z[t0] >= cfg.entryZ() && z[t0] < cfg.stopZ());
        assertTrue(Math.abs(z[t0 + 1]) <= cfg.exitZ());
        assertTrue(z[t0 + 2] >= cfg.entryZ() && z[t0 + 2] < cfg.stopZ());
        assertTrue(Math.abs(z[t0 + 3]) <= cfg.exitZ());
        assertTrue(maxAbsFinite(z, 0, t0) < cfg.entryZ());
        assertTrue(maxAbsFinite(z, t0 + 4, n) < cfg.entryZ());
        Backtest.Result res = Backtest.run(y, x, 10.0, 1.0, cfg);
        assertEquals(2, res.trades().size());
        assertEquals(new Backtest.Trade(t0 + 1, t0 + 2, -1, res.trades().get(0).pnl(), false),
                res.trades().get(0));
        assertEquals(new Backtest.Trade(t0 + 3, t0 + 4, -1, res.trades().get(1).pnl(), false),
                res.trades().get(1));
        assertEquals(res.trades().get(0).exitDay() + 1, res.trades().get(1).entryDay());
    }

    @Test
    public void equityNonPositiveMetricsDefined() {
        // proposed test 14 (MIN-11): a 4x jump on the short leg loses 1.5x
        // capital: total_return = -1.5 exactly, ann_return pinned to -1.0,
        // max_drawdown = 1.5, finite Sharpe, no NaN anywhere
        int n = 45;
        int w = 40;
        int t0 = 40;
        double[] s = new double[n];
        double[] x = new double[n];
        double[] y = new double[n];
        for (int t = 0; t < n; t++) {
            s[t] = t < t0 ? 0.2 * Math.sin(0.7 * t) : -0.5; // long-spread entry band
            x[t] = t < t0 + 2 ? 50.0 : 200.0;                // short leg quadruples
            y[t] = 100.0 + s[t];                              // alpha 50, beta 1
        }
        Backtest.Config cfg = new Backtest.Config(w, 2.0, 0.5, 4.0, 0.0, 0.0, 1e6);
        Backtest.Result res = Backtest.run(y, x, 50.0, 1.0, cfg);
        assertEquals(1, res.trades().size());
        Backtest.Trade tr = res.trades().get(0);
        assertEquals(t0 + 1, tr.entryDay());
        assertEquals(t0 + 3, tr.exitDay());
        assertEquals(1, tr.direction());
        assertTrue(tr.stopped());
        assertEquals(-1.5e6, tr.pnl(), 1e-6); // q_x = -1e4 shares * 150
        assertEquals(-5.0e5, res.equity()[n - 1], 1e-6);
        Map<String, Double> m = res.metrics();
        assertEquals(-1.5, m.get("total_return"), 1e-12);
        assertEquals(-1.0, m.get("ann_return"), 0.0);
        assertEquals(1.5, m.get("max_drawdown"), 1e-12);
        double ret = -1.5;
        double mean = ret / n;
        double var = ((n - 1) * mean * mean + (ret - mean) * (ret - mean)) / (n - 1);
        assertEquals(mean / Math.sqrt(var) * Math.sqrt(252.0), m.get("sharpe"), 1e-10);
        assertEquals(3.5, m.get("turnover"), 1e-12); // 1e6 in, 2.5e6 out
        for (double v : m.values()) {
            assertTrue(Double.isFinite(v));
        }
    }

    @Test
    public void configRejectsNonFinite() {
        // proposed test 5 (MAJ-2): NaN/inf thresholds and costs are errors
        double nan = Double.NaN;
        double inf = Double.POSITIVE_INFINITY;
        assertThrows(IllegalArgumentException.class,
                () -> new Backtest.Config(60, 2.0, 0.5, 4.0, nan, 0.5, 1e6));
        assertThrows(IllegalArgumentException.class,
                () -> new Backtest.Config(60, 2.0, 0.5, 4.0, 1.0, inf, 1e6));
        assertThrows(IllegalArgumentException.class,
                () -> new Backtest.Config(60, nan, 0.5, 4.0, 1.0, 0.5, 1e6));
        assertThrows(IllegalArgumentException.class,
                () -> new Backtest.Config(60, 2.0, 0.5, nan, 1.0, 0.5, 1e6));
        assertThrows(IllegalArgumentException.class,
                () -> new Backtest.Config(60, 2.0, -inf, 4.0, 1.0, 0.5, 1e6));
        assertThrows(IllegalArgumentException.class,
                () -> new Backtest.Config(60, 2.0, 0.5, inf, 1.0, 0.5, 1e6));
        assertThrows(IllegalArgumentException.class,
                () -> new Backtest.Config(60, 2.0, 0.5, 4.0, 1.0, 0.5, nan));
    }

    @Test
    public void walkForwardLengthMismatchIsTypedError() {
        // proposed test 7 (MAJ-5): previously copyOfRange zero-padded x and the
        // run failed later with a misleading "strictly positive" message
        double[] y = TestData.pair().get("stock_y");
        double[] x100 = TestData.slice(TestData.pair().get("stock_x"), 0, 100);
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> Backtest.walkForward(y, x100, TestData.btConfig(), 0.6));
        assertTrue(e.getMessage(), e.getMessage().contains("equal length"));
    }

    @Test
    public void walkForwardSplitExact() {
        // proposed test 8 (MIN-10): floor(T * frac) is the intended integer
        // (700 * 0.7 == 489.99999999999994 in binary floating point)
        Backtest.Config cfg = new Backtest.Config(5, 50.0, 0.5, 60.0, 1.0, 0.5, 1e6);
        int[] lengths = {700, 100, 1500};
        double[] fracs = {0.7, 0.29, 0.6};
        int[] expected = {490, 29, 900};
        for (int c = 0; c < lengths.length; c++) {
            double[] y = new double[lengths[c]];
            double[] x = new double[lengths[c]];
            for (int t = 0; t < lengths[c]; t++) {
                y[t] = 100.0 + Math.sin(0.3 * t);
                x[t] = 50.0 + Math.cos(0.2 * t);
            }
            assertEquals(expected[c], Backtest.walkForward(y, x, cfg, fracs[c]).split());
        }
    }

    @Test
    public void computeMetricsValidation() {
        java.util.List<Backtest.Trade> none = java.util.List.of();
        assertThrows(IllegalArgumentException.class,
                () -> Backtest.computeMetrics(new double[0], 1e6, none));
        assertThrows(IllegalArgumentException.class,
                () -> Backtest.computeMetrics(new double[] {1e6, Double.NaN}, 1e6, none));
        assertThrows(IllegalArgumentException.class,
                () -> Backtest.computeMetrics(new double[] {1e6}, 0.0, none));
        Map<String, Double> m = Backtest.computeMetrics(new double[] {1e6}, 1e6, none);
        assertEquals(0.0, m.get("sharpe"), 0.0);
        assertEquals(0.0, m.get("total_return"), 0.0);
        assertEquals(0.0, m.get("max_drawdown"), 0.0);
    }
}
