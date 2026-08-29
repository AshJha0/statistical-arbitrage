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
}
