package com.quant.statarb;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.List;
import java.util.Map;
import org.junit.Test;

/**
 * Golden-value suite: every case in ../data/golden/golden.json is recomputed
 * from the bundled CSVs and must agree within its stated tolerance.
 */
public class GoldenTest {

    // Heavy pipelines computed once and shared across cases.
    private static EngleGranger.Result egInsample;
    private static Kalman.Result kalman;
    private static Backtest.Result backtest;

    private static synchronized EngleGranger.Result egInsample() {
        if (egInsample == null) {
            int nIs = TestData.insampleDays();
            egInsample = EngleGranger.test(
                    TestData.slice(TestData.pair().get("stock_y"), 0, nIs),
                    TestData.slice(TestData.pair().get("stock_x"), 0, nIs),
                    TestData.config().get("adf_maxlag").intValue());
        }
        return egInsample;
    }

    private static synchronized Kalman.Result kalman() {
        if (kalman == null) {
            kalman = Kalman.hedge(TestData.pair().get("stock_y"),
                    TestData.pair().get("stock_x"),
                    TestData.config().get("kalman_delta"),
                    TestData.config().get("kalman_r"));
        }
        return kalman;
    }

    private static synchronized Backtest.Result backtest() {
        if (backtest == null) {
            EngleGranger.Result eg = egInsample();
            backtest = Backtest.run(TestData.pair().get("stock_y"),
                    TestData.pair().get("stock_x"), eg.alpha(), eg.beta(),
                    TestData.btConfig());
        }
        return backtest;
    }

    private static double compute(String name, Map<String, Object> inputs) {
        double[] y = TestData.pair().get("stock_y");
        double[] x = TestData.pair().get("stock_x");
        int nIs = TestData.insampleDays();
        int maxlag = TestData.config().get("adf_maxlag").intValue();

        switch (name) {
            case "ols_pair_alpha":
                return Ols.fit(y, x, true).params()[0];
            case "ols_pair_beta":
                return Ols.fit(y, x, true).params()[1];
            case "ols_pair_r2":
                return Ols.fit(y, x, true).r2();
            case "adf_random_walk_tstat":
            case "adf_random_walk_reject": {
                double[] col = TestData.walks().get((String) inputs.get("column"));
                Adf.Result res = Adf.test(col, maxlag, 1);
                return name.endsWith("tstat") ? res.tstat() : (res.reject(0.05) ? 1.0 : 0.0);
            }
            case "eg_spurious_coint": {
                EngleGranger.Result eg = EngleGranger.test(
                        TestData.walks().get("walk_a"), TestData.walks().get("walk_b"), maxlag);
                return eg.cointegrated(0.05) ? 1.0 : 0.0;
            }
            case "adf_resid_insample_tstat":
                return egInsample().adf().tstat();
            case "eg_pair_insample_coint":
                return egInsample().cointegrated(0.05) ? 1.0 : 0.0;
            case "halflife_insample_spread":
                return Signals.halfLife(egInsample().resid());
            case "johansen_pair_trace_r0":
            case "johansen_pair_trace_r1":
            case "johansen_pair_rank": {
                double[][] lv = new double[nIs][2];
                for (int t = 0; t < nIs; t++) {
                    lv[t][0] = y[t];
                    lv[t][1] = x[t];
                }
                Johansen.Result jo = Johansen.traceTest(lv);
                if (name.endsWith("r0")) {
                    return jo.traceStats()[0];
                }
                if (name.endsWith("r1")) {
                    return jo.traceStats()[1];
                }
                return jo.rank();
            }
            case "kalman_beta_day500":
            case "kalman_beta_day1200":
            case "kalman_beta_final":
                return kalman().beta()[((Double) inputs.get("day")).intValue()];
            case "zscore_day200":
            case "zscore_day1100": {
                EngleGranger.Result eg = egInsample();
                double[] z = Signals.rollingZscore(
                        Signals.spread(y, x, eg.alpha(), eg.beta()),
                        TestData.config().get("window").intValue());
                return z[((Double) inputs.get("day")).intValue()];
            }
            case "backtest_total_return":
            case "backtest_sharpe":
            case "backtest_max_drawdown":
            case "backtest_n_trades":
                return backtest().metrics().get(name.substring("backtest_".length()));
            default:
                throw new AssertionError("unknown golden case: " + name);
        }
    }

    @Test
    public void allGoldenCasesAgree() {
        List<Map<String, Object>> cases = TestData.golden();
        assertTrue("expected at least 18 golden cases", cases.size() >= 18);
        for (Map<String, Object> c : cases) {
            String name = (String) c.get("name");
            @SuppressWarnings("unchecked")
            Map<String, Object> inputs = (Map<String, Object>) c.get("inputs");
            @SuppressWarnings("unchecked")
            double expect = (Double) ((Map<String, Object>) c.get("expect")).get("value");
            double tol = (Double) c.get("tol");
            double got = compute(name, inputs);
            assertEquals("golden case " + name, expect, got, tol);
        }
    }

    @Test
    public void integerCasesAreExact() {
        // trade count and rank must match exactly, not just within tolerance
        assertEquals(31.0, backtest().metrics().get("n_trades"), 0.0);
        double[] y = TestData.pair().get("stock_y");
        double[] x = TestData.pair().get("stock_x");
        int nIs = TestData.insampleDays();
        double[][] lv = new double[nIs][2];
        for (int t = 0; t < nIs; t++) {
            lv[t][0] = y[t];
            lv[t][1] = x[t];
        }
        assertEquals(1, Johansen.traceTest(lv).rank());
    }

    @Test
    public void goldenSchemaIsFlat() {
        for (Map<String, Object> c : TestData.golden()) {
            assertEquals(java.util.Set.of("name", "inputs", "expect", "tol"), c.keySet());
            @SuppressWarnings("unchecked")
            Map<String, Object> expect = (Map<String, Object>) c.get("expect");
            assertEquals(java.util.Set.of("value"), expect.keySet());
            assertTrue((Double) c.get("tol") > 0.0);
        }
    }
}
