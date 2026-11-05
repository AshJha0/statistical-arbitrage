package com.quant.statarb;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Golden-value suite: EVERY case in ../data/golden/golden.json is recomputed
 * from the bundled CSVs and must agree within its stated tolerance. Cases with
 * an {@code inputs.kind} are dispatched on it, the older ones on their name;
 * there is no enumerated subset, so a new case type fails loudly.
 */
public class GoldenTest {

    /** Exact number of golden cases (kept in step with generate_data.py). */
    static final int GOLDEN_COUNT = 928;

    private static final Map<String, String> SIZING = Map.of(
            "limits_beta", Backtest.BETA_HEDGED, "limits_dollar", Backtest.EQUAL_DOLLAR,
            "plain_beta", Backtest.BETA_HEDGED, "plain_dollar", Backtest.EQUAL_DOLLAR);

    private static final Map<String, Backtest.Result> BT = new HashMap<>();
    private static final Map<String, Backtest.RollingResult> RW = new HashMap<>();
    private static Backtest.WalkForwardResult wf;
    private static EngleGranger.Result egInsample;

    private static double[] y() {
        return TestData.pair().get("stock_y");
    }

    private static double[] x() {
        return TestData.pair().get("stock_x");
    }

    private static Backtest.Config scenario(String name) {
        Backtest.Config c = TestData.btConfig().withSizing(SIZING.get(name));
        return name.startsWith("limits") ? c : c.withLimits(0.0, 0, 0);
    }

    private static synchronized EngleGranger.Result eg() {
        if (egInsample == null) {
            int n = TestData.insampleDays();
            egInsample = EngleGranger.test(TestData.slice(y(), 0, n), TestData.slice(x(), 0, n),
                    TestData.config().get("adf_maxlag").intValue());
        }
        return egInsample;
    }

    private static synchronized Backtest.Result bt(String name) {
        return BT.computeIfAbsent(name, k ->
                Backtest.run(y(), x(), eg().alpha(), eg().beta(), scenario(k)));
    }

    private static synchronized Backtest.WalkForwardResult wf() {
        if (wf == null) {
            wf = Backtest.walkForward(y(), x(), scenario("limits_beta"),
                    TestData.config().get("insample_frac"));
        }
        return wf;
    }

    private static synchronized Backtest.RollingResult rw(String name, int train, int test) {
        return RW.computeIfAbsent(name + "/" + train + "/" + test, k ->
                Backtest.rollingWalkForward(y(), x(), scenario(name), train, test));
    }

    private static double sum(double[] a) {
        double s = 0.0;
        for (double v : a) {
            s += v;
        }
        return s;
    }

    private static double field(Map<String, Double> metrics, List<Backtest.Trade> trades,
                                double[] equity, String field) {
        switch (field) {
            case "n_stops": {
                int c = 0;
                for (Backtest.Trade t : trades) {
                    c += t.stopped() ? 1 : 0;
                }
                return c;
            }
            case "equity_sum":
                return sum(equity);
            default:
                return metrics.get(field);
        }
    }

    private static double field(Backtest.Result r, String f) {
        return field(r.metrics(), r.trades(), r.equity(), f);
    }

    private static int intOf(Map<String, Object> in, String key) {
        return ((Double) in.get(key)).intValue();
    }

    private static double kind(Map<String, Object> in) {
        String kind = (String) in.get("kind");
        String f = (String) in.get("field");
        switch (kind) {
            case "backtest":
                return field(bt((String) in.get("scenario")), f);
            case "backtest_equity":
                return bt((String) in.get("scenario")).equity()[intOf(in, "day")];
            case "backtest_trade": {
                Backtest.Trade t = bt((String) in.get("scenario")).trades().get(intOf(in, "index"));
                switch (f) {
                    case "entry":
                        return t.entryDay();
                    case "exit":
                        return t.exitDay();
                    case "direction":
                        return t.direction();
                    case "pnl":
                        return t.pnl();
                    case "stopped":
                        return t.stopped() ? 1.0 : 0.0;
                    default:
                        throw new AssertionError("unknown trade field " + f);
                }
            }
            case "walk_forward": {
                Backtest.WalkForwardResult w = wf();
                String seg = (String) in.get("segment");
                if (seg.equals("fit")) {
                    return switch (f) {
                        case "alpha" -> w.alpha();
                        case "beta" -> w.beta();
                        case "split" -> w.split();
                        default -> throw new AssertionError("unknown fit field " + f);
                    };
                }
                return field(seg.equals("insample") ? w.insample() : w.oos(), f);
            }
            case "rolling": {
                Backtest.RollingResult r = rw((String) in.get("scenario"),
                        intOf(in, "train"), intOf(in, "test"));
                return f.equals("n_blocks") ? r.blocks().size()
                        : field(r.metrics(), r.trades(), r.equity(), f);
            }
            case "rolling_block": {
                Backtest.RollingBlock b = rw((String) in.get("scenario"),
                        intOf(in, "train"), intOf(in, "test")).blocks().get(intOf(in, "block"));
                return switch (f) {
                    case "start" -> b.start();
                    case "end" -> b.end();
                    case "alpha" -> b.alpha();
                    case "beta" -> b.beta();
                    case "skipped" -> b.skipped() ? 1.0 : 0.0;
                    default -> b.result().metrics().get(f);
                };
            }
            case "margin":
                return margin();
            default:
                throw new AssertionError("unknown golden kind: " + kind);
        }
    }

    /** Smallest ||z| - th| over entry/exit/stop and every z path of the goldens. */
    private static double margin() {
        List<double[]> paths = new ArrayList<>();
        for (String n : SIZING.keySet()) {
            paths.add(bt(n).zscore());
        }
        paths.add(wf().insample().zscore());
        paths.add(wf().oos().zscore());
        for (String n : new String[] {"limits_beta", "plain_beta"}) {
            for (Backtest.RollingBlock b : rw(n, 500, 100).blocks()) {
                paths.add(b.result().zscore());
            }
        }
        Map<String, Double> c = TestData.config();
        double best = Double.POSITIVE_INFINITY;
        for (double[] p : paths) {
            for (double z : p) {
                if (!Double.isNaN(z)) {
                    for (String k : new String[] {"entry_z", "exit_z", "stop_z"}) {
                        best = Math.min(best, Math.abs(Math.abs(z) - c.get(k)));
                    }
                }
            }
        }
        return best;
    }

    private static double[][] fxLogs() {
        Map<String, double[]> fx = TestData.fx();
        int n = fx.get("eurusd").length;
        double[][] out = new double[n][3];
        for (int t = 0; t < n; t++) {
            out[t][0] = Math.log(fx.get("eurusd")[t]);
            out[t][1] = Math.log(fx.get("gbpusd")[t]);
            out[t][2] = Math.log(fx.get("eurgbp")[t]);
        }
        return out;
    }

    private static double[] column(double[][] m, int lo, int hi, int j) {
        double[] out = new double[hi - lo];
        for (int t = lo; t < hi; t++) {
            out[t - lo] = m[t][j];
        }
        return out;
    }

    private static double[][] cols(double[][] m, int j0, int j1) {
        double[][] out = new double[m.length][j1 - j0];
        for (int t = 0; t < m.length; t++) {
            System.arraycopy(m[t], j0, out[t], 0, j1 - j0);
        }
        return out;
    }

    private static double named(String name, Map<String, Object> in) {
        double[] y = y();
        double[] x = x();
        Map<String, Double> cfg = TestData.config();
        int nIs = TestData.insampleDays();
        int maxlag = cfg.get("adf_maxlag").intValue();

        if (name.startsWith("adf_fx") || name.startsWith("johansen_fx")) {
            double[][] ly = fxLogs();
            if (name.startsWith("adf_fx")) {
                double[] resid = Ols.fit(column(ly, 0, ly.length, 0), cols(ly, 1, 3), true).resid();
                Adf.Result r = Adf.test(resid, intOf(in, "maxlag"), intOf(in, "nvars"));
                return switch (name.substring(name.lastIndexOf('_') + 1)) {
                    case "tstat" -> r.tstat();
                    case "crit1" -> r.crit1();
                    case "crit5" -> r.crit5();
                    case "crit10" -> r.crit10();
                    case "pvalue" -> r.pvalue();
                    default -> throw new AssertionError(name);
                };
            }
            Johansen.Result jo = Johansen.traceTest(ly);
            String tail = name.substring("johansen_fx_".length());
            if (tail.equals("rank")) {
                return jo.rank();
            }
            int i = tail.charAt(tail.length() - 1) - '0';
            return tail.startsWith("trace_r") ? jo.traceStats()[i] : jo.beta()[i][0];
        }
        if (name.startsWith("kalman_alpha") || name.startsWith("kalman_z")
                || name.startsWith("kalman_beta")) {
            boolean base = name.startsWith("kalman_beta");
            Kalman.Result km = Kalman.hedge(y, x,
                    base ? cfg.get("kalman_delta") : (Double) in.get("delta"),
                    base ? cfg.get("kalman_r") : (Double) in.get("r"));
            double[] path = name.startsWith("kalman_alpha") ? km.alpha()
                    : name.startsWith("kalman_z") ? km.z() : km.beta();
            return path[intOf(in, "day")];
        }
        if (name.startsWith("adf_resid_lag")) {
            Adf.Result r = Adf.test(eg().resid(), intOf(in, "maxlag"), intOf(in, "nvars"));
            return switch (name.substring(name.lastIndexOf('_') + 1)) {
                case "tstat" -> r.tstat();
                case "crit5" -> r.crit5();
                case "nobs" -> r.nobs();
                default -> throw new AssertionError(name);
            };
        }
        switch (name) {
            case "select_lag_resid":
                return Adf.selectLag(eg().resid(), intOf(in, "max_lag"), intOf(in, "nvars"));
            case "select_lag_walk":
                return Adf.selectLag(TestData.walks().get((String) in.get("column")),
                        intOf(in, "max_lag"), intOf(in, "nvars"));
            case "eg_both_insample_coint":
            case "eg_both_insample_x_on_y_tstat": {
                EngleGranger.BothResult both = EngleGranger.testBoth(
                        TestData.slice(y, 0, nIs), TestData.slice(x, 0, nIs), maxlag);
                return name.endsWith("coint") ? (both.cointegrated(0.05) ? 1.0 : 0.0)
                        : both.xOnY().adf().tstat();
            }
            case "halflife_growth_is_inf": {
                double[] s = new double[intOf(in, "n")];
                for (int t = 0; t < s.length; t++) {
                    s[t] = Math.pow(1.1, t);
                }
                return Double.isInfinite(Signals.halfLife(s)) ? 1.0 : 0.0;
            }
            case "halflife_alternating_zero": {
                double[] s = new double[intOf(in, "n")];
                for (int t = 0; t < s.length; t++) {
                    s[t] = t % 2 == 0 ? 1.0 : -1.0;
                }
                return Signals.halfLife(s);
            }
            case "ols_pair_alpha":
                return Ols.fit(y, x, true).params()[0];
            case "ols_pair_beta":
                return Ols.fit(y, x, true).params()[1];
            case "ols_pair_r2":
                return Ols.fit(y, x, true).r2();
            case "adf_random_walk_tstat":
            case "adf_random_walk_reject":
            case "adf_random_walk_crit5":
            case "adf_random_walk_pvalue": {
                Adf.Result r = Adf.test(TestData.walks().get((String) in.get("column")),
                        maxlag, 1);
                if (name.endsWith("tstat")) {
                    return r.tstat();
                }
                if (name.endsWith("reject")) {
                    return r.reject(0.05) ? 1.0 : 0.0;
                }
                return name.endsWith("crit5") ? r.crit5() : r.pvalue();
            }
            case "eg_spurious_coint":
                return EngleGranger.test(TestData.walks().get("walk_a"),
                        TestData.walks().get("walk_b"), maxlag).cointegrated(0.05) ? 1.0 : 0.0;
            case "adf_resid_insample_tstat":
                return eg().adf().tstat();
            case "adf_resid_insample_crit5":
                return eg().adf().crit5();
            case "eg_pair_insample_coint":
                return eg().cointegrated(0.05) ? 1.0 : 0.0;
            case "halflife_insample_spread":
                return Signals.halfLife(eg().resid());
            case "johansen_pair_trace_r0":
            case "johansen_pair_trace_r1":
            case "johansen_pair_rank": {
                double[][] lv = new double[nIs][2];
                for (int t = 0; t < nIs; t++) {
                    lv[t][0] = y[t];
                    lv[t][1] = x[t];
                }
                Johansen.Result jo = Johansen.traceTest(lv);
                return name.endsWith("r0") ? jo.traceStats()[0]
                        : name.endsWith("r1") ? jo.traceStats()[1] : jo.rank();
            }
            case "zscore_day200":
            case "zscore_day1100":
                return Signals.rollingZscore(Signals.spread(y, x, eg().alpha(), eg().beta()),
                        cfg.get("window").intValue())[intOf(in, "day")];
            case "backtest_total_return":
            case "backtest_sharpe":
            case "backtest_max_drawdown":
            case "backtest_n_trades":
                return bt("limits_beta").metrics().get(name.substring("backtest_".length()));
            default:
                throw new AssertionError("unknown golden case: " + name);
        }
    }

    @Test
    public void allGoldenCasesAgree() {
        List<Map<String, Object>> cases = TestData.golden();
        assertEquals("golden case count must be exact", GOLDEN_COUNT, cases.size());
        Set<String> names = new java.util.HashSet<>();
        int kinds = 0;
        for (Map<String, Object> c : cases) {
            String name = (String) c.get("name");
            assertTrue("duplicate case name " + name, names.add(name));
            @SuppressWarnings("unchecked")
            Map<String, Object> inputs = (Map<String, Object>) c.get("inputs");
            @SuppressWarnings("unchecked")
            double expect = (Double) ((Map<String, Object>) c.get("expect")).get("value");
            double tol = (Double) c.get("tol");
            double got;
            if (inputs.containsKey("kind")) {
                kinds++;
                got = kind(inputs);
                if ("margin".equals(inputs.get("kind"))) {
                    assertTrue("z threshold margin " + got, got > 1.0e-6);
                    assertTrue("stored margin " + expect, expect > 1.0e-6);
                }
            } else {
                got = named(name, inputs);
            }
            assertEquals("golden case " + name, expect, got, tol);
        }
        assertEquals("kind-dispatched cases", 874, kinds);
    }

    @Test
    public void integerCasesAreExact() {
        // rank must match exactly, not just within tolerance
        double[] y = y();
        double[] x = x();
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
            assertEquals(Set.of("name", "inputs", "expect", "tol"), c.keySet());
            @SuppressWarnings("unchecked")
            Map<String, Object> inputs = (Map<String, Object>) c.get("inputs");
            for (Object v : inputs.values()) {
                assertTrue(v instanceof Double || v instanceof String);
            }
            @SuppressWarnings("unchecked")
            Map<String, Object> expect = (Map<String, Object>) c.get("expect");
            assertEquals(Set.of("value"), expect.keySet());
            assertTrue((Double) c.get("tol") > 0.0);
        }
    }
}
