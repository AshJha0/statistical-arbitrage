package com.quant.statarb;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Map;

/**
 * statarb end-to-end demo.
 *
 * <p>Sections:
 * <ol>
 *   <li>Cointegration diagnostics on the bundled equity pair (EG + Johansen).</li>
 *   <li>Kalman vs static OLS beta tracking through the day-1050 regime break.</li>
 *   <li>Full backtest report: in-sample vs walk-forward out-of-sample.</li>
 *   <li>Spurious-regression teaching case (independent random walks).</li>
 * </ol>
 *
 * <p>Run: {@code bash demo.sh} from the java/ directory (after {@code bash build.sh}).
 */
public final class Demo {

    private Demo() {
    }

    private static void rule(String title) {
        StringBuilder sb = new StringBuilder("\n=== ").append(title).append(' ');
        int pad = Math.max(0, 66 - title.length());
        sb.append("=".repeat(pad));
        System.out.println(sb);
    }

    private static void printMetrics(String label, Backtest.Result res) {
        Map<String, Double> m = res.metrics();
        System.out.printf(
                "  %-12s ret %7.2f%%  ann %7.2f%%  sharpe %6.2f  maxDD %5.2f%%  "
                        + "trades %3d  win %4.1f%%  hold %5.1fd  turn %5.1fx  stops %d%n",
                label, m.get("total_return") * 100.0, m.get("ann_return") * 100.0,
                m.get("sharpe"), m.get("max_drawdown") * 100.0,
                m.get("n_trades").intValue(), m.get("win_rate") * 100.0,
                m.get("avg_holding_days"), m.get("turnover"), res.nStops());
    }

    /**
     * Entry point; prints the compact report and exits 0.
     *
     * @param args optional first argument: path to the data directory
     *             (default {@code ../data} relative to the working directory)
     */
    public static void main(String[] args) {
        Path data = Paths.get(args.length > 0 ? args[0] : "../data");
        Map<String, Double> cfg = DataIo.loadConfig(data.resolve("config.json"));
        Map<String, double[]> pair = DataIo.loadCsv(data.resolve("pair_equity.csv"));
        Map<String, double[]> fx = DataIo.loadCsv(data.resolve("fx_triangle.csv"));
        Map<String, double[]> walks = DataIo.loadCsv(data.resolve("random_walks.csv"));
        double[] y = pair.get("stock_y");
        double[] x = pair.get("stock_x");
        int nIs = (int) Math.floor(y.length * cfg.get("insample_frac") + Backtest.SPLIT_EPS);
        int maxlag = cfg.get("adf_maxlag").intValue();
        Backtest.Config btCfg = Backtest.Config.fromMap(cfg);

        System.out.println("statarb demo — cointegration, Kalman hedging, pairs backtest");
        System.out.printf("pair_equity.csv: %d days, in-sample = first %d days "
                + "(true beta 1.7, break at day 1050)%n", y.length, nIs);

        rule(String.format("1. Cointegration diagnostics (in-sample, days 0..%d)", nIs - 1));
        double[] yIn = java.util.Arrays.copyOfRange(y, 0, nIs);
        double[] xIn = java.util.Arrays.copyOfRange(x, 0, nIs);
        EngleGranger.Result eg = EngleGranger.test(yIn, xIn, maxlag);
        System.out.printf("  EG step 1  y = %.4f + %.4f x   (R^2 %.4f)%n",
                eg.alpha(), eg.beta(), eg.r2());
        System.out.printf("  EG step 2  ADF(resid) t = %.4f  crit 5%% (2-var) = %.2f  p ~ %.4f%n",
                eg.adf().tstat(), eg.adf().crit5(), eg.adf().pvalue());
        System.out.printf("  verdict    cointegrated at 5%%: %s%n", eg.cointegrated(0.05));
        System.out.printf("  half-life  %.2f days%n", Signals.halfLife(eg.resid()));
        double[][] lv = new double[nIs][2];
        for (int t = 0; t < nIs; t++) {
            lv[t][0] = y[t];
            lv[t][1] = x[t];
        }
        Johansen.Result jo = Johansen.traceTest(lv);
        for (int r = 0; r < 2; r++) {
            String flag = jo.traceStats()[r] > jo.crit95()[r] ? ">" : "<";
            System.out.printf("  Johansen   trace(r=%d) = %8.3f %s cv95 %6.2f%n",
                    r, jo.traceStats()[r], flag, jo.crit95()[r]);
        }
        System.out.printf("  Johansen   rank = %d, cointegrating vector (1, %.4f)  "
                + "[true beta 1.7]%n", jo.rank(), jo.beta()[1][0]);
        double[] eurusd = fx.get("eurusd");
        double[][] fxLv = new double[eurusd.length][3];
        for (int t = 0; t < eurusd.length; t++) {
            fxLv[t][0] = Math.log(eurusd[t]);
            fxLv[t][1] = Math.log(fx.get("gbpusd")[t]);
            fxLv[t][2] = Math.log(fx.get("eurgbp")[t]);
        }
        Johansen.Result jfx = Johansen.traceTest(fxLv);
        System.out.printf("  FX triangle (3 assets, logs): rank = %d, leading vector "
                + "(%.2f, %.2f, %.2f)  [triangle: (1, -1, -1)]%n",
                jfx.rank(), jfx.beta()[0][0], jfx.beta()[1][0], jfx.beta()[2][0]);

        rule("2. Kalman vs static OLS through the regime break");
        Kalman.Result km = Kalman.hedge(y, x, cfg.get("kalman_delta"), cfg.get("kalman_r"));
        double bOls = Ols.fit(y, x, true).params()[1];
        double[] trueBeta = new double[y.length];
        java.util.Arrays.fill(trueBeta, 1.7);
        for (int t = 1050; t <= 1080; t++) {
            trueBeta[t] = 1.7 + (1.1 - 1.7) * (t - 1050) / 30.0;
        }
        for (int t = 1081; t < y.length; t++) {
            trueBeta[t] = 1.1;
        }
        System.out.println("  day    true beta   kalman beta   static OLS beta");
        for (int d : new int[] {250, 500, 900, 1049, 1100, 1200, 1499}) {
            System.out.printf("  %5d   %8.3f   %10.4f   %12.4f%n",
                    d, trueBeta[d], km.beta()[d], bOls);
        }
        double teK = 0.0;
        double teO = 0.0;
        int cnt = y.length - 300;
        for (int t = 300; t < y.length; t++) {
            teK += (km.beta()[t] - trueBeta[t]) * (km.beta()[t] - trueBeta[t]);
            teO += (bOls - trueBeta[t]) * (bOls - trueBeta[t]);
        }
        System.out.printf("  beta tracking RMSE (day 300+): kalman %.4f vs static OLS %.4f "
                + "-> kalman adapts, OLS cannot%n",
                Math.sqrt(teK / cnt), Math.sqrt(teO / cnt));

        rule("3. Pairs backtest — in-sample vs walk-forward out-of-sample");
        Backtest.WalkForwardResult wf =
                Backtest.walkForward(y, x, btCfg, cfg.get("insample_frac"));
        System.out.printf("  hedge estimated on days 0..%d: alpha %.4f, beta %.4f%n",
                wf.split() - 1, wf.alpha(), wf.beta());
        printMetrics("in-sample", wf.insample());
        printMetrics("OOS (40%)", wf.oos());
        System.out.println("  degradation is honest: the OOS window contains the day-1050");
        System.out.println("  break; stop-losses cap the damage of the broken relationship.");
        System.out.println("  (walk_forward is ONE fit and ONE split; the refitting version follows.)");
        Backtest.RollingResult rw = Backtest.rollingWalkForward(y, x, btCfg, 500, 100);
        System.out.printf("  rolling walk-forward (train 500, test 100): %d blocks, "
                + "total return %.4f, %d trades, %d stops%n", rw.blocks().size(),
                rw.metrics().get("total_return"), rw.trades().size(), rw.nStops());

        rule("4. Spurious-regression teaching case (independent random walks)");
        double[] wa = walks.get("walk_a");
        double[] wb = walks.get("walk_b");
        Adf.Result a1 = Adf.test(wa, maxlag, 1);
        EngleGranger.Result sp = EngleGranger.test(wa, wb, maxlag);
        double slopeT = Ols.fit(wa, wb, true).tstats()[1];
        System.out.printf("  regressing walk_a on walk_b: slope t-stat = %.1f —%n", slopeT);
        System.out.println("  'significant' by the usual +/-2 rule, yet the walks are independent.");
        System.out.printf("  ADF(walk_a) t = %.3f (unit root not rejected) and%n", a1.tstat());
        System.out.printf("  ADF(resid) t = %.3f vs 5%% crit %.2f -> cointegrated: %s%n",
                sp.adf().tstat(), sp.adf().crit5(), sp.cointegrated(0.05));
        System.out.println("  verdict: spurious relationship — NO backtest is run on this pair.");

        System.out.println("\ndemo complete.");
    }
}
