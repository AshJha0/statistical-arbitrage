package com.quant.statarb;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Map;

/**
 * Small latency and throughput harness (no dependency). This is NOT JMH.
 *
 * <p>Per-update latencies are measured over batches of {@value #BATCH} updates
 * with {@link System#nanoTime()} and divided by the batch size, so each reported
 * figure is a batch average: p50/p99/p99.9/max are percentiles of those batch
 * averages, not of single updates, and hide sub-batch outliers. Results are
 * folded into a sink that is printed, so the JIT cannot drop the work. A long
 * warm-up runs first. Treat numbers as indicative for one machine and one run.
 *
 * <p>Run: {@code bash bench.sh [dataDir]} (after {@code bash build.sh}).
 */
public final class Bench {

    private static final int BATCH = 64;
    private static final int WARMUP_BATCHES = 200_000;
    private static final int MEASURED_BATCHES = 100_000;

    private static double sink;

    private Bench() {
    }

    private static void report(String name, double[] perUpdateNs) {
        double[] a = perUpdateNs.clone();
        Arrays.sort(a);
        int n = a.length;
        System.out.printf("%-18s p50 %8.1f  p99 %8.1f  p99.9 %8.1f  max %10.1f  (ns/update, batch avg)%n",
                name, a[n / 2], a[(int) Math.min(n - 1, Math.floor(n * 0.99))],
                a[(int) Math.min(n - 1, Math.floor(n * 0.999))], a[n - 1]);
    }

    private static void benchRollingZ(double[] s) {
        RollingZ rz = new RollingZ(60);
        int m = s.length;
        int idx = 0;
        double acc = 0.0;
        double[] out = new double[MEASURED_BATCHES];
        for (int b = 0; b < WARMUP_BATCHES + MEASURED_BATCHES; b++) {
            long t0 = System.nanoTime();
            for (int i = 0; i < BATCH; i++) {
                double z = rz.update(s[idx]);
                acc += Double.isNaN(z) ? 0.0 : z;
                idx = idx + 1 == m ? 0 : idx + 1;
            }
            long dt = System.nanoTime() - t0;
            if (b >= WARMUP_BATCHES) {
                out[b - WARMUP_BATCHES] = (double) dt / BATCH;
            }
        }
        sink += acc;
        report("RollingZ.update", out);
    }

    private static void benchKalman(double[] y, double[] x, double delta, double r) {
        KalmanStep ks = new KalmanStep(delta, r);
        int m = y.length;
        int idx = 0;
        double acc = 0.0;
        double[] out = new double[MEASURED_BATCHES];
        for (int b = 0; b < WARMUP_BATCHES + MEASURED_BATCHES; b++) {
            long t0 = System.nanoTime();
            for (int i = 0; i < BATCH; i++) {
                acc += ks.update(x[idx], y[idx]);
                idx = idx + 1 == m ? 0 : idx + 1;
            }
            long dt = System.nanoTime() - t0;
            if (b >= WARMUP_BATCHES) {
                out[b - WARMUP_BATCHES] = (double) dt / BATCH;
            }
        }
        sink += acc + ks.beta();
        report("KalmanStep.update", out);
    }

    private static void throughput(String name, int items, int warm, int reps, Runnable r) {
        for (int i = 0; i < warm; i++) {
            r.run();
        }
        long t0 = System.nanoTime();
        for (int i = 0; i < reps; i++) {
            r.run();
        }
        double sec = (System.nanoTime() - t0) / 1e9;
        System.out.printf("%-18s %12.0f obs/s  (%d obs x %d runs, %.2f s)%n",
                name, (double) items * reps / sec, items, reps, sec);
    }

    /**
     * Entry point.
     *
     * @param args optional data directory (default {@code ../data})
     */
    public static void main(String[] args) {
        Path data = Paths.get(args.length > 0 ? args[0] : "../data");
        Map<String, Double> cfg = DataIo.loadConfig(data.resolve("config.json"));
        Map<String, double[]> pair = DataIo.loadCsv(data.resolve("pair_equity.csv"));
        double[] y = pair.get("stock_y");
        double[] x = pair.get("stock_x");
        double[] spread = Signals.spread(y, x, 0.0, 1.7);
        double delta = cfg.get("kalman_delta");
        double r = cfg.get("kalman_r");
        Backtest.Config bc = Backtest.Config.fromMap(cfg);
        System.out.println("statarb Bench: batches of " + BATCH + ", warm-up " + WARMUP_BATCHES
                + " batches, " + MEASURED_BATCHES + " measured. Per-update figures are batch"
                + " averages; this is not JMH.");
        benchRollingZ(spread);
        benchKalman(y, x, delta, r);
        int w = bc.window();
        throughput("rollingZscore", spread.length, 2000, 5000,
                () -> sink += Signals.rollingZscore(spread, w)[spread.length - 1]);
        throughput("Backtest.run", y.length, 1000, 3000,
                () -> sink += Backtest.run(y, x, 0.0, 1.7, bc).equity()[y.length - 1]);
        System.out.println("sink " + sink);
    }
}
