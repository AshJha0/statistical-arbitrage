package com.quant.statarb;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;

/**
 * Shared fixtures for the test suite: bundled datasets, config, and the
 * cross-language golden cases (loaded from ../data relative to java/).
 */
final class TestData {

    static final Path DATA = Paths.get("..", "data");

    private static Map<String, double[]> pair;
    private static Map<String, double[]> fx;
    private static Map<String, double[]> walks;
    private static Map<String, Double> config;
    private static List<Map<String, Object>> golden;

    private TestData() {
    }

    static synchronized Map<String, double[]> pair() {
        if (pair == null) {
            pair = DataIo.loadCsv(DATA.resolve("pair_equity.csv"));
        }
        return pair;
    }

    static synchronized Map<String, double[]> fx() {
        if (fx == null) {
            fx = DataIo.loadCsv(DATA.resolve("fx_triangle.csv"));
        }
        return fx;
    }

    static synchronized Map<String, double[]> walks() {
        if (walks == null) {
            walks = DataIo.loadCsv(DATA.resolve("random_walks.csv"));
        }
        return walks;
    }

    static synchronized Map<String, Double> config() {
        if (config == null) {
            config = DataIo.loadConfig(DATA.resolve("config.json"));
        }
        return config;
    }

    /** The golden cases as parsed maps: name, inputs, expect, tol. */
    @SuppressWarnings("unchecked")
    static synchronized List<Map<String, Object>> golden() {
        if (golden == null) {
            String text;
            try {
                text = Files.readString(DATA.resolve("golden").resolve("golden.json"));
            } catch (IOException e) {
                throw new IllegalStateException("cannot read golden.json", e);
            }
            Map<String, Object> root = (Map<String, Object>) Json.parse(text);
            golden = (List<Map<String, Object>>) (List<?>) root.get("cases");
        }
        return golden;
    }

    /** In-sample day count floor(T * insample_frac) for the bundled pair. */
    static int insampleDays() {
        return (int) Math.floor(pair().get("stock_y").length * config().get("insample_frac")
                + Backtest.SPLIT_EPS);
    }

    static double[] slice(double[] a, int from, int to) {
        return java.util.Arrays.copyOfRange(a, from, to);
    }

    /** The backtest config built from the bundled config.json. */
    static Backtest.Config btConfig() {
        Map<String, Double> c = config();
        return new Backtest.Config(c.get("window").intValue(), c.get("entry_z"),
                c.get("exit_z"), c.get("stop_z"), c.get("cost_bps"),
                c.get("slippage_bps"), c.get("capital"));
    }
}
