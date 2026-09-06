package com.quant.statarb;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Tiny CSV / flat-JSON loaders so the package stays dependency-free.
 *
 * <p>Validation (pinned across languages): the file must exist and hold a
 * header plus at least one data row; every row must have exactly as many
 * cells as the header; every cell must parse as a FINITE number (NaN/inf
 * are rejected, not propagated); column names must be unique; and if a
 * column named {@code day} exists it must be strictly increasing — an
 * unsorted, duplicated or re-merged time index silently changes every lag,
 * z-score and fill in the backtest. The models further assume the rows are
 * EQUALLY SPACED consecutive observations (next-close execution, lag-1
 * differences, sqrt(252)); gaps are the caller's responsibility
 * (API_SPEC section 10).
 */
public final class DataIo {

    /** Name of the optional time-index column validated on load. */
    public static final String DAY_COLUMN = "day";

    private DataIo() {
    }

    /**
     * Load a simple headered numeric CSV into {column name -&gt; double[]}.
     *
     * @throws IllegalArgumentException on a missing file, empty file,
     *     ragged, non-numeric or non-finite rows, duplicate column names, or a
     *     {@code day} column that is not strictly increasing
     */
    public static Map<String, double[]> loadCsv(Path path) {
        List<String> lines;
        try {
            lines = Files.readAllLines(path);
        } catch (IOException e) {
            throw new IllegalArgumentException("no such data file: " + path, e);
        }
        // drop leading/trailing blank lines only (mirrors Python's strip());
        // an interior blank line is a ragged row, as in the other ports
        while (!lines.isEmpty() && lines.get(lines.size() - 1).isBlank()) {
            lines.remove(lines.size() - 1);
        }
        while (!lines.isEmpty() && lines.get(0).isBlank()) {
            lines.remove(0);
        }
        if (lines.size() < 2) {
            throw new IllegalArgumentException("empty data file: " + path);
        }
        String[] header = lines.get(0).split(",", -1);
        int ncol = header.length;
        for (int j = 0; j < ncol; j++) {
            header[j] = header[j].trim();
            for (int k = 0; k < j; k++) {
                if (header[k].equals(header[j])) {
                    throw new IllegalArgumentException("duplicate column names in " + path);
                }
            }
        }
        int nrow = lines.size() - 1;
        double[][] data = new double[ncol][nrow];
        for (int i = 0; i < nrow; i++) {
            String[] parts = lines.get(i + 1).split(",", -1);
            if (parts.length != ncol) {
                throw new IllegalArgumentException("ragged row " + (i + 2) + " in " + path);
            }
            for (int j = 0; j < ncol; j++) {
                double v;
                try {
                    v = Double.parseDouble(parts[j].trim());
                } catch (NumberFormatException e) {
                    throw new IllegalArgumentException(
                            "non-numeric value at row " + (i + 2) + " in " + path, e);
                }
                if (!Double.isFinite(v)) {
                    throw new IllegalArgumentException(
                            "non-finite value at row " + (i + 2) + " in " + path);
                }
                data[j][i] = v;
            }
        }
        Map<String, double[]> out = new LinkedHashMap<>();
        for (int j = 0; j < ncol; j++) {
            out.put(header[j], data[j]);
        }
        double[] day = out.get(DAY_COLUMN);
        if (day != null) {
            for (int i = 1; i < day.length; i++) {
                if (!(day[i] > day[i - 1])) {
                    throw new IllegalArgumentException(
                            "day column must be strictly increasing (sorted, no duplicates) in "
                                    + path);
                }
            }
        }
        return out;
    }

    /**
     * Load the bundled {@code config.json} — a flat JSON object of finite
     * numeric scalars — into {key -&gt; double}.
     *
     * @throws IllegalArgumentException on a missing file or malformed /
     *     non-flat / non-numeric / non-finite content (booleans, strings and
     *     NaN/Infinity literals are rejected)
     */
    public static Map<String, Double> loadConfig(Path path) {
        String text;
        try {
            text = Files.readString(path);
        } catch (IOException e) {
            throw new IllegalArgumentException("no such config file: " + path, e);
        }
        return parseFlatJson(text, path);
    }

    private static Map<String, Double> parseFlatJson(String text, Path path) {
        Map<String, Double> out = new LinkedHashMap<>();
        int[] pos = {0};
        skipWs(text, pos);
        expect(text, pos, '{', path);
        skipWs(text, pos);
        if (peek(text, pos, path) == '}') {
            pos[0]++;
            return out;
        }
        while (true) {
            skipWs(text, pos);
            String key = parseString(text, pos, path);
            skipWs(text, pos);
            expect(text, pos, ':', path);
            skipWs(text, pos);
            out.put(key, parseNumber(text, pos, path));
            skipWs(text, pos);
            char c = peek(text, pos, path);
            pos[0]++;
            if (c == '}') {
                break;
            }
            if (c != ',') {
                throw new IllegalArgumentException(
                        "malformed config JSON in " + path + " at offset " + (pos[0] - 1));
            }
        }
        skipWs(text, pos);
        if (pos[0] != text.length()) {
            throw new IllegalArgumentException("trailing content in config JSON " + path);
        }
        return out;
    }

    private static void skipWs(String s, int[] pos) {
        while (pos[0] < s.length() && Character.isWhitespace(s.charAt(pos[0]))) {
            pos[0]++;
        }
    }

    private static char peek(String s, int[] pos, Path path) {
        if (pos[0] >= s.length()) {
            throw new IllegalArgumentException("unexpected end of config JSON " + path);
        }
        return s.charAt(pos[0]);
    }

    private static void expect(String s, int[] pos, char c, Path path) {
        if (peek(s, pos, path) != c) {
            throw new IllegalArgumentException(
                    "malformed config JSON in " + path + ": expected '" + c
                            + "' at offset " + pos[0]);
        }
        pos[0]++;
    }

    private static String parseString(String s, int[] pos, Path path) {
        expect(s, pos, '"', path);
        StringBuilder sb = new StringBuilder();
        while (true) {
            char c = peek(s, pos, path);
            pos[0]++;
            if (c == '"') {
                return sb.toString();
            }
            if (c == '\\') {
                throw new IllegalArgumentException(
                        "escapes not supported in flat config JSON " + path);
            }
            sb.append(c);
        }
    }

    private static double parseNumber(String s, int[] pos, Path path) {
        int start = pos[0];
        while (pos[0] < s.length() && "+-0123456789.eE".indexOf(s.charAt(pos[0])) >= 0) {
            pos[0]++;
        }
        if (start == pos[0]) {
            throw new IllegalArgumentException(
                    "config JSON in " + path + " must hold numeric scalars (offset "
                            + start + ")");
        }
        double v;
        try {
            v = Double.parseDouble(s.substring(start, pos[0]));
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(
                    "bad number in config JSON " + path + " at offset " + start, e);
        }
        if (!Double.isFinite(v)) {
            throw new IllegalArgumentException(
                    "config JSON in " + path + " must hold finite numeric scalars (offset "
                            + start + ")");
        }
        return v;
    }
}
