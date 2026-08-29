package com.quant.statarb;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Tiny CSV / flat-JSON loaders so the package stays dependency-free.
 */
public final class DataIo {

    private DataIo() {
    }

    /**
     * Load a simple headered numeric CSV into {column name -&gt; double[]}.
     *
     * @throws IllegalArgumentException on a missing file, empty file, or
     *     ragged/non-numeric rows
     */
    public static Map<String, double[]> loadCsv(Path path) {
        List<String> lines;
        try {
            lines = Files.readAllLines(path);
        } catch (IOException e) {
            throw new IllegalArgumentException("no such data file: " + path, e);
        }
        while (!lines.isEmpty() && lines.get(lines.size() - 1).isBlank()) {
            lines.remove(lines.size() - 1);
        }
        if (lines.size() < 2) {
            throw new IllegalArgumentException("empty data file: " + path);
        }
        String[] header = lines.get(0).split(",");
        int ncol = header.length;
        int nrow = lines.size() - 1;
        double[][] data = new double[ncol][nrow];
        for (int i = 0; i < nrow; i++) {
            String[] parts = lines.get(i + 1).split(",");
            if (parts.length != ncol) {
                throw new IllegalArgumentException("ragged row " + (i + 2) + " in " + path);
            }
            for (int j = 0; j < ncol; j++) {
                try {
                    data[j][i] = Double.parseDouble(parts[j].trim());
                } catch (NumberFormatException e) {
                    throw new IllegalArgumentException(
                            "non-numeric value at row " + (i + 2) + " in " + path, e);
                }
            }
        }
        Map<String, double[]> out = new LinkedHashMap<>();
        for (int j = 0; j < ncol; j++) {
            out.put(header[j].trim(), data[j]);
        }
        return out;
    }

    /**
     * Load the bundled {@code config.json} — a flat JSON object of numeric
     * scalars — into {key -&gt; double}.
     *
     * @throws IllegalArgumentException on a missing file or malformed /
     *     non-flat / non-numeric content
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
        try {
            return Double.parseDouble(s.substring(start, pos[0]));
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(
                    "bad number in config JSON " + path + " at offset " + start, e);
        }
    }
}
