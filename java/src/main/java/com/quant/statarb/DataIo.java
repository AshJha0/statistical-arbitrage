package com.quant.statarb;

import java.io.IOException;
import java.math.BigInteger;
import java.nio.ByteBuffer;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Pattern;

/**
 * Tiny CSV / flat-JSON loaders so the package stays dependency-free.
 *
 * <p>Validation (pinned across languages, API_SPEC section 10): the file must
 * be valid UTF-8, exist and hold a header plus at least one data row; every row
 * must have exactly as many cells as the header; every cell, after trimming
 * ASCII spaces and tabs, must fully match the plain number grammar
 * {@code [+-]?([0-9]+\.?[0-9]*|\.[0-9]+)([eE][+-]?[0-9]+)?} and parse to a
 * FINITE double (so {@code NaN}, {@code Infinity}, hex floats, {@code _}
 * separators and type suffixes such as {@code 1.5f} or {@code 2d} are rejected
 * before Java's own parser sees them); column names must be unique; and if a
 * column named {@code day} exists its cells must be plain integers and the
 * column strictly increasing. The models assume EQUALLY SPACED consecutive
 * rows (next-close execution, lag-1 differences, sqrt(252)); gaps are the
 * caller's responsibility.
 *
 * <p>{@link #loadConfig} accepts exactly one flat JSON object (RFC 8259 number
 * grammar, nothing after the closing brace) holding finite numbers.
 */
public final class DataIo {

    /** Name of the optional time-index column validated on load. */
    public static final String DAY_COLUMN = "day";

    /** Config keys that must be written as JSON integers ({@code 60}, not {@code 60.0}). */
    public static final Set<String> INTEGER_KEYS =
            Set.of("window", "adf_maxlag", "max_hold_days", "cooldown_days");

    private static final Pattern NUMBER =
            Pattern.compile("[+-]?(?:[0-9]+\\.?[0-9]*|\\.[0-9]+)(?:[eE][+-]?[0-9]+)?");
    private static final Pattern INTEGER = Pattern.compile("[+-]?[0-9]+");
    private static final Pattern JSON_NUMBER =
            Pattern.compile("-?(?:0|[1-9][0-9]*)(?:\\.[0-9]+)?(?:[eE][+-]?[0-9]+)?");
    private static final BigInteger TWO_53 = BigInteger.ONE.shiftLeft(53);

    private DataIo() {
    }

    private static String readUtf8(Path path, String what) {
        if (!Files.isRegularFile(path)) {
            throw new IllegalArgumentException("no such " + what + " file: " + path);
        }
        byte[] bytes;
        try {
            bytes = Files.readAllBytes(path);
        } catch (IOException e) {
            throw new IllegalArgumentException("cannot read " + what + " file: " + path, e);
        }
        try {
            return StandardCharsets.UTF_8.newDecoder()
                    .onMalformedInput(CodingErrorAction.REPORT)
                    .onUnmappableCharacter(CodingErrorAction.REPORT)
                    .decode(ByteBuffer.wrap(bytes)).toString();
        } catch (CharacterCodingException e) {
            throw new IllegalArgumentException(
                    what + " file is not valid UTF-8: " + path, e);
        }
    }

    private static List<String> splitLines(String text) {
        List<String> lines = new ArrayList<>();
        int start = 0;
        int n = text.length();
        for (int i = 0; i < n; i++) {
            char c = text.charAt(i);
            if (c == '\n' || c == '\r') {
                lines.add(text.substring(start, i));
                if (c == '\r' && i + 1 < n && text.charAt(i + 1) == '\n') {
                    i++;
                }
                start = i + 1;
            }
        }
        if (start < n) {
            lines.add(text.substring(start));
        }
        return lines;
    }

    private static String trimBlanks(String cell) {
        int lo = 0;
        int hi = cell.length();
        while (lo < hi && (cell.charAt(lo) == ' ' || cell.charAt(lo) == '\t')) {
            lo++;
        }
        while (hi > lo && (cell.charAt(hi - 1) == ' ' || cell.charAt(hi - 1) == '\t')) {
            hi--;
        }
        return cell.substring(lo, hi);
    }

    /**
     * Load a simple headered numeric CSV into {column name -&gt; double[]}.
     *
     * @param path the CSV file
     * @return columns in file order
     * @throws IllegalArgumentException on a missing or non-UTF-8 file, empty
     *     file, ragged rows, cells outside the number grammar, non-finite
     *     cells, duplicate column names, a non-integer {@code day} cell, or a
     *     {@code day} column that is not strictly increasing
     */
    public static Map<String, double[]> loadCsv(Path path) {
        List<String> lines = splitLines(readUtf8(path, "data"));
        // drop leading/trailing blank lines only; an interior blank line is a
        // ragged row, as in the other ports
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
            header[j] = header[j].strip();
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
                String cell = trimBlanks(parts[j]);
                Pattern grammar = DAY_COLUMN.equals(header[j]) ? INTEGER : NUMBER;
                if (!grammar.matcher(cell).matches()) {
                    throw new IllegalArgumentException(
                            "non-numeric value at row " + (i + 2) + " in " + path);
                }
                double v = Double.parseDouble(cell);
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
     * Load a flat JSON object of finite numeric scalars (config.json).
     *
     * @param path the JSON file
     * @return key to value, in file order
     * @throws IllegalArgumentException on a missing or non-UTF-8 file, anything
     *     but one flat object of finite RFC 8259 numbers, trailing content
     *     (including after {@code {}}), an integer key
     *     ({@link #INTEGER_KEYS}) written with a fraction or exponent, or an
     *     integer of magnitude above 2^53
     */
    public static Map<String, Double> loadConfig(Path path) {
        return parseFlatJson(readUtf8(path, "config"), path);
    }

    private static Map<String, Double> parseFlatJson(String text, Path path) {
        Map<String, Double> out = new LinkedHashMap<>();
        int[] pos = {0};
        skipWs(text, pos);
        expect(text, pos, '{', path);
        skipWs(text, pos);
        if (peek(text, pos, path) == '}') {
            pos[0]++;
        } else {
            while (true) {
                skipWs(text, pos);
                String key = parseString(text, pos, path);
                skipWs(text, pos);
                expect(text, pos, ':', path);
                skipWs(text, pos);
                out.put(key, parseNumber(text, pos, path, key));
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
        }
        skipWs(text, pos);
        if (pos[0] != text.length()) {
            throw new IllegalArgumentException("trailing content in config JSON " + path);
        }
        return out;
    }

    private static void skipWs(String s, int[] pos) {
        while (pos[0] < s.length()) {
            char c = s.charAt(pos[0]);
            if (c != ' ' && c != '\t' && c != '\n' && c != '\r') {
                return;
            }
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
            if (c < 0x20) {
                throw new IllegalArgumentException(
                        "control character in config JSON string " + path);
            }
            if (c == '\\') {
                char e = peek(s, pos, path);
                pos[0]++;
                switch (e) {
                    case '"', '\\', '/' -> sb.append(e);
                    case 'b' -> sb.append('\b');
                    case 'f' -> sb.append('\f');
                    case 'n' -> sb.append('\n');
                    case 'r' -> sb.append('\r');
                    case 't' -> sb.append('\t');
                    case 'u' -> {
                        if (pos[0] + 4 > s.length()) {
                            throw new IllegalArgumentException(
                                    "bad \\u escape in config JSON " + path);
                        }
                        String hex = s.substring(pos[0], pos[0] + 4);
                        for (int i = 0; i < 4; i++) {
                            if (Character.digit(hex.charAt(i), 16) < 0
                                    || hex.charAt(i) > 'f') {
                                throw new IllegalArgumentException(
                                        "bad \\u escape in config JSON " + path);
                            }
                        }
                        sb.append((char) Integer.parseInt(hex, 16));
                        pos[0] += 4;
                    }
                    default -> throw new IllegalArgumentException(
                            "bad escape in config JSON string " + path);
                }
            } else {
                sb.append(c);
            }
        }
    }

    private static double parseNumber(String s, int[] pos, Path path, String key) {
        int start = pos[0];
        // take the maximal run of number-ish characters, then check the grammar
        // on the whole token, so "1f", "0x10", "1_0" and "1.5 2" cannot slip through
        while (pos[0] < s.length() && isTokenChar(s.charAt(pos[0]))) {
            pos[0]++;
        }
        String tok = s.substring(start, pos[0]);
        if (tok.isEmpty() || !JSON_NUMBER.matcher(tok).matches()) {
            throw new IllegalArgumentException(
                    "config JSON in " + path + " must hold finite numeric scalars (key '"
                            + key + "', offset " + start + ")");
        }
        boolean integer = tok.indexOf('.') < 0 && tok.indexOf('e') < 0 && tok.indexOf('E') < 0;
        if (integer && new BigInteger(tok).abs().compareTo(TWO_53) > 0) {
            throw new IllegalArgumentException(
                    "config key '" + key + "' is out of range in " + path);
        }
        if (INTEGER_KEYS.contains(key) && !integer) {
            throw new IllegalArgumentException(
                    "config key '" + key + "' must be an integer in " + path);
        }
        double v = Double.parseDouble(tok);
        if (!Double.isFinite(v)) {
            throw new IllegalArgumentException(
                    "config JSON in " + path + " must hold finite numeric scalars (key '"
                            + key + "')");
        }
        return v;
    }

    private static boolean isTokenChar(char c) {
        return (c >= '0' && c <= '9') || c == '+' || c == '-' || c == '.'
                || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
    }
}
