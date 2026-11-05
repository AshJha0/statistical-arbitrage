package com.quant.statarb;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import org.junit.Test;

/** Loader validation: day-column monotonicity, non-finite cells, config scalars. */
public class DataIoTest {

    private static Path temp(String text) throws IOException {
        Path p = Files.createTempFile("statarb_test", ".csv");
        p.toFile().deleteOnExit();
        Files.writeString(p, text);
        return p;
    }

    private static void expectCsvError(String text, String needle) throws IOException {
        Path p = temp(text);
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> DataIo.loadCsv(p));
        assertTrue(e.getMessage(), e.getMessage().contains(needle));
    }

    @Test
    public void csvDayColumnMustBeMonotone() throws IOException {
        // proposed test 16 (MAJ-6)
        expectCsvError("day,a\n0,1\n1,2\n2,3\n2,4\n3,5\n", "day column must be strictly increasing");
        expectCsvError("day,a\n0,1\n2,2\n1,3\n3,4\n", "day column must be strictly increasing");
        // gaps are allowed (strictly increasing); a file without `day` is not checked
        Map<String, double[]> gap = DataIo.loadCsv(temp("day,a\n0,1\n1,2\n3,3\n4,4\n"));
        assertArrayEquals(new double[] {0, 1, 3, 4}, gap.get("day"), 0.0);
        Map<String, double[]> nod = DataIo.loadCsv(temp("b,a\n5,1\n2,2\n"));
        assertArrayEquals(new double[] {5, 2}, nod.get("b"), 0.0);
    }

    @Test
    public void csvRejectsNonFiniteRaggedNonNumeric() throws IOException {
        expectCsvError("day,a\n0,1\n1,nan\n", "non-numeric");
        expectCsvError("day,a\n0,1\n1,NaN\n", "non-numeric");
        expectCsvError("day,a\n0,1\n1,Infinity\n", "non-numeric");
        expectCsvError("day,a\n0,1\n1,1e999\n", "non-finite");
        expectCsvError("day,a\n0,1\n1\n", "ragged");
        expectCsvError("day,a\n0,1\n\n1,2\n", "ragged"); // interior blank line
        expectCsvError("day,a\n0,1\n1,abc\n", "non-numeric");
        expectCsvError("day,a\n", "empty");
        expectCsvError("day,day\n0,1\n", "duplicate column");
        assertThrows(IllegalArgumentException.class,
                () -> DataIo.loadCsv(Path.of("/nonexistent/nope.csv")));
        // CRLF endings and trailing blank lines are tolerated
        Map<String, double[]> ok = DataIo.loadCsv(temp("day,a\r\n0,1\r\n1,2\r\n\r\n"));
        assertArrayEquals(new double[] {1, 2}, ok.get("a"), 0.0);
    }

    @Test
    public void bundledFilesPassValidation() {
        for (Map<String, double[]> d : java.util.List.of(TestData.pair(), TestData.fx(),
                TestData.walks())) {
            double[] day = d.get("day");
            assertEquals(1500, day.length);
            for (int i = 0; i < 1500; i++) {
                assertEquals(i, day[i], 0.0);
            }
        }
    }

    @Test
    public void configRejectsNonFiniteAndNonScalar() throws IOException {
        for (String text : new String[] {"{\"cost_bps\": NaN, \"window\": 60}",
                "{\"cost_bps\": Infinity}", "{\"cost_bps\": true}", "[1, 2]",
                "{\"a\": \"x\"}"}) {
            Path p = temp(text);
            assertThrows(text, IllegalArgumentException.class, () -> DataIo.loadConfig(p));
        }
        assertThrows(IllegalArgumentException.class,
                () -> DataIo.loadConfig(Path.of("/nonexistent/nope.json")));
        Map<String, Double> cfg = DataIo.loadConfig(temp("{\"a\": 1.5, \"b\": -2e3}"));
        assertEquals(1.5, cfg.get("a"), 0.0);
        assertEquals(-2000.0, cfg.get("b"), 0.0);
    }

    @Test
    public void csvPlainNumberGrammarOnly() throws IOException {
        for (String cell : new String[] {"0x10", "0x1p3", "1_000", "1f", "1d", "1F", "1D",
            "1.5f", "2d", "1.5L", "1e", "e5", ".", "+", "--1", "+ 1", "+1 . 5", "1 5", "1.5.2",
            "Infinity", "NaN", "\u0663", "1\u00a0", ""}) {
            Path p = temp("a,b\n1," + cell + "\n");
            assertThrows("cell '" + cell + "'", IllegalArgumentException.class,
                    () -> DataIo.loadCsv(p));
        }
    }

    @Test
    public void csvAcceptsPlainNumbers() throws IOException {
        String[] cells = {"1", "-1.5", "+2.", ".5", "1e3", "2.5E-1", " 7 ", "\t8\t", "+1.e2"};
        double[] want = {1.0, -1.5, 2.0, 0.5, 1000.0, 0.25, 7.0, 8.0, 100.0};
        for (int i = 0; i < cells.length; i++) {
            Map<String, double[]> m = DataIo.loadCsv(temp("a\n" + cells[i] + "\n"));
            assertEquals(cells[i], want[i], m.get("a")[0], 0.0);
        }
    }

    @Test
    public void csvDayCellsMustBeIntegers() throws IOException {
        for (String cell : new String[] {"1.0", "1.5", "1e0"}) {
            expectCsvError("day,a\n0,1\n" + cell + ",2\n", "non-numeric");
        }
    }

    @Test
    public void invalidUtf8IsReportedAsSuch() throws IOException {
        Path p = Files.createTempFile("statarb_test", ".csv");
        p.toFile().deleteOnExit();
        Files.write(p, new byte[] {'a', ',', 'b', '\n', '1', ',', (byte) 0xff, '\n'});
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> DataIo.loadCsv(p));
        assertTrue(e.getMessage(), e.getMessage().contains("not valid UTF-8"));
        assertFalse(e.getMessage(), e.getMessage().contains("no such"));
        Path q = Files.createTempFile("statarb_test", ".json");
        q.toFile().deleteOnExit();
        Files.write(q, new byte[] {'{', '"', 'a', (byte) 0xc3, '"', ':', '1', '}'});
        e = assertThrows(IllegalArgumentException.class, () -> DataIo.loadConfig(q));
        assertTrue(e.getMessage(), e.getMessage().contains("not valid UTF-8"));
        // valid multi-byte UTF-8 in a header is fine
        Files.write(p, "caf\u00e9,b\n1,2\n".getBytes(StandardCharsets.UTF_8));
        assertEquals(1.0, DataIo.loadCsv(p).get("caf\u00e9")[0], 0.0);
    }

    @Test
    public void configTrailingContentRejected() throws IOException {
        for (String text : new String[] {"{\"window\": 60} trailing", "{\"window\": 60}{\"a\": 1}",
            "{} x", "{}{}", "{} {\"a\": 1}", "{\"a\": 1},", "{\"a\": 1}\u0000", ""}) {
            Path p = temp(text);
            assertThrows(text, IllegalArgumentException.class, () -> DataIo.loadConfig(p));
        }
        assertTrue(DataIo.loadConfig(temp("  {}  \n")).isEmpty());
        assertEquals(60.0, DataIo.loadConfig(temp("{\"window\": 60}\n\n")).get("window"), 0.0);
    }

    @Test
    public void configStrictNumberGrammarAndIntegerKeys() throws IOException {
        for (String text : new String[] {"{\"window\": 60.0}", "{\"window\": 6e1}",
            "{\"max_hold_days\": 1.5}", "{\"cooldown_days\": 2.0}", "{\"adf_maxlag\": 1.0}",
            "{\"a\": 0x10}", "{\"a\": 1_000}", "{\"a\": 1f}", "{\"a\": 1d}", "{\"a\": 1.5f}",
            "{\"a\": +1}", "{\"a\": +1 . 5}", "{\"a\": 1 . 5}", "{\"a\": .5}", "{\"a\": 1.}",
            "{\"a\": 01}", "{\"a\": -}", "{\"a\": 0x1p3}", "{\"a\": 1e999}",
            "{\"a\": 1" + "0".repeat(400) + "}", "{\"a\": 9007199254740993}",
            "{\"a\": -9007199254740993}", "{\"a\": 1,}", "{\"a\" 1}", "{\"a\": null}",
            "{\"a\": [1]}", "{\"a\": {\"b\": 1}}", "{a: 1}", "{\"a\": \"\u0001\"}"}) {
            Path p = temp(text);
            assertThrows(text, IllegalArgumentException.class, () -> DataIo.loadConfig(p));
        }
        Map<String, Double> ok = DataIo.loadConfig(temp(
                "{\"window\": 60, \"a\": 9007199254740992, \"b\": 1.5E+2, \"c\": -0, \"d\": 0.5e-1}"));
        assertEquals(60.0, ok.get("window"), 0.0);
        assertEquals(9007199254740992.0, ok.get("a"), 0.0);
        assertEquals(150.0, ok.get("b"), 0.0);
        assertEquals(0.05, ok.get("d"), 0.0);
    }

    @Test
    public void bundledConfigCarriesRiskLimits() {
        Map<String, Double> c = TestData.config();
        assertEquals(0.02, c.get("pnl_stop_frac"), 0.0);
        assertEquals(60.0, c.get("max_hold_days"), 0.0);
        assertEquals(20.0, c.get("cooldown_days"), 0.0);
    }
}
