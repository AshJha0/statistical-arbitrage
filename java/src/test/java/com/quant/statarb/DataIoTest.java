package com.quant.statarb;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import java.io.IOException;
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
        expectCsvError("day,a\n0,1\n1,nan\n", "non-numeric"); // Java parses only "NaN"
        expectCsvError("day,a\n0,1\n1,NaN\n", "non-finite");
        expectCsvError("day,a\n0,1\n1,Infinity\n", "non-finite");
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
}
