//! Tiny CSV / JSON loaders (the crate's only dependency is `serde_json`
//! for the flat config file).
//!
//! Validation (pinned across languages): the file must exist and hold a
//! header plus at least one data row; every row must have exactly as many
//! cells as the header; every cell must parse as a FINITE number
//! (`nan`/`inf` are rejected, not propagated); column names must be
//! unique; and if a column named `day` exists it must be strictly
//! increasing — an unsorted, duplicated or re-merged time index silently
//! changes every lag, z-score and fill in the backtest. The models further
//! assume the rows are EQUALLY SPACED consecutive observations (next-close
//! execution, lag-1 differences, `sqrt(252)`); gaps are the caller's
//! responsibility (API_SPEC section 10).

use std::collections::HashMap;
use std::path::Path;

use crate::error::{Result, StatarbError};

/// Name of the optional time-index column validated on load.
pub const DAY_COLUMN: &str = "day";

/// Load a simple headered numeric CSV into `{column_name: values}`.
///
/// # Errors
///
/// [`StatarbError::Data`] on a missing/empty file, ragged, non-numeric or
/// non-finite rows, duplicate column names, or a `day` column that is not
/// strictly increasing.
pub fn load_csv(path: &Path) -> Result<HashMap<String, Vec<f64>>> {
    let text = std::fs::read_to_string(path)
        .map_err(|_| StatarbError::Data(format!("no such data file: {}", path.display())))?;
    let lines: Vec<&str> = text.trim().lines().map(|l| l.trim_end_matches('\r')).collect();
    if lines.len() < 2 {
        return Err(StatarbError::Data(format!(
            "empty data file: {}",
            path.display()
        )));
    }
    let header: Vec<String> = lines[0].split(',').map(|h| h.trim().to_string()).collect();
    let ncol = header.len();
    for a in 0..ncol {
        if header[a + 1..].contains(&header[a]) {
            return Err(StatarbError::Data(format!(
                "duplicate column names in {}",
                path.display()
            )));
        }
    }
    let mut cols: Vec<Vec<f64>> = vec![Vec::with_capacity(lines.len() - 1); ncol];
    for (i, line) in lines[1..].iter().enumerate() {
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() != ncol {
            return Err(StatarbError::Data(format!(
                "ragged row {} in {}",
                i + 2,
                path.display()
            )));
        }
        for (j, part) in parts.iter().enumerate() {
            let v: f64 = part.trim().parse().map_err(|_| {
                StatarbError::Data(format!(
                    "non-numeric value at row {} in {}",
                    i + 2,
                    path.display()
                ))
            })?;
            if !v.is_finite() {
                return Err(StatarbError::Data(format!(
                    "non-finite value at row {} in {}",
                    i + 2,
                    path.display()
                )));
            }
            cols[j].push(v);
        }
    }
    if let Some(j) = header.iter().position(|h| h == DAY_COLUMN) {
        if cols[j].windows(2).any(|w| !(w[1] > w[0])) {
            return Err(StatarbError::Data(format!(
                "day column must be strictly increasing (sorted, no duplicates) in {}",
                path.display()
            )));
        }
    }
    Ok(header.into_iter().zip(cols).collect())
}

/// Load the bundled `config.json` (flat JSON object of finite numeric
/// scalars).
///
/// # Errors
///
/// [`StatarbError::Data`] on a missing file, invalid JSON, a non-object
/// document, or a value that is not a finite number (booleans, strings,
/// nested values and NaN/Infinity are rejected).
pub fn load_config(path: &Path) -> Result<HashMap<String, f64>> {
    let text = std::fs::read_to_string(path)
        .map_err(|_| StatarbError::Data(format!("no such config file: {}", path.display())))?;
    let value: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| StatarbError::Data(format!("invalid JSON in {}: {e}", path.display())))?;
    let obj = value
        .as_object()
        .ok_or_else(|| StatarbError::Data("config.json must contain a JSON object".to_string()))?;
    let mut out = HashMap::with_capacity(obj.len());
    for (k, v) in obj {
        let num = v.as_f64().filter(|x| x.is_finite()).ok_or_else(|| {
            StatarbError::Data(format!("config key '{k}' is not a finite numeric scalar"))
        })?;
        out.insert(k.clone(), num);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp(name: &str, text: &str) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!("statarb_{}_{name}", std::process::id()));
        std::fs::write(&p, text).unwrap();
        p
    }

    fn expect_data_error(name: &str, text: &str, needle: &str) {
        let p = temp(name, text);
        match load_csv(&p) {
            Err(StatarbError::Data(m)) => assert!(m.contains(needle), "{m}"),
            other => panic!("expected Data error containing '{needle}', got {other:?}"),
        }
        let _ = std::fs::remove_file(p);
    }

    #[test]
    fn missing_file_is_data_error() {
        assert!(matches!(
            load_csv(Path::new("/nonexistent/nope.csv")),
            Err(StatarbError::Data(_))
        ));
        assert!(matches!(
            load_config(Path::new("/nonexistent/nope.json")),
            Err(StatarbError::Data(_))
        ));
    }

    #[test]
    fn csv_day_column_must_be_monotone() {
        // proposed test 16 (MAJ-6)
        expect_data_error("dup.csv", "day,a\n0,1\n1,2\n2,3\n2,4\n3,5\n",
                          "day column must be strictly increasing");
        expect_data_error("unsorted.csv", "day,a\n0,1\n2,2\n1,3\n3,4\n",
                          "day column must be strictly increasing");
        // gaps are allowed; a file without `day` is not checked
        let p = temp("gap.csv", "day,a\n0,1\n1,2\n3,3\n4,4\n");
        assert_eq!(load_csv(&p).unwrap()["day"], vec![0.0, 1.0, 3.0, 4.0]);
        let p = temp("nod.csv", "b,a\n5,1\n2,2\n");
        assert_eq!(load_csv(&p).unwrap()["b"], vec![5.0, 2.0]);
    }

    #[test]
    fn csv_rejects_nonfinite_ragged_nonnumeric() {
        expect_data_error("nan.csv", "day,a\n0,1\n1,nan\n", "non-finite");
        expect_data_error("inf.csv", "day,a\n0,1\n1,inf\n", "non-finite");
        expect_data_error("rag.csv", "day,a\n0,1\n1\n", "ragged");
        expect_data_error("blank.csv", "day,a\n0,1\n\n1,2\n", "ragged");
        expect_data_error("txt.csv", "day,a\n0,1\n1,abc\n", "non-numeric");
        expect_data_error("empty.csv", "day,a\n", "empty");
        expect_data_error("dupcol.csv", "day,day\n0,1\n", "duplicate column");
        // CRLF endings and trailing blank lines are tolerated
        let p = temp("crlf.csv", "day,a\r\n0,1\r\n1,2\r\n\r\n");
        assert_eq!(load_csv(&p).unwrap()["a"], vec![1.0, 2.0]);
    }

    #[test]
    fn config_rejects_nonfinite_and_nonscalar() {
        for (name, text) in [
            ("c1.json", "{\"cost_bps\": NaN, \"window\": 60}"),
            ("c2.json", "{\"cost_bps\": true}"),
            ("c3.json", "[1, 2]"),
            ("c4.json", "{\"a\": \"x\"}"),
        ] {
            let p = temp(name, text);
            assert!(matches!(load_config(&p), Err(StatarbError::Data(_))), "{text}");
        }
        let p = temp("ok.json", "{\"a\": 1.5, \"b\": -2e3}");
        let cfg = load_config(&p).unwrap();
        assert_eq!(cfg["a"], 1.5);
        assert_eq!(cfg["b"], -2000.0);
    }
}
