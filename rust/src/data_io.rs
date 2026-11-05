//! Tiny CSV / JSON loaders (the crate's only dependency is `serde_json`
//! for the flat config file).
//!
//! Validation (pinned across languages): the file must exist and hold a
//! header plus at least one data row; every row must have exactly as many
//! cells as the header; every cell, after trimming ASCII spaces and tabs,
//! must match the plain decimal/exponent grammar and be FINITE (`nan`,
//! `inf`, hex, `_` separators, type suffixes and overflow are rejected
//! whatever Rust's own parser would accept); column names must be unique;
//! and if a column named `day` exists its cells must be plain integers and
//! it must be strictly increasing — an unsorted, duplicated or re-merged time index silently
//! changes every lag, z-score and fill in the backtest. The models further
//! assume the rows are EQUALLY SPACED consecutive observations (next-close
//! execution, lag-1 differences, `sqrt(252)`); gaps are the caller's
//! responsibility (API_SPEC section 10).

use std::collections::HashMap;
use std::path::Path;

use crate::error::{Result, StatarbError};

/// Name of the optional time-index column validated on load.
pub const DAY_COLUMN: &str = "day";

/// Config keys that must be written as JSON integers (`60`, not `60.0`).
pub const INTEGER_KEYS: [&str; 4] = ["window", "adf_maxlag", "max_hold_days", "cooldown_days"];

/// Largest integer magnitude accepted in any config key (2^53).
const MAX_SAFE_INT: i128 = 1 << 53;

/// `[+-]?([0-9]+\.?[0-9]*|\.[0-9]+)([eE][+-]?[0-9]+)?`, checked by hand.
fn matches_number_grammar(cell: &str) -> bool {
    let b = cell.as_bytes();
    let mut i = 0;
    if i < b.len() && (b[i] == b'+' || b[i] == b'-') {
        i += 1;
    }
    let int_start = i;
    while i < b.len() && b[i].is_ascii_digit() {
        i += 1;
    }
    let int_digits = i - int_start;
    let mut frac_digits = 0;
    if i < b.len() && b[i] == b'.' {
        i += 1;
        let f0 = i;
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
        }
        frac_digits = i - f0;
    }
    if int_digits == 0 && frac_digits == 0 {
        return false;
    }
    if i < b.len() && (b[i] == b'e' || b[i] == b'E') {
        i += 1;
        if i < b.len() && (b[i] == b'+' || b[i] == b'-') {
            i += 1;
        }
        let e0 = i;
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
        }
        if i == e0 {
            return false;
        }
    }
    i == b.len()
}

/// `[+-]?[0-9]+`.
fn matches_integer_grammar(cell: &str) -> bool {
    let digits = cell.strip_prefix(['+', '-']).unwrap_or(cell);
    !digits.is_empty() && digits.bytes().all(|c| c.is_ascii_digit())
}

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
    let lines: Vec<&str> = text
        .trim()
        .lines()
        .map(|l| l.trim_end_matches('\r'))
        .collect();
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
            let cell = part.trim_matches([' ', '\t']);
            let is_day = header[j] == DAY_COLUMN;
            let grammar_ok = if is_day {
                matches_integer_grammar(cell)
            } else {
                matches_number_grammar(cell)
            };
            let v: f64 = if grammar_ok { cell.parse().ok() } else { None }.ok_or_else(|| {
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
        if cols[j]
            .windows(2)
            .any(|w| w[1].partial_cmp(&w[0]) != Some(std::cmp::Ordering::Greater))
        {
            return Err(StatarbError::Data(format!(
                "day column must be strictly increasing (sorted, no duplicates) in {}",
                path.display()
            )));
        }
    }
    Ok(header.into_iter().zip(cols).collect())
}

/// Load the bundled `config.json` (one flat JSON object of finite numeric
/// scalars, nothing after it).
///
/// The keys in [`INTEGER_KEYS`] must be written as integers (`60` is
/// accepted, `60.0` and `6e1` are not); an integer of magnitude above 2^53
/// is rejected in any key.
///
/// # Errors
///
/// [`StatarbError::Data`] on a missing file, invalid JSON (including
/// trailing content), a non-object document, a value that is not a finite
/// number (booleans, strings, nested values and NaN/Infinity are
/// rejected), an integer key holding a non-integer, or an out-of-range
/// integer.
pub fn load_config(path: &Path) -> Result<HashMap<String, f64>> {
    let text = std::fs::read_to_string(path)
        .map_err(|_| StatarbError::Data(format!("no such config file: {}", path.display())))?;
    // `arbitrary_precision` keeps the number text, so integers beyond i64
    // are still seen as integers (as in the Python reference).
    let value: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| StatarbError::Data(format!("invalid JSON in {}: {e}", path.display())))?;
    let obj = value
        .as_object()
        .ok_or_else(|| StatarbError::Data("config.json must contain a JSON object".to_string()))?;
    let mut out = HashMap::with_capacity(obj.len());
    for (k, v) in obj {
        let bad = || StatarbError::Data(format!("config key '{k}' is not a finite numeric scalar"));
        let serde_json::Value::Number(n) = v else {
            return Err(bad());
        };
        let text = n.to_string();
        let num = text
            .parse::<f64>()
            .ok()
            .filter(|x| x.is_finite())
            .ok_or_else(bad)?;
        let is_int = matches_integer_grammar(&text);
        if is_int {
            let mag = text
                .trim_start_matches(['+', '-'])
                .trim_start_matches('0')
                .parse::<i128>()
                .unwrap_or(i128::MAX);
            if mag > MAX_SAFE_INT {
                return Err(StatarbError::Data(format!(
                    "config key '{k}' is out of range"
                )));
            }
        } else if INTEGER_KEYS.contains(&k.as_str()) {
            return Err(StatarbError::Data(format!(
                "config key '{k}' must be an integer"
            )));
        }
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
        expect_data_error(
            "dup.csv",
            "day,a\n0,1\n1,2\n2,3\n2,4\n3,5\n",
            "day column must be strictly increasing",
        );
        expect_data_error(
            "unsorted.csv",
            "day,a\n0,1\n2,2\n1,3\n3,4\n",
            "day column must be strictly increasing",
        );
        // gaps are allowed; a file without `day` is not checked
        let p = temp("gap.csv", "day,a\n0,1\n1,2\n3,3\n4,4\n");
        assert_eq!(load_csv(&p).unwrap()["day"], vec![0.0, 1.0, 3.0, 4.0]);
        let p = temp("nod.csv", "b,a\n5,1\n2,2\n");
        assert_eq!(load_csv(&p).unwrap()["b"], vec![5.0, 2.0]);
    }

    #[test]
    fn csv_rejects_nonfinite_ragged_nonnumeric() {
        expect_data_error("nan.csv", "day,a\n0,1\n1,nan\n", "non-numeric");
        expect_data_error("inf.csv", "day,a\n0,1\n1,inf\n", "non-numeric");
        expect_data_error(
            "ovf.csv",
            "day,a
0,1
1,1e999
",
            "non-finite",
        );
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
            assert!(
                matches!(load_config(&p), Err(StatarbError::Data(_))),
                "{text}"
            );
        }
        let p = temp("ok.json", "{\"a\": 1.5, \"b\": -2e3}");
        let cfg = load_config(&p).unwrap();
        assert_eq!(cfg["a"], 1.5);
        assert_eq!(cfg["b"], -2000.0);
    }
}
