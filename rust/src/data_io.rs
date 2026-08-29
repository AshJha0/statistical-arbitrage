//! Tiny CSV / JSON loaders so the crate needs nothing beyond serde_json.

use std::collections::HashMap;
use std::path::Path;

use crate::error::{Result, StatarbError};

/// Load a simple headered numeric CSV into `{column_name: values}`.
///
/// # Errors
///
/// [`StatarbError::Data`] on a missing/empty file or ragged/non-numeric
/// rows.
pub fn load_csv(path: &Path) -> Result<HashMap<String, Vec<f64>>> {
    let text = std::fs::read_to_string(path)
        .map_err(|_| StatarbError::Data(format!("no such data file: {}", path.display())))?;
    let lines: Vec<&str> = text.trim().lines().collect();
    if lines.len() < 2 {
        return Err(StatarbError::Data(format!(
            "empty data file: {}",
            path.display()
        )));
    }
    let header: Vec<String> = lines[0].split(',').map(|h| h.trim().to_string()).collect();
    let ncol = header.len();
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
            cols[j].push(v);
        }
    }
    Ok(header.into_iter().zip(cols).collect())
}

/// Load the bundled `config.json` (flat JSON object of numeric scalars).
///
/// # Errors
///
/// [`StatarbError::Data`] on a missing file, invalid JSON, a non-object
/// document, or a non-numeric value.
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
        let num = v.as_f64().ok_or_else(|| {
            StatarbError::Data(format!("config key '{k}' is not a numeric scalar"))
        })?;
        out.insert(k.clone(), num);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
