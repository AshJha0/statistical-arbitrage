/// \file data_io.hpp
/// \brief Tiny CSV / flat-JSON loaders so the library stays dependency-free.
///
/// Validation (pinned across languages): the file must exist and hold a
/// header plus at least one data row; every row must have exactly as many
/// cells as the header; every cell must parse as a FINITE number
/// (nan/inf are rejected, not propagated); column names must be unique;
/// and if a column named `day` exists it must be strictly increasing —
/// an unsorted, duplicated or re-merged time index silently changes every
/// lag, z-score and fill in the backtest.  The models further assume the
/// rows are EQUALLY SPACED consecutive observations (next-close
/// execution, lag-1 differences, sqrt(252)); gaps are the caller's
/// responsibility (API_SPEC section 10).
#pragma once

#include <map>
#include <string>
#include <vector>

namespace statarb {

/// Name of the optional time-index column validated on load.
inline const char* const kDayColumn = "day";

/// Load a simple headered numeric CSV into {column name -> double column}.
/// \throws std::invalid_argument on a missing/empty file, ragged,
///         non-numeric or non-finite rows, duplicate column names, or a
///         `day` column that is not strictly increasing.
std::map<std::string, std::vector<double>> load_csv(const std::string& path);

/// Load the bundled config.json — a FLAT JSON object of finite numeric
/// scalars (that is all the bundled configuration uses; nested values,
/// strings, booleans and NaN/Infinity are rejected).
/// \throws std::invalid_argument on a missing file or malformed content.
std::map<std::string, double> load_config(const std::string& path);

}  // namespace statarb
