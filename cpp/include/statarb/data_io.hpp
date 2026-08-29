/// \file data_io.hpp
/// \brief Tiny CSV / flat-JSON loaders so the library stays dependency-free.
#pragma once

#include <map>
#include <string>
#include <vector>

namespace statarb {

/// Load a simple headered numeric CSV into {column name -> double column}.
/// \throws std::invalid_argument on a missing/empty file or
///         ragged/non-numeric rows.
std::map<std::string, std::vector<double>> load_csv(const std::string& path);

/// Load the bundled config.json — a FLAT JSON object of numeric scalars
/// (that is all the bundled configuration uses; nested values are
/// rejected).
/// \throws std::invalid_argument on a missing file or malformed content.
std::map<std::string, double> load_config(const std::string& path);

}  // namespace statarb
