#include "statarb/data_io.hpp"

#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace statarb {

namespace {

/// Trim ASCII whitespace from both ends.
std::string trim(const std::string& s) {
    std::size_t b = 0, e = s.size();
    while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
    return s.substr(b, e - b);
}

std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> parts;
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ',')) parts.push_back(cell);
    if (!line.empty() && line.back() == ',') parts.emplace_back();
    return parts;
}

double parse_double(const std::string& text, bool& ok) {
    const std::string t = trim(text);
    if (t.empty()) {
        ok = false;
        return 0.0;
    }
    char* end = nullptr;
    const double v = std::strtod(t.c_str(), &end);
    ok = end == t.c_str() + t.size();
    return v;
}

}  // namespace

std::map<std::string, std::vector<double>> load_csv(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::invalid_argument("no such data file: " + path);
    }
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        lines.push_back(line);
    }
    // Drop leading/trailing blank lines only (mirrors Python's strip());
    // an interior blank line is a ragged row, as in the other ports.
    while (!lines.empty() && trim(lines.back()).empty()) lines.pop_back();
    while (!lines.empty() && trim(lines.front()).empty()) lines.erase(lines.begin());
    if (lines.size() < 2) {
        throw std::invalid_argument("empty data file: " + path);
    }
    std::vector<std::string> header = split_csv_line(lines[0]);
    for (std::string& h : header) h = trim(h);
    const std::size_t ncol = header.size();
    for (std::size_t a = 0; a < ncol; ++a) {
        for (std::size_t b = a + 1; b < ncol; ++b) {
            if (header[a] == header[b]) {
                throw std::invalid_argument("duplicate column names in " + path);
            }
        }
    }

    std::map<std::string, std::vector<double>> out;
    std::vector<std::vector<double>> cols(ncol);
    for (auto& c : cols) c.reserve(lines.size() - 1);
    for (std::size_t i = 1; i < lines.size(); ++i) {
        const std::vector<std::string> parts = split_csv_line(lines[i]);
        if (parts.size() != ncol) {
            throw std::invalid_argument("ragged row " + std::to_string(i + 1) +
                                        " in " + path);
        }
        for (std::size_t j = 0; j < ncol; ++j) {
            bool ok = false;
            const double v = parse_double(parts[j], ok);
            if (!ok) {
                throw std::invalid_argument("non-numeric value at row " +
                                            std::to_string(i + 1) + " in " + path);
            }
            if (!std::isfinite(v)) {
                throw std::invalid_argument("non-finite value at row " +
                                            std::to_string(i + 1) + " in " + path);
            }
            cols[j].push_back(v);
        }
    }
    for (std::size_t j = 0; j < ncol; ++j) {
        if (header[j] == kDayColumn) {
            const std::vector<double>& day = cols[j];
            for (std::size_t i = 1; i < day.size(); ++i) {
                if (!(day[i] > day[i - 1])) {
                    throw std::invalid_argument(
                        "day column must be strictly increasing (sorted, no duplicates) in " +
                        path);
                }
            }
        }
    }
    for (std::size_t j = 0; j < ncol; ++j) out[header[j]] = std::move(cols[j]);
    return out;
}

std::map<std::string, double> load_config(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::invalid_argument("no such config file: " + path);
    }
    std::stringstream buf;
    buf << in.rdbuf();
    const std::string text = buf.str();

    // Minimal parser for a FLAT JSON object {"key": number, ...} — all
    // the bundled config.json uses.  Anything else is rejected.
    std::size_t i = 0;
    auto skip_ws = [&]() {
        while (i < text.size() && std::isspace(static_cast<unsigned char>(text[i]))) ++i;
    };
    auto fail = [&](const std::string& why) -> std::invalid_argument {
        return std::invalid_argument("malformed config " + path + ": " + why);
    };
    skip_ws();
    if (i >= text.size() || text[i] != '{') throw fail("expected '{'");
    ++i;
    std::map<std::string, double> cfg;
    skip_ws();
    if (i < text.size() && text[i] == '}') return cfg;  // empty object
    while (true) {
        skip_ws();
        if (i >= text.size() || text[i] != '"') throw fail("expected key string");
        const std::size_t kb = ++i;
        while (i < text.size() && text[i] != '"') ++i;
        if (i >= text.size()) throw fail("unterminated key");
        const std::string key = text.substr(kb, i - kb);
        ++i;
        skip_ws();
        if (i >= text.size() || text[i] != ':') throw fail("expected ':'");
        ++i;
        skip_ws();
        const std::size_t vb = i;
        while (i < text.size() && text[i] != ',' && text[i] != '}') ++i;
        bool ok = false;
        const double v = parse_double(text.substr(vb, i - vb), ok);
        if (!ok || !std::isfinite(v)) {
            throw fail("value for key '" + key + "' is not a finite numeric scalar");
        }
        cfg[key] = v;
        skip_ws();
        if (i >= text.size()) throw fail("unterminated object");
        if (text[i] == '}') break;
        if (text[i] != ',') throw fail("expected ',' or '}'");
        ++i;
    }
    return cfg;
}

}  // namespace statarb
