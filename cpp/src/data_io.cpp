#include "statarb/data_io.hpp"

#include <charconv>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace statarb {

namespace {

bool is_digit(char c) { return c >= '0' && c <= '9'; }

/// Trim ASCII whitespace (header cells, lines).
std::string trim(const std::string& s) {
    std::size_t b = 0, e = s.size();
    auto ws = [](char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f'; };
    while (b < e && ws(s[b])) ++b;
    while (e > b && ws(s[e - 1])) --e;
    return s.substr(b, e - b);
}

/// Trim ASCII spaces and tabs only (data cells).
std::string trim_cell(const std::string& s) {
    std::size_t b = 0, e = s.size();
    while (b < e && (s[b] == ' ' || s[b] == '\t')) ++b;
    while (e > b && (s[e - 1] == ' ' || s[e - 1] == '\t')) --e;
    return s.substr(b, e - b);
}

std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> parts;
    std::string cell;
    for (char c : line) {
        if (c == ',') {
            parts.push_back(cell);
            cell.clear();
        } else {
            cell.push_back(c);
        }
    }
    parts.push_back(cell);
    return parts;
}

/// Plain decimal grammar  [+-]?([0-9]+\.?[0-9]*|\.[0-9]+)([eE][+-]?[0-9]+)?
/// (full match).
bool matches_number(const std::string& t) {
    std::size_t i = 0;
    const std::size_t n = t.size();
    if (i < n && (t[i] == '+' || t[i] == '-')) ++i;
    std::size_t int_digits = 0, frac_digits = 0;
    while (i < n && is_digit(t[i])) { ++i; ++int_digits; }
    if (i < n && t[i] == '.') {
        ++i;
        while (i < n && is_digit(t[i])) { ++i; ++frac_digits; }
    }
    if (int_digits == 0 && frac_digits == 0) return false;
    if (i < n && (t[i] == 'e' || t[i] == 'E')) {
        ++i;
        if (i < n && (t[i] == '+' || t[i] == '-')) ++i;
        std::size_t exp_digits = 0;
        while (i < n && is_digit(t[i])) { ++i; ++exp_digits; }
        if (exp_digits == 0) return false;
    }
    return i == n;
}

/// Plain integer grammar [+-]?[0-9]+ (full match).
bool matches_integer(const std::string& t) {
    std::size_t i = 0;
    if (i < t.size() && (t[i] == '+' || t[i] == '-')) ++i;
    if (i == t.size()) return false;
    for (; i < t.size(); ++i) {
        if (!is_digit(t[i])) return false;
    }
    return true;
}

/// Locale-independent parse of text that already matched matches_number.
/// Returns false on overflow (a value that does not fit a double).
/// Underflow yields 0 like Python's float().
bool parse_number(const std::string& text, double& out) {
    const char* first = text.data();
    const char* last = first + text.size();
    bool neg = false;
    if (first != last && *first == '+') {
        ++first;
    } else if (first != last && *first == '-') {
        neg = true;
        ++first;
    }
    double v = 0.0;
    const std::from_chars_result r = std::from_chars(first, last, v);
    if (r.ec == std::errc() && r.ptr == last) {
        out = neg ? -v : v;
        return true;
    }
    if (r.ec == std::errc::result_out_of_range) {
        // Decide overflow vs underflow from the decimal order of magnitude.
        long long int_digits = 0;  // significant digits before the point
        long long lead_zero_frac = 0;
        const char* p = first;
        bool seen_nonzero = false;
        while (p != last && is_digit(*p)) {
            if (*p != '0') seen_nonzero = true;
            if (seen_nonzero) ++int_digits;
            ++p;
        }
        if (p != last && *p == '.') {
            ++p;
            while (p != last && is_digit(*p)) {
                if (!seen_nonzero) {
                    if (*p == '0') ++lead_zero_frac;
                    else seen_nonzero = true;
                }
                ++p;
            }
        }
        long long expo = 0;
        if (p != last && (*p == 'e' || *p == 'E')) {
            ++p;
            bool eneg = false;
            if (*p == '+') ++p;
            else if (*p == '-') { eneg = true; ++p; }
            for (; p != last; ++p) {
                if (expo < 100000000LL) expo = expo * 10 + (*p - '0');
            }
            if (eneg) expo = -expo;
        }
        const long long magnitude = (int_digits > 0 ? int_digits : -lead_zero_frac) + expo;
        if (magnitude > 0) return false;  // overflow
        out = neg ? -0.0 : 0.0;           // underflow
        return true;
    }
    return false;
}

[[noreturn]] void bad(const std::string& what) { throw std::invalid_argument(what); }

}  // namespace

std::map<std::string, std::vector<double>> load_csv(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        bad("no such data file: " + path);
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
        bad("empty data file: " + path);
    }
    std::vector<std::string> header = split_csv_line(lines[0]);
    for (std::string& h : header) h = trim(h);
    const std::size_t ncol = header.size();
    for (std::size_t a = 0; a < ncol; ++a) {
        for (std::size_t b = a + 1; b < ncol; ++b) {
            if (header[a] == header[b]) {
                bad("duplicate column names in " + path);
            }
        }
    }

    std::map<std::string, std::vector<double>> out;
    std::vector<std::vector<double>> cols(ncol);
    for (auto& c : cols) c.reserve(lines.size() - 1);
    for (std::size_t i = 1; i < lines.size(); ++i) {
        const std::vector<std::string> parts = split_csv_line(lines[i]);
        if (parts.size() != ncol) {
            bad("ragged row " + std::to_string(i + 1) + " in " + path);
        }
        for (std::size_t j = 0; j < ncol; ++j) {
            const std::string cell = trim_cell(parts[j]);
            const bool is_day = header[j] == kDayColumn;
            if (!(is_day ? matches_integer(cell) : matches_number(cell))) {
                bad("non-numeric value at row " + std::to_string(i + 1) + " in " + path);
            }
            double v = 0.0;
            if (!parse_number(cell, v) || !std::isfinite(v)) {
                bad("non-finite value at row " + std::to_string(i + 1) + " in " + path);
            }
            cols[j].push_back(v);
        }
    }
    for (std::size_t j = 0; j < ncol; ++j) {
        if (header[j] == kDayColumn) {
            const std::vector<double>& day = cols[j];
            for (std::size_t i = 1; i < day.size(); ++i) {
                if (!(day[i] > day[i - 1])) {
                    bad("day column must be strictly increasing (sorted, no duplicates) in " +
                        path);
                }
            }
        }
    }
    for (std::size_t j = 0; j < ncol; ++j) out[header[j]] = std::move(cols[j]);
    return out;
}

namespace {

/// Keys that must be written as JSON integers.
bool is_integer_key(const std::string& k) {
    return k == "window" || k == "adf_maxlag" || k == "max_hold_days" ||
           k == "cooldown_days";
}

void append_utf8(std::string& s, unsigned cp) {
    if (cp < 0x80) {
        s.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        s.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        s.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        s.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        s.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

}  // namespace

std::map<std::string, double> load_config(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        bad("no such config file: " + path);
    }
    std::stringstream buf;
    buf << in.rdbuf();
    const std::string text = buf.str();

    // Strict parser for ONE flat JSON object {"key": number, ...}.  RFC 8259
    // whitespace and number grammar; nothing may follow the closing brace.
    std::size_t i = 0;
    const auto skip_ws = [&]() {
        while (i < text.size() &&
               (text[i] == ' ' || text[i] == '\t' || text[i] == '\n' || text[i] == '\r')) {
            ++i;
        }
    };
    const auto fail = [&](const std::string& why) -> std::invalid_argument {
        return std::invalid_argument("malformed config " + path + ": " + why);
    };
    const auto hex4 = [&](std::size_t at) -> unsigned {
        if (at + 4 > text.size()) throw fail("bad \\u escape");
        unsigned v = 0;
        for (std::size_t k = at; k < at + 4; ++k) {
            const char c = text[k];
            v <<= 4;
            if (c >= '0' && c <= '9') v |= static_cast<unsigned>(c - '0');
            else if (c >= 'a' && c <= 'f') v |= static_cast<unsigned>(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v |= static_cast<unsigned>(c - 'A' + 10);
            else throw fail("bad \\u escape");
        }
        return v;
    };
    const auto parse_string = [&]() {
        std::string key;
        ++i;  // opening quote
        while (true) {
            if (i >= text.size()) throw fail("unterminated key");
            const char c = text[i];
            if (c == '"') { ++i; break; }
            if (static_cast<unsigned char>(c) < 0x20) throw fail("control character in key");
            if (c != '\\') { key.push_back(c); ++i; continue; }
            if (++i >= text.size()) throw fail("unterminated key");
            const char e = text[i++];
            switch (e) {
                case '"': key.push_back('"'); break;
                case '\\': key.push_back('\\'); break;
                case '/': key.push_back('/'); break;
                case 'b': key.push_back('\b'); break;
                case 'f': key.push_back('\f'); break;
                case 'n': key.push_back('\n'); break;
                case 'r': key.push_back('\r'); break;
                case 't': key.push_back('\t'); break;
                case 'u': {
                    unsigned cp = hex4(i);
                    i += 4;
                    if (cp >= 0xD800 && cp < 0xDC00 && i + 1 < text.size() &&
                        text[i] == '\\' && text[i + 1] == 'u') {
                        const unsigned lo = hex4(i + 2);
                        if (lo >= 0xDC00 && lo < 0xE000) {
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                            i += 6;
                        }
                    }
                    append_utf8(key, cp);
                    break;
                }
                default: throw fail("bad escape in key");
            }
        }
        return key;
    };

    skip_ws();
    if (i >= text.size() || text[i] != '{') throw fail("expected '{'");
    ++i;
    std::map<std::string, double> cfg;
    skip_ws();
    if (i < text.size() && text[i] == '}') {
        ++i;
    } else {
        while (true) {
            skip_ws();
            if (i >= text.size() || text[i] != '"') throw fail("expected key string");
            const std::string key = parse_string();
            skip_ws();
            if (i >= text.size() || text[i] != ':') throw fail("expected ':'");
            ++i;
            skip_ws();
            // RFC 8259 number: -?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?
            const std::size_t vb = i;
            if (i < text.size() && text[i] == '-') ++i;
            if (i >= text.size() || !is_digit(text[i])) {
                throw fail("value for key '" + key + "' is not a finite numeric scalar");
            }
            if (text[i] == '0') {
                ++i;
            } else {
                while (i < text.size() && is_digit(text[i])) ++i;
            }
            bool integral = true;
            if (i < text.size() && text[i] == '.') {
                integral = false;
                ++i;
                std::size_t d = 0;
                while (i < text.size() && is_digit(text[i])) { ++i; ++d; }
                if (d == 0) throw fail("bad number for key '" + key + "'");
            }
            if (i < text.size() && (text[i] == 'e' || text[i] == 'E')) {
                integral = false;
                ++i;
                if (i < text.size() && (text[i] == '+' || text[i] == '-')) ++i;
                std::size_t d = 0;
                while (i < text.size() && is_digit(text[i])) { ++i; ++d; }
                if (d == 0) throw fail("bad number for key '" + key + "'");
            }
            const std::string tok = text.substr(vb, i - vb);
            double v = 0.0;
            if (!parse_number(tok, v) || !std::isfinite(v)) {
                throw fail("value for key '" + key + "' is not a finite numeric scalar");
            }
            if (integral) {
                // Exact test on the digit string: magnitude above 2^53 is rejected.
                const std::string digits = tok.substr(tok[0] == '-' ? 1 : 0);
                if (digits.size() > 16 ||
                    (digits.size() == 16 && digits > "9007199254740992")) {
                    throw fail("config key '" + key + "' is out of range");
                }
            }
            if (is_integer_key(key) && !integral) {
                throw fail("config key '" + key + "' must be an integer");
            }
            cfg[key] = v;
            skip_ws();
            if (i >= text.size()) throw fail("unterminated object");
            if (text[i] == '}') { ++i; break; }
            if (text[i] != ',') throw fail("expected ',' or '}'");
            ++i;
        }
    }
    skip_ws();
    if (i != text.size()) throw fail("trailing content after the object");
    return cfg;
}

}  // namespace statarb
