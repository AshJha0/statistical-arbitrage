#include "statarb/signals.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include "statarb/linalg.hpp"
#include "statarb/ols.hpp"

namespace statarb {

std::vector<double> spread(const std::vector<double>& y,
                           const std::vector<double>& x,
                           double alpha, double beta) {
    if (y.size() != x.size()) {
        throw std::invalid_argument("y and x must be 1-D arrays of equal length");
    }
    if (!all_finite(y) || !all_finite(x)) {
        throw std::invalid_argument("inputs must be finite (no NaN/inf)");
    }
    if (!std::isfinite(alpha) || !std::isfinite(beta)) {
        throw std::invalid_argument("alpha and beta must be finite");
    }
    std::vector<double> s(y.size());
    for (std::size_t t = 0; t < y.size(); ++t) s[t] = y[t] - (alpha + beta * x[t]);
    return s;
}

std::vector<double> rolling_zscore(const std::vector<double>& s, int window) {
    if (window < 2) {
        throw std::invalid_argument("window must be an integer >= 2");
    }
    const std::size_t t_len = s.size();
    if (static_cast<std::size_t>(window) > t_len) {
        throw std::invalid_argument("window (" + std::to_string(window) +
                                    ") larger than series length (" +
                                    std::to_string(t_len) + ")");
    }
    if (!all_finite(s)) {
        throw std::invalid_argument("inputs must be finite (no NaN/inf)");
    }
    const std::size_t w = static_cast<std::size_t>(window);
    std::vector<double> z(t_len, std::numeric_limits<double>::quiet_NaN());
    for (std::size_t t = w - 1; t < t_len; ++t) {
        NeumaierSum sum;
        for (std::size_t i = t - w + 1; i <= t; ++i) sum.add(s[i]);
        const double mean = sum.value() / static_cast<double>(w);
        NeumaierSum acc;
        for (std::size_t i = t - w + 1; i <= t; ++i) {
            const double d = s[i] - mean;
            acc.add(d * d);
        }
        const double sd = std::sqrt(acc.value() / static_cast<double>(w - 1));
        if (sd > kZscoreStdFloor) z[t] = (s[t] - mean) / sd;
    }
    return z;
}

double half_life(const std::vector<double>& s) {
    if (s.size() < 12) {
        throw std::invalid_argument("need at least 12 observations for half-life");
    }
    if (!all_finite(s)) {
        throw std::invalid_argument("inputs must be finite (no NaN/inf)");
    }
    const auto [mn, mx] = std::minmax_element(s.begin(), s.end());
    if (*mx - *mn == 0.0) {
        throw std::invalid_argument("series is constant: half-life undefined");
    }
    // AR(1) regression (pinned): ds_t = c + rho * s_{t-1} + e_t.
    std::vector<double> ds(s.size() - 1), lag(s.size() - 1);
    for (std::size_t t = 1; t < s.size(); ++t) {
        ds[t - 1] = s[t] - s[t - 1];
        lag[t - 1] = s[t - 1];
    }
    const OLSResult fit = ols(ds, lag, /*intercept=*/true);
    const double rho = fit.params[1];
    if (rho >= 0.0) return std::numeric_limits<double>::infinity();
    if (1.0 + rho <= 0.0) return 0.0;
    return -std::log(2.0) / std::log(1.0 + rho);
}

}  // namespace statarb
