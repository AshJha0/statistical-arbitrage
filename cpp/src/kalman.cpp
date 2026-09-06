#include "statarb/kalman.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace statarb {

KalmanResult kalman_hedge(const std::vector<double>& y,
                          const std::vector<double>& x,
                          double delta, double r) {
    if (y.size() != x.size()) {
        throw std::invalid_argument("series length mismatch: len(y)=" +
                                    std::to_string(y.size()) + ", len(x)=" +
                                    std::to_string(x.size()));
    }
    if (y.size() < 2) {
        throw std::invalid_argument("need at least 2 observations");
    }
    if (!all_finite(y) || !all_finite(x)) {
        throw std::invalid_argument("inputs must be finite (no NaN/inf)");
    }
    if (!(delta > 0.0 && delta < 1.0)) {
        throw std::invalid_argument("delta must lie strictly inside (0, 1)");
    }
    // Negated comparisons so NaN fails; +inf is rejected explicitly.
    if (!(r > 0.0) || !std::isfinite(r)) {
        throw std::invalid_argument("observation variance r must be > 0 and finite");
    }

    const std::size_t t_len = y.size();
    const double q = delta / (1.0 - delta);  // RiskMetrics-style Q scale
    double s0 = 0.0, s1 = 0.0;               // state (alpha, beta)
    // Posterior covariance P, kept symmetric; pinned prior 1e3 * I.
    double p00 = kP0Scale, p01 = 0.0, p11 = kP0Scale;

    KalmanResult res;
    res.alpha.resize(t_len);
    res.beta.resize(t_len);
    res.z.resize(t_len);
    res.innovation.resize(t_len);
    res.s_var.resize(t_len);

    for (std::size_t t = 0; t < t_len; ++t) {
        // Predict: state unchanged, P_pred = P + Q.
        const double pp00 = p00 + q;
        const double pp01 = p01;
        const double pp11 = p11 + q;
        const double xt = x[t];
        // Innovation e = y - H s with H = [1, x_t].
        const double e = y[t] - (s0 + s1 * xt);
        // Innovation variance S = H P_pred H' + r.
        const double ph0 = pp00 + pp01 * xt;  // (P_pred H')_0
        const double ph1 = pp01 + pp11 * xt;  // (P_pred H')_1
        const double s_var = ph0 + ph1 * xt + r;
        // Gain K = P_pred H' / S.
        const double k0 = ph0 / s_var;
        const double k1 = ph1 / s_var;
        s0 += k0 * e;
        s1 += k1 * e;
        // Joseph form: P = (I - K H) P_pred (I - K H)' + K r K'.
        const double i00 = 1.0 - k0;        // (I - K H) entries
        const double i01 = -k0 * xt;
        const double i10 = -k1;
        const double i11 = 1.0 - k1 * xt;
        // M = (I - K H) P_pred
        const double m00 = i00 * pp00 + i01 * pp01;
        const double m01 = i00 * pp01 + i01 * pp11;
        const double m10 = i10 * pp00 + i11 * pp01;
        const double m11 = i10 * pp01 + i11 * pp11;
        // P = M (I - K H)' + r K K'
        double n00 = m00 * i00 + m01 * i01 + r * k0 * k0;
        double n01 = m00 * i10 + m01 * i11 + r * k0 * k1;
        double n10 = m10 * i00 + m11 * i01 + r * k1 * k0;
        double n11 = m10 * i10 + m11 * i11 + r * k1 * k1;
        // Symmetrise against rounding.
        p00 = n00;
        p01 = 0.5 * (n01 + n10);
        p11 = n11;

        res.alpha[t] = s0;
        res.beta[t] = s1;
        res.innovation[t] = e;
        res.s_var[t] = s_var;
        res.z[t] = e / std::sqrt(s_var);
    }

    res.final_cov = Matrix(2, 2);
    res.final_cov(0, 0) = p00;
    res.final_cov(0, 1) = p01;
    res.final_cov(1, 0) = p01;
    res.final_cov(1, 1) = p11;
    return res;
}

}  // namespace statarb
