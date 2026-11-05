#include "statarb/kalman.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace statarb {

namespace {
void check_params(double delta, double r) {
    if (!(delta > 0.0 && delta < 1.0)) {
        throw std::invalid_argument("delta must lie strictly inside (0, 1)");
    }
    // Negated comparisons so NaN fails; +inf is rejected explicitly.
    if (!(r > 0.0) || !std::isfinite(r)) {
        throw std::invalid_argument("observation variance r must be > 0 and finite");
    }
}
}  // namespace

KalmanStep::KalmanStep(double delta, double r) : q_(0.0), r_(r) {
    check_params(delta, r);
    q_ = delta / (1.0 - delta);  // RiskMetrics-style Q scale
}

Matrix KalmanStep::covariance() const {
    Matrix m(2, 2);
    m(0, 0) = p00_;
    m(0, 1) = p01_;
    m(1, 0) = p01_;
    m(1, 1) = p11_;
    return m;
}

KalmanOut KalmanStep::update(double x, double y) {
    if (!(std::isfinite(x) && std::isfinite(y))) {
        throw std::invalid_argument("inputs must be finite (no NaN/inf)");
    }
    const double r = r_;
    // Scalar order pinned (API_SPEC section 6).
    const double p00 = p00_ + q_;
    const double p01 = p01_;
    const double p11 = p11_ + q_;
    const double e = y - (a_ + x * b_);
    const double ph0 = p00 + p01 * x;  // P_pred H'
    const double ph1 = p01 + p11 * x;
    const double s = ph0 + x * ph1 + r;
    const double k0 = ph0 / s;
    const double k1 = ph1 / s;
    a_ += k0 * e;
    b_ += k1 * e;
    // Joseph form with A = I - K H
    const double a00 = 1.0 - k0;
    const double a01 = -k0 * x;
    const double a10 = -k1;
    const double a11 = 1.0 - k1 * x;
    const double b00 = a00 * p00 + a01 * p01;  // B = A P_pred
    const double b01 = a00 * p01 + a01 * p11;
    const double b10 = a10 * p00 + a11 * p01;
    const double b11 = a10 * p01 + a11 * p11;
    const double n00 = b00 * a00 + b01 * a01 + r * (k0 * k0);
    const double n01 = b00 * a10 + b01 * a11 + r * (k0 * k1);
    const double n10 = b10 * a00 + b11 * a01 + r * (k1 * k0);
    const double n11 = b10 * a10 + b11 * a11 + r * (k1 * k1);
    p00_ = n00;
    p01_ = 0.5 * (n01 + n10);
    p11_ = n11;
    innovation_ = e;
    s_var_ = s;
    return KalmanOut{a_, b_, e / std::sqrt(s)};
}

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
    check_params(delta, r);

    const std::size_t t_len = y.size();
    KalmanStep step(delta, r);
    KalmanResult res;
    res.alpha.resize(t_len);
    res.beta.resize(t_len);
    res.z.resize(t_len);
    res.innovation.resize(t_len);
    res.s_var.resize(t_len);
    // One recursion for batch and streaming, so the two cannot diverge.
    for (std::size_t t = 0; t < t_len; ++t) {
        const KalmanOut o = step.update(x[t], y[t]);
        res.alpha[t] = o.alpha;
        res.beta[t] = o.beta;
        res.z[t] = o.z;
        res.innovation[t] = step.innovation();
        res.s_var[t] = step.s_var();
    }
    res.final_cov = step.covariance();
    return res;
}

}  // namespace statarb
