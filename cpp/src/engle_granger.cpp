#include "statarb/engle_granger.hpp"

#include "statarb/ols.hpp"

namespace statarb {

EGResult engle_granger(const std::vector<double>& y,
                       const std::vector<double>& x, int maxlag) {
    const OLSResult fit = ols(y, x, /*intercept=*/true);
    EGResult res;
    res.alpha = fit.params[0];
    res.beta = fit.params[1];
    res.resid = fit.resid;
    res.r2 = fit.r2;
    // Residuals are estimated -> judge against the 2-variable EG table.
    res.adf = adf_test(fit.resid, maxlag, /*nvars=*/2);
    return res;
}

}  // namespace statarb
