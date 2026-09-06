# Statistical Arbitrage (Cointegration, Kalman Filter, Pairs Backtest)

A self-contained statistical-arbitrage toolkit implemented four times — Python
(reference), C++17, Rust, and Java 21 — against one pinned numerical contract
([`API_SPEC.md`](API_SPEC.md)). Every algorithm (OLS, ADF, Engle–Granger,
Johansen-lite, Kalman dynamic hedge, dollar-neutral pairs backtest) is specified
down to the solver so that all four languages agree to tight tolerances,
enforced by a shared golden-value suite.

The package name is `statarb` everywhere: Python package `statarb`, C++
namespace `statarb`, Rust crate `statarb`, Java package `com.quant.statarb`.

## What it does

| Feature | Notes |
|---|---|
| OLS regression | Normal equations + LU solve (pinned), plain SEs, t-stats, R² |
| ADF unit-root test | Fixed lag (default 1), constant/no-trend, MacKinnon (2010) finite-sample critical values `b_inf + b1/T + b2/T² + b3/T³` at the regression sample size (T_eff ≥ 20 enforced), interpolated approximate p-value |
| Half-life of mean reversion | AR(1) fit on the spread, `hl = -ln 2 / ln(1+ρ)` with guards |
| Engle–Granger two-step | Hedge OLS then residual ADF against the 2- or 3-variable EG table |
| Johansen-lite trace test | 2–3 assets, VECM with unrestricted constant, k = 1 lagged difference, native Cholesky + cyclic Jacobi eigensolver, Osterwald–Lenum 95% table |
| Kalman dynamic hedge | Random-walk state (αₜ, βₜ), RiskMetrics-style `Q = δ/(1−δ)·I`, Joseph-form covariance update, innovation z-scores |
| Rolling z-score signal | Current-observation-inclusive window, ddof = 1, zero-variance guard |
| Pairs backtest | FLAT/LONG/SHORT state machine, next-close execution (no look-ahead), equal-dollar legs, bps costs + slippage, stop-losses; a pending entry at the end of data is discarded, never force-closed the same day |
| Walk-forward split | Hedge estimated in-sample (60%), traded out-of-sample (40%) — degradation shown honestly |
| Teaching cases | Regime break at day 1050 (Kalman adapts, static OLS cannot), spurious regression on independent random walks (no trade) |
| Input validation | Every public entry point rejects NaN/inf, length mismatches, empty/constant/collinear inputs and out-of-domain parameters with the language's error type (`ValueError` / `std::invalid_argument` / `Err(StatarbError)` / `IllegalArgumentException`); the CSV loader refuses non-finite cells and a non-monotone `day` column |

Everything is deterministic: no RNG at runtime (randomness lives only in
`data/generate_data.py`, fixed seeds).

## Directory layout

```
statistical-arbitrage/
  README.md               this file
  LEARN.md                theory: unit roots, cointegration, Kalman, pairs practice
  COOKBOOK.md             task-oriented recipes in all four languages
  API_SPEC.md             the pinned cross-language numerical contract
  docs/
    ARCHITECTURE.md       component design, data flow, numerical decisions
    GITHUB_PAGES.md       publishing guide for the docs site: https://ashjha0.github.io/statistical-arbitrage/
    diagrams/*.mmd        raw Mermaid sources
  data/
    pair_equity.csv       1500 days, cointegrated pair (true beta 1.7, break at day 1050)
    fx_triangle.csv       EURUSD vs GBPUSD*EURGBP basket (cointegration in logs)
    random_walks.csv      two independent walks — the spurious-regression case
    config.json           signal/cost parameters shared by all demos
    generate_data.py      regenerates everything (fixed seeds; cross-checks vs statsmodels)
    golden/golden.json    24 golden cases every language must reproduce
  python/                 reference implementation (numpy only in the package)
  cpp/                    C++17 library + GoogleTest suite + demo
  rust/                   Rust library + integration tests + demo binary
  java/                   Java 21 sources + JUnit 4 tests + demo
```

## Build, run, test

The `.sh` helpers are invoked with `bash` below so they work on a fresh
clone regardless of file modes.

### Python (3.11; numpy required, pytest + statsmodels only for the tests)

```bash
cd python
pip install -r requirements.txt         # numpy; pytest + statsmodels for tests
PYTHONPATH=src python3 demo.py          # run the demo
PYTHONPATH=src python3 -m pytest -q     # run the tests (86 pass)
```

### C++ (g++ 13, CMake ≥ 3.16, GoogleTest via `find_package(GTest)`)

```bash
cd cpp
bash build.sh                           # = cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j2
./build/statarb_demo                    # run the demo (from cpp/, reads ../data)
ctest --test-dir build --output-on-failure   # run the tests (63 pass)
```

### Rust (1.95, edition 2021; only dependency `serde_json` for the config loader)

```bash
cd rust
cargo run --bin demo --release          # run the demo
cargo test --release                    # run the tests (74 pass; `cargo build --release` is warning-free)
```

### Java (21; JUnit 4 at /usr/share/java/junit4.jar and Hamcrest at /usr/share/java/hamcrest.jar; no Maven/Gradle)

```bash
cd java
bash build.sh                           # javac -Xlint:all -Werror everything into out/
bash demo.sh                            # run the demo main
bash test.sh                            # JUnitCore on all *Test classes (65 pass)
```

## Demo output (Python reference, truncated)

```
statarb demo — cointegration, Kalman hedging, pairs backtest
pair_equity.csv: 1500 days, in-sample = first 900 days (true beta 1.7, break at day 1050)

=== 1. Cointegration diagnostics (in-sample, days 0..899) =============
  EG step 1  y = 17.3603 + 1.6389 x   (R^2 0.9793)
  EG step 2  ADF(resid) t = -5.8244  crit 5% (2-var) = -3.34  p ~ 0.0001
  verdict    cointegrated at 5%: True
  half-life  8.39 days
  Johansen   trace(r=0) =   37.572 > cv95  15.41
  Johansen   trace(r=1) =    3.460 < cv95   3.76
  Johansen   rank = 1, cointegrating vector (1, -1.6455)  [true beta 1.7]
  FX triangle (3 assets, logs): rank = 1, leading vector (1.00, -1.00, -1.00)  [triangle: (1, -1, -1)]

=== 2. Kalman vs static OLS through the regime break ==================
  day    true beta   kalman beta   static OLS beta
    250      1.700       1.6650         1.5178
    500      1.700       1.6921         1.5178
   1049      1.700       1.7161         1.5178
   1200      1.100       1.1001         1.5178
   1499      1.100       1.1532         1.5178
  beta tracking RMSE (day 300+): kalman 0.0237 vs static OLS 0.2879 -> kalman adapts, OLS cannot

=== 3. Pairs backtest — in-sample vs walk-forward out-of-sample =======
  hedge estimated on days 0..899: alpha 17.3603, beta 1.6389
  in-sample    ret    7.20%  ann    1.97%  sharpe   1.49  maxDD  0.91%  trades  21  win 90.5%  hold   9.8d  turn  41.9x  stops 1
  OOS (40%)    ret  -10.30%  ann   -4.46%  sharpe  -1.37  maxDD 15.44%  trades   9  win 55.6%  hold  24.2d  turn  17.9x  stops 1
  degradation is honest: the OOS window contains the day-1050
  break; stop-losses cap the damage of the broken relationship.

=== 4. Spurious-regression teaching case (independent random walks) ===
  regressing walk_a on walk_b: slope t-stat = 8.4 —
  'significant' by the usual +/-2 rule, yet the walks are independent.
  ADF(walk_a) t = -1.095 (unit root not rejected) and
  ADF(resid) t = -1.154 vs 5% crit -3.34 -> cointegrated: False
  verdict: spurious relationship — NO backtest is run on this pair.

demo complete.
```

## Golden-value cross-language testing

`data/golden/golden.json` holds 24 named cases — OLS coefficients and R² on the
bundled pair, ADF t-stats (residuals and random walk) together with their
MacKinnon (2010) finite-sample 5% critical values and an interpolated p-value,
Engle–Granger verdicts, Johansen trace statistics and rank, half-life, filtered
Kalman betas at pinned days, rolling z-scores at pinned days, and full backtest
metrics (total return, Sharpe, max drawdown, exact trade count of 31). Each
case records its inputs, one expected scalar, and an absolute tolerance (1e-6
to 1e-10; booleans encoded as 0.0/1.0, integers matched exactly). Every
language's golden suite asserts the exact case count.

Every language's test suite loads this same file and asserts agreement, so a
port that deviates from the pinned contract — a different lag convention, a
non-Joseph covariance update, a different z-score windowing — fails its golden
suite immediately. The values are produced by the Python reference, which
`data/generate_data.py` first cross-validates against statsmodels (OLS
coefficients/SEs, ADF t-stats and `mackinnoncrit` critical values,
`coint_johansen` eigenvalues and trace statistics) and an independent numpy
generalised-eigenproblem solve, and refuses to write on any disagreement.

Test counts (all passing, verified on Linux x86-64 with the toolchain above):
Python 86, C++ 63, Rust 74, Java 65. Beyond the golden layer every port
carries the same behavioural suite: accounting identity, exact dollar
neutrality at entry, next-close execution, the end-of-data discard rule on
hand-built spreads with closed-form z-scores, every-truncation
"no zero-day trade" property check on the bundled pair, stop/re-entry
timing, negative-equity metrics, NaN/inf parameter rejection, walk-forward
split exactness and length checks, collinear Johansen inputs, and CSV/config
loader validation.

## Real-world usage notes

What a practitioner gets, precisely:

- **Units and conventions.** Prices are raw levels in whatever unit you pass
  (the bundled equity pair uses price levels; the FX example uses *log*
  rates, because cross-rate identities are multiplicative). `alpha`, `beta`,
  the Kalman `r` and the prior `P₀ = 1e3·I` are all in those units — `r` is a
  price-unit variance, not a fraction, and `P₀ = 1e3` is not diffuse for
  four-digit prices (the first ~50 filtered betas are prior-dominated). Costs
  are basis points of traded notional; `capital` is gross notional and every
  leg is sized to `capital/2` at each entry regardless of accumulated P&L
  (returns are on notional, not compounded on equity; no leverage cap, no
  borrow/financing on the short leg).
- **Time grid.** One row = one observation, assumed equally spaced; the
  annualisation constant 252 and the next-close execution lag are hard-coded.
  The loader enforces a strictly increasing `day` column and finite cells but
  does not know about calendars, holidays, corporate actions or intraday bars —
  align, adjust and de-duplicate upstream. Rolling re-runs are safe: a signal
  on the last available close is never executed and force-closed at the same
  close.
- **Statistics.** ADF/EG critical values are MacKinnon (2010) finite-sample
  response surfaces at the regression sample size; tests with fewer than 20
  effective observations are refused rather than extrapolated. The p-value is a
  piecewise-linear approximation and is never used for a verdict. Johansen is
  the 2–3 asset, k = 1, unrestricted-constant, 95%-trace-only variant with no
  small-sample correction. Half-life sentinels `+inf` and `0.0` both mean "do
  not trade this spread".
- **Stops.** `|z| ≥ stop_z` in position flags a stop regardless of whether the
  overshoot is adverse or favourable, and re-entry in the same direction is
  allowed the very next day (no cool-down); a break regime can therefore
  whipsaw. Treat `stopped` as "model disagreement", not as a P&L stop.
- **What is validated.** Every public entry point checks lengths, finiteness,
  emptiness, constancy/collinearity, parameter domains and window bounds and
  fails with the language's error type; no NaN is ever silently propagated
  into a metric. Config structs are re-validated at run time in C++/Rust.
- **Deliberately out of scope.** Calendars/day-count, `periods_per_year`,
  intraday data, financing and borrow costs, market impact, position
  compounding, rolling ADF/half-life diagnostics helpers, automatic lag
  selection, Johansen deterministic-term variants and p-values, structured
  error codes beyond Rust's enum. These are documented limits, not hidden ones.

This is a verified reference implementation of the pinned algorithms with
cross-language golden agreement and edge-case tests; it is not a trading
system, and nothing here is investment advice.

## References

Works the code implements or the documentation relies on:

- Dickey, D. A. & Fuller, W. A. (1979). "Distribution of the Estimators for
  Autoregressive Time Series with a Unit Root." *Journal of the American
  Statistical Association* 74(366), 427–431. https://doi.org/10.1080/01621459.1979.10482531
- Said, S. E. & Dickey, D. A. (1984). "Testing for Unit Roots in
  Autoregressive-Moving Average Models of Unknown Order." *Biometrika* 71(3),
  599–607. https://doi.org/10.1093/biomet/71.3.599
- MacKinnon, J. G. (2010). "Critical Values for Cointegration Tests." Queen's
  Economics Department Working Paper No. 1227, Queen's University.
  https://www.econ.queensu.ca/sites/econ.queensu.ca/files/wpaper/qed_wp_1227.pdf
  — the response-surface coefficients embedded in every port (revision of
  MacKinnon 1991, in Engle & Granger, eds., *Long-Run Economic Relationships*,
  Oxford University Press, ch. 13).
- MacKinnon, J. G. (1994). "Approximate Asymptotic Distribution Functions for
  Unit-Root and Cointegration Tests." *Journal of Business & Economic
  Statistics* 12(2), 167–176. https://doi.org/10.1080/07350015.1994.10510005
  — the exact p-value response surface that this package deliberately
  approximates by interpolation.
- Engle, R. F. & Granger, C. W. J. (1987). "Co-integration and Error
  Correction: Representation, Estimation, and Testing." *Econometrica* 55(2),
  251–276. https://doi.org/10.2307/1913236
- Phillips, P. C. B. & Ouliaris, S. (1990). "Asymptotic Properties of
  Residual Based Tests for Cointegration." *Econometrica* 58(1), 165–193.
  https://doi.org/10.2307/2938339 — why residual-based tests need their own
  (more negative) critical values.
- Granger, C. W. J. & Newbold, P. (1974). "Spurious Regressions in
  Econometrics." *Journal of Econometrics* 2(2), 111–120.
  https://doi.org/10.1016/0304-4076(74)90034-7
- Johansen, S. (1988). "Statistical Analysis of Cointegration Vectors."
  *Journal of Economic Dynamics and Control* 12(2–3), 231–254.
  https://doi.org/10.1016/0165-1889(88)90041-3
- Johansen, S. (1991). "Estimation and Hypothesis Testing of Cointegration
  Vectors in Gaussian Vector Autoregressive Models." *Econometrica* 59(6),
  1551–1580. https://doi.org/10.2307/2938278
- Johansen, S. & Juselius, K. (1990). "Maximum Likelihood Estimation and
  Inference on Cointegration — with Applications to the Demand for Money."
  *Oxford Bulletin of Economics and Statistics* 52(2), 169–210.
  https://doi.org/10.1111/j.1468-0084.1990.mp52002003.x
- Osterwald-Lenum, M. (1992). "A Note with Quantiles of the Asymptotic
  Distribution of the Maximum Likelihood Cointegration Rank Test Statistics."
  *Oxford Bulletin of Economics and Statistics* 54(3), 461–472.
  https://doi.org/10.1111/j.1468-0084.1992.tb00013.x — the 95% trace table.
- Uhlenbeck, G. E. & Ornstein, L. S. (1930). "On the Theory of the Brownian
  Motion." *Physical Review* 36(5), 823–841.
  https://doi.org/10.1103/PhysRev.36.823 — the OU process behind the half-life.
- Kalman, R. E. (1960). "A New Approach to Linear Filtering and Prediction
  Problems." *Journal of Basic Engineering* 82(1), 35–45.
  https://doi.org/10.1115/1.3662552
- Bucy, R. S. & Joseph, P. D. (1968). *Filtering for Stochastic Processes with
  Applications to Guidance*. Interscience/Wiley, New York. — the Joseph-form
  covariance update.
- Chan, E. P. (2013). *Algorithmic Trading: Winning Strategies and Their
  Rationale*. Wiley, Hoboken. — half-life estimation (ch. 2) and the
  `Q = δ/(1−δ)·I`, `δ = 1e-4`, `R = 1e-3` Kalman pairs parameterisation
  (ch. 3).
- Gatev, E., Goetzmann, W. N. & Rouwenhorst, K. G. (2006). "Pairs Trading:
  Performance of a Relative-Value Arbitrage Rule." *Review of Financial
  Studies* 19(3), 797–827. https://doi.org/10.1093/rfs/hhj020
- Vidyamurthy, G. (2004). *Pairs Trading: Quantitative Methods and Analysis*.
  Wiley, Hoboken.
- Sharpe, W. F. (1994). "The Sharpe Ratio." *Journal of Portfolio Management*
  21(1), 49–58. https://doi.org/10.3905/jpm.1994.409501
- Lo, A. W. (2002). "The Statistics of Sharpe Ratios." *Financial Analysts
  Journal* 58(4), 36–52. https://doi.org/10.2469/faj.v58.n4.2453 — the
  autocorrelation caveat on √252 annualisation (not applied here).
- Khandani, A. E. & Lo, A. W. (2007). "What Happened to the Quants in August
  2007?" *Journal of Investment Management* 5(4), 29–78.
  https://doi.org/10.2139/ssrn.1015987
- Avellaneda, M. & Lee, J.-H. (2010). "Statistical Arbitrage in the US
  Equities Market." *Quantitative Finance* 10(7), 761–782.
  https://doi.org/10.1080/14697680903124632
- Murray, M. P. (1994). "A Drunk and Her Dog: An Illustration of Cointegration
  and Error Correction." *The American Statistician* 48(1), 37–39.
  https://doi.org/10.1080/00031305.1994.10476017
- Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University Press,
  Princeton. — OLS (ch. 8), unit roots through VECMs (chs. 15–20).
- Harvey, A. C. (1989). *Forecasting, Structural Time Series Models and the
  Kalman Filter*. Cambridge University Press, Cambridge.
- Press, W. H., Teukolsky, S. A., Vetterling, W. T. & Flannery, B. P. (2007).
  *Numerical Recipes: The Art of Scientific Computing*, 3rd ed. Cambridge
  University Press, §11.1 — the cyclic Jacobi rotation.
- Golub, G. H. & Van Loan, C. F. (2013). *Matrix Computations*, 4th ed. Johns
  Hopkins University Press, Baltimore, §8.5 — Jacobi methods.
- Neumaier, A. (1974). "Rundungsfehleranalyse einiger Verfahren zur Summation
  endlicher Summen." *Zeitschrift für Angewandte Mathematik und Mechanik*
  54(1), 39–51. https://doi.org/10.1002/zamm.19740540106 — the compensated
  summation used for the long cross-product sums (C++, Rust); Kahan, W.
  (1965). "Further Remarks on Reducing Truncation Errors." *Communications of
  the ACM* 8(1), 40. https://doi.org/10.1145/363707.363723 (Java).

## License

MIT License — see [`LICENSE`](LICENSE). Copyright (c) 2026 Ashish Jha.
Provided "as is", without warranty of any kind; not investment advice.
