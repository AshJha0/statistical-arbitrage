# Statistical Arbitrage (Cointegration, Kalman Filter, Pairs Backtest)

A self-contained statistical-arbitrage toolkit implemented four times — Python
(reference), C++17, Rust, and Java 21 — against one pinned numerical contract
([`API_SPEC.md`](API_SPEC.md)). Every algorithm (OLS, ADF, Engle–Granger,
Johansen-lite, Kalman dynamic hedge, dollar-neutral pairs backtest) is specified
down to the solver so that all four languages agree to tight tolerances,
enforced by a shared golden-value suite.

**Status (v1.2.0).** A teaching and research-infrastructure library on
synthetic data. It is not a trading system and nothing in it is evidence of
edge; see [Limitations and what this is not](#limitations-and-what-this-is-not)
before reading any number below. Changes since v1.1.0 are in
[`CHANGELOG.md`](CHANGELOG.md).

The package name is `statarb` everywhere: Python package `statarb`, C++
namespace `statarb`, Rust crate `statarb`, Java package `com.quant.statarb`.

## What it does

| Feature | Notes |
|---|---|
| OLS regression | Normal equations + LU solve (pinned), plain SEs, t-stats, R² |
| ADF unit-root test | Fixed lag (default 1) or BIC-selected lag (`select_lag`), constant/no-trend, MacKinnon (2010) finite-sample critical values `b_inf + b1/T + b2/T² + b3/T³` at the regression sample size (T_eff ≥ 20 enforced), interpolated approximate p-value |
| Half-life of mean reversion | AR(1) fit on the spread, `hl = -ln 2 / ln(1+ρ)` with guards |
| Engle–Granger two-step | Hedge OLS then residual ADF against the 2- or 3-variable EG table; `engle_granger_both` runs both normalisations (y on x, x on y) and calls a pair cointegrated only if both reject |
| Johansen-lite trace test | 2–3 assets, VECM with unrestricted constant, k = 1 lagged difference, native Cholesky + cyclic Jacobi eigensolver, Osterwald–Lenum 95% table |
| Kalman dynamic hedge | Random-walk state (αₜ, βₜ), RiskMetrics-style `Q = δ/(1−δ)·I`, Joseph-form covariance update, innovation z-scores |
| Rolling z-score signal | Current-observation-inclusive window, ddof = 1, zero-variance guard; plain left-to-right sums, pinned in every port. Configs where the window cannot reach `entry_z` (or `stop_z` with no P&L stop) are rejected |
| Streaming API | `RollingZ(window)` and `KalmanStep(delta, r)`: O(1) per update, no allocation in `update`. `KalmanStep` matches the batch filter to 1e-12; `RollingZ` uses running sums and matches batch to 1e-9 (it drifts about 5e-8 after 50,000 updates at a level of 1e4), so batch stays the trading reference |
| Pairs backtest | FLAT/LONG/SHORT state machine, next-close execution (no look-ahead), bps costs + slippage, z-stop; a pending entry at the end of data is discarded, never force-closed the same day. Rejects `beta <= 0` |
| Sizing | `sizing = "beta_hedged"` (default): legs held in the hedge ratio, so P&L tracks the spread; gross notional equals `capital` at entry. `"equal_dollar"` is the v1.1.0 rule (`capital/2` per leg) |
| Risk limits | `pnl_stop_frac` (stop when open P&L <= -frac x capital), `max_hold_days`, `cooldown_days` after a stopped exit. All default to 0 (off). `data/config.json` turns them on at 0.02 / 60 / 20: round numbers fixed in advance, not tuned |
| Single holdout | `walk_forward`: ONE fit on the first 60%, ONE trade of the last 40%. The name is kept for compatibility; it is a holdout, not a rolling procedure |
| Rolling walk-forward | `rolling_walk_forward(y, x, config, train, test)`: refit every `test` days on the last `train` days, trade the next block with history-only warm-up, flat at each block end, stitched equity. A block whose fit has `beta <= 0` stays flat and is flagged `skipped` |
| Research helpers (Python only) | `statarb.research`: block-bootstrap Sharpe interval, probabilistic and deflated Sharpe, cost sensitivity, CUSUM break test, Benjamini–Hochberg, Kalman grid-MLE with innovation diagnostics, `scan_pairs`. Not ported, not in the goldens, not on the trading path |
| Teaching cases | Regime break at day 1050 (Kalman tracks beta; static OLS cannot; neither is shown to make money across the break), spurious regression on independent random walks (no trade) |
| Input validation | Every public entry point rejects NaN/inf, length mismatches, empty/constant/collinear inputs and out-of-domain parameters with the language's error type (`ValueError` / `std::invalid_argument` / `Err(StatarbError)` / `IllegalArgumentException`); the loaders refuse non-finite cells, a non-monotone `day` column, trailing JSON content, hex, `_` and type suffixes, and non-integer `window`/day fields |

Everything is deterministic: no RNG at runtime (randomness lives only in
`data/generate_data.py`, fixed seeds).

## Limitations and what this is not

**Real-data result (2026-11-05): null.** A preregistered study on US equities, 2020-2026 holdout, gave net Sharpe -0.42 and about zero gross return. See [research/real_data_2026_11/RESULTS.md](research/real_data_2026_11/RESULTS.md).

Read this before the numbers.

- **No real data.** The three CSVs come from `data/generate_data.py`, and the
  pair's generator is the model the tests assume. Over 300 seeds with no
  break, the out-of-sample Sharpe is 1.44; that recovers the generator and is
  not evidence of anything (review C1, `review/2026-11-05/REVIEW.md`).
- **Not tradable as it stands.** Fractional quantities, both legs filled at
  one close, a flat 1.5 bps per side, no borrow, financing, impact, capacity
  or locate check, no restricted lists or corporate actions, no portfolio or
  sector limits, no broker adapter or reconciliation. One pair at a time.
- **Not low latency.** The batch functions are O(T x W) for the z-score and
  allocate freely. The streaming API (`RollingZ`, `KalmanStep`) is O(1) per
  update, but there are no benchmarks and no book, feed or order layer.
- **Not a research pipeline.** `statarb.research` gives a bootstrap interval,
  deflated Sharpe, a break test and a multiple-testing-aware scan, in Python
  only. There is still no universe, no survivorship handling and no
  preregistered holdout. In the v1.1.0 review, 32 of 780 independent
  random-walk pairs passed Engle–Granger on one window; their mean in-sample
  Sharpe was 0.76 and out-of-sample -0.16 (C2).
- **The Kalman filter is never traded by the library, and trading it did not
  work.** "Kalman adapts" is true of beta tracking only (RMSE 0.024 vs 0.288
  for static OLS). In the review, trading the filtered spread on days
  150–1049 gave Sharpe 0.05 (35 trades), against 1.49 for an estimated static
  hedge on a different window (in-sample days 0–899) and 1.72 with the true
  beta. One seed. The cause measured: `r = 1e-3` is about 2,800 times below
  the spread variance (2.82), so beta absorbs part of the signal (C3).
- **Stops do not cap the damage.** The z-stop alone did not: it cost $17k on
  day 1053, and re-entry on day 1059 lost $124k, more than the whole net loss,
  because the rolling mean follows the break and z decays (C4). With the P&L
  stop, max-hold and cool-down on, the single-holdout out-of-sample result is
  -4.91% with max drawdown 6.86%, against -12.35% and 15.28% with none
  (demo, section 3). That is a smaller loss, still a loss. Across 300 seeds
  with the break the out-of-sample Sharpe was -1.9, negative 99.7% of the
  time (measured before the limits existed; not re-measured with them).
- **`walk_forward` is a single holdout.** `rolling_walk_forward` is the
  rolling version. Its bundled result (return -0.29%, Sharpe -0.05, 95%
  block-bootstrap interval [-1.11, 1.45]) is flattered: the last three of the 10 blocks (indices
  7–9) fit `beta <= 0` and sat flat. Fewer trades, not better trades.
- **Round risk limits, not tuned.** 2% / 60 days / 20 days were fixed before
  looking at the out-of-sample window. They have not been validated on any
  other data.
- **Economics, synthetic.** From `review/2026-11-05/experiments/exp4_economics.py`:
  300 seeds of this repo's generator at best-case parameters. No-break, zero
  cost: 1.70% a year on gross notional, 5.1 round trips a year, in the market
  23% of days. Break-even cost 16.7 bps a side; net 1.55% at 1.5 bps, 0.69%
  at 10. A break window averages -12.6% (5th percentile -18.2%), so break-even
  is one break per about 10.5 years per pair; the profile is a small steady
  gain with a rare large loss. These describe the generator, not any market.
  No real-market profit has been measured.

## Directory layout

```
statistical-arbitrage/
  README.md               this file
  LEARN.md                theory: unit roots, cointegration, Kalman, pairs practice
  COOKBOOK.md             task-oriented recipes in all four languages
  CHANGELOG.md            what changed per release
  API_SPEC.md             the pinned cross-language numerical contract
  docs/
    ARCHITECTURE.md       component design, data flow, numerical decisions
    GITHUB_PAGES.md       publishing guide for the docs site: https://ashjha0.github.io/statistical-arbitrage/
    index.html            the project site
    diagrams/*.mmd        raw Mermaid sources
  data/
    pair_equity.csv       1500 days, cointegrated pair (true beta 1.7, break at day 1050)
    fx_triangle.csv       EURUSD vs GBPUSD*EURGBP basket (cointegration in logs)
    random_walks.csv      two independent walks — the spurious-regression case
    config.json           signal, cost and risk-limit parameters shared by all demos
    generate_data.py      regenerates everything (fixed seeds; cross-checks vs statsmodels)
    golden/golden.json    928 golden cases every language must reproduce
  .github/workflows/      CI for all four languages plus a data-regeneration check
  python/                 reference implementation (numpy only in the package;
                          research.py holds the Python-only research helpers)
  cpp/                    C++17 library + GoogleTest suite + demo
  rust/                   Rust library + integration tests + demo binary
  java/                   Java 21 sources + JUnit 4 tests + demo (Maven pom.xml or plain javac)
```

## Build, run, test

The `.sh` helpers are invoked with `bash` below so they work on a fresh
clone regardless of file modes.

CI (`.github/workflows/ci.yml`) runs on every push to `main` and every pull
request: Python 3.11 and 3.12 with ruff, mypy and pytest; C++ with g++ and
clang++, plus an ASan/UBSan build; Rust with `cargo fmt --check`, clippy at
`-D warnings` and the tests; Java with Maven; and a job that regenerates
`data/` and fails if it differs from the committed files. Passing locally
does not replace it.

### Python (3.11+; numpy required, pytest + statsmodels only for the tests)

```bash
cd python
pip install -r requirements.txt         # numpy; pytest + statsmodels for tests
PYTHONPATH=src python3 demo.py          # run the demo
PYTHONPATH=src python3 -m pytest -q     # run the tests (1113 pass; the golden cases are most of them)
ruff check . && mypy src/statarb        # lint and types, as CI does
```

### C++ (g++ 13 or clang++, CMake ≥ 3.16; GoogleTest from the system if found, else fetched v1.14.0)

```bash
cd cpp
bash build.sh                           # = cmake -S . -B build && cmake --build build -j2 (Release is the default build type)
./build/statarb_demo                    # run the demo (from cpp/, reads ../data)
ctest --test-dir build --output-on-failure   # run the tests (107 pass)
# sanitizers: cmake -S . -B build -DSTATARB_SANITIZE=ON   (ASan + UBSan)
# the build uses -ffp-contract=off so a*b+c is never fused, which cross-language agreement needs
```

### Rust (1.95, edition 2021; only dependency `serde_json` for the config loader)

```bash
cd rust
cargo run --bin demo --release          # run the demo
cargo test --release                    # run the tests (115 pass; `cargo build --release` is warning-free)
cargo fmt --check && cargo clippy --all-targets --locked -- -D warnings   # as CI does
```

### Java (21; Maven, or plain javac with JUnit 4 and Hamcrest jars)

```bash
cd java
bash build.sh                           # javac -Xlint:all -Werror --release 21 into out/ (tests only if the jars are found)
bash demo.sh                            # run the demo main
bash test.sh                            # `mvn -B -q test` if Maven is installed, else JUnitCore (109 pass)
# without Maven, set JUNIT_JAR and HAMCREST_JAR (or JUNIT_HOME) if they are not in /usr/share/java or ~/.m2
```

## Demo output (Python reference)

Real output of `PYTHONPATH=src python demo.py` in `python/`. The data is
synthetic.

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
    900      1.700       1.6690         1.5178
   1049      1.700       1.7161         1.5178
   1100      1.100       1.1469         1.5178
   1200      1.100       1.1001         1.5178
   1499      1.100       1.1532         1.5178
  beta tracking RMSE (day 300+): kalman 0.0237 vs static OLS 0.2879
  (beta tracking only: the library never trades the Kalman hedge, and a
   traded Kalman lost most of the pre-break edge in the review, one seed)

=== 3. Pairs backtest: holdout, rolling walk-forward, risk limits =====
  risk limits: P&L stop 2% of capital, max hold 60d, cool-down 20d
  (round numbers fixed in advance, not tuned on this data)
  single holdout, hedge from days 0..899: alpha 17.3603, beta 1.6389
  in-sample    ret    6.81%  ann    1.86%  sharpe   1.40  maxDD  0.87%  trades  19  win 89.5%  hold  10.2d  turn  37.9x  stops 1
  OOS limits   ret   -4.91%  ann   -2.09%  sharpe  -0.78  maxDD  6.86%  trades   9  win 55.6%  hold  15.4d  turn  18.0x  stops 3
  OOS none     ret  -12.35%  ann   -5.39%  sharpe  -1.54  maxDD 15.28%  trades   9  win 55.6%  hold  24.2d  turn  17.9x  stops 1
  rolling walk-forward, refit every 100d on the last 500d (10 blocks, days 500+):
  roll limits  ret   -0.29%  ann   -0.07%  sharpe  -0.05  maxDD  5.55%  trades  15  win 73.3%  hold  10.1d  turn  29.9x  stops 2
  roll none    ret   -8.33%  ann   -2.17%  sharpe  -1.04  maxDD 13.25%  trades  15  win 73.3%  hold  12.5d  turn  29.8x  stops 1
  rolling Sharpe -0.05, 95% block-bootstrap interval [-1.11, 1.45]; P(true Sharpe > 0) = 0.46
  CUSUM on the spread: in-sample p = 0.149, full sample p = 0.012, largest shift near day 1057
  limits shrink the OOS loss but it is still a loss. The last three rolling blocks fit
  beta <= 0 and stayed flat, which flatters that result. The data is synthetic:
  none of this is evidence of edge.

=== 4. Spurious-regression teaching case (independent random walks) ===
  regressing walk_a on walk_b: slope t-stat = 8.4 —
  'significant' by the usual +/-2 rule, yet the walks are independent.
  ADF(walk_a) t = -1.095 (unit root not rejected) and
  ADF(resid) t = -1.154 vs 5% crit -3.34 -> cointegrated: False
  verdict: spurious relationship — NO backtest is run on this pair.

demo complete.
```

Section 3 is the point. In-sample Sharpe 1.40 is on the sample that chose the
hedge. Out of sample, the P&L stop, max-hold and cool-down cut the loss from
-12.35% to -4.91% and the drawdown from 15.28% to 6.86%; it is still a loss.
The rolling result looks near break-even partly because the last three blocks sat flat.

## Golden-value cross-language testing

`data/golden/golden.json` holds 928 named cases (the count follows the trade lists) — OLS coefficients and R² on the
bundled pair, ADF t-stats (residuals and random walk) together with their
MacKinnon (2010) finite-sample 5% critical values and an interpolated p-value,
Engle–Granger verdicts, Johansen trace statistics and rank, half-life, filtered
Kalman betas and z-scores at pinned days, rolling z-scores at pinned days,
`select_lag`, ADF with `nvars = 3`, half-life sentinels, and backtests in both
sizing modes and with the risk limits on: metrics, equity at pinned days and
the trade list (entry, exit, direction, P&L, stopped), plus `walk_forward` and
`rolling_walk_forward` metrics. The full-sample backtest on the bundled pair
with `config.json` has 29 trades, return 3.70%, Sharpe 0.30, max drawdown
6.32%. One case records the smallest margin between a z-score and a threshold
(a last-bit difference in a port could flip a trade only if that margin were
tiny); the tests assert it exceeds 1e-6. Each
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

Test counts (all passing): Python 1113 (run on Windows for this release; CI
runs 3.11 and 3.12 on Linux), C++ 107, Rust 115, Java
109. Beyond the golden layer every port
carries the same behavioural suite: accounting identity, gross notional equal
to `capital` at entry, next-close execution, the end-of-data discard rule on
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
  are basis points of traded notional; `capital` is gross notional at each
  entry regardless of accumulated P&L (returns are on notional, not
  compounded on equity; no leverage cap, no borrow/financing on the short
  leg). With the default `beta_hedged` sizing the legs are held in the ratio
  `1 : beta` by quantity, so gross is `capital` but the position is not
  dollar-neutral; `equal_dollar` is neutral at entry but carries a net
  exposure whenever beta differs from y/x (the review measured Sharpe 0.43
  equal-dollar vs 1.76 beta-hedged on `y = 150 + 0.5x`, 100 seeds).
  `beta <= 0` is rejected.
- **Time grid.** One row = one observation, assumed equally spaced; the
  annualisation constant 252 and the next-close execution lag are hard-coded.
  The loader enforces a strictly increasing `day` column, finite cells and a
  plain decimal/exponent number grammar, but
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
  overshoot is adverse or favourable. With `cooldown_days = 0` (the library
  default) re-entry is allowed the very next day, so a break regime can
  whipsaw; `config.json` sets 20. `pnl_stop_frac` is the actual loss limit
  and takes priority over the z rules. Treat a z-stop as "model
  disagreement"; its measured effect is in the limitations above.
- **What is validated.** Every public entry point checks lengths, finiteness,
  emptiness, constancy/collinearity, parameter domains and window bounds and
  fails with the language's error type; no NaN is ever silently propagated
  into a metric. Config structs are re-validated at run time in C++/Rust.
- **Deliberately out of scope.** Calendars/day-count, `periods_per_year`,
  intraday data, financing and borrow costs, market impact, position
  compounding, rolling ADF/half-life diagnostics helpers, Johansen
  deterministic-term variants and p-values, structured
  error codes beyond Rust's enum. These are documented limits, not hidden ones.

This is a reference implementation of the pinned algorithms with
cross-language golden agreement and edge-case tests, plus Python research
helpers. It is not a trading system, the data is synthetic, and nothing here
is investment advice.

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
