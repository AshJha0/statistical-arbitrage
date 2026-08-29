# 09 — Statistical Arbitrage (Cointegration, Kalman Filter, Pairs Backtest)

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
| ADF unit-root test | Fixed lag (default 1), constant/no-trend, embedded MacKinnon-style critical values, interpolated approximate p-value |
| Half-life of mean reversion | AR(1) fit on the spread, `hl = -ln 2 / ln(1+ρ)` with guards |
| Engle–Granger two-step | Hedge OLS then residual ADF against the 2- or 3-variable EG table |
| Johansen-lite trace test | 2–3 assets, VECM with unrestricted constant, k = 1 lagged difference, native Cholesky + cyclic Jacobi eigensolver, Osterwald–Lenum 95% table |
| Kalman dynamic hedge | Random-walk state (αₜ, βₜ), RiskMetrics-style `Q = δ/(1−δ)·I`, Joseph-form covariance update, innovation z-scores |
| Rolling z-score signal | Current-observation-inclusive window, ddof = 1, zero-variance guard |
| Pairs backtest | FLAT/LONG/SHORT state machine, next-close execution (no look-ahead), equal-dollar legs, bps costs + slippage, stop-losses |
| Walk-forward split | Hedge estimated in-sample (60%), traded out-of-sample (40%) — degradation shown honestly |
| Teaching cases | Regime break at day 1050 (Kalman adapts, static OLS cannot), spurious regression on independent random walks (no trade) |

Everything is deterministic: no RNG at runtime (randomness lives only in
`data/generate_data.py`, fixed seeds).

## Directory layout

```
09-statistical-arbitrage/
  README.md               this file
  LEARN.md                theory: unit roots, cointegration, Kalman, pairs practice
  COOKBOOK.md             task-oriented recipes in all four languages
  API_SPEC.md             the pinned cross-language numerical contract
  docs/
    ARCHITECTURE.md       component design, data flow, numerical decisions
    diagrams/*.mmd        raw Mermaid sources
  data/
    pair_equity.csv       1500 days, cointegrated pair (true beta 1.7, break at day 1050)
    fx_triangle.csv       EURUSD vs GBPUSD*EURGBP basket (cointegration in logs)
    random_walks.csv      two independent walks — the spurious-regression case
    config.json           signal/cost parameters shared by all demos
    generate_data.py      regenerates everything (fixed seeds; cross-checks vs statsmodels)
    golden/golden.json    21 golden cases every language must reproduce
  python/                 reference implementation (numpy only in the package)
  cpp/                    C++17 library + GoogleTest suite + demo
  rust/                   Rust library + integration tests + demo binary
  java/                   Java 21 sources + JUnit 4 tests + demo
```

## Build, run, test

### Python (3.11; numpy required, statsmodels only used by tests as cross-check)

```bash
cd python
PYTHONPATH=src python3 demo.py          # run the demo
PYTHONPATH=src pytest -q                # run the tests
```

### C++ (g++ 13, CMake ≥ 3.28, GoogleTest via find_package)

```bash
cd cpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j2
./build/demo                            # run the demo
ctest --test-dir build --output-on-failure   # run the tests
```

### Rust (1.95, edition 2021)

```bash
cd rust
cargo run --bin demo --release          # run the demo
cargo test                              # run the tests
```

### Java (21; JUnit 4 at /usr/share/java/junit4.jar, no Maven/Gradle)

```bash
cd java
./build.sh                              # javac everything into out/
./demo.sh                               # run the demo main
./test.sh                               # JUnitCore on all *Test classes
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

`data/golden/golden.json` holds 21 named cases — OLS coefficients and R² on the
bundled pair, ADF t-stats (residuals and random walk), Engle–Granger verdicts,
Johansen trace statistics and rank, half-life, filtered Kalman betas at pinned
days, rolling z-scores at pinned days, and full backtest metrics (total return,
Sharpe, max drawdown, exact trade count of 31). Each case records its inputs,
one expected scalar, and an absolute tolerance (1e-6 to 1e-10; booleans encoded
as 0.0/1.0, integers matched exactly).

Every language's test suite loads this same file and asserts agreement, so a
port that deviates from the pinned contract — a different lag convention, a
non-Joseph covariance update, a different z-score windowing — fails its golden
suite immediately. The values are produced by the Python reference, which
`data/generate_data.py` first cross-validates against statsmodels (OLS and ADF)
and refuses to write on any disagreement.

## License

Educational use. No warranty; not investment advice.
