# ARCHITECTURE — statarb

Design notes for the four parallel implementations (Python reference, C++,
Rust, Java) of the statistical-arbitrage toolkit. The single source of truth
for every algorithm is [`../API_SPEC.md`](../API_SPEC.md); this document
explains how the code is organised, why the numerical choices were made, and
how correctness is enforced across languages.

## 1. Component responsibilities

| Component (Python module) | Responsibility | Depends on |
|---|---|---|
| `ols` | Normal-equations OLS: params, plain SEs, t-stats, residuals, R², condition guard | — |
| `adf` | ADF regression builder, embedded MacKinnon (2010) response surfaces (nvars 1/2/3) evaluated at T_eff, interpolated p-value, `reject` verdict, BIC `select_lag` | `ols` |
| `signals` | Spread construction, rolling z-score (inclusive window, ddof 1, zero-variance guard, sequential sums), streaming `RollingZ`, half-life | `ols` |
| `engle_granger` | Two-step test: hedge OLS + residual ADF against the nvars=2/3 table; `engle_granger_both` runs both normalisations | `ols`, `adf` |
| `johansen` | VECM trace test: partialling-out, product moments, native Cholesky, native cyclic Jacobi eigensolver, Osterwald–Lenum table, rank estimate | — (self-contained linear algebra) |
| `kalman` | (αₜ, βₜ) random-walk filter, Joseph-form update, innovation z-scores; `KalmanStep` is the one-tick form the batch filter loops over | — |
| `backtest` | `BacktestConfig` validation (including dead-window rejection), FLAT/LONG/SHORT state machine, sizing modes, P&L stop / max-hold / cool-down, cost accounting, pinned metrics, `walk_forward` (single holdout) and `rolling_walk_forward` | `ols`, `signals` |
| `data_io` | Minimal CSV/JSON loaders with validation: finite cells, no ragged rows, unique headers, strictly increasing `day` column, strict number grammar, no trailing JSON content (keeps the package numpy-only / dependency-light) | — |
| `trading` (Python only, paper only) | `orders` (order state machine), `broker` (`Broker` interface, `PaperBroker` fill model), `execution` (`PairExecutor`: two-leg pacing, unwind), `risk` (`RiskEngine`: pre-trade limits, kill switch), `ledger` (independent books, reconciliation, attribution, `AuditLog`), `runner` (`run_paper` day loop). No network code, no credentials, no live-order path; a live adapter is an owner decision. Not ported, not in the goldens | `backtest` (rule parameters), `signals` |
| `research` (Python only) | Block-bootstrap Sharpe interval, probabilistic and deflated Sharpe, cost sensitivity, CUSUM break test, Benjamini–Hochberg, Kalman grid-MLE with innovation diagnostics, `scan_pairs`. Not ported, not in the goldens, not on the trading path | `backtest`, `kalman`, `engle_granger` |

The dependency graph is a shallow DAG — everything statistical funnels
through `ols`, and `johansen` deliberately stands alone with its own tiny
linear-algebra kernel (Cholesky + Jacobi) so that no library eigensolver can
introduce cross-language drift.

## 2. Data flow

```mermaid
flowchart TD
    subgraph inputs [Bundled inputs]
        CSV["pair_equity.csv / fx_triangle.csv / random_walks.csv"]
        CFG["config.json"]
    end
    CSV --> IO["data_io: load_csv\n(strict number grammar, finite cells,\nmonotone day column)"]
    CFG --> IOC["data_io: load_config\n(no trailing content, integer window/days)"]
    IO --> EG["engle_granger / engle_granger_both\n(hedge alpha, beta + residual ADF,\noptional select_lag)"]
    IO --> JO["johansen_trace\n(rank, cointegrating vectors)"]
    IO --> KF["kalman_hedge / KalmanStep\n(dynamic alpha_t, beta_t, innovations)"]
    EG -- "not cointegrated" --> STOPX["no trade\n(spurious pair rejected)"]
    EG -- "alpha, beta (must be > 0)" --> SPR["signals: spread\ns = y - (a + b x)"]
    EG -- "residuals" --> HL["signals: half_life"]
    SPR --> Z["signals: rolling_zscore / RollingZ\n(window W, ddof 1, sequential sums)"]
    IOC --> BTCFG["BacktestConfig\n(thresholds, costs, sizing,\npnl_stop_frac, max_hold_days, cooldown_days;\nreject dead windows)"]
    Z --> BT["backtest_pair\nstate machine + accounting"]
    BTCFG --> BT
    BT --> MET["metrics: return, Sharpe,\nmaxDD, trades, turnover, stops"]
    IO --> WF["walk_forward\n(single holdout: 60% fit / 40% trade)"]
    IO --> RWF["rolling_walk_forward\n(refit every test days on last train days,\nflat at block ends, stitched equity)"]
    WF --> BT
    RWF --> BT
    MET --> RES["research.py (Python only)\nbootstrap CI, deflated Sharpe, CUSUM,\nBH, scan_pairs"]
    MET --> GOLD["golden.json\n928 cross-language cases"]
    JO --> GOLD
    KF --> GOLD
    HL --> GOLD
```

Two properties of this flow are load-bearing:

- **The cointegration gate.** `backtest_pair` is only reached with hedge
  parameters that came from a passing Engle–Granger (or Johansen) verdict;
  the spurious random-walk pair is diagnosed and never traded. The demo
  encodes this discipline explicitly.
- **The holdout split.** `walk_forward` re-enters the same
  `backtest_pair` twice with one frozen (α, β): a single holdout, not a
  rolling procedure (the name is kept for compatibility). The out-of-sample
  z-score warms up inside its own segment, so no statistic computed after the
  split uses pre-split data. The split index is `floor(T·frac + 1e-9)` so that
  binary-float products (`700 × 0.7 = 489.99999999999994`) cannot move it.
- **Rolling walk-forward.** `rolling_walk_forward` runs one backtest per
  block on days `[start − (W−1), end)` with that block's own OLS fit. The W−1
  earlier days only warm the z-score; entries are ignored before `start`;
  the end-of-data rule flattens at `end − 1`; cool-down state does not cross
  blocks. A block whose fit has β ≤ 0 stays flat and is flagged `skipped`
  instead of failing the run. Block P&L is added to a running equity to give
  one curve. The consequence to remember: skipped blocks add zero, so a
  result with skipped blocks is flattered (on the bundled pair the last three
  of 10 blocks are skipped).

## 3. Core types and call structure

```mermaid
classDiagram
    class OLSResult {
        +params : vector
        +se : vector
        +tstats : vector
        +resid : vector
        +r2 : double
        +sigma2 : double
        +nobs : int
        +nparams : int
    }
    class ADFResult {
        +tstat : double
        +pvalue : double
        +crit : map
        +maxlag : int
        +nvars : int
        +reject(alpha) bool
    }
    class EGResult {
        +alpha : double
        +beta : double
        +resid : vector
        +r2 : double
        +adf : ADFResult
        +cointegrated(alpha) bool
    }
    class EGBothResult {
        +y_on_x : EGResult
        +x_on_y : EGResult
        +cointegrated(alpha) bool
    }
    class JohansenResult {
        +eigenvalues : vector
        +trace_stats : vector
        +crit_95 : vector
        +rank : int
        +beta : matrix
        +t_eff : int
    }
    class KalmanResult {
        +alpha : vector
        +beta : vector
        +z : vector
        +innovation : vector
        +s_var : vector
        +final_cov : matrix2x2
    }
    class KalmanStep {
        +update(x, y) alpha, beta, z
    }
    class RollingZ {
        +update(s) z
    }
    class BacktestConfig {
        +window : int
        +entry_z : double
        +exit_z : double
        +stop_z : double
        +cost_bps : double
        +slippage_bps : double
        +capital : double
        +sizing : beta_hedged or equal_dollar
        +pnl_stop_frac : double
        +max_hold_days : int
        +cooldown_days : int
    }
    class Trade {
        +entry_day : int
        +exit_day : int
        +direction : int
        +pnl : double
        +stopped : bool
    }
    class BacktestResult {
        +equity : vector
        +daily_pnl : vector
        +daily_cost : vector
        +zscore : vector
        +trades : List~Trade~
        +metrics : map
    }
    class WalkForwardResult {
        +alpha : double
        +beta : double
        +split : int
        +insample : BacktestResult
        +oos : BacktestResult
    }
    class RollingBlock {
        +start : int
        +end : int
        +alpha : double
        +beta : double
        +skipped : bool
        +result : BacktestResult
    }
    class RollingWalkForwardResult {
        +blocks : List~RollingBlock~
        +start_day : int
        +equity : vector
        +trades : List~Trade~
        +metrics : map
    }
    EGResult --> ADFResult : nests
    EGResult ..> OLSResult : built from
    ADFResult ..> OLSResult : t-stat from
    EGBothResult --> EGResult : two directions
    BacktestResult --> Trade : collects
    WalkForwardResult --> BacktestResult : IS + OOS
    RollingWalkForwardResult --> RollingBlock : blocks
    RollingBlock --> BacktestResult : block backtest
    BacktestResult ..> BacktestConfig : parameterised by
    KalmanResult ..> KalmanStep : batch loops over
```

Each language expresses these as its idiomatic equivalent: frozen dataclasses
(Python), plain structs (C++/Rust), records/immutable classes (Java). Field
names and semantics are identical everywhere; only casing follows local
convention.

## 4. The backtest day loop (pinned event order)

The trickiest contract in the project is the per-day sequence — it encodes
"signal at close t, execute at close t+1" with zero look-ahead:

```mermaid
sequenceDiagram
    participant P as Prices day t
    participant BT as backtest_pair
    participant SM as State machine
    participant EQ as Equity ledger
    P->>BT: y_t, x_t
    BT->>EQ: 1. mark-to-market pnl_t (quantities from day t-1)
    BT->>SM: 2. execute pending order (decided at close t-1)
    SM-->>EQ: entry/exit costs on traded notional
    Note over SM: a pending ENTRY is executed only if t < T-1<br/>(an entry signalled at T-2 is discarded, never opened-and-closed at one close)
    alt t == T-1 and position open
        BT->>SM: 2b. force-close at final close
        SM-->>EQ: exit cost, Trade appended
    end
    BT->>EQ: E_t = E_(t-1) + pnl_t - cost_t
    alt t < T-1
        BT->>SM: 3. evaluate z_t, set at most one pending order
        Note over SM: in position, in this order:<br/>1 P&L stop (open P&L <= -pnl_stop_frac x capital): stop<br/>2 z-stop (|z| >= stop_z): stop<br/>3 max_hold_days reached: exit (not a stop)<br/>4 |z| <= exit_z: exit<br/>FLAT: entry if entry_z <= |z| < stop_z and past the cool-down<br/>(cooldown_days after a stopped exit)<br/>NaN z: no action
    end
```

Mark-to-market strictly precedes execution, so a fill never earns same-day
P&L; the signal step runs last and only writes a *pending* order. At the
signal step the P&L stop is checked first (open P&L net of costs so far),
then the z-stop, then the max-hold exit; a stopped exit starts the cool-down.
The stop therefore executes one close after it is detected, so the realised
loss can exceed `pnl_stop_frac × capital`. A pending
entry on the final day is discarded rather than executed and force-closed
at the same close (which would report a phantom zero-holding round trip that
paid two costs — the bug class a rolling daily re-run would otherwise hit
every day). Any reordering changes the golden trade lists (29 trades on the bundled pair with `config.json`) and every
downstream metric — which is exactly why the order is pinned, golden-tested,
and property-tested over every truncation of the bundled pair.

## 5. Numerical design decisions and trade-offs

**Normal equations, not QR/SVD (OLS).** Design matrices here have k ≤ 4
columns, and the golden tolerances (1e−10 on coefficients) require every
language's rounding error to stay well inside that band. LU on X'X is
trivially portable (a 4×4 solve); QR pivoting orders and Householder sign
conventions differ across libraries. Cost: X'X squares the condition number
— mitigated by the pinned cond ≤ 1e12 guard that converts near-collinearity
into a clean error. Trade-off accepted: this OLS is not suitable for large
ill-conditioned designs, and says so.

**Accumulation strategy differs per port for the long cross products;
agreement there is by tolerance. The rolling z-score is pinned bitwise.** The
1500-term cross products are summed by numpy/BLAS
pairwise-style accumulation (Python), Neumaier compensated summation (C++
and Rust) and Kahan compensated summation (Java). All three compensated
schemes are accurate to a few ulps of the exact sum, so the four ports agree
within the golden tolerances (1e−6 … 1e−10) *despite* different rounding
sequences. The exception is the rolling z-score: mean and squared
deviations are plain left-to-right sums in every port (no pairwise, Neumaier
or Kahan), because the trade decision `|z| >= entry_z` is a hard branch and a
last-bit difference could flip a trade and change the whole path. The tests
assert the smallest margin between a z-score and a threshold on the bundled
data exceeds 1e−6, and the C++ build uses `-ffp-contract=off` so `a*b+c` is
never fused. The C++ port
deliberately avoids `long double`, whose width is 80 bits on x86 gcc/clang,
64 bits (plain double) on MSVC and 128-bit software arithmetic on AArch64 —
a portability trap for tolerance-based golden tests.

**Fixed ADF lag by default, opt-in BIC selection, MacKinnon (2010)
response-surface critical values, interpolated p-value.** The default lag
stays fixed so the statistic does not depend on a selector. `select_lag` is
pinned tightly enough to agree across ports (one common sample, strict
improvement to move to a larger lag), and `engle_granger_both` removes the
choice of normalisation by requiring both directions to reject. Critical values are the
published finite-sample response surfaces `b_inf + b1/T + b2/T² + b3/T³`
(nine rows × four coefficients, embedded verbatim, evaluated in Horner
form at the regression's own T_eff) — the same numbers statsmodels'
`mackinnoncrit` uses, cross-validated against it by `generate_data.py`. The
surfaces were fitted on samples of 20 and more, so `T_eff < 20` is refused
instead of extrapolated. The p-value remains a documented three-knot
piecewise-linear approximation clamped to [1e−4, 0.9999]; the *verdict*
never touches it (direct comparison with the finite-sample value).

**Native Cholesky + cyclic Jacobi for Johansen.** The canonical-correlation
eigenproblem is solved in symmetric form via `A = L⁻¹ S₁₀S₀₀⁻¹S₀₁ L⁻ᵀ`.
Library eigensolvers (LAPACK, Eigen, ndarray, java built-ins) differ in
algorithm, sweep order, and eigenvector signs; a hand-rolled cyclic Jacobi
with a pinned rotation formula, sweep order, tolerance (1e−14) and sweep cap
(100) produces identical eigenvalues everywhere. On 2×2/3×3 symmetric PSD
matrices Jacobi converges quadratically in a handful of sweeps; hitting the
sweep cap is defined as non-fatal (best iterate returned) because it cannot
occur for these sizes. Eigenvalues are clipped to [0, 1−1e−12] before
`ln(1−λ)` so the trace statistic can never produce −inf.

**Joseph-form Kalman update.** `(I−KH)P` can lose symmetry/PSD-ness under
rounding; Joseph form `(I−KH)P(I−KH)' + rKK'` is PSD by construction, and
an explicit `P := (P+P')/2` symmetrisation removes residual asymmetry bits.
For a 2×2 state the extra flops are negligible and the payoff is 1e−8
cross-language agreement over 1500 filtering steps. The prior (s₀ = 0,
P₀ = 1e3·I) is pinned because the first ~50 filtered betas depend on it.

**Beta-hedged sizing by default (changed in v1.2.0).** v1.1.0 put
`capital/2` on each leg and let β act only in the signal. That trades a
different spread from the one tested whenever β differs from y/x, and the
review measured it: Sharpe 0.43 equal-dollar vs 1.76 beta-hedged on
`y = 150 + 0.5x` (100 seeds). The default now holds `u = capital/(y + βx)` of
y against `−βu` of x, so P&L follows the spread and gross notional is exactly
`capital` at entry; the cost is that the position is not dollar-neutral.
`equal_dollar` remains as an option. β ≤ 0 is rejected, because a negative
hedge would put both legs on the wrong side.

**Risk limits are round numbers, not tuned.** `pnl_stop_frac = 0.02`,
`max_hold_days = 60` and `cooldown_days = 20` in `config.json` were fixed
before the out-of-sample window was examined. All default to off in code. A
config whose window cannot reach `entry_z` (|z| is bounded by `(W−1)/√W`), or
reach `stop_z` with no P&L stop to bound the loss, is rejected as dead.

**Streaming alongside batch.** `KalmanStep` is the single step function the
batch filter loops over, so the two cannot diverge (1e−12). `RollingZ` uses a
ring buffer and running sums (Welford while filling), which is O(1) and
allocation-free but not bitwise equal to batch: it matches to 1e−9 at the
bundled length and drifts about 5e−8 after 50,000 updates at a level of 1e4.
Batch is the trading reference.

**Undefined metrics are pinned, not left to IEEE.** `ann_return =
(1+total)^(252/N) − 1` has no real value once equity is ≤ 0; every port pins
it to exactly −1 in that case (max drawdown then exceeds 1 but stays
finite). Every metric is asserted finite on a synthetic total-loss fixture.

**Determinism.** No RNG at runtime anywhere; randomness exists only in
`data/generate_data.py` with fixed seeds (42, 7, 100). All iteration orders,
tolerances, and tie-breaks are pinned. Any two runs of any implementation
produce byte-identical metrics.

## 6. Error-handling strategy

One validation contract (API_SPEC §9), four idiomatic surfaces:

| Language | Mechanism | Example |
|---|---|---|
| Python | `raise ValueError("...")` | `"window (61) larger than series length (60)"` |
| C++ | `throw std::invalid_argument` | same message text |
| Rust | `Result<T, StatarbError>` (hand-written enum: `LengthMismatch`, `NonFinite`, `ConstantSeries`, `Singular`, `TooShort`, `InvalidParam`, `Data`; no panic on bad input) | `Err(StatarbError::InvalidParam("window (61) larger than series length (60)"))` |
| Java | `throw IllegalArgumentException` | same message text |

Design rules:

- **Validate at the boundary, then trust.** Every public entry point checks
  lengths, emptiness, finiteness, domains (δ ∈ (0,1), r finite and > 0,
  0 ≤ maxlag ≤ 1000, nvars ∈ {1,2,3}, finite thresholds and costs plus
  their ordering) before any arithmetic; internal helpers assume clean
  inputs. Domain checks are written as *negated* comparisons
  (`!(r > 0.0 && isfinite(r))`) so that NaN — which compares false to
  everything — cannot slip through a `<= 0` test. Ports with mutable config
  structs (C++, Rust) re-run the validation inside `backtest_pair` /
  `walk_forward`. The CSV loader is a boundary too: non-finite cells and a
  non-monotone `day` column are refused before any model sees them.
- **Errors for misuse, guarded values for in-domain degeneracy.** A constant
  series into ADF is an *error*; a zero-variance z-score window is *NaN*
  (no signal); ρ ≥ 0 in half-life is *+inf* (no reversion). The line: if
  the caller could not have known in advance, return a defined sentinel;
  if the input violates the contract, refuse loudly.
- **No silent NaN propagation.** NaN appears only where the spec defines it
  (z-score warm-up and zero-variance windows) and the backtest state machine
  treats it as "no action" explicitly. A NaN parameter (cost, threshold,
  Kalman `r`) is an error at construction, never a NaN equity curve with a
  plausible-looking Sharpe of 0.0.
- **Typed errors even for degenerate shapes.** Empty arrays raise the
  language's validation error (never `ArrayIndexOutOfBounds` / UB / a Rust
  panic); `walk_forward` checks lengths before slicing; `maxlag` is bounded
  and computed in 64-bit so no signed overflow is reachable.
- **Non-convergence policy.** The Jacobi sweep cap is the only iterative
  loop; hitting it returns the best iterate (documented, cannot happen at
  n ≤ 3) rather than throwing — matching the "report, don't crash"
  convention for calibration-type code.

## 7. Testing strategy

Four layers, mirrored in every language:

1. **Golden values** (`data/golden/golden.json`, exactly 928 cases — each
   suite asserts the count; it follows the trade lists, so suites loop and
   dispatch rather than enumerate names): end-to-end scalars — OLS
   coefficients, ADF t-stats (including `nvars = 3` and maxlag 0/3) with their
   finite-sample 5% critical values and an interpolated p-value, `select_lag`,
   EG/Johansen verdicts and trace stats (including n = 3 and the leading
   vector), half-life and its sentinels, Kalman betas and z at pinned days,
   z-scores at pinned days, and backtests in both sizing modes and with the
   risk limits on: metrics, equity at pinned days and the trade list (entry,
   exit, direction, P&L, stopped), plus `walk_forward` and
   `rolling_walk_forward` metrics and the threshold-margin case. Tolerances
   1e−6 to 1e−10. This is the
   cross-language contract-enforcement layer; the values are written by
   `generate_data.py` only after the Python reference passes a statsmodels
   cross-check of OLS (coefficients, SEs), ADF (t-stat and `mackinnoncrit`
   critical values) and Johansen (`coint_johansen` plus an independent numpy
   generalised-eigen solve).
2. **Analytic identities**: OLS vs the closed-form two-variable solution;
   the backtest accounting identity `E_end − capital = Σpnl − Σcost` at
   1e−10; gross notional equal to `capital` at entry (and exact dollar-neutrality
   for `equal_dollar`); Joseph-form P kept PSD (2×2
   closed-form eigenvalues on the final posterior); Johansen eigenvalue
   ordering λ₁ ≥ λ₂ ≥ 0; the MacKinnon surface hand-evaluated at T = 20,
   50, 898 and 1498.
3. **Synthetic-spread fixtures with derived expectations** (all four
   ports): a single spike on a zero background has the closed-form z-score
   `(W−1)/√W` (and `−1/√W` the next day), which pins the stop-band-while-flat
   rule; small sinusoidal backgrounds with plateaus pin the entry → stop →
   exit timing (`exit_day == t_stop + 1`, cost equals bps × recomputed exit
   notional), same-direction re-entry with no cool-down, P&L-stop priority,
   max-hold and cool-down timing, the end-of-data
   discard rule, walk-forward split exactness (700 × 0.7 → 490), and the
   negative-equity metrics (total return −1.5 exactly, annualised −1,
   drawdown 1.5, finite Sharpe). Each fixture recomputes the z-path with an
   independent rolling mean/std before asserting the backtest's trades.
4. **Property checks**: every truncation of the bundled pair (T = 70 … 1500)
   yields only trades with `exit_day > entry_day`; the p-value is monotone
   in t; z-score warm-up length equals W−1 for a grid of windows; noiseless
   linear data is recovered exactly over a coefficient grid.
5. **Behavioural/statistical checks**: ADF strongly rejects on bundled
   white noise, fails to reject on the bundled random walk; Kalman with
   δ → 0 converges to static OLS β within a loose band; stops fire in
   the break regime; holdout OOS degradation present; streaming equals batch
   (Kalman 1e−12, `RollingZ` 1e−9). These show the code behaves as specified;
   they do not show a strategy works.
6. **Error paths**: every §6 validation raises its language's error type
   with the specified conditions — NaN/inf parameters, empty arrays, length
   mismatches in `walk_forward`, `beta <= 0`, dead configs, strict loader
   grammar (trailing content, hex, `_`, suffixes), mutated C++/Rust configs, collinear
   Johansen levels, non-finite matrices in the eigensolver, bad CSV/config
   content — plus the C++ test that previously skipped zero-day trades now
   asserts they never occur.

Python additionally cross-checks OLS, ADF critical values and Johansen
against statsmodels — in tests and `generate_data.py` only; the shipped
package depends on numpy alone. Counts at this release: Python 1248, C++
107, Rust 115, Java 109 tests, all passing.

**Continuous integration** (`.github/workflows/ci.yml`, new in v1.2.0): on
every push to `main` and pull request, Python 3.11 and 3.12 (ruff, mypy,
pytest); C++ with g++ and clang++ and a separate ASan/UBSan build; Rust with
`cargo fmt --check`, clippy at `-D warnings` and `cargo test --release
--locked`; Java through Maven; and a job that regenerates `data/` and runs
`data/check_regenerated.py`: CSVs and config must match byte for byte, and
golden values within each case's tolerance, because BLAS builds differ in the
last digits. The runner is pinned to `ubuntu-24.04`. Before v1.2.0 every cross-language
claim rested on manual runs.

## 7a. Research studies

`research/` sits outside the four-language contract. Each study has a
preregistration committed before it runs, a Python script using the library,
and aggregate results. Raw Yahoo prices stay in a git-ignored cache. Both
studies so far are null: [pairs](../research/real_data_2026_11/RESULTS.md) and
[sector-residual](../research/sector_residual_2026_11/RESULTS.md).

## 8. Performance notes

The workloads are small (T = 1500, n ≤ 3, k ≤ 4), so clarity beats
micro-optimisation everywhere; still, the shapes are worth recording:

- `ols`, `adf_test`, `engle_granger`, `half_life`: O(T·k²) — microseconds.
- `johansen_trace`: O(T·n²) for moments + O(n³) per Jacobi sweep; converges
  in ≤ ~10 sweeps on 3×3.
- `kalman_hedge`: strictly O(T) with constant 2×2 algebra per step;
  1500 steps in well under a millisecond in the native languages. The
  Python loop is interpreter-bound (~ms) — acceptable for a reference.
- `rolling_zscore`: O(T·W) as written (recomputing each window with plain
  sequential sums). The batch version stays this way on purpose: running sums
  accumulate rounding error and would break the 1e−10 golden z-scores and the
  bitwise trade decisions. `RollingZ` is the O(1)-per-update alternative for a
  live feed, accurate to 1e−9 against batch at the bundled length. At
  T·W ≈ 90k multiply-adds the batch loop is instantaneous.
- `backtest_pair`: single O(T) pass; the dominant cost is the z-score.

None of this is a tick-to-trade claim. The batch code allocates freely
(`Vec<Vec<f64>>`, `double[][]`), and there is no book or feed. `KalmanStep.update` and
`RollingZ.update` allocate nothing, which is the property a tick-level
consumer would need. Since v1.2.1 there are harnesses, `rust/src/bin/bench.rs`
and `java/.../Bench.java`, that time those two updates in batches of 64 and
report percentiles of the batch averages (median of three runs on one noisy
Windows machine, not JMH; `max` is operating-system jitter):

| ns per update | p50 | p99 | p99.9 | max |
|---|---:|---:|---:|---:|
| Rust `RollingZ` | 6.2 | 25.0 | 45.3 | 2684 |
| Rust `KalmanStep` | 17.2 | 29.7 | 232.8 | 2137 |
| Java `RollingZ` | 14.1 | 29.7 | 182.8 | 2477 |
| Java `KalmanStep` | 31.3 | 50.0 | 268.8 | 3155 |

Throughput on the bundled pair: Rust rolling z-score 21.9M bars/s and backtest
13.6M; Java 10.47M and 7.37M observations/s. Jacobi rotations in Rust and
Java update only rows and columns p and q in place (same operands, same order,
bit-identical). Not done: Java flat matrices and removing the boxed `Map` from
results, because both change public API; Rust OLS is unchanged because it is
at most 4x4. Reading the table is covered in LEARN.md section 10c.

Memory: everything is O(T) vectors; the largest allocation is the handful of
1500-length paths in the backtest result. No parallelism is used or needed;
determinism would forbid reduction reordering anyway.

## 9. Trading layer (Python only, paper only)

`statarb.trading` sits outside the four-language contract: it is not in
`API_SPEC.md`, not ported, and not covered by goldens. It has no network code,
holds no credentials and has no live-order path. `Broker` is the seam a real
adapter could implement; writing one is an owner decision. The layer exists to
show what sits between a signal and a P&L number: whole shares, partial fills,
spread and impact, leg risk, pre-trade limits, an independent ledger and an
audit trail.

Assumptions to keep in view: ADV and every cost are assumed (the CSV has no
volume); limits are checked at decision prices, so a later price move can take
exposure past one without a trade; one pair at a time.

**Order lifecycle** (`orders.py`; source `diagrams/order_lifecycle.mmd`). Any
move not drawn raises `IllegalTransition`.

```mermaid
stateDiagram-v2
    [*] --> NEW : submit
    NEW --> PARTIAL : fill of part of qty
    NEW --> FILLED : fill of all qty
    NEW --> CANCELLED : cancel
    NEW --> REJECTED : broker refuses, e.g. BORROW_UNAVAILABLE
    PARTIAL --> PARTIAL : another partial fill
    PARTIAL --> FILLED : last shares fill
    PARTIAL --> CANCELLED : cancel keeps filled_qty
    FILLED --> [*]
    CANCELLED --> [*]
    REJECTED --> [*]
    note right of PARTIAL : Any other move raises IllegalTransition.\nA cancel on a finished order changes nothing.\nResending a client id returns the original order.
```

**Paper flow** (`runner.py`; source `diagrams/paper_flow.mmd`). Per bar: borrow
accrues and the broker matches orders, the ledger books the fills and
reconciles against the broker, equity feeds the drawdown check, then the
signal runs. A reconciliation break trips the kill switch.

```mermaid
flowchart LR
    BAR["bar t: mid, adv, prev_close\nper symbol (Quote)"] --> SIG["RollingZ on the spread\nrule from BacktestConfig"]
    SIG -- "enter or exit at the close" --> EX["PairExecutor\nsizes child orders, paces the two legs"]
    EX -- "each child order" --> RK["RiskEngine.check\nreducing orders always pass"]
    RK -- "approved" --> BR["Broker.submit\nPaperBroker: next-bar fill,\nspread, impact, participation cap"]
    RK -- "rejected, reasons logged" --> EX
    BR -- "fills on a later bar" --> LG["Ledger.ingest\nindependent positions, cash, P&L"]
    LG --> REC["reconcile against broker\nbreak trips the kill switch"]
    LG --> EQ["equity -> RiskEngine.on_equity\ndrawdown latches the kill switch"]
    EQ -. "killed: flatten, no new risk" .-> EX
    SIG -. "signal event" .-> AUD[("AuditLog\nappend-only JSONL")]
    RK -. "risk_decision" .-> AUD
    LG -. "fill, reconciliation_break" .-> AUD
    NOTE["Paper only: no network code,\nno credentials, no live order path"]:::note
    classDef note fill:#fff8ec,stroke:#999,color:#333
```

**Pair entry and leg risk** (`execution.py`; source
`diagrams/pair_entry_leg_risk.mmd`). A leg may have at most
`ceil((m + b) x target)` shares sent, where `m` is the progress of the slower
leg and `b` is `max_imbalance_frac`. A leading leg therefore cannot run more
than `b` ahead, whatever the fill model does. After `timeout_bars` or a rejected
child the entry is cancelled and unwound with the same pacing. Exits and
unwinds never give up.

```mermaid
sequenceDiagram
    participant S as Signal
    participant E as PairExecutor
    participant R as RiskEngine
    participant B as PaperBroker
    S->>E: enter(target y -3097, x +5077)
    Note over E: slowest leg progress m = 0, bound b = 0.5\nsend up to ceil((m + b) x target) per leg
    E->>R: check y sell 1549, then x buy 2539
    R-->>E: approved both
    E->>B: submit both children
    B-->>E: bar t+1 y fills 1549, x not yet eligible (latency)
    B-->>E: bar t+2 x partial fill 1925 of 2539
    Note over E: progress y 0.50, x 0.38\nslowest m = 0.38, so allowed = ceil(0.88 x target)
    E->>R: check y sell 1174, x buy 1925
    R-->>E: approved
    E->>B: submit second children
    B-->>E: more partial fills each bar
    alt both legs reach 100 percent
        E-->>S: outcome FILLED
    else timeout_bars passed or a child rejected
        E->>B: cancel open children
        Note over E: UNWIND back to the start, paced the same way\nimbalance vs original target stays under 2b plus rounding
        E->>R: check reducing orders, always approved
        E->>B: submit unwind children
        B-->>E: filled, retried each bar until done
        E-->>S: outcome UNWOUND
    end
    Note over E,B: exit and unwind never give up:\na rejected child is retried on the next bar
```

**Risk** (`risk.py`). Exposure is measured at the worst case: position plus
every open order on the same side plus the new order, at the current mid. A
*reducing* order, one that can only move a position toward zero and cannot cross
it, always passes and is still logged, so risk can come off under the kill
switch, in a restricted name or on bad data. The drawdown switch latches and is
cleared only by `reset_kill_switch`, which needs an operator and a note.

**Ledger** (`ledger.py`). It rebuilds cash and positions from fills and its own
borrow accrual, so reconciling against the broker compares two separate
calculations. Identity at any mark: equity = initial + spread P&L at mid
- half-spread - impact - commission - borrow.

## 10. Research protocol and studies (outside the contract)

Source diagrams: `diagrams/research_protocol.mmd` and
`diagrams/sector_residual_pipeline.mmd`.

```mermaid
flowchart TD
    Q["Question and rule written down\nparameters round and fixed"] --> PR["PREREGISTRATION.md committed\nbefore any data is downloaded\ngit history is the timestamp"]
    PR --> DEV["Development period run\nfor pipeline bugs only\nparameters may not change"]
    DEV -- "bug found" --> FIX["fix code, record as deviation"]
    FIX --> DEV
    DEV -- "pipeline sound" --> HO["Holdout run ONCE\nno change since the development run"]
    HO --> GATES{"All pass gates met?\nSharpe, PSR, bootstrap CI,\ndrawdown, beta, capacity,\ndeflated Sharpe if trials > 1"}
    GATES -- "yes" --> WEAK["Report a pass as weak evidence\nsurvivor universe: needs point-in-time data"]
    GATES -- "no" --> NULL["Report a null, no tuning\nany variant is a NEW study\nwith a new preregistration"]
    NULL --> NEXT["Next study discloses this trial\ndeflated Sharpe counts it"]
```

The sector-residual signal, fixed in its preregistration:

```mermaid
flowchart TD
    PX["Daily adjusted closes\n63 stocks in 8 sector groups + 8 sector ETFs"] --> RET["Simple returns, last 60 days"]
    RET --> OLS["OLS: stock on its sector ETF\nbeta and residuals"]
    OLS --> X["X = cumulative sum of residuals"]
    X --> AR["AR(1) on X: X_k = a + b X_(k-1) + e"]
    AR --> EL{"0 < b < 1 and\nhalf-life <= 30 days?"}
    EL -- "no" --> NONE["ineligible: opens nothing,\nclose if open"]
    EL -- "yes" --> S["s = (X_t - m) / sigma\nm = a/(1-b), sigma = sd(e)/sqrt(1-b^2)"]
    S --> RULE{"flat: |s| >= 1.25 opens\nopen: |s| <= 0.5, 20 days,\nor ineligible exits"}
    RULE --> EXE["Signal at close t,\nexecute at close t+1"]
    EXE --> POS["One unit of stock hedged with\n-beta units of the sector ETF\nbeta fixed at entry"]
    POS --> PNL["Daily P&L = dir x (r_stock - beta x r_etf)\ncosts 5 bps stock, 2 bps ETF, borrow 1% a year"]
    PNL --> GATE["Holdout gates, run once\nresult: null, net Sharpe -0.32"]
```
