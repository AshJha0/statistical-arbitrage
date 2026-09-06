# LEARN — Statistical Arbitrage: Cointegration, Mean Reversion, and the Kalman Filter

This document teaches the theory behind the `statarb` toolkit: why price
levels and price differences behave so differently, why regressing one random
walk on another produces convincing nonsense, what cointegration actually is,
how the Engle–Granger and Johansen tests detect it, how mean reversion is
quantified through the half-life, why a Kalman filter tracks a drifting hedge
ratio when OLS cannot, and — the part textbooks skip — how pairs trades are
actually run, costed, and killed in practice.

All worked numbers come from the bundled data and match
`data/golden/golden.json`, so you can reproduce every figure by running the
demo or tests in any of the four languages.

---

## 1. Stationarity and unit roots: why prices are I(1) and returns are I(0)

### 1.1 Intuition

Ask two questions of any time series:

1. Does it have a "home" — a level it keeps returning to?
2. Does the uncertainty about its future value stop growing as you look
   further ahead?

For daily **returns** the answer to both is yes: tomorrow's return is drawn
from roughly the same distribution as today's, centred near zero, and knowing
today's return tells you almost nothing about next month's. Returns are
(approximately) **stationary**, denoted I(0) — integrated of order zero.

For **prices** the answer to both is no. A stock at 100 today has no
statistical pull back to 80 just because it traded there last year. Shocks are
*permanent*: a bad earnings day moves the level forever, it does not decay.
The variance of the k-day-ahead price grows linearly in k. Prices are
(approximately) **random walks** — I(1), integrated of order one — meaning
the *first difference* (the return) is stationary but the level is not.

### 1.2 The mathematics

An AR(1) process

```math
y_t = \phi\, y_{t-1} + \varepsilon_t, \qquad \varepsilon_t \sim \text{iid}(0, \sigma^2)
```

behaves in two qualitatively different regimes:

- **|φ| < 1 (stationary).** Back-substitution gives
  $y_t = \sum_{j\ge 0} \phi^j \varepsilon_{t-j}$: old shocks decay
  geometrically. Mean 0, variance $\sigma^2/(1-\phi^2)$ — finite and
  constant. The series mean-reverts.
- **φ = 1 (unit root).** Now $y_t = y_0 + \sum_{j=1}^{t}\varepsilon_j$:
  every shock persists forever, $\operatorname{Var}(y_t) = t\sigma^2$ grows
  without bound, and there is no mean to revert to. This is the random walk.

"y has a unit root" and "y is I(1)" both mean: the level is nonstationary but
Δy is stationary. Efficient-market logic explains why prices sit at (or very
near) φ = 1: if prices predictably reverted, the reversion would be traded
away. Returns, being the differences of an I(1) series, are I(0).

The entire machinery of classical regression — t-statistics with Student-t
distributions, standard errors shrinking at rate $1/\sqrt{T}$ — is built on
stationarity. Feed it I(1) data and it silently produces garbage, which is the
next section.

---

## 2. Spurious regression: the Granger–Newbold warning

### 2.1 The experiment

Granger and Newbold (1974) simulated two *independent* random walks and
regressed one on the other:

```math
y_t = a + b\, x_t + u_t
```

By construction the true b is 0 and nothing about x helps predict y. Yet in
their simulations the t-statistic on b exceeded 2 about **three-quarters of
the time**, R² was routinely large, and the Durbin–Watson statistic was near
zero (residuals massively autocorrelated). The regression "discovers" a strong
relationship that does not exist.

Why: two independent random walks both wander, and over any finite sample two
wandering paths will drift broadly up or down together (or apart) purely by
chance. OLS fits that common drift. The usual t-statistic assumes the residual
is stationary noise; here the residual is itself (close to) a random walk, so
the standard error is wildly understated. Formally, under I(1) regressors the
t-stat diverges at rate $\sqrt{T}$ instead of converging — more data makes
the illusion *stronger*, not weaker.

### 2.2 The bundled demonstration

`data/random_walks.csv` contains two independent geometric random walks. The
demo regresses `walk_a` on `walk_b`:

- slope t-statistic ≈ **8.4** — "significant" at any conventional level by
  the naive ±2 rule;
- ADF on `walk_a` levels: t = **−1.0945** against a 5% critical value of
  −2.8635 (MacKinnon 2010, T_eff = 1498; golden `adf_random_walk_crit5`) —
  the unit root is *not* rejected (golden `adf_random_walk_tstat`,
  `adf_random_walk_reject`);
- ADF on the regression residuals: t = **−1.154** against the 2-variable
  Engle–Granger 5% value of −3.3402 (T_eff = 1498) — not cointegrated
  (golden `eg_spurious_coint` = 0.0).

The residual test is the tell: the "relationship" leaves behind a
nonstationary residual, so it is spurious, and the toolkit refuses to
backtest the pair. Rule of thumb from the practice floor: **any regression of
levels on levels is presumed spurious until a residual stationarity test says
otherwise.**

---

## 3. Cointegration: when levels regression *is* legitimate

### 3.1 Concept

Two I(1) series $y_t$ and $x_t$ are **cointegrated** if some linear
combination

```math
s_t = y_t - \beta x_t
```

is I(0), i.e. stationary. Each series wanders freely, but they wander
*together*: the spread $s_t$ is tied to a fixed mean by some economic force —
shared cash flows (dual-listed shares), arbitrage (ETF vs basket, ADR vs
local line), triangle identities in FX, or substitutable businesses.

The classic metaphor (Murray, 1994): a drunk and her dog. The drunk follows a
random walk; the dog wanders too; but the leash guarantees the *distance
between them* is stationary even though neither position is. β is the leash's
"exchange rate", and $s_t$ is the leash tension you trade on.

Cointegration is what rescues levels regression: when the combination is
genuinely stationary, the OLS estimate of β is not only consistent but
**super-consistent** — it converges at rate $T$ instead of $\sqrt{T}$,
because the nonstationary variation in x is enormous relative to the
stationary residual.

Vocabulary: $(1, -\beta)$ is the **cointegrating vector**; the number of
independent stationary combinations among n series is the **cointegration
rank** r (0 ≤ r < n for I(1) systems).

### 3.2 Why traders care

If $s_t$ is stationary with mean μ, then whenever the spread stretches far
from μ, the *statistical expectation* is that it comes back. Long the cheap
leg, short the rich leg, sized so the position depends only on the spread —
that is a pairs trade. No view on the market, only on the leash.

---

## 4. The ADF test: mechanics and critical values

### 4.1 The regression

The Augmented Dickey–Fuller test asks: does ρ = 0 in

```math
\Delta y_t = c + \rho\, y_{t-1} + \sum_{i=1}^{p} \phi_i\, \Delta y_{t-i} + e_t
```

Subtracting $y_{t-1}$ from the AR representation maps "unit root" to
ρ = 0 and "stationary" to ρ < 0 (ρ = φ − 1 in the AR(1) case). The lagged
differences ("augmentation") soak up short-run autocorrelation so that $e_t$
is approximately white; this implementation pins **p = maxlag, default 1, no
automatic lag selection** — automatic selection (AIC etc.) is common but
makes results depend on the selector, which is fatal for cross-language
reproducibility. The statistic is the plain OLS t-ratio on ρ:

```math
t_{ADF} = \hat{\rho} / \operatorname{se}(\hat{\rho})
```

With T raw observations and lag p, the usable rows are t = p+1 … T−1, so
the effective sample is $T_{\text{eff}} = T - 1 - p$ with $k = 2 + p$
parameters.

### 4.2 Why special critical values

Under the null ρ = 0 the regressor $y_{t-1}$ is itself a random walk, so the
t-ratio does **not** follow a Student-t or normal distribution. Its limit is
the Dickey–Fuller distribution — a functional of Brownian motion, shifted
left and left-skewed. Using ±1.96 would reject a true unit root far too
often. The embedded values are MacKinnon's (2010) **response surfaces**
(constant, no trend): the asymptotic value plus finite-sample correction
terms in the effective regression size $T_{\text{eff}}$,

```math
cv(T_{\text{eff}}) = \beta_\infty + \frac{\beta_1}{T_{\text{eff}}}
  + \frac{\beta_2}{T_{\text{eff}}^2} + \frac{\beta_3}{T_{\text{eff}}^3}
```

whose asymptotic column $\beta_\infty$ is

| nvars | use                        |    1%    |    5%    |   10%    |
|-------|----------------------------|----------|----------|----------|
| 1     | plain ADF on one series    | −3.43035 | −2.86154 | −2.56677 |
| 2     | EG residual test, 2 vars   | −3.89644 | −3.33613 | −3.04445 |
| 3     | EG residual test, 3 vars   | −4.29374 | −3.74066 | −3.45218 |

(the full coefficient table is in API_SPEC §2). The corrections matter in
exactly the samples desks use: the 5% value for one series is −2.8615
asymptotically but **−2.92 at T_eff = 50 and −3.02 at T_eff = 20**; for the
2-variable EG case −3.336 becomes −3.46 at T_eff = 50. A rolling 60-day pair
screen that used the asymptotic numbers would declare cointegration for
residual t-statistics in (−3.46, −3.34] that the finite-sample test rejects —
systematic over-detection. The toolkit therefore always evaluates the
surface at the actual regression size, and refuses tests with fewer than 20
effective observations (the smallest sample in MacKinnon's simulations)
rather than extrapolate.

You reject the unit root when the t-stat is *more negative* than the critical
value. Note −2.86 vs −1.65 for a one-sided normal test at 5%: the unit-root
null is much harder to reject than a textbook zero-coefficient null.

Rows 2 and 3 exist because of a subtlety covered in §5: testing *estimated*
residuals requires still-more-negative thresholds.

### 4.3 The pinned p-value approximation

The exact MacKinnon (1994) response-surface p-value is deliberately not
implemented. Instead the p-value is a documented approximation:
piecewise-linear interpolation of the significance level through the knots
(cv₁%, 0.01), (cv₅%, 0.05), (cv₁₀%, 0.10) evaluated at the test's own
T_eff, extrapolating with the edge slopes, clamped to [1e−4, 0.9999].
Worked example on the in-sample EG residuals (t = −5.8244, nvars = 2,
T_eff = 898, so cv₁% = −3.9087 and cv₅% = −3.3429): t ≤ cv₅%, so use the
1%–5% segment with slope (0.05 − 0.01)/(−3.3429 + 3.9087) = 0.0707 per
unit t; p = 0.01 + 0.0707 × (−5.8244 + 3.9087) = −0.126 → clamped to
**0.0001**, which is what the demo prints. On the bundled random walk
(t = −1.0945, T_eff = 1498, cv₅% = −2.8635, cv₁₀% = −2.5678) the 5%–10%
segment gives p = 0.05 + 0.169 × (−1.0945 + 2.8635) = **0.3491** (golden
`adf_random_walk_pvalue`). The *verdict* (`reject`) never uses the
p-value — it compares the t-stat to the finite-sample table directly, so
the approximation cannot change any decision at the three supported
levels.

### 4.4 Power: the test's real weakness

The ADF test has notoriously low power against alternatives close to the
null. A stationary series with φ = 0.99 (half-life ~69 days) looks almost
identical to a random walk in a few years of daily data; the test will often
fail to reject even though the series mean-reverts. Practical consequences:
"fail to reject" is *not* "is a random walk"; and slowly-reverting spreads —
often the ones that look most tradeable on a chart — are precisely where the
test is least conclusive. More data helps, but regime changes (§9) punish
long samples.

---

## 5. Engle–Granger two-step: the workhorse, and its limits

### 5.1 The procedure

1. **Hedge regression.** Estimate $y_t = a + b\,x_t + e_t$ by OLS. Under
   cointegration, $\hat b$ is super-consistent for β.
2. **Residual ADF.** Run the §4 ADF (same fixed lag) on $\hat e_t$. If the
   residuals are stationary, y and x are cointegrated; $\hat e_t$ *is* the
   tradeable spread.

Crucial detail — the critical values change. OLS *chooses* $(\hat a,\hat b)$
to make the residuals as small (and as stationary-looking) as possible.
Testing these optimised residuals with the plain 1-variable table would
over-reject: you would "find" cointegration too often. Hence the nvars = 2
row (−3.90/−3.34/−3.04), and the nvars = 3 row when the first-step
regression has two regressors. This is the single most common implementation
mistake in home-grown EG tests.

### 5.2 Worked example (bundled pair, first 900 days)

- Step 1: $y = 17.3603 + 1.6389\,x$, R² = 0.9793 (true generator beta 1.7 —
  OLS lands close but not exactly, as it should on one noisy sample).
- Step 2: ADF(resid, maxlag 1) t = **−5.8244** (golden
  `adf_resid_insample_tstat`), far below the 2-variable 5% value −3.3429 at
  T_eff = 898 (golden `adf_resid_insample_crit5`) → cointegrated at 5%
  (golden `eg_pair_insample_coint` = 1.0).

### 5.3 Limitations (honest list)

- **Normalisation dependence.** Regressing y on x versus x on y gives
  different hedge ratios (they are not reciprocals unless R² = 1) and can,
  in marginal cases, give different verdicts. Johansen treats the system
  symmetrically.
- **One cointegrating vector only.** With n ≥ 3 series there can be up to
  n − 1 independent stationary combinations; EG can only ever exhibit one.
- **Two-step estimation error.** The residual test conditions on an
  estimated β; the correction via shifted critical values is asymptotic.
- **No dynamics.** EG says nothing about *how fast* the spread reverts or
  which leg does the adjusting — the VECM (§6) does.
- **Low power inherited from ADF**, amplified by estimation error.

For a 2-asset pairs screen EG is usually fine and is what most desks run
first; Johansen is the cross-check and the multi-asset tool.

---

## 6. Johansen's approach: VECM and rank tests, accessibly

### 6.1 From VAR to VECM

Stack n price series into $Y_t$. The Vector Error-Correction Model with one
lagged difference (the case pinned here) is

```math
\Delta Y_t = c + \Gamma\, \Delta Y_{t-1} + \Pi\, Y_{t-1} + e_t
```

Everything hinges on the n×n matrix Π, the "error-correction" term:

- **rank(Π) = 0**: Π = 0, levels never feed back into changes — no
  cointegration, the system is n unrelated random walks (in this sense).
- **rank(Π) = n**: the levels themselves are stationary (no unit roots
  anywhere).
- **0 < rank(Π) = r < n**: the interesting case. Π factors as
  $\Pi = \alpha\beta'$ where β's r columns are the cointegrating vectors
  (the stationary combinations $\beta'Y_{t-1}$) and α's entries are
  **adjustment speeds** — how hard each series is pulled back when the
  combinations stray. The cointegration rank *is* the rank of Π.

### 6.2 How the test works (the accessible version)

Johansen's procedure is a canonical-correlation analysis between "today's
changes" and "yesterday's levels", after removing what short-run dynamics
already explain:

1. Partial the constant and $\Delta Y_{t-1}$ out of both $\Delta Y_t$ and
   $Y_{t-1}$, leaving residual blocks $R_0$ and $R_1$.
2. Form moment matrices $S_{ij} = R_i'R_j/T_{\text{eff}}$ and solve the
   generalised eigenvalue problem
   $|\lambda S_{11} - S_{10}S_{00}^{-1}S_{01}| = 0$.
3. The eigenvalues $1 > \lambda_1 \ge \dots \ge \lambda_n \ge 0$ are squared
   canonical correlations: $\lambda_i$ measures how much predictive power the
   i-th level-combination has for changes. A λ near 0 means that combination
   is useless for prediction — it is a pure random-walk direction. A
   meaningfully positive λ means that combination pulls the system — it is a
   cointegrating direction. Its eigenvector is the cointegrating vector.
4. **Trace statistic** for each hypothesised rank r:

   ```math
   \text{trace}(r) = -T_{\text{eff}} \sum_{i=r+1}^{n} \ln(1-\lambda_i)
   ```

   which tests "rank ≤ r" against "rank > r" using the smallest n − r
   eigenvalues. Test sequentially from r = 0 upward; the estimated rank is
   the first r whose trace statistic falls below its critical value
   (Osterwald–Lenum 1992, 95%, unrestricted constant: 3.76 / 15.41 / 29.68
   for n − r = 1/2/3).

Numerically, this implementation makes the problem symmetric via a Cholesky
factor of $S_{11}$ and diagonalises with a native cyclic Jacobi rotation
(2×2/3×3 matrices, so Jacobi converges in a few sweeps) — pinned so all four
languages produce bit-comparable eigenvalues.

### 6.3 Worked example

In-sample equity pair (900 days): trace(0) = **37.572** > 15.41 → reject
rank 0; trace(1) = **3.460** < 3.76 → cannot reject rank ≤ 1. Estimated
rank **1** with cointegrating vector (1, −1.6455) — one stationary spread,
hedge ratio agreeing with EG's 1.6389 to ~0.4% (golden
`johansen_pair_trace_r0/_r1`, `johansen_pair_rank`).

FX triangle in *logs*: eurusd, gbpusd, eurgbp with basket = eurgbp×gbpusd.
Since log(eurusd) ≈ log(eurgbp) + log(gbpusd) + stationary noise, the
theoretical cointegrating vector is (1, −1, −1) — and the test recovers rank
1 with exactly that leading vector. This is the FX convention worth
memorising: **multiplicative identities (cross rates, baskets) become linear
cointegrating relations only in logs**, so FX stat-arb runs on log prices,
while equity pairs are commonly run on raw prices (as here) or log prices
depending on desk convention.

### 6.4 Honest limitations of this "Johansen-lite"

Fixed k = 1 lagged difference, only the unrestricted-constant case, only the
95% trace column, no small-sample (Bartlett) correction, no max-eigenvalue
variant, no p-values, n ≤ 3. Real research code (e.g. statsmodels
`coint_johansen`) handles the five deterministic cases and general lags. The
point here is the mechanism, pinned tightly enough to reproduce across
languages.

---

## 7. Mean reversion and the half-life: the OU connection

### 7.1 From AR(1) to Ornstein–Uhlenbeck

A stationary spread is naturally modelled in continuous time as an
Ornstein–Uhlenbeck process:

```math
ds_t = \kappa(\mu - s_t)\,dt + \sigma\,dW_t
```

κ is the reversion speed. Sampled daily, an OU process is *exactly* an
AR(1): $s_{t+1} = \mu(1-e^{-\kappa}) + e^{-\kappa}s_t + \eta_t$. So fit the
pinned regression

```math
\Delta s_t = c + \rho\, s_{t-1} + e_t \quad\Rightarrow\quad 1+\rho = e^{-\kappa}
```

The deterministic part of the OU solution decays as $e^{-\kappa t}$; the time
for half the displacement to decay is

```math
t_{1/2} = \frac{\ln 2}{\kappa} = -\frac{\ln 2}{\ln(1+\rho)}
```

which is the pinned formula. Guards: ρ ≥ 0 means no mean reversion at all →
half-life +∞ (do not trade this); 1 + ρ ≤ 0 means an overshooting flip →
returned as 0.

### 7.2 Worked example and use

In-sample EG residuals: $\hat\rho = -0.0793$, so
hl = −ln 2 / ln(0.9207) = **8.39 days** (golden `halflife_insample_spread`).

The half-life is the desk's sizing and patience parameter all in one:

- **Signal window**: the rolling z-score window should be a few half-lives
  (here W = 60 ≈ 7 half-lives — comfortably long enough to estimate the
  mean and dispersion the spread reverts to).
- **Expected holding period**: entries at |z| = 2 exiting at |z| = 0.5
  should take on the order of two half-lives — consistent with the ~10-day
  average holds the in-sample backtest reports.
- **Sanity filter**: half-lives of hours are usually microstructure noise
  and uncapturable after costs; half-lives of months tie up capital and are
  statistically indistinguishable from non-reversion in samples you can
  trust. Most equity pairs desks want days-to-weeks.

Caveat: the half-life estimate inherits AR(1)-fit noise and is biased short
in small samples; treat it as an order of magnitude, not a forecast.

---

## 8. Kalman filtering for a dynamic hedge ratio

### 8.1 Why a static β is not enough

OLS assumes one β for all time. Real hedge ratios drift: capital structures,
index weights, business mix, and market regimes change. The bundled pair
makes this brutal: true β is 1.7 until day 1049, ramps to 1.1 by day 1080,
and stays there. Full-sample OLS answers **1.5178** (golden `ols_pair_beta`)
— a number that was *never* the true beta on any single day, with full-sample
R² collapsing to 0.139. Any spread built from it is mis-hedged everywhere.

### 8.2 State-space formulation

Treat the intercept and hedge ratio as an unobserved state that random-walks:

```math
\text{state: } s_t = \begin{pmatrix}\alpha_t\\ \beta_t\end{pmatrix},\quad
s_t = s_{t-1} + w_t,\quad \operatorname{Cov}(w_t)=Q
```
```math
\text{observation: } y_t = H_t s_t + \varepsilon_t,\quad
H_t = [\,1,\; x_t\,],\quad \operatorname{Var}(\varepsilon_t) = r
```

Each day the filter carries a Gaussian belief (mean s, covariance P) about
(α, β) and updates it in two moves:

```text
predict:  P_pred = P + Q                    # belief diffuses: the world may have moved
innovate: e = y_t − H s                     # surprise in today's price
          S = H P_pred H' + r               # how surprising is surprising?
gain:     K = P_pred H' / S                 # trust in the surprise
update:   s ← s + K e
          P ← (I − K H) P_pred (I − K H)' + r K K'    # Joseph form
```

The gain K is the whole story: it balances *state uncertainty* (P + Q)
against *observation noise* (r). Noisy prices → small K → smooth, slowly
adapting β. Large Q → big K → nervous, fast-adapting β. The pinned
RiskMetrics-style scaling

```math
Q = \frac{\delta}{1-\delta} I_2, \qquad 0 < \delta < 1
```

collapses that trade-off into one dial: δ → 0 makes the state noise vanish
and the filter degenerates into **recursive least squares** — it converges
to the static OLS β (a property the test suite checks). Larger δ forgets the
past faster; δ = 1e−4 with r = 1e−3 (the bundled config) tracks the day-1050
break within ~50 days while staying stable before it. The prior is pinned at
s₀ = 0, P₀ = 1000·I — diffuse enough that the first few observations
dominate it.

Worked numbers (golden `kalman_beta_*`): filtered β = 1.69206 at day 500
(true 1.7), 1.10012 at day 1200 (true 1.1), 1.15319 at day 1499. Tracking
RMSE from day 300: **0.024 for the Kalman filter vs 0.288 for static OLS**.

### 8.3 Why the Joseph form is pinned

The textbook covariance update $P = (I-KH)P_{\text{pred}}$ is algebraically
correct but numerically fragile: it subtracts nearly-equal quantities, and
rounding can leave P asymmetric or indefinite (a negative eigenvalue), after
which gains misbehave and the filter can diverge. The Joseph form

```math
P = (I-KH)\,P_{\text{pred}}\,(I-KH)' + K r K'
```

is a sum of a congruence of a PSD matrix and a PSD rank-one term — it stays
symmetric PSD under rounding *by construction* (the implementation also
re-symmetrises P := (P+P')/2 to kill the last bits of asymmetry). It costs a
few extra flops on a 2×2 matrix, i.e. nothing, and keeps the four ports
within the 1e−8 golden tolerance over 1500 filtering steps even though each
accumulates rounding differently. The test suite verifies PSD-ness via the
2×2 closed-form eigenvalues on the final posterior.

### 8.4 Innovation z-scores

The filter emits $z_t = e_t / \sqrt{S_t}$ — each day's price surprise in
units of its own predicted standard deviation. This is an alternative,
self-normalising trading signal: unlike the rolling z-score it needs no
window, adapts its scale automatically, and reflects the *current* hedge. Its
cost is model dependence: mis-set δ or r and the z-scores are mis-scaled.

---

## 9. Pairs trading in practice

### 9.1 The signal pipeline as implemented

1. Estimate (α, β) in-sample (EG on the first 900 days: 17.3603, 1.6389).
2. Spread $s_t = y_t - (\alpha + \beta x_t)$.
3. Rolling z-score, window W = 60, **current observation included**,
   std with ddof = 1, NaN for the first W−1 days and for zero-variance
   windows (golden `zscore_day200` = −1.17204).
4. State machine: enter short spread at z ≥ 2 (long at z ≤ −2), exit at
   |z| ≤ 0.5, stop at |z| ≥ 4. If |z| ≥ 4 while flat: **do not enter** — a
   4σ dislocation is more likely a break than a bargain.
5. **Execution lag (pinned): a signal at the day-t close trades at the day
   t+1 close.** Backtests that fill at the same close they signalled on
   embed look-ahead and can turn a losing strategy into a paper star.
   Position sizing is equal-dollar per leg (capital/2 each side), exactly
   dollar-neutral at entry; β lives in the *signal*, not the leg sizes, and
   quantities are then frozen for the life of the trade. A signal on the
   last available close is *discarded*, never executed and force-closed at
   the same close: a rolling daily re-run therefore never reports a
   phantom zero-day round trip that paid two costs.

### 9.2 Costs

Each execution pays (cost_bps + slippage_bps) on traded notional — here
1.5 bps combined, charged on ~capital of notional at entry and again at exit,
so ~3 bps of capital per round trip. That sounds small; it is not. The
in-sample backtest turns over 41.9× capital, so costs consume ~0.6% of
capital — roughly a tenth of the gross P&L. Stat-arb strategies live or die
on the cost line: gross Sharpe is easy, net Sharpe is the job. Real desks
additionally face borrow fees on the short leg, ticket costs, and market
impact that grows with size (§10).

### 9.3 Walk-forward honesty

Fitting and evaluating on the same data flatters everything: β is tuned to
the very sample it trades. The pinned protocol estimates the hedge on the
first 60% and trades the last 40% with parameters frozen; the OOS z-score
warms up inside its own segment so no information crosses the split.

Bundled result: in-sample Sharpe **+1.49** becomes out-of-sample **−1.37**.
Part of that is ordinary estimation decay; most of it is the next subsection.

### 9.4 Regime breaks: why pairs die

The generator breaks the pair at day 1050 (β ramps 1.7 → 1.1; the spread's
AR coefficient rises from 0.92 to 0.995, i.e. half-life ~8 days → ~138
days: economically, reversion stops). This is the canonical failure mode of
pairs trading in the wild — mergers, delistings, index rebalances, business
model divergence (think Kodak/Fuji), or a funding regime change sever the
leash. The spread does not revert to the old mean because *there is no old
mean any more*; a mean-reversion system will keep averaging into it.

What the toolkit shows honestly:

- The static-hedge backtest across the break loses money (full-sample golden
  metrics: total return **−1.91%**, Sharpe **−0.137**, max drawdown
  **14.26%**, 31 trades) — the stop-loss at |z| = 4 triggers in the broken
  regime and caps the damage, which is exactly its job.
- The Kalman filter *adapts* β through the break (§8) — the right tool if
  the relationship survives with a new ratio; no hedge-ratio filter helps if
  the relationship is simply gone.

Diagnostics that flag a dying pair before the P&L does: half-life estimates
drifting up on rolling windows, rolling ADF t-stats decaying toward zero,
Kalman innovation z-scores running persistently one-sided, and — most
reliable of all — knowing the corporate/news calendar of both legs.

### 9.5 How a desk would use this

A realistic daily loop around the toolkit, in the order the functions are
meant to be called:

1. **Universe and hygiene.** Load adjusted closes (splits, dividends,
   corporate actions applied upstream), one row per trading day, aligned
   across both legs. `load_csv` refuses non-finite cells and a non-monotone
   `day` column; it does *not* fill gaps — that is your data pipeline's job.
2. **Screen candidates.** For each economically linked pair run
   `engle_granger` on a fitting window of at least 250 days (the finite-sample
   critical values are used at whatever T_eff you supply, but power below a
   year of data is poor — §4.4) and demand `cointegrated(0.05)`; confirm
   with `johansen_trace` rank ≥ 1, and require `half_life` in a tradeable
   band (days to a few weeks — both sentinels `+inf` and `0.0` mean "skip").
   Screening thousands of pairs guarantees false positives (§11.8): rank by
   economic story first, statistics second.
3. **Fix the hedge and signal.** Use the in-sample (α, β) with
   `rolling_zscore(spread(...), W)` where W ≈ several half-lives, or run
   `kalman_hedge` and use its `beta` path if the ratio is expected to drift.
   Never re-estimate the hedge on the data you evaluate.
4. **Evaluate honestly.** `walk_forward` with the same `BacktestConfig` you
   would trade: costs at or above what your broker charges, `stop_z` set
   where a dislocation is more likely a break than a bargain. Judge the
   out-of-sample segment only; the bundled example's in-sample Sharpe +1.49
   versus out-of-sample −1.37 is the lesson.
5. **Run and monitor.** Each day append the new close, re-run the z-score,
   act on the *pending* order the state machine emits for the next close,
   and re-run the diagnostics on a trailing window: rolling ADF t drifting
   toward zero, half-life lengthening, Kalman innovation z-scores running
   one-sided, or a stop firing all say "the leash may have snapped".
6. **Know the limits.** Returns are on gross notional (no compounding, no
   financing, no borrow), one-day execution lag only, no calendar, no
   impact model; size for the fact that everyone else with the same
   textbook holds the same spread (§10).

### 9.6 Equity vs FX conventions

- **Equity pairs**: price levels (or log prices) of two stocks; borrow
  availability and fees on the short leg matter; dividends and corporate
  actions must be adjusted or they masquerade as spread jumps; dollar
  neutrality is the standard sizing (as pinned here); beta/sector
  neutrality is the institutional refinement.
- **FX**: cointegration lives in **log** rates because cross-rate identities
  are multiplicative (§6.3); the "pair" is often one rate vs a synthetic
  basket; costs are spreads and rollover/carry rather than borrow; notionals
  are naturally symmetric so dollar-neutrality is automatic, but carry can
  dominate a slow spread's P&L and must be modelled.

---

## 10. Risk management of statistical arbitrage

**Stop-losses.** A z-score stop (|z| ≥ 4 here) is a model-disagreement
trigger, not a price stop: it fires when the spread does something the
stationarity assumption says should almost never happen, which is evidence
the assumption is wrong. The bundled break regime shows stops converting an
unbounded structural loss into a capped one. The cost is whipsaw: some stops
fire at what was, in hindsight, the best entry ever. Accept it; the
alternative is Kodak/Fuji with leverage. Time stops (exit after k×half-life
regardless) are a common complement.

**Capacity.** Each pair's edge is a few basis points of reversion on limited
liquid size; impact costs grow superlinearly in participation. Scaling a
stat-arb book means more pairs, not bigger pairs — and the marginal pair is
always worse than the average one. Turnover × impact is the binding
constraint long before Sharpe is.

**Crowding — August 2007.** During 6–9 August 2007, quantitative
equity-market-neutral funds suffered enormous, highly correlated losses over
a few days while index markets were calm — the "quant quake". The accepted
post-mortem (Khandani–Lo, 2007): many funds held overlapping
factor/stat-arb portfolios; one or more large players deleveraged rapidly
(likely to meet credit-related losses elsewhere); their unwinding moved
prices *against* everyone holding the same spreads; mark-to-market losses
triggered further deleveraging at other funds — a fire-sale spiral in
positions, not in fundamentals. Many spreads snapped back within days;
funds that had cut risk at the bottom locked in the losses. Lessons that
are now doctrine: your true risk includes *who else holds your trade*;
leverage plus crowding converts a stationary spread into a run-prone one;
deleveraging policy (how fast, in what order) matters as much as entry
signals; and backtests on tranquil samples cannot show any of this.

**Model risk controls in this codebase**, in miniature: refusing to trade
non-cointegrated pairs (the spurious case is diagnosed and no backtest is
run), walk-forward evaluation as the default honesty check, stop-losses in
the state machine, and an accounting identity test (equity change ≡ Σpnl −
Σcosts at 1e−10) so the backtester itself cannot quietly leak money.

---

## 11. Common pitfalls and numerical issues

1. **Standard t-tables on unit-root or EG statistics.** Both the ADF and the
   EG residual test need their own (more negative) critical values; using
   ±1.96, or using the 1-variable ADF row for estimated residuals,
   over-detects stationarity/cointegration.
2. **Same-close execution.** Signal at close t, fill at close t — a subtle
   look-ahead worth several Sharpe points on mean-reversion systems. Pinned
   here as next-close execution.
3. **Z-score window off-by-one.** Including vs excluding the current
   observation changes every signal date; this contract pins *inclusive*,
   ddof = 1. Cross-language golden tests at 1e−10 catch any drift.
4. **Plain Kalman covariance update.** $(I-KH)P$ loses PSD-ness under
   rounding; symptoms are negative variances and exploding gains. Use Joseph
   form + symmetrisation (pinned), verify PSD in tests.
5. **Normal equations without a condition guard.** $X'X$ squares the
   condition number; with a constant regressor next to the intercept it is
   exactly singular. The pinned guard (cond ≤ 1e12) turns garbage into a
   clean error. Fine for k ≤ 4; use QR for big design matrices.
6. **Zero-variance windows.** A flat spread window makes the z-score 0/0;
   the pinned guard emits NaN (no signal), never ±inf into the state
   machine.
7. **Half-life edge cases.** ρ ≥ 0 must map to +∞ (and "do not trade"), not
   to a negative half-life from blindly evaluating the formula.
8. **Survivorship and pair-selection bias.** Screening thousands of pairs
   for the best in-sample ADF *guarantees* false discoveries (multiple
   testing); demand economic linkage first, statistics second, and validate
   out-of-sample.
9. **Ignoring the constant/trend specification.** ADF and Johansen verdicts
   change with the deterministic terms included; this toolkit pins
   constant/no-trend (ADF) and unrestricted constant (Johansen) — know what
   your library defaults to.
10. **Log vs level inconsistency.** FX identities require logs; mixing a
    log-based signal with level-based sizing (or vice versa) mis-scales
    every position.
11. **Asymptotic critical values on short windows.** The 5% ADF value is
    −2.86 only as T → ∞; at 50 observations it is −2.92, at 20 it is −3.02,
    and the EG 2-variable value moves from −3.34 to −3.46 at T = 50. Rolling
    screens on 60-day windows with the asymptotic table over-detect
    cointegration; the toolkit evaluates the MacKinnon (2010) surface at the
    actual sample size and refuses T_eff < 20.
12. **Unsorted or duplicated dates.** A CSV exported in a different order or
    with a duplicated row after a merge silently changes every lag,
    z-score and fill. The loaders refuse a `day` column that is not strictly
    increasing; equal spacing is still the caller's responsibility.
13. **End-of-data phantom trades.** A backtester that executes a
    last-day entry and force-closes it at the same close reports an extra
    losing round trip on every rolling re-run. The pinned rule discards a
    pending entry at the end of data.

---

## 12. Interview-style Q&A

**Q1. Why can't you regress one stock price on another and trust the
t-statistic?**
Both series are I(1). With nonstationary regressor and residual, the OLS
t-statistic diverges with sample size instead of settling into a t
distribution — two independent random walks produce |t| > 2 roughly 75% of
the time (Granger–Newbold). The regression is spurious unless the residual is
stationary, i.e. unless the series are cointegrated — which is exactly what
the EG residual test checks.

**Q2. Define cointegration in one sentence and give the trading
interpretation.**
Two (or more) I(1) series are cointegrated if some fixed linear combination
of them is I(0); that combination is a stationary spread whose deviations
from its mean are statistically expected to revert — the tradeable object in
pairs trading.

**Q3. Why does the Engle–Granger residual test use different critical values
from the plain ADF test?**
Because the residuals are estimated: OLS picks the coefficients that make the
residuals look as stationary as possible, biasing the ADF statistic downward.
Compensating requires more negative critical values, growing with the number
of estimated variables (asymptotically −2.862 → −3.336 → −3.741 at 5% for
1/2/3 variables, and more negative still in finite samples).

**Q4. What does the rank of Π in a VECM tell you, and how does the trace test
find it?**
rank(Π) = r is the number of independent cointegrating relations: Π = αβ′
with β holding the r stationary combinations and α the adjustment speeds. The
trace test orders the canonical-correlation eigenvalues and tests rank ≤ r
via $-T\sum_{i>r}\ln(1-\lambda_i)$ sequentially from r = 0; the first
non-rejection gives the rank.

**Q5. Your spread has AR(1) coefficient 0.92 on daily data. Half-life?**
ρ = φ − 1 = −0.08; hl = −ln 2 / ln(0.92) ≈ 8.3 days. (The bundled pair's
in-sample spread: 8.39 days.) Trade horizon: entries should resolve in
roughly one to three weeks.

**Q6. Why prefer a Kalman filter to rolling OLS for the hedge ratio?**
Rolling OLS gives every observation in the window equal weight and full
weight zero outside it — a step-shaped memory with a hard tuning choice, and
it re-estimates from scratch each day. The Kalman filter is recursive
(O(1) per day), weights the past with geometric decay controlled by one
parameter (δ via Q = δ/(1−δ)·I), provides an uncertainty estimate P around
the hedge, and emits self-normalised innovation z-scores. As δ → 0 it
recovers static OLS, so rolling/static OLS is a special case, not a
competitor.

**Q7. What is the Joseph form and why bother on a 2×2 problem?**
$P = (I-KH)P_{\text{pred}}(I-KH)' + KrK'$ — algebraically identical to
$(I-KH)P_{\text{pred}}$ but structurally symmetric PSD, so floating-point
rounding cannot produce negative variances that destabilise the gain. On 2×2
the extra cost is nil and it makes cross-language agreement at 1e−8
achievable, which the plain form does not guarantee.

**Q8. Why does the backtest execute at the *next* close, and what happens if
you don't?**
The signal uses the day-t close, so the earliest realistic fill is after that
close — pinned here as the t+1 close. Same-close fills let the strategy trade
on information contemporaneous with its price, a look-ahead that
systematically flatters mean-reversion (you always "catch" the extreme tick).
It is one of the most common sources of unreproducible backtest Sharpe.

**Q9. Your pair passed EG at 1% and then lost money for six months straight.
Name three explanations and the diagnostic for each.**
(a) Regime break — the cointegrating relation ended (merger, business
change): rolling ADF t decays toward zero, half-life estimates blow up,
Kalman innovations run one-sided. (b) Data-mined false positive — the pair
was selected from a large screen: no economic linkage story, OOS behaviour
inconsistent with in-sample from day one. (c) Crowding/deleveraging — the
spread is statistically intact but other holders are unwinding: widening
happens fast, on volume, correlated with other quant spreads (August 2007
pattern). Stops address (a) and (c); selection discipline addresses (b).

**Q10. What actually happened in August 2007, and what is the portable
lesson?**
Over roughly 6–9 August 2007, market-neutral quant funds with overlapping
holdings suffered large simultaneous losses as one or more big players
rapidly deleveraged, pushing common spreads against every similar book and
forcing further unwinds — a liquidity spiral invisible to single-strategy
backtests; many spreads reverted within days. Portable lesson: crowded
positions carry a risk factor that is not in your covariance matrix — the
other holders' leverage and funding — so size assuming you may have to exit
alongside everyone who holds what you hold.

---

## 13. Further reading

- Granger, C. & Newbold, P. (1974). "Spurious Regressions in Econometrics."
  *Journal of Econometrics* 2, 111–120. — the two-random-walks experiment.
- Engle, R. & Granger, C. (1987). "Co-integration and Error Correction:
  Representation, Estimation and Testing." *Econometrica* 55, 251–276.
- Johansen, S. (1991). "Estimation and Hypothesis Testing of Cointegration
  Vectors in Gaussian Vector Autoregressive Models." *Econometrica* 59,
  1551–1580.
- Osterwald-Lenum, M. (1992). "A Note with Quantiles of the Asymptotic
  Distribution of the ML Cointegration Rank Test Statistics." *Oxford
  Bulletin of Economics and Statistics* 54, 461–472. — the trace tables.
- MacKinnon, J. G. (2010). "Critical Values for Cointegration Tests."
  Queen's Economics Department Working Paper 1227. — the finite-sample
  response-surface critical values embedded verbatim in every port.
- MacKinnon, J. G. (1994). "Approximate Asymptotic Distribution Functions
  for Unit-Root and Cointegration Tests." *Journal of Business & Economic
  Statistics* 12(2), 167–176. — the exact p-value surface this package
  replaces by piecewise-linear interpolation.
- Hamilton, J. (1994). *Time Series Analysis*. Princeton. — chapters 15–20
  cover unit roots through VECMs rigorously.
- Harvey, A. (1989). *Forecasting, Structural Time Series Models and the
  Kalman Filter*. Cambridge. — state-space methods, including numerical
  stability.
- Vidyamurthy, G. (2004). *Pairs Trading: Quantitative Methods and
  Analysis*. Wiley.
- Chan, E. (2013). *Algorithmic Trading: Winning Strategies and Their
  Rationale*. Wiley. — half-life estimation and Kalman pairs in practice.
- Avellaneda, M. & Lee, J.-H. (2010). "Statistical Arbitrage in the US
  Equities Market." *Quantitative Finance* 10, 761–782.
- Khandani, A. & Lo, A. (2007). "What Happened to the Quants in August
  2007?" *Journal of Investment Management* 5(4). — the quant quake
  post-mortem.
- Murray, M. (1994). "A Drunk and Her Dog: An Illustration of Cointegration
  and Error Correction." *The American Statistician* 48, 37–39.
