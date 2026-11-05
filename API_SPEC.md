# statarb — API Specification (pinned numerical contract)

This document is the cross-language contract for the `statistical-arbitrage`
repository (cointegration, Kalman filter, pairs backtest).
Every implementation (Python `statarb`, C++ `statarb::`, Rust crate
`statarb`, Java `com.quant.statarb`) MUST reproduce the algorithms below
exactly as pinned; agreement is enforced by `data/golden/golden.json`.
The Python package under `python/src/statarb/` is the reference.

Notation: series are 1-D arrays of doubles indexed `t = 0..T-1`.
"Pinned" means: not an implementation choice — do it this way.

---

## 1. OLS (`ols`)

Model: `y = X b + e`, optional intercept prepended as the FIRST column.

Pinned solution method: **normal equations with an LU solve** —

```
b      = solve(X'X, X'y)               # no QR, no SVD
sigma2 = RSS / (n - k),  RSS = e'e,  e = y - X b
cov(b) = sigma2 * inv(X'X)
se_i   = sqrt(cov_ii),   t_i = b_i / se_i        # plain SEs, no HAC
R^2    = 1 - RSS/TSS
TSS    = sum((y - mean(y))^2)  with intercept; y'y without
```

Validation (raise/throw/Err on violation):
- `len(y) == rows(X)` else "series length mismatch";
- all inputs finite (no NaN/inf);
- `n > k` (more observations than parameters);
- `cond(X'X) <= 1e12` else "(near-)singular / collinear" — this also
  rejects a constant regressor next to an intercept;
- `TSS > 0` else "y is constant: R^2 undefined".

Result fields: `params` (intercept first), `se`, `tstats`, `resid`,
`r2`, `sigma2`, `nobs`, `nparams`.

## 2. ADF test (`adf_test`)

Regression (pinned, constant / no-trend case, FIXED lag; `adf_test` never
picks its own lag, `select_lag` below is a separate explicit call):

```
dy_t = c + rho * y_{t-1} + sum_{i=1..p} phi_i * dy_{t-i} + e_t
```

- `p = maxlag`, integer `>= 0`, default `1`.
- Usable rows: `t = p+1 .. T-1` (0-indexed); effective nobs
  `T_eff = T - 1 - p`; number of parameters `k = 2 + p`.
- Statistic: plain OLS t-stat on `rho` (section 1 machinery), i.e. the
  coefficient at index 1 (right after the intercept).
- Cross-check target: statsmodels
  `adfuller(y, maxlag=p, regression='c', autolag=None)` t-stat, |diff| < 1e-8.

Critical values (pinned): the **MacKinnon (2010) finite-sample response
surfaces**, constant / no-trend case, evaluated at the effective
regression sample size `T_eff` (exactly what statsmodels' `adfuller`
passes to `mackinnoncrit`):

```
cv(T_eff) = b_inf + b_1 / T_eff + b_2 / T_eff^2 + b_3 / T_eff^3
```

Coefficients embedded verbatim (Table 1 of MacKinnon 2010, "no trend"
panel; every language embeds these exact numbers; evaluated in Horner
form `b_inf + inv*(b_1 + inv*(b_2 + inv*b_3))`, `inv = 1/T_eff`):

| nvars | use                      | level | b_inf    | b_1      | b_2     | b_3     |
|-------|--------------------------|-------|----------|----------|---------|---------|
| 1     | plain ADF on one series  | 1%    | -3.43035 | -6.5393  | -16.786 | -79.433 |
|       |                          | 5%    | -2.86154 | -2.8903  | -4.234  | -40.040 |
|       |                          | 10%   | -2.56677 | -1.5384  | -2.809  | 0       |
| 2     | EG residual test, 2 vars | 1%    | -3.89644 | -10.9519 | -33.527 | 0       |
|       |                          | 5%    | -3.33613 | -6.1101  | -6.823  | 0       |
|       |                          | 10%   | -3.04445 | -4.2412  | -2.720  | 0       |
| 3     | EG residual test, 3 vars | 1%    | -4.29374 | -14.4354 | -33.195 | 47.433  |
|       |                          | 5%    | -3.74066 | -8.5632  | -10.852 | 27.982  |
|       |                          | 10%   | -3.45218 | -6.2143  | -3.718  | 0       |

The `b_inf` column is the asymptotic table (exposed separately as
`CRITICAL_VALUES` / `adf_critical_values(nvars)` /
`critical_values(nvars, None)` / `Adf.criticalValues(nvars)`); the test
result's `crit` field always carries the finite-sample values at its own
`nobs`. Worked values: nvars = 1, `T_eff = 1498` -> 5% cv `-2.8634713`;
nvars = 2, `T_eff = 898` -> 5% cv `-3.3429426`; nvars = 1, `T_eff = 20`
-> 5% cv `-3.0216450`.

Verdict `reject(alpha)`: `tstat < cv(alpha)`, direct comparison with the
finite-sample value, `alpha` restricted to {0.01, 0.05, 0.10} (anything
else is an error).

Approximate p-value (documented approximation; the exact MacKinnon 1994
p-value response surface is deliberately NOT implemented): piecewise-linear
interpolation of significance level through the knots
`(cv_1%, 0.01), (cv_5%, 0.05), (cv_10%, 0.10)` evaluated at the same
`T_eff` (asymptotic knots when no sample size is given):
- `tstat <= cv_5%`: use the segment through (cv_1%, 0.01)-(cv_5%, 0.05);
- otherwise the segment through (cv_5%, 0.05)-(cv_10%, 0.10);
- linear extrapolation beyond the outer knots with the edge slopes;
- clamp to `[1e-4, 0.9999]`; a NaN t-stat is an error, +/-inf (an
  exact-fit regression) clamps to the bound on its side.

Validation: `maxlag` integer in `[0, 1000]` (documented upper bound —
no integer overflow for any input); `nvars` in {1,2,3}; finite input;
non-constant series; `T_eff >= max(k + 10, 20)` else "series too short"
(20 is the smallest sample size in MacKinnon's simulations; below it the
response surface is an extrapolation and the test is refused).

### Lag selection (`select_lag(y, max_lag, nvars)`)

Returns the integer lag in `0..max_lag` with the smallest BIC. Every
candidate uses the SAME rows `t = max_lag+1 .. T-1` (`n = T - 1 - max_lag`),
with regressors as in the ADF regression, so the BICs are comparable:

```
BIC(p) = n * ln(RSS_p / n) + (2 + p) * ln(n)
```

Evaluate `p = 0, 1, ..` in order and replace the best only on a strictly
smaller BIC, so a tie goes to the smaller lag. `RSS_p == 0` gives
`-inf` for the first term. Validation is that of
`adf_test(y, max_lag, nvars)`; `nvars` does not affect the result.
Cross-check: statsmodels `adfuller(y, maxlag=m, regression='c',
autolag='BIC')` picks the same lag.

## 3. Half-life of mean reversion (`half_life`)

AR(1) regression on the spread (pinned): `ds_t = c + rho * s_{t-1} + e_t`
by OLS with intercept; then

```
half_life = -ln(2) / ln(1 + rho)
```

Guards (pinned): `rho >= 0` -> `+inf` (no mean reversion);
`1 + rho <= 0` -> `0.0`. Both sentinels mean "the AR(1) fit is not
interpretable as a half-life — do not trade this spread": `0.0` covers
the oscillating (`-2 < rho < -1`) and explosive (`rho <= -2`) sign-flipping
fits alike and deliberately does not distinguish them, so a tradeability
filter must be two-sided (`0 < hl < H`). Validation: `T >= 12`, finite,
non-constant.

## 4. Engle-Granger two-step (`engle_granger`)

1. Hedge regression `y = a + b x + e` by section-1 OLS.
2. Section-2 ADF on the residuals `e` with the SAME fixed `maxlag`,
   judged against the `nvars=2` critical-value row (2-variable EG case;
   the residuals are estimated, so the nvars=1 row would over-reject).
   For a 3-variable residual regression use the `nvars=3` row.
3. `cointegrated(alpha)` = residual ADF `reject(alpha)`.

Result: `alpha`, `beta`, `resid`, `r2`, nested ADF result.

`engle_granger_both(y, x, maxlag)` runs the test as y on x and as x on y
(the two can disagree) and returns both results as `y_on_x`, `x_on_y`.
Its `cointegrated(alpha)` is true only when BOTH directions reject.

## 5. Johansen-lite trace test (`johansen_trace`)

Scope (honest limitations, pinned): n = 2 or 3 assets ONLY, VECM with
unrestricted constant, exactly k = 1 lagged difference, only the 95%
trace column. No small-sample correction, no max-eigenvalue variant,
no p-values.

Algorithm, step by step (T raw rows, `T_eff = T - 2`):

1. `dY_t = Y_t - Y_{t-1}` (T-1 rows). Usable rows `t = 2..T-1`:
   `Z0 = dY_t` (T_eff x n), `Z1 = Y_{t-1}` (T_eff x n),
   `ZD = [1, dY_{t-1}]` (T_eff x (n+1)).
2. Partial out ZD by multivariate OLS (normal equations, one solve per
   block): `R0 = Z0 - ZD * solve(ZD'ZD, ZD'Z0)`, same for `R1` from Z1.
   Error if `cond(ZD'ZD) > 1e12`.
3. Product moments: `S_ij = R_i' R_j / T_eff` for ij in {00, 01, 10, 11}.
4. Solve `|lambda S11 - S10 inv(S00) S01| = 0` via the SYMMETRIC form:
   - native Cholesky `S11 = L L'` (column algorithm; error if a pivot
     <= 0);
   - `A = inv(L) * (S10 * solve(S00, S01)) * inv(L)'`, then force
     symmetry `A := (A + A')/2`;
   - eigen-decomposition of A by the native cyclic **Jacobi rotation**
     algorithm (pinned — no library eigensolver): sweep upper-triangle
     pairs (p,q) in row-major order; for `|a_pq| > 1e-14` rotate with
     `theta = (a_qq - a_pp)/(2 a_pq)`,
     `t = sign(theta)/(|theta| + sqrt(theta^2+1))` (t = 1 when
     theta = 0), `c = 1/sqrt(t^2+1)`, `s = t*c`; stop when the
     off-diagonal Frobenius norm < 1e-14 or after 100 sweeps;
   - eigenvalues sorted DESCENDING, clipped into `[0, 1 - 1e-12]`;
   - cointegrating vectors `beta_i = inv(L)' v_i`, each normalised so
     its first component is +1 (when |first| > 1e-12).
5. Trace statistics, r = 0..n-1:
   `trace(r) = -T_eff * sum_{i=r..n-1} ln(1 - lambda_i)`
   (lambda index i counts the descending eigenvalues; the sum for rank
   r skips the r largest).

Embedded Osterwald-Lenum (1992) 95% trace critical values, model with
unrestricted constant, keyed by `n - r` (pinned verbatim):

| n - r | cv 95% |
|-------|--------|
| 1     |  3.76  |
| 2     | 15.41  |
| 3     | 29.68  |

Rank estimate: smallest r with `trace(r) < cv(n - r)`, else n.

Validation: 2-D input; n in {2,3}; `maxlag_diff == 1`; `T >= 20`;
finite; no constant column.

## 6. Kalman dynamic hedge (`kalman_hedge`)

Model (pinned):

```
state        s_t = (alpha_t, beta_t)',   s_t = s_{t-1} + w_t
observation  y_t = H_t s_t + eps_t,      H_t = [1, x_t]
Q = (delta / (1 - delta)) * I_2          # RiskMetrics-style scaling — exact form
R = r                                    # scalar observation variance
s_0 = (0, 0)',   P_0 = 1e3 * I_2         # pinned prior
```

Per-observation recursion, t = 0..T-1 (the very first observation is
already filtered against the prior):

```
P_pred = P + Q                       # predict (state prediction = s)
e_t    = y_t - H_t s                 # innovation
S_t    = H_t P_pred H_t' + r         # innovation variance
K_t    = P_pred H_t' / S_t           # gain (2-vector)
s      = s + K_t e_t
P      = (I - K_t H_t) P_pred (I - K_t H_t)' + K_t r K_t'   # JOSEPH form
P      = (P + P')/2                  # symmetrise against rounding
z_t    = e_t / sqrt(S_t)             # innovation z-score
```

The Joseph covariance update is pinned: algebraically equal to
`(I - K H) P_pred` but keeps P symmetric PSD under rounding.
`delta -> 0` recovers recursive least squares (static OLS in the limit).

Outputs: `alpha[t]`, `beta[t]`, `z[t]`, `innovation[t]`, `s_var[t]`,
final 2x2 posterior `final_cov`.

### Streaming (`KalmanStep(delta, r)`)

`update(x, y)` filters one observation and returns `(alpha, beta, z)`; O(1),
no allocation in `update` (a port may return through a small value type).
Note the argument order: x first. Constructor validation is that of
`kalman_hedge`; a non-finite x or y is an error and leaves the state
unchanged.

Scalar order (pinned, because `P_0 = 1e3 I` makes early updates sensitive
to rounding at the 1e-11 level). State `a, b, p00, p01, p11`, `q = delta/(1-delta)`:

```
p00 += q; p11 += q                     # local copies, P_pred
e   = y - (a + x*b)
ph0 = p00 + p01*x;  ph1 = p01 + p11*x
S   = ph0 + x*ph1 + r
k0  = ph0/S;  k1 = ph1/S
a  += k0*e;  b += k1*e
a00 = 1 - k0;  a01 = -k0*x;  a10 = -k1;  a11 = 1 - k1*x
b00 = a00*p00 + a01*p01;  b01 = a00*p01 + a01*p11
b10 = a10*p00 + a11*p01;  b11 = a10*p01 + a11*p11
p00' = b00*a00 + b01*a01 + r*(k0*k0)
n01  = b00*a10 + b01*a11 + r*(k0*k1)
n10  = b10*a00 + b11*a01 + r*(k1*k0)
p11' = b10*a10 + b11*a11 + r*(k1*k1)
p01' = 0.5*(n01 + n10)
z   = e / sqrt(S)
```

`kalman_hedge` is a loop over this step in every port, so batch and
streaming cannot diverge (tests assert agreement to 1e-12 on alpha, beta
and z).

Validation: equal lengths, `T >= 2`, finite inputs,
`0 < delta < 1` STRICTLY (bounds excluded), `r > 0` AND finite (NaN and
+inf are rejected in every port — written as negated comparisons so NaN
cannot pass).

## 7. Rolling z-score (`rolling_zscore`)

Pinned windowing — the current observation IS included:

```
z_t = (s_t - mean(s_{t-W+1..t})) / std(s_{t-W+1..t})    # std with ddof=1
```

- `t < W-1`: `z_t = NaN` (insufficient history).
- Zero-variance guard: sample std `<= 1e-12` -> `z_t = NaN` (never inf).
- Validation: integer `W >= 2`; `W <= T` else "window larger than
  series"; finite input.

Summation (pinned): plain left-to-right sums, one accumulator starting at
0.0, in every port. No numpy pairwise sum, no Neumaier or Kahan (Python's
builtin `sum` on floats is compensated from 3.12, so it is not used either):

```
total = 0; for i in t-W+1..t: total += s_i
m     = total / W
ss    = 0; for i in t-W+1..t: d = s_i - m; ss += d*d
sd    = sqrt(ss / (W-1))
z_t   = (s_t - m) / sd          # when sd > 1e-12, else NaN
```

The largest |z| a window can produce is `(W-1)/sqrt(W)` (section 8 uses this
to reject dead configs).

### Streaming (`RollingZ(window)`)

`update(s)` returns z, or NaN during warm-up and on zero variance, using a
ring buffer and running moments; O(1), no allocation in `update`. While
filling, Welford: `n += 1; d = s - mean; mean += d/n; m2 += d*(s - mean)`.
Once full, with `old` the value leaving the ring:
`mean' = mean + (s - old)/W; m2 += (s - old)*(s - mean' + old - mean);
m2 = max(m2, 0)`. Then `sd = sqrt(m2/(W-1))`, the same 1e-12 floor, and
`z = (s - mean)/sd`. Non-finite input is an error and leaves the state
unchanged; `W` must be an integer >= 2.

This is NOT bitwise equal to the batch function: tests assert agreement to
1e-9 at the bundled length. Rounding drifts slowly with the update count
(about 5e-8 after 50,000 updates at a level of 1e4). The batch function stays
the trading reference.

Spread definition used everywhere: `s_t = y_t - (alpha + beta * x_t)`.

## 8. Pairs backtest (`backtest_pair`)

### Timing (pinned — no look-ahead)

**A signal computed from the day-t close is executed at the day t+1
close, at day t+1 prices.** Day-t P&L marks the position held since the
day t-1 close. Per-day sequence, t = 0..T-1:

1. Mark-to-market: `pnl_t = q_y (y_t - y_{t-1}) + q_x (x_t - x_{t-1})`
   (0 for t = 0), with quantities as of the previous close.
2. Execute the single pending order (decided at the day t-1 close), if
   any, at day-t close prices.
3. If `t == T-1` and a position is still open: force-close it at the
   final close (a normal closed trade, `stopped = false`). A pending
   ENTRY at the end of data is DISCARDED: step 2 executes an entry only
   when `t < T-1`, so a signal fired at `T-2` never produces a zero-day
   round trip that is opened and force-closed at the same close (which
   would charge two costs and distort trade count, win rate, holding
   period and Sharpe). A pending EXIT at `T-1` executes normally.
4. `E_t = E_{t-1} + pnl_t - cost_t` with `E_{-1} = capital`.
5. If `t < T-1`, evaluate the signal `z_t` (section 7 z-score of the
   spread) and set at most one pending order (rules below).

### State machine (states: FLAT, LONG spread, SHORT spread)

Signal rules at day t (NaN z: no action; at most one pending order):

- FLAT, and `t >= e + cooldown_days` for the latest STOPPED exit executed
  on day e: `entry_z <= z_t < stop_z` -> pending SHORT-spread entry;
  `-stop_z < z_t <= -entry_z` -> pending LONG-spread entry;
  `|z_t| >= stop_z` while flat -> NO entry (pinned).
- In position, first match wins:
  1. `pnl_stop_frac > 0` and open trade P&L `<= -pnl_stop_frac * capital`
     -> pending STOP exit (`stopped = true`). Open trade P&L is the marks
     accrued since entry minus the entry cost; the exit cost is not yet
     known and is not included.
  2. `|z_t| >= stop_z` -> pending STOP exit (`stopped = true`).
  3. `max_hold_days > 0` and `t - entry_day >= max_hold_days` -> pending
     exit (`stopped = false`).
  4. `|z_t| <= exit_z` -> pending normal exit.
- Rules 1 and 3 do not read z, so they fire on a NaN z too. Rules 2 and 4
  and all entries need a non-NaN z.
- Limits are 0 = off. Execution is still next-close, so a trade can last
  `max_hold_days + 1` days.
- The cool-down follows stopped exits only (rules 1 and 2), not rule 3,
  rule 4 or the final force-close. With `cooldown_days = 0` a signal on the
  exit day itself may re-enter, as before.

The limits in `data/config.json` (0.02, 60, 20) are round numbers fixed in
advance. They were not tuned on this data.

### Position sizing (pinned)

Entry at the day-t close, direction `dir` (+1 long spread = long y /
short x; -1 short spread). Config `sizing`:

```
"beta_hedged" (default):  u = capital / (y_t + beta * x_t)
                          q_y = dir * u
                          q_x = -dir * beta * u
"equal_dollar":           q_y = dir  * (capital/2) / y_t
                          q_x = -dir * (capital/2) / x_t
```

Both have gross notional `|q_y| y_t + |q_x| x_t = capital` at entry.
`beta_hedged` holds the legs in the hedge ratio, so the position's P&L
tracks the spread, but it is not dollar-neutral. `equal_dollar` is exactly
dollar-neutral at entry, but its P&L also carries a net exposure whenever
beta differs from y/x. Quantities are held fixed for the life of the trade
(no daily rebalance).

`beta > 0` is required (`beta <= 0` or NaN is an error in `backtest_pair`):
a pair needing a negative hedge is out of scope.

### Cost accounting (pinned)

Charged at each execution close on traded notional:

```
cost = (cost_bps + slippage_bps) / 1e4 * (|dq_y| * y_t + |dq_x| * x_t)
```

Entry trades exactly capital of notional, so it costs
`capital * total_bps`; the exit trades the same quantities at exit prices.
Per-trade net P&L = accrued daily pnl while
holding minus its entry and exit costs.

Accounting identity (tested at 1e-10 relative to capital):
`E_{T-1} - capital == sum_t pnl_t - sum_t cost_t`.

### Configuration (validated on construction AND at run time)

`window >= 2` (integer); `entry_z, exit_z, stop_z, cost_bps,
slippage_bps` all finite (NaN/inf rejected); `0 <= exit_z < entry_z <
stop_z`; `cost_bps, slippage_bps >= 0`; `capital > 0` finite; `sizing` one
of the two names; `pnl_stop_frac >= 0` finite; `max_hold_days,
cooldown_days >= 0` integers (all three default 0).

Dead configs are rejected, with `reach = (W-1)/sqrt(W)`:
`reach <= entry_z` is an error (no entry can ever fire);
`reach <= stop_z` is an error unless `pnl_stop_frac > 0`, because then the
z stop can never fire and nothing else bounds the loss (`max_hold_days`
alone does not count).

Ports whose
config is a mutable struct (C++, Rust) re-run the same validation inside
`backtest_pair` and `walk_forward`, so mutating a field after
construction cannot bypass it (Python's dataclass is frozen, Java's
record is immutable). Price inputs must be finite and strictly positive;
equal lengths; `window <= T`.

### Metrics (pinned formulas; N = number of days)

```
daily_ret_t      = (E_t - E_{t-1}) / capital        # return on gross capital
total_return     = (E_{N-1} - capital) / capital
ann_return       = (1 + total_return)^(252/N) - 1
                   (pinned to exactly -1.0 when total_return <= -1, i.e.
                   the equity curve ends at or below zero — the power is
                   undefined there; NaN is never returned)
sharpe           = mean(daily_ret) / std(daily_ret, ddof=1) * sqrt(252)
                   (0.0 when std == 0 or N < 2)
max_drawdown     = max_t (peak_t - E_t) / peak_t,  peak_t = running max E
                   (positive fraction; 0 when equity never dips; may
                   exceed 1 when equity goes negative — the loss exceeded
                   the peak — but is always finite)
n_trades         = closed round trips (force-close counts, discarded
                   pending entries do not)
win_rate         = wins / n_trades   (win: net trade pnl > 0; 0 if none)
avg_holding_days = mean(exit_day - entry_day)       (0 if no trades)
turnover         = total traded notional / capital
```

No-trade guard: with no trades all of the above are exactly 0 and the
equity curve is flat at `capital` (no NaN/inf anywhere).
`compute_metrics` itself rejects an empty or non-finite equity curve and
a non-positive / non-finite capital.

Returns are on gross notional `capital`, not on running equity: gross
notional is `capital` at every entry regardless of P&L to date (no
compounding, no leverage cap, no financing or borrow on the short leg).

### Single holdout (`walk_forward`)

Despite the name, kept for compatibility, this is ONE fit and ONE split, not
a rolling procedure; see `rolling_walk_forward` below.

`split = floor(T * insample_frac + 1e-9)` with `0 < insample_frac < 1`
(default 0.6). The `1e-9` nudge (pinned) removes binary-float artefacts:
`700 * 0.7 == 489.99999999999994` in IEEE double, and the intended split
is 490 (likewise `100 * 0.29 -> 29`, `1500 * 0.6 -> 900`). `y` and `x`
must have equal length (typed error otherwise, before any slicing). Estimate `(alpha, beta)` by section-1 OLS on days
`0..split-1`; run the backtest on days `0..split-1` (in-sample) and,
with the SAME parameters, on days `split..T-1` (out-of-sample). The OOS
z-score warms up inside the OOS segment only — no information crosses
the split. Error when either segment is shorter than `window`, and when
the fit gives `beta <= 0` (section 8 rule).

### Rolling walk-forward (`rolling_walk_forward(y, x, config, train, test)`)

For `k = 0, 1, ..` while `k*test + train < T`:

```
fit   = section-1 OLS of y on x over days [k*test, k*test + train)
start = k*test + train
end   = min(start + test, T)
```

Each block is its own backtest on days `[start - (W-1), end)` with the
block's `(alpha, beta)`:

- The W-1 prior days make z valid on day `start` from history only. (The
  contract's "window days before" needs only W-1, because the window includes
  the current day. An earlier first day would change nothing.)
- Entry signals are ignored before day `start`, so the first possible entry
  executes on `start + 1`. A position never exists during warm-up.
- The usual end-of-data rule flattens the position at `end - 1` (forced
  close, `stopped = false`) and discards a pending entry.
- Cool-down state does not carry from one block to the next.
- A block whose fit gives `beta <= 0` or NaN is not an error: it stays flat
  (no trades, equity flat, z all NaN, metrics zero) with `skipped = true`.
  Its fitted `alpha`, `beta` are still reported.

Block result: `start, end, alpha, beta, skipped`, and a backtest result
trimmed to days `[start, end)`. Trade days are absolute day indices, and block
metrics use only the block days (N = end - start; turnover from the block).

Stitching: `running = capital`; for each block append
`running + (block_equity - capital)`, then set `running` to the last value.
The stitched curve has `T - train` points, the first being day `train`.
Result: `blocks`, `start_day = train`, `equity`, `trades` (all blocks),
`metrics = compute_metrics(equity, capital, trades)` with `turnover` the sum
of block turnovers.

Validation: equal lengths; `train`, `test` integers; `test >= 1`;
`train >= window`; `train < T`.

## 9. Error behavior

Python raises `ValueError` with a descriptive message; C++ throws
`std::invalid_argument`; Rust returns `Result<_, StatarbError>` (never
panics on any input); Java throws `IllegalArgumentException` (never an
`ArrayIndexOutOfBoundsException`, including on empty arrays). Errors are
raised for: series length mismatch (including in `walk_forward` and
`rolling_walk_forward`), `beta <= 0` in the backtest, dead configs
(section 8), unknown `sizing`, invalid risk limits or `train`/`test`,
non-finite inputs (series, matrices, hedge parameters, equity curves),
constant series in OLS/ADF/half-life, (near-)collinear designs
(cond > 1e12, including collinear Johansen level columns), series too
short (ADF `T_eff < max(k+10, 20)`), window larger than series or < 2,
non-positive prices in the backtest, `delta` outside the open interval
(0,1) or NaN, `r` not a finite number > 0, `maxlag < 0` or `> 1000`,
`nvars` outside {1,2,3}, Johansen n outside {2,3} or `maxlag_diff != 1`,
invalid or non-finite backtest config (also when re-validated at run
time), `alpha` outside {0.01, 0.05, 0.10} in `reject`, NaN t-stat in the
p-value interpolation, a CSV `day` column that is not strictly
increasing, CSV cells outside the number grammar, non-finite CSV cells or
config values, non-integer integer config fields, trailing JSON content,
non-finite input to `RollingZ.update` or `KalmanStep.update`. Non-convergence
(Jacobi sweep cap, 100 sweeps) is not an error: the best iterate is
returned (matrices here are 2x2/3x3 and converge). Half-life sentinels
(`+inf`, `0.0`) are returned values, not errors — both mean "not
tradeable" (section 3).
Deterministic everywhere; NO RNG at runtime in the ported API (RNG lives
in `data/generate_data.py` and in the caller-seeded bootstrap of the
Python-only `statarb.research`, section 12).

## 10. Bundled data and configuration

- `data/pair_equity.csv` (`day, stock_y, stock_x`): 1500 days;
  `y = 12 + beta_t x + OU spread`; true beta 1.7 through day 1049,
  linear ramp to 1.1 over days 1050..1080, then 1.1; spread AR(1)
  phi = 0.92 pre-break, 0.995 after (the regime break). Seed 42.
- `data/fx_triangle.csv` (`day, eurusd, gbpusd, eurgbp, basket`):
  `basket = eurgbp * gbpusd`; `log(eurusd) = log(basket) + OU(0.9)`.
  Seed 7. Cointegration lives in LOG prices for the 3-asset case.
- `data/random_walks.csv` (`day, walk_a, walk_b`): independent
  geometric random walks — the spurious-regression case. Seed 100.
- `data/config.json` (flat scalars): `window` 60, `entry_z` 2.0,
  `exit_z` 0.5, `stop_z` 4.0, `cost_bps` 1.0, `slippage_bps` 0.5,
  `capital` 1e6, `insample_frac` 0.6, `adf_maxlag` 1, `significance`
  0.05, `kalman_delta` 1e-4, `kalman_r` 1e-3, `pnl_stop_frac` 0.02,
  `max_hold_days` 60, `cooldown_days` 20. The file stays flat and numeric:
  `sizing` is not a config.json key, it is set in code and defaults to
  `beta_hedged`.

In-sample = the first `floor(1500 * 0.6 + 1e-9) = 900` days.

Loader contract (`load_csv`, all languages): header plus at least one
data row; no ragged rows; unique column names; every cell, after trimming
ASCII spaces and tabs, fully matches the plain number grammar

```
[+-]?([0-9]+\.?[0-9]*|\.[0-9]+)([eE][+-]?[0-9]+)?
```

and parses to a FINITE double. So `nan`, `inf`, `Infinity`, hex (`0x10`),
`_` separators, type suffixes (`1f`, `1d`, `1L`), non-ASCII digits and
overflow (`1e999`) are all rejected, whatever the language's own parser would
accept: check the grammar BEFORE calling it. If a column named `day` exists its
cells must match `[+-]?[0-9]+` (so `1.0` is rejected) and it must be **strictly
increasing** (unsorted, duplicated or re-merged time indices are refused
at the boundary). Leading/trailing blank lines are ignored; an interior
blank line is a ragged row.

`load_config` accepts one flat JSON object and nothing after it (trailing
content, including a second object, is an error). Values are finite JSON
numbers (RFC 8259 grammar; `NaN`, `Infinity`, hex, `_`, suffixes, booleans,
strings and nested values are rejected). The keys `window`, `adf_maxlag`,
`max_hold_days`, `cooldown_days` must be written as integers: `60` is
accepted, `60.0` and `6e1` are not. An integer of magnitude above 2^53 is
rejected in any key.

Time-grid assumption (documented, not validated beyond monotonicity):
every model treats consecutive rows as EQUALLY SPACED observations —
lag-1 differences in ADF/Johansen/half-life, next-close execution in the
backtest, and the `sqrt(252)` / `252/N` annualisation assume one row per
trading day. Gaps (holidays, missing prints) or non-daily bars are the
caller's responsibility; there is no calendar, day-count or
`periods_per_year` parameter.

## 11. Golden cases (`data/golden/golden.json`)

Flat schema, unchanged: `{"cases": [{name, inputs, expect: {value}, tol}]}`.
`inputs` values are scalars (number or string), `expect.value` is one number,
`tol` is absolute. Booleans are encoded 0.0/1.0; integer expectations must
match exactly. All values are computed from the CSVs as shipped, with
config.json parameters; hedge parameters for z-score and `backtest` cases come
from Engle-Granger on the first 900 days.

928 cases; every language's golden suite asserts this exact count.
The list length depends on the trade counts, so a port should loop over the
file and dispatch, not enumerate names.

### Scenarios

`inputs.scenario` names a backtest configuration:

| scenario | sizing | risk limits |
|----------|--------|-------------|
| limits_beta | beta_hedged | config.json (0.02, 60, 20) |
| limits_dollar | equal_dollar | config.json |
| plain_beta | beta_hedged | all 0 |
| plain_dollar | equal_dollar | all 0 |

### Cases dispatched on `inputs.kind`

A `field` is one of `total_return, ann_return, sharpe, max_drawdown,
n_trades, win_rate, avg_holding_days, turnover, n_stops, equity_sum`
(`equity_sum` = sum of the equity curve in day order, tol 1e-3).

| kind | other inputs | value | tol |
|------|--------------|-------|-----|
| backtest | scenario, field | full-sample backtest metric; names `bt_<scenario>_<field>` | 1e-8 (counts 1e-9) |
| backtest_equity | scenario, day | equity at day in 0, 250, 500, 900, 1050, 1200, 1499 | 1e-6 |
| backtest_trade | scenario, index, field | trade `index`: entry, exit, direction, pnl, stopped | pnl 1e-6, others 1e-9 |
| walk_forward | segment, field, scenario limits_beta | segment insample or oos: a field above. Segment fit: alpha, beta, split | as above; fit 1e-9 |
| rolling | scenario, train 500, test 100, field | stitched metric, or `n_blocks` | as above |
| rolling_block | scenario, train, test, block, field | start, end, alpha, beta, skipped, total_return, sharpe, n_trades | 1e-8 |
| margin | none | see below | 1e-9 |

`zscore_threshold_margin` (kind margin) is the smallest `||z| - th|` over
`th` in entry_z, exit_z, stop_z and every z path above: the backtest, both
walk_forward segments and every non-skipped rolling block. Ports assert only
that the stored value exceeds 1e-6: a trade list could otherwise flip on
rounding. Python also recomputes it. `generate_data.py` refuses to write
goldens below that bound.

Scenarios limits_beta and plain_beta are the ones used by kind rolling.

### Cases dispatched on name (no `kind`)

| case | definition | tol |
|------|------------|-----|
| ols_pair_alpha / _beta / _r2 | full-sample OLS y~x on the pair | 1e-10 |
| adf_resid_insample_tstat | ADF t (maxlag 1) on in-sample EG residuals | 1e-8 |
| adf_resid_insample_crit5 | MacKinnon (2010) 5% cv, nvars 2, T_eff 898 | 1e-10 |
| adf_resid_lag0_ / lag3_ tstat, crit5, nobs | same residuals, maxlag 0 and 3, nvars 2 | 1e-8, 1e-10, 1e-9 |
| adf_fx_nvars3_tstat / _crit1 / _crit5 / _crit10 / _pvalue | ADF (maxlag 1, nvars 3) on residuals of OLS with intercept: log eurusd on log gbpusd, log eurgbp; full sample | 1e-8, then 1e-10 |
| eg_pair_insample_coint | EG verdict at 5% (1.0 = cointegrated) | 1e-9 |
| eg_both_insample_x_on_y_tstat | residual ADF t of x on y, in-sample | 1e-8 |
| eg_both_insample_coint | both-direction verdict at 5% | 1e-9 |
| adf_random_walk_tstat | ADF t on walk_a levels | 1e-8 |
| adf_random_walk_reject | ADF verdict at 5% (0.0 = fails to reject) | 1e-9 |
| adf_random_walk_crit5 | MacKinnon (2010) 5% cv, nvars 1, T_eff 1498 | 1e-10 |
| adf_random_walk_pvalue | interpolated p-value through the T_eff-1498 knots | 1e-10 |
| select_lag_resid | `select_lag` on in-sample EG residuals, max_lag 5, nvars 2 | 1e-9 |
| select_lag_walk | `select_lag` on walk_a, max_lag 5, nvars 1 | 1e-9 |
| eg_spurious_coint | EG verdict walk_a~walk_b (0.0; no backtest is run) | 1e-9 |
| johansen_pair_trace_r0 / _r1 | in-sample 2-asset trace stats | 1e-6 |
| johansen_pair_rank | estimated rank | 1e-9 |
| johansen_fx_trace_r0 / _r1 / _r2 | log eurusd, gbpusd, eurgbp, full sample | 1e-6 |
| johansen_fx_vector0 / 1 / 2 | components of the leading vector (first is +1) | 1e-6 |
| johansen_fx_rank | estimated rank | 1e-9 |
| halflife_insample_spread | half-life of in-sample EG residuals | 1e-8 |
| halflife_growth_is_inf | `s_t = 1.1^t`, t = 0..19: 1.0 when half-life is +inf (JSON has no inf) | 1e-9 |
| halflife_alternating_zero | `s_t = (-1)^t`, t = 0..19: half-life, the 0.0 sentinel | 1e-9 |
| kalman_beta_day500 / _day1200 / _final | filtered beta at pinned days (delta 1e-4, r 1e-3) | 1e-8 |
| kalman_alpha_day500 / _day1200 / _final | filtered alpha | 1e-8 |
| kalman_z_day500 / _day1200 / _final | innovation z | 1e-8 |
| zscore_day200 / _day1100 | rolling z (window 60) of the in-sample-hedge spread | 1e-10 |
| backtest_total_return / _sharpe / _max_drawdown | scenario limits_beta | 1e-8 |
| backtest_n_trades | scenario limits_beta, closed trades | 1e-9 |

Regenerate with `python3 data/generate_data.py`, which first
cross-validates the reference against statsmodels (OLS coefficients and
standard errors; ADF t-statistics AND finite-sample critical values via
`mackinnoncrit`; `select_lag` via `adfuller` BIC selection; Johansen
eigenvalues/trace statistics via `coint_johansen` and an independent numpy
generalised-eigen solve) and refuses to write golden values on any
disagreement. The CSVs are byte-identical across regenerations (the RNG
streams are unchanged).

## 12. Python-only research helpers (`statarb.research`)

Not ported and not in the goldens; numpy and the standard library only.
`block_bootstrap_sharpe_ci` (circular blocks, caller's seed),
`probabilistic_sharpe`, `expected_max_sharpe`, `deflated_sharpe`,
`cost_sensitivity`, `cusum_break_test`, `benjamini_hochberg`,
`kalman_grid_mle` (with innovation z sd and lag-1 autocorrelation) and
`scan_pairs` (EG both ways, BH on the larger p-value, then half-life and
`beta > 0` filters). The docstrings give the formulas and sources. These are
judgement aids; none of them feeds the trading path.
