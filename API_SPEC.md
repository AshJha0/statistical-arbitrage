# statarb — API Specification (pinned numerical contract)

This document is the cross-language contract for project 09
(statistical arbitrage: cointegration, Kalman filter, pairs backtest).
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

Regression (pinned, constant / no-trend case, FIXED lag — no automatic
lag selection):

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

Embedded critical-value tables (MacKinnon-style asymptotic values,
constant no-trend case; pinned verbatim, keyed by `nvars`):

| nvars | use                        |   1%  |   5%  |  10%  |
|-------|----------------------------|-------|-------|-------|
| 1     | plain ADF on one series    | -3.43 | -2.86 | -2.57 |
| 2     | EG residual test, 2 vars   | -3.90 | -3.34 | -3.04 |
| 3     | EG residual test, 3 vars   | -4.29 | -3.74 | -3.45 |

Verdict `reject(alpha)`: `tstat < cv(alpha)`, direct table lookup,
`alpha` restricted to {0.01, 0.05, 0.10} (anything else is an error).

Approximate p-value (documented approximation; the exact MacKinnon
response surface is deliberately NOT implemented): piecewise-linear
interpolation of significance level through the knots
`(cv_1%, 0.01), (cv_5%, 0.05), (cv_10%, 0.10)`:
- `tstat <= cv_5%`: use the segment through (cv_1%, 0.01)-(cv_5%, 0.05);
- otherwise the segment through (cv_5%, 0.05)-(cv_10%, 0.10);
- linear extrapolation beyond the outer knots with the edge slopes;
- clamp to `[1e-4, 0.9999]`.

Validation: `maxlag` integer `>= 0`; `nvars` in {1,2,3}; finite input;
non-constant series; `T_eff >= k + 10` else "series too short".

## 3. Half-life of mean reversion (`half_life`)

AR(1) regression on the spread (pinned): `ds_t = c + rho * s_{t-1} + e_t`
by OLS with intercept; then

```
half_life = -ln(2) / ln(1 + rho)
```

Guards (pinned): `rho >= 0` -> `+inf` (no mean reversion);
`1 + rho <= 0` -> `0.0`. Validation: `T >= 12`, finite, non-constant.

## 4. Engle-Granger two-step (`engle_granger`)

1. Hedge regression `y = a + b x + e` by section-1 OLS.
2. Section-2 ADF on the residuals `e` with the SAME fixed `maxlag`,
   judged against the `nvars=2` critical-value row (2-variable EG case;
   the residuals are estimated, so the nvars=1 row would over-reject).
   For a 3-variable residual regression use the `nvars=3` row.
3. `cointegrated(alpha)` = residual ADF `reject(alpha)`.

Result: `alpha`, `beta`, `resid`, `r2`, nested ADF result.

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

Validation: equal lengths, `T >= 2`, finite inputs,
`0 < delta < 1` STRICTLY (bounds excluded), `r > 0`.

## 7. Rolling z-score (`rolling_zscore`)

Pinned windowing — the current observation IS included:

```
z_t = (s_t - mean(s_{t-W+1..t})) / std(s_{t-W+1..t})    # std with ddof=1
```

- `t < W-1`: `z_t = NaN` (insufficient history).
- Zero-variance guard: sample std `<= 1e-12` -> `z_t = NaN` (never inf).
- Validation: integer `W >= 2`; `W <= T` else "window larger than
  series"; finite input.

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
   ENTRY at the end of data is discarded.
4. `E_t = E_{t-1} + pnl_t - cost_t` with `E_{-1} = capital`.
5. If `t < T-1`, evaluate the signal `z_t` (section 7 z-score of the
   spread) and set at most one pending order (rules below).

### State machine (states: FLAT, LONG spread, SHORT spread)

Signal rules at day t (NaN z: no action; at most one pending order):

- FLAT: `entry_z <= z_t < stop_z` -> pending SHORT-spread entry;
  `-stop_z < z_t <= -entry_z` -> pending LONG-spread entry;
  `|z_t| >= stop_z` while flat -> NO entry (pinned).
- In position: `|z_t| >= stop_z` -> pending STOP exit (flagged);
  else `|z_t| <= exit_z` -> pending normal exit.
- Re-entry after any exit follows the same FLAT rules (no cool-down).

### Position sizing (pinned)

Entry at the day-t close, direction `dir` (+1 long spread = long y /
short x; -1 short spread):

```
q_y = dir  * (capital/2) / y_t
q_x = -dir * (capital/2) / x_t
```

Equal-dollar legs: the position is EXACTLY dollar-neutral at entry
(each leg carries capital/2 of notional; tested). The hedge ratio beta
enters through the spread/signal, not the leg sizing. Quantities are
held fixed for the life of the trade (no daily rebalance).

### Cost accounting (pinned)

Charged at each execution close on traded notional:

```
cost = (cost_bps + slippage_bps) / 1e4 * (|dq_y| * y_t + |dq_x| * x_t)
```

Entry and exit each trade capital of notional, so each costs
`capital * total_bps`. Per-trade net P&L = accrued daily pnl while
holding minus its entry and exit costs.

Accounting identity (tested at 1e-10 relative to capital):
`E_{T-1} - capital == sum_t pnl_t - sum_t cost_t`.

### Configuration (validated on construction)

`window >= 2` (integer); `0 <= exit_z < entry_z < stop_z`;
`cost_bps, slippage_bps >= 0`; `capital > 0` finite. Price inputs must
be finite and strictly positive; equal lengths; `window <= T`.

### Metrics (pinned formulas; N = number of days)

```
daily_ret_t      = (E_t - E_{t-1}) / capital        # return on gross capital
total_return     = (E_{N-1} - capital) / capital
ann_return       = (1 + total_return)^(252/N) - 1
sharpe           = mean(daily_ret) / std(daily_ret, ddof=1) * sqrt(252)
                   (0.0 when std == 0 or N < 2)
max_drawdown     = max_t (peak_t - E_t) / peak_t,  peak_t = running max E
                   (positive fraction; 0 when equity never dips)
n_trades         = closed round trips (force-close counts, discarded
                   pending entries do not)
win_rate         = wins / n_trades   (win: net trade pnl > 0; 0 if none)
avg_holding_days = mean(exit_day - entry_day)       (0 if no trades)
turnover         = total traded notional / capital
```

No-trade guard: with no trades all of the above are exactly 0 and the
equity curve is flat at `capital` (no NaN/inf anywhere).

### Walk-forward (`walk_forward`)

`split = floor(T * insample_frac)` with `0 < insample_frac < 1`
(default 0.6). Estimate `(alpha, beta)` by section-1 OLS on days
`0..split-1`; run the backtest on days `0..split-1` (in-sample) and,
with the SAME parameters, on days `split..T-1` (out-of-sample). The OOS
z-score warms up inside the OOS segment only — no information crosses
the split. Error when either segment is shorter than `window`.

## 9. Error behavior

Python raises `ValueError` with a descriptive message; C++ throws
`std::invalid_argument`; Rust returns `Result<_, StatarbError>`; Java
throws `IllegalArgumentException`. Errors are raised for: series length
mismatch, non-finite inputs, constant series in OLS/ADF/half-life,
(near-)collinear designs (cond > 1e12), series too short, window larger
than series or < 2, non-positive prices in the backtest, `delta`
outside the open interval (0,1), `r <= 0`, `maxlag < 0`, `nvars`
outside {1,2,3}, Johansen n outside {2,3} or `maxlag_diff != 1`,
invalid backtest config ordering, `alpha` outside {0.01, 0.05, 0.10}
in `reject`. Non-convergence (Jacobi sweep cap) is not an error: the
best iterate is returned (matrices here are 2x2/3x3 and converge).
Deterministic everywhere; NO RNG at runtime (RNG lives only in
`data/generate_data.py`).

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
  0.05, `kalman_delta` 1e-4, `kalman_r` 1e-3.

In-sample = the first `floor(1500 * 0.6) = 900` days.

## 11. Golden cases (`data/golden/golden.json`)

Flat schema; every case has scalar `inputs`, one scalar
`expect.value`, and an absolute tolerance `tol`. Booleans are encoded
0.0/1.0; integer expectations (rank, trade count) must match exactly.
All values are computed from the CSVs as shipped, with config.json
parameters; hedge parameters for z-score/backtest cases come from
Engle-Granger on the first 900 days.

| case | definition | tol |
|------|------------|-----|
| ols_pair_alpha / _beta / _r2 | full-sample OLS y~x on the pair | 1e-10 |
| adf_resid_insample_tstat | ADF t (maxlag 1) on in-sample EG residuals | 1e-8 |
| eg_pair_insample_coint | EG verdict at 5% (1.0 = cointegrated) | 1e-9 |
| adf_random_walk_tstat | ADF t on walk_a levels | 1e-8 |
| adf_random_walk_reject | ADF verdict at 5% (0.0 = fails to reject) | 1e-9 |
| eg_spurious_coint | EG verdict walk_a~walk_b (0.0; no backtest is run) | 1e-9 |
| johansen_pair_trace_r0 / _r1 | in-sample 2-asset trace stats | 1e-6 |
| johansen_pair_rank | estimated rank (1) | 1e-9 |
| halflife_insample_spread | half-life of in-sample EG residuals | 1e-8 |
| kalman_beta_day500 / _day1200 / _final | filtered beta at pinned days (delta 1e-4, r 1e-3) | 1e-8 |
| zscore_day200 / _day1100 | rolling z (window 60) of the in-sample-hedge spread | 1e-10 |
| backtest_total_return / _sharpe / _max_drawdown | full-sample backtest, in-sample hedge, config.json | 1e-8 |
| backtest_n_trades | closed trades (31, exact integer) | 1e-9 |

21 cases. Regenerate with `python3 data/generate_data.py`, which first
cross-validates the reference OLS/ADF against statsmodels and refuses
to write golden values on any disagreement.
