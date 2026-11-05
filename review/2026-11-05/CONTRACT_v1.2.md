# v1.2.0 contract (all lanes follow this; API_SPEC.md becomes the final form)

## Ported to all four languages
1. Backtest input: reject beta <= 0 (language's error type); a pair needing a negative hedge is out of scope.
2. Config `sizing`: "beta_hedged" (DEFAULT) or "equal_dollar". beta_hedged: u = capital / (y + beta*x) at execution close; q_y = dir*u; q_x = -dir*beta*u (gross = capital at entry). equal_dollar = the old rule.
3. Config risk limits, all default 0 = off, validated >= 0 and finite (integers for days):
   - `pnl_stop_frac`: at signal step, if open trade P&L (net of costs so far) <= -pnl_stop_frac*capital, pending STOP exit (stopped=true). Checked before z rules; takes priority.
   - `max_hold_days`: at signal step, if t - entry_day >= max_hold_days, pending exit (stopped=false), after pnl and z-stop rules.
   - `cooldown_days`: after any stopped exit executed on day e, no entry signal while t < e + cooldown_days (t = signal day).
   data/config.json turns on 0.02 / 60 / 20. These are round risk limits, NOT tuned; docs must say so.
4. Dead-config rejection: reject window where (W-1)/sqrt(W) <= entry_z (entry unreachable). stop_z unreachable is allowed only if pnl_stop_frac > 0; otherwise reject.
5. Rolling z summation PINNED: plain sequential left-to-right sums for mean and for squared deviations, in every port (no numpy pairwise, Neumaier or Kahan in rolling_zscore). Python uses an explicit loop or equivalent exact order.
6. `rolling_walk_forward(y, x, config, train, test)`: for k = 0,1,..: fit OLS on [k*test, k*test+train); score days [k*test+train, min(+test, T)) using the fit; z warm-up uses the window days before the test block (history only, no future). Flat at each block end (force close by running each block as its own backtest on warm-up + block, with entries only inside the block). Result: per-block metrics plus stitched equity (each block's P&L added to running equity) and compute_metrics on it. The old `walk_forward` stays, documented as a SINGLE HOLDOUT, name kept for compatibility.
7. Streaming API, O(1) per update, no allocation in update: `RollingZ(window)` (ring buffer; update(s) -> z or NaN) and `KalmanStep(delta, r)` (update(x, y) -> alpha, beta, z). Tests assert equality with batch: Kalman exact to 1e-12; RollingZ to 1e-9 (running sums, so not pinned bitwise; batch stays the trading reference).
8. ADF: `select_lag(y, max_lag, nvars)`: min BIC across lags 0..max_lag on a common sample. `engle_granger_both(y, x, maxlag)`: runs both directions; cointegrated only if both reject.
9. Loaders strict and aligned: reject trailing JSON content, hex, `_`, type suffixes, non-integer `window`/day fields; plain decimal/exponent grammar only.

## Python only (python/src/statarb/research.py, numpy only, not ported, not in goldens)
block-bootstrap Sharpe CI (seeded), probabilistic and deflated Sharpe, cost sensitivity, CUSUM break test on spread, Benjamini-Hochberg over pair p-values, Kalman grid-MLE with innovation diagnostics (z sd, lag-1 autocorrelation), `scan_pairs` (EG both directions + BH + half-life filter).

## Goldens
Keep the format. Add cases: trade list per backtest (entry, exit, direction, pnl, stopped), equity at pinned days plus sum, ann_return, win_rate, hold, turnover, n_stops, both sizing modes, config with risk limits on, walk_forward and rolling_walk_forward metrics, Johansen n=3 trace and leading vector, Kalman alpha/z at pinned days, ADF nvars=3 and maxlag 0/3, half-life sentinels, select_lag. Record min |z - threshold| margin as a case; test asserts it exceeds 1e-6.

## Rules
No "interview" anywhere. Docs voice: plain, sourced, no hype. No commits or pushes by lanes: the coordinator commits. Stay in your own paths.
