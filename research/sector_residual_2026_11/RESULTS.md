# Sector-residual reversal study: results

**Outcome: null. The book was deployed and traded often, but its small gross return did not cover costs.**

Protocol: [PREREGISTRATION.md](PREREGISTRATION.md), committed in `1404b15` before the strategy was run. The holdout ran once; nothing changed after the development run.
This is the second trial on these years, after the [pairs study](../real_data_2026_11/RESULTS.md).
Figures: `results_dev.json`, `results_holdout.json`. Reproduce with `python residual.py dev`, then `holdout`. Yahoo data can be revised.

One choice was not in the preregistration: the deflated Sharpe's variance input is the variance of the two studies' holdout Sharpes. It was fixed before the holdout ran and does not change the outcome, since the undeflated probabilistic Sharpe already fails.

## Holdout gates (2020-01-02 to 2026-10-30, 1,717 days)

| Gate | Required | Result | |
|---|---|---|---|
| 1. Net Sharpe | >= 1.0 | -0.32 | FAIL |
| 2. PSR vs 0 | >= 0.95 | 0.21 | FAIL |
| 2. Bootstrap 95% CI lower bound | > 0 | -1.09 (upper 0.42) | FAIL |
| 3. Max drawdown | <= 10% | 5.02% | pass |
| 4. Beta to SPY | within 0.10 | 0.006 | pass |
| 5. Capacity, median entry | >= $1m | $5.85m | pass |
| 6. Deflated Sharpe, 2 trials | >= 0.95 | 0.18 | FAIL |

## Other figures

| | Development 2012-2019 | Holdout 2020-2026 |
|---|---|---|
| Gross return a year | 0.07% | 0.31% |
| Gross Sharpe | 0.10 | 0.26 |
| Net return a year | -0.61% | -0.37% |
| Net Sharpe | -0.87 | -0.32 |
| Trades | 4,432 | 3,800 |
| Average open positions (of 63) | 23.8 | 23.8 |
| Average hold, days | 10.8 | 10.7 |
| Win rate, after costs | 55% | 55% |
| Net Sharpe at 2 / 10 / 20 bps | -0.49 / -1.49 / -2.73 | -0.09 / -0.69 / -1.44 |

Holdout net Sharpe by year: 2020 0.98, 2021 -0.90, 2022 -0.20, 2023 -2.21, 2024 -0.08, 2025 -0.89, 2026 -0.05.

## What this shows
- **It fixed what pairs could not:** about 24 positions open and 3,800 trades, with neutrality as designed.
- **Costs now matter.** Gross is slightly positive but not distinguishable from zero. Even at 2 bps the net Sharpe is below zero.
- **Only 2020 was positive,** a year of unusual dislocation, so the small gross return may not be a stable effect.
- **This is an upper bound.** The universe is survivors, which flatters reversal.

## Limits
Daily closes with a one-day delay, one parameter set, 63 large caps, and constant-notional positions with rebalancing cost ignored. Shorter-horizon or intraday versions were not tested, and this does not show they fail.

## Decision
Reported as a null, with no tuning. Two preregistered ideas have now failed on free daily data. Further variants on the same data and years would mostly be searching for luck.
