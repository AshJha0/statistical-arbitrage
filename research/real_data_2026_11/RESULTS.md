# Real-data pairs study: results

**Outcome: null. The v1.2.0 rule showed no edge on real US data, even before costs and with a universe biased in its favour.**

Protocol: [PREREGISTRATION.md](PREREGISTRATION.md), committed in `4733077` before any data was downloaded.
The holdout ran once. No parameter or code changed between the development and holdout runs.
Figures: `results_dev.json`, `results_holdout.json`. Reproduce with `python study.py dev`, then `holdout`. Yahoo data can be revised, so numbers may shift slightly.

## Holdout gates (2020-01-02 to 2026-10-30, 1,708 days)

| Gate | Required | Result | |
|---|---|---|---|
| 1. Net Sharpe | >= 1.0 | -0.42 | FAIL |
| 2. PSR vs 0 | >= 0.95 | 0.13 | FAIL |
| 2. Bootstrap 95% CI lower bound | > 0 | -1.15 (upper 0.41) | FAIL |
| 3. Max drawdown | <= 10% | 1.10% | pass, see below |
| 4. Beta to SPY | within 0.10 | 0.00003 | pass, see below |
| 5. Capacity, median entry | >= $1m | $3.07m | pass |

Gates 3 and 4 say little: the book was almost always nearly empty, so small drawdown and beta are expected.

## Other figures

| | Development 2012-2019 | Holdout 2020-2026 |
|---|---|---|
| Gross return a year, on gross capital | 0.01% | -0.10% |
| Net return a year | -0.04% | -0.13% |
| Net Sharpe | -0.16 | -0.42 |
| Trades | 78 | 46 |
| Win rate | 68% | 52% |
| Stopped trades | 17 | 16 |
| Periods with no pairs | 25% | 36% |
| Net Sharpe at 2 / 10 / 20 bps | -0.05 / -0.33 / -0.67 | -0.36 / -0.64 / -0.74 |
| Equal-dollar sizing, net Sharpe | -0.02 | -0.19 |

## What this shows
- **Costs are not the problem.** Gross return is about zero in both periods.
- **Few pairs survive selection:** 0 to 5 of 20 slots per period, and none in four of the last five holdout periods. Multiple-testing control and the two-way test are strict, which is the intent, but the book is barely deployed.
- **Losses are larger than the stop.** The 2% stop acts at the next close: the worst holdout trade lost 5.7% of pair capital.
- **The result is an upper bound.** The universe is today's survivors. Kellogg (`K`) is absent from the source, an example of that bias.

## Limits
One rule, one data source, one universe, and few trades, so the confidence interval is wide: this does not prove that no pairs strategy works. Selection's completeness rule looks at the trading window, a small look-ahead that favours the strategy. Capacity uses 1% of the thinner leg's volume as gross.

## Decision
Under the preregistered rule this is reported as a null, with no tuning. The repo stays a teaching library. A different idea, such as sector-residual baskets, is a new study with a new preregistration.
