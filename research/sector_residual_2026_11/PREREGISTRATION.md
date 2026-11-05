# Preregistration: sector-residual reversal study

Written 2026-11-05, before this strategy was run on any data and before the sector ETF prices were downloaded. Git history is the timestamp.
Everything below is fixed. Any change after this commit is reported as a deviation.

## Disclosure
This is the SECOND trial on these years. The pairs study (`research/real_data_2026_11`) already used the same stocks and holdout dates, and its null result is known. Stock prices are already cached. The deflated Sharpe therefore uses 2 trials. The rule follows the published style of Avellaneda & Lee (2010, cited from memory), with round parameters, not values fitted here.

## Question
Does a sector-neutral residual reversal book earn a positive net return on real US data?

## Data and universe
Same source and same 63 stock tickers in the same 8 groups as the pairs study (Kellogg is unavailable). Raw prices are not committed. Survivors only: this flatters reversal strategies, so a pass is weak evidence and needs point-in-time confirmation.
Hedges: banks XLF, energy XLE, staples XLP, health XLV, tech XLK, industrials XLI, utilities XLU, retail XLY.

## Signal, at each close t, for each stock with 60 complete days
1. Daily simple returns over days t-59..t. OLS of stock on its ETF with intercept gives beta and residuals.
2. X = cumulative sum of the residuals (intercept removed).
3. AR(1) on X: X_k = a + b X_{k-1} + e. Eligible only if 0 < b < 1 and half-life -ln2/ln(b) <= 30 days.
4. m = a/(1-b); sigma = sd(e)/sqrt(1-b^2); s = (X_t - m)/sigma.

## Trading
- Flat: s >= 1.25 opens short, s <= -1.25 opens long. An ineligible stock opens nothing.
- Open: exit when |s| <= 0.5, when the position has been held 20 days, or when the stock becomes ineligible.
- Signal at close t executes at close t+1.
- Position: one unit of stock notional, hedged with -beta units of ETF, beta fixed at entry. Notional is held constant daily; the rebalancing cost of that is ignored.

## Accounting
Daily P&L = sum over open positions of dir * (r_stock - beta * r_etf), using returns from the close after execution.
Capital = 2 x number of stocks in the universe that day (a unit of stock and of hedge per name). Empty capacity earns 0, no cash interest.
Costs: 5 bps per side on stock notional; 2 bps on the change in net ETF notional per sector. Borrow 1% a year on short stock notional.

## Periods
Development 2012-01-01 to 2019-12-31: pipeline bugs only, parameters may not change.
Holdout 2020-01-01 to 2026-10-30: run once. Gates use this only.

## Pass gates (holdout, all required)
1. Net annualised Sharpe >= 1.0.
2. Probabilistic Sharpe vs 0 >= 0.95, and block-bootstrap 95% CI lower bound > 0 (block 20, 5000 draws, seed 20261105).
3. Max drawdown <= 10% of capital.
4. |beta to SPY| <= 0.10.
5. Capacity: 1% of the 20-day median dollar volume at entry is >= $1m for the median entry.
6. Deflated Sharpe >= 0.95 with 2 trials.

## Also reported, not gates
Gross return first, then development results, average positions, turnover, trades, win rate, net Sharpe at 2, 5, 10, 20 bps stock cost, Sharpe by calendar year, worst day.

## Outcome rule
Fail any gate: report a null and do not tune. Any variant is a new study.
