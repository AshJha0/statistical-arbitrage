# Preregistration: real-data pairs study

Written 2026-11-05, before any price data was downloaded for this study. Git history is the timestamp.
Everything below is fixed. Any change after this commit is reported as a deviation.

## Question
Does the v1.2.0 pairs rule, with parameters fixed in advance, earn a positive net return on real US data?

## Data
Yahoo Finance public chart endpoint, daily split- and dividend-adjusted close plus volume, no account. Raw prices are not committed.
Known bias: the universe is names that survive today, so results are an UPPER BOUND. A pass needs confirmation on point-in-time data; a fail is strong evidence.

## Universe
Sector groups (pairs only inside a group):
- banks: JPM BAC C WFC USB PNC GS MS
- energy: XOM CVX COP EOG SLB OXY VLO MPC
- staples: KO PEP PG CL KMB GIS K HSY
- health: JNJ PFE MRK ABT LLY BMY AMGN GILD
- tech: MSFT AAPL ORCL IBM INTC CSCO TXN QCOM
- industrials: CAT DE HON MMM GE UNP CSX NSC
- utilities: DUK SO AEP D EXC XEL ED WEC
- retail: WMT TGT HD LOW COST KR DG DLTR

Linked pairs: GOOGL/GOOG, FOXA/FOX, NWSA/NWS, LEN/PHM, V/MA, GLD/IAU, SPY/IVV, EWA/EWC, XLE/VDE, KO/PEP.
A name is used in a period only with complete data in formation and trading windows.

## Periods
Trading periods start each 1 Jan and 1 Jul, 126 trading days at most. Formation: the prior 504 trading days.
- Development: periods starting 2012-01 to 2019-07. For finding pipeline bugs only. Parameters may not change.
- Holdout: periods starting 2020-01 to 2026-07, ending 2026-10-30. Run once. Gates use this only.

## Selection, per period, formation data only
Prices rebased to 100 at formation start. For each candidate: `engle_granger_both` with the lag from `select_lag` (max 5); p = larger of the two directions' interpolated p-values; Benjamini-Hochberg at q = 0.05 across all candidates; keep beta > 0 and 5 <= half-life <= 60 days. Take up to 20 pairs by lowest p, ties by ticker order. y is the alphabetically first ticker.

## Trading
`backtest_pair`, hedge fixed from formation OLS, on rebased prices from 59 days before the period to its end, entries inside the period only, flat at period end.
window 60, entry 2.0, exit 0.5, stop 4.0, beta_hedged, pnl_stop_frac 0.02, max_hold_days 60, cooldown_days 20.
Cost 5 bps per side all-in. Borrow 1% a year on short notional while open.

## Portfolio
20 equal slots of gross capital. Daily return = sum of pair P&L / total gross capital; empty slots earn 0. No cash interest.

## Pass gates (holdout, all required)
1. Net annualised Sharpe >= 1.0.
2. Probabilistic Sharpe vs 0 >= 0.95, and block-bootstrap 95% CI lower bound > 0 (block 20, 5000 draws, seed 20261105).
3. Max drawdown <= 10% of gross.
4. |beta to SPY| <= 0.10.
5. Capacity: at 1% of 20-day median dollar volume of the thinner leg, the median entry supports >= $1m gross per pair.

## Also reported, not gates
Development results, gross return, turnover, trades, win rate, stop counts, share of periods with no pairs, net Sharpe at 2, 5, 10, 20 bps, worst trades, equal-dollar sizing variant.

## Outcome rule
Fail any gate: report it as a null and do not tune. Any later variant is a new study with a new preregistration.
