# statistical-arbitrage v1.1.0: review, 2026-11-05

Verdict: a careful, well-tested teaching library. It is not quant research, and it is not low-latency code.
It has no real data, no pair selection, and no strategy with evidence of edge.
Experiments are in `experiments/` (exp2 stops at a bug fixed in exp3; they use absolute local paths and expect to run side by side).

## Verified good
- Tests: Python 86/86 and Rust 74/74 passed. C++ demo clean under ASan/UBSan, g++ 15, 0 warnings (library + demo only; gtest suite NOT run). Java main compiled and demo ran; JUnit suite NOT run.
- Python, Java and Rust demos differ only cosmetically. C++ figures for sections 2-3 matched by eye.
- MacKinnon surfaces match statsmodels exactly. EG size is correct under white-noise errors (4.6-5.5%).
- Timing is look-ahead free; the Kalman recursion and accounting read correctly.
- Kalman beats rolling OLS on beta RMSE (0.024 vs 1.1-1.2). The reviewer expected otherwise and was wrong.

## Critical
C1. No real data. All three datasets are synthetic, and the pair's DGP is the model under test. Across 300 seeds with no break, OOS Sharpe is 1.44: that recovers the generator, it is not evidence.
C2. No research pipeline: no universe, pair selection, multiple-testing control, or rolling re-estimation. On 780 pure random-walk pairs, 32 passed EG; their mean IS Sharpe was 0.76 and OOS -0.16. The best was 1.77 IS, -0.11 OOS.
C3. Kalman is never traded in the library, and it fails when traded. Days 150-1049, rolling z on the causal residual: Sharpe 0.05 (35 trades), 0.24 with a grid-MLE tune. Static hedge with the true beta: 1.72 (oracle); estimated static hedge, IS days 0-899: 1.49 (different window). One seed. Mechanism: r=1e-3 is ~2,800x below the spread variance (2.82), so beta absorbs the signal. Beta error x price (~0.025 x 100) exceeds the spread sd (1.68); filtered spread vs true correlation is 0.18; innovation z sd is 0.65-0.70, not 1. "Kalman adapts" (README, site) is true of beta RMSE only.
C4. "Stop-losses cap the damage" (README:127, demo) is false. The stop cost -$17k on day 1053; re-entry on day 1059 lost -$124k unstopped, more than the whole -$103k net loss. OOS max DD is 15.4% vs 0.91% IS. The rolling mean follows the break, so z decays and the stop cannot fire. Across 300 seeds with the break: OOS Sharpe -1.9, negative 99.7% of the time.
C5. Sizing ignores the hedge ratio. Equal dollars do not trade the tested spread. y=150+0.5x, 100 seeds: Sharpe 0.43 equal-dollar vs 1.76 beta-hedged. The bundled pair hides this (y/x 1.82 vs beta 1.64). Negative beta is accepted and gets wrong legs: -0.01 vs 1.62 (`backtest.py`, no beta check).

## High
H1. Fixed ADF lag has bad size under negative MA errors: 31-33% at MA -0.5, 95-97% at MA -0.8 (nominal 5%). Those MA values are deliberately strong. y|x and x|y verdicts disagree up to 32%. Need lag selection and both normalisations.
H2. No break detection. EG on days 1050-1499 says cointegrated (t=-9.16) across the break; 900-1499 correctly does not. A single-window test cannot be the gate.
H3. "Walk-forward" is one 60/40 holdout with one fit, losing 59 OOS warm-up days. Rename it or implement rolling refits.
H4. No uncertainty anywhere. IS Sharpe 1.49 has SE ~0.77 (approx. 0 to 3.0), 21 trades, flat 75% of days. Need block bootstrap, deflated Sharpe, PSR.
H5. Cost model: flat 1.5 bps/side; no spread, impact, borrow, financing, capacity. IS Sharpe: 1.49 at 1.5 bps, 0.74 at 10, -0.10 at 20.
H6. No CI and no branch protection. All cross-language claims rest on manual runs.
H7. Goldens: 24 scalars, one dataset, one config. Not covered cross-language: trade lists, equity paths, walk_forward, Johansen n=3 and vectors, sentinels, error paths. Ports also sum differently (numpy pairwise, Neumaier, naive, Kahan), and `z >= entry_z` is a hard branch, so a last-bit difference can change a trade path. Closest margin on golden data is 3.5e-4, so it never shows. [port audit]
H8. Nothing is tick-capable: batch only, O(T*W) z-score, no streaming API, no benchmarks, Vec<Vec<f64>>/double[][], allocating Jacobi rotations, no book/feed/order/risk layer. Any low-latency reading is unsupported. [port audit]

## Medium
- `alpha` has no effect on results (rolling z is shift-invariant; verified): the fitted equilibrium is thrown away.
- Validated configs can be dead: |z| is capped at (W-1)/sqrt(W), so W<=17 never reaches stop_z=4 and W=5 never reaches entry_z=2.
- Entry is not rechecked at execution. Costs and Sharpe mix non-compounded P&L with a geometric annualised return.
- Kalman Q uses the same variance for alpha ($) and beta (unitless). Levels rather than logs make spread noise price-dependent.
- FX triangle residual (~18 bps sd, phi 0.9) is not a realistic parity deviation.
- Port audit: C++/Java config parsers accept invalid JSON; CSV number grammar differs per port; `window: 60.5` truncates in C++/Rust demos; jacobi_eigh symmetry and tie order differ; absolute JACOBI_TOL runs all 100 sweeps on X'X; compute_metrics peak=0 gives three answers.
- Build: gtest mandatory, no default Release, no Rust release profile or clippy, Java has hardcoded Linux jar paths and no Maven/Gradle.
- Owner rule breach: the word "interview" at LEARN.md:737 and docs/index.html:90 (the live site matches).

## Fix order
1. Now, small: correct README/site claims (C3, C4, walk-forward label); remove "interview"; reject or handle beta<=0; beta-hedged sizing; add CI for all four languages, branch protection.
2. Engine: P&L stop, time stop, cool-down; fixed equilibrium mean option; dynamic hedge in the backtester; streaming step() API; rolling refit; cost, borrow and impact model.
3. Statistics: lag selection, both normalisations, break tests (CUSUM, Gregory-Hansen), Kalman MLE/EM with diagnostics, bootstrap CIs, deflated Sharpe, FDR on pair selection.
4. Research: free real data, survivorship-aware universe, sector-constrained selection, preregistered holdout, portfolio of pairs with market/sector neutrality. A null result is acceptable; publish it.
5. Cross-language and latency: trade-level goldens, one pinned summation, threshold-margin tests, fuzzed loaders; O(1) rolling stats, fixed-size matrices, benchmarks with p50/p99/p99.9. Consider cutting Java or stating the ports are learning material.

# Addendum: business, functional, profit, tradability

All figures are from `experiments/exp4_economics.py`: 300 seeds of the repo's own synthetic generator, best-case parameters. They describe the generator, not any market, and no real-market profit has been measured.

## Profit
- No-break, zero cost: 1.70%/yr gross on gross notional; 5.1 round trips/yr; 10.2x turnover; in market 23% of days; 33.5 bps per round trip.
- Break-even all-in cost: 16.7 bps/side. Net: 1.55% at 1.5 bps, 1.19% at 5, 0.69% at 10.
- Borrow is minor for general collateral (0.03%/yr at 25 bps) and material for hard-to-borrow names (0.34%/yr at 3%).
- Tail decides it: a break window returns -12.6% on average (p5 -18.2%), about 16 points worse than normal. Break-even is one break per ~10.5 years per pair. Gain is small and steady, loss is rare and large: short-volatility shaped. Real pairs (M&A, earnings, regulation) can be expected to break more often than that; measure it.
- $10m gross on one pair earns about $155k/yr in the best case, before any break, staff, data or financing.

## Tradability gaps
Pre-trade: no ADV/capacity limit, locate/borrow check, restricted list, corporate actions, earnings/M&A filter, or calendars.
Execution: same-close fill for both legs is assumed. No leg risk, partial fills, auction logic, order types, or fill model. No share rounding (fractional quantities).
Risk: no portfolio, sector or factor limits, gross/net limits, drawdown kill switch, or volatility-scaled sizing. Dollar neutrality decays after entry because positions never rebalance.
Operations: no broker adapter, position reconciliation, paper trading, P&L attribution, audit trail, or monitoring.

## Functional gaps
One pair at a time; no multi-pair book; no scanner; no daily signal/run command; no config for dates or calendars; no report output; no market-data adapter. The library answers "was this pair cointegrated?", not "what do I trade tomorrow, and how much?".

## Business position
- Plain distance/cointegration pairs in liquid large caps is a crowded, published strategy; the academic record shows returns decaying after publication (Gatev et al. 2006; Do & Faff 2010; cited from memory, check before quoting). Expect no edge from the method alone.
- Plausible niches to test: sector-residual or PCA baskets rather than two names; ETF vs constituents; dual listings and share classes, where the relationship has an economic link; event-aware filters. Rates or FX relative value could reuse the owner's other repos.
- Four language ports add no revenue or alpha. For a single operator, one research language plus one production path is the right cost.
- Realistic value today: portfolio and teaching asset. Position it that way until real-data net results exist.

## Go/no-go gates before any capital
1. Real-data, survivorship-aware, preregistered holdout: net Sharpe > 1 after full costs, deflated for trials.
2. Measured break rate and loss, at portfolio level, with drawdown inside a stated budget.
3. Capacity at 1% ADV exceeds the capital that covers running costs.
4. At least 3 months of paper trading with slippage within modelled cost.
5. Live orders only by owner decision.
If gate 1 fails, publish the null and keep the repo as teaching material.
