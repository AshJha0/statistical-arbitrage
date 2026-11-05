# Changelog

Format: newest first. Numbers marked synthetic come from the bundled
generated data and are not evidence of edge. The review behind v1.2.0 is
`review/2026-11-05/REVIEW.md`; the contract is
`review/2026-11-05/CONTRACT_v1.2.md`, folded into `API_SPEC.md`.

## Unreleased

- Docs brought in line with both studies; Java artifact version set to 1.2.0.
- Sector-residual reversal study, preregistered: null result (holdout net Sharpe -0.32). See `research/sector_residual_2026_11/RESULTS.md`.
- Real-data pairs study, preregistered: null result (holdout net Sharpe -0.42). See `research/real_data_2026_11/RESULTS.md`.
- CI: action versions updated, runner pinned to ubuntu-24.04.

## 1.2.0

Changes follow the v1.1.0 review. Several change results: backtest numbers
from v1.1.0 are not comparable (default sizing changed, `config.json` now
turns risk limits on).

### Changed behaviour (all four languages)

- Backtest input with `beta <= 0` (or NaN) is an error. A pair needing a
  negative hedge is out of scope (review C5: v1.1.0 gave the wrong legs).
- New config `sizing`: `beta_hedged` (default) or `equal_dollar` (the v1.1.0
  rule). `beta_hedged` holds `capital / (y + beta*x)` of y against
  `-beta` times that of x, so gross notional is `capital` at entry. It is not
  dollar-neutral. Review C5: Sharpe 0.43 equal-dollar vs 1.76 beta-hedged on
  `y = 150 + 0.5x`, 100 seeds.
- New risk limits, all default 0 (off): `pnl_stop_frac`, `max_hold_days`,
  `cooldown_days`. `data/config.json` sets 0.02, 60, 20: round numbers fixed
  in advance, not tuned.
- Dead configs are rejected: a window whose maximum |z|, `(W-1)/sqrt(W)`,
  cannot reach `entry_z`; and one that cannot reach `stop_z` unless
  `pnl_stop_frac > 0`.
- Rolling z-score sums are pinned to plain left-to-right order in every port,
  so a last-bit difference cannot flip a trade (review H7). The C++ build adds
  `-ffp-contract=off`.
- Loaders are strict and aligned: no trailing JSON content, hex, `_` or type
  suffixes, and `window` and day fields must be integers.

### Added (all four languages)

- `rolling_walk_forward(y, x, config, train, test)`: refit every block,
  history-only z warm-up, flat at block ends, stitched equity, blocks with
  `beta <= 0` skipped. `walk_forward` is unchanged and now documented as a
  single holdout (review H3).
- Streaming API, O(1) per update with no allocation: `RollingZ(window)` and
  `KalmanStep(delta, r)`. Kalman matches batch to 1e-12; `RollingZ` to 1e-9
  (it drifts about 5e-8 after 50,000 updates at a level of 1e4).
- `select_lag(y, max_lag, nvars)` (minimum BIC on a common sample) and
  `engle_granger_both` (cointegrated only if both directions reject), review H1.
- Golden suite grown from 24 to 928 cases: trade lists, equity at pinned
  days, both sizing modes, risk limits on, holdout and rolling metrics,
  Johansen n = 3 and leading vector, Kalman alpha and z, ADF `nvars = 3`,
  half-life sentinels, `select_lag`, and the minimum threshold margin.

### Added (Python only)

- `statarb.research`: block-bootstrap Sharpe interval, probabilistic and
  deflated Sharpe, cost sensitivity, CUSUM break test, Benjamini–Hochberg,
  Kalman grid-MLE with innovation diagnostics, `scan_pairs`. Not ported, not
  in the goldens.
- `pyproject.toml` (ruff, mypy settings).

### Build and CI

- `.github/workflows/ci.yml`: Python 3.11 and 3.12 (ruff, mypy, pytest); C++
  with g++ and clang++ and an ASan/UBSan build; Rust with `cargo fmt --check`,
  clippy `-D warnings` and `cargo test --release --locked`; Java with Maven;
  a job that regenerates `data/` and fails on any difference.
- C++: default build type Release; `STATARB_SANITIZE` and
  `STATARB_BUILD_TESTS` options; GoogleTest fetched (v1.14.0) when not
  installed.
- Java: `pom.xml`; `build.sh` and `test.sh` locate JUnit and Hamcrest jars
  through `JUNIT_JAR`, `HAMCREST_JAR`, `JUNIT_HOME`, `/usr/share/java` or
  `~/.m2` instead of hard-coded Linux paths.
- Rust: release profile with LTO.

### Documentation

- Corrected claims. "Stop-losses cap the damage" was false: the z-stop alone
  did not, and with a P&L stop, max-hold and cool-down the single-holdout
  out-of-sample result is -4.91% with max drawdown 6.86%, against -12.35% and
  15.28% with none: a smaller loss, still a loss. "Kalman adapts" is true of
  beta tracking only; trading the Kalman spread gave Sharpe 0.05 against 1.49
  for a static hedge on a different window (one seed). `walk_forward` is a
  single holdout. Rolling walk-forward (-0.29%, Sharpe -0.05, interval
  [-1.11, 1.45]) is flattered because the last three of 10 blocks fit
  `beta <= 0` and sat flat.
- New "Limitations and what this is not" sections: synthetic data, not low
  latency, not tradable as is, synthetic economics from the review addendum.
- The word the owner banned in public repositories was removed from LEARN.md
  and the site; the LEARN.md section is now "Knowledge check".
- Demo output refreshed from a real run; diagrams updated; this changelog added.
- Python test count 1113 (was 86). C++, Rust and Java counts: 107,
  115, 109.

### Known limits

See the README limitations section. In short: no real data, no research
universe, not tick-capable, and the benchmark for any claim of edge is a
preregistered real-data holdout that does not exist yet.

## 1.1.0

Previous release (reviewed 2026-11-05). Fixed the end-of-data phantom trade,
moved ADF and Engle–Granger to MacKinnon (2010) finite-sample critical
values with `T_eff >= 20`, added finite-parameter checks and day-column
monotonicity, and grew the goldens from 21 to 24 cases. Dated detail is in
the git history.
