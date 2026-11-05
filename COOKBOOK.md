# COOKBOOK — statarb recipes

Task-oriented recipes for the `statarb` toolkit, each in all four languages.
Python snippets run as-is from the `python/` directory with
`PYTHONPATH=src python3 snippet.py` (they locate the bundled data relative to
the project root). C++/Rust/Java snippets follow the pinned API contract in
[`API_SPEC.md`](API_SPEC.md): C++ namespace `statarb` (headers
`<statarb/*.hpp>`, series as `std::vector<double>`, levels as the row-major
`statarb::Matrix`), Rust crate `statarb` (fallible functions return
`Result<_, StatarbError>`; paths are `&std::path::Path`), Java package
`com.quant.statarb` (series as `double[]`, results are nested records such
as `Adf.Result`, `EngleGranger.Result`, `Backtest.Config`; errors are
`IllegalArgumentException`; paths are `java.nio.file.Path`).

The C++, Rust and Java snippets are written so that, concatenated in order
inside one `main` (with the `#include` / `use` / `import` lines hoisted),
they form a complete program: recipe 2 defines `y`, `x` and `eg`, recipe 7
defines `cfg`, and later recipes reuse them. They are compiled and run
against the real libraries as part of the release check (see the note at
the end).

All numbers quoted in comments are the golden values for the bundled data,
or, where a comment says "demo", the Python demo's output. The data is
synthetic: these recipes show how to call the code, not that a strategy
works. Recipes 13–15 (streaming, lag selection, research helpers) are
Python-only except where stated. Recipes 17–20 use the paper-trading layer
(`statarb.trading`): Python only, paper only, no live broker. Their code was run
from `python/` and the output shown is the real output. Recipes 21–23 (benchmarks,
reproducing the studies, a preregistration template) give commands and a template;
see each for what was not re-run.

---

## 1. How do I test a series for a unit root (ADF)?

**Python**

```python
from statarb import adf_test, load_csv

walks = load_csv("../data/random_walks.csv")
res = adf_test(walks["walk_a"], maxlag=1, nvars=1)
print(f"t = {res.tstat:.4f}, p ~ {res.pvalue:.4f}, 5% cv = {res.crit['5%']}")
print("unit root rejected at 5%:", res.reject(0.05))   # False: it IS a walk
```

**C++**

```cpp
#include <iostream>
#include <statarb/adf.hpp>
#include <statarb/data_io.hpp>

auto walks = statarb::load_csv("../data/random_walks.csv");
statarb::ADFResult adf = statarb::adf_test(walks.at("walk_a"), /*maxlag=*/1, /*nvars=*/1);
std::cout << "t = " << adf.tstat << ", 5% cv = " << adf.crit.cv5   // -2.8635 at T_eff 1498
          << ", reject@5% = " << adf.reject(0.05) << "\n";        // false: it IS a walk
```

**Rust**

```rust
use std::path::Path;
use statarb::{adf_test, load_csv};

let walks = load_csv(Path::new("../data/random_walks.csv"))?;
let adf = adf_test(&walks["walk_a"], 1, 1)?;
println!("t = {:.4}, 5% cv = {:.4}, reject@5% = {}", adf.tstat, adf.crit[1], adf.reject(0.05)?);
```

**Java**

```java
import java.nio.file.Paths;
import com.quant.statarb.Adf;
import com.quant.statarb.DataIo;

var walks = DataIo.loadCsv(Paths.get("../data/random_walks.csv"));
Adf.Result adf = Adf.test(walks.get("walk_a"), 1, 1);
System.out.printf("t = %.4f, 5%% cv = %.4f, reject@5%% = %b%n",
    adf.tstat(), adf.crit5(), adf.reject(0.05));
```

Note: `reject` compares the t-stat directly to the MacKinnon (2010)
finite-sample critical value at the regression's own sample size
(`res.crit`); the p-value is a documented piecewise-linear approximation and
is never used for the verdict. Tests with fewer than 20 effective
observations are refused. (Python: `res.crit['5%']`.)

---

## 2. How do I test two price series for cointegration (Engle–Granger)?

**Python**

```python
from statarb import engle_granger, load_csv

pair = load_csv("../data/pair_equity.csv")
y, x = pair["stock_y"][:900], pair["stock_x"][:900]   # in-sample window
eg = engle_granger(y, x, maxlag=1)
print(f"hedge: y = {eg.alpha:.4f} + {eg.beta:.4f} x  (R2 {eg.r2:.4f})")
print(f"resid ADF t = {eg.adf.tstat:.4f} vs 5% cv {eg.adf.crit['5%']}")
print("cointegrated at 5%:", eg.cointegrated(0.05))    # True; t = -5.8244
```

**C++**

```cpp
#include <vector>
#include <statarb/engle_granger.hpp>

auto pair = statarb::load_csv("../data/pair_equity.csv");
const std::vector<double> y_all = pair.at("stock_y"), x_all = pair.at("stock_x");
const std::vector<double> y(y_all.begin(), y_all.begin() + 900);   // in-sample window
const std::vector<double> x(x_all.begin(), x_all.begin() + 900);
statarb::EGResult eg = statarb::engle_granger(y, x, /*maxlag=*/1);
std::cout << "beta = " << eg.beta << ", resid ADF t = " << eg.adf.tstat
          << ", coint@5% = " << eg.cointegrated(0.05) << "\n";     // 1.6389, -5.8244, true
```

**Rust**

```rust
use statarb::engle_granger;

let pair = load_csv(Path::new("../data/pair_equity.csv"))?;
let (y_all, x_all) = (&pair["stock_y"], &pair["stock_x"]);
let (y, x) = (y_all[..900].to_vec(), x_all[..900].to_vec());     // in-sample window
let eg = engle_granger(&y, &x, 1)?;
println!("beta = {:.4}, resid ADF t = {:.4}, coint@5% = {}", eg.beta, eg.adf.tstat, eg.cointegrated(0.05)?);
```

**Java**

```java
import java.util.Arrays;
import com.quant.statarb.EngleGranger;

var pair = DataIo.loadCsv(Paths.get("../data/pair_equity.csv"));
double[] yAll = pair.get("stock_y"), xAll = pair.get("stock_x");
double[] y = Arrays.copyOfRange(yAll, 0, 900);                    // in-sample window
double[] x = Arrays.copyOfRange(xAll, 0, 900);
EngleGranger.Result eg = EngleGranger.test(y, x, 1);
System.out.printf("beta = %.4f, resid ADF t = %.4f, coint@5%% = %b%n",
    eg.beta(), eg.adf().tstat(), eg.cointegrated(0.05));
```

The residual ADF is automatically judged against the 2-variable
Engle–Granger critical values (−3.3429 at 5% for T_eff = 898), not the plain
ADF row — using the 1-variable row here would over-detect cointegration.

---

## 3. How do I build the spread and its z-score signal?

**Python**

```python
from statarb import engle_granger, load_csv, rolling_zscore, spread

pair = load_csv("../data/pair_equity.csv")
y, x = pair["stock_y"], pair["stock_x"]
eg = engle_granger(y[:900], x[:900])            # hedge from in-sample only
s = spread(y, x, eg.alpha, eg.beta)             # s_t = y_t - (a + b x_t)
z = rolling_zscore(s, window=60)                # first 59 entries are NaN
print(f"z[200] = {z[200]:.6f}")                 # golden: -1.1720390869
```

**C++**

```cpp
#include <statarb/signals.hpp>

std::vector<double> s = statarb::spread(y_all, x_all, eg.alpha, eg.beta);
std::vector<double> z = statarb::rolling_zscore(s, /*window=*/60);
// z[t] is NaN for t < 59 and for zero-variance windows
std::cout << "z[200] = " << z[200] << "\n";                       // -1.1720390869
```

**Rust**

```rust
use statarb::{rolling_zscore, spread};

let s = spread(y_all, x_all, eg.alpha, eg.beta)?;
let z = rolling_zscore(&s, 60)?;   // z[t] = f64::NAN for t < 59
println!("z[200] = {:.10}", z[200]);                               // -1.1720390869
```

**Java**

```java
import com.quant.statarb.Signals;

double[] s = Signals.spread(yAll, xAll, eg.alpha(), eg.beta());
double[] z = Signals.rollingZscore(s, 60);   // Double.NaN for t < 59
System.out.printf("z[200] = %.10f%n", z[200]);                     // -1.1720390869
```

Pinned windowing: the current observation is **included** in the window and
the standard deviation uses ddof = 1. Both choices shift every signal date if
changed — the golden z-scores pin them at 1e−10.

---

## 4. How do I measure the half-life of mean reversion?

**Python**

```python
from statarb import engle_granger, half_life, load_csv

pair = load_csv("../data/pair_equity.csv")
eg = engle_granger(pair["stock_y"][:900], pair["stock_x"][:900])
hl = half_life(eg.resid)
print(f"half-life = {hl:.2f} days")     # golden: 8.39 days
```

**C++**

```cpp
double hl = statarb::half_life(eg.resid);   // +inf if rho >= 0 (no reversion)
std::cout << "half-life = " << hl << " days\n";                   // 8.39
```

**Rust**

```rust
let hl = statarb::half_life(&eg.resid)?;    // f64::INFINITY if rho >= 0
println!("half-life = {hl:.2} days");                              // 8.39
```

**Java**

```java
double hl = Signals.halfLife(eg.resid());   // POSITIVE_INFINITY if rho >= 0
System.out.printf("half-life = %.2f days%n", hl);                  // 8.39
```

Interpretation: choose the z-score window as several half-lives (60 ≈ 7×8.39
here) and expect holding periods of one to two half-lives. `+inf` means the
AR(1) fit found no reversion and `0.0` means the fit flips sign every step —
both sentinels say "do not trade that spread".

---

## 5. How do I run the Johansen trace test on 2 or 3 assets?

**Python**

```python
import numpy as np
from statarb import johansen_trace, load_csv

pair = load_csv("../data/pair_equity.csv")
levels = np.column_stack([pair["stock_y"][:900], pair["stock_x"][:900]])
jo = johansen_trace(levels, maxlag_diff=1)
for r in range(2):
    print(f"trace(r={r}) = {jo.trace_stats[r]:8.3f}  cv95 = {jo.crit_95[r]}")
print("rank =", jo.rank, " vector:", jo.beta[:, 0])   # rank 1, (1, -1.6455)
```

**C++**

```cpp
#include <statarb/johansen.hpp>
#include <statarb/linalg.hpp>

// levels is a T x n row-major statarb::Matrix: one ROW per day, one column per asset
statarb::Matrix levels(900, 2);
for (std::size_t t = 0; t < 900; ++t) { levels(t, 0) = y[t]; levels(t, 1) = x[t]; }
statarb::JohansenResult jo = statarb::johansen_trace(levels, /*maxlag_diff=*/1);
std::cout << "rank = " << jo.rank << ", trace(0) = " << jo.trace_stats[0]
          << ", vector (1, " << jo.beta(1, 0) << ")\n";           // 1, 37.572, -1.6455
```

**Rust**

```rust
use statarb::johansen_trace;

// levels is a Vec of T rows, each a Vec of n asset values (NOT one Vec per asset)
let levels: Vec<Vec<f64>> = (0..900).map(|t| vec![y[t], x[t]]).collect();
let jo = johansen_trace(&levels, 1)?;
println!("rank = {}, trace(0) = {:.3}, vector (1, {:.4})", jo.rank, jo.trace_stats[0], jo.beta[1][0]);
```

**Java**

```java
import com.quant.statarb.Johansen;

double[][] levels = new double[900][2];         // T rows x n columns
for (int t = 0; t < 900; t++) { levels[t][0] = y[t]; levels[t][1] = x[t]; }
Johansen.Result jo = Johansen.traceTest(levels, 1);
System.out.printf("rank = %d, trace(0) = %.3f, vector (1, %.4f)%n",
    jo.rank(), jo.traceStats()[0], jo.beta()[1][0]);
```

Passing one series per row (e.g. `{col_y, col_x}`) is the classic mistake:
it is a 2 × 900 matrix, which the ports reject with "n = 2 or 3 assets
only" — a message that looks like a data problem. Exactly collinear columns
are rejected as "singular".

For FX use **log** rates — cross-rate identities are multiplicative, so the
(1, −1, −1) triangle relation is linear only in logs:

```python
fx = load_csv("../data/fx_triangle.csv")
logs = np.column_stack([np.log(fx["eurusd"]), np.log(fx["gbpusd"]),
                        np.log(fx["eurgbp"])])
print(johansen_trace(logs).rank)   # 1, leading vector ~ (1, -1, -1)
```

Scope is pinned: n ∈ {2, 3}, k = 1 lagged difference, unrestricted constant,
95% trace column only.

---

## 6. How do I track a dynamic hedge ratio with the Kalman filter?

**Python**

```python
from statarb import kalman_hedge, load_csv

pair = load_csv("../data/pair_equity.csv")
km = kalman_hedge(pair["stock_y"], pair["stock_x"], delta=1e-4, r=1e-3)
print(f"beta[500]  = {km.beta[500]:.6f}")    # golden: 1.692056 (true 1.7)
print(f"beta[1200] = {km.beta[1200]:.6f}")   # golden: 1.100117 (true 1.1)
print(f"innovation z[1200] = {km.z[1200]:.3f}")
```

**C++**

```cpp
#include <statarb/kalman.hpp>

statarb::KalmanResult km = statarb::kalman_hedge(y_all, x_all, /*delta=*/1e-4, /*r=*/1e-3);
std::cout << "beta[500] = " << km.beta[500] << ", beta[1200] = " << km.beta[1200]
          << ", beta today = " << km.beta.back() << "\n";        // 1.692056, 1.100117, 1.153189
```

**Rust**

```rust
let km = statarb::kalman_hedge(y_all, x_all, 1e-4, 1e-3)?;
let beta_today = km.beta[km.beta.len() - 1];
println!("beta[500] = {:.6}, beta[1200] = {:.6}, today = {beta_today:.6}", km.beta[500], km.beta[1200]);
```

**Java**

```java
import com.quant.statarb.Kalman;

Kalman.Result km = Kalman.hedge(yAll, xAll, 1e-4, 1e-3);
double betaToday = km.beta()[km.beta().length - 1];
System.out.printf("beta[500] = %.6f, beta[1200] = %.6f, today = %.6f%n",
    km.beta()[500], km.beta()[1200], betaToday);
```

Tuning: `delta` is the forgetting dial (`Q = delta/(1-delta) * I`). Smaller
delta → smoother beta, slower adaptation; `delta -> 0` converges to static
OLS. `r` is the observation noise in price-unit variance; both must satisfy
0 < delta < 1 (strictly) and 0 < r < ∞ (NaN and inf are rejected) or the
call errors. `km.z` are innovation z-scores — a window-free alternative
trading signal.

---

## 7. How do I run a full pairs backtest with costs?

**Python**

```python
from statarb import (BacktestConfig, backtest_pair, config_from_dict,
                     engle_granger, load_config, load_csv)

pair = load_csv("../data/pair_equity.csv")
y, x = pair["stock_y"], pair["stock_x"]
eg = engle_granger(y[:900], x[:900])                 # in-sample hedge
# config.json turns the risk limits on (P&L stop 2%, max hold 60d, cool-down
# 20d); BacktestConfig() alone leaves them off. Sizing defaults to "beta_hedged".
cfg = config_from_dict(load_config("../data/config.json"))
res = backtest_pair(y, x, eg.alpha, eg.beta, cfg)
m = res.metrics
print(f"total {m['total_return']:.4%}  sharpe {m['sharpe']:.3f}  "
      f"maxDD {m['max_drawdown']:.2%}  trades {int(m['n_trades'])}")
# golden: 3.6955%, 0.301, 6.32%, 29 trades
print("stopped trades:", res.n_stops)             # 4
# the same run with explicit knobs:
cfg2 = BacktestConfig(window=60, entry_z=2.0, exit_z=0.5, stop_z=4.0,
                      cost_bps=1.0, slippage_bps=0.5, capital=1e6,
                      sizing="beta_hedged", pnl_stop_frac=0.02,
                      max_hold_days=60, cooldown_days=20)
```

**C++**

```cpp
#include <statarb/backtest.hpp>

// Construct through the validating constructor (window, entry_z, exit_z,
// stop_z, cost_bps, slippage_bps, capital). This seven-argument form leaves
// the v1.2.0 fields at their defaults (beta_hedged sizing, risk limits off);
// API_SPEC.md section 8 lists sizing, pnl_stop_frac, max_hold_days and
// cooldown_days.
// The members are public, but backtest_pair re-validates the struct, so a
// later `cfg.exit_z = 5.0;` is caught with std::invalid_argument at run time.
const statarb::BacktestConfig cfg(60, 2.0, 0.5, 4.0, 1.0, 0.5, 1e6);
statarb::BacktestResult res = statarb::backtest_pair(y_all, x_all, eg.alpha, eg.beta, cfg);
std::cout << "total " << res.metrics.total_return << "  sharpe " << res.metrics.sharpe
          << "  maxDD " << res.metrics.max_drawdown << "  trades " << res.trades.size()
          << "  stops " << res.n_stops() << "\n";               // limits off: -0.0249, -0.169, 0.1392, 31 (2 stops)
```

**Rust**

```rust
use statarb::{backtest_pair, BacktestConfig};

// BacktestConfig::new validates (this form: beta_hedged sizing, limits off; see
// API_SPEC.md section 8 for the other fields); a struct literal that bypasses
// it is re-validated inside backtest_pair / walk_forward (Err(InvalidParam)).
let cfg = BacktestConfig::new(60, 2.0, 0.5, 4.0, 1.0, 0.5, 1e6)?;
let res = backtest_pair(y_all, x_all, eg.alpha, eg.beta, &cfg)?;
println!("total {:.4}  sharpe {:.3}  maxDD {:.4}  trades {}  stops {}",
    res.metrics.total_return, res.metrics.sharpe, res.metrics.max_drawdown,
    res.trades.len(), res.n_stops());                             // limits off: -0.0249, -0.169, 0.1392, 31 (2 stops)
```

**Java**

```java
import com.quant.statarb.Backtest;

Backtest.Config cfg = new Backtest.Config(60, 2.0, 0.5, 4.0, 1.0, 0.5, 1e6);
Backtest.Result res = Backtest.run(yAll, xAll, eg.alpha(), eg.beta(), cfg);
System.out.printf("total %.4f  sharpe %.3f  maxDD %.4f  trades %d  stops %d%n",
    res.metrics().get("total_return"), res.metrics().get("sharpe"),
    res.metrics().get("max_drawdown"), res.trades().size(), res.nStops());
```

Pinned mechanics worth knowing: signals computed at the day-t close execute
at the day t+1 close (no look-ahead); with the default `sizing =
"beta_hedged"` the legs are `capital / (y + beta*x)` units of y against
`beta` times that of x, so gross notional equals `capital` and the P&L tracks
the spread (not dollar-neutral); `"equal_dollar"` puts `capital/2` on each
leg (neutral at entry, but it trades a different spread when beta differs
from y/x); `beta <= 0` is an error. |z| ≥ stop_z while flat means *no
entry*. The risk limits are checked at the signal step in this order:
`pnl_stop_frac` (a stop), then the z-stop, then `max_hold_days` (an exit that
is not a stop); after a stopped exit, `cooldown_days` blocks new entries. A
position open on the last day is force-closed as a normal trade, while a
pending *entry* on the last day is discarded (no zero-day round trip).
Thresholds and costs must be finite — a config loaded from JSON with
`"cost_bps": NaN` is refused rather than producing a NaN equity curve and a
Sharpe of 0.00. A window too short to reach `entry_z` is refused too:
|z| cannot exceed `(W-1)/sqrt(W)`, so W = 5 never reaches 2. The 2% / 60 /
20 limits in `config.json` are round numbers, not tuned.

---

## 8. How do I run an out-of-sample evaluation (holdout and rolling)?

**Python**

```python
from statarb import (config_from_dict, load_config, load_csv,
                     rolling_walk_forward, walk_forward)

pair = load_csv("../data/pair_equity.csv")
y, x = pair["stock_y"], pair["stock_x"]
cfg = config_from_dict(load_config("../data/config.json"))   # limits on

# walk_forward is a SINGLE HOLDOUT: one fit on 60%, one trade of the other 40%.
wf = walk_forward(y, x, cfg, insample_frac=0.6)
print(f"hedge from days 0..{wf.split-1}: alpha {wf.alpha:.4f} beta {wf.beta:.4f}")
print(f"IS  sharpe {wf.insample.metrics['sharpe']:.2f}")   # +1.40
print(f"OOS sharpe {wf.oos.metrics['sharpe']:.2f}")        # -0.78 (break!)

# rolling_walk_forward refits every `test` days on the last `train` days.
rw = rolling_walk_forward(y, x, cfg, train=500, test=100)
print(len(rw.blocks), "blocks from day", rw.start_day)     # 10 blocks from day 500
print(f"total {rw.metrics['total_return']:.2%}  sharpe {rw.metrics['sharpe']:.2f}")
# -0.29%, -0.05. The last three blocks (indices 7-9) fit beta <= 0 and stayed
# flat (block.skipped is True), which flatters the result.
print([b.skipped for b in rw.blocks])
```

**C++**

```cpp
statarb::WalkForwardResult wf = statarb::walk_forward(y_all, x_all, cfg, 0.6);
std::cout << "split " << wf.split << "  IS " << wf.insample.metrics.sharpe
          << "  OOS " << wf.oos.metrics.sharpe << "\n";           // limits off: 900, +1.60, -1.54
```

**Rust**

```rust
let wf = statarb::walk_forward(y_all, x_all, &cfg, 0.6)?;
println!("split {}  IS {:.2}  OOS {:.2}", wf.split, wf.insample.metrics.sharpe, wf.oos.metrics.sharpe);
```

**Java**

```java
var wf = Backtest.walkForward(yAll, xAll, cfg, 0.6);
System.out.printf("split %d  IS %.2f  OOS %.2f%n", wf.split(),
    wf.insample().metrics().get("sharpe"), wf.oos().metrics().get("sharpe"));
```

The hedge is estimated by OLS on the first 60% only, and the out-of-sample
z-score warms up inside the OOS segment — nothing crosses the split. The
bundled pair degrades out-of-sample because the OOS window contains the
day-1050 regime break; that degradation is the lesson, not a bug. With the
limits off the same split gives OOS -12.35% and 15.28% drawdown; with them on,
-4.91% and 6.86%: a smaller loss, still a loss.

`rolling_walk_forward` fits OLS on `[k*test, k*test+train)`, trades the next
`test` days with a z-score warmed up on earlier days only, is flat at every
block end, and stitches block P&L into one equity curve with `T - train`
points starting at day `train`. A block whose fit gives `beta <= 0` is
`skipped` (flat, zero metrics), not an error. The ports expose the same
function with the same arguments; see API_SPEC.md section 8.

---

## 9. How do I detect a spurious pair and refuse to trade it?

**Python**

```python
from statarb import adf_test, engle_granger, load_csv, ols

walks = load_csv("../data/random_walks.csv")
a, b = walks["walk_a"], walks["walk_b"]
print("naive slope t-stat:", round(float(ols(a, b).tstats[1]), 1))  # ~8.4 (!)
eg = engle_granger(a, b, maxlag=1)
if not eg.cointegrated(0.05):                      # golden: not cointegrated
    print("spurious relationship — refusing to backtest")
else:
    raise AssertionError("should not happen on independent walks")
```

**C++**

```cpp
statarb::EGResult eg_sp = statarb::engle_granger(walks.at("walk_a"), walks.at("walk_b"), 1);
if (!eg_sp.cointegrated(0.05)) {
    std::cout << "spurious — no backtest\n";       // this branch runs
}
```

**Rust**

```rust
let eg_sp = statarb::engle_granger(&walks["walk_a"], &walks["walk_b"], 1)?;
if !eg_sp.cointegrated(0.05)? {
    println!("spurious — no backtest");            // this branch runs
}
```

**Java**

```java
EngleGranger.Result egSp = EngleGranger.test(walks.get("walk_a"), walks.get("walk_b"), 1);
if (!egSp.cointegrated(0.05)) {
    System.out.println("spurious — no backtest");  // this branch runs
}
```

Gate every candidate pair this way. A large slope t-statistic on levels is
meaningless (Granger–Newbold); only a stationary residual licenses a trade.

---

## 10. How do I run plain OLS and read the diagnostics?

**Python**

```python
from statarb import load_csv, ols

pair = load_csv("../data/pair_equity.csv")
fit = ols(pair["stock_y"], pair["stock_x"], intercept=True)
print(f"alpha {fit.params[0]:.6f}  beta {fit.params[1]:.6f}  R2 {fit.r2:.4f}")
# golden full-sample: alpha 13.2994, beta 1.5178, R2 0.1385 (break-polluted!)
print(f"se(beta) {fit.se[1]:.4f}  t(beta) {fit.tstats[1]:.2f}  nobs {fit.nobs}")
```

**C++**

```cpp
#include <statarb/ols.hpp>

statarb::OLSResult fit = statarb::ols(y_all, x_all, /*intercept=*/true);
// fit.params[0] = intercept, fit.params[1] = slope; fit.se, fit.tstats, fit.resid, fit.r2
std::cout << "beta = " << fit.params[1] << ", r2 = " << fit.r2 << "\n";   // 1.5178, 0.1385
```

**Rust**

```rust
let fit = statarb::ols(y_all, x_all, true)?;
println!("beta = {:.6}, r2 = {:.4}", fit.params[1], fit.r2);      // 1.517790, 0.1385
```

**Java**

```java
import com.quant.statarb.Ols;

Ols.Result fit = Ols.fit(yAll, xAll, true);
System.out.printf("beta = %.6f, r2 = %.4f%n", fit.params()[1], fit.r2());  // 1.517790, 0.1385
```

The solver is pinned (normal equations + LU); a condition-number guard on
X'X (limit 1e12) rejects collinear designs — including a constant regressor
next to the intercept — with a clear error instead of returning garbage.

---

## 11. How do I load the bundled data and configuration?

**Python**

```python
from statarb import load_config, load_csv

cfg = load_config("../data/config.json")     # flat dict of scalars
pair = load_csv("../data/pair_equity.csv")   # {column_name: float array}
print(sorted(pair), "window =", cfg["window"], "capital =", cfg["capital"])
```

**C++**

```cpp
auto config = statarb::load_config("../data/config.json");   // std::map<std::string, double>
auto pair2  = statarb::load_csv("../data/pair_equity.csv");  // std::map<std::string, std::vector<double>>
std::cout << "window = " << config.at("window") << ", days = " << pair2.at("day").size() << "\n";
```

**Rust**

```rust
let config = statarb::load_config(Path::new("../data/config.json"))?;   // HashMap<String, f64>
let pair2  = statarb::load_csv(Path::new("../data/pair_equity.csv"))?;  // HashMap<String, Vec<f64>>
println!("window = {}, days = {}", config["window"], pair2["day"].len());
```

**Java**

```java
var config = DataIo.loadConfig(Paths.get("../data/config.json"));  // Map<String, Double>
var pair2  = DataIo.loadCsv(Paths.get("../data/pair_equity.csv")); // Map<String, double[]>
System.out.printf("window = %.0f, days = %d%n", config.get("window"), pair2.get("day").length);
```

The loaders validate: missing file, empty file, ragged rows, non-numeric or
non-finite cells, duplicate column names, and a `day` column that is not
strictly increasing (unsorted or duplicated rows) all raise descriptive
errors; `load_config` rejects non-finite or non-numeric values. Python, C++
and Java ship their own tiny CSV/JSON readers; the Rust crate hand-rolls the
CSV reader and uses `serde_json` for the flat config file.

---

## 12. How do I handle invalid inputs?

Every validation failure is a typed error with a descriptive message —
never a NaN result or a crash. Python raises `ValueError`; C++ throws
`std::invalid_argument`; Rust returns `Err(StatarbError)`; Java throws
`IllegalArgumentException`.

**Python**

```python
import numpy as np
from statarb import BacktestConfig, adf_test, kalman_hedge, ols, rolling_zscore

for fn, args in [
    (ols,            (np.ones(50), np.arange(50.0))),          # constant y
    (adf_test,       (np.sin(np.arange(14.0)),),),             # too short
    (rolling_zscore, (np.arange(10.0), 11)),                   # window > T
    (kalman_hedge,   (np.arange(9.0), np.arange(10.0))),       # length mismatch
]:
    try:
        fn(*args)
    except ValueError as e:
        print(f"{fn.__name__}: {e}")
try:
    BacktestConfig(entry_z=2.0, exit_z=2.5)                    # ordering broken
except ValueError as e:
    print("config:", e)
```

**C++**

```cpp
#include <cmath>
#include <stdexcept>

try {
    statarb::kalman_hedge(y, x, /*delta=*/1.5, /*r=*/1e-3);   // delta not in (0,1)
} catch (const std::invalid_argument& e) {
    std::cerr << e.what() << "\n";
}
try {
    statarb::BacktestConfig(60, 2.0, 0.5, 4.0, std::nan(""), 0.5, 1e6);   // NaN cost
} catch (const std::invalid_argument& e) {
    std::cerr << e.what() << "\n";                           // thresholds and costs must be finite
}
```

**Rust**

```rust
match statarb::kalman_hedge(&y, &x, 1.5, 1e-3) {
    Err(e) => eprintln!("{e}"),          // delta must lie strictly inside (0,1)
    Ok(_) => unreachable!(),
}
if let Err(e) = BacktestConfig::new(60, 2.0, 0.5, 4.0, f64::NAN, 0.5, 1e6) {
    eprintln!("{e}");                    // thresholds and costs must be finite
}
```

**Java**

```java
try {
    Kalman.hedge(y, x, 1.5, 1e-3);       // delta not in (0,1)
} catch (IllegalArgumentException e) {
    System.err.println(e.getMessage());
}
try {
    new Backtest.Config(60, 2.0, 0.5, 4.0, Double.NaN, 0.5, 1e6);   // NaN cost
} catch (IllegalArgumentException e) {
    System.err.println(e.getMessage());  // thresholds and costs must be finite
}
```

The full list of guarded conditions (length mismatch, non-finite input,
constant series, collinearity, short series, window bounds, non-positive
prices, delta/r/maxlag/nvars/alpha domains, backtest threshold ordering and
finiteness, CSV day-column monotonicity) is in API_SPEC.md §9.

---

## 13. How do I update a rolling z-score or a Kalman hedge one tick at a time?

```python
import numpy as np
from statarb import (KalmanStep, RollingZ, engle_granger, kalman_hedge,
                     load_csv, rolling_zscore, spread)

pair = load_csv("../data/pair_equity.csv")
y, x = pair["stock_y"], pair["stock_x"]
eg = engle_granger(y[:900], x[:900])
s = spread(y, x, eg.alpha, eg.beta)

rz = RollingZ(60)                          # ring buffer, O(1) per update
z = np.array([rz.update(v) for v in s])    # NaN during warm-up and on zero variance
batch = rolling_zscore(s, 60)
ok = ~np.isnan(batch)
print(np.max(np.abs(z[ok] - batch[ok])))   # 5.2e-13 here; the tests bound it at 1e-9

ks = KalmanStep(1e-4, 1e-3)                # note the argument order below: x first
alpha, beta, zk = ks.update(x[0], y[0])
```

`KalmanStep` runs the same scalar recursion as `kalman_hedge` and matches it to
1e-12. `RollingZ` keeps running sums, so it is not bitwise equal to the batch
function and drifts slowly with the update count (about 5e-8 after 50,000
updates at a level of 1e4). The batch function is the trading reference; use
the streaming forms to follow a live feed, then re-check against batch. A
non-finite input is an error and leaves the state unchanged. C++, Rust and
Java expose `RollingZ` and `KalmanStep` with the same contract (API_SPEC.md
sections 6 and 7).

## 14. How do I choose the ADF lag, and test a pair in both directions?

```python
from statarb import engle_granger_both, select_lag

lag = select_lag(eg.resid, max_lag=4, nvars=2)     # smallest BIC on a common sample; 0 on this data
both = engle_granger_both(y[:900], x[:900], maxlag=lag)
print(both.cointegrated())                          # True only if BOTH directions reject
print(both.y_on_x.adf.tstat, both.x_on_y.adf.tstat) # -6.03, -5.99 at lag 0 (-5.82, -5.78 at lag 1)
```

A fixed lag can have the wrong size when the errors are autocorrelated (the
review measured 31–33% rejections at a nominal 5% under MA(-0.5) errors), and
the Engle–Granger verdict can differ between `y on x` and `x on y`. Selecting
the lag and requiring both directions is the cautious default for screening.
`select_lag` and `engle_granger_both` exist in every port.

## 15. How do I put an interval on a result, and scan many pairs? (Python only)

```python
import numpy as np
from statarb.research import (block_bootstrap_sharpe_ci, cost_sensitivity,
                              cusum_break_test, scan_pairs)

returns = np.diff(rw.equity) / cfg.capital         # rw from recipe 8
ci = block_bootstrap_sharpe_ci(returns, seed=1)    # circular blocks, seeded
print(f"{ci.sharpe:.2f} [{ci.lower:.2f}, {ci.upper:.2f}]")   # -0.05 [-1.11, 1.45]

rows = cost_sensitivity(y[:900], x[:900], eg.alpha, eg.beta, cfg)
print([(r["total_bps"], round(r["sharpe"], 2)) for r in rows])
# in-sample Sharpe at 0, 1.5, 3, 5, 10, 20 bps: 1.52 1.40 1.28 1.13 0.73 -0.04

print(cusum_break_test(eg.resid).pvalue)           # 0.149 in-sample, no break flagged
print(cusum_break_test(y - (eg.alpha + eg.beta * x)).pvalue)   # 0.012 full sample

for row in scan_pairs({"Y": y[:900], "X": x[:900]}):
    print(row.y, row.x, row.adjusted_pvalue, row.selected)
```

`statarb.research` also has `probabilistic_sharpe`, `deflated_sharpe`,
`expected_max_sharpe`, `benjamini_hochberg` and `kalman_grid_mle` (which
reports the innovation z sd and lag-1 autocorrelation; a well-specified
filter has sd near 1 and autocorrelation near 0). They are judgement aids, not
part of the trading path, and are not in the goldens. An interval says how
much the number could move by chance; it does not make a synthetic result
real.

---

## 16. How do I run a real-data study honestly?

Commit a preregistration first (recipe 23 has a template), with the data, rules, dates and pass gates. Run the development period for bugs only, then the holdout once, and report a failure as a null. Worked examples: [`research/real_data_2026_11`](research/real_data_2026_11) and [`research/sector_residual_2026_11`](research/sector_residual_2026_11). Both were null.

## 17. How do I run a paper trade end to end and read the audit trail? (Python only)

**Paper trading only.** `statarb.trading` simulates fills. It has no network
code, no credentials and no live-order path. ADV (average daily volume) is an
assumption because the CSV has no volume, and every cost below is assumed. The
prices are synthetic.

```python
import json
import tempfile
from collections import Counter
from pathlib import Path

from statarb import load_config, load_csv, walk_forward
from statarb.backtest import config_from_dict
from statarb.trading import (ExecutionConfig, PaperBrokerConfig, PaperRunConfig,
                             RiskLimits, run_paper)

data = load_csv("../data/pair_equity.csv")
y, x = data["stock_y"], data["stock_x"]
rule = config_from_dict(load_config("../data/config.json"))
wf = walk_forward(y, x, rule)                     # hedge fitted on the first 60%
capital = rule.capital

run = PaperRunConfig(adv_y=100_000.0, adv_x=100_000.0, seed=42)   # ADV is assumed
broker = PaperBrokerConfig(initial_cash=capital, half_spread_bps=1.0, spread_jitter=0.3,
                           impact_bps=30.0, max_participation=0.02, liquidity_jitter=0.3,
                           commission_bps=0.5, borrow_bps_annual=50.0,
                           leg_latency_bars={"stock_x": 1})
limits = RiskLimits(max_gross=1.3 * capital, max_net=0.6 * capital,
                    max_name_notional=0.9 * capital, max_order_adv_frac=0.05,
                    price_band_frac=0.15, max_drawdown_frac=0.10,
                    locates=frozenset({"stock_y", "stock_x"}))
execution = ExecutionConfig(max_imbalance_frac=0.5, timeout_bars=8)

with tempfile.TemporaryDirectory() as folder:
    path = Path(folder) / "audit.jsonl"
    res = run_paper(y[wf.split:], x[wf.split:], wf.alpha, wf.beta, rule, run, broker, limits,
                    execution, str(path))
    lines = [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines()]

a = res.attribution
print(f"net P&L {res.equity[-1] - capital:,.0f} = at mid {a.spread:,.0f} - costs {a.costs:,.0f} - borrow {a.borrow:,.0f}")
print("entries", res.entries, "outcomes", res.outcomes, "fills", res.fills)
print("rejections", res.rejections or "none", "| kill switch", res.kill_reason or "not tripped",
      "| breaks", len(res.breaks))
print("audit records", len(lines), dict(Counter(r["event"] for r in lines)))

first = next(r["seq"] for r in lines if r["event"] == "signal" and r["action"] == "enter")
print("\nFirst entry, in audit order:")
for r in lines[first:first + 9]:
    keep = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()
            if k in ("seq", "bar", "event", "action", "z", "symbol", "side", "qty", "price",
                     "approved", "reasons", "reducing", "client_id")}
    print(" ", json.dumps(keep, sort_keys=True))
```

Output (a real run; the audit file is in a temporary folder and deleted):

```text
net P&L 3,953 = at mid 12,841 - costs 7,383 - borrow 1,505
entries 8 outcomes {'FILLED': 7, 'FLAT': 8, 'UNWOUND': 0} fills 119
rejections none | kill switch not tripped | breaks 0
audit records 227 {'signal': 16, 'risk_decision': 92, 'fill': 119}

First entry, in audit order:
  {"action": "enter", "bar": 106, "event": "signal", "seq": 0, "z": 2.2011}
  {"approved": true, "bar": 106, "client_id": "pair-00000001", "event": "risk_decision", "qty": 1549, "reasons": [], "reducing": false, "seq": 1, "side": "SELL", "symbol": "stock_y"}
  {"approved": true, "bar": 106, "client_id": "pair-00000002", "event": "risk_decision", "qty": 2539, "reasons": [], "reducing": false, "seq": 2, "side": "BUY", "symbol": "stock_x"}
  {"bar": 107, "client_id": "pair-00000001", "event": "fill", "price": 171.2854, "qty": 1549, "seq": 3, "side": "SELL", "symbol": "stock_y"}
  {"bar": 108, "client_id": "pair-00000002", "event": "fill", "price": 92.4641, "qty": 1925, "seq": 4, "side": "BUY", "symbol": "stock_x"}
  {"approved": true, "bar": 108, "client_id": "pair-00000003", "event": "risk_decision", "qty": 1925, "reasons": [], "reducing": false, "seq": 5, "side": "BUY", "symbol": "stock_x"}
  {"approved": true, "bar": 108, "client_id": "pair-00000004", "event": "risk_decision", "qty": 1174, "reasons": [], "reducing": false, "seq": 6, "side": "SELL", "symbol": "stock_y"}
  {"bar": 109, "client_id": "pair-00000002", "event": "fill", "price": 92.8183, "qty": 614, "seq": 7, "side": "BUY", "symbol": "stock_x"}
  {"bar": 109, "client_id": "pair-00000004", "event": "fill", "price": 172.9098, "qty": 1174, "seq": 8, "side": "SELL", "symbol": "stock_y"}
```

How to read it:

- **Attribution.** `net P&L = at mid - costs - borrow`: 12,841 - 7,383 - 1,505
  = 3,953. "At mid" values the trades at mid prices; the rest is what the
  simulated market charged. `python demo_paper.py` splits the costs into
  half-spread (1,471), impact (5,171) and commission (742).
- **Do not read the sign as an edge.** The same data, frictionless, loses
  45,526 and the backtest loses 49,084. Delaying both legs by two bars with no
  costs gives -16,782 and by four +19,743, so this P&L is very sensitive to
  entry timing. It is one synthetic path with about 9 trades.
- **Audit order is causal order.** `seq` 0 is the signal at bar 106 (z = 2.20,
  short the spread). `seq` 1 and 2 are the risk decisions for the two child
  orders, both approved (`reducing` false: they add risk). `seq` 3 is the fill of
  the y leg at bar 107; `seq` 4 is the x leg at bar 108, which fills 1,925 of
  its 2,539 shares (it has one bar of latency and a participation cap). Then the
  executor, seeing the legs at different progress, sends a second pair of
  children (`seq` 5 and 6, sized 1,925 and 1,174) and fills continue. Section
  10a.2 of LEARN.md works through those sizes.
- **Event kinds.** `signal`, `risk_decision`, `fill`, `flatten` (a stop, the
  kill switch or the end-of-data wind-down), `reconciliation_break`,
  `kill_switch_tripped` and `kill_switch_reset`. The file is append-only
  JSONL, keyed by bar number rather than wall-clock time, so a seeded run
  replays byte for byte. `run_paper(..., audit_path=None)` keeps records in
  memory only.
- **Checks to run on every result.** `res.rejections`, `res.kill_reason`,
  `res.breaks`, `res.outcomes['UNWOUND']` and `res.final_positions`. Here: no
  rejections, no kill switch, no breaks, nothing unwound, nothing left open.

---

## 18. How do I add a risk limit and test it? (Python only)

The checks live in `RiskEngine.check`. The least invasive way to add one is a
subclass that calls the base check first, then applies the extra limit to
orders that increase risk. Keep the rule that **reducing orders always pass**:
a limit that blocks the order that cuts exposure makes the breach worse.

```python
from dataclasses import replace
from enum import Enum

from statarb.trading import Order, Quote, RiskEngine, RiskLimits, Side


class ExtraReason(Enum):
    MAX_SHARES_NAME = "MAX_SHARES_NAME"


class SharesCapEngine(RiskEngine):
    """RiskEngine plus one limit: at most `max_shares` (long or short) in any one name."""

    def __init__(self, limits, max_shares, sink=None):
        super().__init__(limits, sink)
        self.max_shares = max_shares

    def check(self, order, positions, open_orders, quotes, bar):
        decision = super().check(order, positions, open_orders, quotes, bar)
        if decision.reducing:                      # the rule that lets risk come off
            return decision
        # worst case on the order's side, as in the base class
        held = positions.get(order.symbol, 0)
        working = sum(o.remaining for o in open_orders
                      if o.symbol == order.symbol and o.side is order.side)
        after = abs(held + int(order.side) * (working + order.qty))
        if after <= self.max_shares:
            return decision
        reasons = decision.reasons + (ExtraReason.MAX_SHARES_NAME,)
        self._record({"event": "risk_decision", "bar": bar, "client_id": order.client_id,
                      "symbol": order.symbol, "approved": False,
                      "reasons": [r.value for r in reasons], "note": "extra limit"})
        return replace(decision, approved=False, reasons=reasons)


limits = RiskLimits(max_gross=2e6, max_net=1e6, max_name_notional=1e6,
                    locates=frozenset({"A"}))
engine = SharesCapEngine(limits, max_shares=1_000)
quotes = {"A": Quote(mid=100.0, adv=1e6, prev_close=100.0)}

def test(side, qty, held, expect_ok):
    order = Order("t1", "A", side, qty)
    d = engine.check(order, {"A": held} if held else {}, [], quotes, bar=0)
    print(f"{side.name:4} {qty:5} held {held:5} -> approved={d.approved} "
          f"reducing={d.reducing} reasons={[r.value for r in d.reasons]}")
    assert d.approved is expect_ok

test(Side.BUY, 900, 0, True)            # inside the cap
test(Side.BUY, 1_100, 0, False)         # over the cap
test(Side.BUY, 200, 900, False)         # 900 held + 200 = 1,100
test(Side.SELL, 1_500, 1_200, True)     # crosses to -300: not reducing, but inside the cap
test(Side.SELL, 2_000, 900, False)      # crosses to -1,100: not reducing, over the cap
test(Side.SELL, 500, 1_500, True)       # already over the cap (price drift, a fill): reducing, passes

engine.trip(bar=1, reason="MANUAL")     # kill switch: new risk is refused, reducing is not
test(Side.BUY, 10, 0, False)
test(Side.SELL, 500, 1_500, True)
print("all checks passed;", len(engine.log), "decisions logged")
```

```text
BUY    900 held     0 -> approved=True reducing=False reasons=[]
BUY   1100 held     0 -> approved=False reducing=False reasons=['MAX_SHARES_NAME']
BUY    200 held   900 -> approved=False reducing=False reasons=['MAX_SHARES_NAME']
SELL  1500 held  1200 -> approved=True reducing=False reasons=[]
SELL  2000 held   900 -> approved=False reducing=False reasons=['MAX_SHARES_NAME']
SELL   500 held  1500 -> approved=True reducing=True reasons=[]
BUY     10 held     0 -> approved=False reducing=False reasons=['KILL_SWITCH']
SELL   500 held  1500 -> approved=True reducing=True reasons=[]
all checks passed; 12 decisions logged
```

What the cases pin down: inside the cap passes; over the cap is refused with the
reason recorded in the log; a sell that crosses zero is checked as new risk (the
sell of 2,000 against 900 held lands at -1,100 and is refused); an order that only moves a
position back toward zero passes even when the position is already over the cap
(1,500 shares against a cap of 1,000); and under the kill switch new risk is
refused (`KILL_SWITCH`) while a reducing sell still goes through. The logged
decision count includes the extra refusals the subclass appended. For a permanent
limit, add the field to `RiskLimits` and a member to `RiskReason` in
`statarb/trading/risk.py`, and put the cases above into
`python/tests/test_trading_risk_execution.py`.

Remember what limits can and cannot do: they are checked at decision prices, so a
later price move can take exposure past a limit with no trade.

---

## 19. How do I write a `Broker` adapter against the interface? (Python only, paper only)

`Broker` is a `typing.Protocol` with seven methods: `submit`, `cancel`, `order`,
`open_orders`, `positions`, `cash` and `fills`. Anything with those methods can
drive `PairExecutor`. A live adapter would translate each call to a venue API;
that is an owner decision and nothing in this repository does it. The example
below forwards to `PaperBroker` and counts calls, so it can be run and tested
with no connection of any kind.

```python
from collections import Counter

from statarb.trading import (Broker, ExecutionConfig, Order, PairExecutor, PaperBroker,
                             PaperBrokerConfig, Quote, RiskEngine, RiskLimits, Side)


class RecordingBroker:
    """An adapter: same seven methods as `Broker`, here forwarding to PaperBroker.

    A real adapter would translate each call to a venue API instead. This one is
    paper only and opens no connection. `on_bar` is the simulator's clock, not part
    of `Broker`; a live adapter would receive fills from the venue instead.
    """

    def __init__(self, inner):
        self.inner = inner
        self.calls = Counter()

    def submit(self, order):
        self.calls["submit"] += 1
        return self.inner.submit(order)

    def cancel(self, client_id):
        self.calls["cancel"] += 1
        return self.inner.cancel(client_id)

    def order(self, client_id):
        return self.inner.order(client_id)

    def open_orders(self):
        return self.inner.open_orders()

    def positions(self):
        return self.inner.positions()

    def cash(self):
        return self.inner.cash()

    def fills(self):
        return self.inner.fills()

    def on_bar(self, bar, quotes):
        return self.inner.on_bar(bar, quotes)


broker = RecordingBroker(PaperBroker(PaperBrokerConfig(max_participation=0.05), seed=1))
print("implements Broker:", isinstance(broker, Broker))        # structural check only

# Contract checks a new adapter should pass before anything drives it.
a = Order("c-1", "A", Side.BUY, 100)
first = broker.submit(a)
again = broker.submit(Order("c-1", "A", Side.BUY, 100))        # same id, same order: no-op
assert (first.client_id, again.status, len(broker.open_orders())) == ("c-1", first.status, 1)
try:
    broker.submit(Order("c-1", "A", Side.BUY, 999))            # same id, different order
except ValueError as err:
    print("reused id refused:", err)
assert broker.cancel("c-1").status.value == "CANCELLED"
assert broker.cancel("c-1").status.value == "CANCELLED"        # cancelling a finished order changes nothing
assert broker.open_orders() == []

# Drive a pair through it, with risk checks, exactly as run_paper does.
limits = RiskLimits(max_gross=5e6, max_net=2e6, max_name_notional=3e6,
                    locates=frozenset({"A", "B"}))
executor = PairExecutor(broker, RiskEngine(limits), ("A", "B"), ExecutionConfig(0.5, 6))
executor.enter(0, {"A": 4_000, "B": -3_000})
for bar in range(1, 12):
    quotes = {"A": Quote(50.0, 1e5, 50.0), "B": Quote(80.0, 1e5, 80.0)}
    broker.on_bar(bar, quotes)
    executor.step(bar, quotes)
    if not executor.active:
        break
print("bar", bar, "outcome", executor.last_outcome.value, "positions", broker.positions())
print("calls", dict(broker.calls), "fills", len(broker.fills()))
```

```text
implements Broker: True
reused id refused: client id c-1 reused for a different order
bar 3 outcome FILLED positions {'A': 4000, 'B': -3000}
calls {'submit': 7, 'cancel': 2} fills 4
```

The contract an adapter must keep, which the first block of the script checks:

- `submit` of an order with a client id already seen and the same symbol, side
  and quantity returns the original and does nothing; the same id with anything
  different raises `ValueError`. Idempotence makes a retried send safe.
- `cancel` of a finished order changes nothing.
- Returned orders are snapshots; mutating one does not change the broker.
- Fills carry a unique `fill_id`; the ledger refuses to book one twice.

`isinstance(x, Broker)` checks method names only, not behaviour. Keep the
checks above in a test of your own. `on_bar` is the simulator's clock and is not
part of the interface; a real venue delivers fills asynchronously.

---

## 20. How do I reconcile a ledger and diagnose a break? (Python only)

`Ledger` rebuilds positions and cash from fills and its own borrow accrual, so
`reconcile(broker, bar)` compares two independent calculations. Each break has a
kind, a key and both values, and is also written to the audit log.

```python
from dataclasses import replace

from statarb.trading import Ledger, Order, PaperBroker, PaperBrokerConfig, Quote, Side

cfg = PaperBrokerConfig(initial_cash=1_000_000.0, half_spread_bps=1.0, impact_bps=10.0,
                        max_participation=0.5, commission_bps=0.5)
broker = PaperBroker(cfg, seed=7)
ledger = Ledger(cfg.initial_cash)
quotes = {"A": Quote(50.0, 1e5, 50.0), "B": Quote(80.0, 1e5, 80.0)}

broker.submit(Order("o-1", "A", Side.BUY, 1_000))
broker.submit(Order("o-2", "B", Side.SELL, 500))
broker.on_bar(0, quotes)
fills = broker.fills()
print("broker fills:", [(f.fill_id, f.symbol, f.signed_qty, round(f.price, 4)) for f in fills])

def show(title, breaks):
    print(f"\n{title}: {len(breaks)} break(s)")
    for b in breaks:
        print(f"  {b.kind.value:18} {b.key:8} ledger {b.ledger:>14,.2f}  broker {b.broker:>14,.2f}")

# Case 1: the ledger never saw the second fill (a dropped message).
ledger.apply_fill(fills[0])
show("1. second fill not booked", ledger.reconcile(broker, bar=0))

# Case 2: ingest() is safe with the full history; it books only what is missing.
print("\ningest booked", ledger.ingest(broker.fills()), "new fill")
show("2. after ingest", ledger.reconcile(broker, bar=0))

# Case 3: same fill ids, but the ledger booked a wrong price on one (a bad feed).
ledger2 = Ledger(cfg.initial_cash)
ledger2.apply_fill(replace(fills[0], price=fills[0].price + 0.05))
ledger2.apply_fill(fills[1])
show("3. wrong price booked", ledger2.reconcile(broker, bar=0))

# Case 4: the ledger holds a fill the broker never reported.
ghost = replace(fills[0], fill_id="ghost#1")
ledger.apply_fill(ghost)
show("4. fill unknown to broker", ledger.reconcile(broker, bar=0))
print("\naudit events written:", [r["event"] for r in ledger.audit.records])
```

```text
broker fills: [('o-1#1', 'A', 1000, 50.01), ('o-2#2', 'B', -500, 79.9863)]

1. second fill not booked: 3 break(s)
  POSITION           B        ledger           0.00  broker        -500.00
  CASH               cash     ledger     949,987.50  broker     989,978.67
  MISSING_IN_LEDGER  o-2#2    ledger           0.00  broker           1.00

ingest booked 1 new fill

2. after ingest: 0 break(s)

3. wrong price booked: 1 break(s)
  CASH               cash     ledger     989,928.67  broker     989,978.67

4. fill unknown to broker: 3 break(s)
  POSITION           A        ledger       2,000.00  broker       1,000.00
  CASH               cash     ledger     939,966.17  broker     989,978.67
  MISSING_AT_BROKER  ghost#1  ledger           1.00  broker           0.00

audit events written: ['fill', 'reconciliation_break', 'reconciliation_break', 'reconciliation_break', 'fill', 'fill', 'reconciliation_break', 'reconciliation_break', 'reconciliation_break']
```

Reading the pattern:

| Breaks seen | Likely cause | Fix |
|---|---|---|
| `POSITION` + `CASH` + `MISSING_IN_LEDGER` | the ledger missed a fill (dropped message) | `ledger.ingest(broker.fills())`; it books only what it has not seen (case 2 shows zero breaks) |
| `CASH` only | positions agree, so a wrong price, fee or borrow was booked (case 3: a price 0.05 high on 1,000 shares is a 50.00 cash break) | find the fill whose booked value differs; correct the source, do not adjust cash |
| `POSITION` + `CASH` + `MISSING_AT_BROKER` | the ledger booked a fill the broker never reported (case 4) | do not trade until it is explained; the fill may be a duplicate or belong to another account |

In `run_paper`, any break trips the kill switch with reason
`RECONCILIATION_BREAK` and the runner flattens. Clearing it needs
`reset_kill_switch(bar, operator, note)`, which is itself written to the audit
log. A reset restarts the drawdown peak from the last equity, so it is a risk
decision.

---

## 21. How do I run the Rust and Java benchmarks?

```bash
# Rust (from rust/)
cargo run --release --bin bench

# Java (from java/)
bash build.sh
bash bench.sh            # optional argument: data directory
```

Both time `RollingZ.update` and `KalmanStep.update` in batches of 64 updates
and report percentiles of the batch averages, then throughput of the batch
rolling z-score and the backtest on the bundled pair. The figures published in
the README are the median of three runs on one noisy Windows machine (ns per
update):

| | p50 | p99 | p99.9 | max |
|---|---:|---:|---:|---:|
| Rust `RollingZ` | 6.2 | 25.0 | 45.3 | 2684 |
| Rust `KalmanStep` | 17.2 | 29.7 | 232.8 | 2137 |
| Java `RollingZ` | 14.1 | 29.7 | 182.8 | 2477 |
| Java `KalmanStep` | 31.3 | 50.0 | 268.8 | 3155 |

Throughput there: Rust rolling z-score 21.9M bars/s, backtest 13.6M; Java 10.47M
and 7.37M observations/s. Your numbers will differ. Close other programs, run
more than once and compare p50; treat p99.9 and `max` as operating-system
jitter. These are not tick-to-trade figures and the Java harness is not JMH.
LEARN.md section 10c explains how to read them. These two commands were not
re-run for this cookbook; the table is from the release measurement.

---

## 22. How do I reproduce the two real-data studies?

Both scripts read daily Yahoo Finance data over the public chart endpoint (no
account), cache it in `research/real_data_2026_11/.cache/` (git-ignored) and
never commit raw prices. They need network access and numpy, and run from
the study folder:

```bash
cd research/real_data_2026_11
python study.py dev          # development period 2012-2019: pipeline bugs only
python study.py holdout      # 2020-2026: the one run

cd ../sector_residual_2026_11
python residual.py dev
python residual.py holdout   # reuses the pairs study's price cache
```

Compare against `results_dev.json` and `results_holdout.json` in each folder.
Yahoo adjusts history retroactively and some names change, so numbers can move
slightly. The committed results, for reference:

| Study | Holdout net Sharpe | Gate 1 (>= 1.0) | Gross return a year | Trades |
|---|---:|---|---:|---:|
| Pairs (`real_data_2026_11`) | -0.42 | FAIL | -0.10% | 46 |
| Sector-residual (`sector_residual_2026_11`) | -0.32 | FAIL | +0.31% | 3,800 |

Both are null. Do not rerun the holdout to "try again": the preregistration says
it runs once, and a variant is a new study (recipe 23). These two commands need
the network and were not re-run for this cookbook.

---

## 23. How do I start a new preregistered study from a template?

Create `research/<name>_<yyyy_mm>/PREREGISTRATION.md` from the template below,
fill every field with a number or a rule, commit it **before** downloading any
data or running anything, and add `study.py` only after that commit. Git
history is the timestamp. If an earlier study used the same years or names,
disclose it and count it as a trial (`deflated_sharpe` takes the number of
trials).

```markdown
# Preregistration: <short name>

Written <date>, before this strategy was run on any data. Git history is the timestamp.
Everything below is fixed. Any change after this commit is reported as a deviation.

## Disclosure
Earlier trials on these years or names: <none | list>. Trials counted for the deflated Sharpe: <N>.

## Question
<one yes/no question>

## Data and universe
Source, adjustment, names, groups. Known biases (survivors only? point-in-time?).

## Signal
<exact formula; every parameter a round number fixed now>

## Trading
Entry, exit, stop, maximum hold, sizing, execution lag (signal at close t, trade at close t+1).

## Accounting
Costs per side by instrument, borrow, how capital is defined, what is ignored.

## Periods
Development <from> to <to>: pipeline bugs only, parameters may not change.
Holdout <from> to <to>: run once. Gates use this only.

## Pass gates (holdout, all required)
1. Net annualised Sharpe >= <x>.
2. Probabilistic Sharpe vs 0 >= 0.95, and block-bootstrap 95% CI lower bound > 0 (block, draws, seed fixed).
3. Max drawdown <= <x>% of capital.
4. |beta to SPY| <= <x>.
5. Capacity at entry >= $<x>.
6. Deflated Sharpe >= 0.95 with <N> trials.

## Also reported, not gates
Gross return, development results, turnover, trades, win rate, net Sharpe at several costs, Sharpe by calendar year, worst day.

## Outcome rule
Fail any gate: report a null and do not tune. Any variant is a new study.
```

After the preregistration commit: write the script, run `dev` and fix bugs only,
run `holdout` once, and publish `RESULTS.md` with the gate table, the other
figures, what the result shows, its limits, and the decision. Report a miss as a
null. The two existing studies are worked examples
(`research/real_data_2026_11` and `research/sector_residual_2026_11`). See also
recipe 16 and LEARN.md section 9.9.

---

## Verifying the non-Python snippets

Every C++, Rust and Java block in recipes 1–12 is extracted in order, the
`#include` / `use` / `import` lines hoisted, the rest wrapped in a `main`,
and the result compiled against the built library and run from the
language's own directory (so `../data` resolves). This is how the recipes
are checked at release; the Python blocks (including 13–15 and 17–20) are run directly
from `python/` with `PYTHONPATH=src`. The commands are:

```bash
# C++ (from cpp/, after bash build.sh)
g++ -std=c++17 -Iinclude cookbook_main.cpp build/libstatarb.a -o cookbook && ./cookbook
# Rust (from rust/): put cookbook_main.rs under src/bin/ and
cargo run --release --bin cookbook_main
# Java (from java/, after bash build.sh)
javac -cp out/main -d /tmp/cb CookbookMain.java && java -cp out/main:/tmp/cb CookbookMain
```
