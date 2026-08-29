# COOKBOOK — statarb recipes

Task-oriented recipes for the `statarb` toolkit, each in all four languages.
Python snippets run as-is from the `python/` directory with
`PYTHONPATH=src python3 snippet.py` (they locate the bundled data relative to
the project root). C++/Rust/Java snippets follow the pinned API contract in
[`API_SPEC.md`](API_SPEC.md): C++ namespace `statarb` (headers
`<statarb/*.hpp>`, series as `std::vector<double>`), Rust crate `statarb`
(fallible functions return `Result<_, StatarbError>`), Java package
`com.quant.statarb` (series as `double[]`, errors as
`IllegalArgumentException`).

All numbers quoted in comments are the golden values for the bundled data.

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
#include <statarb/adf.hpp>
#include <statarb/data_io.hpp>

auto walks = statarb::load_csv("../data/random_walks.csv");
statarb::ADFResult res = statarb::adf_test(walks["walk_a"], /*maxlag=*/1, /*nvars=*/1);
std::cout << "t = " << res.tstat << ", reject@5% = " << res.reject(0.05) << "\n";
```

**Rust**

```rust
use statarb::{adf_test, load_csv};

let walks = load_csv("../data/random_walks.csv")?;
let res = adf_test(&walks["walk_a"], 1, 1)?;
println!("t = {:.4}, reject@5% = {}", res.tstat, res.reject(0.05)?);
```

**Java**

```java
import com.quant.statarb.Adf;
import com.quant.statarb.AdfResult;
import com.quant.statarb.DataIo;

var walks = DataIo.loadCsv("../data/random_walks.csv");
AdfResult res = Adf.test(walks.get("walk_a"), 1, 1);
System.out.printf("t = %.4f, reject@5%% = %b%n", res.tstat(), res.reject(0.05));
```

Note: `reject` compares the t-stat to the embedded critical-value table
directly; the p-value is a documented piecewise-linear approximation and is
never used for the verdict.

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
#include <statarb/engle_granger.hpp>

statarb::EGResult eg = statarb::engle_granger(y, x, /*maxlag=*/1);
std::cout << "beta = " << eg.beta
          << ", coint@5% = " << eg.cointegrated(0.05) << "\n";
```

**Rust**

```rust
use statarb::engle_granger;

let eg = engle_granger(&y, &x, 1)?;
println!("beta = {:.4}, coint@5% = {}", eg.beta, eg.cointegrated(0.05)?);
```

**Java**

```java
import com.quant.statarb.EngleGranger;
import com.quant.statarb.EgResult;

EgResult eg = EngleGranger.test(y, x, 1);
System.out.printf("beta = %.4f, coint@5%% = %b%n", eg.beta(), eg.cointegrated(0.05));
```

The residual ADF is automatically judged against the 2-variable
Engle–Granger critical values (−3.34 at 5%), not the plain ADF row — using
the 1-variable row here would over-detect cointegration.

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

std::vector<double> s = statarb::spread(y, x, eg.alpha, eg.beta);
std::vector<double> z = statarb::rolling_zscore(s, /*window=*/60);
// z[t] is NaN for t < 59 and for zero-variance windows
```

**Rust**

```rust
use statarb::{rolling_zscore, spread};

let s = spread(&y, &x, eg.alpha, eg.beta)?;
let z = rolling_zscore(&s, 60)?;   // z[t] = f64::NAN for t < 59
```

**Java**

```java
import com.quant.statarb.Signals;

double[] s = Signals.spread(y, x, eg.alpha(), eg.beta());
double[] z = Signals.rollingZscore(s, 60);   // Double.NaN for t < 59
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
#include <statarb/signals.hpp>

double hl = statarb::half_life(eg.resid);   // +inf if rho >= 0 (no reversion)
```

**Rust**

```rust
let hl = statarb::half_life(&eg.resid)?;    // f64::INFINITY if rho >= 0
```

**Java**

```java
double hl = Signals.halfLife(eg.resid());   // POSITIVE_INFINITY if rho >= 0
```

Interpretation: choose the z-score window as several half-lives (60 ≈ 7×8.39
here) and expect holding periods of one to two half-lives. `+inf` means the
AR(1) fit found no reversion — do not trade that spread.

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

// levels: column-major, one std::vector<double> per asset
statarb::JohansenResult jo =
    statarb::johansen_trace({col_y, col_x}, /*maxlag_diff=*/1);
std::cout << "rank = " << jo.rank << "\n";
```

**Rust**

```rust
use statarb::johansen_trace;

let jo = johansen_trace(&[col_y, col_x], 1)?;   // one Vec<f64> per asset
println!("rank = {}, trace(0) = {:.3}", jo.rank, jo.trace_stats[0]);
```

**Java**

```java
import com.quant.statarb.Johansen;
import com.quant.statarb.JohansenResult;

double[][] levels = {colY, colX};               // one column per asset
JohansenResult jo = Johansen.traceTest(levels, 1);
System.out.println("rank = " + jo.rank());
```

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

statarb::KalmanResult km = statarb::kalman_hedge(y, x, /*delta=*/1e-4, /*r=*/1e-3);
double beta_today = km.beta.back();
```

**Rust**

```rust
let km = statarb::kalman_hedge(&y, &x, 1e-4, 1e-3)?;
let beta_today = *km.beta.last().unwrap();
```

**Java**

```java
import com.quant.statarb.Kalman;
import com.quant.statarb.KalmanResult;

KalmanResult km = Kalman.hedge(y, x, 1e-4, 1e-3);
double betaToday = km.beta()[km.beta().length - 1];
```

Tuning: `delta` is the forgetting dial (`Q = delta/(1-delta) * I`). Smaller
delta → smoother beta, slower adaptation; `delta -> 0` converges to static
OLS. `r` is the observation noise; both must satisfy 0 < delta < 1
(strictly) and r > 0 or the call errors. `km.z` are innovation z-scores —
a window-free alternative trading signal.

---

## 7. How do I run a full pairs backtest with costs?

**Python**

```python
from statarb import BacktestConfig, backtest_pair, engle_granger, load_csv

pair = load_csv("../data/pair_equity.csv")
y, x = pair["stock_y"], pair["stock_x"]
eg = engle_granger(y[:900], x[:900])                 # in-sample hedge
cfg = BacktestConfig(window=60, entry_z=2.0, exit_z=0.5, stop_z=4.0,
                     cost_bps=1.0, slippage_bps=0.5, capital=1e6)
res = backtest_pair(y, x, eg.alpha, eg.beta, cfg)
m = res.metrics
print(f"total {m['total_return']:.4%}  sharpe {m['sharpe']:.3f}  "
      f"maxDD {m['max_drawdown']:.2%}  trades {int(m['n_trades'])}")
# golden: -1.9113%, -0.137, 14.26%, 31 trades; stops fired in the break regime
print("stopped trades:", res.n_stops)
```

**C++**

```cpp
#include <statarb/backtest.hpp>

statarb::BacktestConfig cfg;      // defaults match config.json
cfg.window = 60; cfg.entry_z = 2.0; cfg.exit_z = 0.5; cfg.stop_z = 4.0;
cfg.cost_bps = 1.0; cfg.slippage_bps = 0.5; cfg.capital = 1e6;
statarb::BacktestResult res = statarb::backtest_pair(y, x, eg.alpha, eg.beta, cfg);
std::cout << "sharpe = " << res.metrics.at("sharpe") << "\n";
```

**Rust**

```rust
use statarb::{backtest_pair, BacktestConfig};

let cfg = BacktestConfig { window: 60, entry_z: 2.0, exit_z: 0.5, stop_z: 4.0,
                           cost_bps: 1.0, slippage_bps: 0.5, capital: 1e6 };
let res = backtest_pair(&y, &x, eg.alpha, eg.beta, &cfg)?;
println!("sharpe = {:.3}, trades = {}", res.metrics.sharpe, res.trades.len());
```

**Java**

```java
import com.quant.statarb.Backtest;
import com.quant.statarb.BacktestConfig;
import com.quant.statarb.BacktestResult;

BacktestConfig cfg = new BacktestConfig(60, 2.0, 0.5, 4.0, 1.0, 0.5, 1e6);
BacktestResult res = Backtest.run(y, x, eg.alpha(), eg.beta(), cfg);
System.out.printf("sharpe = %.3f%n", res.metrics().get("sharpe"));
```

Pinned mechanics worth knowing: signals computed at the day-t close execute
at the day t+1 close (no look-ahead); legs are equal-dollar (capital/2 each)
so entries are exactly dollar-neutral; |z| ≥ stop_z while flat means *no
entry*; a position open on the last day is force-closed as a normal trade.

---

## 8. How do I run an honest walk-forward evaluation?

**Python**

```python
from statarb import BacktestConfig, load_csv, walk_forward

pair = load_csv("../data/pair_equity.csv")
cfg = BacktestConfig()          # defaults = config.json values
wf = walk_forward(pair["stock_y"], pair["stock_x"], cfg, insample_frac=0.6)
print(f"hedge from days 0..{wf.split-1}: alpha {wf.alpha:.4f} beta {wf.beta:.4f}")
print(f"IS  sharpe {wf.insample.metrics['sharpe']:.2f}")   # +1.49
print(f"OOS sharpe {wf.oos.metrics['sharpe']:.2f}")        # -1.37 (break!)
```

**C++**

```cpp
statarb::WalkForwardResult wf = statarb::walk_forward(y, x, cfg, 0.6);
std::cout << "IS "  << wf.insample.metrics.at("sharpe")
          << " OOS " << wf.oos.metrics.at("sharpe") << "\n";
```

**Rust**

```rust
let wf = statarb::walk_forward(&y, &x, &cfg, 0.6)?;
println!("IS {:.2} OOS {:.2}", wf.insample.metrics.sharpe, wf.oos.metrics.sharpe);
```

**Java**

```java
var wf = Backtest.walkForward(y, x, cfg, 0.6);
System.out.printf("IS %.2f OOS %.2f%n",
    wf.insample().metrics().get("sharpe"), wf.oos().metrics().get("sharpe"));
```

The hedge is estimated by OLS on the first 60% only, and the out-of-sample
z-score warms up inside the OOS segment — nothing crosses the split. The
bundled pair degrades hard out-of-sample because the OOS window contains the
day-1050 regime break; that degradation is the lesson, not a bug.

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
statarb::EGResult eg = statarb::engle_granger(walk_a, walk_b, 1);
if (!eg.cointegrated(0.05)) {
    std::cout << "spurious — no backtest\n";       // this branch runs
}
```

**Rust**

```rust
let eg = statarb::engle_granger(&walk_a, &walk_b, 1)?;
if !eg.cointegrated(0.05)? {
    println!("spurious — no backtest");            // this branch runs
}
```

**Java**

```java
EgResult eg = EngleGranger.test(walkA, walkB, 1);
if (!eg.cointegrated(0.05)) {
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

statarb::OLSResult fit = statarb::ols(y, x, /*intercept=*/true);
// fit.params[0] = intercept, fit.params[1] = slope; fit.resid, fit.r2, ...
```

**Rust**

```rust
let fit = statarb::ols(&y, &x, true)?;
println!("beta = {:.6}, r2 = {:.4}", fit.params[1], fit.r2);
```

**Java**

```java
import com.quant.statarb.Ols;
import com.quant.statarb.OlsResult;

OlsResult fit = Ols.fit(y, x, true);
System.out.printf("beta = %.6f, r2 = %.4f%n", fit.params()[1], fit.r2());
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
#include <statarb/data_io.hpp>

auto cfg  = statarb::load_config("../data/config.json");
auto pair = statarb::load_csv("../data/pair_equity.csv");
// pair["stock_y"], pair["stock_x"] are std::vector<double>
```

**Rust**

```rust
let cfg  = statarb::load_config("../data/config.json")?;
let pair = statarb::load_csv("../data/pair_equity.csv")?;
```

**Java**

```java
var cfg  = DataIo.loadConfig("../data/config.json");
var pair = DataIo.loadCsv("../data/pair_equity.csv");
```

The loaders validate: missing file, empty file, ragged rows, and non-numeric
cells all raise descriptive errors. Every language ships its own tiny
CSV/JSON reader so the libraries stay dependency-light.

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
try {
    statarb::kalman_hedge(y, x, /*delta=*/1.5, /*r=*/1e-3);   // delta not in (0,1)
} catch (const std::invalid_argument& e) {
    std::cerr << e.what() << "\n";
}
```

**Rust**

```rust
match statarb::kalman_hedge(&y, &x, 1.5, 1e-3) {
    Err(e) => eprintln!("{e}"),          // delta must lie strictly inside (0,1)
    Ok(_) => unreachable!(),
}
```

**Java**

```java
try {
    Kalman.hedge(y, x, 1.5, 1e-3);       // delta not in (0,1)
} catch (IllegalArgumentException e) {
    System.err.println(e.getMessage());
}
```

The full list of guarded conditions (length mismatch, non-finite input,
constant series, collinearity, short series, window bounds, non-positive
prices, delta/r/maxlag/nvars/alpha domains, backtest threshold ordering) is
in API_SPEC.md §9.
