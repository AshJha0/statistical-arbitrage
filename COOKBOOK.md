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

// Construct through the validating constructor (window, entry_z, exit_z,
// stop_z, cost_bps, slippage_bps, capital); the defaults equal config.json.
// The members are public, but backtest_pair re-validates the struct, so a
// later `cfg.exit_z = 5.0;` is caught with std::invalid_argument at run time.
const statarb::BacktestConfig cfg(60, 2.0, 0.5, 4.0, 1.0, 0.5, 1e6);
statarb::BacktestResult res = statarb::backtest_pair(y_all, x_all, eg.alpha, eg.beta, cfg);
std::cout << "total " << res.metrics.total_return << "  sharpe " << res.metrics.sharpe
          << "  maxDD " << res.metrics.max_drawdown << "  trades " << res.trades.size()
          << "  stops " << res.n_stops() << "\n";               // -0.0191, -0.137, 0.1426, 31
```

**Rust**

```rust
use statarb::{backtest_pair, BacktestConfig};

// BacktestConfig::new validates; a struct literal that bypasses it is
// re-validated inside backtest_pair / walk_forward (Err(InvalidParam)).
let cfg = BacktestConfig::new(60, 2.0, 0.5, 4.0, 1.0, 0.5, 1e6)?;
let res = backtest_pair(y_all, x_all, eg.alpha, eg.beta, &cfg)?;
println!("total {:.4}  sharpe {:.3}  maxDD {:.4}  trades {}  stops {}",
    res.metrics.total_return, res.metrics.sharpe, res.metrics.max_drawdown,
    res.trades.len(), res.n_stops());                             // -0.0191, -0.137, 0.1426, 31
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
at the day t+1 close (no look-ahead); legs are equal-dollar (capital/2 each)
so entries are exactly dollar-neutral; |z| ≥ stop_z while flat means *no
entry*; a position open on the last day is force-closed as a normal trade,
while a pending *entry* on the last day is discarded (no zero-day round
trip). Thresholds and costs must be finite — a config loaded from JSON with
`"cost_bps": NaN` is refused rather than producing a NaN equity curve and a
Sharpe of 0.00.

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
statarb::WalkForwardResult wf = statarb::walk_forward(y_all, x_all, cfg, 0.6);
std::cout << "split " << wf.split << "  IS " << wf.insample.metrics.sharpe
          << "  OOS " << wf.oos.metrics.sharpe << "\n";           // 900, +1.49, -1.37
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

## Verifying the non-Python snippets

Every C++, Rust and Java block above is extracted in order, the
`#include` / `use` / `import` lines hoisted, the rest wrapped in a `main`,
and the result compiled against the built library and run from the
language's own directory (so `../data` resolves). This is how the recipes
were checked for this release; the commands are:

```bash
# C++ (from cpp/, after bash build.sh)
g++ -std=c++17 -Iinclude cookbook_main.cpp build/libstatarb.a -o cookbook && ./cookbook
# Rust (from rust/): put cookbook_main.rs under src/bin/ and
cargo run --release --bin cookbook_main
# Java (from java/, after bash build.sh)
javac -cp out/main -d /tmp/cb CookbookMain.java && java -cp out/main:/tmp/cb CookbookMain
```
