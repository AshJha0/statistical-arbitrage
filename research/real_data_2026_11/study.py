"""Real-data pairs study. Protocol: PREREGISTRATION.md. Run: python study.py dev | holdout

Raw prices are cached in .cache/ (git-ignored) and never committed.
"""
from __future__ import annotations

import itertools
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "python" / "src"))
from statarb import BacktestConfig, backtest_pair, engle_granger_both, half_life, ols  # noqa: E402
from statarb.adf import select_lag  # noqa: E402
from statarb.research import (  # noqa: E402
    benjamini_hochberg,
    block_bootstrap_sharpe_ci,
    probabilistic_sharpe,
)

GROUPS = {
    "banks": "JPM BAC C WFC USB PNC GS MS", "energy": "XOM CVX COP EOG SLB OXY VLO MPC",
    "staples": "KO PEP PG CL KMB GIS K HSY", "health": "JNJ PFE MRK ABT LLY BMY AMGN GILD",
    "tech": "MSFT AAPL ORCL IBM INTC CSCO TXN QCOM", "industrials": "CAT DE HON MMM GE UNP CSX NSC",
    "utilities": "DUK SO AEP D EXC XEL ED WEC", "retail": "WMT TGT HD LOW COST KR DG DLTR",
}
LINKED = ["GOOGL/GOOG", "FOXA/FOX", "NWSA/NWS", "LEN/PHM", "V/MA", "GLD/IAU", "SPY/IVV",
          "EWA/EWC", "XLE/VDE", "KO/PEP"]
CACHE = HERE / ".cache"
FORM, MAXDAYS, SLOTS, CAP, WARM = 504, 126, 20, 1_000_000.0, 59
END = "2026-10-30"


def fetch(t: str) -> dict:
    path = CACHE / f"{t}.json"
    if not path.exists():
        url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{t}?period1=1230768000"
               "&period2=1793404800&interval=1d&events=div,split")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        r = json.load(urllib.request.urlopen(req, timeout=30))["chart"]["result"][0]
        q = r["indicators"]["quote"][0]
        path.write_text(json.dumps({"t": r["timestamp"], "adj": r["indicators"]["adjclose"][0]["adjclose"],
                                    "close": q["close"], "vol": q["volume"]}))
        time.sleep(0.5)
    return json.loads(path.read_text())


def to_day(s: int) -> np.datetime64:
    return np.datetime64(int(s), "s").astype("datetime64[D]")


def load():
    CACHE.mkdir(exist_ok=True)
    names = sorted({n for g in GROUPS.values() for n in g.split()} | {n for p in LINKED for n in p.split("/")})
    raw, missing = {}, []
    for n in names:
        try:
            raw[n] = fetch(n)
        except Exception as e:  # recorded in the results, not hidden
            missing.append(f"{n}: {e}")
    days = np.array([to_day(s) for s in raw["SPY"]["t"]])
    adj, dv = {}, {}
    for n, d in raw.items():
        ix = {to_day(s): i for i, s in enumerate(d["t"])}
        a = np.full(len(days), np.nan)
        v = np.full(len(days), np.nan)
        for j, day in enumerate(days):
            i = ix.get(day)
            if i is not None and None not in (d["adj"][i], d["close"][i], d["vol"][i]):
                a[j], v[j] = d["adj"][i], d["close"][i] * d["vol"][i]
        adj[n], dv[n] = a, v
    keep = days <= np.datetime64(END)
    return days[keep], {k: v[keep] for k, v in adj.items()}, {k: v[keep] for k, v in dv.items()}, missing


def candidates():
    for g in GROUPS.values():
        yield from itertools.combinations(sorted(g.split()), 2)
    for p in LINKED:
        yield tuple(sorted(p.split("/")))


def period_starts(days, first, last):
    out = []
    for y in range(first[0], last[0] + 1):
        for m in (1, 7):
            if first <= (y, m) <= last:
                out.append(int(np.searchsorted(days, np.datetime64(f"{y}-{m:02d}-01"))))
    return out


def select(adj, s, e):
    rows, f0 = [], s - FORM
    for a, b in dict.fromkeys(candidates()):
        if a not in adj or b not in adj:
            continue
        if not (np.all(np.isfinite(adj[a][f0:e])) and np.all(np.isfinite(adj[b][f0:e]))):
            continue
        y, x = 100 * adj[a][f0:s] / adj[a][f0], 100 * adj[b][f0:s] / adj[b][f0]
        try:
            lag = select_lag(ols(y, x).resid, 5, 2)
            f = engle_granger_both(y, x, lag)
        except ValueError:
            continue
        p = max(f.y_on_x.adf.pvalue, f.x_on_y.adf.pvalue)
        rows.append((p, a, b, f.y_on_x.alpha, f.y_on_x.beta, half_life(f.y_on_x.resid)))
    if not rows:
        return []
    rej, _ = benjamini_hochberg(np.array([r[0] for r in rows]), 0.05)
    good = [r for r, k in zip(rows, rej, strict=True) if k and r[4] > 0 and 5 <= r[5] <= 60]
    return sorted(good, key=lambda r: (r[0], r[1], r[2]))[:SLOTS]


def period_end(k, starts, last):
    return min(starts[k + 1] if k + 1 < len(starts) else last, starts[k] + MAXDAYS)


def run(days, adj, dv, starts, cost, sizing, last, picks):
    out, trades, caps = [], [], []
    cfg = BacktestConfig(window=60, entry_z=2.0, exit_z=0.5, stop_z=4.0, cost_bps=cost, slippage_bps=0.0,
                         capital=CAP, sizing=sizing, pnl_stop_frac=0.02, max_hold_days=60, cooldown_days=20)
    for k, s in enumerate(starts):
        e = period_end(k, starts, last)
        port = np.zeros(e - s)
        for _, a, b, al, be, _hl in picks[k]:
            f0, w0 = s - FORM, s - WARM
            y, x = 100 * adj[a][w0:e] / adj[a][f0], 100 * adj[b][w0:e] / adj[b][f0]
            r = backtest_pair(y, x, al, be, cfg)
            d = np.diff(np.r_[CAP, r.equity])
            for t in r.trades:
                u = CAP / (y[t.entry_day] + be * x[t.entry_day])
                span = slice(t.entry_day, t.exit_day)
                short = be * u * x[span] if t.direction > 0 else u * y[span]
                d[span] -= 0.01 / 252 * short
                i = w0 + t.entry_day
                caps.append(0.01 * min(np.median(dv[a][i - 20:i]), np.median(dv[b][i - 20:i])))
                trades.append((str(days[i]), a, b, t.holding_days, t.pnl / CAP, t.stopped))
            port += d[WARM:]
        out.append(port / (SLOTS * CAP))
    return np.concatenate(out), trades, caps


def sharpe(r):
    return float(r.mean() / r.std(ddof=1) * 252 ** 0.5) if r.std() > 0 else 0.0


def report(mode):
    days, adj, dv, missing = load()
    span = ((2012, 1), (2019, 7)) if mode == "dev" else ((2020, 1), (2026, 7))
    starts = period_starts(days, *span)
    last = int(np.searchsorted(days, np.datetime64("2020-01-01"))) if mode == "dev" else len(days)
    picks = [select(adj, s, period_end(k, starts, last)) for k, s in enumerate(starts)]
    r, tr, caps = run(days, adj, dv, starts, 5.0, "beta_hedged", last, picks)
    idx = np.concatenate([np.arange(s, period_end(k, starts, last)) for k, s in enumerate(starts)])
    spy = np.diff(np.log(adj["SPY"]))[idx - 1]
    eq = 1 + np.cumsum(r)
    npairs = [len(p) for p in picks]
    res = {"mode": mode, "missing": missing, "days": len(r), "first": str(days[idx[0]]), "last": str(days[idx[-1]]),
           "ann_net": float(r.mean() * 252), "sharpe": sharpe(r), "max_dd": float(np.max(np.maximum.accumulate(eq) - eq)),
           "beta_spy": float(np.cov(r, spy)[0, 1] / spy.var(ddof=1)),
           "capacity_median": float(np.median(caps)) if caps else 0.0, "trades": len(tr),
           "win": float(np.mean([t[4] > 0 for t in tr])) if tr else 0.0, "stops": int(sum(t[5] for t in tr)),
           "pairs_per_period": npairs, "empty_periods": float(np.mean(np.array(npairs) == 0)),
           "worst": sorted(tr, key=lambda t: t[4])[:5]}
    if r.std() > 0:
        ci = block_bootstrap_sharpe_ci(r, 20, 5000, 0.95, 20261105)
        res["psr"] = probabilistic_sharpe(r)
        res["ci"] = repr(ci)
    res["cost_sharpe"] = {str(c): sharpe(run(days, adj, dv, starts, c, "beta_hedged", last, picks)[0])
                          for c in (2.0, 5.0, 10.0, 20.0)}
    res["equal_dollar_sharpe"] = sharpe(run(days, adj, dv, starts, 5.0, "equal_dollar", last, picks)[0])
    res["gross_ann"] = float(run(days, adj, dv, starts, 0.0, "beta_hedged", last, picks)[0].mean() * 252)
    (HERE / f"results_{mode}.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, default=str))


if __name__ == "__main__":
    report(sys.argv[1])
