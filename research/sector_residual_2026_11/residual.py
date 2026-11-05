"""Sector-residual reversal study. Protocol: PREREGISTRATION.md.

Run: python residual.py dev | holdout.  Raw prices stay in the pairs study's git-ignored cache.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "real_data_2026_11"))
import study as base  # noqa: E402
from statarb.research import (  # noqa: E402
    block_bootstrap_sharpe_ci,
    deflated_sharpe,
    probabilistic_sharpe,
)

ETF = {"banks": "XLF", "energy": "XLE", "staples": "XLP", "health": "XLV", "tech": "XLK",
       "industrials": "XLI", "utilities": "XLU", "retail": "XLY"}
W, ENTRY, EXIT, HOLD, MAX_HL = 60, 1.25, 0.5, 20, 30.0
PAIRS_HOLDOUT_SHARPE = -0.420219401596257


def aligned(days, name):
    d = base.fetch(name)
    ix = {base.to_day(s): i for i, s in enumerate(d["t"])}
    out = np.full(len(days), np.nan)
    for j, day in enumerate(days):
        i = ix.get(day)
        if i is not None and d["adj"][i] is not None:
            out[j] = d["adj"][i]
    return out


def s_score(ri, re):
    """s, beta for one 60-day window; None when ineligible."""
    xc, yc = re - re.mean(), ri - ri.mean()
    den = xc @ xc
    if den <= 0.0:
        return None
    beta = (xc @ yc) / den
    X = np.cumsum(yc - beta * xc)  # centred residuals: the intercept is removed
    a0, a1 = X[:-1], X[1:]
    c0 = a0 - a0.mean()
    d0 = c0 @ c0
    if d0 <= 0.0:
        return None
    b = (c0 @ (a1 - a1.mean())) / d0
    if not 0.0 < b < 1.0 or -math.log(2.0) / math.log(b) > MAX_HL:
        return None
    a = a1.mean() - b * a0.mean()
    e = a1 - a - b * a0
    sigma = e.std(ddof=1) / math.sqrt(1.0 - b * b)
    if sigma <= 0.0:
        return None
    return (X[-1] - a / (1.0 - b)) / sigma, beta


def signals(ret, eret, lo, hi):
    n = ret.shape[1]
    S = np.full((hi, n), np.nan)
    B = np.full((hi, n), np.nan)
    for t in range(lo, hi):
        for i in range(n):
            w = ret[t - W + 1:t + 1, i]
            f = eret[t - W + 1:t + 1, i]
            if np.all(np.isfinite(w)) and np.all(np.isfinite(f)):
                out = s_score(w, f)
                if out is not None:
                    S[t, i], B[t, i] = out
    return S, B


def simulate(ret, eret, S, B, sect, lo, hi, cost_bps, dv):
    n = ret.shape[1]
    pos, beta, held = np.zeros(n), np.zeros(n), np.zeros(n, dtype=int)
    pend = {}
    net, gross, npos, trades, caps = [], [], [], [], []
    open_pnl = np.zeros(n)
    sectors = sorted(set(sect))
    def etf_net(p, b):
        return np.array([-(p * b)[[k for k in range(n) if sect[k] == s]].sum() for s in sectors])
    for t in range(lo, hi):
        live = pos != 0
        r_i = np.nan_to_num(ret[t])
        r_e = np.nan_to_num(eret[t])
        pnl_i = pos * (r_i - beta * r_e)
        open_pnl += pnl_i
        pnl = pnl_i.sum()
        borrow = 0.01 / 252 * (pos < 0).sum()
        before = etf_net(pos, beta)
        last = t == hi - 1
        if last:
            pend = {i: 0 for i in np.nonzero(pos)[0]}
        cost = 0.0
        for i, d in pend.items():
            cost += cost_bps / 1e4
            if d == 0:
                trades.append((int(held[i]), float(open_pnl[i] - 2 * cost_bps / 1e4)))
                pos[i] = beta[i] = 0.0
            else:
                pos[i], beta[i], held[i], open_pnl[i] = d, B[t - 1, i], 0, 0.0
                caps.append(0.01 * np.median(dv[t - 20:t, i]))
        pend = {}
        cost += 2.0 / 1e4 * np.abs(etf_net(pos, beta) - before).sum()
        held[pos != 0] += 1
        count = max(int(np.isfinite(ret[t]).sum()), 1)
        gross.append(pnl / (2 * count))
        net.append((pnl - cost - borrow) / (2 * count))
        npos.append(int(live.sum()))
        if last:
            break
        for i in range(n):
            s = S[t, i]
            if pos[i] == 0:
                if np.isfinite(s) and abs(s) >= ENTRY:
                    pend[i] = -1 if s > 0 else 1
            elif not np.isfinite(s) or abs(s) <= EXIT or held[i] >= HOLD:
                pend[i] = 0
    return np.array(net), np.array(gross), np.array(npos), trades, caps


def report(mode):
    days, adj, dv, _ = base.load()
    names, sect, etf = [], [], {}
    for g, tick in base.GROUPS.items():
        etf[g] = aligned(days, ETF[g])
        for n in tick.split():
            if n in adj:
                names.append(n)
                sect.append(g)
    P = np.column_stack([adj[n] for n in names])
    ret = np.full_like(P, np.nan)
    ret[1:] = P[1:] / P[:-1] - 1
    E = np.column_stack([etf[g] for g in sect])
    eret = np.full_like(E, np.nan)
    eret[1:] = E[1:] / E[:-1] - 1
    D = np.column_stack([dv[n] for n in names])
    a, b = ("2012-01-01", "2020-01-01") if mode == "dev" else ("2020-01-01", "2026-10-31")
    lo, hi = (int(np.searchsorted(days, np.datetime64(x))) for x in (a, b))
    S, B = signals(ret, eret, lo, hi)
    r, g, npos, tr, caps = simulate(ret, eret, S, B, sect, lo, hi, 5.0, D)
    sh = lambda v: float(v.mean() / v.std(ddof=1) * 252 ** 0.5)  # noqa: E731
    spy = np.diff(np.log(adj["SPY"]))[lo - 1:hi - 1]
    eq = 1 + np.cumsum(r)
    ci = block_bootstrap_sharpe_ci(r, 20, 5000, 0.95, 20261105)
    psr_var = float(np.var([PAIRS_HOLDOUT_SHARPE / 252 ** 0.5, sh(r) / 252 ** 0.5], ddof=1))
    yrs = days[lo:hi].astype("datetime64[Y]").astype(int) + 1970
    res = {"mode": mode, "days": len(r), "names": len(names), "gross_ann": float(g.mean() * 252),
           "net_ann": float(r.mean() * 252), "sharpe": sh(r), "gross_sharpe": sh(g), "psr": probabilistic_sharpe(r),
           "ci": repr(ci), "dsr_2_trials": deflated_sharpe(r, 2, psr_var),
           "max_dd": float(np.max(np.maximum.accumulate(eq) - eq)), "beta_spy": float(np.cov(r, spy)[0, 1] / spy.var(ddof=1)),
           "capacity_median": float(np.median(caps)), "trades": len(tr), "avg_positions": float(npos.mean()),
           "avg_hold": float(np.mean([t[0] for t in tr])), "win": float(np.mean([t[1] > 0 for t in tr])),
           "worst_day": float(r.min()),
           "by_year": {int(y): round(sh(r[yrs == y]), 2) for y in np.unique(yrs) if (yrs == y).sum() > 20},
           "cost_sharpe": {c: sh(simulate(ret, eret, S, B, sect, lo, hi, c, D)[0]) for c in (2.0, 5.0, 10.0, 20.0)}}
    (HERE / f"results_{mode}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


if __name__ == "__main__":
    report(sys.argv[1])
