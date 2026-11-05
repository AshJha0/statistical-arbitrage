import sys,warnings
warnings.filterwarnings('ignore')
sys.dont_write_bytecode=True
R='C:/Work/Claude/statistical-arbitrage/statistical-arbitrage'
sys.path.insert(0,R+'/python/src')
import numpy as np
from statarb import *
from statarb.backtest import walk_forward
def dgp(seed,brk,T=1500):
    r=np.random.default_rng(seed);x=100*np.exp(np.cumsum(r.normal(2e-4,.01,T)));e=r.normal(0,.6,T);s=np.zeros(T)
    for t in range(1,T): s[t]=(.92 if (t<1050 or not brk) else .995)*s[t-1]+e[t]
    b=np.full(T,1.7)
    if brk: b[1050:1081]=np.linspace(1.7,1.1,31);b[1081:]=1.1
    return 12+b*x+s,x
free=BacktestConfig(cost_bps=0,slippage_bps=0);base=BacktestConfig()
G=[];
for sd in range(300):
    y,x=dgp(sd,False);w=walk_forward(y,x,free,.6).oos;m=w.metrics;N=len(w.equity);yrs=N/252
    inmkt=sum(t.holding_days for t in w.trades)/N
    G.append((m['total_return']/yrs,m['n_trades']/yrs,m['turnover']/yrs,inmkt,m['total_return']/max(m['n_trades'],1)*1e4,m['max_drawdown']))
G=np.array(G);g,tr,to,im,pt,dd=G.mean(0)
print(f'no-break OOS, zero cost: gross {g:.2%}/yr, trades/yr {tr:.1f}, turnover {to:.1f}x/yr, in market {im:.0%}, gross per round trip {pt:.1f}bps of capital, maxDD {dd:.2%}')
print(f'break-even all-in cost per side: {g/to*1e4:.1f}bps')
for c in (1.5,5,10):
    print(f' net at {c}bps/side: {g-to*c/1e4:.2%}/yr')
for b in (.0025,.01,.03):
    print(f' borrow {b:.2%}/yr on short leg (half gross, in market): drag {b*.5*im:.3%}/yr')
L=[]
for sd in range(300):
    y,x=dgp(sd,True);L.append(walk_forward(y,x,base,.6).oos.metrics['total_return'])
Y=[]
for sd in range(300):
    y,x=dgp(sd,False);Y.append(walk_forward(y,x,base,.6).oos.metrics['total_return'])
L=np.array(L);Y=np.array(Y);loss=Y.mean()-L.mean();net=Y.mean()/(600/252)
print(f'net normal {net:.2%}/yr; break-window return mean {L.mean():.2%} p5 {np.percentile(L,5):.2%}; incremental loss per break {loss:.2%}; breakeven break rate {net/loss:.3f}/yr = 1 per {loss/net:.1f} yrs')
print(f'$ on $10m gross/pair at net: {net*1e7:,.0f}/yr')
