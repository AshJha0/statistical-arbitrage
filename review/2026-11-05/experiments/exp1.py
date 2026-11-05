import sys,json,math,warnings
warnings.filterwarnings('ignore')
R='C:/Work/Claude/statistical-arbitrage/statistical-arbitrage'
sys.dont_write_bytecode=True
sys.path.insert(0,R+'/python/src')
import numpy as np, csv
from statarb import *
from statarb.adf import RESPONSE_SURFACE
from statarb.backtest import walk_forward
d=np.genfromtxt(R+'/data/pair_equity.csv',delimiter=',',names=True)
y,x=d['stock_y'],d['stock_x']
cfg=BacktestConfig()
wf=walk_forward(y,x,cfg,0.6)
print('A base',{k:round(v,4) for k,v in wf.oos.metrics.items()}, 'stops',wf.oos.n_stops)
print(' OOS trades',[(t.entry_day+900,t.exit_day+900,t.direction,round(t.pnl),t.stopped) for t in wf.oos.trades])
# B MacKinnon vs statsmodels
from statsmodels.tsa.adfvalues import mackinnoncrit
from statsmodels.tsa import adfvalues as av
for N in (1,2,3):
  for n in (30,100,900):
    print('B',N,n,np.round(critical_values(N,n),4),np.round(mackinnoncrit(N,'c',n),4))
print('B sm tau_2010 c',[np.round(av.tau_2010s['c'][i],4).tolist() if hasattr(av,'tau_2010s') else None for i in range(3)] if hasattr(av,'tau_2010s') else [k for k in dir(av) if 'tau' in k])
# C hedge vs sizing: sized ratio vs beta
print('C y/x ratio pre-break',(y[:900]/x[:900]).mean(),'post',(y[1100:]/x[1100:]).mean(),'beta',wf.beta)
# D PnL with beta-hedged quantities and Kalman
def bt(y,x,z,beta_t,cfg,mode):
    T=len(y);bps=(cfg.cost_bps+cfg.slippage_bps)/1e4;eq=[];E=cfg.capital;st=0;qy=qx=0;pend=None;n=0
    for t in range(T):
        pnl=qy*(y[t]-y[t-1])+qx*(x[t]-x[t-1]) if t else 0;c=0
        if pend is not None:
            if pend!=0 and st==0 and t<T-1:
                b=beta_t[t-1]
                if mode=='dollar': qy=pend*cfg.capital/2/y[t];qx=-pend*cfg.capital/2/x[t]
                else:
                    u=cfg.capital/(y[t]+abs(b)*x[t]); qy=pend*u;qx=-pend*u*b
                c+=(abs(qy)*y[t]+abs(qx)*x[t])*bps;st=pend;n+=1
            elif pend==0 and st!=0:
                c+=(abs(qy)*y[t]+abs(qx)*x[t])*bps;qy=qx=0;st=0
            pend=None
        if t==T-1 and st: c+=(abs(qy)*y[t]+abs(qx)*x[t])*bps;qy=qx=0;st=0
        E+=pnl-c;eq.append(E)
        if t<T-1 and not math.isnan(z[t]):
            if st==0:
                if cfg.entry_z<=z[t]<cfg.stop_z:pend=-1
                elif -cfg.stop_z<z[t]<=-cfg.entry_z:pend=1
            elif abs(z[t])>=cfg.stop_z or abs(z[t])<=cfg.exit_z:pend=0
    e=np.array(eq);r=np.diff(np.r_[cfg.capital,e])/cfg.capital
    return dict(ret=round(e[-1]/cfg.capital-1,4),sh=round(r.mean()/r.std(ddof=1)*252**.5,2),dd=round(((np.maximum.accumulate(e)-e)/np.maximum.accumulate(e)).max(),4),n=n)
ys,xs=y[900:],x[900:]
s=ys-(wf.alpha+wf.beta*xs);z=rolling_zscore(s,60)
print('D static dollar',bt(ys,xs,z,np.full(600,wf.beta),cfg,'dollar'))
print('D static beta  ',bt(ys,xs,z,np.full(600,wf.beta),cfg,'beta'))
# Kalman diagnostics
k=kalman_hedge(y,x,1e-4,1e-3)
zz=k.z[100:]
ac=lambda v,l=1:np.corrcoef(v[:-l],v[l:])[0,1]
print('E kalman innov z std',zz.std(),'ac1',ac(zz),'|z|>2 frac',(abs(zz)>2).mean(),'median S',np.median(k.s_var),'beta daily chg sd',np.diff(k.beta[300:]).std())
# true residual var
res=(y-(12+1.7*x))[:1050];print('E true spread var',res.var(),'vs r=1e-3 ratio',res.var()/1e-3)
tb=np.where(np.arange(1500)<1050,1.7,1.1);tb[1050:1081]=np.linspace(1.7,1.1,31)
rm=lambda b:np.sqrt(np.mean((b[300:]-tb[300:])**2))
roll=np.full(1500,np.nan)
for W in (60,120,250):
    rb=np.array([np.nan]*W+[np.polyfit(x[t-W:t],y[t-W:t],1)[0] for t in range(W,1500)])
    print('E rolling OLS',W,'RMSE',rm(np.nan_to_num(rb,nan=1.7)))
print('E kalman RMSE',rm(k.beta),'static',rm(np.full(1500,wf.beta)))
# Kalman traded, causal: spread e_t=y-a_{t-1}-b_{t-1}x
ap=np.r_[k.alpha[0],k.alpha[:-1]];bp=np.r_[k.beta[0],k.beta[:-1]]
e=y-ap-bp*x
for nm,zs in (('innov z',k.z),('roll z of innov',rolling_zscore(e,60))):
    print('F kalman OOS',nm,bt(ys,xs,zs[900:],k.beta[900:],cfg,'beta'),' full',bt(y[100:],x[100:],zs[100:],k.beta[100:],cfg,'beta'))
# G costs sensitivity + param grid
for c in (0.5,1.5,5,10,20):
    c2=BacktestConfig(cost_bps=c,slippage_bps=0);w=walk_forward(y,x,c2,.6)
    print('G cost',c,'IS',round(w.insample.metrics['sharpe'],2),'OOS',round(w.oos.metrics['sharpe'],2))
# H sign bug: negative beta
rng=np.random.default_rng(1);xx=100*np.exp(np.cumsum(rng.normal(0,.01,1000)));s0=np.zeros(1000)
for t in range(1,1000):s0[t]=.9*s0[t-1]+rng.normal(0,.6)
yy=300-1.0*xx+s0
f=ols(yy,xx);print('H beta',f.params,backtest_pair(yy,xx,f.params[0],f.params[1],cfg).metrics['sharpe'])
# I stop-reachability
for W in (5,10,17,18,20):print('I W',W,'max|z|',(W-1)/W**.5)
# J alpha irrelevance
print('J alpha irrelevant',backtest_pair(y,x,0,1.6,cfg).metrics['total_return']==backtest_pair(y,x,999,1.6,cfg).metrics['total_return'])
# K window sharpe CI
r=np.diff(np.r_[1e6,wf.insample.equity])/1e6;sr=wf.insample.metrics['sharpe'];Y=len(r)/252
print('K IS SR se',((1+.5*sr**2)/Y)**.5,'frac days flat',(r==0).mean())
