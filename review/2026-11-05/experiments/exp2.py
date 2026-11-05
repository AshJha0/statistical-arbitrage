import sys,math,warnings
warnings.filterwarnings('ignore')
R='C:/Work/Claude/statistical-arbitrage/statistical-arbitrage'
sys.path.insert(0,R+'/python/src')
import numpy as np
from statarb import *
from statarb.backtest import walk_forward
exec(open('exp1.py').read().split('# D PnL')[1].split('ys,xs=')[0].split('\n',1)[1])  # bt()
cfg=BacktestConfig()
d=np.genfromtxt(R+'/data/pair_equity.csv',delimiter=',',names=True);y,x=d['stock_y'],d['stock_x']
def dgp(seed,brk=True,T=1500):
    r=np.random.default_rng(seed);x=100*np.exp(np.cumsum(r.normal(2e-4,.01,T)));e=r.normal(0,.6,T);s=np.zeros(T)
    for t in range(1,T):
        s[t]=(.92 if (t<1050 or not brk) else .995)*s[t-1]+e[t]
    b=np.full(T,1.7)
    if brk: b[1050:1081]=np.linspace(1.7,1.1,31);b[1081:]=1.1
    return 12+b*x+s,x
# 1 multi-seed
for brk in (False,True):
    I=[];O=[];St=[];
    for sd in range(300):
        yy,xx=dgp(sd,brk);w=walk_forward(yy,xx,cfg,.6);I.append(w.insample.metrics['sharpe']);O.append(w.oos.metrics['sharpe'])
    I,O=np.array(I),np.array(O)
    print('1 break',brk,'IS SR mean/p5/p95',I.mean().round(2),np.percentile(I,[5,95]).round(2),'OOS',O.mean().round(2),np.percentile(O,[5,95]).round(2),'P(OOS<0)',(O<0).mean())
# 2 selection on pure random walks
r=np.random.default_rng(5);N=40;T=1500
P=100*np.exp(np.cumsum(r.normal(0,.015,(T,N)),axis=0))
sel=[];tot=0
for i in range(N):
    for j in range(i+1,N):
        tot+=1
        g=engle_granger(P[:900,i],P[:900,j])
        if g.cointegrated(.05) and g.beta>0:
            w=walk_forward(P[:,i],P[:,j],cfg,.6);sel.append((w.insample.metrics['sharpe'],w.oos.metrics['sharpe'],half_life(g.resid)))
sel=np.array(sel);print('2 pairs',tot,'pass EG',len(sel),round(len(sel)/tot,3),'IS SR mean',sel[:,0].mean().round(2),'top decile IS',np.percentile(sel[:,0],90).round(2),'OOS mean',sel[:,1].mean().round(2),'best-IS pair OOS',sel[sel[:,0].argmax(),:2].round(2))
# 3 EG size with fixed lag under AR(2)-ish/MA diffs + normalisation disagreement
def resid_rw(T,r,ma):
    u=r.normal(size=T+1);return np.cumsum(u[1:]+ma*u[:-1])
for ma in (0,-.5,-.8):
  for T in (250,900):
    r=np.random.default_rng(9);rej=0;dis=0;n=1500
    for _ in range(n):
        a,b=resid_rw(T,r,ma),resid_rw(T,r,ma)
        v1=engle_granger(a,b).cointegrated(.05);v2=engle_granger(b,a).cointegrated(.05);rej+=v1;dis+=v1!=v2
    print('3 MA',ma,'T',T,'size',round(rej/n,3),'y|x vs x|y disagree',round(dis/n,3))
print('3 pair both dirs',engle_granger(y[:900],x[:900]).adf.tstat,engle_granger(x[:900],y[:900]).adf.tstat)
# 3b does EG detect the break? rolling verdict on post-break
for a,b in ((900,1500),(1050,1500),(1100,1500)):print('3b EG',a,b,engle_granger(y[a:b],x[a:b]).adf.tstat.round(2),engle_granger(y[a:b],x[a:b]).cointegrated())
# 4 Kalman: pre-break only, repo params vs MLE-tuned
def ll(k,b=100):return -.5*np.sum(np.log(k.s_var[b:])+k.innovation[b:]**2/k.s_var[b:])
best=max(((ll(kalman_hedge(y[:900],x[:900],dl,rr)),dl,rr) for dl in (1e-9,1e-8,1e-7,1e-6,1e-5,1e-4) for rr in (1e-3,1e-2,.1,1,3,10)))
print('4 ll repo',ll(kalman_hedge(y[:900],x[:900],1e-4,1e-3)),'best',best)
tb=np.where(np.arange(1500)<1050,1.7,1.1);tb[1050:1081]=np.linspace(1.7,1.1,31)
for nm,dl,rr in (('repo',1e-4,1e-3),('mle',best[1],best[2])):
    k=kalman_hedge(y,x,dl,rr);ap=np.r_[k.alpha[0],k.alpha[:-1]];bp=np.r_[k.beta[0],k.beta[:-1]];e=y-ap-bp*x
    pre=slice(150,1050)
    sp_true=(y-12-tb*x)
    print('4',nm,'beta rmse pre',np.sqrt(np.mean((k.beta[pre]-1.7)**2)).round(4),'post-ramp',np.sqrt(np.mean((k.beta[1081:]-1.1)**2)).round(4),'corr(filtered spread,true) pre',np.corrcoef(y[pre]-k.alpha[pre]-k.beta[pre]*x[pre],sp_true[pre])[0,1].round(3),'innov z sd',k.z[pre].std().round(2),'z ac1',np.corrcoef(k.z[150:1049],k.z[151:1050])[0,1].round(2))
    print('   traded pre-break rollz',bt(y[pre],x[pre],rolling_zscore(e,60)[pre],k.beta[pre],cfg,'beta'))
s=y-(wfa:=walk_forward(y,x,cfg,.6)).alpha-wfa.beta*x
print('4 static true-beta pre-break',bt(y[150:1050],x[150:1050],rolling_zscore(y-1.7*x,60)[150:1050],np.full(900,1.7),cfg,'beta'))
# 5 sizing: big intercept pair
r=np.random.default_rng(3);res=[]
for sd in range(100):
    r=np.random.default_rng(sd);T=1000;xx=100*np.exp(np.cumsum(r.normal(0,.015,T)));s0=np.zeros(T)
    for t in range(1,T):s0[t]=.9*s0[t-1]+r.normal(0,.6)
    yy=150+.5*xx+s0;f=ols(yy,xx);z=rolling_zscore(yy-f.params[1]*xx,60);bb=np.full(T,f.params[1])
    res.append((bt(yy,xx,z,bb,cfg,'dollar')['sh'],bt(yy,xx,z,bb,cfg,'beta')['sh']))
print('5 alpha-heavy pair (y=150+.5x): SR equal-dollar vs beta-hedged',np.mean(res,axis=0).round(2),'sd',np.std(res,axis=0).round(2))
yy=300-xx+s0;f=ols(yy,xx);z=rolling_zscore(yy-f.params[1]*xx,60)
print('5 neg beta repo',round(backtest_pair(yy,xx,f.params[0],f.params[1],cfg).metrics['sharpe'],2),'correct legs',bt(yy,xx,z,np.full(T,f.params[1]),cfg,'beta')['sh'])
