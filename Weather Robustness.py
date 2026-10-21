"""
=====================================================================
 EXPERIMENT 6 - Robustness under imperfect weather forecasts
=====================================================================
"""
import os, numpy as np, pandas as pd, time
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# ------------------------- SET THIS -------------------------
DATA_PATH = "/content/model_ready.csv"
OUT_DIR   = "/content/"
N_SEEDS   = 3
NOISE_LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]   # fraction of each weather var's std
# ------------------------- config ---------------------------
LOOKBACK, HORIZON, STRIDE = 168, 168, 6
HIDDEN, EMB_DIM, LAYERS = 64, 8, 1
EPOCHS, PATIENCE, BATCH, LR, VAL_FRAC = 40, 6, 128, 1e-3, 0.15
CAL_SPAN_W = 8
LAM_Q, LAM_VOL, LAM_SMOOTH = 0.3, 0.02, 0.005
QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]
Q_LO, Q_MED, Q_HI = 0, 2, 4
ALPHA, CQR_COVER = 0.10, 0.93
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DMAS = [f"DMA {c}" for c in "ABCDEFGHIJ"]
WEATHER = ["Rain","Temperature","Humidity","Windspeed"]      # perturbed (cols 0-3 of CTX)
CTX = WEATHER + ["hour_sin","hour_cos","wk_sin","wk_cos","holiday"]
LAGW = [168, 336, 504, 672]

# ----------------------- data / evaluation -----------------------
def load_model():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found. Upload model_ready.csv to Colab first.")
    m=pd.read_csv(DATA_PATH,index_col=0); m.index=pd.to_datetime(m.index,utc=True); return m

def eval_windows(m):
    ev=m["eval_week"].values; blocks=[]; start=None
    for i,v in enumerate(ev):
        if v and start is None: start=i
        elif not v and start is not None: blocks.append((start,i)); start=None
    if start is not None: blocks.append((start,len(ev)))
    return blocks

def pi_metrics(y,yhat):
    err=np.abs(np.asarray(y)-np.asarray(yhat)); return np.nanmean(err[:24]),np.nanmean(err[24:168]),np.nanmax(err[:24])

def prob_metrics(y,lo,hi,qall,taus):
    y=np.asarray(y,float); lo=np.asarray(lo,float); hi=np.asarray(hi,float); ok=~np.isnan(y)
    cov=np.mean((y[ok]>=lo[ok])&(y[ok]<=hi[ok])); mpiw=np.mean(hi[ok]-lo[ok])
    pb=[np.mean(np.maximum(t*(y[ok]-qall[k][ok]),(t-1)*(y[ok]-qall[k][ok]))) for k,t in enumerate(taus)]
    w=(hi[ok]-lo[ok]).copy(); b=y[ok]<lo[ok]; a=y[ok]>hi[ok]
    w[b]+=2/ALPHA*(lo[ok][b]-y[ok][b]); w[a]+=2/ALPHA*(y[ok][a]-hi[ok][a])
    return cov,mpiw,float(np.mean(pb)),float(np.mean(w))

def seasonal_ref(x):
    ref=np.full_like(x,np.nan)
    for t in range(len(x)):
        vals=[x[t-L] for L in LAGW if t-L>=0]
        if vals: ref[t]=np.mean(vals)
    ref[np.isnan(ref)]=np.nanmean(x); return ref

def dma_norm(m,origin):
    cal=origin-CAL_SPAN_W*168
    return {d:(float(np.nanmean(m[d].values[:cal])),float(np.nanstd(m[d].values[:cal])+1e-6)) for d in DMAS}

def build(m,lo,hi,stats,refs):
    base=m[CTX].values.astype(np.float32); E,D,T,I,R=[],[],[],[],[]
    for di,d in enumerate(DMAS):
        mu,sd=stats[d]; dem=(m[d].values.astype(np.float32)-mu)/sd; ref=refs[d]
        first=max(0,lo-LOOKBACK); last=hi-HORIZON-LOOKBACK
        for s in range(first,last+1,STRIDE):
            ee=s+LOOKBACK; de=ee+HORIZON
            if ee<lo: continue
            E.append(np.column_stack([dem[s:ee],base[s:ee]])); D.append(base[ee:de])
            T.append(dem[ee:de]); I.append(di); R.append((np.mean(ref[ee:de])-mu)/sd)
    return (np.asarray(E,np.float32),np.asarray(D,np.float32),np.asarray(T,np.float32),
            np.asarray(I,np.int64),np.asarray(R,np.float32))

# ----------------------------- model -----------------------------
class QuantileNet(nn.Module):
    def __init__(s,ne,nd,nq):
        super().__init__(); s.emb=nn.Embedding(len(DMAS),EMB_DIM)
        s.enc=nn.LSTM(ne+EMB_DIM,HIDDEN,LAYERS,batch_first=True)
        s.dec=nn.LSTM(nd+EMB_DIM,HIDDEN,LAYERS,batch_first=True)
        s.mu=nn.Linear(HIDDEN,1); s.dn=nn.Linear(HIDDEN,nq//2); s.up=nn.Linear(HIDDEN,nq//2)
    def forward(s,ex,dx,did):
        e=s.emb(did); ee=e.unsqueeze(1).expand(-1,ex.size(1),-1); ed=e.unsqueeze(1).expand(-1,dx.size(1),-1)
        _,(h,c)=s.enc(torch.cat([ex,ee],-1)); o,_=s.dec(torch.cat([dx,ed],-1),(h,c))
        mu=s.mu(o); low=torch.flip(mu-torch.cumsum(F.softplus(s.dn(o)),-1),[-1]); high=mu+torch.cumsum(F.softplus(s.up(o)),-1)
        return torch.cat([low,mu,high],-1), mu

def train_quantile(m,origin,refs,seed,use_physics):
    torch.manual_seed(seed); np.random.seed(seed)
    stats=dma_norm(m,origin); cal_lo=origin-CAL_SPAN_W*168
    E,D,T,I,R=build(m,0,cal_lo,stats,refs)
    n=len(E); cut=int(n*(1-VAL_FRAC)); idx=np.arange(n)
    def dl(ii,sh): return DataLoader(TensorDataset(torch.tensor(E[ii]),torch.tensor(D[ii]),torch.tensor(T[ii]),torch.tensor(I[ii]),torch.tensor(R[ii])),batch_size=BATCH,shuffle=sh)
    tr,va=dl(idx[:cut],True),dl(idx[cut:],False)
    net=QuantileNet(1+len(CTX),len(CTX),len(QUANTILES)-1).to(DEVICE)
    opt=torch.optim.Adam(net.parameters(),lr=LR); best,bs,bad=np.inf,None,0
    for ep in range(EPOCHS):
        net.train()
        for eb,db,tb,ib,rb in tr:
            eb,db,tb,ib,rb=[x.to(DEVICE) for x in (eb,db,tb,ib,rb)]; opt.zero_grad()
            q,mu=net(eb,db,ib); qb=mu.detach()+(q-mu); loss=F.mse_loss(q[...,Q_MED],tb)
            for k,t in enumerate(QUANTILES):
                if k==Q_MED: continue
                d=tb-qb[...,k]; loss=loss+LAM_Q*torch.mean(torch.maximum(t*d,(t-1)*d))
            if use_physics:
                md=q[...,Q_MED]; loss=loss+LAM_VOL*torch.mean((md.mean(1)-rb)**2)
                d2=md[:,2:]-2*md[:,1:-1]+md[:,:-2]; loss=loss+LAM_SMOOTH*torch.mean(d2**2)
            loss.backward(); opt.step()
        net.eval(); vs=vn=0
        with torch.no_grad():
            for eb,db,tb,ib,rb in va:
                eb,db,tb,ib=[x.to(DEVICE) for x in (eb,db,tb,ib)]
                q,_=net(eb,db,ib); vs+=F.mse_loss(q[...,Q_MED],tb.to(DEVICE)).item()*len(eb); vn+=len(eb)
        v=vs/max(vn,1)
        if v<best-1e-5: best,bs,bad=v,{k:val.cpu().clone() for k,val in net.state_dict().items()},0
        else:
            bad+=1
            if bad>=PATIENCE: break
    if bs: net.load_state_dict(bs)
    return net,stats

def cqr_corr(net,m,origin,stats,refs):
    cal_lo=origin-CAL_SPAN_W*168; E,D,T,I,R=build(m,cal_lo,origin,stats,refs); corr={}; net.eval()
    with torch.no_grad():
        for di,d in enumerate(DMAS):
            sel=I==di
            if sel.sum()==0: corr[d]=0.0; continue
            mu,sd=stats[d]; q,_=net(torch.tensor(E[sel]).to(DEVICE),torch.tensor(D[sel]).to(DEVICE),torch.tensor(I[sel]).to(DEVICE))
            q=q.cpu().numpy(); lo=q[...,Q_LO]*sd+mu; hi=q[...,Q_HI]*sd+mu; y=T[sel]*sd+mu
            Esc=np.maximum(lo-y,y-hi).ravel(); nE=len(Esc); lv=min(1.0,np.ceil((nE+1)*CQR_COVER)/nE)
            corr[d]=float(np.quantile(Esc,lv))
    return corr

def predict(net,m,origin,stats,corr,noise=None):
    """noise[dma] = array(HORIZON,4) added to the future weather cols (0-3)."""
    base=m[CTX].values.astype(np.float32); out={}; net.eval()
    with torch.no_grad():
        for di,d in enumerate(DMAS):
            mu,sd=stats[d]; dem=(m[d].values.astype(np.float32)-mu)/sd
            enc=np.column_stack([dem[origin-LOOKBACK:origin],base[origin-LOOKBACK:origin]])[None]
            dec=base[origin:origin+HORIZON].copy()
            if noise is not None:
                dec[:, :4]=dec[:, :4]+noise[d]; dec[:,0]=np.clip(dec[:,0],0,None)   # rain>=0
            q,_=net(torch.tensor(enc).to(DEVICE),torch.tensor(dec[None]).to(DEVICE),torch.tensor([di]).to(DEVICE))
            q=q.cpu().numpy()[0]*sd+mu; qa=[np.clip(q[:,k],0,None) for k in range(len(QUANTILES))]
            lo=np.clip(q[:,Q_LO]-corr[d],0,None); hi=np.clip(q[:,Q_HI]+corr[d],0,None)
            out[d]=dict(median=qa[Q_MED],lo=lo,hi=hi,qall=qa)
    return out

def ensemble(preds_list):
    ens={}
    for wi in preds_list[0]:
        s,e,_=preds_list[0][wi]; pe={}
        for d in DMAS:
            pe[d]=dict(median=np.mean([p[wi][2][d]["median"] for p in preds_list],0),
                       lo=np.mean([p[wi][2][d]["lo"] for p in preds_list],0),
                       hi=np.mean([p[wi][2][d]["hi"] for p in preds_list],0),
                       qall=[np.mean([p[wi][2][d]["qall"][k] for p in preds_list],0) for k in range(len(QUANTILES))])
        ens[wi]=(s,e,pe)
    return ens

def score(m,pred):
    rows=[]
    for wi,(s,e,p) in pred.items():
        for d in DMAS:
            y=m[d].iloc[s:e].values.astype(float).copy()
            imp=m[d+"_imp"].iloc[s:e].values.astype(bool); y[imp]=np.nan; P=p[d]
            pi1,_,_=pi_metrics(y,P["median"]); cov,mpiw,pb,wk=prob_metrics(y,P["lo"],P["hi"],P["qall"],QUANTILES)
            rows.append(dict(PI1=pi1,COV90=cov,MPIW=mpiw,Winkler=wk))
    return pd.DataFrame(rows).mean(numeric_only=True)

if __name__=="__main__":
    t0=time.time(); m=load_model(); wins=eval_windows(m)
    refs={d:seasonal_ref(m[d].values.astype(np.float32)) for d in DMAS}
    wstd={w:float(m[w].std()) for w in WEATHER}
    print(f"Device:{DEVICE} | seeds={N_SEEDS}")
    print("weather std:", {k:round(v,2) for k,v in wstd.items()})

    # --- train both settings ONCE: store (net, stats, corr) per setting/seed/window ---
    trained={True:[], False:[]}
    for phys in (True, False):
        for si in range(N_SEEDS):
            seed=42+si; per_win={}
            for wi,(s,e) in enumerate(wins,1):
                net,stats=train_quantile(m,s,refs,seed,phys)
                corr=cqr_corr(net,m,s,stats,refs)
                per_win[wi]=(s,e,net,stats,corr)
            trained[phys].append(per_win)
        print(f"  trained physics={phys} ({N_SEEDS} seeds)  [{time.time()-t0:.0f}s]")

    # --- precompute a fixed noise draw per level/window/dma (shared ON & OFF) ---
    def make_noise(level, wi):
        rng=np.random.default_rng(10_000*wi + int(level*1000))
        return {d: rng.standard_normal((HORIZON,4))*np.array([level*wstd[w] for w in WEATHER],np.float32)
                for d in DMAS}

    # --- evaluate every noise level for ON and OFF ---
    rows=[]
    for level in NOISE_LEVELS:
        res={}
        for phys in (True, False):
            seed_preds=[]
            for per_win in trained[phys]:
                pw={}
                for wi,(s,e,net,stats,corr) in per_win.items():
                    noise = None if level==0.0 else make_noise(level, wi)
                    pw[wi]=(s,e,predict(net,m,s,stats,corr,noise))
                seed_preds.append(pw)
            res[phys]=score(m,ensemble(seed_preds))
        rows.append(dict(level=level,
                         PI1_on=res[True].PI1,  COV_on=res[True].COV90,  Wink_on=res[True].Winkler,
                         PI1_off=res[False].PI1,COV_off=res[False].COV90,Wink_off=res[False].Winkler))
        print(f"  noise {level:.2f} | ON  PI1={res[True].PI1:.3f} COV={res[True].COV90:.3f} Wink={res[True].Winkler:.3f}"
              f" | OFF PI1={res[False].PI1:.3f} COV={res[False].COV90:.3f} Wink={res[False].Winkler:.3f}")
    R=pd.DataFrame(rows)
    print("\n=== Experiment 6: robustness to weather-forecast error ===")
    print(R.round(4).to_string(index=False))
    R.to_csv(OUT_DIR+"exp6_weather_robust.csv",index=False)
    # degradation relative to clean inputs
    base_on=R.loc[R.level==0,"PI1_on"].values[0]; base_off=R.loc[R.level==0,"PI1_off"].values[0]
    print(f"\nPoint degradation at max noise:  physics-ON +{100*(R.PI1_on.iloc[-1]/base_on-1):.1f}%"
          f"  vs  physics-OFF +{100*(R.PI1_off.iloc[-1]/base_off-1):.1f}%")
    print(f"Saved -> {OUT_DIR}exp6_weather_robust.csv   ({time.time()-t0:.0f}s)")

    # --------------------- plots (inline, not saved) ---------------------
    fig,ax=plt.subplots(1,3,figsize=(15,4.3)); ON,OFF="#1a9850","#d73027"
    ax[0].plot(R.level,R.PI1_on,"o-",color=ON,lw=2,label="Physics ON")
    ax[0].plot(R.level,R.PI1_off,"s--",color=OFF,lw=2,label="Physics OFF")
    ax[0].set_ylabel("PI1 (day-1 MAE, L/s)"); ax[0].set_title("Point accuracy vs weather error")
    ax[1].plot(R.level,R.COV_on,"o-",color=ON,lw=2,label="Physics ON")
    ax[1].plot(R.level,R.COV_off,"s--",color=OFF,lw=2,label="Physics OFF")
    ax[1].axhline(0.90,ls=":",color="k",lw=1,label="90% target")
    ax[1].set_ylabel("Coverage"); ax[1].set_title("Calibration vs weather error")
    ax[2].plot(R.level,R.Wink_on,"o-",color=ON,lw=2,label="Physics ON")
    ax[2].plot(R.level,R.Wink_off,"s--",color=OFF,lw=2,label="Physics OFF")
    ax[2].set_ylabel("Winkler (lower=better)"); ax[2].set_title("Uncertainty quality vs weather error")
    for a in ax:
        a.set_xlabel("Weather-forecast noise (fraction of variable std)"); a.grid(alpha=0.25); a.legend()
    plt.tight_layout(); plt.show()
