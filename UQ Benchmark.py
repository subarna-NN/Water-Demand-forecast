"""
=====================================================================
 EXPERIMENT 5 - Benchmark of uncertainty-quantification (UQ) methods
=====================================================================
"""
import os, numpy as np, pandas as pd, time
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# ------------------------- SET THIS -------------------------
DATA_PATH = "/content/model_ready.csv"
OUT_DIR   = "/content/"
N_SEEDS   = 3             # ensemble size per method (2-5). 3 is a good balance.
# ------------------------- config ---------------------------
LOOKBACK, HORIZON, STRIDE = 168, 168, 6
HIDDEN, EMB_DIM, LAYERS = 64, 8, 1
EPOCHS, PATIENCE, BATCH, LR, VAL_FRAC = 40, 6, 128, 1e-3, 0.15
CAL_SPAN_W = 8
LAM_Q, LAM_VOL, LAM_SMOOTH = 0.3, 0.02, 0.005
QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]
Q_LO, Q_MED, Q_HI = 0, 2, 4
ZQ = [-1.6449, -0.6745, 0.0, 0.6745, 1.6449]   # normal quantiles for Gaussian
ALPHA, CQR_COVER = 0.10, 0.93
MC_T = 30                 # MC-dropout forward passes
MC_P = 0.20               # MC-dropout rate (fair, standard value)
AR_SAMPLES = 100          # DeepAR sample paths
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DMAS = [f"DMA {c}" for c in "ABCDEFGHIJ"]
CTX = ["Rain","Temperature","Humidity","Windspeed",
       "hour_sin","hour_cos","wk_sin","wk_cos","holiday"]
LAGW = [168, 336, 504, 672]

# ----------------------- data / evaluation -----------------------
def load_model():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found. Upload model_ready.csv to Colab first.")
    m = pd.read_csv(DATA_PATH, index_col=0); m.index = pd.to_datetime(m.index, utc=True)
    return m

def eval_windows(m):
    ev=m["eval_week"].values; blocks=[]; start=None
    for i,v in enumerate(ev):
        if v and start is None: start=i
        elif not v and start is not None: blocks.append((start,i)); start=None
    if start is not None: blocks.append((start,len(ev)))
    return blocks

def pi_metrics(y,yhat):
    err=np.abs(np.asarray(y)-np.asarray(yhat))
    return np.nanmean(err[:24]), np.nanmean(err[24:168]), np.nanmax(err[:24])

def prob_metrics(y,lo,hi,qall,taus):
    y=np.asarray(y,float); lo=np.asarray(lo,float); hi=np.asarray(hi,float); ok=~np.isnan(y)
    cov=np.mean((y[ok]>=lo[ok])&(y[ok]<=hi[ok])); mpiw=np.mean(hi[ok]-lo[ok])
    pb=[np.mean(np.maximum(t*(y[ok]-qall[k][ok]),(t-1)*(y[ok]-qall[k][ok]))) for k,t in enumerate(taus)]
    w=(hi[ok]-lo[ok]).copy(); below=y[ok]<lo[ok]; above=y[ok]>hi[ok]
    w[below]+=2/ALPHA*(lo[ok][below]-y[ok][below]); w[above]+=2/ALPHA*(y[ok][above]-hi[ok][above])
    return cov, mpiw, float(np.mean(pb)), float(np.mean(w))

def seasonal_ref(x):
    ref=np.full_like(x,np.nan)
    for t in range(len(x)):
        vals=[x[t-L] for L in LAGW if t-L>=0]
        if vals: ref[t]=np.mean(vals)
    ref[np.isnan(ref)]=np.nanmean(x); return ref

# ---------------- sample builders (shared) ----------------
def dma_norm(m, origin_idx):
    cal_lo=origin_idx-CAL_SPAN_W*168
    return {d:(float(np.nanmean(m[d].values[:cal_lo])), float(np.nanstd(m[d].values[:cal_lo])+1e-6)) for d in DMAS}

def build_seq2seq(m, lo_idx, hi_idx, stats, refs):
    """Encoder/decoder/target/ids/ref samples with target window in [lo,hi)."""
    base=m[CTX].values.astype(np.float32); E,D,T,I,R=[],[],[],[],[]
    for di,d in enumerate(DMAS):
        mu,sd=stats[d]; dem=(m[d].values.astype(np.float32)-mu)/sd; ref=refs[d]
        first=max(0,lo_idx-LOOKBACK); last=hi_idx-HORIZON-LOOKBACK
        for s in range(first,last+1,STRIDE):
            ee=s+LOOKBACK; de=ee+HORIZON
            if ee<lo_idx: continue
            E.append(np.column_stack([dem[s:ee],base[s:ee]])); D.append(base[ee:de])
            T.append(dem[ee:de]); I.append(di); R.append((np.mean(ref[ee:de])-mu)/sd)
    return (np.asarray(E,np.float32),np.asarray(D,np.float32),np.asarray(T,np.float32),
            np.asarray(I,np.int64),np.asarray(R,np.float32))

# ============================ MODELS ============================
class QuantileNet(nn.Module):          # QR / CQR / Ours
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

class GaussNet(nn.Module):             # parametric mean+variance
    def __init__(s,ne,nd):
        super().__init__(); s.emb=nn.Embedding(len(DMAS),EMB_DIM)
        s.enc=nn.LSTM(ne+EMB_DIM,HIDDEN,LAYERS,batch_first=True)
        s.dec=nn.LSTM(nd+EMB_DIM,HIDDEN,LAYERS,batch_first=True)
        s.mu=nn.Linear(HIDDEN,1); s.ls=nn.Linear(HIDDEN,1)
    def forward(s,ex,dx,did):
        e=s.emb(did); ee=e.unsqueeze(1).expand(-1,ex.size(1),-1); ed=e.unsqueeze(1).expand(-1,dx.size(1),-1)
        _,(h,c)=s.enc(torch.cat([ex,ee],-1)); o,_=s.dec(torch.cat([dx,ed],-1),(h,c))
        return s.mu(o).squeeze(-1), s.ls(o).squeeze(-1)      # mu, log_sigma

class MCDropNet(nn.Module):            # point net with dropout kept on at test
    def __init__(s,ne,nd):
        super().__init__(); s.emb=nn.Embedding(len(DMAS),EMB_DIM)
        s.enc=nn.LSTM(ne+EMB_DIM,HIDDEN,LAYERS,batch_first=True)
        s.dec=nn.LSTM(nd+EMB_DIM,HIDDEN,LAYERS,batch_first=True)
        s.do=nn.Dropout(MC_P); s.head=nn.Linear(HIDDEN,1)
    def forward(s,ex,dx,did):
        e=s.emb(did); ee=e.unsqueeze(1).expand(-1,ex.size(1),-1); ed=e.unsqueeze(1).expand(-1,dx.size(1),-1)
        _,(h,c)=s.enc(torch.cat([ex,ee],-1)); o,_=s.dec(torch.cat([dx,ed],-1),(h,c))
        return s.head(s.do(o)).squeeze(-1)

class DeepAR(nn.Module):               # autoregressive Gaussian
    def __init__(s,ncov):
        super().__init__(); s.emb=nn.Embedding(len(DMAS),EMB_DIM)
        s.lstm=nn.LSTM(1+ncov+EMB_DIM,HIDDEN,LAYERS,batch_first=True)
        s.mu=nn.Linear(HIDDEN,1); s.ls=nn.Linear(HIDDEN,1)
    def step(s,prev_y,cov,did,hc):
        # one timestep: prev_y (B,), cov (B,ncov), did (B,) -> add seq dim of 1
        e=s.emb(did).unsqueeze(1)                       # (B,1,EMB)
        x=torch.cat([prev_y.view(-1,1,1),cov.unsqueeze(1),e],-1)  # (B,1,1+ncov+EMB)
        o,hc=s.lstm(x,hc)                               # o (B,1,HIDDEN)
        return s.mu(o).squeeze(1).squeeze(-1), s.ls(o).squeeze(1).squeeze(-1), hc  # (B,),(B,)
    def forward(s,y_seq,cov_seq,did):   # teacher forcing over whole seq
        e=s.emb(did).unsqueeze(1).expand(-1,cov_seq.size(1),-1)
        x=torch.cat([y_seq.unsqueeze(-1),cov_seq,e],-1); o,_=s.lstm(x)
        return s.mu(o).squeeze(-1), s.ls(o).squeeze(-1)

def gauss_nll(mu,ls,y):
    s=F.softplus(ls)+1e-3
    return torch.mean(0.5*np.log(2*np.pi)+torch.log(s)+0.5*((y-mu)/s)**2)

# ============================ TRAINERS ============================
def _loaders(E,D,T,I,R,seed):
    torch.manual_seed(seed); np.random.seed(seed)
    n=len(E); cut=int(n*(1-VAL_FRAC)); idx=np.arange(n)
    def dl(ii,sh):
        ds=TensorDataset(torch.tensor(E[ii]),torch.tensor(D[ii]),torch.tensor(T[ii]),
                         torch.tensor(I[ii]),torch.tensor(R[ii]))
        return DataLoader(ds,batch_size=BATCH,shuffle=sh)
    return dl(idx[:cut],True), dl(idx[cut:],False)

def train_quantile(m,origin,refs,seed,use_physics):
    stats=dma_norm(m,origin); cal_lo=origin-CAL_SPAN_W*168
    E,D,T,I,R=build_seq2seq(m,0,cal_lo,stats,refs); tr,va=_loaders(E,D,T,I,R,seed)
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
    cal_lo=origin-CAL_SPAN_W*168; E,D,T,I,R=build_seq2seq(m,cal_lo,origin,stats,refs); corr={}
    net.eval()
    with torch.no_grad():
        for di,d in enumerate(DMAS):
            sel=I==di
            if sel.sum()==0: corr[d]=0.0; continue
            mu,sd=stats[d]; q,_=net(torch.tensor(E[sel]).to(DEVICE),torch.tensor(D[sel]).to(DEVICE),torch.tensor(I[sel]).to(DEVICE))
            q=q.cpu().numpy(); lo=q[...,Q_LO]*sd+mu; hi=q[...,Q_HI]*sd+mu; y=T[sel]*sd+mu
            Esc=np.maximum(lo-y,y-hi).ravel(); nE=len(Esc); lv=min(1.0,np.ceil((nE+1)*CQR_COVER)/nE)
            corr[d]=float(np.quantile(Esc,lv))
    return corr

def predict_quantile(net,m,origin,stats,corr):
    base=m[CTX].values.astype(np.float32); out={}; net.eval()
    with torch.no_grad():
        for di,d in enumerate(DMAS):
            mu,sd=stats[d]; dem=(m[d].values.astype(np.float32)-mu)/sd
            enc=np.column_stack([dem[origin-LOOKBACK:origin],base[origin-LOOKBACK:origin]])[None]
            dec=base[origin:origin+HORIZON][None]
            q,_=net(torch.tensor(enc).to(DEVICE),torch.tensor(dec).to(DEVICE),torch.tensor([di]).to(DEVICE))
            q=q.cpu().numpy()[0]*sd+mu; qa=[np.clip(q[:,k],0,None) for k in range(len(QUANTILES))]
            lo=np.clip(q[:,Q_LO]-corr[d],0,None); hi=np.clip(q[:,Q_HI]+corr[d],0,None)
            out[d]=dict(median=qa[Q_MED],lo=lo,hi=hi,qall=qa)
    return out

def train_gauss(m,origin,refs,seed):
    stats=dma_norm(m,origin); E,D,T,I,R=build_seq2seq(m,0,origin,stats,refs); tr,va=_loaders(E,D,T,I,R,seed)
    net=GaussNet(1+len(CTX),len(CTX)).to(DEVICE); opt=torch.optim.Adam(net.parameters(),lr=LR); best,bs,bad=np.inf,None,0
    for ep in range(EPOCHS):
        net.train()
        for eb,db,tb,ib,rb in tr:
            eb,db,tb,ib=[x.to(DEVICE) for x in (eb,db,tb,ib)]; opt.zero_grad()
            mu,ls=net(eb,db,ib); loss=gauss_nll(mu,ls,tb.to(DEVICE)); loss.backward(); opt.step()
        net.eval(); vs=vn=0
        with torch.no_grad():
            for eb,db,tb,ib,rb in va:
                eb,db,tb,ib=[x.to(DEVICE) for x in (eb,db,tb,ib)]
                mu,ls=net(eb,db,ib); vs+=gauss_nll(mu,ls,tb.to(DEVICE)).item()*len(eb); vn+=len(eb)
        v=vs/max(vn,1)
        if v<best-1e-5: best,bs,bad=v,{k:val.cpu().clone() for k,val in net.state_dict().items()},0
        else:
            bad+=1
            if bad>=PATIENCE: break
    if bs: net.load_state_dict(bs)
    return net,stats

def predict_gauss(net,m,origin,stats,_):
    base=m[CTX].values.astype(np.float32); out={}; net.eval()
    with torch.no_grad():
        for di,d in enumerate(DMAS):
            mu,sd=stats[d]; dem=(m[d].values.astype(np.float32)-mu)/sd
            enc=np.column_stack([dem[origin-LOOKBACK:origin],base[origin-LOOKBACK:origin]])[None]
            dec=base[origin:origin+HORIZON][None]
            mn,ls=net(torch.tensor(enc).to(DEVICE),torch.tensor(dec).to(DEVICE),torch.tensor([di]).to(DEVICE))
            mn=mn.cpu().numpy()[0]; sg=F.softplus(ls).cpu().numpy()[0]+1e-3
            qa=[np.clip((mn+z*sg)*sd+mu,0,None) for z in ZQ]
            out[d]=dict(median=qa[Q_MED],lo=qa[Q_LO],hi=qa[Q_HI],qall=qa)
    return out

def train_mcdrop(m,origin,refs,seed):
    stats=dma_norm(m,origin); E,D,T,I,R=build_seq2seq(m,0,origin,stats,refs); tr,va=_loaders(E,D,T,I,R,seed)
    net=MCDropNet(1+len(CTX),len(CTX)).to(DEVICE); opt=torch.optim.Adam(net.parameters(),lr=LR); best,bs,bad=np.inf,None,0
    for ep in range(EPOCHS):
        net.train()
        for eb,db,tb,ib,rb in tr:
            eb,db,tb,ib=[x.to(DEVICE) for x in (eb,db,tb,ib)]; opt.zero_grad()
            loss=F.mse_loss(net(eb,db,ib),tb.to(DEVICE)); loss.backward(); opt.step()
        net.eval(); vs=vn=0
        with torch.no_grad():
            for eb,db,tb,ib,rb in va:
                eb,db,tb,ib=[x.to(DEVICE) for x in (eb,db,tb,ib)]
                vs+=F.mse_loss(net(eb,db,ib),tb.to(DEVICE)).item()*len(eb); vn+=len(eb)
        v=vs/max(vn,1)
        if v<best-1e-5: best,bs,bad=v,{k:val.cpu().clone() for k,val in net.state_dict().items()},0
        else:
            bad+=1
            if bad>=PATIENCE: break
    if bs: net.load_state_dict(bs)
    return net,stats

def predict_mcdrop(net,m,origin,stats,_):
    base=m[CTX].values.astype(np.float32); out={}
    net.train()   # keep dropout ACTIVE at inference (MC-dropout)
    with torch.no_grad():
        for di,d in enumerate(DMAS):
            mu,sd=stats[d]; dem=(m[d].values.astype(np.float32)-mu)/sd
            enc=np.column_stack([dem[origin-LOOKBACK:origin],base[origin-LOOKBACK:origin]])[None]
            dec=base[origin:origin+HORIZON][None]
            et=torch.tensor(np.repeat(enc,MC_T,0)).to(DEVICE); dt=torch.tensor(np.repeat(dec,MC_T,0)).to(DEVICE)
            it=torch.tensor([di]*MC_T).to(DEVICE)
            samp=net(et,dt,it).cpu().numpy()*sd+mu                # (MC_T, H)
            qa=[np.clip(np.quantile(samp,t,axis=0),0,None) for t in QUANTILES]
            out[d]=dict(median=qa[Q_MED],lo=qa[Q_LO],hi=qa[Q_HI],qall=qa)
    return out

def train_deepar(m,origin,refs,seed):
    stats=dma_norm(m,origin); torch.manual_seed(seed); np.random.seed(seed)
    base=m[CTX].values.astype(np.float32); Y,C,I=[],[],[]
    L=LOOKBACK+HORIZON
    for di,d in enumerate(DMAS):
        mu,sd=stats[d]; dem=(m[d].values.astype(np.float32)-mu)/sd
        for s in range(0,origin-L+1,STRIDE):
            Y.append(dem[s:s+L]); C.append(base[s:s+L]); I.append(di)
    Y=np.asarray(Y,np.float32); C=np.asarray(C,np.float32); I=np.asarray(I,np.int64)
    n=len(Y); cut=int(n*(1-VAL_FRAC))
    def dl(a,b,sh): return DataLoader(TensorDataset(torch.tensor(Y[a:b]),torch.tensor(C[a:b]),torch.tensor(I[a:b])),batch_size=BATCH,shuffle=sh)
    tr,va=dl(0,cut,True),dl(cut,n,False)
    net=DeepAR(len(CTX)).to(DEVICE); opt=torch.optim.Adam(net.parameters(),lr=LR); best,bs,bad=np.inf,None,0
    for ep in range(EPOCHS):
        net.train()
        for yb,cb,ib in tr:
            yb,cb,ib=yb.to(DEVICE),cb.to(DEVICE),ib.to(DEVICE); opt.zero_grad()
            # teacher forcing: predict y_t from y_{t-1}; shift by one
            mu,ls=net(yb[:,:-1],cb[:,1:],ib); loss=gauss_nll(mu,ls,yb[:,1:]); loss.backward(); opt.step()
        net.eval(); vs=vn=0
        with torch.no_grad():
            for yb,cb,ib in va:
                yb,cb,ib=yb.to(DEVICE),cb.to(DEVICE),ib.to(DEVICE)
                mu,ls=net(yb[:,:-1],cb[:,1:],ib); vs+=gauss_nll(mu,ls,yb[:,1:]).item()*len(yb); vn+=len(yb)
        v=vs/max(vn,1)
        if v<best-1e-5: best,bs,bad=v,{k:val.cpu().clone() for k,val in net.state_dict().items()},0
        else:
            bad+=1
            if bad>=PATIENCE: break
    if bs: net.load_state_dict(bs)
    return net,stats

def predict_deepar(net,m,origin,stats,_):
    base=m[CTX].values.astype(np.float32); out={}; net.eval()
    with torch.no_grad():
        for di,d in enumerate(DMAS):
            mu,sd=stats[d]; dem=(m[d].values.astype(np.float32)-mu)/sd
            cov=base[origin-LOOKBACK:origin+HORIZON]                 # covariates over cond+pred
            did=torch.tensor([di]*AR_SAMPLES).to(DEVICE)
            # condition on the lookback with teacher forcing
            y_hist=torch.tensor(np.tile(dem[origin-LOOKBACK:origin],(AR_SAMPLES,1))).to(DEVICE)
            c_hist=torch.tensor(np.tile(cov[1:LOOKBACK+1],(AR_SAMPLES,1,1))).to(DEVICE)
            e=net.emb(did).unsqueeze(1).expand(-1,c_hist.size(1),-1)
            x=torch.cat([y_hist.unsqueeze(-1),c_hist,e],-1); o,hc=net.lstm(x)
            prev=torch.tensor(np.tile(dem[origin-1],(AR_SAMPLES,))).float().to(DEVICE)
            paths=[]
            for t in range(HORIZON):
                ct=torch.tensor(np.tile(cov[LOOKBACK+t],(AR_SAMPLES,1))).to(DEVICE)
                mn,ls,hc=net.step(prev,ct,did,hc); sg=F.softplus(ls)+1e-3   # (S,),(S,)
                prev=mn+sg*torch.randn_like(sg)
                paths.append(prev.cpu().numpy())
            samp=np.stack(paths,1)*sd+mu                            # (AR_SAMPLES, H)
            qa=[np.clip(np.quantile(samp,t,axis=0),0,None) for t in QUANTILES]
            out[d]=dict(median=qa[Q_MED],lo=qa[Q_LO],hi=qa[Q_HI],qall=qa)
    return out

# ============================ RUN ============================
METHODS = {
 "QR":        dict(train=lambda m,o,r,s: train_quantile(m,o,r,s,False), pred=predict_quantile, calib=False),
 "CQR":       dict(train=lambda m,o,r,s: train_quantile(m,o,r,s,False), pred=predict_quantile, calib=True),
 "Gaussian":  dict(train=train_gauss,   pred=predict_gauss,   calib=False),
 "MC-Dropout":dict(train=train_mcdrop,  pred=predict_mcdrop,  calib=False),
 "DeepAR":    dict(train=train_deepar,  pred=predict_deepar,  calib=False),
 "BCP-PINN":  dict(train=lambda m,o,r,s: train_quantile(m,o,r,s,True),  pred=predict_quantile, calib=True),
}

def ensemble(runs):
    ens={}
    for wi in runs[0]:
        s,e,_=runs[0][wi]; pe={}
        for d in DMAS:
            pe[d]=dict(median=np.mean([r[wi][2][d]["median"] for r in runs],0),
                       lo=np.mean([r[wi][2][d]["lo"] for r in runs],0),
                       hi=np.mean([r[wi][2][d]["hi"] for r in runs],0),
                       qall=[np.mean([r[wi][2][d]["qall"][k] for r in runs],0) for k in range(len(QUANTILES))])
        ens[wi]=(s,e,pe)
    return ens

def score(m,pred):
    rows=[]
    for wi,(s,e,p) in pred.items():
        for d in DMAS:
            y=m[d].iloc[s:e].values.astype(float).copy()
            imp=m[d+"_imp"].iloc[s:e].values.astype(bool); y[imp]=np.nan; P=p[d]
            pi1,pi2,pi3=pi_metrics(y,P["median"]); cov,mpiw,pb,wk=prob_metrics(y,P["lo"],P["hi"],P["qall"],QUANTILES)
            rows.append(dict(dma=d,PI1=pi1,COV90=cov,MPIW=mpiw,Pinball=pb,Winkler=wk))
    return pd.DataFrame(rows)

if __name__=="__main__":
    t0=time.time(); m=load_model(); wins=eval_windows(m)
    refs={d:seasonal_ref(m[d].values.astype(np.float32)) for d in DMAS}
    print(f"Device:{DEVICE} | {len(wins)} weeks | {N_SEEDS} seeds/method | methods: {list(METHODS)}")
    table=[]
    for name,cfg in METHODS.items():
        runs=[]
        for si in range(N_SEEDS):
            seed=42+si; per_win={}
            for wi,(s,e) in enumerate(wins,1):
                net,stats=cfg["train"](m,s,refs,seed)
                corr=cqr_corr(net,m,s,stats,refs) if cfg["calib"] else {d:0.0 for d in DMAS}
                per_win[wi]=(s,e,cfg["pred"](net,m,s,stats,corr))
            runs.append(per_win)
        sdf=score(m,ensemble(runs)).mean(numeric_only=True)
        table.append(dict(Method=name,PI1=sdf.PI1,COV90=sdf.COV90,MPIW=sdf.MPIW,Pinball=sdf.Pinball,Winkler=sdf.Winkler))
        print(f"  {name:11s} PI1={sdf.PI1:.3f} COV90={sdf.COV90:.3f} MPIW={sdf.MPIW:.3f} Pinball={sdf.Pinball:.3f} Winkler={sdf.Winkler:.3f}")
    T=pd.DataFrame(table).set_index("Method")
    print("\n=== Experiment 5: UQ-method comparison (mean over 4 weeks x 10 DMAs) ===")
    print(T.round(4).to_string())
    T.to_csv(OUT_DIR+"exp5_uq_comparison.csv")
    print(f"\nSaved -> {OUT_DIR}exp5_uq_comparison.csv   ({time.time()-t0:.0f}s)")

    # ---- plot: coverage (target 0.90) and Winkler (lower=better) ----
    fig,ax=plt.subplots(1,2,figsize=(12,4.2))
    cols=["#2c7bb6","#abd9e9","#fdae61","#f46d43","#9970ab","#1a9850"]
    ax[0].bar(T.index,T["COV90"],color=cols); ax[0].axhline(0.90,ls="--",color="k",lw=1,label="90% target")
    ax[0].set_title("Coverage (closer to 0.90 is better)"); ax[0].set_ylim(0.6,1.0); ax[0].legend(); ax[0].tick_params(axis="x",rotation=30)
    ax[1].bar(T.index,T["Winkler"],color=cols); ax[1].set_title("Winkler score (lower is better)"); ax[1].tick_params(axis="x",rotation=30)
    plt.tight_layout(); plt.show()
