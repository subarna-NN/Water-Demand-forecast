import os, numpy as np, pandas as pd, time
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# ------------------------- SET THIS -------------------------
DATA_PATH = "/content/model_ready.csv"     # where you uploaded the clean data
OUT_DIR   = "/content/"                     # where results are written
PLOT_DMAS = ["DMA E", "DMA C", "DMA B"]     # which zones to plot (big / small / weather-sensitive)
PLOT_WINDOW = 1                             # which eval week (1-4) to plot
# ------------------------------------------------------------

# ----------------------------- config -----------------------------
SEED, LOOKBACK, HORIZON, STRIDE = 42, 168, 168, 6
HIDDEN, EMB_DIM, LAYERS, DROPOUT = 64, 8, 1, 0.0
EPOCHS, PATIENCE, BATCH, LR, VAL_FRAC = 40, 6, 128, 1e-3, 0.15
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DMAS = [f"DMA {c}" for c in "ABCDEFGHIJ"]
CTX = ["Rain","Temperature","Humidity","Windspeed",
       "hour_sin","hour_cos","wk_sin","wk_cos","holiday"]   # known context feats
torch.manual_seed(SEED); np.random.seed(SEED)

# ----------------------- evaluation (inlined) ----------------------
def load_model():
    path = DATA_PATH
    if not os.path.exists(path):
        # On Colab: if the file wasn't uploaded, open an upload dialog automatically.
        try:
            from google.colab import files
            print("model_ready.csv not found at", DATA_PATH,
                  "\n-> Opening upload box. Choose your model_ready.csv file:")
            up = files.upload()
            path = list(up.keys())[0]
        except Exception:
            raise FileNotFoundError(
                f"{DATA_PATH} not found. Upload model_ready.csv to the Colab "
                f"session (left panel > Files > upload), then run again.")
    m = pd.read_csv(path, index_col=0)
    m.index = pd.to_datetime(m.index, utc=True)
    return m

def eval_windows(m):
    ev = m["eval_week"].values; blocks=[]; start=None
    for i,v in enumerate(ev):
        if v and start is None: start=i
        elif not v and start is not None: blocks.append((start,i)); start=None
    if start is not None: blocks.append((start,len(ev)))
    return blocks

def pi_metrics(y_true, y_pred):
    err = np.abs(np.asarray(y_true) - np.asarray(y_pred))
    return np.nanmean(err[:24]), np.nanmean(err[24:168]), np.nanmax(err[:24])

# ----------------------------- model -----------------------------
class Seq2SeqLSTM(nn.Module):
    def __init__(self, n_enc, n_dec, n_dma):
        super().__init__()
        self.emb = nn.Embedding(n_dma, EMB_DIM)
        self.encoder = nn.LSTM(n_enc+EMB_DIM, HIDDEN, LAYERS, batch_first=True, dropout=DROPOUT)
        self.decoder = nn.LSTM(n_dec+EMB_DIM, HIDDEN, LAYERS, batch_first=True, dropout=DROPOUT)
        self.head = nn.Linear(HIDDEN, 1)
    def forward(self, enc_x, dec_x, dma_id):
        e = self.emb(dma_id)
        e_enc = e.unsqueeze(1).expand(-1, enc_x.size(1), -1)
        e_dec = e.unsqueeze(1).expand(-1, dec_x.size(1), -1)
        _, (h,c) = self.encoder(torch.cat([enc_x, e_enc], -1))
        out,_ = self.decoder(torch.cat([dec_x, e_dec], -1), (h,c))
        return self.head(out).squeeze(-1)

# ----------------------- sample construction -----------------------
def build_samples(m, origin_idx, dma_stats):
    base = m[CTX].values.astype(np.float32)
    enc_l, dec_l, tgt_l, id_l = [], [], [], []
    for d_i, dma in enumerate(DMAS):
        mu, sd = dma_stats[dma]
        dem = (m[dma].values.astype(np.float32) - mu) / sd
        last_start = origin_idx - HORIZON - LOOKBACK
        for s in range(0, last_start+1, STRIDE):
            ee = s+LOOKBACK; de = ee+HORIZON
            enc_l.append(np.column_stack([dem[s:ee], base[s:ee]]))
            dec_l.append(base[ee:de]); tgt_l.append(dem[ee:de]); id_l.append(d_i)
    return (np.asarray(enc_l,np.float32), np.asarray(dec_l,np.float32),
            np.asarray(tgt_l,np.float32), np.asarray(id_l,np.int64))

def train_for_window(m, origin_idx):
    dma_stats = {dma:(float(np.nanmean(m[dma].values[:origin_idx])),
                      float(np.nanstd(m[dma].values[:origin_idx])+1e-6)) for dma in DMAS}
    enc,dec,tgt,ids = build_samples(m, origin_idx, dma_stats)
    n = len(enc); cut = int(n*(1-VAL_FRAC)); idx = np.arange(n)
    def dl(ii, sh):
        ds = TensorDataset(torch.tensor(enc[ii]),torch.tensor(dec[ii]),
                           torch.tensor(tgt[ii]),torch.tensor(ids[ii]))
        return DataLoader(ds, batch_size=BATCH, shuffle=sh)
    tr_dl, va_dl = dl(idx[:cut],True), dl(idx[cut:],False)
    model = Seq2SeqLSTM(1+len(CTX), len(CTX), len(DMAS)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR); loss_fn = nn.MSELoss()
    best, best_state, bad = np.inf, None, 0
    for ep in range(EPOCHS):
        model.train()
        for eb,db,tb,ib in tr_dl:
            eb,db,tb,ib = [x.to(DEVICE) for x in (eb,db,tb,ib)]
            opt.zero_grad(); loss_fn(model(eb,db,ib), tb).backward(); opt.step()
        model.eval(); vs=vn=0
        with torch.no_grad():
            for eb,db,tb,ib in va_dl:
                eb,db,tb,ib = [x.to(DEVICE) for x in (eb,db,tb,ib)]
                vs += loss_fn(model(eb,db,ib),tb).item()*len(eb); vn += len(eb)
        va = vs/max(vn,1)
        if va < best-1e-5: best,best_state,bad = va,{k:v.cpu().clone() for k,v in model.state_dict().items()},0
        else:
            bad += 1
            if bad>=PATIENCE: break
    if best_state: model.load_state_dict(best_state)
    return model, dma_stats

def predict_window(model, m, origin_idx, dma_stats):
    base = m[CTX].values.astype(np.float32); preds={}; model.eval()
    with torch.no_grad():
        for d_i,dma in enumerate(DMAS):
            mu,sd = dma_stats[dma]
            dem = (m[dma].values.astype(np.float32)-mu)/sd
            enc = np.column_stack([dem[origin_idx-LOOKBACK:origin_idx],
                                   base[origin_idx-LOOKBACK:origin_idx]])[None]
            dec = base[origin_idx:origin_idx+HORIZON][None]
            p = model(torch.tensor(enc).to(DEVICE), torch.tensor(dec).to(DEVICE),
                      torch.tensor([d_i]).to(DEVICE)).cpu().numpy()[0]
            preds[dma] = np.clip(p*sd+mu, 0, None)
    return preds

# ----------------------------- run -----------------------------
if __name__ == "__main__":
    t0=time.time(); m=load_model(); wins=eval_windows(m)
    print(f"Device: {DEVICE} | eval windows: {len(wins)}")
    rows=[]; saved_preds={}       # saved_preds[window_index] = {dma: pred array}
    for wi,(s,e) in enumerate(wins,1):
        print(f"\n[Window {wi}/{len(wins)}] origin {m.index[s]} ... training")
        model,stats = train_for_window(m, s)
        preds = predict_window(model, m, s, stats)
        saved_preds[wi] = (s, e, preds)
        for dma in DMAS:
            y = m[dma].iloc[s:e].values.astype(float).copy()
            imp = m[dma+"_imp"].iloc[s:e].values.astype(bool); y[imp]=np.nan
            pi1,pi2,pi3 = pi_metrics(y, preds[dma])
            rows.append(dict(window=str(m.index[s].date()), dma=dma, PI1=pi1,PI2=pi2,PI3=pi3))
        d = pd.DataFrame(rows); d = d[d.window==str(m.index[s].date())]
        print(f"  window mean  PI1={d.PI1.mean():.3f}  PI2={d.PI2.mean():.3f}  PI3={d.PI3.mean():.3f}")
    detail = pd.DataFrame(rows)
    summary = detail[["PI1","PI2","PI3"]].mean(); summary.name="LSTM"
    print("\n=== LSTM baseline (mean over 4 windows x 10 DMAs, L/s) ===")
    print(summary.round(4).to_string())
    detail.to_csv(OUT_DIR+"baseline_lstm_detail.csv", index=False)
    summary.to_frame().T.to_csv(OUT_DIR+"baseline_lstm_summary.csv")
    print(f"\nSaved results CSVs to {OUT_DIR}   ({time.time()-t0:.0f}s)")

    # ----------------------- plots (inline, not saved) -----------------------
    s, e, preds = saved_preds[PLOT_WINDOW]
    hours = np.arange(HORIZON)
    fig, axes = plt.subplots(len(PLOT_DMAS), 1, figsize=(11, 2.6*len(PLOT_DMAS)), sharex=True)
    if len(PLOT_DMAS) == 1: axes = [axes]
    for ax, dma in zip(axes, PLOT_DMAS):
        y = m[dma].iloc[s:e].values.astype(float)
        ax.plot(hours, y,          color="#1b9e77", lw=1.8, label="Actual")
        ax.plot(hours, preds[dma], color="#d95f02", lw=1.8, ls="--", label="LSTM forecast")
        ax.axvspan(0, 24, color="#7570b3", alpha=0.08)      # shade the first day (PI1/PI3)
        ax.set_ylabel(f"{dma}\n(L/s)"); ax.grid(alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Forecast hour (0–168 = one week ahead)")
    fig.suptitle(f"LSTM baseline — forecast vs actual (eval week {PLOT_WINDOW}, "
                 f"origin {m.index[s].date()})", y=1.0, fontsize=11)
    plt.tight_layout(); plt.show()
