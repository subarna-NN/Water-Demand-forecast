import pandas as pd, numpy as np

DMAS = [f"DMA {c}" for c in "ABCDEFGHIJ"]

def load_model():
    m = pd.read_csv("/home/claude/water_pinn/data/model_ready.csv", index_col=0)
    m.index = pd.to_datetime(m.index, utc=True)
    return m

def eval_windows(m):
    """Return list of (start_idx, end_idx) for the 4 official 168h eval weeks."""
    ev = m["eval_week"].values; blocks=[]; start=None
    for i,v in enumerate(ev):
        if v and start is None: start=i
        elif not v and start is not None: blocks.append((start,i)); start=None
    if start is not None: blocks.append((start,len(ev)))
    return blocks

def pi_metrics(y_true, y_pred):
    """y_true,y_pred: arrays of length 168 (one DMA, one week). Returns PI1,PI2,PI3."""
    err = np.abs(np.asarray(y_true) - np.asarray(y_pred))
    pi1 = np.nanmean(err[:24])        # MAE first day
    pi2 = np.nanmean(err[24:168])     # MAE days 2-7
    pi3 = np.nanmax(err[:24])         # worst-hour error, first day
    return pi1, pi2, pi3

def evaluate_forecaster(m, forecast_fn, label, exclude_imputed=True):
    wins = eval_windows(m); rows=[]
    for (s,e) in wins:
        for dma in DMAS:
            y = m[dma].iloc[s:e].values.astype(float).copy()
            if exclude_imputed:
                imp = m[dma + "_imp"].iloc[s:e].values.astype(bool)
                y[imp] = np.nan
            yhat = forecast_fn(m, s, dma)
            pi1,pi2,pi3 = pi_metrics(y, yhat)
            rows.append(dict(window=m.index[s].date(), dma=dma,
                             PI1=pi1, PI2=pi2, PI3=pi3))
    df = pd.DataFrame(rows)
    summary = df[["PI1","PI2","PI3"]].mean()
    summary.name = label
    return df, summary
