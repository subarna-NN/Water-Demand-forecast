import pandas as pd, numpy as np

RAW = "/home/claude/bwdf_raw/"
OUT = "/home/claude/water_pinn/data/"
SHORT_GAP = 3  # hours

def load(name):
    df = pd.read_csv(RAW + name, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True)
    return df

inf  = load("dma-inflows.csv")
wth  = load("weather.csv")
cal  = load("calendar.csv")
prop = pd.read_csv(RAW + "dma-properties.csv", index_col=0)

# --- regular hourly index (fill any missing timestamps too) ---
full_idx = pd.date_range(inf.index.min(), inf.index.max(), freq="h")
inf = inf.reindex(full_idx); wth = wth.reindex(full_idx); cal = cal.reindex(full_idx)

report = []
def log(s): report.append(s); print(s)

log(f"Rows (hourly): {len(full_idx)}  |  {full_idx.min()} -> {full_idx.max()}")

def seasonal_profile_fill(s):
    """Fill NaNs with mean of same (weekday, hour) across available data."""
    idx = s.index
    key = idx.weekday * 24 + idx.hour          # 0..167 hour-of-week
    prof = s.groupby(key).transform("mean")    # aligned seasonal mean
    return s.fillna(prof)

def clean_series(s):
    mask = s.isna()                            # original NaN positions
    # tier 1: short-gap linear interpolation
    s2 = s.interpolate(method="linear", limit=SHORT_GAP, limit_area="inside")
    # tier 2: seasonal hour-of-week profile for the rest
    s2 = seasonal_profile_fill(s2)
    # safety: any residual (e.g. profile itself empty) -> global mean
    s2 = s2.fillna(s2.mean())
    return s2, mask

# --- clean inflows ---
inf_clean = pd.DataFrame(index=full_idx); inf_mask = pd.DataFrame(index=full_idx)
for c in inf.columns:
    inf_clean[c], inf_mask[c + "_imp"] = clean_series(inf[c])
inf_clean = inf_clean.clip(lower=0)            # physics: flow >= 0

# --- clean weather ---
wth_clean = pd.DataFrame(index=full_idx)
for c in wth.columns:
    wth_clean[c], _ = clean_series(wth[c])
wth_clean["Rain"] = wth_clean["Rain"].clip(lower=0)

log("\nImputed % per DMA after cleaning (should be 0 remaining NaN):")
log((inf_clean.isna().mean() * 100).round(3).to_string())
log("\nShare of each DMA that was imputed (%):")
log((inf_mask.mean() * 100).round(2).to_string())

# --- temporal features ---
feat = pd.DataFrame(index=full_idx)
feat["month"]   = full_idx.month
feat["day"]     = full_idx.day
feat["hour"]    = full_idx.hour
feat["weekday"] = full_idx.weekday
# cyclical encodings (better for NN than raw integers)
feat["hour_sin"] = np.sin(2*np.pi*full_idx.hour/24)
feat["hour_cos"] = np.cos(2*np.pi*full_idx.hour/24)
feat["wk_sin"]   = np.sin(2*np.pi*full_idx.weekday/7)
feat["wk_cos"]   = np.cos(2*np.pi*full_idx.weekday/7)
# calendar
feat["holiday"]  = cal["Holiday"].fillna(False).astype(int)
feat["eval_week"]= cal["Evaluation week"].fillna(False).astype(int)
feat["iteration"]= cal["Iteration"].fillna(0).astype(int)

# --- assemble model-ready table ---
model = pd.concat([inf_clean, wth_clean, feat, inf_mask], axis=1)
model.index.name = "Datetime"
model.to_csv(OUT + "model_ready.csv")
inf_clean.to_csv(OUT + "inflows_clean.csv")
prop.to_csv(OUT + "dma_properties.csv")
log(f"\nmodel_ready.csv shape: {model.shape}")
log(f"columns: {list(model.columns)}")

with open(OUT + "clean_report.txt", "w") as f:
    f.write("\n".join(report))
print("\nSAVED ->", OUT + "model_ready.csv")
