import pandas as pd, numpy as np
import xgboost as xgb
import sys; sys.path.insert(0, "/home/claude/water_pinn/code")
from eval_utils import load_model, evaluate_forecaster, DMAS

WEATHER = ["Rain","Temperature","Humidity","Windspeed"]
TEMP    = ["hour_sin","hour_cos","wk_sin","wk_cos","month","holiday"]
LAGW    = [168, 336, 504, 672]   # 1..4 weeks

def build_features(series, m):
    """Return feature DataFrame aligned to m.index for one DMA's series."""
    X = m[WEATHER + TEMP].copy()
    s = pd.Series(series, index=m.index)
    for L in LAGW:
        X[f"lag{L}"] = s.shift(L)
    X["lag_mean"] = pd.concat([s.shift(L) for L in LAGW], axis=1).mean(axis=1)
    X["lag_std"]  = pd.concat([s.shift(L) for L in LAGW], axis=1).std(axis=1)
    return X

def make_xgb_forecaster():
    cache = {}
    def fn(m, origin_idx, dma):
        key = (dma, origin_idx)
        series = m[dma].values.astype(float)
        X = build_features(series, m)
        y = pd.Series(series, index=m.index)
        train_mask = np.zeros(len(m), bool); train_mask[:origin_idx] = True
        # drop rows with NaN lags
        valid = X.notna().all(axis=1).values & train_mask
        Xtr, ytr = X[valid], y[valid]
        model = xgb.XGBRegressor(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            n_jobs=4, objective="reg:squarederror")
        model.fit(Xtr, ytr)
        Xte = X.iloc[origin_idx:origin_idx+168]
        pred = model.predict(Xte)
        return np.clip(pred, 0, None)
    return fn

if __name__ == "__main__":
    m = load_model()
    detail, summary = evaluate_forecaster(m, make_xgb_forecaster(), "XGBoost")
    print("=== XGBoost baseline ===")
    print("\nMean over 4 windows x 10 DMAs (L/s):")
    print(summary.round(4).to_string())
    print("\nPer-window averages:")
    print(detail.groupby("window")[["PI1","PI2","PI3"]].mean().round(3).to_string())
    detail.to_csv("/home/claude/water_pinn/results/baseline_xgb_detail.csv", index=False)
    summary.to_frame().T.to_csv("/home/claude/water_pinn/results/baseline_xgb_summary.csv")
    print("\nSAVED -> results/baseline_xgb_*.csv")
