import pandas as pd, numpy as np
import sys; sys.path.insert(0, "/home/claude/water_pinn/code")
from eval_utils import load_model, evaluate_forecaster, DMAS

def naive_4week(m, origin_idx, dma, n_weeks=4):
    """Predict each of next 168h as mean of same hour-of-week over prior n_weeks."""
    series = m[dma].values.astype(float)        # clean series
    preds = np.empty(168)
    for h in range(168):
        # positions origin-168*k + h for k=1..n_weeks   (same hour-of-week, prior weeks)
        vals = [series[origin_idx + h - 168*k] for k in range(1, n_weeks+1)
                if origin_idx + h - 168*k >= 0]
        preds[h] = np.nanmean(vals) if vals else np.nan
    return preds

if __name__ == "__main__":
    m = load_model()
    detail, summary = evaluate_forecaster(m, naive_4week, "Naive-4wk")
    print("=== Naive 4-week seasonal baseline ===")
    print("\nMean over 4 windows x 10 DMAs (L/s):")
    print(summary.round(4).to_string())
    print("\nPer-window averages:")
    print(detail.groupby("window")[["PI1","PI2","PI3"]].mean().round(3).to_string())
    print("\nPer-DMA averages:")
    print(detail.groupby("dma")[["PI1","PI2","PI3"]].mean().round(3).to_string())
    detail.to_csv("/home/claude/water_pinn/results/baseline_naive_detail.csv", index=False)
    summary.to_frame().T.to_csv("/home/claude/water_pinn/results/baseline_naive_summary.csv")
    print("\nSAVED -> results/baseline_naive_*.csv")
