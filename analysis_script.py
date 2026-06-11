"""
AIS Trajectory Gap Reconstruction — Full Analysis Script
Computes all metrics, win rates, and generates figures for the technical report.

Configurations analyzed:
  - S0/C0: 3B model with 5-bin SOG (STOP/LOW/MOD/HIGH/VHIGH) and 8-dir COG
  - S1/C1: 3B model with 10-bin SOG (<SOG_*>) and 16-point compass COG
  - 1.5B  : 1.5B model with S0/C0 config (gaps 30, 45 only, different test set)

Baselines: LSTM, GRU, BiLSTM
Gap sizes considered: 5, 10, 30, 45 (steps; 1 step = 10 min → 50, 100, 300, 450 min)
"""

import math
import os
import sys
import warnings
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# 0.  PATH SETUP
# ──────────────────────────────────────────────────────────────────────────────

BASE = Path("/home/av15650@ens.ad.etsmtl.ca/Downloads/Musit_instruction_tuning_3b_withOUTmlp+enhance sogcog")
BASE_1B5 = Path("/home/av15650@ens.ad.etsmtl.ca/Downloads/Musit_instruction_tuning (1.5b")

OUT = BASE / "report_outputs"
OUT.mkdir(exist_ok=True)
(OUT / "tables").mkdir(exist_ok=True)
(OUT / "figures").mkdir(exist_ok=True)

GAPS = [5, 10, 30, 45]          # gap sizes to analyze (steps)
GAP_MIN = {5: 50, 10: 100, 30: 300, 45: 450}  # equivalent minutes

print("=== AIS Gap Reconstruction — Analysis Script ===\n")

# ──────────────────────────────────────────────────────────────────────────────
# 1.  METRIC FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl   = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2)**2
    return R * 2 * math.asin(math.sqrt(max(0.0, a)))

def compute_per_sample_metrics(df_pred):
    """
    Given a DataFrame with columns
        sample_id, trajectory_id, gap_steps, gap_step,
        pred_lat, pred_lon, true_lat, true_lon
    return a DataFrame with per-sample ADE, FDE, DTW, MAE, MSE.
    Note: MAE_km == ADE_km (both = mean haversine distance).
    """
    records = []
    for sid, grp in df_pred.groupby("sample_id"):
        grp = grp.sort_values("gap_step")
        preds = list(zip(grp["pred_lat"], grp["pred_lon"]))
        trues = list(zip(grp["true_lat"], grp["true_lon"]))
        n = min(len(preds), len(trues))
        if n == 0:
            continue
        dists = [haversine_km(trues[i][0], trues[i][1], preds[i][0], preds[i][1]) for i in range(n)]
        ade = float(np.mean(dists))
        fde = dists[-1]
        # DTW (same-length sequences → diagonal = sum of pointwise distances)
        dtw = float(np.sum(dists))
        mae = ade  # MAE = mean absolute error = ADE
        mse = float(np.mean([d**2 for d in dists]))
        records.append({
            "sample_id":     sid,
            "trajectory_id": grp["trajectory_id"].iloc[0],
            "gap_steps":     int(grp["gap_steps"].iloc[0]),
            "ADE_km":        ade,
            "FDE_km":        fde,
            "DTW_km":        dtw,
            "MAE_km":        mae,
            "MSE_km2":       mse,
        })
    return pd.DataFrame(records)

# ──────────────────────────────────────────────────────────────────────────────
# 2.  LOAD DATA  — S0/C0 configuration
# ──────────────────────────────────────────────────────────────────────────────

print("Loading S0/C0 data ...")

s0c0_llm = pd.read_csv(BASE / "outputs/resultof_enhanced(sog_cog)/evaluation/test_predictions.csv")
s0c0_llm = s0c0_llm[s0c0_llm["gap_steps"].isin(GAPS)].copy()

s0c0_lstm_raw   = pd.read_csv(BASE / "outputs/baselines/lstm_predictions.csv")
s0c0_gru_raw    = pd.read_csv(BASE / "outputs/baselines/gru_predictions.csv")
s0c0_bilstm_raw = pd.read_csv(BASE / "outputs/baselines/bilstm_predictions.csv")
s0c0_interp_raw = pd.read_csv(BASE / "outputs/baselines/linear_interp_predictions.csv")

for df in [s0c0_lstm_raw, s0c0_gru_raw, s0c0_bilstm_raw, s0c0_interp_raw]:
    df.query("gap_steps in @GAPS", inplace=True)

s0c0_lstm   = compute_per_sample_metrics(s0c0_lstm_raw)
s0c0_gru    = compute_per_sample_metrics(s0c0_gru_raw)
s0c0_bilstm = compute_per_sample_metrics(s0c0_bilstm_raw)
s0c0_interp = compute_per_sample_metrics(s0c0_interp_raw)

# ──────────────────────────────────────────────────────────────────────────────
# 3.  LOAD DATA  — S1/C1 configuration
# ──────────────────────────────────────────────────────────────────────────────

print("Loading S1/C1 data ...")

s1c1_llm = pd.read_csv(BASE / "outputs/S1_C2/evaluation_c1/test_predictions.csv")
s1c1_llm = s1c1_llm[s1c1_llm["gap_steps"].isin(GAPS)].copy()

s1c1_lstm_raw   = pd.read_csv(BASE / "outputs/S1_C2/baselines/lstm_predictions.csv")
s1c1_gru_raw    = pd.read_csv(BASE / "outputs/S1_C2/baselines/gru_predictions.csv")
s1c1_bilstm_raw = pd.read_csv(BASE / "outputs/S1_C2/baselines/bilstm_predictions.csv")
s1c1_interp_raw = pd.read_csv(BASE / "outputs/S1_C2/baselines/linear_interp_predictions.csv")

for df in [s1c1_lstm_raw, s1c1_gru_raw, s1c1_bilstm_raw, s1c1_interp_raw]:
    df.query("gap_steps in @GAPS", inplace=True)

s1c1_lstm   = compute_per_sample_metrics(s1c1_lstm_raw)
s1c1_gru    = compute_per_sample_metrics(s1c1_gru_raw)
s1c1_bilstm = compute_per_sample_metrics(s1c1_bilstm_raw)
s1c1_interp = compute_per_sample_metrics(s1c1_interp_raw)

# ──────────────────────────────────────────────────────────────────────────────
# 4.  LOAD DATA  — 1.5B model (gaps 30, 45 only, different test set)
# ──────────────────────────────────────────────────────────────────────────────

print("Loading 1.5B data ...")

b15_llm_raw  = pd.read_csv(BASE_1B5 / "outputs/evaluation/test_predictions.csv")
b15_llm_raw  = b15_llm_raw[b15_llm_raw["gap_steps"].isin([30, 45])].copy()
b15_llm      = b15_llm_raw.copy()

b15_lstm_raw   = pd.read_csv(BASE_1B5 / "outputs/baselines/lstm_predictions.csv")
b15_gru_raw    = pd.read_csv(BASE_1B5 / "outputs/baselines/gru_predictions.csv")
b15_bilstm_raw = pd.read_csv(BASE_1B5 / "outputs/baselines/bilstm_predictions.csv")

for df in [b15_lstm_raw, b15_gru_raw, b15_bilstm_raw]:
    df.query("gap_steps in [30, 45]", inplace=True)

b15_lstm   = compute_per_sample_metrics(b15_lstm_raw)
b15_gru    = compute_per_sample_metrics(b15_gru_raw)
b15_bilstm = compute_per_sample_metrics(b15_bilstm_raw)

# ──────────────────────────────────────────────────────────────────────────────
# 5.  LOAD GAP METADATA  (for behavioral analysis)
# ──────────────────────────────────────────────────────────────────────────────

print("Loading gap metadata ...")

meta = pd.read_csv(BASE / "processed/gap_metadata.csv")
meta_test = meta[meta["split"] == "test"].copy()
meta_test = meta_test[meta_test["gap_steps"].isin(GAPS)].copy()
print(f"  Test metadata: {len(meta_test)} rows")

# ──────────────────────────────────────────────────────────────────────────────
# 6.  PER-GAP AGGREGATE TABLES
# ──────────────────────────────────────────────────────────────────────────────

print("\n=== PER-GAP AGGREGATE METRICS ===\n")

def agg_table(df_llm, df_lstm, df_gru, df_bilstm, label="S0/C0"):
    rows = []
    for g in GAPS:
        for model_name, df in [
            (f"LLM ({label})", df_llm),
            ("LSTM",   df_lstm),
            ("GRU",    df_gru),
            ("BiLSTM", df_bilstm),
        ]:
            sub = df[df["gap_steps"] == g]
            if len(sub) == 0:
                continue
            ade = sub["ADE_km"].mean()
            fde = sub["FDE_km"].mean()
            dtw = sub["DTW_km"].mean()
            mae = sub["MAE_km"].mean() if "MAE_km" in sub.columns else ade
            mse = sub["MSE_km2"].mean() if "MSE_km2" in sub.columns else float("nan")
            rows.append({
                "Gap": g, "Gap_min": GAP_MIN[g], "Model": model_name,
                "ADE_km":  round(ade, 3),
                "FDE_km":  round(fde, 3),
                "DTW_km":  round(dtw, 3),
                "MAE_km":  round(mae, 3),
                "MSE_km2": round(mse, 3),
                "N": len(sub),
            })
    return pd.DataFrame(rows)

tbl_s0c0 = agg_table(s0c0_llm, s0c0_lstm, s0c0_gru, s0c0_bilstm, "S0/C0")
tbl_s1c1 = agg_table(s1c1_llm, s1c1_lstm, s1c1_gru, s1c1_bilstm, "S1/C1")

print("--- S0/C0 test set ---")
print(tbl_s0c0.to_string(index=False))
print()
print("--- S1/C1 test set ---")
print(tbl_s1c1.to_string(index=False))

tbl_s0c0.to_csv(OUT / "tables/aggregate_s0c0.csv", index=False)
tbl_s1c1.to_csv(OUT / "tables/aggregate_s1c1.csv", index=False)

# ──────────────────────────────────────────────────────────────────────────────
# 7.  WIN-RATE ANALYSIS  (per-sample)
# ──────────────────────────────────────────────────────────────────────────────

print("\n=== WIN-RATE ANALYSIS ===\n")

def win_rate(llm_df, base_df, metric="ADE_km", label_llm="LLM", label_base="LSTM"):
    """
    Returns a DataFrame with per-gap win rates.
    LLM wins when its metric < baseline metric on the same sample.
    """
    records = []
    for g in GAPS:
        llm_g  = llm_df[llm_df["gap_steps"] == g][["sample_id", metric]].rename(columns={metric: "llm"})
        base_g = base_df[base_df["gap_steps"] == g][["sample_id", metric]].rename(columns={metric: "base"})
        if len(llm_g) == 0 or len(base_g) == 0:
            continue
        merged = llm_g.merge(base_g, on="sample_id")
        if len(merged) == 0:
            continue
        wins = (merged["llm"] < merged["base"]).sum()
        total = len(merged)
        records.append({
            "Gap": g,
            "Gap_min": GAP_MIN[g],
            f"LLM wins vs {label_base}": wins,
            "Total": total,
            f"Win% vs {label_base}": round(100 * wins / total, 1),
        })
    return pd.DataFrame(records)

ALL_METRICS = ["ADE_km", "FDE_km", "DTW_km", "MAE_km", "MSE_km2"]

def full_win_analysis(llm_df, lstm_df, gru_df, bilstm_df, cfg_label="S0/C0"):
    """Compute win rates for ADE, FDE, DTW, MAE, MSE for all baselines."""
    results = {}
    for metric in ALL_METRICS:
        if metric not in llm_df.columns:
            continue
        for base_name, base_df in [("LSTM", lstm_df), ("GRU", gru_df), ("BiLSTM", bilstm_df)]:
            if metric not in base_df.columns:
                continue
            key = f"{metric} vs {base_name}"
            df = win_rate(llm_df, base_df, metric=metric, label_llm=f"LLM({cfg_label})", label_base=base_name)
            results[key] = df
    return results

wins_s0c0 = full_win_analysis(s0c0_llm, s0c0_lstm, s0c0_gru, s0c0_bilstm, "S0/C0")
wins_s1c1 = full_win_analysis(s1c1_llm, s1c1_lstm, s1c1_gru, s1c1_bilstm, "S1/C1")

print("--- S0/C0: LLM win rates vs baselines (ADE) ---")
for base in ["LSTM", "GRU", "BiLSTM"]:
    df = wins_s0c0[f"ADE_km vs {base}"]
    print(f"  vs {base}:")
    for _, r in df.iterrows():
        print(f"    Gap {int(r['Gap'])} ({int(r['Gap_min'])} min): {int(r[f'LLM wins vs {base}'])}/{int(r['Total'])} = {r[f'Win% vs {base}']}%")

print()
print("--- S1/C1: LLM win rates vs baselines (ADE) ---")
for base in ["LSTM", "GRU", "BiLSTM"]:
    df = wins_s1c1[f"ADE_km vs {base}"]
    print(f"  vs {base}:")
    for _, r in df.iterrows():
        print(f"    Gap {int(r['Gap'])} ({int(r['Gap_min'])} min): {int(r[f'LLM wins vs {base}'])}/{int(r['Total'])} = {r[f'Win% vs {base}']}%")

# Also FDE win rates
print()
print("--- S0/C0: LLM win rates vs BiLSTM (FDE) ---")
df = wins_s0c0["FDE_km vs BiLSTM"]
for _, r in df.iterrows():
    print(f"  Gap {int(r['Gap'])} ({int(r['Gap_min'])} min): {r['Win% vs BiLSTM']}%")

# Save win rate tables (all metrics)
for cfg, wins in [("s0c0", wins_s0c0), ("s1c1", wins_s1c1)]:
    for key, df in wins.items():
        fname = key.replace(" ", "_").replace("/", "_").replace("2", "2")
        df.to_csv(OUT / f"tables/winrate_{cfg}_{fname}.csv", index=False)

# ──────────────────────────────────────────────────────────────────────────────
# 8.  S0/C0 vs S1/C1 COMPARISON  (same test samples)
# ──────────────────────────────────────────────────────────────────────────────

print("\n=== S0/C0 vs S1/C1 COMPARISON ===\n")

def cfg_win_rate(df1, df2, metric="ADE_km", label1="S0/C0", label2="S1/C1"):
    """S1/C1 wins when its metric < S0/C0 metric."""
    records = []
    for g in GAPS:
        d1 = df1[df1["gap_steps"] == g][["sample_id", metric]].rename(columns={metric: "cfg1"})
        d2 = df2[df2["gap_steps"] == g][["sample_id", metric]].rename(columns={metric: "cfg2"})
        merged = d1.merge(d2, on="sample_id")
        if len(merged) == 0:
            continue
        wins2 = (merged["cfg2"] < merged["cfg1"]).sum()
        total = len(merged)
        records.append({
            "Gap": g, "Gap_min": GAP_MIN[g],
            f"{label2} wins": wins2,
            "Total": total,
            f"{label2} Win%": round(100 * wins2 / total, 1),
        })
    return pd.DataFrame(records)

for metric in ALL_METRICS:
    if metric not in s0c0_llm.columns or metric not in s1c1_llm.columns:
        continue
    df = cfg_win_rate(s0c0_llm, s1c1_llm, metric=metric)
    print(f"S1/C1 better than S0/C0 ({metric}):")
    print(df.to_string(index=False))
    print()
    df.to_csv(OUT / f"tables/cfg_comparison_{metric}.csv", index=False)

# ──────────────────────────────────────────────────────────────────────────────
# 9.  1.5B MODEL ANALYSIS  (gaps 30, 45 only)
# ──────────────────────────────────────────────────────────────────────────────

print("\n=== 1.5B MODEL ANALYSIS (gaps 30, 45 only, separate test set) ===\n")

b15_gaps = [30, 45]

for g in b15_gaps:
    llm_g  = b15_llm[b15_llm["gap_steps"] == g]
    lstm_g = b15_lstm[b15_lstm["gap_steps"] == g]
    gru_g  = b15_gru[b15_gru["gap_steps"] == g]
    bi_g   = b15_bilstm[b15_bilstm["gap_steps"] == g]
    print(f"Gap {g} ({GAP_MIN[g]} min), N={len(llm_g)} (1.5B test set):")
    for name, df in [("LLM (1.5B)", llm_g), ("LSTM", lstm_g), ("GRU", gru_g), ("BiLSTM", bi_g)]:
        if len(df) > 0:
            print(f"  {name}: ADE={df['ADE_km'].mean():.3f}  FDE={df['FDE_km'].mean():.3f}  DTW={df['DTW_km'].mean():.3f}")
    print()

# Win rates for 1.5B
print("1.5B LLM win rates vs baselines (ADE_km):")
for g in b15_gaps:
    llm_g = b15_llm[b15_llm["gap_steps"] == g][["sample_id", "ADE_km"]].rename(columns={"ADE_km": "llm"})
    for base_name, base_df in [("LSTM", b15_lstm), ("GRU", b15_gru), ("BiLSTM", b15_bilstm)]:
        base_g = base_df[base_df["gap_steps"] == g][["sample_id", "ADE_km"]].rename(columns={"ADE_km": "base"})
        merged = llm_g.merge(base_g, on="sample_id")
        if len(merged) == 0:
            print(f"  Gap {g} vs {base_name}: no matching samples")
            continue
        wins = (merged["llm"] < merged["base"]).sum()
        pct = 100 * wins / len(merged)
        print(f"  Gap {g} ({GAP_MIN[g]} min) vs {base_name}: {wins}/{len(merged)} = {pct:.1f}%")

# ──────────────────────────────────────────────────────────────────────────────
# 10. BEHAVIORAL ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

print("\n=== BEHAVIORAL ANALYSIS ===\n")

# 10a. Gap duration analysis
print("Gap duration statistics (from gap_metadata):")
for g in GAPS:
    sub = meta_test[meta_test["gap_steps"] == g]
    print(f"  Gap {g}: observable_gap_min={sub['observable_gap_duration_min'].mean():.1f} "
          f"(min={sub['observable_gap_duration_min'].min():.1f}, "
          f"max={sub['observable_gap_duration_min'].max():.1f})")

# 10b. Mean dt analysis
print("\nMean inter-observation interval (mean_dt_inside_window_min):")
for g in GAPS:
    sub = meta_test[meta_test["gap_steps"] == g]
    print(f"  Gap {g}: mean_dt={sub['mean_dt_inside_window_min'].mean():.2f} min "
          f"(std={sub['mean_dt_inside_window_min'].std():.2f})")

# 10c.  Join LLM per-sample metrics with gap metadata for behavioral analysis
# Use S1/C1 (best 3B config) as reference
llm_meta = s1c1_llm.merge(
    meta_test[["sample_id", "gap_steps", "observable_gap_duration_min",
               "mean_dt_inside_window_min", "hidden_internal_duration_min",
               "max_dt_inside_window_min"]],
    on=["sample_id", "gap_steps"], how="inner"
)

# S1/C1 BiLSTM per-sample
bilstm_meta = s1c1_bilstm.merge(
    meta_test[["sample_id", "gap_steps", "observable_gap_duration_min",
               "mean_dt_inside_window_min"]],
    on=["sample_id", "gap_steps"], how="inner"
)

# Merge LLM and BiLSTM on same samples
compare_meta = llm_meta.merge(
    bilstm_meta[["sample_id", "gap_steps", "ADE_km", "FDE_km"]].rename(
        columns={"ADE_km": "bilstm_ADE", "FDE_km": "bilstm_FDE"}),
    on=["sample_id", "gap_steps"]
)
compare_meta["llm_wins_ade"] = compare_meta["ADE_km"] < compare_meta["bilstm_ADE"]

print(f"\nSamples in LLM-BiLSTM comparison (S1/C1): {len(compare_meta)}")

# Stationary analysis: use observable gap duration
# Approximate "stationary" as small total displacement = large gap but short actual distance
# Use mean_dt as proxy for speed (shorter dt = more frequent = possibly moving faster)
compare_meta["fast_sampling"] = compare_meta["mean_dt_inside_window_min"] < 10.0

print("\nLLM (S1/C1) win rate vs BiLSTM by gap and sampling regime (ADE):")
for g in GAPS:
    sub = compare_meta[compare_meta["gap_steps"] == g]
    fast = sub[sub["fast_sampling"]]
    slow = sub[~sub["fast_sampling"]]
    for lbl, ss in [("all", sub), ("fast_sampling (<10 min dt)", fast), ("slow_sampling (>=10 min dt)", slow)]:
        if len(ss) == 0:
            continue
        wr = ss["llm_wins_ade"].mean() * 100
        print(f"  Gap {g} | {lbl}: LLM wins {wr:.1f}% ({ss['llm_wins_ade'].sum()}/{len(ss)})")

# Observable gap duration vs LLM win rate
print("\nLLM (S1/C1) win rate vs BiLSTM by observable gap duration quartile:")
compare_meta["obs_dur_quartile"] = pd.qcut(
    compare_meta["observable_gap_duration_min"], q=4, labels=["Q1 (short)", "Q2", "Q3", "Q4 (long)"]
)
tbl_dur = compare_meta.groupby(["gap_steps", "obs_dur_quartile"])["llm_wins_ade"].agg(["mean", "count"])
tbl_dur["win_pct"] = (tbl_dur["mean"] * 100).round(1)
print(tbl_dur[["win_pct", "count"]].to_string())

# Save behavioral summary
compare_meta.to_csv(OUT / "tables/behavioral_comparison_s1c1_vs_bilstm.csv", index=False)

# ──────────────────────────────────────────────────────────────────────────────
# 11. COMPREHENSIVE SUMMARY TABLE
# ──────────────────────────────────────────────────────────────────────────────

print("\n=== COMPREHENSIVE SUMMARY TABLES ===\n")

# Build one table per gap with all models (S0/C0 and S1/C1 test sets)
summary_rows = []
for g in GAPS:
    for cfg, llm_df, lstm_df, gru_df, bi_df in [
        ("S0/C0", s0c0_llm, s0c0_lstm, s0c0_gru, s0c0_bilstm),
        ("S1/C1", s1c1_llm, s1c1_lstm, s1c1_gru, s1c1_bilstm),
    ]:
        for mname, mdf in [
            (f"LLM ({cfg})", llm_df),
            ("LSTM", lstm_df),
            ("GRU", gru_df),
            ("BiLSTM", bi_df),
        ]:
            sub = mdf[mdf["gap_steps"] == g]
            if len(sub) == 0:
                continue
            summary_rows.append({
                "Gap_steps": g,
                "Gap_min": GAP_MIN[g],
                "Config": cfg,
                "Model": mname,
                "N": len(sub),
                "ADE_km":  round(sub["ADE_km"].mean(), 3),
                "FDE_km":  round(sub["FDE_km"].mean(), 3),
                "DTW_km":  round(sub["DTW_km"].mean(), 3),
                "MAE_km":  round(sub["MAE_km"].mean(), 3) if "MAE_km" in sub.columns else round(sub["ADE_km"].mean(), 3),
                "MSE_km2": round(sub["MSE_km2"].mean(), 3) if "MSE_km2" in sub.columns else float("nan"),
            })

summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False))
summary_df.to_csv(OUT / "tables/full_summary.csv", index=False)

# ──────────────────────────────────────────────────────────────────────────────
# 12. FIGURE 1 — Per-gap ADE comparison (S0/C0 test set)
# ──────────────────────────────────────────────────────────────────────────────

print("\nGenerating figures ...")

colors = {
    "LLM (S0/C0)": "#1f77b4",
    "LLM (S1/C1)": "#ff7f0e",
    "LSTM":         "#2ca02c",
    "GRU":          "#d62728",
    "BiLSTM":       "#9467bd",
    "LLM (1.5B)":  "#8c564b",
}

def make_metric_plot(tbl, metric, title, fname, highlight_best=True):
    fig, axes = plt.subplots(1, len(GAPS), figsize=(16, 5), sharey=False)
    for ax, g in zip(axes, GAPS):
        sub = tbl[tbl["Gap"] == g].copy()
        models = sub["Model"].tolist()
        vals   = sub[metric].tolist()
        cs = [colors.get(m, "#7f7f7f") for m in models]
        bars = ax.bar(range(len(models)), vals, color=cs, edgecolor="black", linewidth=0.5)
        if highlight_best:
            best_idx = int(np.argmin(vals))
            bars[best_idx].set_edgecolor("gold")
            bars[best_idx].set_linewidth(2.5)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=30, ha="right", fontsize=8)
        ax.set_title(f"Gap {g} ({GAP_MIN[g]} min)", fontsize=10)
        ax.set_ylabel(metric if g == GAPS[0] else "")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + max(vals)*0.01, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=7)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT / f"figures/{fname}", dpi=150, bbox_inches="tight")
    plt.close()

# ADE
make_metric_plot(tbl_s0c0, "ADE_km",  "ADE per Gap — S0/C0 Test Set",  "fig1_s0c0_ade.png")
make_metric_plot(tbl_s1c1, "ADE_km",  "ADE per Gap — S1/C1 Test Set",  "fig2_s1c1_ade.png")
# FDE
make_metric_plot(tbl_s0c0, "FDE_km",  "FDE per Gap — S0/C0 Test Set",  "fig3_s0c0_fde.png")
make_metric_plot(tbl_s1c1, "FDE_km",  "FDE per Gap — S1/C1 Test Set",  "fig4_s1c1_fde.png")
# DTW
make_metric_plot(tbl_s0c0, "DTW_km",  "DTW per Gap — S0/C0 Test Set",  "fig_s0c0_dtw.png")
make_metric_plot(tbl_s1c1, "DTW_km",  "DTW per Gap — S1/C1 Test Set",  "fig_s1c1_dtw.png")
# MAE (= ADE; identical values, separate figure for completeness)
make_metric_plot(tbl_s0c0, "MAE_km",  "MAE per Gap — S0/C0 Test Set",  "fig_s0c0_mae.png")
make_metric_plot(tbl_s1c1, "MAE_km",  "MAE per Gap — S1/C1 Test Set",  "fig_s1c1_mae.png")
# MSE
make_metric_plot(tbl_s0c0, "MSE_km2", "MSE per Gap — S0/C0 Test Set",  "fig_s0c0_mse.png")
make_metric_plot(tbl_s1c1, "MSE_km2", "MSE per Gap — S1/C1 Test Set",  "fig_s1c1_mse.png")

# ──────────────────────────────────────────────────────────────────────────────
# 13. FIGURE 2 — Win-rate heatmaps
# ──────────────────────────────────────────────────────────────────────────────

def win_rate_heatmap(wins_dict, cfg_label, metric="ADE_km", fname="winrate_heatmap.png"):
    baselines = ["LSTM", "GRU", "BiLSTM"]
    data = np.zeros((len(GAPS), len(baselines)))
    for j, base in enumerate(baselines):
        key = f"{metric} vs {base}"
        df = wins_dict.get(key, pd.DataFrame())
        for i, g in enumerate(GAPS):
            row = df[df["Gap"] == g]
            if len(row) > 0:
                data[i, j] = row.iloc[0][f"Win% vs {base}"]
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(baselines)))
    ax.set_xticklabels(baselines, fontsize=11)
    ax.set_yticks(range(len(GAPS)))
    ax.set_yticklabels([f"Gap {g} ({GAP_MIN[g]} min)" for g in GAPS], fontsize=10)
    for i in range(len(GAPS)):
        for j in range(len(baselines)):
            ax.text(j, i, f"{data[i,j]:.0f}%", ha="center", va="center",
                    fontsize=11, color="black", fontweight="bold")
    plt.colorbar(im, ax=ax, label="LLM Win Rate (%)")
    ax.set_title(f"LLM ({cfg_label}) Win Rate vs Baselines — {metric}", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT / f"figures/{fname}", dpi=150, bbox_inches="tight")
    plt.close()

win_rate_heatmap(wins_s0c0, "S0/C0", "ADE_km",  "fig5_winrate_s0c0_ade.png")
win_rate_heatmap(wins_s1c1, "S1/C1", "ADE_km",  "fig6_winrate_s1c1_ade.png")
win_rate_heatmap(wins_s0c0, "S0/C0", "FDE_km",  "fig7_winrate_s0c0_fde.png")
win_rate_heatmap(wins_s1c1, "S1/C1", "FDE_km",  "fig8_winrate_s1c1_fde.png")
win_rate_heatmap(wins_s0c0, "S0/C0", "DTW_km",  "fig_winrate_s0c0_dtw.png")
win_rate_heatmap(wins_s1c1, "S1/C1", "DTW_km",  "fig_winrate_s1c1_dtw.png")
win_rate_heatmap(wins_s0c0, "S0/C0", "MAE_km",  "fig_winrate_s0c0_mae.png")
win_rate_heatmap(wins_s1c1, "S1/C1", "MAE_km",  "fig_winrate_s1c1_mae.png")
win_rate_heatmap(wins_s0c0, "S0/C0", "MSE_km2", "fig_winrate_s0c0_mse.png")
win_rate_heatmap(wins_s1c1, "S1/C1", "MSE_km2", "fig_winrate_s1c1_mse.png")

# ──────────────────────────────────────────────────────────────────────────────
# 14. FIGURE 3 — ADE distribution per gap (boxplot)
# ──────────────────────────────────────────────────────────────────────────────

def boxplot_metric(llm_df, lstm_df, gru_df, bilstm_df, cfg_label, metric, ylabel, fname):
    fig, axes = plt.subplots(1, len(GAPS), figsize=(18, 5), sharey=False)
    model_names = [f"LLM\n({cfg_label})", "LSTM", "GRU", "BiLSTM"]
    for ax, g in zip(axes, GAPS):
        data_per_model = []
        for mdf in [llm_df, lstm_df, gru_df, bilstm_df]:
            col = metric if metric in mdf.columns else "ADE_km"
            sub = mdf[mdf["gap_steps"] == g][col].values
            data_per_model.append(sub)
        bp = ax.boxplot(data_per_model, labels=model_names, patch_artist=True,
                        medianprops=dict(color="black", linewidth=2))
        cs = [colors[f"LLM ({cfg_label})"], colors["LSTM"], colors["GRU"], colors["BiLSTM"]]
        for patch, c in zip(bp["boxes"], cs):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        ax.set_title(f"Gap {g} ({GAP_MIN[g]} min)", fontsize=10)
        ax.set_ylabel(ylabel if g == GAPS[0] else "")
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle(f"{ylabel} Distribution — {cfg_label} Test Set", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT / f"figures/{fname}", dpi=150, bbox_inches="tight")
    plt.close()

# ADE boxplots
boxplot_metric(s0c0_llm, s0c0_lstm, s0c0_gru, s0c0_bilstm, "S0/C0", "ADE_km",  "ADE (km)",   "fig9_boxplot_s0c0.png")
boxplot_metric(s1c1_llm, s1c1_lstm, s1c1_gru, s1c1_bilstm, "S1/C1", "ADE_km",  "ADE (km)",   "fig10_boxplot_s1c1.png")
# FDE boxplots
boxplot_metric(s0c0_llm, s0c0_lstm, s0c0_gru, s0c0_bilstm, "S0/C0", "FDE_km",  "FDE (km)",   "fig_boxplot_s0c0_fde.png")
boxplot_metric(s1c1_llm, s1c1_lstm, s1c1_gru, s1c1_bilstm, "S1/C1", "FDE_km",  "FDE (km)",   "fig_boxplot_s1c1_fde.png")
# DTW boxplots
boxplot_metric(s0c0_llm, s0c0_lstm, s0c0_gru, s0c0_bilstm, "S0/C0", "DTW_km",  "DTW (km)",   "fig_boxplot_s0c0_dtw.png")
boxplot_metric(s1c1_llm, s1c1_lstm, s1c1_gru, s1c1_bilstm, "S1/C1", "DTW_km",  "DTW (km)",   "fig_boxplot_s1c1_dtw.png")
# MAE boxplots
boxplot_metric(s0c0_llm, s0c0_lstm, s0c0_gru, s0c0_bilstm, "S0/C0", "MAE_km",  "MAE (km)",   "fig_boxplot_s0c0_mae.png")
boxplot_metric(s1c1_llm, s1c1_lstm, s1c1_gru, s1c1_bilstm, "S1/C1", "MAE_km",  "MAE (km)",   "fig_boxplot_s1c1_mae.png")
# MSE boxplots
boxplot_metric(s0c0_llm, s0c0_lstm, s0c0_gru, s0c0_bilstm, "S0/C0", "MSE_km2", "MSE (km²)",  "fig_boxplot_s0c0_mse.png")
boxplot_metric(s1c1_llm, s1c1_lstm, s1c1_gru, s1c1_bilstm, "S1/C1", "MSE_km2", "MSE (km²)",  "fig_boxplot_s1c1_mse.png")

# ──────────────────────────────────────────────────────────────────────────────
# 15. FIGURE 4 — S0/C0 vs S1/C1 LLM direct comparison
# ──────────────────────────────────────────────────────────────────────────────

for metric in ["ADE_km", "FDE_km", "DTW_km"]:
    fig, axes = plt.subplots(1, len(GAPS), figsize=(14, 4))
    for ax, g in zip(axes, GAPS):
        s0_vals = s0c0_llm[s0c0_llm["gap_steps"] == g]["ADE_km"].values if metric == "ADE_km" else \
                  s0c0_llm[s0c0_llm["gap_steps"] == g][metric].values
        s1_vals = s1c1_llm[s1c1_llm["gap_steps"] == g]["ADE_km"].values if metric == "ADE_km" else \
                  s1c1_llm[s1c1_llm["gap_steps"] == g][metric].values
        ax.scatter(s0_vals, s1_vals, alpha=0.5, s=25, color="#1f77b4")
        lim = max(max(s0_vals), max(s1_vals)) * 1.05
        ax.plot([0, lim], [0, lim], "k--", linewidth=0.8, alpha=0.6)
        ax.set_xlabel(f"S0/C0 {metric}")
        ax.set_ylabel(f"S1/C1 {metric}" if g == GAPS[0] else "")
        ax.set_title(f"Gap {g} ({GAP_MIN[g]} min)")
    fig.suptitle(f"S0/C0 vs S1/C1 LLM — {metric}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fname = f"fig_s0c0_vs_s1c1_{metric.replace('_km','').replace('_km2','')}.png"
    plt.savefig(OUT / f"figures/{fname}", dpi=150, bbox_inches="tight")
    plt.close()

# ──────────────────────────────────────────────────────────────────────────────
# 16. FIGURE 5 — Trajectory visualizations (representative cases)
# ──────────────────────────────────────────────────────────────────────────────

print("Generating trajectory visualizations ...")

# Load per-step LLM predictions for trajectory plots
s0c0_llm_steps = pd.read_csv(BASE / "outputs/baselines/lstm_predictions.csv")
# Use LSTM predictions file structure to get true lat/lon per step
# The LLM predictions are per-sample (no step-level lat/lon in test_predictions.csv)
# Instead, use baseline file for ground truth and derive LLM from geohash decode
# (requires geohash library - use raw step predictions from baseline as proxy for true)

# For trajectory visualization, use S0/C0 LSTM step predictions (has true_lat, true_lon)
# and S0/C0 baseline predictions as comparison
# Note: LLM per-step lat/lon would need the llm_per_step_predictions.csv

llm_step_file = BASE / "outputs/S1_C2/evaluation/llm_per_step_predictions.csv"
if llm_step_file.exists():
    llm_steps = pd.read_csv(llm_step_file)
    llm_steps = llm_steps[llm_steps["gap_steps"].isin(GAPS)].copy()
    print(f"  Loaded LLM per-step predictions: {len(llm_steps)} rows")

    # Pick representative samples for each gap size
    def plot_trajectory(g, sample_idx=0, fname_suffix=""):
        fig, axes = plt.subplots(1, 1, figsize=(7, 5))
        ax = axes

        gap_samples = llm_steps[llm_steps["gap_steps"] == g]["sample_id"].unique()
        if len(gap_samples) == 0:
            return
        sid = gap_samples[sample_idx % len(gap_samples)]

        llm_sub = llm_steps[(llm_steps["sample_id"] == sid) & (llm_steps["gap_steps"] == g)]
        lstm_sub = s0c0_lstm_raw[(s0c0_lstm_raw["sample_id"] == sid) & (s0c0_lstm_raw["gap_steps"] == g)]
        bilstm_sub = s0c0_bilstm_raw[(s0c0_bilstm_raw["sample_id"] == sid) & (s0c0_bilstm_raw["gap_steps"] == g)]

        if len(llm_sub) == 0 or len(lstm_sub) == 0:
            return

        true_lats = llm_sub.sort_values("gap_step")["true_lat"].values
        true_lons = llm_sub.sort_values("gap_step")["true_lon"].values
        llm_lats  = llm_sub.sort_values("gap_step")["pred_lat"].values
        llm_lons  = llm_sub.sort_values("gap_step")["pred_lon"].values
        lstm_lats  = lstm_sub.sort_values("gap_step")["pred_lat"].values
        lstm_lons  = lstm_sub.sort_values("gap_step")["pred_lon"].values

        ax.plot(true_lons, true_lats, "ko-", linewidth=2, markersize=6, label="Ground Truth", zorder=5)
        ax.plot(llm_lons, llm_lats, "b^--", linewidth=1.5, markersize=5, label="LLM (S1/C1)", zorder=4)
        ax.plot(lstm_lons, lstm_lats, "rv--", linewidth=1.5, markersize=5, label="LSTM", zorder=3)

        if len(bilstm_sub) > 0:
            bi_lats = bilstm_sub.sort_values("gap_step")["pred_lat"].values
            bi_lons = bilstm_sub.sort_values("gap_step")["pred_lon"].values
            ax.plot(bi_lons, bi_lats, "ms--", linewidth=1.5, markersize=5, label="BiLSTM", zorder=3)

        ax.scatter([true_lons[0]], [true_lats[0]], c="green", s=100, zorder=6, marker="*", label="Start")
        ax.scatter([true_lons[-1]], [true_lats[-1]], c="red", s=100, zorder=6, marker="*", label="End")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"Trajectory: {sid}\nGap {g} ({GAP_MIN[g]} min)")
        ax.legend(fontsize=8, loc="best")
        plt.tight_layout()
        plt.savefig(OUT / f"figures/traj_gap{g}_{fname_suffix}.png", dpi=150, bbox_inches="tight")
        plt.close()

    # Generate a few trajectory plots per gap
    for g in GAPS:
        for idx in range(min(3, 5)):
            plot_trajectory(g, sample_idx=idx, fname_suffix=f"sample{idx}")
else:
    print("  llm_per_step_predictions.csv not found at expected path — skipping step-level trajectory plots")

# ──────────────────────────────────────────────────────────────────────────────
# 17. FIGURE 6 — win rate vs gap size (line chart)
# ──────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (cfg, wins) in zip(axes, [("S0/C0", wins_s0c0), ("S1/C1", wins_s1c1)]):
    for base in ["LSTM", "GRU", "BiLSTM"]:
        df = wins[f"ADE_km vs {base}"]
        ax.plot(df["Gap_min"], df[f"Win% vs {base}"], "o-", label=f"vs {base}",
                linewidth=2, markersize=8)
    ax.axhline(50, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xlabel("Gap Duration (minutes)")
    ax.set_ylabel("LLM Win Rate (%)")
    ax.set_title(f"LLM ({cfg}) Win Rate vs Baselines (ADE)")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "figures/fig_winrate_vs_gap.png", dpi=150, bbox_inches="tight")
plt.close()

# ──────────────────────────────────────────────────────────────────────────────
# 18. TOKEN ACCURACY SUMMARY
# ──────────────────────────────────────────────────────────────────────────────

print("\n=== TOKEN ACCURACY SUMMARY ===\n")

for cfg, llm_df in [("S0/C0", s0c0_llm), ("S1/C1", s1c1_llm)]:
    print(f"--- {cfg} ---")
    for g in GAPS:
        sub = llm_df[llm_df["gap_steps"] == g]
        if "exact_token_accuracy" in sub.columns and len(sub) > 0:
            ta  = sub["exact_token_accuracy"].mean()
            sem = sub["sequence_exact_match"].mean() if "sequence_exact_match" in sub.columns else float("nan")
            print(f"  Gap {g}: token_acc={ta:.3f}, seq_exact={sem:.3f}, N={len(sub)}")

# ──────────────────────────────────────────────────────────────────────────────
# 19. EXPORT COMBINED WIN-RATE TABLE (report-ready)
# ──────────────────────────────────────────────────────────────────────────────

print("\nBuilding report-ready win-rate table ...")

wr_rows = []
for cfg, llm_df, lstm_df, gru_df, bi_df in [
    ("S0/C0", s0c0_llm, s0c0_lstm, s0c0_gru, s0c0_bilstm),
    ("S1/C1", s1c1_llm, s1c1_lstm, s1c1_gru, s1c1_bilstm),
]:
    for g in GAPS:
        llm_g = llm_df[llm_df["gap_steps"] == g]
        for base_name, base_df in [("LSTM", lstm_df), ("GRU", gru_df), ("BiLSTM", bi_df)]:
            for metric in ALL_METRICS:
                if metric not in llm_g.columns or metric not in base_df.columns:
                    continue
                llm_vals = llm_g[["sample_id", metric]].rename(columns={metric: "llm"})
                base_g   = base_df[base_df["gap_steps"] == g][["sample_id", metric]].rename(columns={metric: "base"})
                merged   = llm_vals.merge(base_g, on="sample_id")
                if len(merged) == 0:
                    continue
                wins = (merged["llm"] < merged["base"]).sum()
                wr_rows.append({
                    "Config": cfg, "Gap": g, "Gap_min": GAP_MIN[g],
                    "Baseline": base_name, "Metric": metric,
                    "LLM_wins": int(wins), "Total": len(merged),
                    "WinRate_%": round(100 * wins / len(merged), 1),
                })

wr_table = pd.DataFrame(wr_rows)
wr_table.to_csv(OUT / "tables/combined_winrates.csv", index=False)
print("Combined win-rate table saved.")
print(wr_table.to_string(index=False))

print(f"\n=== DONE ===\nAll outputs saved to {OUT}")
