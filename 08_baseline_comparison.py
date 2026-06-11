

import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.distance_utils import (
    compute_ade_km, compute_fde_km, compute_dtw_km,
    compute_mae_km, compute_mse_km2,
)
from src.utils.io_utils import load_config, setup_logging

logger = logging.getLogger(__name__)

Coords = List[Tuple[float, float]]

# ── Neural hyperparameters ────────────────────────────────────────────────────
INPUT_DIM  = 7    # lat_n, lon_n, sog_n, cog_sin, cog_cos, dt_n, observed_flag
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT    = 0.2
BATCH_SIZE = 64
MAX_EPOCHS = 100
PATIENCE   = 10
LR         = 1e-3
PREFIX_LEN = 60
SUFFIX_LEN = 60


# ═══════════════════════════════════════════════════════════════════════════════
# Feature engineering
# ═══════════════════════════════════════════════════════════════════════════════

def compute_norm_stats(train_df: pd.DataFrame) -> Dict:
    """Compute normalization stats from training prefix+suffix rows only."""
    obs = train_df[train_df["row_type"].isin(["prefix", "suffix"])]
    stats: Dict = {}
    for col in ["latitude", "longitude", "SOG", "dt_min"]:
        stats[f"{col}_mean"] = float(obs[col].mean())
        stats[f"{col}_std"]  = float(obs[col].std()) + 1e-8
    return stats


def build_feature_matrix(rows: pd.DataFrame, stats: Dict, observed: bool) -> np.ndarray:
    """
    Build (n_rows, 7) float32 feature matrix.
    observed=False → lat/lon/SOG/COG zeroed (gap masking, no leakage).
    """
    n = len(rows)
    if observed:
        lat_n   = (rows["latitude"].values  - stats["latitude_mean"])  / stats["latitude_std"]
        lon_n   = (rows["longitude"].values - stats["longitude_mean"]) / stats["longitude_std"]
        sog_n   = (rows["SOG"].values       - stats["SOG_mean"])       / stats["SOG_std"]
        cog_rad = np.radians(rows["COG"].values.astype(float) % 360.0)
        cog_sin = np.sin(cog_rad)
        cog_cos = np.cos(cog_rad)
    else:
        lat_n = lon_n = sog_n = cog_sin = cog_cos = np.zeros(n, dtype=np.float32)

    dt_n = (rows["dt_min"].values - stats["dt_min_mean"]) / stats["dt_min_std"]
    flag = np.ones(n, dtype=np.float32) if observed else np.zeros(n, dtype=np.float32)

    return np.stack([lat_n, lon_n, sog_n, cog_sin, cog_cos, dt_n, flag], axis=1).astype(np.float32)


def denorm_latlon(arr: np.ndarray, stats: Dict) -> np.ndarray:
    """Denormalize (N, 2) array [lat_n, lon_n] → degrees."""
    lat = arr[:, 0] * stats["latitude_std"]  + stats["latitude_mean"]
    lon = arr[:, 1] * stats["longitude_std"] + stats["longitude_mean"]
    return np.stack([lat, lon], axis=1)


# ═══════════════════════════════════════════════════════════════════════════════
# Data preparation
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_samples(df: pd.DataFrame, gap_steps: int) -> List[Dict]:
    """Return list of dicts {sample_id, trajectory_id, gap_steps, prefix, gap, suffix}."""
    samples: List[Dict] = []
    sub = df[df["gap_steps"] == gap_steps].sort_values(["sample_id", "global_row_index"])
    for sid, grp in sub.groupby("sample_id", sort=False):
        grp    = grp.reset_index(drop=True)
        prefix = grp[grp["row_type"] == "prefix"].reset_index(drop=True)
        gap    = grp[grp["row_type"] == "gap"].reset_index(drop=True)
        suffix = grp[grp["row_type"] == "suffix"].reset_index(drop=True)
        if len(prefix) != PREFIX_LEN or len(gap) != gap_steps or len(suffix) != SUFFIX_LEN:
            logger.warning("Skipping malformed sample: %s (prefix=%d gap=%d suffix=%d)",
                           sid, len(prefix), len(gap), len(suffix))
            continue
        samples.append({
            "sample_id":     sid,
            "trajectory_id": grp["trajectory_id"].iloc[0],
            "gap_steps":     gap_steps,
            "prefix":        prefix,
            "gap":           gap,
            "suffix":        suffix,
        })
    return samples


# ═══════════════════════════════════════════════════════════════════════════════
# Rule-based baselines
# ═══════════════════════════════════════════════════════════════════════════════

def predict_last_known(sample: Dict) -> Coords:
    """Repeat last prefix position for every gap step."""
    last_lat = float(sample["prefix"]["latitude"].iloc[-1])
    last_lon = float(sample["prefix"]["longitude"].iloc[-1])
    return [(last_lat, last_lon)] * sample["gap_steps"]


def predict_linear_interp(sample: Dict) -> Coords:
    """Linearly interpolate between last prefix and first suffix position."""
    p0_lat = float(sample["prefix"]["latitude"].iloc[-1])
    p0_lon = float(sample["prefix"]["longitude"].iloc[-1])
    p1_lat = float(sample["suffix"]["latitude"].iloc[0])
    p1_lon = float(sample["suffix"]["longitude"].iloc[0])
    n = sample["gap_steps"]
    return [
        (p0_lat + (p1_lat - p0_lat) * (i + 1) / (n + 1),
         p0_lon + (p1_lon - p0_lon) * (i + 1) / (n + 1))
        for i in range(n)
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# Neural models
# ═══════════════════════════════════════════════════════════════════════════════

class LSTMForecaster(nn.Module):
    """
    Encodes prefix with LSTM/GRU, projects final hidden state to all gap positions at once.
    One model is trained per gap size to avoid padding.
    """
    def __init__(self, gap_steps: int, rnn_type: str = "lstm"):
        super().__init__()
        self.gap_steps = gap_steps
        rnn_cls        = nn.LSTM if rnn_type == "lstm" else nn.GRU
        self.encoder   = rnn_cls(
            INPUT_DIM, HIDDEN_DIM, NUM_LAYERS,
            batch_first=True,
            dropout=DROPOUT if NUM_LAYERS > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, gap_steps * 2),
        )
        self._is_lstm = rnn_type == "lstm"

    def forward(self, prefix_seq: torch.Tensor) -> torch.Tensor:
        # prefix_seq: (B, PREFIX_LEN, INPUT_DIM)
        _, state = self.encoder(prefix_seq)
        h = state[0] if self._is_lstm else state   # (num_layers, B, H)
        h = h[-1]                                   # last layer: (B, H)
        return self.head(h).view(-1, self.gap_steps, 2)  # (B, gap_steps, 2)


class BiLSTMInterpolator(nn.Module):
    """
    Bidirectional LSTM over the full window (prefix | masked-gap | suffix).
    Hidden states at gap positions are projected to lat/lon predictions.
    Gap rows are fed with lat/lon/SOG/COG zeroed (observed_flag=0): no leakage.
    """
    def __init__(self, gap_steps: int):
        super().__init__()
        self.gap_steps = gap_steps
        self.bilstm    = nn.LSTM(
            INPUT_DIM, HIDDEN_DIM, NUM_LAYERS,
            batch_first=True, bidirectional=True,
            dropout=DROPOUT if NUM_LAYERS > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, 2),
        )

    def forward(self, full_seq: torch.Tensor) -> torch.Tensor:
        # full_seq: (B, PREFIX_LEN + gap_steps + SUFFIX_LEN, INPUT_DIM)
        out, _ = self.bilstm(full_seq)                             # (B, T, 2H)
        gap_h  = out[:, PREFIX_LEN: PREFIX_LEN + self.gap_steps]  # (B, gap_steps, 2H)
        return self.head(gap_h)                                    # (B, gap_steps, 2)


class TransformerGapFiller(nn.Module):
    """
    Full-context Transformer encoder for trajectory gap filling.

    Same input format as BiLSTMInterpolator: full window (prefix | masked-gap | suffix).
    Gap rows are fed with lat/lon/SOG/COG zeroed (observed_flag=0): no leakage.
    Self-attention over the entire sequence; gap hidden states are projected to lat/lon.

    Architecture: Linear(7→d_model) + LearnedPosEmb + TransformerEncoder → Linear(d_model→2)
    """

    def __init__(
        self,
        gap_steps: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.gap_steps  = gap_steps
        self.input_proj = nn.Linear(INPUT_DIM, d_model)
        max_len         = PREFIX_LEN + gap_steps + SUFFIX_LEN
        self.pos_emb    = nn.Embedding(max_len, d_model)
        enc_layer       = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head    = nn.Linear(d_model, 2)

    def forward(self, full_seq: torch.Tensor) -> torch.Tensor:
        # full_seq: (B, PREFIX_LEN + gap_steps + SUFFIX_LEN, INPUT_DIM)
        B, L, _ = full_seq.shape
        pos = torch.arange(L, device=full_seq.device).unsqueeze(0).expand(B, -1)
        h   = self.input_proj(full_seq) + self.pos_emb(pos)        # (B, T, d_model)
        h   = self.encoder(h)                                       # (B, T, d_model)
        gap_h = h[:, PREFIX_LEN: PREFIX_LEN + self.gap_steps, :]   # (B, gap_steps, d_model)
        return self.head(gap_h)                                     # (B, gap_steps, 2)


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════════

class GapDataset(Dataset):
    def __init__(self, samples: List[Dict], stats: Dict, model_type: str):
        self.samples    = samples
        self.stats      = stats
        self.model_type = model_type   # "lstm" | "gru" | "bilstm" | "transformer"

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s           = self.samples[idx]
        prefix_feat = build_feature_matrix(s["prefix"], self.stats, observed=True)   # (60, 7)
        gap_feat    = build_feature_matrix(s["gap"],    self.stats, observed=False)  # (g, 7)
        suffix_feat = build_feature_matrix(s["suffix"], self.stats, observed=True)   # (60, 7)

        # Target: normalized gap lat/lon — never used as model input
        lat_n = (s["gap"]["latitude"].values  - self.stats["latitude_mean"])  / self.stats["latitude_std"]
        lon_n = (s["gap"]["longitude"].values - self.stats["longitude_mean"]) / self.stats["longitude_std"]
        target = np.stack([lat_n, lon_n], axis=1).astype(np.float32)  # (g, 2)

        if self.model_type in ("bilstm", "transformer"):
            full_seq = np.concatenate([prefix_feat, gap_feat, suffix_feat], axis=0)
            return torch.from_numpy(full_seq), torch.from_numpy(target)
        else:
            return torch.from_numpy(prefix_feat), torch.from_numpy(target)


# ═══════════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════════

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
) -> nn.Module:
    """Train with MSE loss and early stopping. Returns model with best val weights."""
    optimizer  = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn    = nn.MSELoss()
    best_val   = float("inf")
    best_wts   = None
    no_improve = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        for batch in train_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch[0].to(device), batch[1].to(device)
                val_loss += loss_fn(model(x), y).item()
        val_loss /= max(len(val_loader), 1)

        if val_loss < best_val - 1e-6:
            best_val   = val_loss
            best_wts   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                logger.info("    Early stop epoch=%d  best_val=%.5f", epoch + 1, best_val)
                break

    model.load_state_dict(best_wts)
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# Neural inference
# ═══════════════════════════════════════════════════════════════════════════════

def neural_predict(
    model: nn.Module,
    samples: List[Dict],
    stats: Dict,
    model_type: str,
    device: str,
) -> Dict[str, Coords]:
    """Run inference → {sample_id: [(lat, lon), ...]}."""
    ds         = GapDataset(samples, stats, model_type)
    loader     = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    sample_ids = [s["sample_id"] for s in samples]
    preds_norm: List[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            x   = batch[0].to(device)
            out = model(x).cpu().numpy()   # (B, gap_steps, 2)
            preds_norm.append(out)

    preds_norm_all = np.concatenate(preds_norm, axis=0)  # (N, gap_steps, 2)
    result: Dict[str, Coords] = {}
    for i, sid in enumerate(sample_ids):
        latlon = denorm_latlon(preds_norm_all[i], stats)  # (gap_steps, 2)
        result[sid] = [(float(r[0]), float(r[1])) for r in latlon]
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Metric helpers
# ═══════════════════════════════════════════════════════════════════════════════

def sample_true_coords(sample: Dict) -> Coords:
    return [(float(r["latitude"]), float(r["longitude"]))
            for _, r in sample["gap"].iterrows()]


def _spatial(pred: Coords, true: Coords) -> Dict:
    n = min(len(pred), len(true))
    if n == 0:
        nan = float("nan")
        return dict(MSE_km2=nan, MAE_km=nan, ADE_km=nan, FDE_km=nan, DTW_km=nan)
    p, t = pred[:n], true[:n]
    return {
        "MSE_km2": compute_mse_km2(p, t),
        "MAE_km":  compute_mae_km(p, t),
        "ADE_km":  compute_ade_km(p, t),
        "FDE_km":  compute_fde_km(p, t),
        "DTW_km":  compute_dtw_km(p, t),
    }


def evaluate_predictions(preds: Dict[str, Coords], samples: List[Dict]) -> Dict:
    keys  = ["MSE_km2", "MAE_km", "ADE_km", "FDE_km", "DTW_km"]
    accum = {k: [] for k in keys}
    for s in samples:
        sid = s["sample_id"]
        if sid not in preds:
            continue
        m = _spatial(preds[sid], sample_true_coords(s))
        for k in keys:
            v = m[k]
            if not math.isnan(v):
                accum[k].append(v)
    return {k: float(np.mean(vs)) if vs else float("nan") for k, vs in accum.items()}


def aggregate_overall(per_gap: Dict[int, Dict], gap_sizes: List[int]) -> Dict:
    """Average per-gap metrics (equal sample count per gap → equivalent to overall mean)."""
    keys  = ["MSE_km2", "MAE_km", "ADE_km", "FDE_km", "DTW_km"]
    accum = {k: [] for k in keys}
    for g in gap_sizes:
        for k, v in per_gap[g].items():
            if not math.isnan(v):
                accum[k].append(v)
    return {k: float(np.mean(vs)) if vs else float("nan") for k, vs in accum.items()}


# ═══════════════════════════════════════════════════════════════════════════════
# Prediction rows for CSV
# ═══════════════════════════════════════════════════════════════════════════════

def build_pred_rows(preds: Dict[str, Coords], samples: List[Dict]) -> List[Dict]:
    rows = []
    for s in samples:
        sid  = s["sample_id"]
        pred = preds.get(sid, [])
        true = sample_true_coords(s)
        for step, (p, t) in enumerate(zip(pred, true)):
            rows.append({
                "sample_id":     sid,
                "trajectory_id": s["trajectory_id"],
                "gap_steps":     s["gap_steps"],
                "gap_step":      step,
                "pred_lat": p[0], "pred_lon": p[1],
                "true_lat": t[0], "true_lon": t[1],
            })
    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# Model factory
# ═══════════════════════════════════════════════════════════════════════════════

def build_model(model_type: str, gap_steps: int) -> nn.Module:
    if model_type == "lstm":
        return LSTMForecaster(gap_steps, rnn_type="lstm")
    if model_type == "gru":
        return LSTMForecaster(gap_steps, rnn_type="gru")
    if model_type == "bilstm":
        return BiLSTMInterpolator(gap_steps)
    if model_type == "transformer":
        return TransformerGapFiller(gap_steps)
    raise ValueError(f"Unknown model_type: {model_type}")


# ═══════════════════════════════════════════════════════════════════════════════
# Comparison table + printing
# ═══════════════════════════════════════════════════════════════════════════════

def build_comparison_table(all_metrics: Dict, gap_sizes: List[int]) -> pd.DataFrame:
    """Wide-format table: one row per model, columns per gap size × metric."""
    rows = []
    for model_name, metrics in all_metrics.items():
        row = {"model": model_name}
        for g in gap_sizes:
            gm = metrics.get(g, {})
            for met in ["ADE_km", "FDE_km", "DTW_km", "MAE_km", "MSE_km2"]:
                row[f"gap{g}_{met}"] = gm.get(met, float("nan"))
        ov = metrics.get("overall", {})
        for met in ["ADE_km", "FDE_km", "DTW_km", "MAE_km", "MSE_km2"]:
            row[f"overall_{met}"] = ov.get(met, float("nan"))
        rows.append(row)
    return pd.DataFrame(rows)


def build_long_table(all_metrics: Dict, gap_sizes: List[int]) -> pd.DataFrame:
    """
    Long-format table: one row per (model, gap_steps) combination.
    Columns: model | gap_steps | ADE_km | FDE_km | DTW_km | MAE_km | MSE_km2
    Also includes an 'overall' row per model (gap_steps='overall').
    """
    rows = []
    for model_name, metrics in all_metrics.items():
        for g in gap_sizes:
            gm = metrics.get(g, {})
            rows.append({
                "model":     model_name,
                "gap_steps": g,
                "ADE_km":    gm.get("ADE_km",  float("nan")),
                "FDE_km":    gm.get("FDE_km",  float("nan")),
                "DTW_km":    gm.get("DTW_km",  float("nan")),
                "MAE_km":    gm.get("MAE_km",  float("nan")),
                "MSE_km2":   gm.get("MSE_km2", float("nan")),
            })
        ov = metrics.get("overall", {})
        rows.append({
            "model":     model_name,
            "gap_steps": "overall",
            "ADE_km":    ov.get("ADE_km",  float("nan")),
            "FDE_km":    ov.get("FDE_km",  float("nan")),
            "DTW_km":    ov.get("DTW_km",  float("nan")),
            "MAE_km":    ov.get("MAE_km",  float("nan")),
            "MSE_km2":   ov.get("MSE_km2", float("nan")),
        })
    return pd.DataFrame(rows)


def print_comparison(df: pd.DataFrame, gap_sizes: List[int],
                     long_df: Optional[pd.DataFrame] = None) -> None:
    # ── Wide summary: ADE per gap size ────────────────────────────────────────
    print("\n" + "=" * 80)
    print("BASELINE COMPARISON  (ADE_km — lower is better)")
    print("=" * 80)
    cols  = ["model"] + [f"gap{g}_ADE_km" for g in gap_sizes] + ["overall_ADE_km"]
    avail = [c for c in cols if c in df.columns]
    print(df[avail].to_string(index=False, float_format="{:.3f}".format))

    print("\nFDE_km:")
    cols_fde  = ["model"] + [f"gap{g}_FDE_km" for g in gap_sizes] + ["overall_FDE_km"]
    avail_fde = [c for c in cols_fde if c in df.columns]
    print(df[avail_fde].to_string(index=False, float_format="{:.3f}".format))

    print("\nDTW_km:")
    cols_dtw  = ["model"] + [f"gap{g}_DTW_km" for g in gap_sizes] + ["overall_DTW_km"]
    avail_dtw = [c for c in cols_dtw if c in df.columns]
    print(df[avail_dtw].to_string(index=False, float_format="{:.1f}".format))

    # ── Long-format table: model | gap_steps | ADE | FDE | DTW ───────────────
    if long_df is not None:
        print("\n" + "─" * 80)
        print("DETAILED TABLE  (model | gap_steps | ADE_km | FDE_km | DTW_km)")
        print("─" * 80)
        # Print numeric gap rows first, then overall
        numeric = long_df[long_df["gap_steps"] != "overall"].copy()
        numeric["gap_steps"] = numeric["gap_steps"].astype(int)
        numeric = numeric.sort_values(["gap_steps", "model"]).reset_index(drop=True)
        overall = long_df[long_df["gap_steps"] == "overall"].copy()
        combined = pd.concat([numeric, overall], ignore_index=True)
        print(combined[["model", "gap_steps", "ADE_km", "FDE_km", "DTW_km"]].to_string(
            index=False, float_format="{:.3f}".format))

    # ── Best model per gap size ───────────────────────────────────────────────
    print("\n" + "─" * 80)
    print("BEST MODEL PER GAP SIZE (ADE_km)")
    for g in gap_sizes:
        col = f"gap{g}_ADE_km"
        if col in df.columns and df[col].notna().any():
            idx = df[col].idxmin()
            print(f"  gap={g:2d}: ★ {df.loc[idx,'model']:<24s}  ADE={df.loc[idx,col]:.3f} km")
    ov = "overall_ADE_km"
    if ov in df.columns and df[ov].notna().any():
        idx = df[ov].idxmin()
        print(f"  overall: ★ {df.loc[idx,'model']:<24s}  ADE={df.loc[idx,ov]:.3f} km")
    print("=" * 80)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    setup_logging()
    cfg_path  = os.path.join(os.path.dirname(__file__), "..", "configs", "geohash_sft_config.yaml")
    cfg       = load_config(cfg_path)
    gap_sizes = cfg["gap_sizes"]
    out_dir   = Path("outputs") / "baselines"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    # ── Load and split data ───────────────────────────────────────────────────
    logger.info("Loading gap_examples.parquet ...")
    df = pd.read_parquet("processed/gap_examples.parquet")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["COG"]       = df["COG"].astype(float) % 360.0

    train_df = df[df["split"] == "train"]
    val_df   = df[df["split"] == "val"]
    test_df  = df[df["split"] == "test"]
    logger.info("Rows — train=%d  val=%d  test=%d", len(train_df), len(val_df), len(test_df))

    # ── Normalization stats from training set ─────────────────────────────────
    stats = compute_norm_stats(train_df)

    all_metrics:   Dict[str, Dict] = {}
    all_pred_rows: Dict[str, List] = {}

    # ════════════════════════════════════════════════════════════════════════════
    # Rule-based baselines (no training required)
    # ════════════════════════════════════════════════════════════════════════════
    rule_baselines = [
        ("last_known",    predict_last_known),
        ("linear_interp", predict_linear_interp),
    ]
    for name, fn in rule_baselines:
        logger.info("── Baseline: %s ──", name)
        per_gap: Dict[int, Dict] = {}
        rows_all: List[Dict]     = []
        for g in gap_sizes:
            samples  = prepare_samples(test_df, g)
            preds    = {s["sample_id"]: fn(s) for s in samples}
            per_gap[g] = evaluate_predictions(preds, samples)
            rows_all.extend(build_pred_rows(preds, samples))
            logger.info("  gap=%d  ADE=%.3f km  FDE=%.3f km",
                        g, per_gap[g]["ADE_km"], per_gap[g]["FDE_km"])

        all_metrics[name]   = {**per_gap, "overall": aggregate_overall(per_gap, gap_sizes)}
        all_pred_rows[name] = rows_all

    # ════════════════════════════════════════════════════════════════════════════
    # Neural baselines
    # ════════════════════════════════════════════════════════════════════════════
    for model_type in ["lstm", "gru", "bilstm", "transformer"]:
        logger.info("── Baseline: %s ──", model_type.upper())
        per_gap: Dict[int, Dict] = {}
        rows_all: List[Dict]     = []

        for g in gap_sizes:
            logger.info("  Training gap=%d ...", g)
            train_samples = prepare_samples(train_df, g)
            val_samples   = prepare_samples(val_df,   g)
            test_samples  = prepare_samples(test_df,  g)

            train_ds = GapDataset(train_samples, stats, model_type)
            val_ds   = GapDataset(val_samples,   stats, model_type)
            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=False)
            val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

            model = build_model(model_type, g).to(device)
            model = train_model(model, train_loader, val_loader, device)

            preds = neural_predict(model, test_samples, stats, model_type, device)
            per_gap[g] = evaluate_predictions(preds, test_samples)
            rows_all.extend(build_pred_rows(preds, test_samples))
            logger.info("  gap=%d  ADE=%.3f km  FDE=%.3f km",
                        g, per_gap[g]["ADE_km"], per_gap[g]["FDE_km"])

            # Free GPU memory between gap sizes
            del model
            if device == "cuda":
                torch.cuda.empty_cache()

        all_metrics[model_type]   = {**per_gap, "overall": aggregate_overall(per_gap, gap_sizes)}
        all_pred_rows[model_type] = rows_all

    # ════════════════════════════════════════════════════════════════════════════
    # Load LLM metrics from existing evaluation
    # ════════════════════════════════════════════════════════════════════════════
    llm_path = Path("outputs") / "evaluation" / "test_metrics.json"
    if llm_path.exists():
        with open(llm_path) as f:
            llm_json = json.load(f)
        llm_m: Dict[int | str, Dict] = {}
        for g in gap_sizes:
            gm = llm_json.get(f"gap_{g}", {})
            llm_m[g] = {k: gm.get(k, float("nan"))
                        for k in ["ADE_km", "FDE_km", "DTW_km", "MAE_km", "MSE_km2"]}
        ov = llm_json.get("overall", {})
        llm_m["overall"] = {k: ov.get(k, float("nan"))
                            for k in ["ADE_km", "FDE_km", "DTW_km", "MAE_km", "MSE_km2"]}
        all_metrics["LLM_Qwen2.5"] = llm_m
        logger.info("Loaded LLM metrics from %s", llm_path)
    else:
        logger.warning("LLM metrics not found at %s — LLM row omitted", llm_path)

    # ════════════════════════════════════════════════════════════════════════════
    # Save outputs
    # ════════════════════════════════════════════════════════════════════════════
    for name, rows in all_pred_rows.items():
        path = out_dir / f"{name}_predictions.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        logger.info("Saved %s", path)

    # Convert int keys to strings for JSON serialization
    json_metrics = {
        model: {str(k): v for k, v in m.items()}
        for model, m in all_metrics.items()
    }
    metrics_path = out_dir / "baseline_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(json_metrics, f, indent=2)
    logger.info("Saved %s", metrics_path)

    comparison_df = build_comparison_table(all_metrics, gap_sizes)
    cmp_path = out_dir / "comparison_table.csv"
    comparison_df.to_csv(cmp_path, index=False)
    logger.info("Saved %s", cmp_path)

    long_df  = build_long_table(all_metrics, gap_sizes)
    long_path = out_dir / "comparison_long.csv"
    long_df.to_csv(long_path, index=False)
    logger.info("Saved long-format table: %s", long_path)

    # Print final comparison (wide summary + long detail + best-per-gap)
    print_comparison(comparison_df, gap_sizes, long_df=long_df)


if __name__ == "__main__":
    main()
