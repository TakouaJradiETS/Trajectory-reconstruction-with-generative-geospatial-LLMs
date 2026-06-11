#!/usr/bin/env python3
"""
03_create_gap_examples.py — Create artificial observation gaps for SFT training.

For each trajectory × gap_size, extracts a deterministic central window:
  prefix (60 obs) | gap (N obs) | suffix (60 obs)

Saves:
  processed/gap_examples.parquet  — all windowed rows
  processed/gap_metadata.csv      — one row per sample with statistics
"""

import logging
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.geohash_utils import is_valid_geohash
from src.utils.io_utils import load_config, load_parquet, save_csv, save_parquet, setup_logging
from src.utils.validation_utils import VALID_COG_TOKENS, VALID_SOG_TOKENS

logger = logging.getLogger(__name__)

PREFIX = 60
SUFFIX = 60


def _compute_dt_min(timestamps: pd.Series) -> list[float]:
    """dt_min for each row in a window; first row = 0.0."""
    ts = timestamps.reset_index(drop=True)
    dt = [0.0]
    for i in range(1, len(ts)):
        delta = (ts.iloc[i] - ts.iloc[i - 1]).total_seconds() / 60.0
        dt.append(float(delta))
    return dt


def _make_sample(
    traj_df: pd.DataFrame,
    gap_steps: int,
    trajectory_id: str,
    mmsi: int,
    split: str,
    window_start: int,
    sample_suffix: str = "",
) -> Optional[tuple[pd.DataFrame, dict]]:
    """
    Extract one sample and return (rows_df, metadata_dict).
    sample_suffix is appended to the sample_id for multi-window augmentation (e.g. "_w0", "_w1").
    Returns None if the window is not valid (should not happen after pre-check).
    """
    required = PREFIX + gap_steps + SUFFIX
    traj_sorted = traj_df.reset_index(drop=True)

    # Sanity
    if window_start + required > len(traj_sorted):
        return None

    window = traj_sorted.iloc[window_start: window_start + required].copy()
    window = window.reset_index(drop=True)

    prefix_end = PREFIX
    gap_end = PREFIX + gap_steps

    # Row type labels
    row_types = (
        ["prefix"] * PREFIX +
        ["gap"] * gap_steps +
        ["suffix"] * SUFFIX
    )
    row_idx_in_section = (
        list(range(PREFIX)) +
        list(range(gap_steps)) +
        list(range(SUFFIX))
    )

    # dt_min over the whole window
    dt_min_vals = _compute_dt_min(window["timestamp"])

    window["row_type"] = row_types
    window["row_index_in_section"] = row_idx_in_section
    window["global_row_index"] = list(range(required))
    window["dt_min"] = dt_min_vals

    # Sample metadata
    sample_id = f"{trajectory_id}_gap{gap_steps}{sample_suffix}"
    prefix_rows = window[window["row_type"] == "prefix"]
    gap_rows = window[window["row_type"] == "gap"]
    suffix_rows = window[window["row_type"] == "suffix"]

    gap_ts = gap_rows["timestamp"]
    prefix_ts = prefix_rows["timestamp"]
    suffix_ts = suffix_rows["timestamp"]

    hidden_start_ts = gap_ts.iloc[0]
    hidden_end_ts = gap_ts.iloc[-1]
    last_prefix_ts = prefix_ts.iloc[-1]
    first_suffix_ts = suffix_ts.iloc[0]

    hidden_dur = (hidden_end_ts - hidden_start_ts).total_seconds() / 60.0
    obs_gap_dur = (first_suffix_ts - last_prefix_ts).total_seconds() / 60.0

    window_dt = window["dt_min"].values[1:]  # exclude first (always 0.0)

    metadata = {
        "sample_id": sample_id,
        "trajectory_id": trajectory_id,
        "MMSI": mmsi,
        "split": split,
        "gap_steps": gap_steps,
        "prefix_steps": PREFIX,
        "suffix_steps": SUFFIX,
        "window_start_index": window_start,
        "gap_start_index": window_start + PREFIX,
        "gap_end_index": window_start + PREFIX + gap_steps - 1,
        "window_end_index": window_start + required - 1,
        "hidden_gap_start_timestamp": str(hidden_start_ts),
        "hidden_gap_end_timestamp": str(hidden_end_ts),
        "last_prefix_timestamp": str(last_prefix_ts),
        "first_suffix_timestamp": str(first_suffix_ts),
        "hidden_internal_duration_min": round(hidden_dur, 4),
        "observable_gap_duration_min": round(obs_gap_dur, 4),
        "max_dt_inside_window_min": round(float(np.max(window_dt)), 4),
        "mean_dt_inside_window_min": round(float(np.mean(window_dt)), 4),
        "median_dt_inside_window_min": round(float(np.median(window_dt)), 4),
        "num_prefix_rows": int(len(prefix_rows)),
        "num_gap_rows": int(len(gap_rows)),
        "num_suffix_rows": int(len(suffix_rows)),
    }

    window["sample_id"] = sample_id
    window["trajectory_id"] = trajectory_id
    window["MMSI"] = mmsi
    window["split"] = split
    window["gap_steps"] = gap_steps

    return window, metadata


def validate_sample(window: pd.DataFrame, metadata: dict, gap_steps: int) -> None:
    """Fail-fast validation of a single generated sample."""
    sid = metadata["sample_id"]
    tid = metadata["trajectory_id"]

    def fail(msg: str):
        raise AssertionError(
            f"Sample validation failed | sample_id={sid} | trajectory_id={tid} "
            f"| gap_steps={gap_steps} | reason: {msg}"
        )

    prefix_rows = window[window["row_type"] == "prefix"]
    gap_rows = window[window["row_type"] == "gap"]
    suffix_rows = window[window["row_type"] == "suffix"]

    # Length checks
    required = PREFIX + gap_steps + SUFFIX
    if len(window) != required:
        fail(f"window length {len(window)} != {required}")
    if len(prefix_rows) != PREFIX:
        fail(f"prefix rows {len(prefix_rows)} != {PREFIX}")
    if len(gap_rows) != gap_steps:
        fail(f"gap rows {len(gap_rows)} != {gap_steps}")
    if len(suffix_rows) != SUFFIX:
        fail(f"suffix rows {len(suffix_rows)} != {SUFFIX}")

    # Order: prefix before gap before suffix
    prefix_idx = prefix_rows.index.tolist()
    gap_idx = gap_rows.index.tolist()
    suffix_idx = suffix_rows.index.tolist()
    if max(prefix_idx) >= min(gap_idx):
        fail("prefix rows do not all precede gap rows")
    if max(gap_idx) >= min(suffix_idx):
        fail("gap rows do not all precede suffix rows")

    # Contiguous
    all_idx = sorted(window.index.tolist())
    if all_idx != list(range(all_idx[0], all_idx[0] + required)):
        fail("window rows are not contiguous")

    # Timestamps strictly increasing
    ts = window["timestamp"].values
    diffs = np.diff(ts.astype("int64"))
    if (diffs <= 0).any():
        bad = np.where(diffs <= 0)[0]
        fail(f"timestamps not strictly increasing at positions {bad[:3].tolist()}")

    # No NaT
    if window["timestamp"].isna().any():
        fail("NaT found in timestamp")

    # Prefix/suffix geohash, sog_token, cog_token validity
    for col in ["geohash_7", "sog_token", "cog_token"]:
        if prefix_rows[col].isna().any() or suffix_rows[col].isna().any():
            fail(f"Null {col} in prefix/suffix")

    for gh in prefix_rows["geohash_7"]:
        if not is_valid_geohash(gh):
            fail(f"Invalid geohash in prefix: {gh}")
    for gh in suffix_rows["geohash_7"]:
        if not is_valid_geohash(gh):
            fail(f"Invalid geohash in suffix: {gh}")

    for tok in prefix_rows["sog_token"]:
        if tok not in VALID_SOG_TOKENS:
            fail(f"Invalid sog_token in prefix: {tok}")
    for tok in suffix_rows["sog_token"]:
        if tok not in VALID_SOG_TOKENS:
            fail(f"Invalid sog_token in suffix: {tok}")

    for tok in prefix_rows["cog_token"]:
        if tok not in VALID_COG_TOKENS:
            fail(f"Invalid cog_token in prefix: {tok}")
    for tok in suffix_rows["cog_token"]:
        if tok not in VALID_COG_TOKENS:
            fail(f"Invalid cog_token in suffix: {tok}")

    # Metadata counts
    if metadata["num_prefix_rows"] != PREFIX:
        fail(f"metadata num_prefix_rows {metadata['num_prefix_rows']} != {PREFIX}")
    if metadata["num_gap_rows"] != gap_steps:
        fail(f"metadata num_gap_rows {metadata['num_gap_rows']} != {gap_steps}")
    if metadata["num_suffix_rows"] != SUFFIX:
        fail(f"metadata num_suffix_rows {metadata['num_suffix_rows']} != {SUFFIX}")


def _get_window_starts(L: int, required: int, num_windows: int) -> list[int]:
    """
    Return up to `num_windows` deterministic, non-overlapping window start positions.

    Windows are spread evenly across the valid range [0, L - required]:
      - window 0: left (start of trajectory)
      - window 1: center
      - window 2: right (end of trajectory)
    For num_windows=1 this reduces to the original central window.
    """
    max_start = L - required
    if max_start < 0:
        return []
    if num_windows == 1:
        return [(L - required) // 2]
    starts = []
    for i in range(num_windows):
        # Evenly spaced from 0 to max_start
        start = int(round(max_start * i / (num_windows - 1))) if num_windows > 1 else max_start // 2
        starts.append(start)
    # Deduplicate (can happen for very short trajectories)
    seen: set[int] = set()
    unique = []
    for s in starts:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return unique


def _resolve_cfg_path() -> str:
    if "--config" in sys.argv:
        idx = sys.argv.index("--config")
        if idx + 1 < len(sys.argv):
            return sys.argv[idx + 1]
    return os.path.join(os.path.dirname(__file__), "..", "configs", "geohash_sft_config.yaml")


def main() -> None:
    setup_logging()
    cfg_path = _resolve_cfg_path()
    cfg = load_config(cfg_path)
    out_dir = cfg["output_dir"]
    # splits_dir: shared directory where trajectory_splits.csv always lives.
    # Defaults to out_dir for backward compatibility (baseline S0/C0 run).
    splits_dir = cfg.get("splits_dir", out_dir)
    gap_sizes = cfg["gap_sizes"]

    # num_windows_per_trajectory: 1 = original single central window (default)
    #                              3 = left / center / right windows (3× data augmentation)
    num_windows = int(cfg.get("num_windows_per_trajectory", 1))
    logger.info("Windows per trajectory per gap size: %d", num_windows)

    tok_path    = os.path.join(out_dir,    "tokenized_trajectories.parquet")
    splits_path = os.path.join(splits_dir, "trajectory_splits.csv")

    df = load_parquet(tok_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    splits_df = pd.read_csv(splits_path)

    split_map = dict(zip(splits_df["trajectory_id"], splits_df["split"]))

    all_rows: list[pd.DataFrame] = []
    all_metadata: list[dict] = []
    skipped: dict[int, int] = {g: 0 for g in gap_sizes}
    created: dict[int, int] = {g: 0 for g in gap_sizes}

    trajectory_ids = sorted(df["trajectory_id"].unique())
    logger.info("Processing %d trajectories × %d gap sizes × %d windows...",
                len(trajectory_ids), len(gap_sizes), num_windows)

    # Sort entire dataframe once for efficiency
    df_sorted = df.sort_values(["trajectory_id", "timestamp"]).reset_index(drop=True)

    for traj_id in tqdm(trajectory_ids, desc="Trajectories"):
        traj_df = df_sorted[df_sorted["trajectory_id"] == traj_id].copy()
        traj_df = traj_df.sort_values("timestamp").reset_index(drop=True)
        mmsi = int(traj_df["MMSI"].iloc[0])
        split = split_map[traj_id]
        L = len(traj_df)

        for gap_steps in gap_sizes:
            required = PREFIX + gap_steps + SUFFIX
            # Multi-window augmentation applies ONLY to training split.
            # Val and test always use 1 center window so the evaluation set
            # stays fixed and comparable across runs.
            actual_windows = num_windows if split == "train" else 1
            window_starts = _get_window_starts(L, required, actual_windows)

            if not window_starts:
                logger.warning(
                    "SKIP: %s gap%d — trajectory too short (%d < %d)",
                    traj_id, gap_steps, L, required,
                )
                skipped[gap_steps] += 1
                continue

            for win_idx, window_start in enumerate(window_starts):
                # Add window suffix only when multiple windows are actually used
                # (val/test always use actual_windows=1 so get no suffix)
                sample_suffix = f"_w{win_idx}" if actual_windows > 1 else ""

                result = _make_sample(
                    traj_df, gap_steps, traj_id, mmsi, split, window_start,
                    sample_suffix=sample_suffix,
                )
                if result is None:
                    logger.warning("SKIP: %s gap%d win%d — window extraction failed",
                                   traj_id, gap_steps, win_idx)
                    skipped[gap_steps] += 1
                    continue

                window, metadata = result
                validate_sample(window, metadata, gap_steps)

                all_rows.append(window)
                all_metadata.append(metadata)
                created[gap_steps] += 1

    logger.info("Created samples per gap size: %s", created)
    logger.info("Skipped trajectories per gap size: %s", skipped)

    # Save gap_examples.parquet
    examples_df = pd.concat(all_rows, ignore_index=True)
    examples_path = os.path.join(out_dir, "gap_examples.parquet")
    save_parquet(examples_df, examples_path)

    # Save gap_metadata.csv
    metadata_df = pd.DataFrame(all_metadata)
    metadata_path = os.path.join(out_dir, "gap_metadata.csv")
    save_csv(metadata_df, metadata_path)

    # Summary
    total = sum(created.values())
    logger.info("Total samples created: %d", total)
    for split_name in ["train", "val", "test"]:
        n = sum(1 for m in all_metadata if m["split"] == split_name)
        logger.info("  %s: %d", split_name, n)

    logger.info("PASS: Gap examples saved → %s and %s", examples_path, metadata_path)


if __name__ == "__main__":
    main()
