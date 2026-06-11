#!/usr/bin/env python3
"""
00_validate_input.py — Validate the raw AIS parquet file.

Fail-fast: raises AssertionError on first validation failure.
"""

import logging
import os
import sys

import numpy as np
import pandas as pd

# Allow running from repo root or src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.io_utils import load_config, load_parquet, setup_logging
from src.utils.validation_utils import (
    validate_coordinates,
    validate_dataframe_schema,
    validate_no_nulls,
    validate_sog_cog,
    validate_timestamps,
    validate_trajectory_integrity,
)

logger = logging.getLogger(__name__)


def validate_input(cfg: dict) -> pd.DataFrame:
    input_path = cfg["input_file"]

    # 1. File exists and is readable as Parquet
    if not os.path.exists(input_path):
        raise AssertionError(f"Input file does not exist: {input_path}")

    logger.info("Loading input file: %s", input_path)
    df = load_parquet(input_path)

    # 2. Required columns
    logger.info("Validating schema...")
    validate_dataframe_schema(df)

    # 3. No missing values
    logger.info("Checking for null values...")
    validate_no_nulls(df)

    # 4. Timestamp convertible
    logger.info("Validating timestamps...")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    validate_timestamps(df)

    # 5. Coordinate bounds
    logger.info("Validating coordinates...")
    validate_coordinates(df)

    # 6. SOG/COG validity
    logger.info("Validating SOG/COG...")
    validate_sog_cog(df)

    # 7. Trajectory integrity (min 2 points, no duplicates, strict timestamps, 1000 unique ids)
    logger.info("Validating trajectory integrity...")
    validate_trajectory_integrity(df, strict_1000=cfg.get("strict_1000", True))

    return df


def print_summary(df: pd.DataFrame) -> None:
    logger.info("=" * 60)
    logger.info("INPUT VALIDATION SUMMARY")
    logger.info("=" * 60)

    logger.info("Columns: %s", list(df.columns))
    logger.info("Rows loaded: %d", len(df))
    logger.info("Unique trajectories: %d", df["trajectory_id"].nunique())

    # Timestamp interval statistics per pair of consecutive rows
    df_sorted = df.sort_values(["trajectory_id", "timestamp"])
    raw = (
        df_sorted.groupby("trajectory_id")["timestamp"]
        .apply(lambda ts: ts.diff().dt.total_seconds().dropna())
    )
    # Flatten regardless of whether groupby returns a Series-of-Series or a flat Series
    all_diffs = raw.explode().astype(float).dropna().values / 60.0  # minutes

    logger.info("Timestamp interval stats (minutes):")
    logger.info("  min dt   : %.2f", np.min(all_diffs))
    logger.info("  mean dt  : %.2f", np.mean(all_diffs))
    logger.info("  median dt: %.2f", np.median(all_diffs))
    logger.info("  max dt   : %.2f", np.max(all_diffs))

    traj_lengths = df.groupby("trajectory_id").size()
    logger.info("Trajectory length stats:")
    logger.info("  min: %d  max: %d  mean: %.1f  median: %.1f",
                traj_lengths.min(), traj_lengths.max(),
                traj_lengths.mean(), traj_lengths.median())

    logger.info("SOG range: [%.3f, %.3f]", df["SOG"].min(), df["SOG"].max())
    logger.info("COG range: [%.3f, %.3f]", df["COG"].min(), df["COG"].max())
    logger.info("=" * 60)


def main() -> None:
    setup_logging()
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "configs", "geohash_sft_config.yaml")
    cfg = load_config(cfg_path)

    df = validate_input(cfg)
    print_summary(df)

    logger.info("PASS: Input file passed all validation checks.")


if __name__ == "__main__":
    main()
