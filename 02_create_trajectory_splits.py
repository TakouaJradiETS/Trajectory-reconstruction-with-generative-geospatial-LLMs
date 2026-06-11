#!/usr/bin/env python3
"""
02_create_trajectory_splits.py — Split 1000 trajectories 800/100/100 by trajectory_id.

Saves: processed/trajectory_splits.csv
"""

import logging
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.io_utils import load_config, load_parquet, save_csv, setup_logging
from src.utils.validation_utils import validate_split_integrity

logger = logging.getLogger(__name__)


def create_splits(trajectory_ids: list[str], cfg: dict, seed: int) -> pd.DataFrame:
    split_cfg = cfg["split"]
    n_train = split_cfg["train_size"]
    n_val = split_cfg["val_size"]
    n_test = split_cfg["test_size"]

    total = len(trajectory_ids)
    assert total == n_train + n_val + n_test, (
        f"Total trajectories {total} != {n_train}+{n_val}+{n_test}"
    )

    rng = np.random.default_rng(seed)
    shuffled = np.array(sorted(trajectory_ids))  # sort first for determinism
    rng.shuffle(shuffled)

    train_ids = shuffled[:n_train]
    val_ids = shuffled[n_train:n_train + n_val]
    test_ids = shuffled[n_train + n_val:]

    rows = (
        [(tid, "train") for tid in train_ids] +
        [(tid, "val") for tid in val_ids] +
        [(tid, "test") for tid in test_ids]
    )
    splits = pd.DataFrame(rows, columns=["trajectory_id", "split"])
    return splits


def main() -> None:
    setup_logging()
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "configs", "geohash_sft_config.yaml")
    cfg = load_config(cfg_path)
    seed = cfg.get("seed", 42)
    out_dir    = cfg["output_dir"]
    splits_dir = cfg.get("splits_dir", out_dir)   # shared; default to out_dir for baseline

    tok_path = os.path.join(out_dir, "tokenized_trajectories.parquet")
    df = load_parquet(tok_path)

    trajectory_ids = df["trajectory_id"].unique().tolist()
    logger.info("Found %d unique trajectory_ids", len(trajectory_ids))

    splits = create_splits(trajectory_ids, cfg, seed)

    # Validate split integrity
    validate_split_integrity(splits)

    out_path = os.path.join(splits_dir, "trajectory_splits.csv")
    save_csv(splits, out_path)

    logger.info("Split counts:")
    for split_name, count in splits["split"].value_counts().items():
        logger.info("  %s: %d", split_name, count)

    logger.info("PASS: Trajectory splits saved → %s", out_path)


if __name__ == "__main__":
    main()
