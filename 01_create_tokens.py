#!/usr/bin/env python3
"""
01_create_tokens.py — Create geohash_7, sog_token, and cog_token columns.

Supports SOG variants S0/S1/S2/S3 and COG variants C0/C1/C2/C3 via config.
For S2/S3, processed/sog_quantile_thresholds.json must exist (run fit_sog_quantiles.py first).

Saves: <output_dir>/tokenized_trajectories.parquet

Usage:
  python src/01_create_tokens.py
  python src/01_create_tokens.py --config configs/variants/S2_C2.yaml
"""

import logging
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.tokenizers.cog_tokenizer import COGTokenizer
from src.tokenizers.sog_tokenizer import SOGTokenizer, load_sog_tokenizer
from src.utils.geohash_utils import encode_geohash, validate_geohash_series
from src.utils.io_utils import load_config, load_parquet, save_parquet, setup_logging

logger = logging.getLogger(__name__)


def _resolve_cfg_path() -> str:
    if "--config" in sys.argv:
        idx = sys.argv.index("--config")
        if idx + 1 < len(sys.argv):
            return sys.argv[idx + 1]
    return os.path.join(os.path.dirname(__file__), "..", "configs", "geohash_sft_config.yaml")


# ------------------------------------------------------------------ geohash

def create_geohash_tokens(df: pd.DataFrame, precision: int = 7) -> pd.Series:
    lat = df["latitude"].astype(float)
    lon = df["longitude"].astype(float)
    return pd.Series(
        [encode_geohash(la, lo, precision) for la, lo in zip(lat, lon)],
        index=df.index,
        dtype=str,
    )


# ------------------------------------------------------------------ SOG

def sog_to_token(sog: float, cfg: dict) -> str:
    """Single-value wrapper — delegates to SOGTokenizer; used by unit tests."""
    variant = cfg.get("sog_variant", "S0")
    # For S2/S3 tests supply quantile_thresholds directly in cfg["_sog_tokenizer"]
    tok = cfg.get("_sog_tokenizer")
    if tok is None:
        if variant in ("S2", "S3"):
            raise ValueError(
                "sog_to_token: pass a pre-built SOGTokenizer via cfg['_sog_tokenizer'] "
                "for S2/S3 (or use create_sog_tokens_from_tokenizer)."
            )
        tok = SOGTokenizer(variant)
    return tok.token_for_sog(sog)


def create_sog_tokens(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """Legacy S0-compatible entry point; used internally and by old tests."""
    sog = df["SOG"].astype(float)
    if sog.isna().any():
        raise AssertionError("SOG contains NaN values")
    if (sog < 0).any():
        raise AssertionError(f"Negative SOG values: {(sog < 0).sum()} rows")
    variant = cfg.get("sog_variant", "S0")
    tok: SOGTokenizer = cfg.get("_sog_tokenizer") or SOGTokenizer(variant)
    return sog.apply(tok.token_for_sog)


# ------------------------------------------------------------------ COG

def cog_to_token(cog_norm: float, cfg: dict = None) -> str:
    """Single-value wrapper — delegates to COGTokenizer; used by unit tests.
    cog_norm must already be in [0, 360).
    cfg is optional; omitting it defaults to C0 for backward compatibility.
    """
    variant = (cfg or {}).get("cog_variant", "C0")
    tok = COGTokenizer(variant)
    return tok.token_for_cog(cog_norm)


def create_cog_tokens(df: pd.DataFrame, cfg: dict = None) -> pd.Series:
    """Variant-aware COG tokenisation."""
    cog = df["COG"].astype(float)
    if cog.isna().any():
        raise AssertionError("COG contains NaN values")
    variant = (cfg or {}).get("cog_variant", "C0")
    tok = COGTokenizer(variant)
    cog_norm = cog % 360.0
    return cog_norm.apply(tok.token_for_cog)


# ------------------------------------------------------------------ Main

def main() -> None:
    setup_logging()
    cfg_path = _resolve_cfg_path()
    cfg = load_config(cfg_path)
    out_dir = cfg.get("output_dir", "processed")

    sog_variant = cfg.get("sog_variant", "S0")
    cog_variant = cfg.get("cog_variant", "C0")
    logger.info("SOG variant: %s  |  COG variant: %s", sog_variant, cog_variant)

    # ------------------------------------------------------------------ Load input
    logger.info("Loading input file: %s", cfg["input_file"])
    df = load_parquet(cfg["input_file"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Sort for determinism (delta tokens would need this; kept for reproducibility)
    df = df.sort_values(["trajectory_id", "timestamp"]).reset_index(drop=True)

    precision = cfg.get("geohash_precision", 7)

    # ------------------------------------------------------------------ Build tokenizers
    # SOG
    q_path = None
    if sog_variant in ("S2", "S3"):
        import os as _os
        q_path = cfg.get(
            "sog_quantile_thresholds_path",
            _os.path.join(out_dir, "sog_quantile_thresholds.json"),
        )
    sog_tok = load_sog_tokenizer(cfg, quantile_path=q_path)
    cfg["_sog_tokenizer"] = sog_tok   # cache for create_sog_tokens

    cog_tok = COGTokenizer(cog_variant)

    # ------------------------------------------------------------------ Create tokens
    logger.info("Creating geohash_7 tokens (precision=%d)...", precision)
    df["geohash_7"] = create_geohash_tokens(df, precision)

    logger.info("Creating sog_token (%s)...", sog_variant)
    df["sog_token"] = create_sog_tokens(df, cfg)

    logger.info("Creating cog_token (%s)...", cog_variant)
    df["cog_token"] = create_cog_tokens(df, cfg)

    # ------------------------------------------------------------------ Validate
    logger.info("Validating geohash_7...")
    validate_geohash_series(df["geohash_7"], precision)

    logger.info("Validating sog_token...")
    invalid_sog = ~df["sog_token"].isin(sog_tok.valid_tokens())
    if invalid_sog.any():
        raise AssertionError(f"Invalid sog_token values: {df['sog_token'][invalid_sog].unique()[:5]}")

    logger.info("Validating cog_token...")
    invalid_cog = ~df["cog_token"].isin(cog_tok.valid_tokens())
    if invalid_cog.any():
        raise AssertionError(f"Invalid cog_token values: {df['cog_token'][invalid_cog].unique()[:5]}")

    # ------------------------------------------------------------------ Distribution
    logger.info("SOG token distribution:\n%s", df["sog_token"].value_counts().to_string())
    logger.info("COG token distribution:\n%s", df["cog_token"].value_counts().to_string())

    # Warn on bins with < 0.5% share (potentially too sparse for reliable learning)
    total = len(df)
    sog_counts = df["sog_token"].value_counts()
    for tok_name, cnt in sog_counts.items():
        pct = 100 * cnt / total
        if pct < 0.5:
            logger.warning("SOG bin '%s' is rare: %.2f%% of observations", tok_name, pct)
    cog_counts = df["cog_token"].value_counts()
    for tok_name, cnt in cog_counts.items():
        pct = 100 * cnt / total
        if pct < 0.5:
            logger.warning("COG bin '%s' is rare: %.2f%% of observations", tok_name, pct)

    # ------------------------------------------------------------------ Save bin configs
    import json
    from pathlib import Path
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    sog_tok.save_config(os.path.join(out_dir, "tokenizer_config_sog_v2.json"))
    cog_tok.save_config(os.path.join(out_dir, "tokenizer_config_cog_v2.json"))

    # Combined special tokens file
    special = sog_tok.special_tokens() + cog_tok.special_tokens()
    combined = {
        "sog_variant": sog_variant,
        "cog_variant": cog_variant,
        "sog_special_tokens": sog_tok.special_tokens(),
        "cog_special_tokens": cog_tok.special_tokens(),
        "all_special_tokens": special,
        "num_added": len(special),
    }
    with open(os.path.join(out_dir, "tokenizer_config_sog_cog_v2.json"), "w") as f:
        json.dump(combined, f, indent=2)

    added_tokens_path = os.path.join(out_dir, "added_special_tokens_sog_cog_v2.txt")
    with open(added_tokens_path, "w") as f:
        for t in special:
            f.write(t + "\n")
    logger.info("Special tokens to add to Qwen vocab: %d → %s", len(special), added_tokens_path)

    # ------------------------------------------------------------------ Save parquet
    out_path = os.path.join(out_dir, "tokenized_trajectories.parquet")
    save_parquet(df, out_path)
    logger.info("PASS: Tokenization complete → %s", out_path)


if __name__ == "__main__":
    main()
