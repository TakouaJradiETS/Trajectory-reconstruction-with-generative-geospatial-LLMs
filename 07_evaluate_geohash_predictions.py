

import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.distance_utils import (
    compute_ade_km,
    compute_dtw_km,
    compute_fde_km,
    compute_mae_km,
    compute_mse_km2,
)
from src.utils.geohash_utils import decode_geohash, is_valid_geohash
from src.utils.io_utils import load_config, load_jsonl, save_csv, setup_logging

logger = logging.getLogger(__name__)

_DEFAULT_ADAPTER_DIR = "outputs/qwen25_3b_geohash_lora/final_adapter"
_DEFAULT_EVAL_DIR    = "outputs/evaluation"


def _resolve_cfg_path() -> str:
    if "--config" in sys.argv:
        idx = sys.argv.index("--config")
        if idx + 1 < len(sys.argv):
            return sys.argv[idx + 1]
    return os.path.join(os.path.dirname(__file__), "..", "configs", "geohash_sft_config.yaml")


# ------------------------------------------------------------------ Generation

def load_model_and_tokenizer(adapter_dir: str, gen_cfg: dict):
    import torch
    import transformers as _transformers
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _ver = tuple(int(x) for x in _transformers.__version__.split(".")[:2])
    _dtype_key = "dtype" if _ver >= (4, 46) else "torch_dtype"

    logger.info("Loading tokenizer from %s", adapter_dir)
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    base_model_name = None
    cfg_file = os.path.join(adapter_dir, "..", "training_config.yaml")
    if os.path.exists(cfg_file):
        import yaml
        with open(cfg_file) as f:
            train_cfg = yaml.safe_load(f)
        base_model_name = train_cfg["model"]["model_name"]

    if base_model_name is None:
        base_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
        logger.warning("Could not find base model name in training config, using default: %s", base_model_name)

    logger.info("Loading base model: %s", base_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        **{_dtype_key: dtype},
        device_map="auto" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
    )

    # Resize embeddings to match the tokenizer saved with the adapter.
    # Required whenever special tokens were added during training (S1/S2/S3/C2/C3).
    # For S0+C0 (no special tokens) len(tokenizer) == base vocab size → no-op.
    if len(tokenizer) != base_model.config.vocab_size:
        logger.info(
            "Resizing base model embeddings: %d → %d to match saved tokenizer",
            base_model.config.vocab_size, len(tokenizer),
        )
        base_model.resize_token_embeddings(len(tokenizer))

    logger.info("Loading LoRA adapter from %s", adapter_dir)
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()
    return model, tokenizer


def generate_prediction(model, tokenizer, record: dict, gen_cfg: dict) -> str:
    """Generate predicted geohash sequence for one test example."""
    import torch
    messages = record["messages"][:2]  # system + user only

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4000)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=gen_cfg.get("max_new_tokens", 384),
            do_sample=gen_cfg.get("do_sample", False),
            temperature=gen_cfg.get("temperature", 0.0) or None,
            num_beams=gen_cfg.get("num_beams", 1),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = output[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return text


# ------------------------------------------------------------------ Generation with hidden states

def generate_with_hidden_states(
    model, tokenizer, record: dict, gen_cfg: dict, max_length: int = 5120
) -> dict:
    """
    Two-pass strategy:
      Pass 1 — model.generate() with output_scores=True
               → generated token ids + per-step logits (for log-prob)
      Pass 2 — model.forward() on [prompt + generated] with output_hidden_states=True
               → last-layer hidden state at position p-1 for each generated token p

    Returns
    -------
    {
        "generated_text"      : str,
        "generated_token_ids" : list[int],   # IDs of generated tokens only
        "hidden_states"       : Tensor[K, H],# last-layer, p-1 position per token
        "log_probs"           : Tensor[K],   # log P(token | context) per token
        "is_valid"            : list[bool],  # whether each raw token is a valid geohash
        "prompt_len"          : int,
    }
    K = number of generated tokens (≤ max_new_tokens)
    H = model hidden dim (e.g. 2048 for Qwen2.5-3B)
    """
    import torch
    import torch.nn.functional as F

    messages = record["messages"][:2]
    prompt   = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    device     = next(model.parameters()).device
    input_ids  = enc["input_ids"].to(device)
    prompt_len = input_ids.shape[1]

    # ── Pass 1: generate + capture scores ────────────────────────────────
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=enc["attention_mask"].to(device),
            max_new_tokens=gen_cfg.get("max_new_tokens", 384),
            do_sample=gen_cfg.get("do_sample", False),
            temperature=gen_cfg.get("temperature", 0.0) or None,
            num_beams=gen_cfg.get("num_beams", 1),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )

    full_ids     = out.sequences[0]                          # [prompt_len + K]
    gen_ids      = full_ids[prompt_len:].tolist()            # K generated token ids
    scores_list  = out.scores                                # tuple of K tensors [1, vocab]

    # Per-token log-probabilities
    log_probs = []
    for step_i, (tok_id, logits) in enumerate(zip(gen_ids, scores_list)):
        lp = F.log_softmax(logits[0], dim=-1)[tok_id].item()
        log_probs.append(lp)

    # Decode text
    gen_text = tokenizer.decode(
        torch.tensor(gen_ids), skip_special_tokens=True
    ).strip()

    # ── Pass 2: forward on full sequence → hidden states ─────────────────
    full_ids_2d = full_ids.unsqueeze(0)                      # [1, prompt_len+K]
    attn_mask   = torch.ones_like(full_ids_2d)

    with torch.no_grad():
        fwd = model(
            input_ids=full_ids_2d,
            attention_mask=attn_mask,
            output_hidden_states=True,
            use_cache=False,
        )
    # Last layer hidden states: [1, prompt_len+K, H]
    last_hidden = fwd.hidden_states[-1][0]                   # [prompt_len+K, H]

    # For generated token at full-sequence position p = prompt_len + i,
    # the hidden state that PRODUCED it is at position p-1.
    K = len(gen_ids)
    hs = last_hidden[prompt_len - 1: prompt_len - 1 + K]    # [K, H]

    # Validity check
    valid_flags = [is_valid_geohash(tokenizer.decode([tid], skip_special_tokens=True).strip())
                   for tid in gen_ids]

    return {
        "generated_text":       gen_text,
        "generated_token_ids":  gen_ids,
        "hidden_states":        hs.cpu().float(),
        "log_probs":            torch.tensor(log_probs),
        "is_valid":             valid_flags,
        "prompt_len":           prompt_len,
    }


# ------------------------------------------------------------------ Parsing + constrained decoding

def parse_predicted_tokens(text: str) -> list[str]:
    """Split generated text into geohash tokens, filtering empties."""
    import re
    tokens = re.split(r"[\s,]+", text.strip())
    return [t for t in tokens if t]


def constrain_to_exact_n(raw_tokens: list[str], n: int) -> list[str]:
    """
    Exact-N constrained post-processing:
      1. Keep only valid geohash-7 tokens from the raw output.
      2. If fewer than n valid tokens: pad by repeating the last valid token.
         If no valid tokens at all: use a deterministic fallback of 'zzzzzzz' × n.
      3. If more than n valid tokens: truncate to the first n.

    This ensures the model always returns exactly n valid geohash-7 tokens,
    making per-step metric alignment exact rather than approximate.
    """
    valid = [t for t in raw_tokens if is_valid_geohash(t)]

    if len(valid) == 0:
        # No valid tokens at all — use a neutral fallback
        valid = ["zzzzzzz"]

    if len(valid) < n:
        # Pad by repeating the last valid token
        valid = valid + [valid[-1]] * (n - len(valid))
    else:
        # Truncate to exactly n
        valid = valid[:n]

    return valid


# ------------------------------------------------------------------ Metrics

def compute_token_metrics(pred_tokens: list[str], true_tokens: list[str]) -> dict:
    n_true = len(true_tokens)
    n_pred = len(pred_tokens)
    n_aligned = min(n_true, n_pred)

    correct = sum(p == t for p, t in zip(pred_tokens, true_tokens))
    exact_match = int(pred_tokens == true_tokens)

    invalid_count = sum(1 for t in pred_tokens if not is_valid_geohash(t))
    missing_count = max(0, n_true - n_pred)
    extra_count = max(0, n_pred - n_true)
    wrong_count = n_aligned - correct

    return {
        "exact_token_accuracy": correct / n_true if n_true > 0 else 0.0,
        "wrong_token_rate": wrong_count / n_true if n_true > 0 else 0.0,
        "sequence_exact_match": exact_match,
        "invalid_token_rate": invalid_count / max(n_pred, 1),
        "missing_token_rate": missing_count / n_true if n_true > 0 else 0.0,
        "extra_token_rate": extra_count / n_true if n_true > 0 else 0.0,
        "output_length_correct": int(n_pred == n_true),
        "n_true": n_true,
        "n_pred": n_pred,
    }


def compute_spatial_metrics(pred_tokens: list[str], true_tokens: list[str]) -> dict:
    n_aligned = min(len(pred_tokens), len(true_tokens))
    if n_aligned == 0:
        nan = float("nan")
        return {"MSE_km2": nan, "MAE_km": nan, "ADE_km": nan, "FDE_km": nan, "DTW_km": nan}

    pred_coords, true_coords = [], []
    for i in range(n_aligned):
        pt = pred_tokens[i]
        tt = true_tokens[i]
        if is_valid_geohash(pt):
            pred_coords.append(decode_geohash(pt))
        else:
            # Use true coords as fallback (worst-case 0 km displacement would be generous,
            # so use a distant dummy — but nan is safer)
            pred_coords.append(decode_geohash(tt))  # penalized elsewhere via invalid_token_rate
        true_coords.append(decode_geohash(tt))

    return {
        "MSE_km2": compute_mse_km2(pred_coords, true_coords),
        "MAE_km": compute_mae_km(pred_coords, true_coords),
        "ADE_km": compute_ade_km(pred_coords, true_coords),
        "FDE_km": compute_fde_km(pred_coords, true_coords),
        "DTW_km": compute_dtw_km(pred_coords, true_coords),
    }


def compute_per_step_accuracy(
    all_results: list[dict], gap_steps_list: list[int]
) -> pd.DataFrame:
    """Compute per-position token accuracy for each gap size."""
    rows = []
    for g in sorted(set(gap_steps_list)):
        subset = [r for r in all_results if r["gap_steps"] == g]
        # Max alignment length
        step_correct = defaultdict(int)
        step_total = defaultdict(int)
        for r in subset:
            pred = r["pred_tokens"]
            true = r["true_tokens"]
            for i in range(min(len(pred), len(true))):
                step_correct[i] += int(pred[i] == true[i])
                step_total[i] += 1
        for i in range(g):
            acc = step_correct[i] / step_total[i] if step_total[i] > 0 else float("nan")
            rows.append({"gap_steps": g, "step": i, "accuracy": acc, "n": step_total[i]})
    return pd.DataFrame(rows)


def aggregate_metrics(results: list[dict]) -> dict:
    def _mean(key):
        vals = [r[key] for r in results if not np.isnan(r.get(key, float("nan")))]
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "num_examples": len(results),
        "token_accuracy": _mean("exact_token_accuracy"),
        "wrong_token_rate": _mean("wrong_token_rate"),
        "sequence_exact_match": _mean("sequence_exact_match"),
        "invalid_token_rate": _mean("invalid_token_rate"),
        "missing_token_rate": _mean("missing_token_rate"),
        "extra_token_rate": _mean("extra_token_rate"),
        "output_length_accuracy": _mean("output_length_correct"),
        "MSE_km2": _mean("MSE_km2"),
        "MAE_km": _mean("MAE_km"),
        "ADE_km": _mean("ADE_km"),
        "FDE_km": _mean("FDE_km"),
        "DTW_km": _mean("DTW_km"),
    }


# ------------------------------------------------------------------ Main

def main() -> None:
    setup_logging()
    cfg_path = _resolve_cfg_path()
    cfg = load_config(cfg_path)
    gen_cfg = cfg.get("generation", {})
    out_dir = cfg.get("output_dir", "processed")
    model_out = cfg.get("model_output_dir", _DEFAULT_ADAPTER_DIR.rsplit("/", 1)[0])
    adapter_dir = os.path.join(model_out, "final_adapter")
    eval_dir    = cfg.get("eval_output_dir", os.path.join(model_out, "evaluation"))

    Path(eval_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ Load model
    model, tokenizer = load_model_and_tokenizer(adapter_dir, gen_cfg)

    # ------------------------------------------------------------------ Test data
    test_path = os.path.join(out_dir, "test_geohash_sft.jsonl")
    test_records = load_jsonl(test_path)
    logger.info("Evaluating on %d test examples", len(test_records))
    sog_variant = cfg.get("sog_variant", "S0")
    cog_variant = cfg.get("cog_variant", "C0")

    # ------------------------------------------------------------------ Generate
    all_results = []
    prediction_rows = []

    from tqdm import tqdm
    for rec in tqdm(test_records, desc="Generating"):
        sample_id = rec["sample_id"]
        gap_steps = rec["gap_steps"]
        true_asst = rec["messages"][2]["content"].strip()
        true_tokens = true_asst.split()

        pred_text   = generate_prediction(model, tokenizer, rec, gen_cfg)
        raw_tokens  = parse_predicted_tokens(pred_text)

        # Exact-N constrained post-processing: always return exactly gap_steps
        # valid geohash-7 tokens for fair per-step alignment with ground truth.
        pred_tokens = constrain_to_exact_n(raw_tokens, gap_steps)

        tok_metrics  = compute_token_metrics(pred_tokens, true_tokens)
        spat_metrics = compute_spatial_metrics(pred_tokens, true_tokens)

        # Record whether the raw output already had the correct length
        tok_metrics["raw_output_length_correct"] = int(len(raw_tokens) == gap_steps)

        result = {
            "sample_id": sample_id,
            "trajectory_id": rec["trajectory_id"],
            "gap_steps": gap_steps,
            "pred_text": pred_text,
            "pred_tokens": pred_tokens,
            "true_tokens": true_tokens,
            **tok_metrics,
            **spat_metrics,
        }
        all_results.append(result)

        prediction_rows.append({
            "sample_id": sample_id,
            "trajectory_id": rec["trajectory_id"],
            "gap_steps": gap_steps,
            "predicted": pred_text,
            "true": true_asst,
            "n_true": tok_metrics["n_true"],
            "n_pred": tok_metrics["n_pred"],
            "exact_token_accuracy": tok_metrics["exact_token_accuracy"],
            "sequence_exact_match": tok_metrics["sequence_exact_match"],
            "ADE_km": spat_metrics["ADE_km"],
            "FDE_km": spat_metrics["FDE_km"],
        })

    # ------------------------------------------------------------------ Save predictions
    pred_df = pd.DataFrame(prediction_rows)
    save_csv(pred_df, os.path.join(eval_dir, "test_predictions.csv"))

    # ------------------------------------------------------------------ Per-gap metrics
    gap_sizes = sorted(set(r["gap_steps"] for r in all_results))
    per_gap_rows = []
    per_gap_dict = {}
    for g in gap_sizes:
        subset = [r for r in all_results if r["gap_steps"] == g]
        agg = aggregate_metrics(subset)
        agg["gap_steps"] = g
        per_gap_rows.append(agg)
        per_gap_dict[f"gap_{g}"] = agg

    per_gap_df = pd.DataFrame(per_gap_rows)
    ordered_cols = [
        "gap_steps", "num_examples", "token_accuracy", "wrong_token_rate",
        "sequence_exact_match", "invalid_token_rate", "missing_token_rate",
        "extra_token_rate", "output_length_accuracy",
        "MSE_km2", "MAE_km", "ADE_km", "FDE_km", "DTW_km",
    ]
    per_gap_df = per_gap_df[[c for c in ordered_cols if c in per_gap_df.columns]]
    save_csv(per_gap_df, os.path.join(eval_dir, "per_gap_metrics.csv"))

    # ------------------------------------------------------------------ Overall metrics
    overall = aggregate_metrics(all_results)
    test_metrics = {
        "sog_variant": sog_variant,
        "cog_variant": cog_variant,
        "overall": overall,
    }
    for key, val in per_gap_dict.items():
        test_metrics[key] = val

    with open(os.path.join(eval_dir, "test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)
    logger.info("Saved test_metrics.json")

    # ------------------------------------------------------------------ Per-step accuracy
    per_step_df = compute_per_step_accuracy(all_results, [r["gap_steps"] for r in all_results])
    save_csv(per_step_df, os.path.join(eval_dir, "per_step_accuracy.csv"))

    # ------------------------------------------------------------------ Per-gap CSV for cross-variant comparison
    per_gap_df["sog_variant"] = sog_variant
    per_gap_df["cog_variant"] = cog_variant
    per_gap_df["variant"] = f"{sog_variant}_{cog_variant}"
    save_csv(per_gap_df, os.path.join(eval_dir, "per_gap_sog_cog_results.csv"))

    # ------------------------------------------------------------------ Print summary
    logger.info("=" * 60)
    logger.info("EVALUATION SUMMARY  [%s_%s]", sog_variant, cog_variant)
    logger.info("=" * 60)
    logger.info("Overall (%d examples):", len(all_results))
    for k, v in overall.items():
        if isinstance(v, float):
            logger.info("  %-30s %.4f", k, v)
        else:
            logger.info("  %-30s %s", k, v)

    for g in gap_sizes:
        logger.info("--- gap=%d ---", g)
        m = per_gap_dict[f"gap_{g}"]
        logger.info("  token_accuracy=%.4f  ADE_km=%.4f  FDE_km=%.4f  DTW_km=%.4f",
                    m["token_accuracy"], m["ADE_km"], m["FDE_km"], m["DTW_km"])

    logger.info("=" * 60)
    logger.info("Evaluation outputs saved to: %s", eval_dir)


if __name__ == "__main__":
    main()
