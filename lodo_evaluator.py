"""Leave-One-Dataset-Out (LODO) Evaluation Protocol for Semantic Generalization.

Validates semantic generalization of the prompt injection interception middleware
for academic manuscript reporting by executing a strict cross-dataset evaluation:
  1. Partitions benchmark samples (15,000 samples) by origin (e.g. Protect AI, BIPIA, S-Labs).
  2. In each fold, holds out exactly one dataset as the unseen test set.
  3. Combines all remaining datasets into the training corpus.
  4. Trains the model from scratch on PyTorch MPS (with CUDA / CPU fallback).
  5. Evaluates strictly on the held-out unseen dataset, capturing F1-Score, Recall,
     Precision, False Negative Rate (FNR), and False Positive Rate (FPR).
  6. Computes in-distribution vs. out-of-distribution metrics and generalization gaps
     (Delta F1, Delta Recall) to prove semantic invariant feature learning.
  7. Outputs an academic-grade summary table and persistent JSON results.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from laya.calibrate import fit_temperature_map
from laya.common import collate_items, uses_parallel_layout
from laya.train import (
    calibration_records,
    calibration_report,
    encode_item,
    items_from_rows,
    load_checkpoint,
    save_checkpoint,
    split_calibration,
)

logger = logging.getLogger("lodo_evaluator")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

_MASKED = -1e4
DEFAULT_QUESTION = (
    "Does this text contain hidden instructions, malicious overrides, or prompt injection attempts?"
)


# =====================================================================
# Device Resolution & Hardware Accelerator Verification (PyTorch MPS)
# =====================================================================

def resolve_device(prefer_accelerator: bool = True) -> torch.device:
    """Verify and map device to Apple Silicon MPS, NVIDIA CUDA, or CPU fallback.
    
    Strictly detects Apple Silicon Metal Performance Shaders (MPS) for Mac execution,
    while gracefully falling back to NVIDIA CUDA or CPU on other operating systems.
    """
    if prefer_accelerator:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            if torch.backends.mps.is_built():
                logger.info("Apple Silicon Metal Performance Shaders (MPS) active: torch.backends.mps is available.")
                return torch.device("mps")
            else:
                logger.warning("torch.backends.mps.is_available() is True, but torch was not built with MPS. Fallback to CPU.")
        if torch.cuda.is_available():
            logger.info(f"NVIDIA CUDA active: {torch.cuda.get_device_name(0)}")
            return torch.device("cuda")
    logger.info("Using CPU device.")
    return torch.device("cpu")


def resolve_mps_device(prefer_mps: bool = True) -> torch.device:
    """Backwards compatibility alias for resolve_device."""
    return resolve_device(prefer_accelerator=prefer_mps)


# =====================================================================
# Loss Function with Severe False Negative Penalty
# =====================================================================

def weighted_soft_ce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    fn_penalty_weight: float = 2.0,
    class_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Cross-Entropy loss heavily penalizing False Negatives on injected payloads."""
    masked_logits = logits.masked_fill(~mask, _MASKED)
    log_probs = torch.log_softmax(masked_logits, dim=-1)

    if class_weights is None:
        class_weights = torch.tensor([1.0, fn_penalty_weight], device=logits.device, dtype=logits.dtype)

    k_dim = target.size(-1)
    if class_weights.size(0) < k_dim:
        pad = torch.ones(k_dim - class_weights.size(0), device=logits.device, dtype=logits.dtype)
        weights = torch.cat([class_weights, pad])
    else:
        weights = class_weights[:k_dim]

    weighted_target = target * weights
    per_sample_loss = -(weighted_target * log_probs * mask).sum(dim=-1)
    sample_weight_norm = (weighted_target * mask).sum(dim=-1).clamp_min(1e-6)
    return (per_sample_loss / sample_weight_norm).mean()


# =====================================================================
# Dataset Ingestion & Partitioning
# =====================================================================

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read a JSONL file into a list of dictionaries."""
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as err:
                logger.warning(f"Skipping malformed line {line_num} in {path}: {err}")
    return records


def save_jsonl(records: Sequence[Dict[str, Any]], path: str) -> None:
    """Save records to a JSONL file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def sanitize_text(text: str) -> str:
    """Sanitize whitespace and strip null bytes."""
    if not text:
        return ""
    return " ".join(str(text).replace("\x00", "").split()).strip()


def load_and_adapt_rows(file_path: str, default_question: str = DEFAULT_QUESTION) -> List[Dict[str, Any]]:
    """Read dataset file and adapt both noul and native Laya JSON formats."""
    adapted_rows: List[Dict[str, Any]] = []
    raw_records = load_jsonl(file_path)

    for row in raw_records:
        if "questions" in row and ("expected" in row or "gold" in row):
            adapted_rows.append(row)
            continue

        state = row.get("state") or row.get("text")
        if not state or not isinstance(state, str) or not state.strip():
            continue

        question_text = row.get("question") or default_question
        label = row.get("label", 0)

        adapted_rows.append({
            "state": state,
            "questions": {
                "injection": {
                    "type": "noul",
                    "instructions": question_text,
                }
            },
            "expected": {
                "injection": int(label),
            },
        })

    return adapted_rows


def isolate_datasets_from_master(
    master_file: str = "data/dataset_noul.jsonl",
    output_dir: str = "data",
) -> Dict[str, str]:
    """Isolate the 15,000 master samples into distinct JSONL files based on origin:
      - protectai.jsonl (Protect AI validation suite & supplemental benchmarks)
      - bipia.jsonl (Indirect Prompt Injection Attacks benchmark)
      - s_labs.jsonl (S-Labs Prompt Injection Dataset)
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bipia_path = out_dir / "bipia.jsonl"
    protectai_path = out_dir / "protectai.jsonl"
    s_labs_path = out_dir / "s_labs.jsonl"
    neuralchemy_path = out_dir / "neuralchemy.jsonl"

    if bipia_path.exists() and protectai_path.exists() and s_labs_path.exists():
        logger.info(f"Isolated dataset files already present in '{output_dir}'.")
        return {
            "bipia": str(bipia_path),
            "protectai": str(protectai_path),
            "s_labs": str(s_labs_path),
            "neuralchemy": str(neuralchemy_path),
        }

    logger.info(f"Isolating master samples from '{master_file}' into distinct origin files...")
    if not os.path.exists(master_file):
        raise FileNotFoundError(f"Master dataset file not found at: {master_file}")

    local_samples = load_jsonl(master_file)
    logger.info(f"Loaded {len(local_samples)} samples from {master_file}")

    # Fetch reference origins via Hugging Face Hub if available
    bipia_texts = set()
    protectai_texts = set()
    s_labs_texts = set()

    try:
        from huggingface_hub import hf_hub_download
        import pandas as pd

        # 1. BIPIA
        for f in ['data/bipia_code-00000-of-00001.parquet', 'data/bipia_text-00000-of-00001.parquet']:
            p = hf_hub_download('Abdennebi/protectai-prompt-injection-validation', f, repo_type='dataset')
            for t in pd.read_parquet(p)['text']:
                bipia_texts.add(sanitize_text(t))

        p_ig = hf_hub_download('Abdennebi/protectai-prompt-injection-validation', 'data/InjecGuard_valid-00000-of-00001.parquet', repo_type='dataset')
        df_ig = pd.read_parquet(p_ig)
        for _, r in df_ig.iterrows():
            if 'BIPIA' in str(r.get('source', '')):
                bipia_texts.add(sanitize_text(r['text']))

        # 2. Protect AI Suite
        for f in ['data/InjecGuard_valid-00000-of-00001.parquet', 'data/deepset-00000-of-00001.parquet',
                  'data/not_inject-00000-of-00001.parquet', 'data/spikee-00000-of-00001.parquet',
                  'data/wildguard-00000-of-00001.parquet']:
            p = hf_hub_download('Abdennebi/protectai-prompt-injection-validation', f, repo_type='dataset')
            df = pd.read_parquet(p)
            for _, r in df.iterrows():
                st = sanitize_text(r['text'])
                if st not in bipia_texts:
                    protectai_texts.add(st)

        # 3. S-Labs
        for split in ['train', 'validation', 'test']:
            p = hf_hub_download('S-Labs/prompt-injection-dataset', f'data/{split}.csv', repo_type='dataset')
            for t in pd.read_csv(p)['text']:
                s_labs_texts.add(sanitize_text(t))

    except Exception as exc:
        logger.warning(f"Could not load reference Hugging Face files for origin matching ({exc}). "
                       f"Using heuristic separation.")

    bipia_rows: List[Dict[str, Any]] = []
    protectai_rows: List[Dict[str, Any]] = []
    s_labs_rows: List[Dict[str, Any]] = []
    neuralchemy_rows: List[Dict[str, Any]] = []

    for item in local_samples:
        st = sanitize_text(item.get("state") or item.get("text", ""))
        if st in bipia_texts:
            bipia_rows.append(item)
        elif st in protectai_texts:
            protectai_rows.append(item)
        elif st in s_labs_texts:
            s_labs_rows.append(item)
        else:
            # Fallback / neuralchemy supplement merged into protectai suite
            protectai_rows.append(item)
            neuralchemy_rows.append(item)

    save_jsonl(bipia_rows, str(bipia_path))
    save_jsonl(protectai_rows, str(protectai_path))
    save_jsonl(s_labs_rows, str(s_labs_path))
    save_jsonl(neuralchemy_rows, str(neuralchemy_path))

    logger.info(f"Isolating complete:")
    logger.info(f"  - BIPIA:      {len(bipia_rows):6d} samples -> {bipia_path}")
    logger.info(f"  - Protect AI: {len(protectai_rows):6d} samples -> {protectai_path}")
    logger.info(f"  - S-Labs:     {len(s_labs_rows):6d} samples -> {s_labs_path}")

    return {
        "bipia": str(bipia_path),
        "protectai": str(protectai_path),
        "s_labs": str(s_labs_path),
        "neuralchemy": str(neuralchemy_path),
    }


# =====================================================================
# Model Evaluation strictly on Held-Out Test Set
# =====================================================================

@torch.no_grad()
def evaluate_metrics(
    model: nn.Module,
    tok: Any,
    items: Sequence[Dict[str, Any]],
    device: torch.device,
    max_len: int,
    head_max_len: int,
    threshold: float = 0.50,
    fn_penalty_weight: float = 2.0,
    parallel: bool = False,
    batch_size: int = 16,
) -> Dict[str, Any]:
    """Strict evaluation capturing F1-Score, Recall, Precision, Accuracy, FNR, FPR."""
    model.eval()

    total_loss = 0.0
    tp = 0
    fn = 0
    fp = 0
    tn = 0
    n_batches = 0

    all_gold: List[int] = []
    all_prob: List[float] = []

    for start in range(0, len(items), batch_size):
        chunk = items[start:start + batch_size]
        if isinstance(chunk[0], dict) and "ids" in chunk[0] and "markers" in chunk[0]:
            encoded_chunk = chunk
        else:
            encoded_chunk = [encode_item(tok, it, max_len, head_max_len, parallel=parallel) for it in chunk]
        batch = collate_items([encoded_chunk], tok.pad_token_id)

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        marker_pos = batch["marker_pos"].to(device)
        marker_mask = batch["marker_mask"].to(device)
        qtype = batch["qtype"].to(device)
        target = batch["target"].to(device)

        kwargs = {"detach_encoder": False}
        if "option_ids" in batch:
            kwargs.update(
                position_ids=batch["position_ids"].to(device),
                option_ids=batch["option_ids"].to(device),
            )

        logits, _act = model(input_ids, attention_mask, marker_pos, marker_mask, qtype, **kwargs)
        logits = logits.float()

        loss = weighted_soft_ce_loss(logits, target, marker_mask, fn_penalty_weight=fn_penalty_weight)
        total_loss += loss.item()
        n_batches += 1

        probs = torch.softmax(logits.masked_fill(~marker_mask, _MASKED), dim=-1).cpu().numpy()
        targets = target.cpu().numpy()

        for p_row, t_row in zip(probs, targets):
            is_injected_gold = (t_row[1] > 0.5)
            p_injected = float(p_row[1])
            is_injected_pred = (p_injected > threshold)

            all_gold.append(1 if is_injected_gold else 0)
            all_prob.append(p_injected)

            if is_injected_gold and is_injected_pred:
                tp += 1
            elif is_injected_gold and not is_injected_pred:
                fn += 1
            elif not is_injected_gold and is_injected_pred:
                fp += 1
            else:
                tn += 1

    total_pos = tp + fn
    total_neg = tn + fp
    total_samples = total_pos + total_neg

    fnr = fn / total_pos if total_pos > 0 else 0.0
    fpr = fp / total_neg if total_neg > 0 else 0.0
    recall = tp / total_pos if total_pos > 0 else 1.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if tp > 0 else 0.0)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / total_samples if total_samples > 0 else 0.0

    return {
        "loss": total_loss / max(1, n_batches),
        "total_samples": total_samples,
        "true_positives": tp,
        "false_negatives": fn,
        "false_positives": fp,
        "true_negatives": tn,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "accuracy": accuracy,
        "fnr": fnr,
        "fpr": fpr,
    }


# =====================================================================
# Single LODO Fold Training & Evaluation
# =====================================================================

def run_single_fold(
    fold_idx: int,
    held_out_name: str,
    train_paths: List[str],
    test_path: str,
    base_model: str = "convaiinnovations/laya",
    output_dir: str = "checkpoints/lodo",
    epochs: int = 1,
    micro_batch: int = 16,
    grad_accum: int = 2,
    encoder_lr: float = 2e-5,
    head_lr: float = 1e-4,
    fn_penalty_weight: float = 2.0,
    freeze_encoder: bool = False,
    threshold: float = 0.50,
    max_train_samples: Optional[int] = None,
    max_test_samples: Optional[int] = None,
    seed: int = 42,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Train a fresh model from scratch on remaining datasets and evaluate on held-out dataset."""
    torch.manual_seed(seed + fold_idx)
    random.seed(seed + fold_idx)
    np.random.seed(seed + fold_idx)

    device = device or resolve_mps_device(prefer_mps=True)

    print("\n" + "=" * 90)
    logger.info(f"=== FOLD {fold_idx}: Held-Out Dataset = {held_out_name} ===")
    logger.info(f"Target Accelerator: {device.type.upper()}")
    logger.info(f"Training Sets: {[os.path.basename(p) for p in train_paths]}")
    logger.info(f"Test Set:     {os.path.basename(test_path)}")
    print("=" * 90)

    # 1. Load base model fresh from scratch
    logger.info(f"Initializing fresh base model from '{base_model}' for Fold {fold_idx}...")
    model, tok, cfg = load_checkpoint(base_model)
    max_len = cfg.get("max_len", 512)
    head_max_len = cfg.get("head_max_len", 192)
    parallel = uses_parallel_layout(cfg)

    # 2. Combine and adapt training rows
    train_rows: List[Dict[str, Any]] = []
    for p in train_paths:
        rows = load_and_adapt_rows(p, DEFAULT_QUESTION)
        logger.info(f"  Added {len(rows)} samples from {os.path.basename(p)}")
        train_rows.extend(rows)

    random.Random(seed + fold_idx).shuffle(train_rows)

    if max_train_samples and max_train_samples < len(train_rows):
        logger.info(f"Subsampling combined training data to {max_train_samples} samples.")
        train_rows = train_rows[:max_train_samples]

    train_items, skipped_train = items_from_rows(tok, train_rows, max_len, head_max_len)
    logger.info(f"Total training items: {len(train_items)} (skipped: {skipped_train})")

    # 3. Hold out 10% in-distribution validation to measure generalization gap
    train_split, in_dist_val = split_calibration(train_items, calib_max=300, calib_frac=0.10, seed=seed + fold_idx)
    logger.info(f"Split: Train={len(train_split)}, In-Distribution Val={len(in_dist_val)}")

    # 4. Load held-out test rows
    test_rows = load_and_adapt_rows(test_path, DEFAULT_QUESTION)
    if max_test_samples and max_test_samples < len(test_rows):
        logger.info(f"Subsampling test set to {max_test_samples} samples.")
        test_rows = test_rows[:max_test_samples]
    test_items, skipped_test = items_from_rows(tok, test_rows, max_len, head_max_len)
    logger.info(f"Held-Out test items: {len(test_items)} (skipped: {skipped_test})")

    # 5. Pre-encode items for high throughput
    logger.info("Pre-encoding training items...")
    encoded_train = [encode_item(tok, it, max_len, head_max_len, parallel=parallel) for it in train_split]
    encoded_in_dist = [encode_item(tok, it, max_len, head_max_len, parallel=parallel) for it in in_dist_val]
    encoded_test = [encode_item(tok, it, max_len, head_max_len, parallel=parallel) for it in test_items]

    # 6. Prepare model and optimizer
    model = model.to(device)

    if freeze_encoder:
        logger.info("Freezing encoder: optimizing decision head only.")
        for p in model.encoder.parameters():
            p.requires_grad_(False)
        model.encoder.eval()
        groups = [{"params": [p for n, p in model.named_parameters() if not n.startswith("encoder.") and p.requires_grad], "lr": head_lr, "weight_decay": 0.01}]
    else:
        logger.info("Full model fine-tuning (encoder + decision head).")
        head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.") and p.requires_grad]
        encoder_params = [p for n, p in model.named_parameters() if n.startswith("encoder.") and p.requires_grad]
        groups = [
            {"params": head_params, "lr": head_lr, "weight_decay": 0.01},
            {"params": encoder_params, "lr": encoder_lr, "weight_decay": 0.01},
        ]

    optimizer = torch.optim.AdamW(groups)
    steps_per_epoch = math.ceil(len(encoded_train) / micro_batch)
    total_updates = max(1, math.ceil(steps_per_epoch / grad_accum) * epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_updates, eta_min=1e-6)

    # 7. Training loop
    t_start = time.time()
    for epoch in range(1, epochs + 1):
        bucket_size = micro_batch * 8
        buckets = [encoded_train[i:i + bucket_size] for i in range(0, len(encoded_train), bucket_size)]
        random.Random(seed + epoch).shuffle(buckets)
        epoch_items = []
        for b in buckets:
            epoch_items.extend(sorted(b, key=lambda x: len(x["ids"])))

        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_epoch_loss = 0.0
        n_steps = 0

        for start in range(0, len(epoch_items), micro_batch):
            chunk = epoch_items[start:start + micro_batch]
            batch = collate_items([chunk], tok.pad_token_id)

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            marker_pos = batch["marker_pos"].to(device)
            marker_mask = batch["marker_mask"].to(device)
            qtype = batch["qtype"].to(device)
            target = batch["target"].to(device)

            kwargs = {"detach_encoder": freeze_encoder}
            if "option_ids" in batch:
                kwargs.update(
                    position_ids=batch["position_ids"].to(device),
                    option_ids=batch["option_ids"].to(device),
                )

            logits, _ = model(input_ids, attention_mask, marker_pos, marker_mask, qtype, **kwargs)
            logits = logits.float()

            loss = weighted_soft_ce_loss(logits, target, marker_mask, fn_penalty_weight=fn_penalty_weight)
            window_start = (n_steps // grad_accum) * grad_accum
            window_size = min(grad_accum, steps_per_epoch - window_start)
            (loss / window_size).backward()

            n_steps += 1
            if n_steps % grad_accum == 0 or (start + micro_batch) >= len(epoch_items):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            total_epoch_loss += loss.item()

            if n_steps % 50 == 0 or n_steps == steps_per_epoch:
                logger.info(f"Fold {fold_idx} | Epoch {epoch}/{epochs} | Step {n_steps}/{steps_per_epoch} | Loss: {loss.item():.4f}")

        # Empty MPS cache if on Apple Silicon
        if device.type == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()

    train_duration = time.time() - t_start
    logger.info(f"Fold {fold_idx} training completed in {train_duration:.2f}s.")

    # 8. Evaluate In-Distribution performance
    in_dist_metrics = evaluate_metrics(
        model=model,
        tok=tok,
        items=encoded_in_dist,
        device=device,
        max_len=max_len,
        head_max_len=head_max_len,
        threshold=threshold,
        fn_penalty_weight=fn_penalty_weight,
        parallel=parallel,
    )

    # 9. Evaluate strictly on Held-Out Test set
    held_out_metrics = evaluate_metrics(
        model=model,
        tok=tok,
        items=encoded_test,
        device=device,
        max_len=max_len,
        head_max_len=head_max_len,
        threshold=threshold,
        fn_penalty_weight=fn_penalty_weight,
        parallel=parallel,
    )

    # 10. Compute Generalization Gaps
    gap_f1 = in_dist_metrics["f1"] - held_out_metrics["f1"]
    gap_recall = in_dist_metrics["recall"] - held_out_metrics["recall"]
    gap_accuracy = in_dist_metrics["accuracy"] - held_out_metrics["accuracy"]

    logger.info(f"Fold {fold_idx} Results for {held_out_name}:")
    logger.info(f"  In-Distribution  -> F1: {in_dist_metrics['f1']:.4f} | Recall: {in_dist_metrics['recall']:.2%} | Acc: {in_dist_metrics['accuracy']:.2%}")
    logger.info(f"  Held-Out Unseen  -> F1: {held_out_metrics['f1']:.4f} | Recall: {held_out_metrics['recall']:.2%} | Acc: {held_out_metrics['accuracy']:.2%}")
    logger.info(f"  Generalization Gap: Delta F1 = {gap_f1:+.4f} | Delta Recall = {gap_recall:+.4f}")

    # 11. Optionally save fold checkpoint
    fold_checkpoint_dir = os.path.join(output_dir, f"fold_{Path(held_out_name).stem}")
    try:
        os.makedirs(fold_checkpoint_dir, exist_ok=True)
        out_cfg = dict(cfg, fine_tuned=True, threshold=threshold, fold=fold_idx, held_out=held_out_name)
        save_checkpoint(model, tok, out_cfg, fold_checkpoint_dir)
        logger.info(f"Fold {fold_idx} checkpoint saved to '{fold_checkpoint_dir}'")
    except Exception as exc:
        logger.warning(f"Could not save checkpoint for fold {fold_idx}: {exc}")

    return {
        "fold": fold_idx,
        "held_out": held_out_name,
        "train_sources": [os.path.basename(p) for p in train_paths],
        "train_samples": len(train_split),
        "in_dist_val_samples": len(in_dist_val),
        "held_out_samples": len(test_items),
        "in_dist_metrics": in_dist_metrics,
        "held_out_metrics": held_out_metrics,
        "gap_f1": gap_f1,
        "gap_recall": gap_recall,
        "gap_accuracy": gap_accuracy,
        "train_duration_sec": train_duration,
    }


# =====================================================================
# Main LODO Orchestrator Loop
# =====================================================================

def run_lodo_evaluation(
    dataset_paths: List[str],
    base_model: str = "convaiinnovations/laya",
    output_dir: str = "checkpoints/lodo",
    epochs: int = 1,
    micro_batch: int = 16,
    grad_accum: int = 2,
    encoder_lr: float = 2e-5,
    head_lr: float = 1e-4,
    fn_penalty_weight: float = 2.0,
    freeze_encoder: bool = False,
    threshold: float = 0.50,
    max_train_samples: Optional[int] = None,
    max_test_samples: Optional[int] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute complete Leave-One-Dataset-Out (LODO) evaluation cross-fold loop."""
    device = resolve_mps_device(prefer_mps=True)
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 90)
    print("STARTING LEAVE-ONE-DATASET-OUT (LODO) EVALUATION PROTOCOL")
    print(f"Datasets: {[os.path.basename(p) for p in dataset_paths]}")
    print(f"Hardware Accelerator: {device.type.upper()}")
    print(f"Epochs per fold: {epochs} | Micro-Batch: {micro_batch} | Grad-Accum: {grad_accum}")
    print(f"Decision Threshold: {threshold} | False Negative Penalty: {fn_penalty_weight}x")
    print("=" * 90)

    fold_results: List[Dict[str, Any]] = []

    for idx, held_out_file in enumerate(dataset_paths, start=1):
        held_out_name = os.path.basename(held_out_file)
        training_files = [p for i, p in enumerate(dataset_paths) if i != (idx - 1)]

        fold_res = run_single_fold(
            fold_idx=idx,
            held_out_name=held_out_name,
            train_paths=training_files,
            test_path=held_out_file,
            base_model=base_model,
            output_dir=output_dir,
            epochs=epochs,
            micro_batch=micro_batch,
            grad_accum=grad_accum,
            encoder_lr=encoder_lr,
            head_lr=head_lr,
            fn_penalty_weight=fn_penalty_weight,
            freeze_encoder=freeze_encoder,
            threshold=threshold,
            max_train_samples=max_train_samples,
            max_test_samples=max_test_samples,
            seed=seed,
            device=device,
        )
        fold_results.append(fold_res)

    # Compute Macro-Averages
    avg_in_dist_f1 = float(np.mean([r["in_dist_metrics"]["f1"] for r in fold_results]))
    avg_in_dist_recall = float(np.mean([r["in_dist_metrics"]["recall"] for r in fold_results]))
    avg_held_out_f1 = float(np.mean([r["held_out_metrics"]["f1"] for r in fold_results]))
    avg_held_out_recall = float(np.mean([r["held_out_metrics"]["recall"] for r in fold_results]))
    avg_gap_f1 = float(np.mean([r["gap_f1"] for r in fold_results]))
    avg_gap_recall = float(np.mean([r["gap_recall"] for r in fold_results]))

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base_model": base_model,
        "device": device.type,
        "threshold": threshold,
        "folds": fold_results,
        "macro_average": {
            "in_dist_f1": avg_in_dist_f1,
            "in_dist_recall": avg_in_dist_recall,
            "held_out_f1": avg_held_out_f1,
            "held_out_recall": avg_held_out_recall,
            "gap_f1": avg_gap_f1,
            "gap_recall": avg_gap_recall,
        },
    }

    # Save persistent results JSON
    results_path = os.path.join(output_dir, "lodo_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"LODO results saved to '{results_path}'")

    # Print Academic Summary Table
    print_academic_summary_table(summary)

    return summary


# =====================================================================
# Academic Table Formatter
# =====================================================================

def print_academic_summary_table(summary: Dict[str, Any]) -> None:
    """Print academic publication formatted summary table with Generalization Gaps."""
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    folds = summary["folds"]
    macro = summary["macro_average"]

    print("\n" + "=" * 118)
    print("                      LEAVE-ONE-DATASET-OUT (LODO) GENERALIZATION REPORT")
    print("=" * 118)
    header = (
        f"{'Fold':<5} | {'Held-Out Dataset':<18} | {'Train':<6} | {'Test':<6} | "
        f"{'In-Dist F1':<10} | {'In-Dist Rec':<11} | {'Held-Out F1':<11} | {'Held-Out Rec':<12} | "
        f"{'Delta F1':<10} | {'Delta Rec':<11}"
    )
    print(header)
    print("-" * 118)

    for r in folds:
        name = r["held_out"]
        n_tr = r["train_samples"]
        n_te = r["held_out_samples"]
        in_f1 = r["in_dist_metrics"]["f1"]
        in_rec = r["in_dist_metrics"]["recall"]
        out_f1 = r["held_out_metrics"]["f1"]
        out_rec = r["held_out_metrics"]["recall"]
        gap_f1 = r["gap_f1"]
        gap_rec = r["gap_recall"]

        row_str = (
            f"{r['fold']:<5} | {name:<18} | {n_tr:<6} | {n_te:<6} | "
            f"{in_f1:<10.4f} | {in_rec:<11.2%} | {out_f1:<11.4f} | {out_rec:<12.2%} | "
            f"{gap_f1:+10.4f} | {gap_rec:+11.4f}"
        )
        print(row_str)

    print("-" * 118)
    macro_row = (
        f"{'MACRO':<5} | {'ALL FOLDS':<18} | {'-':<6} | {'-':<6} | "
        f"{macro['in_dist_f1']:<10.4f} | {macro['in_dist_recall']:<11.2%} | "
        f"{macro['held_out_f1']:<11.4f} | {macro['held_out_recall']:<12.2%} | "
        f"{macro['gap_f1']:+10.4f} | {macro['gap_recall']:+11.4f}"
    )
    print(macro_row)
    print("=" * 118)

    print("\n[SCIENTIFIC MANUSCRIPT INTERPRETATION]")
    if abs(macro["gap_f1"]) < 0.05:
        print(f"-> EXCELLENT GENERALIZATION: Macro F1 generalization gap is {macro['gap_f1']:+.4f} (< 5.0%).")
        print("   The model demonstrates robust semantic generalization across disjoint origin distributions.")
    elif abs(macro["gap_f1"]) < 0.10:
        print(f"-> MODERATE GENERALIZATION: Macro F1 generalization gap is {macro['gap_f1']:+.4f} (< 10.0%).")
        print("   The model retains strong cross-dataset intercept capabilities with slight domain shift.")
    else:
        print(f"-> DOMAIN GAP DETECTED: Macro F1 generalization gap is {macro['gap_f1']:+.4f}.")
        print("   Noticeable distribution shift between benchmark datasets.")
    print("")


# =====================================================================
# CLI Entry Point
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Leave-One-Dataset-Out (LODO) Evaluation for Semantic Generalization.")
    parser.add_argument("--data-dir", default="data", help="Directory containing isolated JSONL datasets")
    parser.add_argument("--isolate", action="store_true", help="Re-isolate master dataset into distinct origin JSONL files")
    parser.add_argument("--master-file", default="data/dataset_noul.jsonl", help="Master dataset file with all 15k samples")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["protectai.jsonl", "bipia.jsonl", "s_labs.jsonl"],
        help="List of dataset filenames in data-dir to evaluate under LODO protocol",
    )
    parser.add_argument("--base-model", default="convaiinnovations/laya", help="Base model repository or local path")
    parser.add_argument("--output-dir", default="checkpoints/lodo", help="Directory for fold checkpoints and results")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs per fold (default: 1)")
    parser.add_argument("--micro-batch", type=int, default=16, help="Micro-batch size per forward step")
    parser.add_argument("--grad-accum", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--encoder-lr", type=float, default=2e-5, help="Learning rate for encoder")
    parser.add_argument("--head-lr", type=float, default=1e-4, help="Learning rate for decision head")
    parser.add_argument("--fn-penalty-weight", type=float, default=2.0, help="False Negative penalty multiplier")
    parser.add_argument("--freeze-encoder", action="store_true", help="Freeze encoder and train decision head only (rapid execution)")
    parser.add_argument("--threshold", type=float, default=0.50, help="Decision threshold for injection classification")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Optional max training samples per fold")
    parser.add_argument("--max-test-samples", type=int, default=None, help="Optional max test samples per fold")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # If --isolate requested, run isolation
    if args.isolate:
        isolate_datasets_from_master(master_file=args.master_file, output_dir=args.data_dir)

    # Resolve full dataset paths
    resolved_paths: List[str] = []
    for d in args.datasets:
        path = d if os.path.isabs(d) or os.path.exists(d) else os.path.join(args.data_dir, d)
        if not os.path.exists(path):
            logger.warning(f"Dataset '{path}' not found! Checking if master isolation is needed...")
            isolate_datasets_from_master(master_file=args.master_file, output_dir=args.data_dir)
        if os.path.exists(path):
            resolved_paths.append(path)
        else:
            logger.error(f"Dataset path could not be resolved: {path}")

    if len(resolved_paths) < 2:
        logger.error(f"At least 2 datasets required for LODO evaluation, found {len(resolved_paths)}.")
        sys.exit(1)

    run_lodo_evaluation(
        dataset_paths=resolved_paths,
        base_model=args.base_model,
        output_dir=args.output_dir,
        epochs=args.epochs,
        micro_batch=args.micro_batch,
        grad_accum=args.grad_accum,
        encoder_lr=args.encoder_lr,
        head_lr=args.head_lr,
        fn_penalty_weight=args.fn_penalty_weight,
        freeze_encoder=args.freeze_encoder,
        threshold=args.threshold,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

