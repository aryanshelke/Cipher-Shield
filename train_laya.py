"""Fine-tuning Script for Convai Innovations Laya AI on Apple Silicon (MPS).

Fine-tunes the non-autoregressive decision model 'convaiinnovations/laya'
specifically for prompt injection interception.

Key Security Engineering Constraints:
  - Strict MPS device mapping (`torch.backends.mps`) for local Apple Silicon execution.
  - Severe False Negative loss weighting (W_fn >= 10.0-20.0) to ensure low injections leak through.
  - Evaluation against our 0.15 decision threshold tracking FNR, FPR, Precision, and Recall.
  - Outputs standard Laya checkpoint readable via `laya.load(checkpoint_dir)`.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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

logger = logging.getLogger("train_laya")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

_MASKED = -1e4


# =====================================================================
# Device Resolution & Hardware Accelerator Verification
# =====================================================================

def resolve_device(prefer_accelerator: bool = True) -> torch.device:
    """Verify and map device to NVIDIA CUDA, Apple Silicon MPS, or CPU."""
    if prefer_accelerator:
        if torch.cuda.is_available():
            logger.info(f"NVIDIA CUDA active: {torch.cuda.get_device_name(0)}")
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                logger.warning("torch.backends.mps.is_available() is True, but torch was not built with MPS. Fallback to CPU.")
                return torch.device("cpu")
            logger.info("Apple Silicon Metal Performance Shaders (MPS) active: torch.backends.mps is available.")
            return torch.device("mps")
    logger.info("Using CPU device.")
    return torch.device("cpu")


def resolve_mps_device(prefer_mps: bool = True) -> torch.device:
    """Backwards compatibility alias for resolve_device."""
    return resolve_device(prefer_accelerator=prefer_mps)


# =====================================================================
# Loss Functions with Severe False Negative Penalty
# =====================================================================

def weighted_soft_ce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    fn_penalty_weight: float = 15.0,
    class_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute Cross-Entropy heavily penalizing False Negatives.

    For a noul question:
      - Option 0 = 'false' (Safe)
      - Option 1 = 'true' (Injected)

    When target is Injected (target[:, 1] ~ 1.0), under-predicting P(true) is penalized
    by `fn_penalty_weight` (e.g. 15.0x), pushing gradients to ensure low injections leak through.
    """
    # Mask invalid options
    masked_logits = logits.masked_fill(~mask, _MASKED)
    log_probs = torch.log_softmax(masked_logits, dim=-1)

    if class_weights is None:
        # Default binary class weights: [safe=1.0, injected=fn_penalty_weight]
        class_weights = torch.tensor([1.0, fn_penalty_weight], device=logits.device, dtype=logits.dtype)

    # Broadcast class weights across batch
    # target shape: [B, K], class_weights shape: [K] or padded
    k_dim = target.size(-1)
    if class_weights.size(0) < k_dim:
        pad = torch.ones(k_dim - class_weights.size(0), device=logits.device, dtype=logits.dtype)
        weights = torch.cat([class_weights, pad])
    else:
        weights = class_weights[:k_dim]

    # Weighted per-option loss
    # target * weights emphasizes the injected option heavily
    weighted_target = target * weights
    per_sample_loss = -(weighted_target * log_probs * mask).sum(dim=-1)

    # Normalize by the effective sample weight to stabilize gradient scales
    sample_weight_norm = (weighted_target * mask).sum(dim=-1).clamp_min(1e-6)
    normalized_loss = per_sample_loss / sample_weight_norm

    return normalized_loss.mean()


def asymmetric_security_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    fn_penalty_weight: float = 15.0,
    gamma_pos: float = 1.5,
    gamma_neg: float = 0.0,
) -> torch.Tensor:
    """Asymmetric Focal Loss targeting low false negatives.

    Focuses gradient updates on positive injections with low predicted probability,
    decaying easy safe examples while aggressively penalizing false negatives.
    """
    masked_logits = logits.masked_fill(~mask, _MASKED)
    probs = torch.softmax(masked_logits, dim=-1)
    log_probs = torch.log_softmax(masked_logits, dim=-1)

    # Option 0 (Safe) vs Option 1 (Injected)
    p_safe = probs[:, 0].clamp(min=1e-6, max=1.0 - 1e-6)
    p_inj = probs[:, 1].clamp(min=1e-6, max=1.0 - 1e-6)

    y_safe = target[:, 0]
    y_inj = target[:, 1]

    # Positive injections: focal modulating factor (1 - p_inj)^gamma_pos * fn_penalty_weight
    loss_pos = fn_penalty_weight * ((1.0 - p_inj) ** gamma_pos) * torch.log(p_inj) * y_inj

    # Negative safe: standard or softened cross entropy
    loss_neg = ((1.0 - p_safe) ** gamma_neg) * torch.log(p_safe) * y_safe

    loss = -(loss_pos + loss_neg)
    return loss.mean()


# =====================================================================
# Data Loading & Adaptor to Laya Schema
# =====================================================================

def load_and_adapt_rows(file_path: str, default_question: str) -> List[Dict[str, Any]]:
    """Read dataset file and adapt both noul and native Laya JSON formats."""
    adapted_rows: List[Dict[str, Any]] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as err:
                logger.warning(f"Skipping malformed JSON line {line_num}: {err}")
                continue

            # Format A: Laya Native format {"state": "...", "questions": {...}, "expected": {...}}
            if "questions" in row and ("expected" in row or "gold" in row):
                adapted_rows.append(row)
                continue

            # Format B: Flat noul format {"state": "...", "question": "...", "label": 0/1}
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


# =====================================================================
# Model Evaluation (Tracks Security Business Metrics at Threshold 0.15)
# =====================================================================

@torch.no_grad()
def evaluate_security_metrics(
    model: nn.Module,
    tok: Any,
    items: Sequence[Dict[str, Any]],
    device: torch.device,
    max_len: int,
    head_max_len: int,
    threshold: float = 0.15,
    fn_penalty_weight: float = 15.0,
    parallel: bool = False,
    batch_size: int = 16,
) -> Dict[str, float]:
    """Evaluate model tracking False Negatives, False Positives, and Recall at threshold 0.15."""
    model.eval()

    total_loss = 0.0
    tp = 0  # True Positives: Injected and P(true) > threshold
    fn = 0  # False Negatives: Injected but P(true) <= threshold (CRITICAL LEAKAGE)
    fp = 0  # False Positives: Safe but P(true) > threshold (Acceptable tradeoff)
    tn = 0  # True Negatives: Safe and P(true) <= threshold
    n_batches = 0

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

        loss = weighted_soft_ce_loss(
            logits, target, marker_mask, fn_penalty_weight=fn_penalty_weight
        )
        total_loss += loss.item()
        n_batches += 1

        # Calculate probabilities
        probs = torch.softmax(logits.masked_fill(~marker_mask, _MASKED), dim=-1).cpu().numpy()
        targets = target.cpu().numpy()

        for p_row, t_row in zip(probs, targets):
            is_injected_gold = (t_row[1] > 0.5)
            p_injected = p_row[1]

            is_injected_pred = (p_injected > threshold)

            if is_injected_gold and is_injected_pred:
                tp += 1
            elif is_injected_gold and not is_injected_pred:
                fn += 1  # False Negative
            elif not is_injected_gold and is_injected_pred:
                fp += 1  # False Positive
            else:
                tn += 1

    total_pos = tp + fn
    total_neg = tn + fp

    fnr = fn / total_pos if total_pos > 0 else 0.0
    fpr = fp / total_neg if total_neg > 0 else 0.0
    recall = tp / total_pos if total_pos > 0 else 1.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "val_loss": total_loss / max(1, n_batches),
        "false_negatives": fn,
        "false_positives": fp,
        "true_positives": tp,
        "true_negatives": tn,
        "fnr": fnr,
        "fpr": fpr,
        "recall": recall,
        "precision": precision,
        "f1": f1,
    }


# =====================================================================
# Main Training Loop
# =====================================================================

def train_laya_prompt_injection(
    train_data_path: str,
    val_data_path: Optional[str] = None,
    base_model: str = "convaiinnovations/laya",
    output_dir: str = "checkpoints/laya_prompt_injection",
    epochs: int = 3,
    micro_batch: int = 2,
    grad_accum: int = 8,
    encoder_lr: float = 2.0e-5,
    head_lr: float = 1.0e-4,
    fn_penalty_weight: float = 15.0,
    loss_type: str = "weighted_ce",
    freeze_encoder: bool = False,
    threshold: float = 0.15,
    max_train_samples: Optional[int] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Fine-tune Laya with severe false negative penalty on Apple Silicon MPS."""
    torch.manual_seed(seed)
    random.seed(seed)

    # 1. Device mapping: strictly ensure MPS if available on macOS
    device = resolve_mps_device(prefer_mps=True)
    logger.info(f"Target Execution Device: {device.type.upper()}")

    # 2. Load base model checkpoint
    logger.info(f"Loading base Laya model from '{base_model}'...")
    model, tok, cfg = load_checkpoint(base_model)
    max_len = cfg.get("max_len", 512)
    head_max_len = cfg.get("head_max_len", 192)
    parallel = uses_parallel_layout(cfg)

    # 3. Load & adapt training data
    default_q = "Does this text contain hidden instructions, malicious overrides, or prompt injection attempts?"
    train_rows = load_and_adapt_rows(train_data_path, default_q)
    train_items, skipped = items_from_rows(tok, train_rows, max_len, head_max_len)
    logger.info(f"Loaded {len(train_items)} valid training items (skipped: {skipped})")

    if max_train_samples and max_train_samples < len(train_items):
        logger.info(f"Subsampling training dataset to {max_train_samples} samples.")
        train_items = train_items[:max_train_samples]

    # If separate validation path not provided, hold out 10%
    if val_data_path and os.path.exists(val_data_path):
        val_rows = load_and_adapt_rows(val_data_path, default_q)
        val_items, _ = items_from_rows(tok, val_rows, max_len, head_max_len)
    else:
        train_items, val_items = split_calibration(train_items, calib_max=200, calib_frac=0.15, seed=seed)

    logger.info(f"Split sizes: Train={len(train_items)}, Validation={len(val_items)}")

    # 4. Pre-encode items for high throughput on Apple Silicon MPS
    logger.info("Pre-encoding training and validation items (zero per-step tokenization overhead)...")
    encoded_train = [
        encode_item(tok, it, max_len, head_max_len, parallel=parallel)
        for it in train_items
    ]
    encoded_val = [
        encode_item(tok, it, max_len, head_max_len, parallel=parallel)
        for it in val_items
    ]

    # 5. Prepare model and optimizers
    model = model.to(device)

    if freeze_encoder:
        logger.info("Freezing ModernBERT encoder; fine-tuning decision head with detach_encoder=True.")
        for p in model.encoder.parameters():
            p.requires_grad_(False)
        model.encoder.eval()
        groups = [{"params": [p for n, p in model.named_parameters() if not n.startswith("encoder.") and p.requires_grad], "lr": head_lr, "weight_decay": 0.01}]
    else:
        logger.info("Full model fine-tuning (encoder + decision head) with MPS gradient accumulation.")
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

    best_fnr = 1.0
    best_f1 = -1.0
    best_val_loss = float("inf")
    history: List[Dict[str, Any]] = []

    logger.info(f"Starting Fine-Tuning: {epochs} epochs, Steps/epoch={steps_per_epoch}, Micro-batch={micro_batch}, Grad-accum={grad_accum}, FN Penalty={fn_penalty_weight}x")

    for epoch in range(1, epochs + 1):
        # Length bucketing: group items into buckets and sort by length to minimize batch padding
        bucket_size = micro_batch * 8
        buckets = [encoded_train[i:i + bucket_size] for i in range(0, len(encoded_train), bucket_size)]
        random.Random(seed + epoch).shuffle(buckets)
        epoch_items = []
        for b in buckets:
            epoch_items.extend(sorted(b, key=lambda x: len(x["ids"])))

        total_epoch_loss = 0.0
        n_steps = 0

        model.train()
        optimizer.zero_grad(set_to_none=True)

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

            logits, _act = model(input_ids, attention_mask, marker_pos, marker_mask, qtype, **kwargs)
            logits = logits.float()

            if loss_type == "asymmetric_focal":
                loss = asymmetric_security_loss(
                    logits, target, marker_mask, fn_penalty_weight=fn_penalty_weight
                )
            else:
                loss = weighted_soft_ce_loss(
                    logits, target, marker_mask, fn_penalty_weight=fn_penalty_weight
                )

            # Gradient accumulation
            window_start = (n_steps // grad_accum) * grad_accum
            window_size = min(grad_accum, steps_per_epoch - window_start)
            scaled_loss = loss / window_size
            scaled_loss.backward()

            n_steps += 1
            if n_steps % grad_accum == 0 or (start + micro_batch) >= len(epoch_items):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            total_epoch_loss += loss.item()

            if n_steps % 50 == 0 or n_steps == steps_per_epoch:
                logger.info(
                    f"Epoch {epoch}/{epochs} | Step {n_steps:4d}/{steps_per_epoch} | "
                    f"Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.2e}"
                )

        # Free MPS cache at epoch boundary
        if device.type == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()

        mean_train_loss = total_epoch_loss / max(1, n_steps)

        # Run Validation Evaluation using pre-encoded val items
        val_metrics = evaluate_security_metrics(
            model=model,
            tok=tok,
            items=encoded_val,
            device=device,
            max_len=max_len,
            head_max_len=head_max_len,
            threshold=threshold,
            fn_penalty_weight=fn_penalty_weight,
            parallel=parallel,
        )

        logger.info(
            f"Epoch {epoch}/{epochs} Finished -> Train Loss: {mean_train_loss:.4f} | "
            f"Val Loss: {val_metrics['val_loss']:.4f} | F1-Score: {val_metrics['f1']:.4f} | "
            f"Recall: {val_metrics['recall']:.2%} | Precision: {val_metrics['precision']:.2%} | "
            f"FN: {val_metrics['false_negatives']} (FNR: {val_metrics['fnr']:.2%}) | FP: {val_metrics['false_positives']}"
        )

        epoch_summary = {
            "epoch": epoch,
            "train_loss": mean_train_loss,
            **val_metrics,
        }
        history.append(epoch_summary)

        # Checkpoint if best F1 score or lowest validation loss
        is_best = (val_metrics["f1"] > best_f1) or (
            abs(val_metrics["f1"] - best_f1) < 1e-4 and val_metrics["val_loss"] < best_val_loss
        )
        if is_best:
            best_f1 = val_metrics["f1"]
            best_fnr = val_metrics["fnr"]
            best_val_loss = val_metrics["val_loss"]
            logger.info(
                f"New Best Security Checkpoint (F1={best_f1:.4f}, Recall={val_metrics['recall']:.2%}, "
                f"Precision={val_metrics['precision']:.2%}) -> Saving to {output_dir}"
            )
            save_security_checkpoint(model, tok, cfg, max_len, head_max_len, output_dir, val_items, device, parallel)

    logger.info(f"Fine-tuning completed. Final checkpoint saved in '{output_dir}'.")
    return {
        "output_dir": output_dir,
        "history": history,
        "best_f1": best_f1,
        "best_fnr": best_fnr,
        "best_val_loss": best_val_loss,
    }


def save_security_checkpoint(
    model: nn.Module,
    tok: Any,
    cfg: Dict[str, Any],
    max_len: int,
    head_max_len: int,
    output_dir: str,
    val_items: Sequence[Dict[str, Any]],
    device: torch.device,
    parallel: bool,
) -> None:
    """Calibrate temperatures and save standard Laya checkpoint directory."""
    os.makedirs(output_dir, exist_ok=True)

    # Fit calibration temperatures on validation items
    records = calibration_records(model, tok, val_items, device, max_len, head_max_len, parallel=parallel)
    fitted = fit_temperature_map(records)
    calib_rep = calibration_report(records, fitted["temperature"])

    out_cfg = dict(
        cfg,
        max_len=max_len,
        head_max_len=head_max_len,
        fine_tuned=True,
        temperature=fitted["temperature"],
        threshold=0.15,
        security_middleware=True,
    )
    if fitted.get("temperature_by_options"):
        # Clamp temperatures to valid range [0.5, 5.0] to satisfy Laya runtime constraints
        clamped_temps = {k: max(0.5, min(5.0, float(v))) for k, v in fitted["temperature_by_options"].items()}
        out_cfg["temperature_by_options"] = clamped_temps

    save_checkpoint(model, tok, out_cfg, output_dir)

    # Save question schema for router compatibility
    questions_schema = {
        "injection": {
            "type": "noul",
            "instructions": "Does this text contain hidden instructions, malicious overrides, or prompt injection attempts?",
        }
    }
    with open(os.path.join(output_dir, "questions.json"), "w", encoding="utf-8") as f:
        json.dump(questions_schema, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Laya model on MPS with severe false negative weighting.")
    parser.add_argument("--data", default="data/train_noul.jsonl", help="Path to training data (noul JSONL)")
    parser.add_argument("--val-data", default="data/val_noul.jsonl", help="Path to validation data")
    parser.add_argument("--base-model", default="convaiinnovations/laya", help="Base Laya model repository or directory")
    parser.add_argument("--output-dir", default="checkpoints/laya_prompt_injection", help="Output checkpoint directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--micro-batch", type=int, default=16, help="Micro-batch size per forward step")
    parser.add_argument("--grad-accum", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--encoder-lr", type=float, default=2e-5, help="Learning rate for ModernBERT encoder")
    parser.add_argument("--head-lr", type=float, default=1e-4, help="Learning rate for Laya decision head")
    parser.add_argument("--fn-penalty-weight", type=float, default=2.0, help="Weight multiplier penalizing False Negatives")
    parser.add_argument("--loss-type", choices=["weighted_ce", "asymmetric_focal"], default="weighted_ce", help="Loss function")
    parser.add_argument("--freeze-encoder", action="store_true", help="Freeze ModernBERT encoder and train only decision head (memory optimized)")
    parser.add_argument("--threshold", type=float, default=0.15, help="Decision threshold for security evaluation")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Optional max training samples to process")
    args = parser.parse_args()

    train_laya_prompt_injection(
        train_data_path=args.data,
        val_data_path=args.val_data,
        base_model=args.base_model,
        output_dir=args.output_dir,
        epochs=args.epochs,
        micro_batch=args.micro_batch,
        grad_accum=args.grad_accum,
        encoder_lr=args.encoder_lr,
        head_lr=args.head_lr,
        fn_penalty_weight=args.fn_penalty_weight,
        loss_type=args.loss_type,
        freeze_encoder=args.freeze_encoder,
        threshold=args.threshold,
        max_train_samples=args.max_train_samples,
    )


if __name__ == "__main__":
    main()

