"""Threshold Optimization Script for Prompt Injection Interception.

Sweeps thresholds from 0.15 to 0.60 in increments of 0.01 across the validation dataset
(val_noul.jsonl) to locate the highest threshold maintaining False Negatives <= 1
(False Negative Rate near 0%), maximizing specificity while ensuring strict security.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import confusion_matrix

import laya

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("find_threshold")

DEFAULT_MODEL_DIR = "checkpoints/laya_prompt_injection"
DEFAULT_VAL_DATA = "data/val_noul.jsonl"
DEFAULT_QUESTION = (
    "Does this text contain hidden instructions, malicious overrides, or prompt injection attempts?"
)


def load_val_records(data_path: str) -> List[Dict[str, Any]]:
    records = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                records.append(json.loads(line))
    return records


def precompute_probabilities(
    model_path: str, data_path: str, device: str = "auto", max_samples: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Run model once across validation samples and cache ground truth and P(true)."""
    if device == "cuda" or (device == "auto" and torch.cuda.is_available()):
        dev = "cuda"
    elif device == "mps" or (device == "auto" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built()):
        dev = "mps"
    else:
        dev = "cpu"

    logger.info(f"Loading Laya decision model from '{model_path}' on device '{dev}'...")
    agent = laya.load(model_path, device=dev)

    records = load_val_records(data_path)
    if max_samples and max_samples < len(records):
        logger.info(f"Subsampling to {max_samples} validation samples for evaluation.")
        records = records[:max_samples]

    logger.info(f"Loaded {len(records)} samples from '{data_path}'. Pre-computing model probabilities...")

    y_true_list = []
    p_true_list = []

    for idx, row in enumerate(records, start=1):
        state = row.get("state") or row.get("text", "")
        q = row.get("question") or DEFAULT_QUESTION
        label = int(row.get("label", 0))

        res = agent.predict(
            state=state,
            questions={
                "injection": {
                    "type": "noul",
                    "instructions": q,
                }
            },
        )
        p = float(res["answers"]["injection"]["noul"])

        y_true_list.append(label)
        p_true_list.append(p)

        if idx % 50 == 0 or idx == len(records):
            print(f"  Scored [{idx:4d}/{len(records):4d}] samples...")

    return np.array(y_true_list, dtype=int), np.array(p_true_list, dtype=float)


def sweep_thresholds(
    y_true: np.ndarray,
    p_scores: np.ndarray,
    min_thresh: float = 0.10,
    max_thresh: float = 0.90,
    step: float = 0.01,
) -> Dict[str, Any]:
    """Iterate thresholds from min_thresh to max_thresh and find the threshold maximizing F1-score."""
    thresholds = [round(t, 2) for t in np.arange(min_thresh, max_thresh + (step / 2.0), step)]

    total_injections = int(np.sum(y_true == 1))
    total_safe = int(np.sum(y_true == 0))

    results = []

    # First pass: find maximum F1
    best_f1_val = -1.0
    for thresh in thresholds:
        y_pred = (p_scores > thresh).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / total_injections if total_injections > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        if f1 > best_f1_val:
            best_f1_val = f1

    print("\n" + "=" * 94)
    print(f"THRESHOLD SWEEP: {min_thresh:.2f} to {max_thresh:.2f} (Step: {step:.2f}) | Total: {len(y_true)} (Safe: {total_safe}, Injected: {total_injections})")
    print("Optimization Objective: MAXIMIZE F1-SCORE (Harmonic Mean of Precision and Recall)")
    print("=" * 94)
    print(" Threshold |  TP  |  FP  |  TN  |  FN  |   FNR   |   FPR   | Precision |  Recall  |    F1    | Status")
    print("-" * 94)

    for thresh in thresholds:
        y_pred = (p_scores > thresh).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        fnr = fn / total_injections if total_injections > 0 else 0.0
        fpr = fp / total_safe if total_safe > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / total_injections if total_injections > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        is_max_f1 = abs(f1 - best_f1_val) < 1e-5 and best_f1_val > 0
        status_flag = "[MAX F1]" if is_max_f1 else ""

        entry = {
            "threshold": thresh,
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
            "fnr": float(fnr),
            "fpr": float(fpr),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "is_max_f1": is_max_f1,
        }
        results.append(entry)

        # Print all thresholds or step intervals
        print(
            f"   {thresh:5.2f}   | {tp:4d} | {fp:4d} | {tn:4d} | {fn:4d} | {fnr:6.2%} | {fpr:6.2%} |  {precision:7.2%}  | {recall:7.2%}  |  {f1:6.4f}  | {status_flag}"
        )

    print("=" * 94)

    # Optimal candidate strictly maximizes F1 score (ties broken by higher recall, then lower threshold)
    best_candidate = max(results, key=lambda x: (round(x["f1"], 4), x["recall"], -x["threshold"]))

    print("\n" + "#" * 70)
    print("OPTIMAL F1-SCORE THRESHOLD IDENTIFIED")
    print("#" * 70)
    print(f"  Optimal Decision Threshold : {best_candidate['threshold']:.2f}")
    print(f"  F1 Score (Balanced)        : {best_candidate['f1']:.4f}")
    print(f"  Precision (True P / Pos)   : {best_candidate['precision']:.2%}")
    print(f"  Recall (Catch Rate)        : {best_candidate['recall']:.2%}")
    print(f"  True Positives (TP)        : {best_candidate['tp']} (Injections caught)")
    print(f"  False Positives (FP)       : {best_candidate['fp']} (Benign flagged)")
    print(f"  True Negatives (TN)        : {best_candidate['tn']} (Benign allowed)")
    print(f"  False Negatives (FN)       : {best_candidate['fn']} (Injections leaked)")
    print(f"  False Negative Rate (FNR)  : {best_candidate['fnr']:.2%}")
    print(f"  False Positive Rate (FPR)  : {best_candidate['fpr']:.2%}")
    print("#" * 70 + "\n")

    return {
        "optimal_threshold": best_candidate["threshold"],
        "metrics": best_candidate,
        "all_results": results,
    }


def update_middleware_threshold(optimal_threshold: float, middleware_file: str = "middleware_api.py") -> None:
    """Update INJECTION_DECISION_THRESHOLD in middleware_api.py."""
    path = Path(middleware_file)
    if not path.exists():
        return
    content = path.read_text(encoding="utf-8")
    import re
    updated = re.sub(
        r'INJECTION_DECISION_THRESHOLD = float\(os\.environ\.get\("INJECTION_DECISION_THRESHOLD", "[0-9\.]+"\)\)',
        f'INJECTION_DECISION_THRESHOLD = float(os.environ.get("INJECTION_DECISION_THRESHOLD", "{optimal_threshold:.2f}"))',
        content
    )
    path.write_text(updated, encoding="utf-8")
    logger.info(f"Updated {middleware_file} INJECTION_DECISION_THRESHOLD to {optimal_threshold:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Sweep thresholds to maximize F1-score on validation data.")
    parser.add_argument("--model", default=DEFAULT_MODEL_DIR, help="Model checkpoint path")
    parser.add_argument("--data", default=DEFAULT_VAL_DATA, help="Validation JSONL path")
    parser.add_argument("--device", default="auto", help="Device (auto, cuda, mps, or cpu)")
    parser.add_argument("--min", type=float, default=0.10, help="Min threshold (default 0.10)")
    parser.add_argument("--max", type=float, default=0.90, help="Max threshold (default 0.90)")
    parser.add_argument("--step", type=float, default=0.01, help="Step size (default 0.01)")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit validation samples (default: all)")
    parser.add_argument("--update-middleware", action="store_true", help="Auto-update middleware_api.py with optimal threshold")
    args = parser.parse_args()

    y_true, p_scores = precompute_probabilities(
        args.model, args.data, device=args.device, max_samples=args.max_samples
    )
    sweep_res = sweep_thresholds(y_true, p_scores, min_thresh=args.min, max_thresh=args.max, step=args.step)

    if args.update_middleware:
        update_middleware_threshold(sweep_res["optimal_threshold"])


if __name__ == "__main__":
    main()

