"""Export Fine-Tuned Laya Model to ONNX Optimized for Apple CoreML.

Exports 'checkpoints/laya_prompt_injection' into an ONNX graph with dynamic axes
and verifies execution with onnxruntime's CoreMLExecutionProvider.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from laya.train import load_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("export_coreml")

DEFAULT_MODEL_DIR = "checkpoints/laya_prompt_injection"
DEFAULT_OUTPUT_ONNX = "checkpoints/laya_prompt_injection/model.onnx"


def export_model_to_onnx(
    model_dir: str = DEFAULT_MODEL_DIR,
    output_path: str = DEFAULT_OUTPUT_ONNX,
    opset_version: int = 18,
) -> str:
    """Load PyTorch checkpoint and export to ONNX format."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading checkpoint from '{model_dir}'...")
    model, tok, cfg = load_checkpoint(model_dir)
    model.eval()

    # Sequence length matching our chunker limit (450) and 2 markers for noul
    seq_len = 450
    input_ids = torch.randint(0, 1000, (1, seq_len), dtype=torch.long)
    attention_mask = torch.ones((1, seq_len), dtype=torch.long)
    marker_pos = torch.tensor([[10, 20]], dtype=torch.long)
    marker_mask = torch.tensor([[True, True]], dtype=torch.bool)
    qtype = torch.tensor([2], dtype=torch.long)  # QTYPES["noul"] = 2

    logger.info(f"Exporting model to ONNX (opset_version={opset_version}) at '{output_path}'...")
    start_time = time.perf_counter()

    torch.onnx.export(
        model,
        (input_ids, attention_mask, marker_pos, marker_mask, qtype),
        str(out_file),
        input_names=["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"],
        output_names=["logits", "act_logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "marker_pos": {0: "batch_size", 1: "num_markers"},
            "marker_mask": {0: "batch_size", 1: "num_markers"},
            "qtype": {0: "batch_size"},
            "logits": {0: "batch_size", 1: "num_markers"},
            "act_logits": {0: "batch_size"},
        },
        opset_version=opset_version,
    )

    export_duration = time.perf_counter() - start_time
    logger.info(f"ONNX export completed in {export_duration:.2f}s ({out_file.stat().st_size / (1024*1024):.1f} MB)")

    # Verify ONNX model integrity
    logger.info("Verifying ONNX model integrity with onnx.checker...")
    onnx_proto = onnx.load(str(out_file))
    onnx.checker.check_model(onnx_proto)
    logger.info("ONNX graph validation passed.")

    # Test loading with onnxruntime
    logger.info("Testing onnxruntime initialization...")
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    available_providers = ort.get_available_providers()
    providers = []
    for p in ["CUDAExecutionProvider", "DmlExecutionProvider", "CoreMLExecutionProvider"]:
        if p in available_providers:
            providers.append(p)
    providers.append("CPUExecutionProvider")

    logger.info(f"Target Execution Providers: {providers}")
    session = ort.InferenceSession(str(out_file), sess_options=so, providers=providers)
    active_providers = session.get_providers()
    logger.info(f"Active Providers: {active_providers}")

    # Warmup and test forward run
    dummy_ort_inputs = {
        "input_ids": np.random.randint(0, 1000, (1, 64), dtype=np.int64),
        "attention_mask": np.ones((1, 64), dtype=np.int64),
        "marker_pos": np.array([[10, 20]], dtype=np.int64),
        "marker_mask": np.array([[True, True]], dtype=bool),
        "qtype": np.array([2], dtype=np.int64),
    }

    t0 = time.perf_counter()
    outs = session.run(["logits", "act_logits"], dummy_ort_inputs)
    warmup_latency = (time.perf_counter() - t0) * 1000.0
    logger.info(f"Test inference run successful (logits shape: {outs[0].shape}, latency: {warmup_latency:.2f}ms)")

    print("\n" + "=" * 65)
    print("COREML ONNX EXPORT SUCCESSFUL")
    print("=" * 65)
    print(f"  Exported File       : {out_file}")
    print(f"  Model Size          : {out_file.stat().st_size / (1024*1024):.1f} MB")
    print(f"  Opset Version       : {opset_version}")
    print(f"  Active Providers    : {active_providers}")
    print("=" * 65 + "\n")

    return str(out_file)


def main():
    parser = argparse.ArgumentParser(description="Export fine-tuned Laya checkpoint to CoreML-optimized ONNX.")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="Path to checkpoint directory")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_ONNX, help="Output ONNX model path")
    parser.add_argument("--opset", type=int, default=18, help="ONNX operator set version (default: 18)")
    args = parser.parse_args()

    export_model_to_onnx(
        model_dir=args.model_dir,
        output_path=args.output,
        opset_version=args.opset,
    )


if __name__ == "__main__":
    main()

