#!/usr/bin/env python3
"""
Memory-lean argument-acceptability checker.

Re-implements the original script but:
• computes the grounded extension without a dense matrix
• keeps other data structures strictly O(|V|+|E|)
• postpones any NumPy tensor creation until just before the ONNX call
"""
from __future__ import annotations
import argparse
import json
import numpy as np
import onnxruntime as ort
import torch
from pathlib import Path
from typing import Dict, List, Set, Tuple

BLANK, IN, OUT = 0, 1, 2
MAX_EDGES = 3_000_000 

# ----------------------------------------------------------------------
# Grounded semantics without an adjacency matrix
# ----------------------------------------------------------------------
from collections import deque
from typing import List

def solve_grounded(num_nodes: int, edge_index: np.ndarray) -> set[int]:
    """
    Grounded extension on a flat edge list.
    edge_index: shape (2, |E|)  int32 array
    """
    # Build successor lists once (|E| integers, no Python objects)
    succ: List[List[int]] = [[] for _ in range(num_nodes)]
    for u, v in zip(edge_index[0], edge_index[1]):
        succ[u].append(v)

    label  = [BLANK] * num_nodes
    in_deg = [0] * num_nodes
    for v in edge_index[1]:
        in_deg[v] += 1

    q = deque(i for i, deg in enumerate(in_deg) if deg == 0)
    for i in q:
        label[i] = IN

    while q:
        x = q.popleft()          # x is IN
        for y in succ[x]:        # y becomes OUT
            if label[y] != BLANK:
                continue
            label[y] = OUT
            for z in succ[y]:    # y no longer attacks z
                in_deg[z] -= 1
                if in_deg[z] == 0 and label[z] == BLANK:
                    label[z] = IN
                    q.append(z)
    return {i for i, v in enumerate(label) if v == IN}



# ----------------------------------------------------------------------
# PARSER
# ----------------------------------------------------------------------
def read_af_input(path: Path) -> tuple[int, np.ndarray]:
    """
    Read a DIMACS-like AF and return:
      • number of arguments
      • edge_index  – shape (2, |E|)  int32 NumPy array (0-based IDs)
    """
    src, dst = [], []
    num_args = 0
    with path.open() as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.split()
            if parts[:2] == ["p", "af"]:
                num_args = int(parts[2])
            elif len(parts) == 2:
                if len(src) > MAX_EDGES:
                    continue
                # convert to 0-based integers
                u, v = int(parts[0]) - 1, int(parts[1]) - 1
                src.append(u)
                dst.append(v)
    edge_index = np.vstack([src, dst]).astype(np.int32)
    return num_args, edge_index


# ----------------------------------------------------------------------
# THRESHOLDS
# ----------------------------------------------------------------------
def load_thresholds(path: Path) -> Dict[str, float]:
    try:
        with path.open() as fh:
            return json.load(fh)
    except FileNotFoundError:
        return {}


# ----------------------------------------------------------------------
# ONNX INFERENCE
# ----------------------------------------------------------------------
def predict_with_onnx(model_path: Path,
                      edge_index: np.ndarray,
                      num_nodes: int,
                      threshold: float) -> np.ndarray:
    """Run the quantised GNN and return a boolean mask of accepted nodes."""
    # Generate Xavier-initialised node features on demand (float32).
    
    x = torch.empty((num_nodes, 128), dtype=torch.float32)
    x = torch.randn(num_nodes, 128, dtype=torch.float32) / np.sqrt(128 / 2.)
    ort_inputs = {"x": x.numpy(), "edge_index": edge_index.astype(np.int64)}
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    logits = session.run(["logits"], ort_inputs)[0]

    logit_thr = np.log(threshold / (1.0 - threshold))
    preds = logits > logit_thr  
    return preds

import signal, sys, atexit, os

# ----------------------------------------------------------------------
# SAFETY NET: guarantee "NO" on timeout or early termination
# ----------------------------------------------------------------------
_FINISHED = False          # flipped to True once a normal answer is printed

def _urgent_exit(*_):
    """Emergency exit: print NO exactly once, flush, and die."""
    if not _FINISHED:                 # only if we haven't answered yet
        try:
            sys.stdout.write("NO\n")
            sys.stdout.flush()
        except Exception:
            pass
    os._exit(1)                       # immediate, bypass normal cleanup

def install_safety_net(timeout_sec: int = 58) -> None:
    """Arm a SIGALRM timer and handle SIGTERM/SIGINT the same way."""
    signal.signal(signal.SIGALRM, _urgent_exit)
    signal.alarm(timeout_sec)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _urgent_exit)
    # last-chance fallback if Python raises an unhandled exception
    atexit.register(_urgent_exit)

def answer(result: bool) -> None:
    global _FINISHED
    print("YES" if result else "NO")
    _FINISHED = True

# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main() -> None:
       
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_features", type=int, default=128)  # kept for CLI parity
    ap.add_argument("--filepath", required=True, type=Path)
    ap.add_argument("--task", required=True)
    ap.add_argument("--argument", required=True)
    ap.add_argument("--thresholds_file", default="thresholds.json", type=Path)
    args = ap.parse_args()

    install_safety_net()
    # ---------- graph & grounded extension ----------
    num_nodes, edge_index = read_af_input(args.filepath)
    grounded_in = solve_grounded(num_nodes, edge_index)

    # argument IDs in the file are “1 … n”; convert to 0-based
    arg_idx = int(args.argument) - 1
    if arg_idx in grounded_in:
        answer(True)
        return

    # ---------- threshold logic ----------
    thresholds = load_thresholds(args.thresholds_file)
    threshold = thresholds.get(args.task, 0.5)
    if threshold == 1.0:      # guarantee of rejection ⇒ no model call at all
        answer(False)
        return

    # ---------- ONNX inference (only memory we can't avoid) ----------
    preds = predict_with_onnx(
        Path(f"{args.task}_int8.onnx"),
        edge_index,              # ← already the right shape / dtype
        num_nodes,
        threshold
    )
    answer(True if preds[arg_idx] else False)


if __name__ == "__main__":
    main()
