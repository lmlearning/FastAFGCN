# FastAFGCN: Efficient ONNX Argumentation Solver

**Quantized graph-neural-network inference for approximate abstract argumentation.** FastAFGCN combines a grounded-extension check with ONNX model inference to answer argument-acceptance queries.

## How it works

1. Read an argumentation framework and compute grounded acceptance.
2. Resolve arguments accepted by the grounded check.
3. For remaining queries, apply the configured threshold and, where needed, a quantized ONNX model.

The repository includes `DC-CO`, `DC-SST`, `DC-ST` and `DS-ST` INT8 ONNX models. These are approximate predictions; the implementation and thresholds define the behaviour for each task.

## Getting started

The inference script imports NumPy, ONNX Runtime and PyTorch. In an environment with these dependencies, run from the repository root so model and threshold paths resolve:

```bash
python solver.py --help
python solver.py --filepath /path/to/framework.af --task DC-CO --argument 1
```

The input parser expects the ICCMA-style `p af N` format with numeric attack edges and argument IDs starting at 1. Supply your own framework file. The command prints `YES` or `NO`.

## Repository guide

- [solver.py](solver.py): input parsing, grounded reasoning and ONNX inference.
- [solver.sh](solver.sh): competition-style shell wrapper.
- [thresholds.json](thresholds.json): task-specific decision thresholds.
- [FastAFGCN.py](FastAFGCN.py): model implementation.
- [export_afgcn_to_onnx.py](export_afgcn_to_onnx.py) and [quantize_onnx.py](quantize_onnx.py): export and quantization utilities.

For the training and solver lineage, see [AFGCN](https://github.com/lmlearning/AFGCN). Runtime and accuracy depend on the model, framework and environment; this repository does not provide a universal performance guarantee.

## License

See [LICENSE](LICENSE).
