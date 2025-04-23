# quantize_onnx.py
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    model_input="afgcn.onnx",
    model_output="afgcn_int8.onnx",
    weight_type=QuantType.QInt8,
)
print("✅ Quantized to afgcn_int8.onnx")
