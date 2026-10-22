#!/usr/bin/env python3
"""Compile the driver-monitoring ONNX into the tinygrad pkl dmonitoringmodeld loads.

The pkl that ships with the Qualcomm tree is a QCOM/NPY build: on a PC/Jetson the
same graph silently runs on CPU (measured ~590 ms/frame here) because its captured
input device is not CUDA. Compiling it locally produces a pkl whose captured inputs
and kernels are on this device, keeping the model's own interface unchanged.

  python -m openpilot.selfdrive.modeld.compile_dmonitoring \
    --onnx openpilot/selfdrive/modeld/models/dmonitoring_model.onnx \
    --output openpilot/selfdrive/modeld/models/dmonitoring_model_cuda_tinygrad.pkl
"""
import argparse
import pickle
import time

from tinygrad import Device, Tensor, TinyJit
from tinygrad.nn.onnx import OnnxRunner

from openpilot.selfdrive.modeld.helpers import MODELS_DIR


def compile_dmonitoring(onnx_path: str, output: str, benchmark_runs: int = 5) -> None:
  device = Device.DEFAULT
  print(f"Compiling dmonitoring model on {device}: {onnx_path}")
  runner = OnnxRunner(str(onnx_path))
  inputs = runner.graph_inputs
  print("  inputs:", {k: (v.shape, v.dtype) for k, v in inputs.items()})

  # Named parameters: dmonitoringmodeld calls model_run(**tensor_inputs).
  def forward(input_img, calib):
    return runner({'input_img': input_img, 'calib': calib})['outputs'].realize()

  jit = TinyJit(forward, prune=True)
  tensors = {}
  for name, value in inputs.items():
    dtype = 'uint8' if str(value.dtype).endswith('uchar') else 'float32'
    tensors[name] = Tensor.zeros(value.shape, dtype=dtype, device=device).realize()

  # Compile before timing: the first call JITs the whole graph.
  jit(**tensors).realize()
  Device[device].synchronize()
  for i in range(benchmark_runs):
    start = time.perf_counter()
    jit(**tensors).realize()
    Device[device].synchronize()
    print(f"  run {i}: {(time.perf_counter() - start) * 1e3:.1f} ms")

  with open(output, 'wb') as f:
    pickle.dump(jit, f)
  print(f"  saved to {output}")


if __name__ == "__main__":
  p = argparse.ArgumentParser()
  p.add_argument('--onnx', default=str(MODELS_DIR / 'dmonitoring_model.onnx'))
  p.add_argument('--output', default=str(MODELS_DIR / 'dmonitoring_model_cuda_tinygrad.pkl'))
  p.add_argument('--benchmark-runs', type=int, default=5)
  args = p.parse_args()
  compile_dmonitoring(args.onnx, args.output, args.benchmark_runs)
