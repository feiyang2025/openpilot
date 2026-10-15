#!/usr/bin/env bash
set -euo pipefail

# AGX Orin CUDA backend deploy script for master-c3-cuda-test.
# Run ON the Orin after pulling the repo.
# Usage: ./deploy.sh [--fp16]
#
# Steps:
#   1. build libtrt_c_api.so from trt_c_api.cc   (TensorRT C ABI)
#   2. build libcuda_transform.so from cuda_transform.cu
#   3. generate metadata.pkl per model dir (get_model_metadata.py)
#   4. (optional) build the TRT .plan from onnx via trtexec
#
# Every artefact is produced here, on this machine. A .plan built elsewhere will
# not deserialize: TensorRT engines are version- and GPU-specific.

REPO_ROOT="$(cd "$(dirname "$0")/../../../../.." && pwd)"
OPENPILOT="$REPO_ROOT/openpilot"
MODELD="$OPENPILOT/selfdrive/modeld"
TRANSFORMS="$MODELD/transforms"
MODELS="$MODELD/models"
ARCH="${CUDA_ARCH:-sm_87}"
PRECISION="${1:---fp16}"
PY="${2:-${VENV_PYTHON:-python3}}"

echo "[deploy] repo=$REPO_ROOT arch=$ARCH precision=$PRECISION"

if ! command -v nvcc >/dev/null 2>&1; then
  echo "ERROR: nvcc not found; install the CUDA toolkit first" >&2
  exit 1
fi
if [ ! -e /usr/include/aarch64-linux-gnu/NvInfer.h ] && [ ! -e /usr/include/NvInfer.h ]; then
  echo "ERROR: NvInfer.h not found. Install the TensorRT dev package matching the runtime libnvinfer.so." >&2
  echo "       A plan and a libtrt_c_api.so built against a different TensorRT will not load." >&2
  exit 1
fi

# --- 1. TensorRT C ABI ---
echo "[deploy] building libtrt_c_api.so"
(cd "$TRANSFORMS" && \
  nvcc -arch="$ARCH" -shared -Xcompiler -fPIC -O2 \
       -o libtrt_c_api.so trt_c_api.cc -lnvinfer)
echo "[deploy] libtrt_c_api.so ok"

# --- 2. CUDA transform ---
echo "[deploy] building libcuda_transform.so"
(cd "$TRANSFORMS" && \
  nvcc -arch="$ARCH" -shared -Xcompiler -fPIC -O2 \
       -o libcuda_transform.so cuda_transform.cu -I.)
echo "[deploy] libcuda_transform.so ok"

# --- 3. metadata ---
# registry.py refuses to guess: a profile whose metadata is missing falls back to
# tinygrad instead of running with hardcoded output slices.
for model_dir in "$MODELS"/*/; do
  name="$(basename "$model_dir")"
  for onnx in "$model_dir"/*.onnx; do
    [ -e "$onnx" ] || continue
    echo "[deploy] metadata for $onnx"
    "$PY" "$MODELD/get_model_metadata.py" "$onnx" || {
      echo "[deploy] WARN metadata failed for $onnx - this profile will not load" >&2
    }
  done
done

# --- 4. engines ---
# The big model takes UINT8 images and concatenates them, which TensorRT 10.16
# refuses to import ("Found unsupported input type of UINT8").
# trt_fix_uint8_inputs.py rewrites those inputs to FLOAT16 and drops the Cast that
# followed; it is a dtype edit, not a numeric one. Models without UINT8 inputs are
# copied through untouched.
if command -v trtexec >/dev/null 2>&1; then
  for model_dir in "$MODELS"/*/; do
    name="$(basename "$model_dir")"
    onnx="$(ls "$model_dir"/*.onnx 2>/dev/null | head -1 || true)"
    [ -n "$onnx" ] || continue
    case "$onnx" in *.trt.onnx) ;; *)
      trt_onnx="$model_dir/$(basename "${onnx%.onnx}").trt.onnx"
      echo "[deploy] fixing UINT8 inputs: $(basename "$onnx")"
      "$PY" "$MODELD/trt_fix_uint8_inputs.py" "$onnx" "$trt_onnx" || true
      [ -f "$trt_onnx" ] && onnx="$trt_onnx"
      ;;
    esac
    out="$model_dir/driving_supercombo_fp16.plan"
    echo "[deploy] trtexec $name -> $(basename "$out")"
    trtexec --onnx="$onnx" --saveEngine="$out" $PRECISION --memPoolSize=workspace:4096 \
      || echo "[deploy] WARN trtexec failed for $onnx" >&2
  done
else
  echo "[deploy] trtexec not found; build the .plan engines separately"
fi

echo
echo "[deploy] done. Both .so files and any .plan must exist; verify with:"
echo "  ls -l $TRANSFORMS/libtrt_c_api.so $TRANSFORMS/libcuda_transform.so"
echo "  python3 -c \"from openpilot.sunnypilot.modeld_v2.gpu_backend.models.registry import REGISTRY;"
echo "print({k: (v.missing_reason() if v else 'unbuilt') for k, v in REGISTRY.items()})\""
echo
echo "Runtime switches:"
echo "  DISABLE_CUDA_BACKEND=1  force the stock tinygrad path"
echo "  OPENPILOT_CUDA_PATCH=0   disable the fork's runtime patches entirely"
