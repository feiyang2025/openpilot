import ctypes
import os
from pathlib import Path

import numpy as np


# TensorRT nvinferDataType_t:
#   0 FLOAT, 1 HALF, 2 INT8, 3 INT32, 4 BOOL, 5 UINT8, 6 FP8, 7 BF16
# 7 is BF16, not HALF. Mapping it to float16 silently reinterprets the buffer.
_TRT_DTYPE_NAMES = {
  0: "float32", 1: "float16", 2: "int8", 3: "int32",
  4: "bool", 5: "uint8", 6: "float8_e4m3fn", 7: "bfloat16",
}

_TRT_DTYPE_TO_NP = {
  0: np.float32,
  1: np.float16,
  2: np.int8,
  3: np.int32,
  4: np.bool_,
  5: np.uint8,
  6: None,   # fp8: TensorRT-specific, no numpy equivalent
}


def _bfloat16_dtype():
  """numpy has no bfloat16; ml_dtypes provides one.

  Returns None when neither is available. Callers must treat that as an error
  rather than falling back to float16 - the two have different exponents, so the
  reinterpretation would produce wrong numbers with no failure anywhere.
  """
  dtype = getattr(np, "bfloat16", None)
  if dtype is not None:
    return np.dtype(dtype)
  try:
    import ml_dtypes
    return np.dtype(ml_dtypes.bfloat16)
  except Exception:
    return None


def _np_dtype(trt_dtype: int) -> np.dtype:
  if trt_dtype not in _TRT_DTYPE_TO_NP and trt_dtype != 7:
    raise RuntimeError(f"unsupported TensorRT dtype {trt_dtype} ({_TRT_DTYPE_NAMES.get(trt_dtype, 'unknown')})")
  if trt_dtype == 7:
    dtype = _bfloat16_dtype()
    if dtype is None:
      raise RuntimeError(
        f"engine declares bfloat16 but neither numpy {np.__version__} nor ml_dtypes provides it; "
        f"reading it as float16 would reinterpret the buffer and produce wrong numbers silently. "
        f"Install ml_dtypes, or rebuild the engine for fp16."
      )
    return dtype
  dtype = _TRT_DTYPE_TO_NP[trt_dtype]
  if dtype is None:
    raise RuntimeError(
      f"engine declares {_TRT_DTYPE_NAMES[trt_dtype]}, which has no numpy equivalent; "
      f"cast it in the ONNX graph before building the plan"
    )
  return np.dtype(dtype)


class _CudaDriver:
  def __init__(self):
    self.lib = ctypes.CDLL("libcuda.so.1")
    for fn, argt, rest in [
      ("cuInit", [ctypes.c_uint], ctypes.c_int),
      ("cuCtxGetCurrent", [ctypes.POINTER(ctypes.c_ulonglong)], ctypes.c_int),
      ("cuCtxSetCurrent", [ctypes.c_ulonglong], ctypes.c_int),
      ("cuMemAlloc_v2", [ctypes.POINTER(ctypes.c_ulonglong), ctypes.c_size_t], ctypes.c_int),
      ("cuMemFree_v2", [ctypes.c_ulonglong], ctypes.c_int),
      ("cuMemcpyHtoDAsync_v2", [ctypes.c_ulonglong, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_ulonglong], ctypes.c_int),
      ("cuMemcpyDtoHAsync_v2", [ctypes.c_void_p, ctypes.c_ulonglong, ctypes.c_size_t, ctypes.c_ulonglong], ctypes.c_int),
      ("cuStreamCreate", [ctypes.POINTER(ctypes.c_ulonglong), ctypes.c_uint], ctypes.c_int),
      ("cuStreamDestroy_v2", [ctypes.c_ulonglong], ctypes.c_int),
      ("cuStreamSynchronize", [ctypes.c_ulonglong], ctypes.c_int),
    ]:
      f = getattr(self.lib, fn)
      f.argtypes = argt
      f.restype = rest
      setattr(self, fn, f)


def _check(ret: int):
  if ret != 0:
    raise RuntimeError(f"CUDA driver error {ret}")


class TrtRunner:
  def __init__(self, engine_path: str, trt_c_api: str | None = None):
    self.engine_path = engine_path
    api = trt_c_api or os.getenv("TRT_C_API_LIBRARY")
    if api is None:
      api = str(Path(__file__).resolve().parents[3] / "selfdrive" / "modeld" / "transforms" / "libtrt_c_api.so")
    if not Path(api).exists():
      raise RuntimeError(
        f"libtrt_c_api.so not found at {api}; build it on-device with "
        f"`nvcc -arch=sm_87 -shared -Xcompiler -fPIC -O2 -o libtrt_c_api.so trt_c_api.cc -lnvinfer` "
        f"or set TRT_C_API_LIBRARY"
      )
    self._cuda = _CudaDriver()
    _check(self._cuda.cuInit(0))
    ctx = ctypes.c_ulonglong()
    ret = self._cuda.cuCtxGetCurrent(ctypes.byref(ctx))
    if ret != 0 or ctx.value == 0:
      raise RuntimeError("No active CUDA context; initialize tinygrad Device['CUDA'] first")
    self._ctx = ctx.value
    self._lib = ctypes.CDLL(str(api))
    self._lib.trt_engine_create.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    self._lib.trt_engine_create.restype = ctypes.c_void_p
    self._lib.trt_engine_destroy.argtypes = [ctypes.c_void_p]
    self._lib.trt_engine_execute.argtypes = [
      ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulonglong), ctypes.c_ulonglong, ctypes.c_ulonglong,
    ]
    self._lib.trt_engine_execute.restype = ctypes.c_int
    self._lib.trt_engine_num_inputs.argtypes = [ctypes.c_void_p]
    self._lib.trt_engine_num_inputs.restype = ctypes.c_int
    self._lib.trt_engine_num_outputs.argtypes = [ctypes.c_void_p]
    self._lib.trt_engine_num_outputs.restype = ctypes.c_int
    self._lib.trt_engine_input_name.argtypes = [ctypes.c_void_p, ctypes.c_int]
    self._lib.trt_engine_input_name.restype = ctypes.c_char_p
    self._lib.trt_engine_output_name.argtypes = [ctypes.c_void_p, ctypes.c_int]
    self._lib.trt_engine_output_name.restype = ctypes.c_char_p
    self._lib.trt_engine_input_dtype.argtypes = [ctypes.c_void_p, ctypes.c_int]
    self._lib.trt_engine_input_dtype.restype = ctypes.c_int
    self._lib.trt_engine_output_dtype.argtypes = [ctypes.c_void_p, ctypes.c_int]
    self._lib.trt_engine_output_dtype.restype = ctypes.c_int
    self._lib.trt_engine_input_rank.argtypes = [ctypes.c_void_p, ctypes.c_int]
    self._lib.trt_engine_input_rank.restype = ctypes.c_int
    self._lib.trt_engine_input_dim.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    self._lib.trt_engine_input_dim.restype = ctypes.c_int
    self._lib.trt_engine_output_shape.argtypes = [
      ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int64), ctypes.c_int,
    ]
    self._lib.trt_engine_output_shape.restype = ctypes.c_int

    with open(engine_path, "rb") as f:
      data = f.read()
    buf = (ctypes.c_char * len(data)).from_buffer_copy(data)
    self._handle = self._lib.trt_engine_create(buf, len(data))
    if not self._handle:
      raise RuntimeError(
        f"trt_engine_create failed for {engine_path}: a plan built by a different "
        f"TensorRT version or a different GPU will not deserialize. Rebuild it here with trtexec."
      )

    self.num_inputs = self._lib.trt_engine_num_inputs(self._handle)
    self.num_outputs = self._lib.trt_engine_num_outputs(self._handle)
    self.input_names = []
    self.input_shapes = []
    for i in range(self.num_inputs):
      name = self._lib.trt_engine_input_name(self._handle, i).decode()
      self.input_names.append(name)
      rank = self._lib.trt_engine_input_rank(self._handle, i)
      if rank < 0:
        raise RuntimeError(f"cannot read the rank of input {name!r}")
      self.input_shapes.append(tuple(self._lib.trt_engine_input_dim(self._handle, i, d) for d in range(rank)))

    self.output_names = [self._lib.trt_engine_output_name(self._handle, i).decode() for i in range(self.num_outputs)]
    dims_out = (ctypes.c_int64 * 8)()
    nd = self._lib.trt_engine_output_shape(self._handle, 0, dims_out, len(dims_out))
    if nd < 0:
      raise RuntimeError("cannot read the first output's shape")
    if nd > len(dims_out):
      raise RuntimeError(f"first output has rank {nd}, more than this build can hold ({len(dims_out)})")
    self.output_shape = tuple(dims_out[:nd])
    dtype_val = self._lib.trt_engine_output_dtype(self._handle, 0)
    self.output_dtype_np = _np_dtype(dtype_val)
    self.output_nbytes = int(np.prod(self.output_shape)) * self.output_dtype_np.itemsize

    # Input shapes/dtypes are read from the engine, never assumed. A vision tensor
    # built for uint8 cannot be fed to a plan compiled for fp16 (project3 measured
    # 0.79 ms/frame for that CPU-side uint8->fp16 conversion).
    self.input_dtypes: dict[str, np.dtype] = {}
    self.input_nbytes: dict[str, int] = {}
    for i, name in enumerate(self.input_names):
      dtype = _np_dtype(self._lib.trt_engine_input_dtype(self._handle, i))
      self.input_dtypes[name] = dtype
      self.input_nbytes[name] = int(np.prod(self.input_shapes[i])) * dtype.itemsize

    self.stream = ctypes.c_ulonglong()
    _check(self._cuda.cuStreamCreate(ctypes.byref(self.stream), 0))
    self.gpu_output = ctypes.c_ulonglong()
    _check(self._cuda.cuMemAlloc_v2(ctypes.byref(self.gpu_output), self.output_nbytes))
    self._input_gpu: dict[str, dict] = {}

  def __call__(self, **kwargs) -> np.ndarray:
    ctx_now = ctypes.c_ulonglong()
    self._cuda.cuCtxGetCurrent(ctypes.byref(ctx_now))
    if ctx_now.value != self._ctx:
      _check(self._cuda.cuCtxSetCurrent(self._ctx))

    addrs = (ctypes.c_ulonglong * self.num_inputs)()
    for i, name in enumerate(self.input_names):
      val = kwargs.get(name)
      if val is None:
        raise KeyError(f"Missing input tensor: {name}")
      if isinstance(val, (int, np.integer)):
        addrs[i] = int(val)
      elif isinstance(val, np.ndarray):
        expect_bytes = self.input_nbytes[name]
        if val.nbytes != expect_bytes:
          raise ValueError(
            f"input {name!r} expects {self.input_shapes[i]} {self.input_dtypes[name]} "
            f"({expect_bytes} B), got {val.shape} {val.dtype} ({val.nbytes} B)"
          )
        info = self._input_gpu.get(name)
        if info is None:
          ptr = ctypes.c_ulonglong()
          _check(self._cuda.cuMemAlloc_v2(ctypes.byref(ptr), expect_bytes))
          info = {"ptr": ptr.value, "nbytes": expect_bytes}
          self._input_gpu[name] = info
        _check(self._cuda.cuMemcpyHtoDAsync_v2(info["ptr"], ctypes.c_void_p(val.ctypes.data), expect_bytes, self.stream))
        addrs[i] = info["ptr"]
      else:
        raise TypeError(f"Unsupported input type for {name}: {type(val).__name__}")

    ret = self._lib.trt_engine_execute(self._handle, addrs, self.gpu_output.value, self.stream.value)
    if ret != 0:
      raise RuntimeError("trt_engine_execute returned non-zero")

    cpu_out = np.empty(self.output_shape, dtype=self.output_dtype_np)
    _check(self._cuda.cuMemcpyDtoHAsync_v2(ctypes.c_void_p(cpu_out.ctypes.data), self.gpu_output, self.output_nbytes, self.stream))
    _check(self._cuda.cuStreamSynchronize(self.stream))
    return cpu_out.astype(np.float32).ravel() if self.output_dtype_np != np.float32 else cpu_out.ravel()

  def close(self) -> None:
    if getattr(self, "_handle", None):
      self._lib.trt_engine_destroy(self._handle)
      self._handle = None
    if getattr(self, "stream", None):
      self._cuda.cuStreamDestroy_v2(self.stream)
    if getattr(self, "gpu_output", None):
      self._cuda.cuMemFree_v2(self.gpu_output)
    for info in getattr(self, "_input_gpu", {}).values():
      self._cuda.cuMemFree_v2(info["ptr"])

  def __del__(self):
    try:
      self.close()
    except Exception:
      pass
