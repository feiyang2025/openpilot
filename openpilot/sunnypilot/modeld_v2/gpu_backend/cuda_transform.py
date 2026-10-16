import ctypes
import os
from pathlib import Path

import numpy as np
from tinygrad.dtype import dtypes
from tinygrad.tensor import Tensor


class CudaTransform:
  """NV12 -> warped vision tensor, executed by libcuda_transform.so.

  The kernel writes into device memory; the resulting pointer is wrapped as a
  tinygrad CUDA Tensor so the existing pipeline can keep passing device pointers
  around. The vision tensor's dtype is a property of the engine, not of this
  class - the caller passes it in (project3 measured BigCombo img=uint8 and
  TGC img=float16, so hardcoding either one is wrong for the other model).
  """

  def __init__(self, model_w: int, model_h: int, temporal_skip: int, library_path: str | None = None,
               dtype: np.dtype = np.uint8):
    path = library_path or os.getenv("CUDA_TRANSFORM_LIBRARY")
    if path is None:
      path = str(Path(__file__).resolve().parents[3] / "selfdrive" / "modeld" / "transforms" / "libcuda_transform.so")
    if not Path(path).exists():
      raise RuntimeError(
        f"libcuda_transform.so not found at {path}; "
        f"build it on-device with: nvcc -arch=sm_87 -shared -O2 -o libcuda_transform.so cuda_transform.cu"
      )
    self.dtype = np.dtype(dtype)
    if self.dtype != np.uint8 and getattr(dtypes, self.dtype.name, None) is None:
      raise RuntimeError(f"tinygrad has no dtype for {self.dtype}")
    self._library = ctypes.CDLL(path)
    self._library.cuda_transform_init.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                                  ctypes.c_int, ctypes.c_int]
    self._library.cuda_transform_init.restype = None
    self._library.cuda_transform_execute.argtypes = [
      ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
      ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
    ]
    self._library.cuda_transform_execute.restype = ctypes.c_void_p
    self._library.cuda_transform_destroy.argtypes = [ctypes.c_void_p]
    self._library.cuda_transform_destroy.restype = None
    self._states: dict[str, ctypes.Array] = {}
    self._model_w = model_w
    self._model_h = model_h
    self._temporal_skip = temporal_skip
    self.last_ptr = 0

  def _state(self, camera: str) -> ctypes.Array:
    if camera not in self._states:
      state = ctypes.create_string_buffer(128)
      # 0 = uint8, 1 = fp16. The kernel has to be told, not just the Python-side
      # dtype: the buffer is allocated by the .so, so a uint8-sized one read as
      # fp16 runs off the end.
      out_dtype = 0 if self.dtype == np.uint8 else 1
      self._library.cuda_transform_init(
        ctypes.byref(state), self._model_w, self._model_h, self._temporal_skip, out_dtype,
      )
      self._states[camera] = state
    return self._states[camera]

  def __call__(self, camera: str, frame, projection: np.ndarray, shape: tuple[int, ...]) -> Tensor:
    # The frame is passed to the .so as an ordinary HOST pointer; the library stages
    # it onto the device itself. Pinning it with cuMemHostRegister(DEVICEMAP) and
    # hoping the host pointer doubles as a device pointer is not viable here: on this
    # platform the driver API reports no usable device from Python (cuDeviceGet ->
    # INVALID_DEVICE, cuMemAlloc -> INVALID_CONTEXT) while the runtime API is fine,
    # so the registration either fails or yields a pointer the kernels fault on.
    data = np.frombuffer(frame.data, dtype=np.uint8)
    proj = np.ascontiguousarray(projection, dtype=np.float32)
    output = self._library.cuda_transform_execute(
      ctypes.byref(self._state(camera)), ctypes.c_void_p(int(data.ctypes.data)),
      frame.width, frame.height, frame.stride, frame.uv_offset,
      proj.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    if not output:
      raise RuntimeError("cuda_transform_execute returned null")
    self.last_ptr = int(output)
    tdtype = dtypes.uint8 if self.dtype == np.uint8 else getattr(dtypes, self.dtype.name)
    return Tensor.from_blob(output, shape, dtype=tdtype, device="CUDA")

  def close(self) -> None:
    for state in self._states.values():
      self._library.cuda_transform_destroy(ctypes.byref(state))
    self._states.clear()

  def __del__(self):
    try:
      self.close()
    except Exception:
      pass
