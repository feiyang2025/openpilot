"""Fork-side runtime patches for the CUDA/TensorRT stack.

These used to live inside upstream files (modeld.py, process_config.py,
camerad.py). They are collected here so those files stay upstream's, and so the
device launcher, the manager and every child process pick them up from one
place via the repo-root `sitecustomize.py`.

What is patched, and why:

* `sunnypilot/modeld_v2/modeld.ModelState` -> `GpuModelState` when a CUDA profile
  with usable engines exists. Upstream only knows tinygrad; on a Jetson the
  TensorRT engines are ~25x faster. Falls back to the stock ModelState whenever
  the profile is missing, the engines are absent, or the backend cannot init.
* `system/camerad/webcam/camerad.Camera` -> the V4L2/DMABUF camera on Linux.

Rules for anything added here:

* never import the modules being patched at startup - use `patch_after_import()`.
  modeld must keep setting its tinygrad device before the first tinygrad import.
* one patch per concern, idempotent, and silent on failure: this module runs in
  every python process on the device (pytest and the modeld worker included) and
  must never be able to break one.
* `OPENPILOT_CUDA_PATCH=0` disables everything.
"""

from __future__ import annotations

import functools
import importlib.machinery
import os
import sys
from collections.abc import Callable

ENABLED = os.environ.get("OPENPILOT_CUDA_PATCH", "1") != "0"

# Set by patch_modeld_state(); read by patch_camerad_camera() so a failed patch
# is reported once, at the point where it actually matters.
_PATCH_STATE: dict[str, str] = {}


def patch_after_import(module_name: str, callback: Callable[[object], bool]) -> bool:
  """Call callback(module) as soon as module_name finishes importing.

  Used instead of importing the target here: the call sites import tinygrad and
  the modeld modules lazily, and importing them at interpreter start would be a
  behaviour change (startup cost, device ordering), not a patch.
  """
  if (module := sys.modules.get(module_name)) is not None:
    try:
      callback(module)
    except Exception:
      pass
    return True

  class _Finder:
    def find_spec(self, fullname, path=None, target=None):
      if fullname != module_name:
        return None
      sys.meta_path.remove(self)
      try:
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
      except Exception:
        return None
      if spec is None or spec.loader is None:
        return None
      execute = spec.loader.exec_module

      def exec_module(module):
        execute(module)
        try:
          callback(module)
        except Exception:
          pass

      try:
        spec.loader.exec_module = exec_module  # type: ignore[method-assign]
      except Exception:
        return None
      return spec

  sys.meta_path.insert(0, _Finder())
  return True


def _cuda_enabled_by_env() -> bool:
  return os.environ.get("DISABLE_CUDA_BACKEND", "0") != "1"


def patch_modeld_state(modeld_v2: object) -> bool:
  """Swap the tinygrad ModelState for GpuModelState on Jetson.

  Only the two construction sites in `_init_combined`'s caller are touched; the
  rest of ModelState is left alone so a fallback keeps the stock behaviour
  exactly.
  """
  if getattr(modeld_v2, "_fork_cuda_modelstate", False):
    return False
  if not _cuda_enabled_by_env():
    return False

  original_init = modeld_v2.ModelState.__init__

  @functools.wraps(original_init)
  def __init__(self, cam_w: int, cam_h: int, chestnut: bool = False):
    if not chestnut:
      original_init(self, cam_w, cam_h, chestnut)
      return
    try:
      from openpilot.sunnypilot.modeld_v2.gpu_model_state import GpuModelState
      from openpilot.sunnypilot.modeld_v2.gpu_backend.models.registry import resolve_profile
      profile = resolve_profile(os.getenv("MODEL_NAME") or None)
      if profile is not None:
        _PATCH_STATE["modeld"] = f"cuda profile={profile.name} mode={profile.mode}"
        GpuModelState.__init__(self, cam_w, cam_h, chestnut)
        return
      _PATCH_STATE["modeld"] = "no usable CUDA profile (engines or metadata missing)"
    except Exception as e:
      _PATCH_STATE["modeld"] = f"{type(e).__name__}: {e}"
    original_init(self, cam_w, cam_h, chestnut)

  modeld_v2.ModelState.__init__ = __init__
  modeld_v2._fork_cuda_modelstate = True
  return True


def patch_camerad_camera(camerad_module: object) -> bool:
  """Use the V4L2/DMABUF camera instead of cv2 on Linux.

  Upstream webcam/camera.py opens /dev/videoN with OpenCV and rotates 180deg.
  v4l2_dmabuf_camera.py handles UYVY + Jetson VIC; the OpenCV path stays as the
  fallback so a missing device or a Darwin host behaves exactly as before.
  """
  if getattr(camerad_module, "_fork_cuda_camera", False):
    return False
  if os.environ.get("USE_V4L2_CAMERA", "1") != "1" or sys.platform == "darwin":
    return False

  try:
    from openpilot.system.camerad.webcam.v4l2_dmabuf_camera import Camera as V4L2Camera
  except Exception as e:
    _PATCH_STATE["camera"] = f"v4l2 camera unavailable: {type(e).__name__}: {e}"
    return False

  camerad_module.Camera = V4L2Camera
  camerad_module._fork_cuda_camera = True
  _PATCH_STATE["camera"] = f"v4l2_dmabuf_camera.Camera ({V4L2Camera.__name__})"
  return True


def apply() -> list[str]:
  """Apply every patch, returns the names that changed something."""
  if not ENABLED:
    return []
  applied: list[str] = []

  for name, module, patch in (
    ("modeld", "openpilot.sunnypilot.modeld_v2.modeld", patch_modeld_state),
    ("camerad", "openpilot.system.camerad.webcam.camerad", patch_camerad_camera),
  ):
    try:
      if patch_after_import(module, patch):
        applied.append(f"hook:{module}")
    except Exception:
      pass

  if applied:
    try:
      from openpilot.common.swaglog import cloudlog
      cloudlog.warning(f"cuda patches active: {applied} {_PATCH_STATE}")
    except Exception:
      pass
  return applied
