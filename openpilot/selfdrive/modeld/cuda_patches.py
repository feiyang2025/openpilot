"""Fork-side runtime patches for the CUDA/TensorRT stack.

These used to live inside upstream files (modeld.py, process_config.py,
camerad.py). They are collected here so those files stay upstream's, and so the
device launcher, the manager and every child process pick them up from one
place via the repo-root `sitecustomize.py`.

What is patched, and why:

* `selfdrive/modeld/modeld.ModelState` -> `GpuModelState` when a CUDA profile
  with usable engines exists. This is the class `process_config.py:260` runs for
  `Runner.stock`; the tinygrad runner at `:320` is a different process with its
  own ModelState and is left alone. Upstream only knows tinygrad; on a Jetson the
  TensorRT engines are far faster. Falls back to the stock ModelState whenever the
  profile is missing, the engines are absent, or the backend cannot init.
* `system/camerad/webcam/camerad.Camera` -> the V4L2/DMABUF camera on Linux.

Rules for anything added here:

* never import the modules being patched at startup - use `patch_after_import()`.
  modeld must keep setting its tinygrad device before the first tinygrad import.
* one patch per concern, idempotent, and silent on failure: this module runs in
  every python process on the device (pytest and the modeld worker included) and
  must never be able to break one.
* `OPENPILOT_CUDA_PATCH=0` disables everything. It is read at call time so the
  escape hatch still works after import.
"""

from __future__ import annotations

import functools
import importlib.machinery
import os
import sys
from collections.abc import Callable

ENABLED_ENV = "OPENPILOT_CUDA_PATCH"


def enabled() -> bool:
  """Read at call time, not import time.

  sitecustomize imports this module before anything can set the variable, so a
  value captured at import would make the escape hatch untestable and, worse,
  unchangeable for the rest of the process's life.
  """
  return os.environ.get(ENABLED_ENV, "1") != "0"

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


def patch_modeld_state(modeld: object) -> bool:
  """Swap the tinygrad ModelState for GpuModelState on Jetson.

  `selfdrive/modeld/modeld.ModelState` is what process_config.py runs for the
  stock runner. Its constructor signature is (cam_w, cam_h, chestnut) and the
  chestnut instance is the big model, so that is where the CUDA backend belongs.

  The whole __init__ is wrapped rather than patched statement by statement:
  stock and CUDA build different state (tinygrad queues vs engine bindings), so
  there is no meaningful shared body to keep. On any failure the stock path runs
  unchanged, and the reason is recorded.
  """
  if getattr(modeld, "_fork_cuda_modelstate", False):
    return False
  if not _cuda_enabled_by_env():
    return False

  original_init = modeld.ModelState.__init__

  # chestnut is the big model; without it the small model runs. Both can be CUDA
  # profiles here, so the requested one is picked by shape rather than by name:
  # merged profiles carry the big models, the split/small one does not. MODEL_NAME
  # still wins when it is set.
  def _pick_profile(chestnut: bool):
    from openpilot.sunnypilot.modeld_v2.gpu_backend.models.registry import REGISTRY, resolve_profile

    requested = os.getenv("MODEL_NAME") or None
    if requested:
      return resolve_profile(requested)
    fallback = None
    for profile in REGISTRY.values():
      if profile is None or not profile.is_ready():
        continue
      if profile.mode == "merged":
        if chestnut:
          return profile
      elif fallback is None:
        fallback = profile
    return fallback if not chestnut else None

  @functools.wraps(original_init)
  def __init__(self, cam_w: int, cam_h: int, chestnut: bool = False):
    try:
      from openpilot.sunnypilot.modeld_v2.gpu_model_state import GpuModelState
      profile = _pick_profile(chestnut)
      if profile is None:
        _PATCH_STATE["modeld"] = "no usable CUDA profile (engines or metadata missing); using tinygrad"
        original_init(self, cam_w, cam_h, chestnut)
        return
      GpuModelState.__init__(self, cam_w, cam_h, chestnut, profile=profile)
      self.signal_shadow = None
      _PATCH_STATE["modeld"] = f"cuda profile={profile.name} mode={profile.mode} chestnut={chestnut}"
    except Exception as e:
      _PATCH_STATE["modeld"] = f"{type(e).__name__}: {e}; using tinygrad"
      original_init(self, cam_w, cam_h, chestnut)

  modeld.ModelState.__init__ = __init__
  modeld._fork_cuda_modelstate = True
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
  if not enabled():
    return []
  applied: list[str] = []

  for name, module, patch in (
    ("modeld", "openpilot.selfdrive.modeld.modeld", patch_modeld_state),
    # modeld_v2 is what actually runs here: the current bundle's runner is tinygrad,
    # so process_config starts modeld_tinygrad and the stock module above is never
    # imported. Without this hook the patch would never fire on this machine.
    ("modeld_v2", "openpilot.sunnypilot.modeld_v2.modeld", patch_modeld_state),
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
