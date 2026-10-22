"""UI helpers for locally compiling and selecting a CUDA driving model."""
from __future__ import annotations

import os
import threading
import time

from openpilot.common.hardware.hw import Paths
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.sunnypilot.modeld_v2 import local_models, pkl_device
from openpilot.system.ui.lib.multilang import tr


def local_cuda_supported() -> bool:
  """Cheap CUDA check for the UI.

  Deliberately does not import tinygrad (pkl_device.current_device()): the UI
  must not pull the compiler and a device into its process just to show a row.
  The compile itself runs in a subprocess, where current_device() is cheap.
  """
  return os.path.exists('/dev/nvidiactl') or os.path.exists('/dev/nvidia0')


class LocalModelCompileJob:
  """One background compile job shared by the Tici and Mici model panels."""

  def __init__(self):
    self._lock = threading.Lock()
    self._state = {'running': False, 'key': None, 'progress': '', 'error': None}
    self._thread: threading.Thread | None = None

  def snapshot(self) -> dict:
    with self._lock:
      return dict(self._state)

  def start(self, key: str) -> bool:
    if not ui_state.is_offroad() or not local_cuda_supported() or local_models.get_spec(key) is None:
      return False
    with self._lock:
      if self._state['running']:
        return False
      self._state = {'running': True, 'key': key, 'progress': tr('Starting compile'), 'error': None}
      self._thread = threading.Thread(target=self._run, args=(key,), name='local-model-compile', daemon=True)
      self._thread.start()
    return True

  def _run(self, key: str) -> None:
    def report(line: str) -> None:
      with self._lock:
        self._state['progress'] = line[-120:]

    try:
      local_models.compile_model(key, progress=report)
      if not pkl_device.set_active_local(key):
        raise RuntimeError(f"{key}: compiled artifact is not available")
    except Exception as e:
      with self._lock:
        self._state.update(running=False, error=str(e), progress='')
    else:
      with self._lock:
        self._state.update(running=False, error=None, progress=tr('Compile complete'))


local_model_compile_job = LocalModelCompileJob()

# status() reads the index and stats the ONNX sources; the panels ask every
# frame, so the settled text is cached for half a second.
_STATUS_TTL = 0.5
_status_cache = {'at': 0.0, 'text': ''}


def local_model_status_text() -> str:
  job = local_model_compile_job.snapshot()
  if job['running']:
    spec = local_models.get_spec(job['key'])
    name = spec.name if spec else job['key']
    return tr('Building {}: {}').format(name, job['progress'])
  if job['error']:
    return tr('Build failed: {}').format(job['error'])

  now = time.monotonic()
  if now - _status_cache['at'] < _STATUS_TTL:
    return _status_cache['text']

  statuses = local_models.status()
  active = next((item['name'] for item in statuses if item['active']), None)
  compiled = sum(item['compiled'] for item in statuses)
  if active:
    text = tr('Selected: {} (applies when modeld starts)').format(active)
  elif compiled:
    text = tr('{} local CUDA model(s) ready; choose one to use').format(compiled)
  else:
    text = tr('No local CUDA model compiled')
  _status_cache.update(at=now, text=text)
  return text


def bundle_device_note(bundle) -> str | None:
  """Compatibility note for a downloaded bundle visible on this device.

  Only CUDA-only devices need it: elsewhere the catalog bundles are built for
  the hardware they run on. The device name is passed explicitly so this never
  imports tinygrad into the UI process.
  """
  if not local_cuda_supported():
    return None
  device = 'CUDA'

  models = getattr(bundle, 'models', None)
  if not isinstance(models, (list, tuple)):
    return None

  for model in models:
    filename = getattr(getattr(model, 'artifact', None), 'fileName', '')
    if not filename or not filename.endswith('.pkl'):
      continue
    path = os.path.join(Paths.model_root(), filename)
    try:
      if not pkl_device.pkl_exists(path):
        continue
      reason = pkl_device.pkl_incompatible_reason(path, device=device)
    except OSError:
      continue
    if reason:
      hints = ', '.join(sorted(pkl_device.pkl_device_hints(path)))
      return tr('Not compatible with this device ({})').format(hints or reason)
  return None
