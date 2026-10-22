"""
Device compatibility for compiled tinygrad .pkl driving models.

comma's catalog ships pkls compiled for Qualcomm (and a few for AMD); this
device runs NVIDIA, so only a pkl whose opcode stream carries the CUDA tag (or
the device-agnostic NPY tag) can load. A pkl built for another backend
unpickles its buffers straight into ops_qcom and dies on /dev/kgsl-3d0 before
modeld can publish anything. Only the opcode header (~1-3 MB) is read here, so
the incompatible pick can be refused before the expensive load.

This module is shared by modeld (load-time guard) and the model manager/UI
("runs on this device?" marking, plus the local-compiled model index).
"""
import json
import os
import struct
import tempfile
import time

from openpilot.common.hardware.hw import Paths

# Device tags tinygrad bakes into a compiled .pkl's opcode stream.
PKL_DEVICE_TAGS = ('QCOM', 'CUDA', 'AMD', 'NPY')

# Local-compiled models (ONNX -> CUDA pkl) live next to the downloaded bundles,
# with their selection recorded in this index.
LOCAL_INDEX_NAME = 'local_models.json'
LOCAL_PKL_SUFFIX = '_local_tinygrad.pkl'


def current_device() -> str:
  from tinygrad import Device
  return Device.DEFAULT.upper()


def _manifest_path(path: str) -> str:
  from openpilot.common.file_chunker import get_manifest_path
  return get_manifest_path(path)


def pkl_exists(path) -> bool:
  return os.path.exists(path) or os.path.exists(_manifest_path(path))


def _open_chunked(path):
  from openpilot.common.file_chunker import open_file_chunked
  return open_file_chunked(path)


def pkl_device_hints(path) -> set[str]:
  try:
    with _open_chunked(path) as f:
      header = f.read(8)
      if len(header) < 8:
        return set()
      opcode_len = struct.unpack('<q', header)[0]
      if opcode_len <= 0:
        return set()
      opcodes = f.read(opcode_len)
  except Exception:
    return set()
  return {tag for tag in PKL_DEVICE_TAGS if tag.encode() in opcodes}


def pkl_incompatible_reason(path, device: str | None = None) -> str | None:
  hints = pkl_device_hints(path)
  device = (device or current_device()).upper()
  if hints and device not in hints:
    return f"pkl targets {sorted(hints)} but device={device}"
  return None


def pkl_compatible(path, device: str | None = None) -> bool:
  """True when the pkl can load here (or has no device hints to judge by)."""
  return pkl_incompatible_reason(path, device=device) is None


# ---- locally compiled model index -------------------------------------------

def local_index_path() -> str:
  return os.path.join(Paths.model_root(), LOCAL_INDEX_NAME)


def read_local_index() -> dict:
  # Normalised shape, so a hand-edited or half-written index cannot crash the
  # callers: {'active': str|None, 'models': {key: {...}}}.
  try:
    with open(local_index_path()) as f:
      data = json.load(f)
    if isinstance(data, dict) and isinstance(data.get('models'), dict):
      data.setdefault('active', None)
      return data
  except (OSError, ValueError):
    pass
  return {'active': None, 'models': {}}


def write_local_index(data: dict) -> None:
  # Written atomically (temp file + fsync + replace): modeld reads this from
  # another process, so a torn write would look like "no local model selected".
  root = Paths.model_root()
  os.makedirs(root, exist_ok=True)
  path = local_index_path()
  fd, tmp = tempfile.mkstemp(prefix='.local_models-', suffix='.json', dir=root)
  try:
    with os.fdopen(fd, 'w') as f:
      json.dump(data, f, indent=2, sort_keys=True)
      f.write('\n')
      f.flush()
      os.fsync(f.fileno())
    os.replace(tmp, path)
  finally:
    try:
      os.unlink(tmp)
    except FileNotFoundError:
      pass


def local_models() -> dict:
  """key -> {name, pkl, model_type, onnx, compiled_at, size}"""
  return read_local_index().get('models', {})


def local_pkl_for(key: str) -> str | None:
  entry = local_models().get(key)
  if not entry:
    return None
  pkl = os.path.join(Paths.model_root(), entry.get('pkl', ''))
  return pkl if pkl_exists(pkl) else None


def active_local_key() -> str | None:
  return read_local_index().get('active')


def active_local_pkl() -> str | None:
  key = active_local_key()
  if not key:
    return None
  pkl = local_pkl_for(key)
  if pkl is None:
    # selection points at a model that is gone; drop it so we don't shadow bundles
    index = read_local_index()
    if index.get('active') == key:
      index['active'] = None
      write_local_index(index)
  return pkl


def _artifact_size(path: str) -> int:
  """Bytes of a pkl, counting the chunk files when it was delivered chunked.

  compile_modeld chunks large pkls and removes the base file, so stat'ing only
  the nominal path would always report 0.
  """
  try:
    if os.path.exists(path):
      return os.path.getsize(path)
    directory, base = os.path.split(path)
    total = 0
    for name in os.listdir(directory):
      if name.startswith(base + '.chunk') and not name.endswith('.chunkmanifest'):
        total += os.path.getsize(os.path.join(directory, name))
    return total
  except OSError:
    return 0


def register_local_model(key: str, pkl_path: str, *, name: str | None = None,
                         model_type: str = '', onnx: list[str] | None = None,
                         activate: bool = False) -> None:
  index = read_local_index()
  models = index.setdefault('models', {})
  models[key] = {
    'name': name or key,
    'pkl': os.path.basename(pkl_path),
    'model_type': model_type,
    'onnx': onnx or [],
    'compiled_at': int(time.time()),
    'size': _artifact_size(pkl_path),
  }
  # A first registration becomes the default only after the artifact is
  # actually present.  The compiler normally leaves a chunk manifest rather
  # than the base file, so use pkl_exists() instead of os.path.exists().
  if pkl_exists(pkl_path) and (activate or index.get('active') is None):
    index['active'] = key
  write_local_index(index)


def set_active_local(key: str | None) -> bool:
  """Remember the user's pick. Unknown keys (and missing pkls) are rejected.

  Mirrors model_catalog.select(): naming a model that is not registered would
  only fail at modeld load time, so it is refused here instead. Returns whether
  the selection was applied.
  """
  index = read_local_index()
  if key is not None and local_pkl_for(key) is None:
    return False
  index['active'] = key
  write_local_index(index)
  return True


def list_local_models() -> list[dict]:
  """Every registered local model, annotated with availability and selection."""
  index = read_local_index()
  active = index.get('active')
  out = []
  for key, entry in sorted(index.get('models', {}).items()):
    item = dict(entry)
    item['key'] = key
    item['available'] = local_pkl_for(key) is not None
    item['active'] = item['available'] and key == active
    out.append(item)
  return out
