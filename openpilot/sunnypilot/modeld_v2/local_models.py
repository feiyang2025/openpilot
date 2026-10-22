"""
Locally compile the in-repo ONNX driving models into CUDA tinygrad pkls.

comma's model catalog only ships pkls built for Qualcomm (and a few for AMD);
none of them load on this NVIDIA device. The ONNX sources, though, are in the
checkout via git-lfs, so the model selector can build its own CUDA pkl with
`compile_modeld.py` and hand modeld a model that actually runs here.

A compiled model lands in `Paths.model_root()` as `driving_<key>_local_tinygrad.pkl`
and is recorded in `local_models.json` (see pkl_device). modeld prefers the
active local model over a downloaded catalog bundle.
"""
import os
import subprocess
import sys
from dataclasses import dataclass, field

from openpilot.common.hardware.hw import Paths
from openpilot.sunnypilot.modeld_v2 import pkl_device

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MODELS_DIR = os.path.join(REPO_ROOT, 'openpilot', 'selfdrive', 'modeld', 'models')

# The camera stream this device actually delivers; compile the model for both so
# switching resolutions later does not need a rebuild.
BENCH_CAMERA_RESOLUTIONS = [(1344, 760), (1928, 1208)]

MODEL_SIZE = (512, 256)

# An unfetched git-lfs object is a small text pointer, not the model. Counting it
# as present would offer a build that can only fail.
LFS_POINTER = b'version https://git-lfs.github.com/spec/v1'


def _is_lfs_pointer(path: str) -> bool:
  try:
    with open(path, 'rb') as f:
      return f.read(len(LFS_POINTER)) == LFS_POINTER
  except OSError:
    return False


@dataclass(frozen=True)
class LocalModelSpec:
  key: str
  name: str
  model_type: str
  onnx: dict[str, str]
  description: str = ''
  camera_resolutions: list[tuple[int, int]] = field(default_factory=lambda: list(BENCH_CAMERA_RESOLUTIONS))

  def onnx_paths(self) -> dict[str, str]:
    return {role: os.path.join(MODELS_DIR, rel) for role, rel in self.onnx.items()}

  def missing_sources(self) -> list[str]:
    return [p for p in self.onnx_paths().values()
            if not pkl_device.pkl_exists(p) or _is_lfs_pointer(p)]

  def pkl_name(self) -> str:
    return f'driving_{self.key}{pkl_device.LOCAL_PKL_SUFFIX}'

  def pkl_path(self) -> str:
    return os.path.join(Paths.model_root(), self.pkl_name())


LOCAL_MODELS: dict[str, LocalModelSpec] = {
  'supercombo': LocalModelSpec(
    key='supercombo',
    name='Supercombo (local CUDA)',
    model_type='supercombo',
    onnx={'supercombo': os.path.join('supercombo', 'driving_supercombo.onnx')},
    description='The bundle small model, recompiled here for CUDA.',
  ),
  'tgc': LocalModelSpec(
    key='tgc',
    name='TGC big (local CUDA)',
    model_type='supercombo',
    onnx={'supercombo': os.path.join('TGC', 'big_driving_supercombo.onnx')},
    description='Big merged model, recompiled here for CUDA (slower, larger).',
  ),
  'filet_ofish': LocalModelSpec(
    key='filet_ofish',
    name='FiletOFish (local CUDA)',
    model_type='vision_policy',
    onnx={'vision': os.path.join('FiletOFish', 'driving_vision.onnx'),
          'policy': os.path.join('FiletOFish', 'driving_policy.onnx')},
    description='Split vision+policy model, recompiled here for CUDA.',
  ),
}


def get_spec(key: str) -> LocalModelSpec | None:
  return LOCAL_MODELS.get(key)


def _build_args(spec: LocalModelSpec, output: str) -> list[str]:
  args = [
    '--model-type', spec.model_type,
    '--model-size', f'{MODEL_SIZE[0]}x{MODEL_SIZE[1]}',
    '--camera-resolutions', *[f'{w}x{h}' for w, h in spec.camera_resolutions],
    '--output', output,
  ]
  paths = spec.onnx_paths()
  if spec.model_type == 'supercombo':
    args += ['--supercombo-onnx', paths['supercombo']]
  else:
    args += ['--vision-onnx', paths['vision'], '--policy-onnx', paths['policy']]
  return args


def compile_model(key: str, progress=None, timeout: float | None = 3600) -> str:
  """Compile `key` to a CUDA pkl and register it (without activating).

  Returns the output path. `progress(line)` receives each compiler stdout line.
  """
  spec = get_spec(key)
  if spec is None:
    raise ValueError(f"unknown local model: {key}")
  missing = spec.missing_sources()
  if missing:
    raise FileNotFoundError(f"{key}: ONNX source missing (git-lfs not pulled?): {missing}")

  os.makedirs(Paths.model_root(), exist_ok=True)
  output = spec.pkl_path()
  env = dict(os.environ)
  env['PYTHONPATH'] = os.pathsep.join(filter(None, [REPO_ROOT, env.get('PYTHONPATH', '')]))

  cmd = [sys.executable, '-m', 'openpilot.sunnypilot.modeld_v2.compile_modeld', *_build_args(spec, output)]
  proc = subprocess.Popen(cmd, cwd=REPO_ROOT, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, bufsize=1)
  try:
    for line in proc.stdout:  # type: ignore[union-attr]
      line = line.rstrip('\n')
      if progress is not None:
        progress(line)
      else:
        print(line, flush=True)
    rc = proc.wait(timeout=timeout)
  except BaseException:
    proc.kill()
    raise
  if rc != 0 or not pkl_device.pkl_exists(output):
    raise RuntimeError(f"{key}: compile_modeld failed (rc={rc})")

  pkl_device.register_local_model(
    key, output, name=spec.name, model_type=spec.model_type,
    onnx=list(spec.onnx_paths().values()), activate=False,
  )
  return output


def is_compiled(key: str) -> bool:
  return pkl_device.local_pkl_for(key) is not None


def status() -> list[dict]:
  """Per-model availability for the UI: sources present, compiled, active."""
  active = pkl_device.active_local_key()
  out = []
  for key, spec in LOCAL_MODELS.items():
    compiled = is_compiled(key)
    out.append({
      'key': key,
      'name': spec.name,
      'description': spec.description,
      'sources_present': not spec.missing_sources(),
      'compiled': compiled,
      'active': compiled and key == active,
      'pkl_path': spec.pkl_path(),
    })
  return out
