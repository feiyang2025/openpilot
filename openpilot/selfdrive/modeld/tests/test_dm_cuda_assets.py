"""Device-tagged DM artifacts: a locally compiled CUDA build wins over the
Qualcomm/NPY pkl that ships with the tree."""
from pathlib import Path

from openpilot.selfdrive.modeld import helpers


def _with_models_dir(monkeypatch, tmp_path):
  monkeypatch.setattr(helpers, 'MODELS_DIR', Path(tmp_path))
  monkeypatch.setattr(helpers, 'tg_device_tag', lambda: 'cuda')


def test_cuda_dm_model_is_preferred(monkeypatch, tmp_path):
  _with_models_dir(monkeypatch, tmp_path)
  (tmp_path / 'dmonitoring_model_tinygrad.pkl').write_bytes(b'legacy')
  (tmp_path / 'dmonitoring_model_cuda_tinygrad.pkl').write_bytes(b'cuda')
  assert helpers.dm_model_pkl_path().name == 'dmonitoring_model_cuda_tinygrad.pkl'


def test_legacy_dm_model_remains_the_fallback(monkeypatch, tmp_path):
  _with_models_dir(monkeypatch, tmp_path)
  (tmp_path / 'dmonitoring_model_tinygrad.pkl').write_bytes(b'legacy')
  assert helpers.dm_model_pkl_path().name == 'dmonitoring_model_tinygrad.pkl'


def test_cuda_warp_is_preferred_per_camera(monkeypatch, tmp_path):
  _with_models_dir(monkeypatch, tmp_path)
  (tmp_path / 'dm_warp_1344x760_tinygrad.pkl').write_bytes(b'legacy')
  (tmp_path / 'dm_warp_1344x760_cuda_tinygrad.pkl').write_bytes(b'cuda')
  assert helpers.dm_warp_path(1344, 760).name == 'dm_warp_1344x760_cuda_tinygrad.pkl'


def test_device_tag_matches_the_hardware_nodes(monkeypatch):
  monkeypatch.setattr(helpers.os.path, 'exists', lambda p: p == '/dev/nvidiactl')
  assert helpers.tg_device_tag() == 'cuda'
  monkeypatch.setattr(helpers.os.path, 'exists', lambda p: p == '/dev/kgsl-3d0')
  assert helpers.tg_device_tag() == 'qcom'
  monkeypatch.setattr(helpers.os.path, 'exists', lambda p: False)
  assert helpers.tg_device_tag() == 'cpu'
