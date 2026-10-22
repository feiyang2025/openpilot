"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

import struct

from openpilot.common.test import OpenpilotTestCase
from openpilot.sunnypilot.modeld_v2 import local_models, pkl_device
from openpilot.sunnypilot.modeld_v2.tests import helpers as tests_helpers

# resolved by name from this module when a test asks for it
tmp_path = tests_helpers.tmp_path


def _write_opcode_file(path, opcodes: bytes):
  with open(path, 'wb') as f:
    f.write(struct.pack('<q', len(opcodes)))
    f.write(opcodes)
  return str(path)


class TestPklDevice(OpenpilotTestCase):
  def test_hints_from_opcode_header(self, tmp_path):
    path = _write_opcode_file(tmp_path / 'a.pkl', b'..CUDA..')
    assert pkl_device.pkl_device_hints(path) == {'CUDA'}

  def test_no_header_yields_no_hints(self, tmp_path):
    path = tmp_path / 'b.pkl'
    path.write_bytes(b'x')
    assert pkl_device.pkl_device_hints(str(path)) == set()

  def test_incompatible_reason(self, tmp_path):
    path = _write_opcode_file(tmp_path / 'q.pkl', b'..QCOM..')
    assert 'QCOM' in pkl_device.pkl_incompatible_reason(path, device='CUDA')
    assert pkl_device.pkl_incompatible_reason(path, device='QCOM') is None
    assert pkl_device.pkl_compatible(path, device='CUDA') is False


class TestLocalIndex(OpenpilotTestCase):
  def setup_method(self):
    self.tmp = self._fixture("tmp_path")
    self._fixture("mocker").patch("openpilot.common.hardware.hw.Paths.model_root",
                                  staticmethod(lambda: self.tmp))

  def _tmp(self):
    return self.tmp

  def test_active_roundtrip(self):
    pkl = self._tmp() / 'driving_supercombo_local_tinygrad.pkl'
    pkl.write_bytes(b'x')
    pkl_device.register_local_model('supercombo', str(pkl), name='Supercombo', activate=True)
    assert pkl_device.active_local_key() == 'supercombo'
    assert pkl_device.active_local_pkl() == str(pkl)

  def test_missing_pkl_clears_active(self):
    pkl = self._tmp() / 'gone.pkl'
    pkl.write_bytes(b'x')
    pkl_device.register_local_model('tgc', str(pkl), activate=True)
    pkl.unlink()
    assert pkl_device.active_local_pkl() is None
    assert pkl_device.active_local_key() is None

  def test_no_index_is_empty(self):
    assert pkl_device.active_local_pkl() is None
    assert pkl_device.local_models() == {}

  def test_select_rejects_unknown_key(self):
    assert pkl_device.set_active_local('nope') is False
    assert pkl_device.active_local_key() is None

  def test_select_rejects_missing_pkl(self):
    pkl_device.register_local_model('tgc', str(self._tmp() / 'never.pkl'))
    assert pkl_device.set_active_local('tgc') is False
    assert pkl_device.active_local_key() is None

  def test_select_and_clear(self):
    pkl = self._tmp() / 'driving_supercombo_local_tinygrad.pkl'
    pkl.write_bytes(b'x')
    pkl_device.register_local_model('supercombo', str(pkl))
    assert pkl_device.set_active_local('supercombo') is True
    assert pkl_device.active_local_key() == 'supercombo'
    assert pkl_device.set_active_local(None) is True
    assert pkl_device.active_local_key() is None

  def test_corrupt_index_tolerated(self):
    with open(pkl_device.local_index_path(), 'w') as f:
      f.write('{"models": []}')
    assert pkl_device.active_local_key() is None
    assert pkl_device.local_models() == {}

  def test_write_leaves_no_temp_files(self):
    pkl = self._tmp() / 'driving_supercombo_local_tinygrad.pkl'
    pkl.write_bytes(b'x')
    pkl_device.register_local_model('supercombo', str(pkl), activate=True)
    leftovers = [p.name for p in self._tmp().iterdir() if p.name.startswith('.local_models-')]
    assert leftovers == []

  def test_chunked_artifact_reports_total_size(self):
    base = self._tmp() / 'driving_supercombo_local_tinygrad.pkl'
    (self._tmp() / 'driving_supercombo_local_tinygrad.pkl.chunkmanifest').write_text('2')
    (self._tmp() / 'driving_supercombo_local_tinygrad.pkl.chunk01of02').write_bytes(b'a' * 10)
    (self._tmp() / 'driving_supercombo_local_tinygrad.pkl.chunk02of02').write_bytes(b'b' * 5)
    pkl_device.register_local_model('supercombo', str(base), activate=True)
    assert pkl_device.local_models()['supercombo']['size'] == 15
    assert pkl_device.active_local_pkl() == str(base)

  def test_list_annotates_availability(self):
    pkl = self._tmp() / 'driving_supercombo_local_tinygrad.pkl'
    pkl.write_bytes(b'x')
    pkl_device.register_local_model('supercombo', str(pkl), activate=True)
    pkl_device.register_local_model('tgc', str(self._tmp() / 'gone.pkl'))
    items = {m['key']: m for m in pkl_device.list_local_models()}
    assert items['supercombo']['available'] is True
    assert items['supercombo']['active'] is True
    assert items['tgc']['available'] is False
    assert items['tgc']['active'] is False


class TestLocalModels(OpenpilotTestCase):
  def setup_method(self):
    self.tmp = self._fixture("tmp_path")
    self._fixture("mocker").patch("openpilot.common.hardware.hw.Paths.model_root",
                                  staticmethod(lambda: self.tmp))
    self._fixture("mocker").patch("openpilot.sunnypilot.modeld_v2.local_models.MODELS_DIR",
                                  str(self.tmp))

  def _tmp(self):
    return self.tmp

  def test_missing_sources_reported(self):
    st = {s['key']: s for s in local_models.status()}
    assert st['supercombo']['sources_present'] is False
    assert st['tgc']['sources_present'] is False

  def test_compile_args_supercombo(self):
    spec = local_models.get_spec('supercombo')
    args = local_models._build_args(spec, '/tmp/out.pkl')
    assert '--supercombo-onnx' in args and '--vision-onnx' not in args
    assert args[args.index('--model-type') + 1] == 'supercombo'

  def test_compile_args_split(self):
    spec = local_models.get_spec('filet_ofish')
    args = local_models._build_args(spec, '/tmp/out.pkl')
    assert '--vision-onnx' in args and '--policy-onnx' in args

  def test_unknown_key_rejected(self):
    with self.assertRaises(ValueError):
      local_models.compile_model('nope')

  def test_lfs_pointer_is_not_a_present_source(self):
    import os
    spec = local_models.get_spec('supercombo')
    path = spec.onnx_paths()['supercombo']
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
      f.write(local_models.LFS_POINTER + b'\noid sha256:deadbeef\nsize 123\n')
    assert path in spec.missing_sources()
    assert spec.missing_sources() == [path]

  def test_a_real_source_is_present(self):
    import os
    spec = local_models.get_spec('supercombo')
    path = spec.onnx_paths()['supercombo']
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
      f.write(b'\x08\x07real onnx bytes')
    assert spec.missing_sources() == []

  def test_status_marks_compiled_and_active(self):
    pkl = self._tmp() / 'driving_supercombo_local_tinygrad.pkl'
    pkl.write_bytes(b'x')
    pkl_device.register_local_model('supercombo', str(pkl), activate=True)
    st = {s['key']: s for s in local_models.status()}
    assert st['supercombo']['compiled'] is True
    assert st['supercombo']['active'] is True
    assert st['tgc']['compiled'] is False
