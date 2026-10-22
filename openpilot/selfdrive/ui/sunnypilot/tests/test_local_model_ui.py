"""
Copyright (c) 2026-, sunnypilot contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

The local-model row's helpers: what the row says, and whether a downloaded
bundle is marked as unusable on this device.
"""
import struct
import struct

from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.ui.sunnypilot import local_model_ui
from openpilot.sunnypilot.modeld_v2.tests import helpers as tests_helpers

# resolved by name from this module when a test asks for it
tmp_path = tests_helpers.tmp_path


def _write_opcode_file(path, opcodes: bytes):
  with open(path, 'wb') as f:
    f.write(struct.pack('<q', len(opcodes)))
    f.write(opcodes)
  return path


class _Model:
  def __init__(self, filename):
    self.artifact = type('Artifact', (), {'fileName': filename})()


class _Bundle:
  def __init__(self, *filenames):
    self.models = [_Model(name) for name in filenames]


class TestBundleDeviceNote(OpenpilotTestCase):
  def setup_method(self):
    self.tmp = self._fixture("tmp_path")
    mocker = self._fixture("mocker")
    mocker.patch("openpilot.common.hardware.hw.Paths.model_root", staticmethod(lambda: str(self.tmp)))
    mocker.patch("openpilot.selfdrive.ui.sunnypilot.local_model_ui.local_cuda_supported", return_value=True)

  def test_marks_a_pkl_built_for_another_device(self):
    _write_opcode_file(self.tmp / 'driving_qcom_tinygrad.pkl', b'..QCOM..')
    note = local_model_ui.bundle_device_note(_Bundle('driving_qcom_tinygrad.pkl'))
    assert note is not None and 'QCOM' in note

  def test_a_cuda_pkl_is_not_marked(self):
    _write_opcode_file(self.tmp / 'driving_cuda_tinygrad.pkl', b'..CUDA..')
    assert local_model_ui.bundle_device_note(_Bundle('driving_cuda_tinygrad.pkl')) is None

  def test_an_unrelated_artifact_is_ignored(self):
    _write_opcode_file(self.tmp / 'driving_qcom.onnx', b'..QCOM..')
    assert local_model_ui.bundle_device_note(_Bundle('driving_qcom.onnx')) is None

  def test_a_bundle_without_models_is_not_marked(self):
    assert local_model_ui.bundle_device_note(object()) is None


class TestLocalCUDA(OpenpilotTestCase):
  def test_reads_the_nvidia_device_nodes(self):
    mocker = self._fixture("mocker")
    mocker.patch("os.path.exists", side_effect=lambda path: path in ('/dev/nvidiactl', '/dev/nvidia0'))
    assert local_model_ui.local_cuda_supported() is True

  def test_false_without_cuda_nodes(self):
    self._fixture("mocker").patch("os.path.exists", return_value=False)
    assert local_model_ui.local_cuda_supported() is False


class TestLocalModelStatusText(OpenpilotTestCase):
  def setup_method(self):
    self.mocker = self._fixture("mocker")
    local_model_ui._status_cache.update(at=0.0, text='')

  def test_no_model_yet(self):
    self.mocker.patch("openpilot.sunnypilot.modeld_v2.local_models.status", return_value=[
      {'key': 'supercombo', 'name': 'Supercombo', 'compiled': False, 'active': False},
    ])
    assert 'No local CUDA model' in local_model_ui.local_model_status_text()

  def test_compiled_but_not_selected(self):
    self.mocker.patch("openpilot.sunnypilot.modeld_v2.local_models.status", return_value=[
      {'key': 'supercombo', 'name': 'Supercombo', 'compiled': True, 'active': False},
    ])
    assert 'choose one' in local_model_ui.local_model_status_text()

  def test_selected_names_the_model(self):
    self.mocker.patch("openpilot.sunnypilot.modeld_v2.local_models.status", return_value=[
      {'key': 'supercombo', 'name': 'Supercombo', 'compiled': True, 'active': True},
    ])
    assert 'Supercombo' in local_model_ui.local_model_status_text()

  def test_a_running_build_wins_over_the_index(self):
    self.mocker.patch.object(local_model_ui.local_model_compile_job, 'snapshot',
                             return_value={'running': True, 'key': 'supercombo', 'progress': 'compiling', 'error': None})
    self.mocker.patch("openpilot.sunnypilot.modeld_v2.local_models.get_spec",
                      return_value=type('Spec', (), {'name': 'Supercombo'})())
    assert 'compiling' in local_model_ui.local_model_status_text()

