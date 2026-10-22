"""PC/CUDA adaptation of the opt-in neural signal observer.

The upstream observer runs its ONNX pair through a Qualcomm session. These
tests build a tiny ONNX pair with the same input/output contract and run it
through the CUDA session, so the PC path is exercised without the (large,
separately distributed) reference models.
"""
import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

onnx = pytest.importorskip('onnx')
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from openpilot.common.repo_tools import load  # noqa: E402
from openpilot.selfdrive.modeld import signal_neural_shadow as shadow  # noqa: E402

neural_signal = load('signal_analysis/neural_signal.py')

pytestmark = pytest.mark.skipif(
  not (os.path.exists('/dev/nvidiactl') or os.path.exists('/dev/nvidia0'))
  or os.path.exists('/dev/kgsl-3d0'),
  reason='needs the CUDA-only device this tree targets',
)


def _save(model, path):
  onnx.save(model, str(path))
  return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _detector(path):
  inp = helper.make_tensor_value_info('images', TensorProto.FLOAT, [1, 3, 416, 416])
  out = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3549, 7])
  boxes = np.zeros((1, 3549, 7), np.float32)
  boxes[0, 0, 2:4] = np.log(50.)   # object size in grid cells
  boxes[0, 0, 4] = 1.             # objectness
  boxes[0, 0, 6] = 1.             # class score the decoder multiplies by
  const = numpy_helper.from_array(boxes, name='const')
  zero = numpy_helper.from_array(np.zeros((), np.float32), name='zero')
  nodes = [helper.make_node('ReduceSum', ['images'], ['s'], keepdims=0),
           helper.make_node('Mul', ['s', 'zero'], ['z']),
           helper.make_node('Add', ['z', 'const'], ['output'])]
  graph = helper.make_graph(nodes, 'detector', [inp], [out], [const, zero])
  return _save(helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)]), path)


def _classifier(path):
  inp = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 224, 224])
  out = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 11])
  const = numpy_helper.from_array(np.zeros((1, 11), np.float32), name='const')
  zero = numpy_helper.from_array(np.zeros((), np.float32), name='zero')
  nodes = [helper.make_node('ReduceSum', ['input'], ['s'], keepdims=0),
           helper.make_node('Mul', ['s', 'zero'], ['z']),
           helper.make_node('Add', ['z', 'const'], ['output'])]
  graph = helper.make_graph(nodes, 'classifier', [inp], [out], [const, zero])
  return _save(helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)]), path)


def _network(tmp_path):
  det, cls = tmp_path / 'detector.onnx', tmp_path / 'classifier.onnx'
  det_sha, cls_sha = _detector(det), _classifier(cls)
  old = neural_signal.NeuralSignal.DETECTOR_SHA, neural_signal.NeuralSignal.CLASSIFIER_SHA
  neural_signal.NeuralSignal.DETECTOR_SHA, neural_signal.NeuralSignal.CLASSIFIER_SHA = det_sha, cls_sha
  try:
    yield neural_signal.NeuralSignal(det, cls, backend='cuda')
  finally:
    neural_signal.NeuralSignal.DETECTOR_SHA, neural_signal.NeuralSignal.CLASSIFIER_SHA = old


def test_cuda_session_runs_the_pinned_onnx_contract(tmp_path):
  gen = _network(tmp_path)
  network = next(gen)
  try:
    rgb = np.zeros((416, 416, 3), np.uint8)
    boxes = network.detect(rgb)
    assert len(boxes) == 1 and boxes[0]['score'] == 1.0
    result = network.classify(rgb, boxes[0]['box'])
    assert result['label'] in neural_signal.NeuralSignal.LABELS
    assert network.backend == 'cuda'
  finally:
    gen.close()


def test_sequence_produces_displayable_track_fields(tmp_path):
  gen = _network(tmp_path)
  network = next(gen)
  try:
    sequence = neural_signal.NeuralSignalSequence(network)
    result = sequence.process(np.zeros((760, 1344, 3), np.uint8), 1.0)
    assert set(result) == {'timestamp', 'detected', 'tracks'}
    assert len(result['tracks']) == 1
    track = result['tracks'][0]
    assert track['source'] == 'detected' and 'label' in track
  finally:
    gen.close()


def test_automatic_backend_prefers_cuda(monkeypatch):
  monkeypatch.setattr(shadow.os.path, 'exists', lambda p: p in ('/dev/nvidiactl', '/dev/nvidia0'))
  assert shadow.automatic_backend() == 'cuda'


def test_automatic_backend_never_uses_cuda_on_a_qualcomm_gpu(monkeypatch):
  monkeypatch.setattr(shadow.os.path, 'exists', lambda p: p in ('/dev/kgsl-3d0', '/dev/nvidiactl'))
  assert shadow.automatic_backend() == 'cpu'


@pytest.mark.parametrize('value,expected', [('auto', 'auto'), ('cuda', 'cuda'), ('cpu', 'cpu')])
def test_backend_requested_accepts_pc_values(tmp_path, value, expected):
  (tmp_path / 'neural_backend').write_text(value)
  assert shadow.backend_requested(tmp_path) == expected


@pytest.mark.parametrize('value', ['qcom', 'snpe', ''])
def test_backend_requested_rejects_qualcomm_and_junk(tmp_path, value):
  (tmp_path / 'neural_backend').write_text(value)
  with pytest.raises(ValueError):
    shadow.backend_requested(tmp_path)


def test_missing_models_leave_the_observer_off(tmp_path):
  (tmp_path / 'enabled').write_text('1')
  (tmp_path / 'neural_enabled').write_text('1')
  (tmp_path / 'neural_backend').write_text('cuda')
  assert shadow.run(tmp_path, duration=0) is None
