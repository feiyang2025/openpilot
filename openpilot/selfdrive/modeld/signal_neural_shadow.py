"""Opt-in neural signal observer for this tree's PC/CUDA hardware.

The upstream observer is written around a Qualcomm execution session. This
version keeps the same artifacts and display contract but runs the two ONNX
models through tinygrad on the local CUDA device (see tools/signal_analysis/
neural_signal.py). CPU stays available as the lightweight camera tracker, not
as a slow neural fallback.

Enable by placing the pinned detector/classifier plus `neural_enabled` (and
optionally `neural_backend`) in the signal shadow directory.
"""
import hashlib
import json
import os
from pathlib import Path
import time

from openpilot.common.repo_tools import load
from openpilot.selfdrive.modeld.signal_color_shadow import DIRECTORY, requested
from openpilot.selfdrive.modeld.signal_tracking_shadow import MAX_INPUT_AGE_MS, copy_nv12_rgb, next_delay

BACKENDS = ('auto', 'cuda', 'cpu')


def neural_requested(directory=DIRECTORY):
  try:
    return (Path(directory) / 'neural_enabled').read_text().strip() == '1'
  except OSError:
    return False


def backend_requested(directory=DIRECTORY):
  try:
    backend = (Path(directory) / 'neural_backend').read_text().strip()
  except OSError:
    backend = 'auto'
  if backend not in BACKENDS:
    raise ValueError(f'invalid neural backend selection: {backend!r} (expected one of {BACKENDS})')
  return backend


def cuda_available() -> bool:
  """A CUDA device without the Qualcomm GPU node this tree does not target."""
  if os.path.exists('/dev/kgsl-3d0'):
    return False
  return os.path.exists('/dev/nvidiactl') or os.path.exists('/dev/nvidia0')


def automatic_backend() -> str:
  return 'cuda' if cuda_available() else 'cpu'


def display_tracks(tracks):
  result = []
  for t in tracks[:20]:
    labels = t['label'].split(',')
    raw = ('red' if 'red' in labels else 'green' if labels == ['green'] else 'unknown') if t['score'] >= .5 else 'unknown'
    result.append(dict(id=t['id'], box=t['box'], age=0, state='unknown',
                       evidence=dict(raw=raw), label=t['label'], score=t['score'],
                       source=t['source'], neural_age=t['neural_age']))
  return result


def _run_tracking(directory, duration):
  from openpilot.selfdrive.modeld.signal_tracking_shadow import run as run_tracking
  return run_tracking(directory, duration, comparison_override=True)


def run(directory=DIRECTORY, duration=None, backend_override=None):
  import fcntl
  directory = Path(directory)
  if not requested(directory) or not neural_requested(directory):
    return
  selection = backend_requested(directory)
  backend = backend_override or (automatic_backend() if selection == 'auto' else selection)
  if backend == 'cpu':
    # Deliberately the cheap tracker: CPU YOLOX cannot keep up on this device.
    return _run_tracking(directory, duration)
  if backend != 'cuda':
    raise ValueError(f'unsupported neural backend: {backend!r}')

  model_dir = directory / 'neural'
  detector, classifier = model_dir / 'detector.onnx', model_dir / 'classifier.onnx'
  if not detector.is_file() or not classifier.is_file():
    # A missing pair must not crash-loop the shadow process (restart_if_crash):
    # leave the observer off and say why.
    from openpilot.common.swaglog import cloudlog
    cloudlog.error(f'signal neural observer disabled: models missing in {model_dir}')
    return

  with (directory / 'worker.lock').open('w') as lock:
    try:
      fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
      return

    import numpy as np
    from msgq.visionipc import VisionIpcClient, VisionStreamType
    from openpilot.common.swaglog import cloudlog
    from openpilot.sunnypilot.carrot.signal_display import display_requested, publish_display

    neural_signal = load('signal_analysis/neural_signal.py')
    network = neural_signal.NeuralSignal(detector, classifier, backend='cuda')
    # Compile/capture before subscribing: startup must not create a camera backlog.
    network.detector.run(None, {'images': np.zeros((1, 3, 416, 416), np.float32)})
    network.classifier.run(None, {'input': np.zeros((1, 3, 224, 224), np.float32)})
    sequence = neural_signal.NeuralSignalSequence(network)
    session = f'neural:{os.getpid()}:{time.monotonic_ns()}'
    source_sha = hashlib.sha256(Path(neural_signal.__file__).read_bytes()).hexdigest()

    client = VisionIpcClient('camerad', VisionStreamType.VISION_STREAM_ROAD, True)
    start = time.monotonic()
    cloudlog.event('signalNeuralShadowLoaded', backend=backend, algorithm_sha256=source_sha,
                   detector_sha256=network.DETECTOR_SHA, classifier_sha256=network.CLASSIFIER_SHA,
                   pid=os.getpid(), control_permission=False, assist_transport=False)
    previous_id, last_file = None, 0.
    while (requested(directory) and neural_requested(directory)
           and backend_requested(directory) == selection
           and (duration is None or time.monotonic() - start < duration)):
      if not client.is_connected() and not client.connect(False):
        sequence = neural_signal.NeuralSignalSequence(network)
        time.sleep(.2)
        continue
      frame = client.recv(timeout_ms=100)
      if frame is None:
        sequence = neural_signal.NeuralSignalSequence(network)
        continue
      fid, eof = int(client.frame_id), int(client.timestamp_eof)
      if fid == previous_id:
        time.sleep(.005)
        continue
      if previous_id is not None and fid < previous_id:
        sequence = neural_signal.NeuralSignalSequence(network)
      gap = None if previous_id is None else fid - previous_id
      previous_id = fid
      begin, cpu = time.monotonic(), time.process_time()
      age = (time.monotonic_ns() - eof) / 1e6
      if not 0 <= age <= MAX_INPUT_AGE_MS:
        sequence = neural_signal.NeuralSignalSequence(network)
        time.sleep(.05)
        continue
      rgb = copy_nv12_rgb(frame)
      if int(frame.frame_id) != fid:
        sequence = neural_signal.NeuralSignalSequence(network)
        continue
      result = sequence.process(rgb, eof / 1e9)
      record = dict(mode='neural_display_only', neural=True, backend=backend,
                    algorithm_sha256=source_sha, stream='road',
                    reference_width=1344, reference_height=760, frame_id=fid,
                    frame_gap=gap, timestamp_eof=eof, session=session,
                    detected=result['detected'], input_age_ms=age,
                    result_age_ms=(time.monotonic_ns() - eof) / 1e6,
                    recorded_monotonic_ns=time.monotonic_ns(),
                    work_ms=(time.monotonic() - begin) * 1000,
                    cpu_ms=(time.process_time() - cpu) * 1000,
                    tracks=display_tracks(result['tracks']),
                    control_permission=False, assist_transport=False)
      cloudlog.event('signalNeuralShadow', **record)
      if display_requested(directory / 'display_enabled'):
        publish_display(record)
      if time.monotonic() - last_file >= 1:
        tmp = directory / 'neural_latest.tmp'
        tmp.write_text(json.dumps(record, allow_nan=False))
        tmp.replace(directory / 'neural_latest.json')
        last_file = time.monotonic()
      time.sleep(next_delay(time.monotonic() - begin, time.process_time() - cpu))
