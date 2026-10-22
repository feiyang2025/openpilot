#!/usr/bin/env python3
"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

from collections.abc import Callable
import os
os.environ['GMMU'] = '0'
from openpilot.common.hardware import COMMA_HARDWARE
from openpilot.selfdrive.modeld.helpers import check_camera_jit, chestnut_present
CHESTNUT_PRESENT = chestnut_present()
# tinygrad picks its default device by opening every backend until one works, 2.6 s of every start
# on a comma 4. Without a chestnut that is QCOM, so name it; tinygrad reads DEV once, at import
if COMMA_HARDWARE and not CHESTNUT_PRESENT:
  os.environ.setdefault('DEV', 'QCOM')
import numpy as np
import threading
import time
from setproctitle import setproctitle
from tinygrad import Device
from tinygrad.tensor import Tensor

import openpilot.cereal.messaging as messaging
from openpilot.cereal import log
from opendbc.car.structs import car
from openpilot.cereal.services import SERVICE_LIST
from openpilot.cereal.messaging import PubMaster, SubMaster
from openpilot.cereal.visionipc import VisionStreamType
from msgq.visionipc import VisionIpcClient, VisionBuf
from openpilot.common.file_chunker import open_file_chunked
from openpilot.common.swaglog import cloudlog
from openpilot.common.params import Params
from openpilot.common.dm import is_dm_disabled
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import config_realtime_process, DT_MDL
from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.common.transformations.model import get_warp_matrix
from openpilot.system import sentry
from openpilot.system.camerad.cameras.nv12_info import get_nv12_info
from openpilot.selfdrive.controls.lib.desire_helper import DesireHelper
from openpilot.selfdrive.controls.lib.drive_helpers import get_accel_from_plan, smooth_value
from openpilot.sunnypilot import jetlink_adapter

from openpilot.selfdrive.modeld.compile_modeld import (
  MODELD_INPUTS,
  make_input_queues as make_stock_input_queues,
)
from openpilot.sunnypilot.modeld_v2.fill_model_msg import fill_model_msg, fill_pose_msg, PublishState, get_curvature_from_output
from openpilot.sunnypilot.modeld_v2.parse_model_outputs import Parser
from openpilot.sunnypilot.modeld_v2.constants import ModelConstants, Plan
from openpilot.sunnypilot.modeld_v2.meta_helper import load_meta_constants
from openpilot.sunnypilot.modeld_v2.camera_offset_helper import CameraOffsetHelper
from openpilot.sunnypilot.modeld_v2.compile_modeld import (derive_frame_skip, make_split_input_queues,
                                                           make_supercombo_input_queues, nv12_copy_size,
                                                           WARP_INPUTS, POLICY_INPUTS)
from openpilot.sunnypilot.livedelay.helpers import get_lat_delay
from openpilot.sunnypilot.modeld_v2.modeld_base import ModelStateBase
from openpilot.sunnypilot.modeld_v2 import pkl_device
from openpilot.sunnypilot.modeld_v2.helpers import load_oob
from openpilot.sunnypilot.models.helpers import get_active_bundle
from openpilot.sunnypilot.selfdrive.controls.lib.relc import RoadEdgeLaneChangeController

PROCESS_NAME = "openpilot.selfdrive.modeld.modeld_tinygrad"
BIG_MODEL_TIMEOUT = 60


# Shared device guard (see pkl_device.py): the same hints decide whether a
# catalog pkl may load here and whether a bundle is marked usable in the UI.
_pkl_exists = pkl_device.pkl_exists
_pkl_device_hints = pkl_device.pkl_device_hints
_pkl_incompatible_reason = pkl_device.pkl_incompatible_reason

# Fallback for a catalog bundle that cannot run here: the CUDA build shipped in the checkout.
# On this bench launch_env.sh points COMBINED_MODEL_PKL at the same file, but clearing the env
# (or selecting a QCOM bundle from the model manager) must not take modeld down with it.
_LOCAL_CUDA_PKL = os.path.join(
  os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
  'selfdrive', 'modeld', 'models', 'driving_tinygrad.pkl')


def _find_driving_pkl(bundle):
  # Deliberately returns an incompatible pkl too: ModelState needs the path to log why it was
  # rejected and to fall back. Device validation lives in ModelState, not here.
  #
  # Precedence: an explicit COMBINED_MODEL_PKL wins (bench override / tests), then a model
  # compiled locally on this device from ONNX (P2-1), then the selected catalog bundle.
  if (override := os.environ.get('COMBINED_MODEL_PKL')) and _pkl_exists(override):
    return override
  if (local := pkl_device.active_local_pkl()) is not None:
    return local
  if bundle is None or not bundle.models:
    return None
  from openpilot.common.hardware.hw import Paths
  model_root = Paths.model_root()

  pkl_name = bundle.models[0].artifact.fileName
  pkl_path = os.path.join(model_root, pkl_name)
  if _pkl_exists(pkl_path):
    return pkl_path
  return None


class FrameMeta:
  frame_id: int = 0
  timestamp_sof: int = 0
  timestamp_eof: int = 0

  def __init__(self, vipc=None):
    if vipc is not None:
      self.frame_id, self.timestamp_sof, self.timestamp_eof = vipc.frame_id, vipc.timestamp_sof, vipc.timestamp_eof


class ModelState(ModelStateBase):
  inputs: dict[str, np.ndarray]
  prev_desire: np.ndarray

  def __init__(self, cam_w: int, cam_h: int, chestnut: bool = False):
    ModelStateBase.__init__(self)

    env_pkl = os.environ.get('COMBINED_MODEL_PKL')
    if env_pkl and _pkl_exists(env_pkl):
      model_bundle = None
    else:
      model_bundle = get_active_bundle(chestnut=chestnut)
    self.generation = model_bundle.generation if model_bundle is not None else None
    overrides = {override.key: override.value for override in model_bundle.overrides} if model_bundle else {}

    self.LAT_SMOOTH_SECONDS = float(overrides.get('lat', ".0"))
    self.LONG_SMOOTH_SECONDS = float(overrides.get('long', ".0"))
    self.MIN_LAT_CONTROL_SPEED = 0.3
    self.PLANPLUS_CONTROL: float = 1.0
    self.chestnut = chestnut

    pkl_path = _find_driving_pkl(model_bundle)
    assert pkl_path is not None, f"No driving pkl found for {'chestnut' if chestnut else 'small model'} — all models must be compiled with compile_modeld.py"
    self._load_with_device_fallback(pkl_path, cam_w, cam_h, model_bundle)

  def _load_with_device_fallback(self, pkl_path, cam_w, cam_h, bundle):
    # A picked bundle that cannot run on this device must degrade to the local CUDA build, not
    # kill modeld: manager would otherwise restart it into the same crash on every onroad cycle.
    reason = _pkl_incompatible_reason(pkl_path)
    if reason is None:
      try:
        self._init_combined(pkl_path, cam_w, cam_h, bundle)
        return
      except Exception as e:
        reason = f"{type(e).__name__}: {e}"
    cloudlog.error(f"model pkl unusable: {pkl_path} (device={Device.DEFAULT}): {reason}")
    fallback = _LOCAL_CUDA_PKL
    if not _pkl_exists(fallback) or os.path.abspath(fallback) == os.path.abspath(pkl_path):
      raise RuntimeError(f"no usable driving model: {pkl_path} ({reason})")
    cloudlog.warning(f"falling back to {fallback} (reason: {reason})")
    self._init_combined(fallback, cam_w, cam_h, None)

  def _init_combined(self, pkl_path, cam_w, cam_h, bundle):
    cloudlog.warning(f"loading combined pkl: {pkl_path}")
    jits = load_oob(open_file_chunked(pkl_path))

    metadata = jits['metadata']
    self.WARP_DEV = metadata.get('warp_dev', 'QCOM') if COMMA_HARDWARE else Device.DEFAULT
    self.DEV = ('AMD' if self.chestnut else 'QCOM') if COMMA_HARDWARE else Device.DEFAULT
    self.QUEUE_DEV = self.DEV
    self.is_run_model = 'run_model' in jits

    nv12_info = get_nv12_info(cam_w, cam_h)
    self.frame_copy_size = nv12_copy_size(*nv12_info[:3])
    self.full_frames: dict = {}
    self._blob_cache: dict = {}
    self.frame_buffers: dict = {}

    if self.is_run_model or 'model' in metadata:
      model_metadata = metadata.get('model', metadata)
      self.input_shapes = model_metadata['input_shapes']
      self.vision_output_slices = model_metadata['output_slices']
      self.policy_output_slices = {}
      self._policy_slices_list = []
      self._combined_model_type = 'supercombo'
      self._vision_input_names = [key for key in self.input_shapes if 'img' in key]
      self.frame_skip = derive_frame_skip({}, self.input_shapes)
      if self.is_run_model:
        self.input_queues, self.numpy_inputs, self.frame_buffers = make_stock_input_queues(
          self.input_shapes, self.frame_skip, device=self.DEV, frame_copy_size=self.frame_copy_size)
        self.frame_views, self.npy = self.frame_buffers, self.numpy_inputs
        check_camera_jit(jits['run_model'], cam_w, cam_h, pkl_path)
        self.run_model, self.run_policy, self.warp = jits['run_model'][(cam_w, cam_h)], None, None
      else:
        self.input_queues, self.numpy_inputs = make_supercombo_input_queues(self.input_shapes, self.frame_skip, device=self.QUEUE_DEV)
        check_camera_jit(jits, cam_w, cam_h, pkl_path)
        self.run_model, self.run_policy, self.warp = None, jits['run_policy'], jits[(cam_w, cam_h)]
    else:
      check_camera_jit(jits, cam_w, cam_h, pkl_path)
      self.run_model, self.run_policy, self.warp = None, jits['run_policy'], jits[(cam_w, cam_h)]
      vision_metadata = metadata['vision']
      policy_keys = [k for k in metadata if k not in ('vision', 'warp_dev')]
      self._combined_model_type = 'split' if policy_keys == ['policy'] else 'multi_policy'
      self.vision_output_slices = vision_metadata['output_slices']
      self._policy_keys = policy_keys
      self._policy_slices_list = [metadata[k]['output_slices'] for k in policy_keys]
      self.policy_output_slices = self._policy_slices_list[0]
      self._has_on_policy = any('on' in k.lower() for k in policy_keys)
      self._vision_input_names = [key for key in vision_metadata['input_shapes'] if 'img' in key]
      first_policy_meta = metadata[policy_keys[0]]
      frame_skip = derive_frame_skip(vision_metadata['input_shapes'], first_policy_meta['input_shapes'])
      self.input_queues, self.numpy_inputs = make_split_input_queues(vision_metadata['input_shapes'],
                                                                     first_policy_meta['input_shapes'],
                                                                     frame_skip, device=self.QUEUE_DEV)

    self._desire_key = next(key for key in self.numpy_inputs if key.startswith('desire'))
    self._road_key = next(key for key in self._vision_input_names if 'big' not in key)
    self._wide_key = next(key for key in self._vision_input_names if 'big' in key)
    self.frame_buf_params = dict.fromkeys(self._vision_input_names, nv12_info)

    is_20hz = bundle.is20hz if bundle else self._combined_model_type in ('split', 'multi_policy')
    if is_20hz:
      from openpilot.sunnypilot.models.split_model_constants import SplitModelConstants
      self.constants = SplitModelConstants()
    else:
      self.constants = ModelConstants()

    self.parser = Parser()
    self.prev_desire = np.zeros(self.constants.DESIRE_LEN, dtype=np.float32)

    if self.warp is not None:
      self.full_frames = {k: Tensor(np.zeros(nv12_info[3], dtype=np.uint8), device=self.WARP_DEV).contiguous().realize() for k in self._vision_input_names}
      self.warp(**{k: self.input_queues[k] for k in WARP_INPUTS}, frame=self.full_frames[self._road_key], big_frame=self.full_frames[self._wide_key])

  def warmup(self) -> None:
    dummy_size = self.frame_copy_size if self.is_run_model else self.frame_buf_params[self._road_key][3]
    dummy_frames = {k: np.zeros(dummy_size, dtype=np.uint8) for k in self._vision_input_names}
    transforms = {k: np.eye(3, dtype=np.float32) for k in [self._road_key, self._wide_key] if k}
    dummy_inputs = {k: np.zeros(v.shape, dtype=v.dtype) for k, v in self.numpy_inputs.items() if k not in ['tfm', 'big_tfm', 'prev_feat']}
    self.run(dummy_frames, transforms, dummy_inputs)
    if self.is_run_model:
      self.input_queues, self.numpy_inputs, self.frame_buffers = make_stock_input_queues(
        self.input_shapes, self.frame_skip, device=self.DEV, frame_copy_size=self.frame_copy_size)
      self.frame_views = self.frame_buffers
      self.npy = self.numpy_inputs
    else:
      for v in self.numpy_inputs.values():
        v[:] = 0
      self.full_frames.clear()
      self._blob_cache.clear()
    self.prev_desire[:] = 0

  @property
  def mlsim(self) -> bool:
    return bool(self.generation is not None and self.generation >= 11)

  @property
  def vision_input_names(self) -> list[str]:
    return self._vision_input_names

  @property
  def desire_key(self) -> str:
    return self._desire_key

  def run(self, bufs: dict[str, VisionBuf], transforms: dict[str, np.ndarray],
          inputs: dict[str, np.ndarray],
          after_enqueue: Callable[[], None] | None = None) -> dict[str, np.ndarray] | None:
    if self.is_run_model:
      for key, buf in bufs.items():
        data = buf.data if hasattr(buf, 'data') else buf
        np.copyto(self.frame_buffers[key], np.frombuffer(data, dtype=np.uint8, count=self.frame_copy_size))
    else:
      for key, buf in bufs.items():
        ptr = np.frombuffer(buf.data, dtype=np.uint8).ctypes.data
        cache_key = (key, ptr)
        if cache_key not in self._blob_cache:
          self._blob_cache[cache_key] = Tensor.from_blob(ptr, (self.frame_buf_params[key][3],), dtype='uint8', device=self.WARP_DEV)
        self.full_frames[key] = self._blob_cache[cache_key]

    desire_key = self.desire_key
    inputs[desire_key][0] = 0
    self.numpy_inputs[desire_key][:] = np.where(inputs[desire_key] - self.prev_desire > .99, inputs[desire_key], 0)
    self.prev_desire[:] = inputs[desire_key]

    for key in ('traffic_convention', 'lateral_control_params', 'action_t'):
      if key in self.numpy_inputs and key in inputs:
        self.numpy_inputs[key][:] = inputs[key]

    self.numpy_inputs['tfm'][:, :] = transforms[self._road_key].reshape(3, 3)
    self.numpy_inputs['big_tfm'][:, :] = transforms[self._wide_key].reshape(3, 3)

    if self.run_model is not None:
      outs, = self.run_model(**{k: self.input_queues[k] for k in MODELD_INPUTS})
      raw_outputs = outs
    else:
      assert self.warp is not None and self.run_policy is not None
      warped = self.warp(**{k: self.input_queues[k] for k in WARP_INPUTS}, frame=self.full_frames[self._road_key], big_frame=self.full_frames[self._wide_key])
      raw_outputs = self.run_policy(**{k: self.input_queues[k] for k in POLICY_INPUTS if k in self.input_queues}, warped=warped)

    if after_enqueue is not None:
      after_enqueue()

    if self._combined_model_type == 'supercombo':
      model_output = raw_outputs.numpy().flatten()
      if self.chestnut and not np.all(np.isfinite(model_output)):
        raise RuntimeError("model output not finite")
      sliced = {k: model_output[np.newaxis, v] for k, v in self.vision_output_slices.items()}
      outputs = self.parser.parse_outputs(sliced)
      if 'prev_feat' in self.numpy_inputs and 'hidden_state' in self.vision_output_slices:
        self.numpy_inputs['prev_feat'][:] = model_output[self.vision_output_slices['hidden_state']]
    else:
      vision_output = raw_outputs[0].numpy().flatten()
      vision_sliced = {k: vision_output[np.newaxis, v] for k, v in self.vision_output_slices.items()}
      outputs = self.parser.parse_vision_outputs(vision_sliced)

      if 'prev_feat' in self.numpy_inputs and 'hidden_state' in self.vision_output_slices:
        self.numpy_inputs['prev_feat'][:] = vision_output[self.vision_output_slices['hidden_state']]

      for i, policy_slices in enumerate(self._policy_slices_list):
        policy_output = raw_outputs[i + 1].numpy().flatten()
        policy_sliced = {k: policy_output[np.newaxis, v] for k, v in policy_slices.items()}
        parsed = self.parser.parse_policy_outputs(policy_sliced)
        if ('off' in self._policy_keys[i]
          and self._has_on_policy
          and any('plan' in self._policy_slices_list[j] for j, k in enumerate(self._policy_keys) if 'on' in k.lower())):

          parsed.pop('plan', None)

        outputs.update(parsed)

      if 'planplus' in outputs and 'plan' in outputs:
        outputs['plan'] = outputs['plan'] + outputs['planplus']

    if 'desired_curvature' in outputs and 'prev_desired_curv' in self.numpy_inputs:
      buf = self.numpy_inputs['prev_desired_curv']
      buf[0, :-1] = buf[0, 1:]
      buf[0, -1, :] = outputs['desired_curvature'][0, :] if not self.mlsim else 0

    return outputs

  def get_action_from_model(self, model_output: dict[str, np.ndarray], prev_action: log.ModelDataV2.Action,
                            lat_action_t: float, long_action_t: float, v_ego: float) -> log.ModelDataV2.Action:
    if 'action' not in model_output:
      plan = model_output['plan'][0]
      desired_accel = get_accel_from_plan(plan[:, Plan.VELOCITY][:, 0], plan[:, Plan.ACCELERATION][:, 0], self.constants.T_IDXS,
                                          action_t=long_action_t)

      curvature_plan = (plan + (self.PLANPLUS_CONTROL - 1.0) * model_output['planplus'][0]
                        if 'planplus' in model_output and self.PLANPLUS_CONTROL != 1.0 else plan)
      desired_curvature = get_curvature_from_output(model_output, curvature_plan, v_ego, lat_action_t, self.mlsim)
    else:
      desired_accel = model_output['action'][0, 1]
      desired_curvature = model_output['action'][0, 0] / (max(1.0, v_ego))**2

    stop = v_ego < 0.3 and desired_accel < 0.1
    desired_accel = smooth_value(desired_accel, prev_action.desiredAcceleration, self.LONG_SMOOTH_SECONDS)

    if self.generation is not None and self.generation >= 10: # smooth curvature for post FOF models
      if v_ego > self.MIN_LAT_CONTROL_SPEED:
        desired_curvature = smooth_value(desired_curvature, prev_action.desiredCurvature, self.LAT_SMOOTH_SECONDS)
      else:
        desired_curvature = prev_action.desiredCurvature

    return log.ModelDataV2.Action(desiredCurvature=float(desired_curvature), desiredAcceleration=float(desired_accel), shouldStop=bool(stop))


def _pick_trt_profile(chestnut: bool):
  """Choose a TensorRT profile without touching MODEL_NAME.

  MODEL_NAME also drives the model manager (it selects which catalog bundle to
  run), so reusing it to pick an engine couples two unrelated decisions. TRT_PROFILE
  is the dedicated override; use it to force FiletOFish/TGC/supercombo by name.
  """
  from openpilot.sunnypilot.modeld_v2.gpu_backend.models.registry import REGISTRY, resolve_profile
  if (requested := os.getenv("TRT_PROFILE")):
    return resolve_profile(requested)
  # supercombo is the merged profile that matches the bundled small model
  # (driving_supercombo.onnx; the local CUDA pkl has the same 2574-wide output), so
  # it is the right stand-in when replacing the tinygrad small runner. FiletOFish
  # is a different split model and must be asked for explicitly.
  names = ("TGC", "BigCombo") if chestnut else ("supercombo",)
  for name in names:
    profile = REGISTRY.get(name)
    if profile is not None and profile.is_ready():
      return profile
  return None


def make_model_state(cam_w: int, cam_h: int, chestnut: bool = False):
  """Build the small/big model state, TRT when asked for and available.

  USE_TRT=1 switches the runner to GpuModelState; any failure (missing profile,
  missing engine, backend init) logs and falls back to the tinygrad ModelState so
  one env var can never take modeld down. `USE_TRT=0` (or unset) keeps the exact
  tinygrad path that runs at 20 Hz today.
  """
  if os.getenv("USE_TRT") == "1":
    try:
      from openpilot.sunnypilot.modeld_v2.gpu_model_state import GpuModelState
      profile = _pick_trt_profile(chestnut)
      if profile is None:
        raise RuntimeError(f"no usable TRT profile for chestnut={chestnut}")
      state = GpuModelState(cam_w, cam_h, chestnut, profile=profile)
      state.warmup()
      cloudlog.warning(f"using TRT model state: profile={profile.name} mode={profile.mode} chestnut={chestnut}")
      return state
    except Exception as e:
      cloudlog.error(f"USE_TRT=1 but TRT init failed ({type(e).__name__}: {e}); using tinygrad")
  return ModelState(cam_w=cam_w, cam_h=cam_h, chestnut=chestnut)


def main(demo=False):
  cloudlog.warning("modeld init")

  sentry.set_tag("daemon", PROCESS_NAME)
  cloudlog.bind(daemon=PROCESS_NAME)
  setproctitle(PROCESS_NAME)

  CHESTNUT = CHESTNUT_PRESENT
  if CHESTNUT:
    os.environ['HCQDEV_WAIT_TIMEOUT_MS'] = '3000'
  # before going realtime: prepare() starts tinygrad's device thread, which would inherit FIFO 54 on core 7
  if not CHESTNUT:
    jetlink_adapter.prepare()

  config_realtime_process(7, 54)

  params = Params()
  params.put_bool("ChestnutLoading", CHESTNUT)
  params.remove("ChestnutActive")

  # visionipc clients
  while True:
    available_streams = VisionIpcClient.available_streams("camerad", block=False)
    if available_streams:
      use_extra_client = VisionStreamType.VISION_STREAM_WIDE_ROAD in available_streams and VisionStreamType.VISION_STREAM_NARROW_ROAD in available_streams
      main_wide_camera = VisionStreamType.VISION_STREAM_NARROW_ROAD not in available_streams
      break
    time.sleep(.1)

  vipc_client_main_stream = VisionStreamType.VISION_STREAM_WIDE_ROAD if main_wide_camera else VisionStreamType.VISION_STREAM_NARROW_ROAD
  vipc_client_main = VisionIpcClient("camerad", vipc_client_main_stream, True)
  vipc_client_extra = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_WIDE_ROAD, False)
  cloudlog.warning(f"vision stream set up, main_wide_camera: {main_wide_camera}, use_extra_client: {use_extra_client}")

  while not vipc_client_main.connect(False):
    time.sleep(0.1)
  while use_extra_client and not vipc_client_extra.connect(False):
    time.sleep(0.1)

  cloudlog.warning(f"connected main cam with buffer size: {vipc_client_main.buffer_len} ({vipc_client_main.width} x {vipc_client_main.height})")
  if use_extra_client:
    cloudlog.warning(f"connected extra cam with buffer size: {vipc_client_extra.buffer_len} ({vipc_client_extra.width} x {vipc_client_extra.height})")

  cloudlog.warning("loading model")
  st = time.monotonic()

  model = None
  if CHESTNUT:
    big_model = None
    def load_big():
      nonlocal big_model
      try:
        m = make_model_state(cam_w=vipc_client_main.width, cam_h=vipc_client_main.height, chestnut=True)
        m.warmup()
        big_model = m
      except Exception:
        cloudlog.exception("chestnut load failed")
    loader = threading.Thread(target=load_big, daemon=True)
    loader.start()
    loader.join(BIG_MODEL_TIMEOUT)
    model = big_model
    if model is None:
      params.put_bool("ChestnutModelError", True)
    params.put_bool("ChestnutActive", model is not None)
    if model is not None:
      params.remove("ChestnutModelError")

  small_model = make_model_state(cam_w=vipc_client_main.width, cam_h=vipc_client_main.height, chestnut=False) if model is None or CHESTNUT else None
  if model is None:
    model = small_model
  if (joined := jetlink_adapter.attach(small_model, vipc_client_main.width, vipc_client_main.height)) is not None:
    model = joined
  params.put_bool("ChestnutLoading", False)
  assert model is not None
  cloudlog.warning(f"models loaded in {time.monotonic() - st:.1f}s, modeld starting")

  # messaging
  pub_socks = ["modelV2", "drivingModelData", "cameraOdometry", "modelDataV2SP"] + (["chestnutState"] if CHESTNUT else [])
  pm = PubMaster(pub_socks)
  sm = SubMaster(["deviceState", "carState", "narrowRoadCameraState", "extrinsicsCalibration",
                  "driverMonitoringState", "carControl", "carControlSP",
                  "lateralDelay", "carStateSP", "carrotManSP"])

  publish_state = PublishState()
  chestnut_state = None
  if CHESTNUT:
    # upstream modeld's module imports every car interface (1.7 s on a comma 4), so only for a chestnut
    from openpilot.selfdrive.modeld.modeld import ChestnutState
    chestnut_state = ChestnutState(pm, model.chestnut)

  # setup filter to track dropped frames
  frame_dropped_filter = FirstOrderFilter(0., 10., 1. / model.constants.MODEL_FREQ)
  frame_id = 0
  last_vipc_frame_id = 0
  run_count = 0

  model_transform_main = np.zeros((3, 3), dtype=np.float32)
  model_transform_extra = np.zeros((3, 3), dtype=np.float32)
  live_calib_seen = False
  buf_main, buf_extra = None, None
  meta_main = FrameMeta()
  meta_extra = FrameMeta()
  camera_offset_helper = CameraOffsetHelper()


  if demo:
    from opendbc.car.car_helpers import get_demo_car_params
    CP = get_demo_car_params()
  else:
    CP = messaging.log_from_bytes(params.get("CarParams", block=True), car.CarParams)
  cloudlog.info("modeld got CarParams: %s", CP.brand)

  # TODO Move smooth seconds to action function
  long_delay = CP.longitudinalActuatorDelay + model.LONG_SMOOTH_SECONDS
  prev_action = log.ModelDataV2.Action()

  DH = DesireHelper()
  meta_constants = load_meta_constants()
  RELC = RoadEdgeLaneChangeController()

  while True:
    buf_main = vipc_client_main.recv()
    meta_main = FrameMeta(vipc_client_main)
    if buf_main is None:
      cloudlog.debug("vipc_client_main no frame")
      continue

    if use_extra_client:
      # Pair by frame id instead of pre-advancing main by 25ms of SOF. The two
      # cameras on this bench run ~25ms out of phase (half a 20Hz frame), so the
      # old timestamp handshake read two main frames per iteration and the model
      # was capped at 10Hz. Waiting for the matching extra frame keeps the loop
      # at MODEL_FREQ; the SOF skew is reported below instead of dropping frames.
      while buf_extra is None or meta_extra.frame_id < meta_main.frame_id:
        buf_extra = vipc_client_extra.recv()
        if buf_extra is None:
          break
        meta_extra = FrameMeta(vipc_client_extra)

      if buf_extra is None:
        cloudlog.debug("vipc_client_extra no frame")
        continue

      if meta_main.frame_id != meta_extra.frame_id:
        cloudlog.error(f"frames out of sync! main: {meta_main.frame_id} ({meta_main.timestamp_sof / 1e9:.5f}),\
                       extra: {meta_extra.frame_id} ({meta_extra.timestamp_sof / 1e9:.5f})")

    else:
      # Use single camera
      buf_extra = buf_main
      meta_extra = meta_main

    sm.update(0)
    desire = DH.desire
    is_rhd = False if is_dm_disabled(params) else sm["driverMonitoringState"].isRHD
    frame_id = sm["narrowRoadCameraState"].frameId
    v_ego = max(sm["carState"].vEgo, 0.)
    if sm.frame % 60 == 0:
      model.lat_delay = get_lat_delay(params, sm["lateralDelay"].lateralDelay)
      model.PLANPLUS_CONTROL = params.get("PlanplusControl", return_default=True)
      camera_offset_helper.set_offset(params.get("CameraOffset", return_default=True))
    lat_delay = model.lat_delay + model.LAT_SMOOTH_SECONDS
    if sm.updated["extrinsicsCalibration"] and sm.seen['narrowRoadCameraState'] and sm.seen['deviceState']:
      device_from_calib_euler = np.array(sm["extrinsicsCalibration"].rpyCalib, dtype=np.float32)
      dc = DEVICE_CAMERAS[(str(sm['deviceState'].deviceType), str(sm['narrowRoadCameraState'].sensor))]
      main_intrinsics = dc.wide_road.intrinsics if main_wide_camera else dc.narrow_road.intrinsics
      model_transform_main = get_warp_matrix(device_from_calib_euler, main_intrinsics, False).astype(np.float32)
      model_transform_extra = get_warp_matrix(device_from_calib_euler, dc.wide_road.intrinsics, True).astype(np.float32)
      model_transform_main, model_transform_extra = camera_offset_helper.update(model_transform_main, model_transform_extra, sm, main_wide_camera)
      live_calib_seen = True

    traffic_convention = np.zeros(2)
    traffic_convention[int(is_rhd)] = 1

    vec_desire = np.zeros(model.constants.DESIRE_LEN, dtype=np.float32)
    if desire >= 0 and desire < model.constants.DESIRE_LEN:
      vec_desire[desire] = 1

    # tracked dropped frames
    vipc_dropped_frames = max(0, meta_main.frame_id - last_vipc_frame_id - 1)
    frames_dropped = frame_dropped_filter.update(min(vipc_dropped_frames, 10))
    if run_count < 10: # let frame drops warm up
      frame_dropped_filter.x = 0.
      frames_dropped = 0.
    run_count = run_count + 1

    frame_drop_ratio = frames_dropped / (1 + frames_dropped)

    bufs = {name: buf_extra if 'big' in name else buf_main for name in model.vision_input_names}
    transforms = {name: model_transform_extra if 'big' in name else model_transform_main for name in model.vision_input_names}

    frame_delay = DT_MDL # compensate for time passed since the frame was captured: current_time - timestamp_eof is 50ms on average
    action_delay = DT_MDL / 2 # middle of the interval between model output (current state) and next frame (expected state)
    lat_action_t = lat_delay + frame_delay + action_delay
    long_action_t = long_delay + frame_delay + action_delay

    # action_t for every model: run() takes what it has a slot for, and a large
    # model joining mid-drive reads it from a frame built for the small one
    inputs:dict[str, np.ndarray] = {
      model.desire_key: vec_desire,
      'traffic_convention': traffic_convention,
      'action_t': np.array([lat_action_t, long_action_t], dtype=np.float32),
    }

    if 'lateral_control_params' in model.numpy_inputs:
      inputs['lateral_control_params'] = np.array([v_ego, lat_delay], dtype=np.float32)

    # a model can change which model drives inside run() (jetlink's joining
    # model counts its handovers); the stall of one is not lag, as for the
    # fallback below, and nor are the drops of the frame it happens on. The
    # joining model hands a large model back on this share of dropped frames,
    # and swaps one in only while nothing is in control
    model.in_control = jetlink_adapter.in_control(sm)
    model.frame_drop_ratio = frame_drop_ratio
    handovers = getattr(model, 'handovers', 0)
    mt1 = time.perf_counter()
    try:
      send_chestnut = (chestnut_state is not None and
                       run_count % round(model.constants.MODEL_FREQ / SERVICE_LIST['chestnutState'].frequency) == 0)
      model_output = model.run(bufs, transforms, inputs, chestnut_state.send if send_chestnut else None)
    except Exception:
      if not params.get_bool("ChestnutActive"):
        raise
      cloudlog.exception("chestnut failed, falling back to small")
      params.put_bool("ChestnutModelError", True)
      params.put_bool("ChestnutActive", False)
      assert small_model is not None
      model = small_model
      if chestnut_state is not None:
        chestnut_state.big = False
      run_count = 0
      model_output = None
    mt2 = time.perf_counter()
    model_execution_time = mt2 - mt1
    if getattr(model, 'handovers', 0) != handovers:
      run_count = 0
      frame_drop_ratio = 0.

    if model_output is not None:
      modelv2_send = messaging.new_message('modelV2')
      drivingdata_send = messaging.new_message('drivingModelData')
      posenet_send = messaging.new_message('cameraOdometry')
      mdv2sp_send = messaging.new_message('modelDataV2SP')
      mdv2sp_send.modelDataV2SP.acceleratorState = getattr(model, 'big_model_state', 'none')

      action = model.get_action_from_model(model_output, prev_action, lat_action_t, long_action_t, v_ego)
      prev_action = action
      fill_model_msg(drivingdata_send, modelv2_send, model_output, action,
                     publish_state, meta_main.frame_id, meta_extra.frame_id, frame_id,
                     frame_drop_ratio, meta_main.timestamp_eof, model_execution_time, live_calib_seen, meta_constants)
      modelv2_send.modelV2.big = model.chestnut
      # as stock modeld's fill_driving_model_data: the qlog's only model message
      drivingdata_send.drivingModelData.big = model.chestnut

      desire_state = modelv2_send.modelV2.meta.desireState
      l_lane_change_prob = desire_state[log.Desire.laneChangeLeft]
      r_lane_change_prob = desire_state[log.Desire.laneChangeRight]
      lane_change_prob = l_lane_change_prob + r_lane_change_prob
      cs_sp = sm['carStateSP']
      left_lane_line_blocked = cs_sp.carrotLaneValid and cs_sp.carrotLeftLineBlocked
      right_lane_line_blocked = cs_sp.carrotLaneValid and cs_sp.carrotRightLineBlocked

      left_edge, right_edge = RELC.update_and_fill(modelv2_send.modelV2, mdv2sp_send.modelDataV2SP, v_ego)
      carrot_man = sm['carrotManSP'] if sm.alive['carrotManSP'] else None
      DH.update(sm['carState'], sm['carControl'].latActive, lane_change_prob, left_edge, right_edge,
                left_lane_line_blocked=left_lane_line_blocked,
                right_lane_line_blocked=right_lane_line_blocked,
                carrot_man=carrot_man)
      modelv2_send.modelV2.meta.laneChangeState = DH.lane_change_state
      modelv2_send.modelV2.meta.laneChangeDirection = DH.lane_change_direction
      mdv2sp_send.valid = modelv2_send.valid
      mdv2sp_send.modelDataV2SP.laneTurnDirection = DH.lane_turn_direction
      mdv2sp_send.valid = modelv2_send.valid
      drivingdata_send.drivingModelData.meta.laneChangeState = DH.lane_change_state
      drivingdata_send.drivingModelData.meta.laneChangeDirection = DH.lane_change_direction

      fill_pose_msg(posenet_send, model_output, meta_main.frame_id, vipc_dropped_frames, meta_main.timestamp_eof, live_calib_seen)
      pm.send('modelV2', modelv2_send)
      pm.send('drivingModelData', drivingdata_send)
      pm.send('cameraOdometry', posenet_send)
      pm.send('modelDataV2SP', mdv2sp_send)
    last_vipc_frame_id = meta_main.frame_id

if __name__ == "__main__":
  try:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', action='store_true', help='A boolean for demo mode.')
    args = parser.parse_args()
    main(demo=args.demo)
  except KeyboardInterrupt:
    cloudlog.warning(f"child {PROCESS_NAME} got SIGINT")
  except Exception:
    sentry.capture_exception()
    raise
