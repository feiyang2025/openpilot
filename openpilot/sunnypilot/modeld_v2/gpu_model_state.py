import os

import numpy as np
from openpilot.cereal import log
from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.controls.lib.drive_helpers import get_accel_from_plan, get_curvature_from_plan, smooth_value, should_stop
from openpilot.sunnypilot.modeld_v2.modeld_base import ModelStateBase

from openpilot.sunnypilot.modeld_v2.gpu_backend.backend import create_gpu_backend
from openpilot.sunnypilot.modeld_v2.gpu_backend.models.registry import resolve_profile
from openpilot.sunnypilot.modeld_v2.constants import ModelConstants, Plan
from openpilot.sunnypilot.modeld_v2.parse_model_outputs import Parser

PLAN_KEYS = ("plan", "desired_curvature", "action")


class GpuModelState(ModelStateBase):
  """CUDA/TensorRT backend for modeld.

  Replaces the tinygrad `ModelState` when a CUDA profile with usable engines is
  available. Falls back to the caller (which should construct the stock
  `ModelState`) on any failure - see `cuda_patches.make_model_state`.
  """

  def __init__(self, cam_w: int, cam_h: int, chestnut: bool = False):
    ModelStateBase.__init__(self)
    self.chestnut = chestnut
    model_name = os.getenv("MODEL_NAME") or Params().get("Model", encoding="utf-8")
    self.profile = resolve_profile(model_name)
    if self.profile is None:
      raise RuntimeError(
        f"no usable CUDA model profile for {model_name or 'default'}; "
        f"searched {[p.name for p in __import__('openpilot.sunnypilot.modeld_v2.gpu_backend.models.registry', fromlist=['REGISTRY']).REGISTRY.values()]}"
      )
    self.backend = create_gpu_backend(self.profile)
    if self.backend is None:
      raise RuntimeError("CUDA backend unavailable (see the create_gpu_backend warning above for the reason)")

    self.constants = ModelConstants
    self.generation = None
    self.is_run_model = False

    self._vision_input_names = list(self.profile.vision_input_names)
    self._road_key = next(k for k in self._vision_input_names if "big" not in k)
    self._wide_key = next(k for k in self._vision_input_names if "big" in k)

    self.input_shapes = dict(self.profile.input_shapes)
    self.vision_output_slices = dict(self.profile.vision_slices)
    self.policy_output_slices = dict(self.profile.policy_slices)

    self.numpy_inputs: dict[str, np.ndarray] = {}
    for k, shape in self.input_shapes.items():
      if k in self._vision_input_names:
        continue
      self.numpy_inputs[k] = np.zeros(shape, dtype=np.float32)

    desire_candidates = [k for k in self.numpy_inputs if k.startswith("desire")]
    if not desire_candidates:
      raise RuntimeError(f"profile {self.profile.name} declares no desire input; got {sorted(self.input_shapes)}")
    self._desire_key = desire_candidates[0]

    self.prev_desire = np.zeros(self.constants.DESIRE_LEN, dtype=np.float32)
    self.lat_delay = 0.0
    self.LAT_SMOOTH_SECONDS = 0.0
    self.LONG_SMOOTH_SECONDS = 0.0
    self.MIN_LAT_CONTROL_SPEED = 0.3
    self.PLANPLUS_CONTROL = 1.0

    self.parser = Parser()
    self._vision_out_size: int | None = None
    cloudlog.warning(
      f"GpuModelState initialized: model={model_name or 'default'} mode={self.profile.mode} "
      f"profile={self.profile.name} vision_out={len(self.vision_output_slices)} slices"
    )

  @property
  def mlsim(self) -> bool:
    return False

  @property
  def vision_input_names(self) -> list[str]:
    return self._vision_input_names

  @property
  def road_key(self) -> str:
    return self._road_key

  @property
  def wide_key(self) -> str:
    return self._wide_key

  @property
  def desire_key(self) -> str:
    return self._desire_key

  def warmup(self) -> None:
    frames = {k: self._zero_frame() for k in self._vision_input_names}
    transforms = {k: np.eye(3, dtype=np.float32) for k in (self._road_key, self._wide_key)}
    inputs = {k: np.zeros(v.shape, dtype=v.dtype) for k, v in self.numpy_inputs.items()}
    self.run(frames, transforms, inputs)
    for v in self.numpy_inputs.values():
      v[:] = 0
    self.prev_desire[:] = 0

  def _zero_frame(self) -> np.ndarray:
    nbytes = int(np.prod(self.input_shapes[self._road_key])) // 1
    return np.zeros(max(nbytes, 1), dtype=np.uint8)

  def slice_outputs(self, model_output: np.ndarray, slices: dict[str, slice]) -> dict[str, np.ndarray]:
    return {k: model_output[np.newaxis, v] for k, v in slices.items()}

  def _update_temporal(self, vision_output: np.ndarray, policy_output: np.ndarray | None) -> None:
    """Shift features_buffer once per frame using the hidden state from the
    engine that actually produced it.

    `vision_output` is indexed by vision slices, `policy_output` by policy
    slices. Only one of the two carries `hidden_state` for a given profile, and
    the slices are relative to that engine's own output - never to a
    concatenation of both.
    """
    slices, source = (self.vision_output_slices, vision_output)
    if "hidden_state" not in slices and policy_output is not None:
      slices, source = self.policy_output_slices, policy_output
    if "hidden_state" not in slices or source is None:
      return
    hidden = source[slices["hidden_state"]]
    feats = self.numpy_inputs.get("features_buffer")
    if feats is None:
      return
    window = self.profile.temporal.features_len if self.profile.temporal else hidden.size
    if feats.shape[-1] != window:
      raise RuntimeError(
        f"features_buffer last dim {feats.shape[-1]} != temporal.features_len {window}; "
        f"metadata and engine disagree"
      )
    feats[..., :-1, :] = feats[..., 1:, :]
    feats[..., -1, :] = hidden.reshape(feats.shape[-1])

  def run(self, bufs: dict[str, object], transforms: dict[str, np.ndarray],
          inputs: dict[str, np.ndarray], after_enqueue=None) -> dict[str, np.ndarray] | None:
    frames = {k: bufs[k] for k in self._vision_input_names if k in bufs}
    if len(frames) != len(self._vision_input_names):
      missing = sorted(set(self._vision_input_names) - set(frames))
      raise RuntimeError(f"missing vision frames for {missing}")
    self.backend.preprocess(frames, transforms)

    desire_in = inputs.get(self._desire_key)
    if desire_in is not None:
      current = np.asarray(desire_in).reshape(-1)[-self.constants.DESIRE_LEN:]
      pulse = np.where(current - self.prev_desire > 0.99, current, 0).astype(np.float32)
      target = self.numpy_inputs[self._desire_key]
      if target.ndim >= 2:
        target[..., :-1, :] = target[..., 1:, :]
        target[..., -1, :] = pulse
      else:
        target[:] = pulse
      self.prev_desire[:] = current

    for k in ("traffic_convention", "lateral_control_params", "action_t"):
      if k in self.numpy_inputs and k in inputs:
        self.numpy_inputs[k][:] = inputs[k]
    for k in ("tfm", "big_tfm"):
      src = self._road_key if k == "tfm" else self._wide_key
      if k in self.numpy_inputs and src in transforms:
        self.numpy_inputs[k][:, :] = transforms[src].reshape(3, 3)

    scalar_inputs = dict(self.numpy_inputs)

    if self.profile.mode == "merged":
      raw = self.backend.infer_merged(scalar_inputs)
      merged = {**self.vision_output_slices, **self.policy_output_slices}
      outputs = self.parser.parse_outputs(self.slice_outputs(raw, merged))
      self._update_temporal(raw, None)
    else:
      vision_out, policy_out = self.backend.infer_split(scalar_inputs)
      if self._vision_out_size is None:
        self._vision_out_size = int(np.asarray(vision_out).size)
      elif self._vision_out_size != int(np.asarray(vision_out).size):
        raise RuntimeError(
          f"vision engine output size changed: {self._vision_out_size} -> {np.asarray(vision_out).size}"
        )
      outputs = self.parser.parse_vision_outputs(self.slice_outputs(vision_out, self.vision_output_slices))
      outputs.update(self.parser.parse_policy_outputs(self.slice_outputs(policy_out, self.policy_output_slices)))
      self._update_temporal(vision_out, policy_out)

    if after_enqueue is not None:
      after_enqueue()

    if not self._outputs_finite(outputs):
      cloudlog.error("model output not finite, dropping frame")
      return None
    return outputs

  def _outputs_finite(self, outputs: dict[str, np.ndarray]) -> bool:
    for key in PLAN_KEYS:
      if key not in outputs:
        continue
      value = np.asarray(outputs[key], dtype=np.float64)
      if value.size and not np.all(np.isfinite(value)):
        cloudlog.error(f"non-finite model output in {key}")
        return False
    return True

  def get_action_from_model(self, model_output: dict[str, np.ndarray], prev_action: log.ModelDataV2.Action,
                            lat_action_t: float, long_action_t: float, v_ego: float) -> log.ModelDataV2.Action:
    if "action" not in model_output:
      plan = model_output["plan"][0]
      desired_accel = get_accel_from_plan(plan[:, Plan.VELOCITY][:, 0], plan[:, Plan.ACCELERATION][:, 0],
                                          self.constants.T_IDXS, action_t=long_action_t)
      desired_curvature = get_curvature_from_plan(plan[:, Plan.T_FROM_CURRENT_EULER][:, 2],
                                                  plan[:, Plan.ORIENTATION_RATE][:, 2],
                                                  self.constants.T_IDXS, v_ego, action_t=lat_action_t)
    else:
      desired_accel = model_output["action"][0, 1]
      desired_curvature = model_output["action"][0, 0] / (max(1.0, v_ego)) ** 2

    stop = should_stop(v_ego, desired_accel)
    desired_accel = smooth_value(desired_accel, prev_action.desiredAcceleration, self.LONG_SMOOTH_SECONDS)
    if v_ego > self.MIN_LAT_CONTROL_SPEED:
      desired_curvature = smooth_value(desired_curvature, prev_action.desiredCurvature, self.LAT_SMOOTH_SECONDS)
    else:
      desired_curvature = prev_action.desiredCurvature
    return log.ModelDataV2.Action(
      desiredCurvature=float(desired_curvature),
      desiredAcceleration=float(desired_accel),
      shouldStop=bool(stop),
    )
