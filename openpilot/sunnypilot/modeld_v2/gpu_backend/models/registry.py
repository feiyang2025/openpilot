import pickle
from pathlib import Path

from openpilot.sunnypilot.modeld_v2.gpu_backend.profile import ModelProfile, TemporalMeta

MODELS_ROOT = Path(__file__).parents[4] / "selfdrive" / "modeld" / "models"

# Output heads produced by the policy stage of a split bundle. Everything else in
# a merged supercombo output belongs to the vision stage.
POLICY_KEYS = frozenset((
  "plan",
  "planplus",
  "lead",
  "lead_prob",
  "desire_state",
  "action",
  "hidden_state",
  "pad",
))

VISION_ENGINE_SHAPES = {
  # BigCombo (merged) 引擎的绑定名
  "img": (1, 12, 128, 256),
  "big_img": (1, 12, 128, 256),
  # FiletOFish (split) 引擎实测绑定名 —— engine 里就叫 input_imgs/big_input_imgs。
  # TrtRunner 按引擎实际绑定名取 kwargs (trt_runner.py:196), 所以每个 profile 必须
  # 用自己 engine 的真实名字, 否则 kwargs.get('img') 拿到 None 直接报错。
  "input_imgs": (1, 12, 128, 256),
  "big_input_imgs": (1, 12, 128, 256),
}


class MetadataError(RuntimeError):
  pass


def _load_metadata(metadata_path: Path) -> dict:
  if not metadata_path.is_file():
    raise MetadataError(f"metadata not found: {metadata_path}")
  try:
    with open(metadata_path, "rb") as f:
      metadata = pickle.load(f)
  except Exception as e:
    raise MetadataError(f"cannot unpickle {metadata_path}: {e}") from e
  if not isinstance(metadata, dict):
    raise MetadataError(f"{metadata_path} holds {type(metadata).__name__}, expected dict")
  return metadata


def _output_slices(metadata: dict, source: Path) -> dict[str, slice]:
  slices = metadata.get("output_slices")
  if not isinstance(slices, dict) or not slices:
    raise MetadataError(f"{source} has no usable output_slices (got {type(slices).__name__})")
  bad = [k for k, v in slices.items() if not isinstance(v, slice)]
  if bad:
    raise MetadataError(f"{source} output_slices has non-slice entries: {bad}")
  return slices


def _scalars(shapes: dict, vision_names) -> dict:
  """Metadata input_shapes lists the image tensors too.

  Those are not scalars: they are filled by the CUDA transform, which hands the
  backend a device pointer. Leaving them in this dict makes callers build a zero
  array for them and then overwrite the pointer, so every inference silently runs
  on a black image. They are dropped here once instead of at each call site.
  """
  return {k: v for k, v in shapes.items() if k not in set(vision_names)}


def _split_slices(slices: dict[str, slice]) -> tuple[dict[str, slice], dict[str, slice]]:
  vision = {k: v for k, v in slices.items() if k not in POLICY_KEYS}
  policy = {k: v for k, v in slices.items() if k in POLICY_KEYS}
  if not vision or not policy:
    raise MetadataError(
      f"cannot split output_slices: vision={sorted(vision)} policy={sorted(policy)}; "
      f"POLICY_KEYS may not match this engine"
    )
  return vision, policy


def _temporal_from_metadata(metadata: dict, source: Path, features_key: str, desire_key: str) -> TemporalMeta:
  shapes = metadata.get("input_shapes")
  if not isinstance(shapes, dict):
    raise MetadataError(f"{source} has no input_shapes")
  for key in (features_key, desire_key):
    if key not in shapes:
      raise MetadataError(f"{source} input_shapes missing {key}; has {sorted(shapes)}")
  fb = tuple(shapes[features_key])
  dr = tuple(shapes[desire_key])
  # axis 1 is the time axis for both (confirmed from the ONNX graph: features_buffer
  # is consumed by Concat(axis=1), desire by Gather(axis=1)).
  if len(fb) < 2 or len(dr) < 2:
    raise MetadataError(f"{source} expects >=2D {features_key}={fb} and {desire_key}={dr}")
  features_len = int(np_prod(fb[2:])) if len(fb) > 2 else fb[1]
  features_windows = fb[1]
  desire_len = int(np_prod(dr[2:])) if len(dr) > 2 else dr[1]
  desire_windows = dr[1]
  if features_windows < 2:
    raise MetadataError(f"{source} features_buffer has {features_windows} time slots, needs >= 2 to shift")
  return TemporalMeta(
    features_len=features_len,
    features_windows=features_windows,
    desire_len=desire_len,
    desire_windows=desire_windows,
    features_includes_current=False,
    desire_includes_current=True,
  )


def np_prod(values) -> int:
  total = 1
  for v in values:
    total *= int(v)
  return total


def _engine(engine_dir: Path, *names: str) -> str | None:
  # A zero-byte engine is a trtexec run that died partway through; it must not win
  # the candidate search, or it shadows a perfectly good plan sitting later in the
  # list and the profile just reports itself unready.
  for name in names:
    path = engine_dir / name
    if path.is_file() and path.stat().st_size > 0:
      return name
  return None


def _filet_ofish() -> ModelProfile:
  engine_dir = MODELS_ROOT / "FiletOFish"
  vision_engine = _engine(engine_dir, "driving_vision_fp16.plan", "driving_vision.plan")
  policy_engine = _engine(engine_dir, "driving_policy_fp16.plan", "driving_policy.plan")
  if vision_engine is None or policy_engine is None:
    raise MetadataError(f"FiletOFish: missing split engines in {engine_dir}")

  vision_meta = _load_metadata(engine_dir / "driving_vision_metadata.pkl")
  policy_meta = _load_metadata(engine_dir / "driving_policy_metadata.pkl")

  return ModelProfile(
    name="FiletOFish",
    mode="split",
    engine_dir=str(engine_dir),
    vision_engine=vision_engine,
    policy_engine=policy_engine,
    vision_input_names=["input_imgs", "big_input_imgs"],
    input_shapes=dict(VISION_ENGINE_SHAPES),
    vision_slices=_output_slices(vision_meta, engine_dir / "driving_vision_metadata.pkl"),
    policy_slices=_output_slices(policy_meta, engine_dir / "driving_policy_metadata.pkl"),
    temporal=_temporal_from_metadata(policy_meta, engine_dir / "driving_policy_metadata.pkl",
                                     "features_buffer", "desire"),
    scalar_input_shapes=_scalars(policy_meta.get("input_shapes") or {}, ["input_imgs", "big_input_imgs"]),
  )


def _big_combo() -> ModelProfile:
  engine_dir = MODELS_ROOT / "BigCombo"
  merged_engine = _engine(engine_dir, "driving_supercombo_fp16.plan", "driving_supercombo.plan")
  if merged_engine is None:
    raise MetadataError(f"BigCombo: missing merged engine in {engine_dir}")

  metadata_path = engine_dir / "driving_supercombo_metadata.pkl"
  metadata = _load_metadata(metadata_path)
  slices = _output_slices(metadata, metadata_path)
  vision, policy = _split_slices(slices)

  shapes = metadata.get("input_shapes") or {}
  if not shapes:
    raise MetadataError(f"{metadata_path} has no input_shapes; cannot size the scalar inputs")

  return ModelProfile(
    name="BigCombo",
    mode="merged",
    engine_dir=str(engine_dir),
    merged_engine=merged_engine,
    vision_input_names=["img", "big_img"],
    input_shapes={**VISION_ENGINE_SHAPES, **shapes},
    vision_slices=vision,
    policy_slices=policy,
    temporal=_temporal_from_metadata(metadata, metadata_path, "features_buffer", "desire_pulse"),
    scalar_input_shapes=_scalars(shapes, ["img", "big_img"]),
  )


def _supercombo() -> ModelProfile:
  # The small model the bundle actually ships (driving_supercombo.onnx): one merged
  # engine, FLOAT16 img inputs after the UINT8 rewrite, features_buffer (1, 24, 512),
  # desire_pulse (1, 25, 8), output 2576 wide. Same layout as BigCombo/TGC, so this
  # is that code path with a different directory.
  engine_dir = MODELS_ROOT / "supercombo"
  # No _fp16 engine on purpose: the 20 Gelu(approximate=0) nodes have no fp16
  # implementation on sm_87, so trtexec fails with "Could not find any
  # implementation for node_conv2d_1 + node_gelu_1". The fp32 engine builds and
  # runs at 14.49 ms / 69 Hz, well inside the 20 Hz budget.
  merged_engine = _engine(engine_dir, "driving_supercombo_fp16.plan", "driving_supercombo_fp32.plan",
                          "driving_supercombo.plan")
  if merged_engine is None:
    raise MetadataError(f"supercombo: missing merged engine in {engine_dir}")

  metadata_path = engine_dir / "driving_supercombo_metadata.pkl"
  metadata = _load_metadata(metadata_path)
  slices = _output_slices(metadata, metadata_path)
  vision, policy = _split_slices(slices)

  shapes = metadata.get("input_shapes") or {}
  if not shapes:
    raise MetadataError(f"{metadata_path} has no input_shapes; cannot size the scalar inputs")

  return ModelProfile(
    name="supercombo",
    mode="merged",
    engine_dir=str(engine_dir),
    merged_engine=merged_engine,
    vision_input_names=["img", "big_img"],
    input_shapes={**VISION_ENGINE_SHAPES, **shapes},
    vision_slices=vision,
    policy_slices=policy,
    temporal=_temporal_from_metadata(metadata, metadata_path, "features_buffer", "desire_pulse"),
    scalar_input_shapes=_scalars(shapes, ["img", "big_img"]),
  )


def _tgc() -> ModelProfile:
  # The big model sunnypilot currently publishes as `big_driving_supercombo.onnx`
  # is TGC-shaped, not the older BigCombo: desire_pulse is (1, 33, 8),
  # features_buffer is (1, 32, 32, 512) and the output is 18452 wide. Same input
  # names and the same merged single-engine layout, so it reuses the BigCombo
  # code path; only the temporal geometry differs and that is read from metadata.
  engine_dir = MODELS_ROOT / "TGC"
  merged_engine = _engine(engine_dir, "driving_supercombo_fp16.plan", "driving_supercombo.plan")
  if merged_engine is None:
    raise MetadataError(f"TGC: missing merged engine in {engine_dir}")

  metadata_path = engine_dir / "driving_supercombo_metadata.pkl"
  metadata = _load_metadata(metadata_path)
  slices = _output_slices(metadata, metadata_path)
  vision, policy = _split_slices(slices)

  shapes = metadata.get("input_shapes") or {}
  if not shapes:
    raise MetadataError(f"{metadata_path} has no input_shapes; cannot size the scalar inputs")

  return ModelProfile(
    name="TGC",
    mode="merged",
    engine_dir=str(engine_dir),
    merged_engine=merged_engine,
    vision_input_names=["img", "big_img"],
    input_shapes={**VISION_ENGINE_SHAPES, **shapes},
    vision_slices=vision,
    policy_slices=policy,
    temporal=_temporal_from_metadata(metadata, metadata_path, "features_buffer", "desire_pulse"),
    scalar_input_shapes=_scalars(shapes, ["img", "big_img"]),
  )


def _build(name: str, factory) -> ModelProfile | None:
  try:
    return factory()
  except Exception as e:
    import logging
    logging.getLogger(__name__).warning(f"profile {name} unusable: {e}")
    return None


REGISTRY: dict[str, ModelProfile | None] = {
  # Order is the default preference. `supercombo` is the small model the bundle
  # ships and is what runs unless a big model is asked for by name; TGC is that big
  # model. FiletOFish is a different split model (632 + 5884) kept for the profile
  # project3 ran on the Orin - it is NOT the bundle's small model, so it stays last.
  "supercombo": _build("supercombo", _supercombo),
  "BigCombo": _build("BigCombo", _big_combo),
  "TGC": _build("TGC", _tgc),
  "FiletOFish": _build("FiletOFish", _filet_ofish),
}


def get_profile(name: str | None) -> ModelProfile | None:
  if not name:
    return None
  return REGISTRY.get(name)


def resolve_profile(name: str | None = None) -> ModelProfile | None:
  """Pick the first profile that is actually usable on this machine.

  Priority:
    1. explicit `name` (from Params Model / env), if it builds and has its engines;
    2. any ready profile in REGISTRY order;
    3. None -> caller falls back to the stock tinygrad ModelState.
  """
  if name:
    profile = get_profile(name)
    if profile is not None and profile.is_ready():
      return profile
  for profile in REGISTRY.values():
    if profile is not None and profile.is_ready():
      return profile
  return None
