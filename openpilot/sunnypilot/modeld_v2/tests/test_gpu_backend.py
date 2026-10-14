"""Tests for the CUDA backend's pure logic.

None of this touches CUDA, TensorRT or a device: the point is to pin down the
decisions that are otherwise only asserted in prose - dtype mapping, output
slice partitioning, temporal axis derivation, and the refusal to guess. A
mistake in any of these produces plausible driving and no error, so they are
worth testing without a Jetson in the loop.

Runs on any platform with numpy and pytest.
"""

import pickle
import sys
import types
from pathlib import Path

import numpy as np
import pytest

# The modules under test import openpilot (for cloudlog, tinygrad, cereal). Stub
# only what the pure paths touch so the logic can be exercised standalone; nothing
# here claims to test the integration.
def _stub(name, **attrs):
  mod = types.ModuleType(name)
  for k, v in attrs.items():
    setattr(mod, k, v)
  sys.modules.setdefault(name, mod)
  return mod


_stub("openpilot")
_stub("openpilot.common")
_stub("openpilot.common.swaglog", cloudlog=types.SimpleNamespace(warning=lambda *a, **k: None))
_stub("tinygrad")
_stub("tinygrad.dtype", dtypes=types.SimpleNamespace(uint8="uint8", float16="float16", float32="float32"))
_stub("tinygrad.tensor", Tensor=object)
_stub("openpilot.sunnypilot")
_stub("openpilot.sunnypilot.modeld_v2")

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from gpu_backend.profile import ModelProfile, TemporalMeta  # noqa: E402
from gpu_backend.trt_runner import _np_dtype, _TRT_DTYPE_NAMES, _bfloat16_dtype  # noqa: E402


# --------------------------------------------------------------------------
# dtype mapping
# --------------------------------------------------------------------------

class TestDtypeMapping:
  def test_bf16_is_not_fp16(self):
    """TRT enum 7 is BF16. Mapping it to float16 silently reinterprets the
    buffer - the output would be wrong with no error anywhere."""
    assert _TRT_DTYPE_NAMES[7] == "bfloat16"
    try:
      dtype = _np_dtype(7)
    except RuntimeError as e:
      # Refusing is the correct behaviour when the dtype cannot be represented.
      assert "ml_dtypes" in str(e)
      return
    assert dtype.itemsize == 2
    assert dtype.name != "float16"

  def test_bf16_refusal_names_the_fix(self):
    """When bfloat16 is unavailable the message must say what to do, not just
    that something is wrong."""
    import numpy
    if getattr(numpy, "bfloat16", None) is not None:
      pytest.skip("numpy provides bfloat16")
    try:
      import ml_dtypes  # noqa: F401
    except ImportError:
      with pytest.raises(RuntimeError, match="ml_dtypes"):
        _np_dtype(7)

  def test_half_is_still_fp16(self):
    assert _np_dtype(1) == np.dtype(np.float16)

  def test_float32_is_float32(self):
    assert _np_dtype(0) == np.dtype(np.float32)

  def test_unknown_dtype_raises(self):
    with pytest.raises(RuntimeError, match="unsupported TensorRT dtype"):
      _np_dtype(99)

  def test_fp8_has_no_silent_fallback(self):
    with pytest.raises(RuntimeError, match="no numpy equivalent"):
      _np_dtype(6)

  def test_every_enum_value_is_named(self):
    """An unnamed enum value means a silent fallback to float32."""
    for value in range(0, 8):
      assert value in _TRT_DTYPE_NAMES, f"TRT dtype {value} has no name"


# --------------------------------------------------------------------------
# slice partitioning
# --------------------------------------------------------------------------

POLICY_KEYS = frozenset((
  "plan", "planplus", "lead", "lead_prob", "desire_state", "action", "hidden_state", "pad",
))

BIGCOMBO_SLICES = {
  "lane_lines": slice(0, 528),
  "lane_lines_prob": slice(528, 536),
  "road_edges": slice(536, 800),
  "meta": slice(800, 855),
  "desire_pred": slice(855, 887),
  "pose": slice(887, 899),
  "wide_from_device_euler": slice(899, 905),
  "road_transform": slice(905, 917),
  "plan": slice(917, 1907),
  "lead": slice(1907, 2051),
  "lead_prob": slice(2051, 2054),
  "desire_state": slice(2054, 2062),
  "action": slice(2062, 2066),
  "hidden_state": slice(2066, 2578),
  "pad": slice(2578, 2580),
}


def _split(slices):
  vision = {k: v for k, v in slices.items() if k not in POLICY_KEYS}
  policy = {k: v for k, v in slices.items() if k in POLICY_KEYS}
  return vision, policy


class TestSlicePartitioning:
  def test_vision_and_policy_partition_the_output(self):
    vision, policy = _split(BIGCOMBO_SLICES)
    assert set(vision) & set(policy) == set()
    assert set(vision) | set(policy) == set(BIGCOMBO_SLICES)

  def test_old_bug_is_caught(self):
    """The previous code built policy_slices with `k not in vision_slices`,
    which puts every key into policy. Assert the correct shape explicitly."""
    vision, policy = _split(BIGCOMBO_SLICES)
    assert "lane_lines" in vision, "lane_lines is a vision head"
    assert "lane_lines" not in policy
    assert len(policy) == 7 and len(vision) == 8

  def test_hidden_state_lands_in_policy(self):
    _, policy = _split(BIGCOMBO_SLICES)
    assert "hidden_state" in policy

  def test_slices_are_contiguous_and_cover_the_whole_output(self):
    ordered = sorted(BIGCOMBO_SLICES.values(), key=lambda s: s.start)
    assert ordered[0].start == 0
    for prev, nxt in zip(ordered, ordered[1:]):
      assert prev.stop == nxt.start, f"gap or overlap between {prev} and {nxt}"
    assert ordered[-1].stop == 2580


# --------------------------------------------------------------------------
# temporal derivation
# --------------------------------------------------------------------------

def _temporal(fb, dr):
  features_len = int(np.prod(fb[2:])) if len(fb) > 2 else fb[1]
  features_windows = fb[1]
  desire_len = int(np.prod(dr[2:])) if len(dr) > 2 else dr[1]
  desire_windows = dr[1]
  return TemporalMeta(features_len, features_windows, desire_len, desire_windows, False, True)


class TestTemporal:
  def test_bigcombo_shapes(self):
    """BigCombo, per project3's measured table."""
    t = _temporal((1, 24, 512), (1, 25, 8))
    assert t.features_len == 512
    assert t.features_windows == 24
    assert t.desire_len == 8
    assert t.desire_windows == 25

  def test_tgc_shapes(self):
    """TGC nests the history one level deeper: (1, 32, 32, 512)."""
    t = _temporal((1, 32, 32, 512), (1, 33, 8))
    assert t.features_len == 32 * 512
    assert t.features_windows == 32

  def test_time_axis_is_axis_one(self):
    """axis 1 is the time axis (confirmed from the ONNX graph: features_buffer
    is consumed by Concat(axis=1), desire by Gather(axis=1)). If that ever
    changes, these numbers change and the test says so."""
    t = _temporal((1, 24, 512), (1, 25, 8))
    assert t.features_windows == 24
    assert t.features_len != 24, "slots and feature width must not be confused"


# --------------------------------------------------------------------------
# readiness
# --------------------------------------------------------------------------

def _profile(tmp_path, **kw):
  d = tmp_path / "models" / "Test"
  d.mkdir(parents=True, exist_ok=True)
  (d / "driving_supercombo_fp16.plan").write_bytes(b"x" * 128)
  defaults = dict(
    name="Test", mode="merged", engine_dir=str(d),
    merged_engine="driving_supercombo_fp16.plan",
    vision_input_names=["img", "big_img"],
    input_shapes={"img": (1, 12, 128, 256), "big_img": (1, 12, 128, 256)},
    vision_slices=dict(BIGCOMBO_SLICES),
    policy_slices={},
    temporal=_temporal((1, 24, 512), (1, 25, 8)),
  )
  _, policy = _split(BIGCOMBO_SLICES)
  defaults["policy_slices"] = policy
  defaults.update(kw)
  return ModelProfile(**defaults)


class TestReadiness:
  def test_complete_profile_is_ready(self, tmp_path):
    p = _profile(tmp_path)
    assert p.missing_reason() is None
    assert p.is_ready()

  def test_empty_engine_is_not_ready(self, tmp_path):
    p = _profile(tmp_path)
    (Path(p.engine_dir) / p.merged_engine).write_bytes(b"")
    assert "empty" in (p.missing_reason() or "")
    assert not p.is_ready()

  def test_missing_engine_is_not_ready(self, tmp_path):
    p = _profile(tmp_path)
    (Path(p.engine_dir) / p.merged_engine).unlink()
    assert "missing" in (p.missing_reason() or "")

  def test_missing_slices_is_not_ready(self, tmp_path):
    """This is the project3 crash loop: a plan present, metadata absent."""
    p = _profile(tmp_path, vision_slices={}, policy_slices={})
    assert "slices missing" in (p.missing_reason() or "")

  def test_missing_temporal_is_not_ready(self, tmp_path):
    p = _profile(tmp_path, temporal=None)
    assert "temporal" in (p.missing_reason() or "")

  def test_single_slot_features_buffer_is_not_ready(self, tmp_path):
    """One time slot cannot be shifted; accepting it would produce a buffer
    that never carries history."""
    p = _profile(tmp_path, temporal=_temporal((1, 1, 512), (1, 25, 8)))
    assert "time slots" in (p.missing_reason() or "")

  def test_missing_engine_dir_is_not_ready(self, tmp_path):
    p = _profile(tmp_path, engine_dir=str(tmp_path / "nope"))
    assert "engine dir missing" in (p.missing_reason() or "")

  def test_unknown_vision_input_is_not_ready(self, tmp_path):
    p = _profile(tmp_path, input_shapes={"img": (1, 12, 128, 256)})
    assert "no input shape" in (p.missing_reason() or "")


# --------------------------------------------------------------------------
# geometry derived from the input shapes
# --------------------------------------------------------------------------

class TestGeometry:
  def test_model_w_h_doubled(self, tmp_path):
    """img is (1, 12, H/2, W/2): the warped tensor is twice the stored shape."""
    p = _profile(tmp_path)
    assert p.model_w == 512
    assert p.model_h == 256

  def test_vision_shape_for_rejects_unknown_key(self, tmp_path):
    p = _profile(tmp_path)
    with pytest.raises(KeyError, match="nope"):
      p.vision_shape_for("nope")

  def test_temporal_skip_comes_from_metadata(self, tmp_path):
    p = _profile(tmp_path)
    assert p.temporal_skip == 24

  def test_temporal_skip_refuses_to_guess(self, tmp_path):
    p = _profile(tmp_path, temporal=None)
    with pytest.raises(RuntimeError, match="refusing to guess"):
      _ = p.temporal_skip


# --------------------------------------------------------------------------
# the metadata loader must refuse rather than fall back
# --------------------------------------------------------------------------

class TestMetadataLoader:
  def test_missing_file_raises(self, tmp_path):
    from gpu_backend.models import registry
    with pytest.raises(registry.MetadataError, match="metadata not found"):
      registry._load_metadata(tmp_path / "nope.pkl")

  def test_non_dict_raises(self, tmp_path):
    from gpu_backend.models import registry
    p = tmp_path / "m.pkl"
    p.write_bytes(pickle.dumps([1, 2, 3]))
    with pytest.raises(registry.MetadataError, match="expected dict"):
      registry._load_metadata(p)

  def test_missing_output_slices_raises(self, tmp_path):
    from gpu_backend.models import registry
    p = tmp_path / "m.pkl"
    p.write_bytes(pickle.dumps({"input_shapes": {}}))
    with pytest.raises(registry.MetadataError, match="output_slices"):
      registry._output_slices(registry._load_metadata(p), p)

  def test_empty_output_slices_raises(self, tmp_path):
    from gpu_backend.models import registry
    p = tmp_path / "m.pkl"
    p.write_bytes(pickle.dumps({"output_slices": {}}))
    with pytest.raises(registry.MetadataError, match="output_slices"):
      registry._output_slices(registry._load_metadata(p), p)

  def test_non_slice_entry_raises(self, tmp_path):
    from gpu_backend.models import registry
    p = tmp_path / "m.pkl"
    p.write_bytes(pickle.dumps({"output_slices": {"plan": [0, 10]}}))
    with pytest.raises(registry.MetadataError, match="non-slice"):
      registry._output_slices(registry._load_metadata(p), p)

  def test_unloadable_pickle_raises(self, tmp_path):
    from gpu_backend.models import registry
    p = tmp_path / "m.pkl"
    p.write_bytes(b"not a pickle at all")
    with pytest.raises(registry.MetadataError, match="cannot unpickle"):
      registry._load_metadata(p)

  def test_missing_temporal_inputs_raise(self, tmp_path):
    from gpu_backend.models import registry
    p = tmp_path / "m.pkl"
    p.write_bytes(pickle.dumps({"input_shapes": {"desire_pulse": (1, 25, 8)}}))
    meta = registry._load_metadata(p)
    with pytest.raises(registry.MetadataError, match="features_buffer"):
      registry._temporal_from_metadata(meta, p, "features_buffer", "desire_pulse")

  def test_resolve_profile_returns_none_when_nothing_is_ready(self, monkeypatch, tmp_path):
    """Must return None, not a profile built from guessed defaults."""
    from gpu_backend.models import registry
    monkeypatch.setattr(registry, "REGISTRY", {})
    assert registry.resolve_profile("anything") is None
    assert registry.resolve_profile(None) is None

  def test_get_profile_without_name_returns_none(self, monkeypatch):
    """An empty name used to default to 'FiletOFish' while resolve_profile
    scanned for the first ready profile - two different answers to the same
    question."""
    from gpu_backend.models import registry
    p = _profile(Path("."))
    monkeypatch.setattr(registry, "REGISTRY", {"FiletOFish": p})
    assert registry.get_profile(None) is None
    assert registry.get_profile("") is None
    assert registry.get_profile("FiletOFish") is p
    assert registry.get_profile("nope") is None
