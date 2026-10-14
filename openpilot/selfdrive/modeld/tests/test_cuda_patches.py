"""Tests for the runtime patch mechanism itself.

cuda_patches is the reason no upstream file has to be edited: sitecustomize runs
it at interpreter start and it swaps classes in place. If that mechanism is
wrong, the fork either silently does nothing or breaks every python process on
the device. Both failure modes are silent, so they get tests.

These exercise the mechanism against fake modules - no CUDA, no TensorRT.
"""

import sys
import types

import pytest

from openpilot.selfdrive.modeld import cuda_patches


def _fresh_module(name):
  """Register an empty module so patch_after_import takes the 'already imported'
  branch, then let the test populate it."""
  sys.modules.pop(name, None)
  mod = types.ModuleType(name)
  sys.modules[name] = mod
  return mod


def _absent(name):
  sys.modules.pop(name, None)
  return name


class TestPatchAfterImport:
  def test_calls_back_on_an_already_imported_module(self):
    mod = _fresh_module("fake_already")
    seen = []
    assert cuda_patches.patch_after_import("fake_already", lambda m: seen.append(m) or True)
    assert seen == [mod]

  def test_does_not_import_the_target(self):
    """Importing at startup is a behaviour change: startup cost, and tinygrad's
    device selection must happen before the first tinygrad import."""
    _absent("fake_absent_never_imported")
    cuda_patches.patch_after_import("fake_absent_never_imported", lambda m: True)
    assert "fake_absent_never_imported" not in sys.modules

  def test_fires_when_the_target_imports_later(self, tmp_path, monkeypatch):
    """The callback must run once the module body has executed - patching before
    exec_module would see an empty module."""
    monkeypatch.syspath_prepend(str(tmp_path))
    name = _absent("deferred_target")
    (tmp_path / "deferred_target.py").write_text("NAME = 'deferred_target'\n")

    fired = []
    assert cuda_patches.patch_after_import(name, lambda m: fired.append(m.NAME) or True)
    assert fired == []

    mod = __import__(name)
    assert mod.NAME == name
    assert fired == [name], "callback must run after the module body executes"

  def test_finder_is_removed_after_use(self, tmp_path, monkeypatch):
    """A finder that stays on sys.meta_path is consulted for every later import
    in the process."""
    monkeypatch.syspath_prepend(str(tmp_path))
    name = _absent("removed_target")
    (tmp_path / "removed_target.py").write_text("VALUE = 1\n")

    before = len(sys.meta_path)
    cuda_patches.patch_after_import(name, lambda m: True)
    assert len(sys.meta_path) == before + 1

    __import__(name)
    assert len(sys.meta_path) == before, "the finder must remove itself once used"

  def test_finder_ignores_other_modules(self):
    cuda_patches.patch_after_import(_absent("fake_only_me"), lambda m: True)
    finder = next(f for f in sys.meta_path if type(f).__name__ == "_Finder")
    assert finder.find_spec("something_else") is None
    sys.meta_path.remove(finder)

  def test_a_raising_callback_does_not_break_the_import(self, tmp_path, monkeypatch):
    """The hook runs inside every python process on the device; it must not be
    able to stop the module from loading."""
    monkeypatch.syspath_prepend(str(tmp_path))
    name = _absent("raising_target")
    (tmp_path / "raising_target.py").write_text("VALUE = 42\n")

    cuda_patches.patch_after_import(name, lambda m: (_ for _ in ()).throw(RuntimeError("boom")))
    mod = __import__(name)
    assert mod.VALUE == 42, "a failing patch must not prevent the import"


class TestIdempotence:
  def test_applying_twice_changes_nothing_the_second_time(self):
    mod = _fresh_module("fake_idem")
    mod.ModelState = type("ModelState", (), {"__init__": lambda self, *a: None})
    assert cuda_patches.patch_modeld_state(mod) is True
    first = mod.ModelState.__init__
    assert cuda_patches.patch_modeld_state(mod) is False
    assert mod.ModelState.__init__ is first

  def test_marker_is_set_on_the_module(self):
    mod = _fresh_module("fake_idem2")
    mod.ModelState = type("ModelState", (), {"__init__": lambda self, *a: None})
    cuda_patches.patch_modeld_state(mod)
    assert mod._fork_cuda_modelstate is True

  def test_camerad_patch_is_idempotent(self):
    mod = _fresh_module("fake_camerad")
    mod.Camera = object
    first = cuda_patches.patch_camerad_camera(mod)
    second = cuda_patches.patch_camerad_camera(mod)
    assert second is False
    assert first in (True, False)


class TestCameraPatchGating:
  def test_disabled_by_env(self, monkeypatch):
    monkeypatch.setenv("USE_V4L2_CAMERA", "0")
    mod = _fresh_module("fake_cam_off")
    original = mod.Camera = object
    cuda_patches.patch_camerad_camera(mod)
    assert mod.Camera is original

  def test_disabled_on_darwin(self, monkeypatch):
    monkeypatch.setenv("USE_V4L2_CAMERA", "1")
    monkeypatch.setattr(cuda_patches.sys, "platform", "darwin")
    mod = _fresh_module("fake_cam_mac")
    original = mod.Camera = object
    cuda_patches.patch_camerad_camera(mod)
    assert mod.Camera is original

  def test_missing_v4l2_module_leaves_the_original(self, monkeypatch):
    monkeypatch.setenv("USE_V4L2_CAMERA", "1")
    monkeypatch.setattr(cuda_patches.sys, "platform", "linux")
    mod = _fresh_module("fake_cam_missing")
    original = mod.Camera = object
    result = cuda_patches.patch_camerad_camera(mod)
    assert result is False
    assert mod.Camera is original
    assert "v4l2 camera unavailable" in cuda_patches._PATCH_STATE.get("camera", "")


class TestMasterSwitch:
  def test_env_zero_disables_everything(self, monkeypatch):
    monkeypatch.setenv("OPENPILOT_CUDA_PATCH", "0")
    assert cuda_patches.enabled() is False
    assert cuda_patches.apply() == []

  def test_default_is_enabled(self, monkeypatch):
    monkeypatch.delenv("OPENPILOT_CUDA_PATCH", raising=False)
    assert cuda_patches.enabled() is True

  def test_the_switch_is_read_at_call_time(self, monkeypatch):
    """sitecustomize imports this module before anything could set the variable.
    A value captured at import would make the escape hatch unchangeable for the
    life of the process - and untestable."""
    assert cuda_patches.enabled() is True
    monkeypatch.setenv("OPENPILOT_CUDA_PATCH", "0")
    assert cuda_patches.enabled() is False


class TestDisableBackendEnv:
  def test_disable_cuda_backend_skips_the_modeld_patch(self, monkeypatch):
    monkeypatch.setenv("DISABLE_CUDA_BACKEND", "1")
    mod = _fresh_module("fake_disable")
    original_init = lambda self, *a: None
    mod.ModelState = type("ModelState", (), {"__init__": original_init})
    assert cuda_patches.patch_modeld_state(mod) is False
    assert mod.ModelState.__init__ is original_init

  def test_disable_cuda_backend_zero_does_not_skip(self, monkeypatch):
    monkeypatch.setenv("DISABLE_CUDA_BACKEND", "0")
    mod = _fresh_module("fake_enable")
    mod.ModelState = type("ModelState", (), {"__init__": lambda self, *a: None})
    assert cuda_patches.patch_modeld_state(mod) is True
