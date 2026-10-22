"""The tuning metadata declared in params_keys.h reaches the settings catalog.

Two separate claims are checked here:

- The reader side: ``_declared_metadata`` turns the compiled key table into
  per-key numbers, and falls back to "not a knob, advanced" when the store has
  no opinion. That runs anywhere.
- The consumer side: ``carrot_catalog`` carries those numbers into the payload
  the app renders. Building the catalog pulls ``carrot.config`` -> hardware ->
  fcntl, so that one only runs on a device; skipping it on a PC host is an
  environment fact, not a gap in the check.
"""
import importlib.util
import unittest

from openpilot.sunnypilot.carrot.server.services.settings_sp import _declared_metadata

DEVICE_ONLY = unittest.skipUnless(importlib.util.find_spec("fcntl") is not None,
                                  "building the carrot catalog needs the device import chain")


class FakeMetadata:
  """A store that answers the metadata accessors, or refuses to."""

  def __init__(self, levels: dict | None = None, tiers: dict | None = None, *, broken: bool = False):
    self.levels = dict(levels or {})
    self.tiers = dict(tiers or {})
    self.broken = broken

  def get_tuning_level(self, key: str) -> int:
    if self.broken:
      raise OSError("no metadata here")
    return self.levels.get(key, 0)

  def get_settings_tier(self, key: str) -> int:
    if self.broken:
      raise OSError("no metadata here")
    return self.tiers.get(key, 1)


class TestDeclaredMetadata(unittest.TestCase):
  def test_a_declared_level_is_reported(self):
    levels, tiers = _declared_metadata(["CarrotEnabled"],
                                       FakeMetadata({"CarrotEnabled": 1}, {"CarrotEnabled": 0}))
    self.assertEqual(levels, {"CarrotEnabled": 1})
    self.assertEqual(tiers, {"CarrotEnabled": 0})

  def test_a_key_that_declares_nothing_is_not_a_knob(self):
    levels, tiers = _declared_metadata(["Unknown"], FakeMetadata())
    self.assertEqual(levels, {})
    self.assertEqual(tiers, {"Unknown": 1})

  def test_a_store_without_the_accessors_reports_no_levels(self):
    # A device whose libparams_c.so predates the metadata behaves exactly like
    # the pre-metadata build: every key is "advanced", none is a knob.
    class Bare:
      pass

    levels, tiers = _declared_metadata(["CarrotEnabled"], Bare())
    self.assertEqual(levels, {})
    self.assertEqual(tiers, {"CarrotEnabled": 1})

  def test_a_store_that_raises_falls_back_per_key(self):
    levels, tiers = _declared_metadata(["CarrotEnabled"], FakeMetadata(broken=True))
    self.assertEqual(levels, {})
    self.assertEqual(tiers, {"CarrotEnabled": 1})

  def test_no_store_at_all_is_not_an_error(self):
    self.assertEqual(_declared_metadata(["CarrotEnabled"], None), ({}, {}))


@DEVICE_ONLY
class TestCatalogCarriesTheMetadata(unittest.TestCase):
  def test_every_entry_carries_both_numbers(self):
    from openpilot.sunnypilot.carrot.server.services.settings_sp import carrot_catalog

    catalog = carrot_catalog()
    self.assertGreater(catalog["count"], 0)
    for name, entry in catalog["by_name"].items():
      self.assertIn("tuningLevel", entry, name)
      self.assertIn("settingsTier", entry, name)
      self.assertIsInstance(entry["tuningLevel"], int)
      self.assertIsInstance(entry["settingsTier"], int)

  def test_the_declared_knob_reports_its_level(self):
    from openpilot.sunnypilot.carrot.server.services.settings_sp import carrot_catalog

    entry = carrot_catalog()["by_name"]["CarrotEnabled"]
    self.assertEqual(entry["tuningLevel"], 1)
    self.assertEqual(entry["settingsTier"], 0)


if __name__ == "__main__":
  unittest.main()
