#pragma once

#include <string>

#include "common/hardware/base.h"

class HardwarePC : public HardwareNone {
public:
  static std::string get_name() { return "pc"; }
  static cereal::InitData::DeviceType get_device_type() { return cereal::InitData::DeviceType::PC; }
  static bool PC() { return true; }
  static int get_voltage() { return 0; }
  static int get_current() { return 0; }
};
