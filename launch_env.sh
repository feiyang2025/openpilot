#!/usr/bin/env bash

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# models get lower priority than ui
# - ui is ~5ms
# - modeld is 20ms
# - DM is 10ms
# in order to run ui at 60fps (16.67ms), we need to allow
# it to preempt the model workloads. we have enough
# headroom for this until ui is moved to the CPU.
export QCOM_PRIORITY=12

if [ -z "$AGNOS_VERSION" ]; then
  export AGNOS_VERSION="19.8-carrot-bt2"
fi

export STAGING_ROOT="/data/safe_staging"

# Jetson dev/bench mode: the Qualcomm-only camerad (spectra.cc) asserts here, so
# use the Python webcamerad for the GMSL cameras. Only applies on non-TICI
# aarch64 (Jetson); real comma hardware is unaffected.
if [ "$(uname -m)" = "aarch64" ] && [ ! -e /TICI ]; then
  export SKIP_FW_QUERY="1"
  export USE_WEBCAM="1"
  # 摄像头节点号由下方 resolve_cams.sh 按设备名动态解析, 勿硬编码
  export CAM_WIDTH="1344"
  export CAM_HEIGHT="760"
  export CAM_BRIGHTNESS="1.0"

  # 本机(Orin)没有高通 Adreno(/dev/kgsl-3d0), 上游模型包里的 *_tinygrad.pkl 是
  # QCOM 编译版, modeld_tinygrad 一加载就崩 FileNotFoundError。指向本机编译的
  # tinygrad pkl(compile_modeld.py 产物, 不入库; 缺失时跑
  # scripts/build_runtime.sh 或参照交付记录阶段 7 生成)。
  # 这个分支同样覆盖 PC aarch64, 但 PC 上没有该 pkl 时 modeld 会明确报错, 可接受。
  if [ -z "${COMBINED_MODEL_PKL:-}" ] && [ -f "$DIR/openpilot/selfdrive/modeld/models/driving_tinygrad.pkl" ]; then
    export COMBINED_MODEL_PKL="$DIR/openpilot/selfdrive/modeld/models/driving_tinygrad.pkl"
  fi
fi

# v4l2 节点号随枚举顺序漂移: 启动时按设备名解析 (ROAD=120° 广角主路 / WIDE=60° 窄角 / DRIVER=USB)
if [ -f "$(dirname "${BASH_SOURCE[0]}")/scripts/resolve_cams.sh" ]; then
  eval "$(bash "$(dirname "${BASH_SOURCE[0]}")/scripts/resolve_cams.sh" 2>/dev/null)"
fi
export ROAD_CAM="${ROAD_CAM:-}"
export WIDE_CAM="${WIDE_CAM:-}"
export DRIVER_CAM="${DRIVER_CAM:-}"

# PC 上跑大 UI (C4 那套是小屏; PC 需要 BIG=1 才是大画布)
export BIG="1"
export SCALE="1.0"
export UI_WIDTH="1920"
export UI_HEIGHT="1048"
