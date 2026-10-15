#!/usr/bin/env python3
"""Chroma / geometry health probe for the *published* camera streams.

Why
---
Tearing and "green picture" look like image bugs but are decided by numbers:

* **Green**: U and V drift away from 128 *together* (especially both downward).
  U<->V swapped shows as magenta/blue, not green - so on a green frame do not
  blame the swap first, blame a wrong layout or a misplaced chroma plane.
* **Tearing**: a mixed frame (this exposure's top half + the previous
  exposure's bottom half) shows up as a discontinuity row, and as a row whose
  content is ~50 ms old.
* **Static frames fool you**: with no motion a mixed frame's two halves are
  identical, and a screenshot reads the compositor's finished frame, so neither
  reveals the problem. Measure the numbers instead.

This reads the frames over VisionIPC, i.e. exactly what modeld/UI receive -
not the camera node - so a problem introduced after capture (layout decode,
stride/alignment, missing sync before send) is what gets measured.
`roadCameraState.image` is empty on this stack (FrameData carries no image
here), so msgq is the only way to see the published bytes.

Usage
-----
    PYTHONPATH="$PWD:$PWD/msgq_repo" .venv/bin/python tools/webcam/chroma_probe.py --dur 5
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time

import numpy as np

try:
  from msgq.visionipc import VisionIpcClient, VisionStreamType
  _ROAD, _WIDE = VisionStreamType.VISION_STREAM_ROAD, VisionStreamType.VISION_STREAM_WIDE_ROAD
except Exception:  # pragma: no cover - schema/fork differences
  VisionIpcClient = None
  _ROAD, _WIDE = 0, 2

from openpilot.cereal import messaging

STREAMS = {"ROAD": _ROAD, "WIDE": _WIDE}


def _plane_stats(data: bytes, width: int, height: int, stride: int) -> dict:
  """Y / UV stats of one NV12 frame. `stride` is the row pitch in bytes."""
  w, h = max(int(width), 2), max(int(height), 2)
  stride = int(stride) or w
  need_y = stride * h
  if len(data) < need_y:
    return {}
  # 注意: count 必须覆盖整个 buffer (含 UV 面), 不能只取 Y 面那段, 否则下面切 UV 会空。
  a = np.frombuffer(data, dtype=np.uint8)
  y = a[:need_y].reshape(h, stride)[:, :w].astype(np.float32)
  uv_off = stride * h
  uv_rows = h // 2
  uv_w = w // 2
  if len(data) < uv_off + stride * uv_rows:
    uv = np.zeros((uv_rows, uv_w, 2), dtype=np.float32)
  else:
    uv = a[uv_off:uv_off + stride * uv_rows].reshape(uv_rows, stride)[:, :uv_w * 2]
    uv = uv.reshape(uv_rows, uv_w, 2).astype(np.float32)
  u, v = uv[..., 0], uv[..., 1]
  return {
    "y_mean": float(y.mean()),
    "y_std": float(y.std()),
    "u128": float(u.mean() - 128.0),
    "v128": float(v.mean() - 128.0),
    "abs_u128": float(abs(u.mean() - 128.0)),
    "abs_v128": float(abs(v.mean() - 128.0)),
    "uv_std": float(uv.std()),
    # A chroma plane that looks like an image means plane offset/geometry is
    # wrong, not the byte order.
    "uv_hf": float(np.abs(np.diff(u, axis=1)).mean()) if u.shape[1] > 1 else 0.0,
    # Row-domain discontinuity: a tear is a sharp jump between adjacent rows.
    "row_jump": float(np.abs(np.diff(y.mean(axis=1))).max()) if h > 1 else 0.0,
  }


def _probe_streams(dur: float, conflate: bool) -> dict:
  clients = {}
  for name, st in STREAMS.items():
    c = VisionIpcClient("camerad", st, conflate)
    if c.connect(False):
      clients[name] = c
  out: dict[str, list[dict]] = {n: [] for n in clients}
  counts = {n: 0 for n in clients}
  sizes = {}
  t0 = time.monotonic()
  while time.monotonic() - t0 < dur:
    for name, c in clients.items():
      buf = c.recv(timeout_ms=50)
      if buf is None:
        continue
      counts[name] += 1
      data = bytes(buf.data)
      sizes[name] = (buf.width, buf.height, buf.stride, len(data))
      st = _plane_stats(data, buf.width, buf.height, buf.stride)
      if st:
        out[name].append(st)
  for c in clients.values():
    try:
      del c
    except Exception:
      pass
  return {"samples": out, "counts": counts, "hz": {n: counts[n] / (time.monotonic() - t0) for n in clients},
          "sizes": sizes, "present": sorted(clients)}


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--dur", type=float, default=5.0)
  ap.add_argument("--conflate", action="store_true", help="same receive mode modeld uses")
  ap.add_argument("--with-model", action="store_true", help="also report modelV2 rate")
  args = ap.parse_args()

  if VisionIpcClient is None:
    print("msgq visionipc 不可用, 无法读已发布帧", file=sys.stderr)
    return 2

  res = _probe_streams(args.dur, args.conflate)
  if not res["present"]:
    print("camerad 没有可连接的流 (camera 还没起来?)")
    return 1

  print(f"=== published-frame chroma/geometry probe ({args.dur:.1f}s) ===")
  for name in res["present"]:
    samples = res["samples"][name]
    w, h, stride, nbytes = res["sizes"].get(name, (0, 0, 0, 0))
    print(f"{name:5s} {res['hz'][name]:5.2f} Hz  n={len(samples)}  {w}x{h} stride={stride} bytes={nbytes}")
    if not samples:
      continue
    keys = ["y_mean", "y_std", "u128", "v128", "abs_u128", "abs_v128", "uv_std", "uv_hf", "row_jump"]
    print("      " + " ".join(f"{k}={statistics.mean(d[k] for d in samples):+.2f}" for k in keys))
  if args.with_model:
    sm = messaging.SubMaster(["modelV2"])
    n = 0
    t0 = time.monotonic()
    while time.monotonic() - t0 < args.dur:
      sm.update(50)
      if sm.updated.get("modelV2"):
        n += 1
    print(f"modelV2 {n / max(1e-6, time.monotonic() - t0):5.2f} Hz  (0 = 模型没在推理)")

  print()
  print("判据: U/V 同时远离 128 (尤其同时偏低) -> 整幅偏绿, 查布局/色度面错位;")
  print("      U<->V 互换 -> 品红/蓝; uv_hf 高 -> UV 面像图像, 是几何错不是字节序错;")
  print("      y_std 很小 (夜里 <15) 时字节序判据本身不可靠 (布局文档 §4.2);")
  print("      帧率掉到 ~17Hz -> 取帧路径整块拷贝吃掉了单帧预算 (布局文档 §1.3)。")
  return 0


if __name__ == "__main__":
  sys.exit(main())