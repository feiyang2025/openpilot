#!/usr/bin/env python3
"""Camera link preflight: wait until every requested /dev/video node really delivers frames.

Why this exists
---------------
`tw_camera_cfg` finishing (or even holding /dev/i2c-9 open) does NOT mean the GMSL
links are locked. Each VI node locks asynchronously, so camerad can STREAMON a node
whose serdes pipeline is not up yet. v4l2_dmabuf_camera._probe_frame_bytesused then
gets no frame within its window, permanently falls back to guessed G_FMT geometry, and
the corresponding cameraState stays not_alive forever -> modeld blocks and never
publishes modelV2, while the road camera (which did lock) still shows video in the UI.

Doing the DQBUF round trip here, before the stack starts, turns that race into a
deterministic wait: the caller does not let camerad run until all nodes produced a frame,
and a node that never produces one triggers a twgmsl reload.

Exit codes
----------
0  every requested node delivered a frame
1  at least one node never delivered a frame (retrying may help)
2  bad usage
3  same as 1, but the VI layer reports request timeouts: the link is down, so
   reloading twgmsl will not fix it and the caller should not spend retries on it
"""

import argparse
import ctypes
import fcntl
import mmap
import os
import select
import struct
import subprocess
import sys
import time

V4L2_BUF_TYPE_VIDEO_CAPTURE = 1
V4L2_MEMORY_MMAP = 1
V4L2_CAP_STREAMING = 0x04000000

VIDIOC_QUERYCAP = 0x80685600
VIDIOC_QUERYBUF = 0xC0585609
VIDIOC_REQBUFS = 0xC0145608
VIDIOC_QBUF = 0xC058560F
VIDIOC_DQBUF = 0xC0585611
VIDIOC_STREAMON = 0x40045612
VIDIOC_STREAMOFF = 0x40045613


class v4l2_requestbuffers(ctypes.Structure):
  _fields_ = [
    ("count", ctypes.c_uint32),
    ("type", ctypes.c_uint32),
    ("memory", ctypes.c_uint32),
    ("reserved", ctypes.c_uint32 * 2),
  ]


class v4l2_buffer(ctypes.Structure):
  _fields_ = [
    ("index", ctypes.c_uint32),
    ("type", ctypes.c_uint32),
    ("bytesused", ctypes.c_uint32),
    ("flags", ctypes.c_uint32),
    ("sequence", ctypes.c_uint32),
    ("memory", ctypes.c_uint32),
    ("m", ctypes.c_uint32),
    ("length", ctypes.c_uint32),
    ("reserved", ctypes.c_uint32 * 2),
    ("timeval_sec", ctypes.c_int64),
    ("timeval_usec", ctypes.c_int64),
  ]


class v4l2_querycap(ctypes.Structure):
  _fields_ = [
    ("driver", ctypes.c_char * 16),
    ("card", ctypes.c_char * 32),
    ("version", ctypes.c_uint32),
    ("capabilities", ctypes.c_uint32),
    ("device_caps", ctypes.c_uint32),
    ("reserved", ctypes.c_uint32 * 14),
  ]


def _dequeue_once(fd, timeout):
  """DQBUF one buffer and requeue it. Returns bytesused, or None on timeout/error."""
  ready, _, _ = select.select([fd], [], [], timeout)
  if not ready:
    return None
  buf = v4l2_buffer()
  buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
  buf.memory = V4L2_MEMORY_MMAP
  try:
    fcntl.ioctl(fd, VIDIOC_DQBUF, buf)
  except OSError:
    return None
  if buf.index >= 8 or buf.bytesused == 0:
    return None
  rq = v4l2_buffer()
  rq.index = buf.index
  rq.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
  rq.memory = V4L2_MEMORY_MMAP
  try:
    fcntl.ioctl(fd, VIDIOC_QBUF, rq)
  except OSError:
    pass
  return int(buf.bytesused)


def wait_for_frames(device, timeout):
  """Stream one node until a real frame arrives. Returns bytesused or None.

  The fd is always torn down (STREAMOFF + REQBUFS(0) + munmap) so the driver is left
  in the state the camera class expects when it opens the node itself.
  """
  try:
    fd = os.open(device, os.O_RDWR | os.O_NONBLOCK)
  except OSError as e:
    print(f"[preflight] {device}: open 失败: {e}", flush=True)
    return None

  mms = []
  try:
    cap = v4l2_querycap()
    try:
      fcntl.ioctl(fd, VIDIOC_QUERYCAP, cap)
    except OSError:
      print(f"[preflight] {device}: QUERYCAP 失败", flush=True)
      return None
    if not cap.capabilities & V4L2_CAP_STREAMING:
      print(f"[preflight] {device}: QUERYCAP 无 STREAMING 能力, 不是采集节点", flush=True)
      return None

    req = v4l2_requestbuffers()
    req.count = 4
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
    req.memory = V4L2_MEMORY_MMAP
    fcntl.ioctl(fd, VIDIOC_REQBUFS, req)
    if req.count < 2:
      print(f"[preflight] {device}: REQBUFS 只给了 {req.count} 个 buffer (驱动未就绪/链路脏)", flush=True)
      return None

    for i in range(req.count):
      qb = v4l2_buffer()
      qb.index = i
      qb.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
      qb.memory = V4L2_MEMORY_MMAP
      fcntl.ioctl(fd, VIDIOC_QUERYBUF, qb)
      # twgmsl 的 vi-output 会报 QUERYBUF length=0（驱动自己写硬件缓冲，不走用户态
      # mmap 偏移）。这里只关心"有没有出帧"，所以 length=0 时跳过 mmap，别去对 0
      # 长度 mmap（EINVAL）。
      if qb.length:
        mms.append(mmap.mmap(fd, qb.length, mmap.MAP_SHARED, mmap.PROT_READ, offset=qb.m))
      buf = v4l2_buffer()
      buf.index = i
      buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
      buf.memory = V4L2_MEMORY_MMAP
      fcntl.ioctl(fd, VIDIOC_QBUF, buf)

    fcntl.ioctl(fd, VIDIOC_STREAMON, struct.pack("I", V4L2_BUF_TYPE_VIDEO_CAPTURE))

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
      used = _dequeue_once(fd, timeout=0.5)
      if used:
        print(f"[preflight] {device}: 首帧到位 bytesused={used}", flush=True)
        return used
    print(f"[preflight] {device}: {timeout:.0f}s 内没有出帧 -> 该路 GMSL 未锁定", flush=True)
    return None
  except OSError as e:
    print(f"[preflight] {device}: ioctl 失败: {e}", flush=True)
    return None
  finally:
    for m in mms:
      try:
        m.close()
      except Exception:
        pass
    try:
      fcntl.ioctl(fd, VIDIOC_STREAMOFF, struct.pack("I", V4L2_BUF_TYPE_VIDEO_CAPTURE))
    except OSError:
      pass
    try:
      rel = v4l2_requestbuffers()
      rel.count = 0
      rel.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
      rel.memory = V4L2_MEMORY_MMAP
      fcntl.ioctl(fd, VIDIOC_REQBUFS, rel)
    except OSError:
      pass
    try:
      os.close(fd)
    except OSError:
      pass


def _link_is_down():
  """True if the VI layer is timing out waiting for frames (link down, not slow)."""
  """If the VI layer is reporting request timeouts, this is not a slow lock.

  `tegra-camrtc-capture-vi ... uncorr_err: request timed out after 2500 ms` means the
  VI channel is waiting for frame/frame-start from a deserializer that never delivers:
  the link is down (serdes/power/cabling), not merely slow. Waiting longer or reloading
  twgmsl will not bring it back; the caller should not keep retrying for nothing.
  """
  try:
    dmesg = subprocess.run(["sudo", "dmesg"], capture_output=True, text=True, timeout=5).stdout
  except Exception:
    return False
  timeouts = [l for l in dmesg.splitlines() if "uncorr_err" in l]
  if not timeouts:
    return False
  print(f"[preflight] dmesg 有 {len(timeouts)} 条 VI uncorr_err (最近一条: {timeouts[-1].strip()})",
        flush=True)
  print("[preflight] 这是链路真没出帧(等再久/重载 twgmsl 都没用), 疑似 serdes/供电/接线;"
        "先查 tw_camera_cfg 日志与硬件, 链路恢复前 wide 路会一直缺帧。", flush=True)
  return True


DEAD_NODES_PATH = "/tmp/cam_dead_nodes"


def _record_dead_nodes(devices):
  """把确认不出帧的节点记下来, 让 camerad 直接跳过对它们的慢探测。

  不这么做的话, 启动要白等两轮: preflight 15s 判死一次, 之后 camerad 的布局探测
  又按 3x8s 重试一遍 (实测 ~42s 才出画面), 而那 27s 得到的结论和 preflight 一模一样。
  """
  try:
    with open(DEAD_NODES_PATH, "w") as f:
      f.write("\n".join(devices) + "\n")
  except OSError:
    pass


def _clear_dead_nodes():
  try:
    os.unlink(DEAD_NODES_PATH)
  except OSError:
    pass


def main():
  ap = argparse.ArgumentParser(description="wait until each /dev/videoN node delivers frames")
  ap.add_argument("devices", nargs="*", help="/dev/videoN list; defaults to $ROAD_CAM/$WIDE_CAM")
  ap.add_argument("--timeout", type=float, default=15.0, help="per-device budget in seconds")
  args = ap.parse_args()

  devices = list(args.devices)
  if not devices:
    ids = [os.getenv("ROAD_CAM", "1"), os.getenv("WIDE_CAM", "2")]
    devices = [f"/dev/video{i}" for i in ids if i]

  present = [d for d in devices if os.path.exists(d)]
  missing = [d for d in devices if d not in present]
  if missing:
    print(f"[preflight] 节点不存在, 跳过: {', '.join(missing)}", flush=True)
  if not present:
    print("[preflight] 没有可测的相机节点", flush=True)
    return 1

  failed = [d for d in present if wait_for_frames(d, args.timeout) is None]
  if failed:
    print(f"[preflight] 未出帧: {', '.join(failed)}", flush=True)
    _record_dead_nodes(failed)
    return 3 if _link_is_down() else 1
  _clear_dead_nodes()
  print(f"[preflight] {len(present)} 路相机全部就绪", flush=True)
  return 0


if __name__ == "__main__":
  sys.exit(main())