# Ubuntu 上手指南

> 面向：在 Ubuntu 24.04 机器上接手这个项目的人
> 仓库：`https://github.com/feiyang2025/openpilot`
> 分支：`master-c3-cuda-test`
> 最后更新：2026-10-14

---

## 0. 不要从 Windows 拷贝目录

代码已经在 GitHub 上了。**直接从 GitHub clone**，不要拷 `E:\Model\增加导航\op-cuda`：

- Windows 上那个 clone 里的 `.git` 记录的是 Windows 路径和 LF/CRLF 转换，拷到 Linux 会出问题
- LFS 对象、`submodule` 指针都可能不一致
- `git clone` 反而更快更干净

**唯一需要从 Windows 带过去的**是本目录（`E:\Model\自动化更新`）里的**规划文档**，它们还没进 Git。里面是需求梳理、调研结论、阶段划分、待办清单——纯参考资料，不影响运行。

如果想省事，直接 clone 之后告诉我文档在哪，我再决定要不要也提交进去。

---

## 1. clone

```bash
git clone --recurse-submodules --branch master-c3-cuda-test \
  https://github.com/feiyang2025/openpilot.git
cd openpilot
```

### 1.1 验证 clone 完整

```bash
# 1) 子模块都拉下来了（不应该有 '-' 前缀）
git submodule status

# 2) 模型是真文件，不是 LFS 指针（指针只有 130 字节左右）
ls -l openpilot/selfdrive/modeld/models/*.onnx

# 3) 分支对
git log --oneline -1     # 应该是 62f8a30 或更新
```

如果 `git submodule status` 前面有 `-`：

```bash
git submodule update --init --recursive
```

如果 `.onnx` 只有 130 字节：

```bash
git lfs pull
```

> 注意：这个 fork 的 **LFS 配额已耗尽**，`.gitattributes` 里排除了大量模型和字体走普通 git blob。所以模型应该是普通文件，不需要 `git lfs pull`——但如果看到指针文本，就跑一下。

---

## 2. 环境准备

```bash
# 系统依赖
sudo apt update
sudo apt install -y \
  build-essential cmake clang git git-lfs libeigen3-dev \
  libopencv-dev python3-dev python3-venv \
  libusb-1.0-0-dev libudev-dev

# Python 必须 3.12（.python-version 写的是 3.12.13，
# pyproject 要求 >= 3.12.3, < 3.13）
python3 --version
```

**WSL 不支持。** 这是项目3 的 webcam README 和 openpilot 工具链共同确认的，`tools/README.md` 也只列了 Ubuntu 24.04 和 WSL 两条路，但 visionipc / cython / scons 在 WSL 下会有问题。用原生 Ubuntu 或虚拟机。

### 2.1 openpilot 自己的 setup

```bash
./tools/op.sh setup
```

这会建 `.venv` 并装依赖（内部走 `uv` + `tools/setup_dependencies.sh`）。完成后：

```bash
source .venv/bin/activate
python --version    # 应该是 3.12.x
```

---

## 3. 先跑不需要 GPU 的测试

这是**最重要的一步**。52 个测试不需要 CUDA / TensorRT / Jetson，几秒钟就能跑完，能立刻验证这次改动的大部分逻辑：

```bash
source .venv/bin/activate

python -m pytest openpilot/sunnypilot/modeld_v2/tests/test_gpu_backend.py -v
python -m pytest openpilot/selfdrive/modeld/tests/test_cuda_patches.py -v
```

期望：`35 passed` 和 `17 passed`。

这些测试覆盖的是**一旦错了就完全没有症状**的部分：

| 测试 | 挡住什么 |
|---|---|
| `test_bf16_is_not_fp16` | TRT dtype 7 被当成 fp16 —— 静默读出错误数字 |
| `test_old_bug_is_caught` | vision/policy 切片划分反了 —— 输出能解析但全是错的 |
| `test_unloadable_pickle_raises` | metadata 缺失时回落到硬编码的 2580 维切片 |
| `test_single_slot_features_buffer_is_not_ready` | 单时间槽的 buffer 无法移位 |
| `test_a_raising_callback_does_not_break_the_import` | 补丁抛异常导致整个模块 import 失败 |

### 3.1 验证 CUDA 补丁能不能被加载

```bash
# 补丁总开关是否生效
OPENPILOT_CUDA_PATCH=0 python -c "
from openpilot.selfdrive.modeld import cuda_patches
print('enabled:', cuda_patches.enabled())
print('applied:', cuda_patches.apply())
"
```

在 PC 上 `applied` 应该是空列表（`patch_after_import` 只挂钩子，不主动 import 目标模块）。这是**正常的**，不是失败。

```bash
# 看看模型注册表为什么没准备好（PC 上必然不可用，这正是要看的原因）
python -c "
from openpilot.sunnypilot.modeld_v2.gpu_backend.models.registry import REGISTRY
for name, p in REGISTRY.items():
    print(name, '->', p.missing_reason() if p else 'metadata missing / build failed')
"
```

PC 上会打印 `engine dir missing: ...` 之类。这是**预期的**——`.plan` 只能在 Orin 上编。

> ⚠️ `import openpilot.*` 必须在仓库根目录下跑，否则 `sitecustomize.py` 不会生效（它靠仓库根在 `sys.path` 上）。这也是**故意设计**：不想在一个裸 `python -c` 里强行拉起整个包。

---

## 4. 跑通采集链路（PC 上）

### 4.1 摄像头

需要一个 USB 摄像头（≥720p、≥78° FOV，比如 Logitech C920）。

```bash
# 确认设备节点
ls -l /dev/video*
v4l2-ctl --list-devices
```

```bash
# 启动（默认用 /dev/video0）
USE_WEBCAM=1 ROAD_CAM=0 python openpilot/system/manager/manager.py
```

如果加宽屏摄像头：

```bash
USE_WEBCAM=1 ROAD_CAM=0 WIDE_CAM=2 python openpilot/system/manager/manager.py
```

**期望**：`modelV2` 输出 **20.00 Hz**，`carState` 稳定无跳变，控制指令能算出来但**不控车**（PC 上没有 panda 连接）。

> 20 Hz 是 `constants.py` 里 `MODEL_RUN_FREQ = 20` 的**调度上限**，不是模型能力上限。看到 20 fps 不要以为"模型慢"。

### 4.2 摄像头补丁

`USE_V4L2_CAMERA=1`（默认）时，补丁会把 `camerad.py` 里的 `Camera` 换成 `v4l2_dmabuf_camera.Camera`。**PC 上不要开**——那个模块是 Jetson VIC 路径：

```bash
USE_WEBCAM=1 USE_V4L2_CAMERA=0 python openpilot/system/manager/manager.py
```

### 4.3 设备匹配

PC 上不做真车 CAN 指纹。用固定 fingerprint：

```bash
# 离线生成 fingerprint
python tools/car_porting/auto_fingerprint.py --route <你的route.zip>
```

或直接写死 Params 里的 `CarPlatformBundle.platform`（`card.py:110` 的 `fixed_fingerprint` 路径）。

---

## 5. CUDA 后端（需要 Jetson AGX Orin）

> ⚠️ **以下全部只在 Orin 上做，且必须现场做。** 所有性能/精度数字在真机实测前一律写 TBD，不许填"业界一般是"。

### 5.1 一次性构建

```bash
cd openpilot/sunnypilot/modeld_v2/gpu_backend
./deploy.sh
```

它会按顺序做：

1. 编译 `libtrt_c_api.so`（TensorRT C ABI，源码在 `openpilot/selfdrive/modeld/transforms/trt_c_api.cc`）
2. 编译 `libcuda_transform.so`（CUDA warp kernel）
3. 为每个模型目录生成 `metadata.pkl`
4. 用 `trtexec` 编 `.plan` 引擎

编译前它会检查 `NvInfer.h` 是否存在——TensorRT 版本不匹配是 `.so` 加载失败最常见的原因。

```bash
# 可覆盖
CUDA_ARCH=sm_87 ./deploy.sh --fp16
```

### 5.2 验证构建结果

```bash
ls -l openpilot/selfdrive/modeld/transforms/lib*.so
ls -l openpilot/selfdrive/modeld/models/*/*.plan
ls -l openpilot/selfdrive/modeld/models/*/*metadata*.pkl

python -c "
from openpilot.sunnypilot.modeld_v2.gpu_backend.models.registry import REGISTRY
for name, p in REGISTRY.items():
    print(name, '->', p.missing_reason() if p else 'build failed')
"
```

**全部 `None` 才算成功。**

### 5.3 dtype 必须从引擎绑定读，不能猜

```bash
python -c "
from openpilot.sunnypilot.modeld_v2.gpu_backend.trt_runner import TrtRunner
r = TrtRunner('openpilot/selfdrive/modeld/models/BigCombo/driving_supercombo_fp16.plan')
for name, dt in r.input_dtypes.items():
    print(f'{name}: {dt}')
"
```

项目3 的实测记录：

| 模型 | `img` dtype | 代价 |
|---|---|---|
| BigCombo | uint8 | 无 |
| TGC | float16 | +0.79 ms/帧，图像带宽翻倍 |

写死任何一个，另一个就是错的。

### 5.4 精度三组对比（**硬停点**）

| 组 | 配置 |
|---|---|
| A | onnxruntime fp32（baseline） |
| B | trtexec 无 fp16 |
| C | trtexec `--fp16` |

**判据**：A vs B vs C 逐帧差异 < 1%。

> 「代码默认 `--fp16`」≠「已决定用 fp16」。默认存在 ≠ 验证通过。没跑完这三组之前，精度状态就是 **TBD**。

### 5.5 测量口径必须分开

混在一起会得出"引擎 84 fps 但系统 44 fps"这种无法解释的结论。

| 口径 | 怎么测 | 用途 |
|---|---|---|
| 引擎本体 | `trtexec --loadEngine`，**必须停栈** | 这张图本身行不行 |
| 经 modeld | 订阅 modelV2 读 `modelExecutionTime` | 管线真实代价 |
| 实际输出 | modelV2 到达频率 | 被 `MODEL_RUN_FREQ=20` 卡住的上限 |

### 5.6 停栈的正确方式

`setproctitle` 会把 cmdline 改写成不含 `openpilot` 字样的名字，`pgrep -f openpilot` **杀不到**：

```bash
# 先确认有几个 modeld 实例
ps -eo pid,ppid,args | grep -E "[s]elfdrive\.|[o]penpilot\.|[m]anager\.py|[w]ebcam\."

# 正确的清理 pattern 集合
[Mm]ANAGER  # 先杀 manager，否则它不停重生子进程
pkill -f '[m]anager\.py'
pkill -f '[l]aunch_chffrplus\.sh'
pkill -f '[s]elfdrive\.[a-z]'
pkill -f '[o]penpilot\.selfdrive\.'
pkill -f '[o]penpilot\.system\.'
pkill -f '[o]penpilot\.tools\.webcam\.camerad'
# watchdog 和 recovery server 挂在 systemd 下，不在进程树里，要单独点名
```

**顺序有讲究**：先杀 manager，再杀子进程，**循环三轮**。

**典型症状**：两个 modeld 抢同一路 visionipc，谁也跑不满。被抢占的那个 `run()` 只耗 0.5 ms（`prepare_only` 早返回），而消息里的 `modelExecutionTime` 是 20 ms——两个数都对，但**测的不是同一个进程**。第一步永远是确认实例数。

---

## 6. 运行时开关

| 开关 | 作用 |
|---|---|
| `OPENPILOT_CUDA_PATCH=0` | 关闭**全部** fork 补丁（兜底用） |
| `DISABLE_CUDA_BACKEND=1` | 关闭 CUDA 后端，回落 stock tinygrad |
| `USE_WEBCAM=1` | 启用 USB 摄像头 |
| `USE_V4L2_CAMERA=0` | 不用 V4L2/VIC 路径（PC 上应该设） |
| `ROAD_CAM` / `WIDE_CAM` / `DRIVER_CAM` | 摄像头设备号 |
| `CUDA_ARCH` | 编译架构，默认 `sm_87`（AGX Orin） |
| `TRT_C_API_LIBRARY` | 覆盖 `libtrt_c_api.so` 路径 |

**出问题时的排查顺序**：`OPENPILOT_CUDA_PATCH=0` → `DISABLE_CUDA_BACKEND=1` → 二者都设还能跑，说明是 fork 的问题不是上游的。

---

## 7. 这次改了什么（`7e75b92..62f8a30`）

**上游文件零改动。** 全部是新增文件 + `.gitattributes` / `.gitignore` 各加几行。所以和上游合并时冲突面接近零。

```
A  sitecustomize.py                          CPython 标准钩子
A  openpilot/selfdrive/modeld/cuda_patches.py
A  openpilot/sunnypilot/modeld_v2/gpu_model_state.py
A  openpilot/sunnypilot/modeld_v2/gpu_backend/*              6 个文件
A  openpilot/selfdrive/modeld/transforms/cuda_*.cu/.h
A  openpilot/selfdrive/modeld/transforms/trt_c_api.cc/.h     ← 之前不存在
A  openpilot/system/camerad/webcam/*_cuda*.py, v4l2_dmabuf_camera.py
A  openpilot/sunnypilot/modeld_v2/tests/test_gpu_backend.py   35 个测试
A  openpilot/selfdrive/modeld/tests/test_cuda_patches.py      17 个测试
A  .github/scripts/*                                          11 个脚本
M  .github/workflows/upstream-sync.yml
M  .gitattributes, .gitignore
```

**被修掉的静默错误**（每一个都会导致"能跑但结果是错的"）：

| 原问题 | 后果 |
|---|---|
| TRT dtype `7: float16`（7 实际是 **BF16**） | 静默读出垃圾数据 |
| `features_buffer` 每帧移两格 | 时序错乱 |
| `_update_temporal` 用 policy 切片切拼接后的数组 | 切出 vision 区的数据 |
| slices 划分条件写反 | 所有键都归 policy |
| 2580 维切片硬编码兜底 | metadata 缺失时用错切片 |
| vision dtype 写死 uint8 | TGC 模型必错 |
| pin 缓存无限增长 + 不 unregister | 显存泄漏 |
| NaN 检查只在 `chestnut=True` 跑 | 小模型路径无防护 |
| L4T 版本号硬编码 | JetPack 一升级就失效 |
| 异常一律吞掉只打 warning | 装了坏的也不知道 |

---

## 8. 已知未解决问题

看 `交付记录.md` 的「已知未解决问题」区。当前 P0：

| 问题 | 说明 |
|---|---|
| `libtrt_c_api.so` 未编译 | **已实现源码**，等 Orin 上编 |
| `libcuda_transform.so` 未编译 | 等 Orin 上编 |
| 无 `.plan` 引擎 | TRT 版本相关，**必须在 Orin 上编** |
| 无 `metadata.pkl` | `registry.py` 现在缺失即报错，不会静默兜底 |
| 精度 TBD | 硬停点 |
| 真零拷贝未接线 | `read_frame_to_fd` / `CudaIPCExport` 是死代码，本轮不做 |

**pin 缓存上限那条特别说明**：我把 `_registered` 改成了 LRU（上限 32 帧）。理论上如果 kernel 还没消费完就被 unpin，会读到被换出的内存。**未验证**。Orin 上第一次跑建议把 `MAX_REGISTERED_FRAMES` 调大到 128，或者先改成只报警不 unpin。

---

## 9. 出问题看哪里

| 症状 | 先查 |
|---|---|
| 补丁完全没生效 | `OPENPILOT_CUDA_PATCH=0` 是不是设了；是不是在仓库根目录外跑的 |
| CUDA 后端没被用 | `python -c "...REGISTRY..."` 看 `missing_reason()` 的具体输出 |
| `trt_engine_create failed` | plan 是别的 TRT 版本或别的 GPU 编的，用 `trtexec` 重编 |
| modeld 崩溃重启 | `cat openpilot/selfdrive/modeld/models/.big_model_failed`；看是不是 metadata 缺失 |
| 摄像头起不来 | `USE_V4L2_CAMERA=0`；`v4l2-ctl --list-devices` |
| 性能数字看着不对 | 先 `ps` 确认只有一个 modeld 实例 |

---

## 10. 分支与同步

| 分支 | 内容 |
|---|---|
| `master-c3` | **纯上游镜像**，永远等于 `mouxangithub/openpilot@master-c3` |
| `master-c3-cuda-test` | **开发主线**，你在用的 |

上游同步自动化（`.github/workflows/upstream-sync.yml`）每天 03:17 UTC 跑：merge 上游 → 有冲突就 opencode 解（最多 2 次）→ 结构检查 + 验证 → **只推 `master-c3-cuda-test`** → 开**草稿 PR** 到 `master-c3`。

**永远不自动合并。** 控车代码的合并权在人，CI 判断不了纵向/横向行为有没有回归。

需要配的 secret：`OPENCODE_API_KEY`（可选 `SYNC_PAT`、`vars.OPENCODE_MODEL`）。

> 上游同步**从不修改 `master-c3`**。这样你随时能 `git diff master-c3-cuda-test` 看同步到底动了什么。

---

## 11. 提交规范

- `交付记录.md` 在仓库根目录，**必须和代码一起 commit，不许进 `.gitignore`**。理由：出故障时你要回滚的是代码；log 放别处 → 代码回滚了 log 还在原地 → 你不知道改之前长什么样。
- 改了东西就在 `交付记录.md` 加一条，7 个字段：时间 / 阶段 / 改了什么(`文件:行号`) / 为什么 / 怎么验证(**具体命令 + 实际输出**) / 能不能退回去 / 已知副作用。
- **未验证 ≠ 已通过。** 没跑就写"未验证"，不许写"应该没问题"。
- 收益数字必须附推导过程（写"提速 27%"就要写"45ms 中 30ms 在 transform，改后 18ms"）。
- **做了没要求之外的改动，一律算失败，回退掉。**
