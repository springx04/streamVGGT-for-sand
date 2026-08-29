# OmniVGGT C++ 边缘 GPU 推理部署说明

本目录用于在不影响 OmniVGGT 原有 Python 训练/推理代码的前提下，增加一套 C++ 边缘部署路径。当前实现路线是：

1. 用 Python 将 `checkpoints/OmniVGGT.safetensors` 导出为固定输入规格的 TorchScript `.pt` 模型。
2. 用 C++/LibTorch 在边缘设备上加载 `.pt` 执行推理。
3. 用 OpenCV 在 C++ 端完成图片读取、resize/crop、张量构造。
4. 用 C++ 端将模型输出保存为点云 `pointcloud.ply`、相机结果 `cameras.txt` 和运行摘要 `summary.txt`。
5. 额外提供 C++ 流式 runner，将 `stream_omnivggt` 中的外部 2D 对齐、变化检测、增量 depth/canvas 融合迁移到 C++ 侧。

必须注意：本方案要求 GPU 推理，不做 CPU fallback。运行时命令应使用 `--device cuda`。

## 为什么选择 TorchScript + LibTorch

OmniVGGT 是一个较大的 PyTorch Transformer 模型，包含 DINOv2 patch embed、多帧 attention、相机/深度辅助输入、DPT dense head 和 camera head。直接手写 C++ 网络实现成本高、风险大，也容易和 Python 版本不一致。

因此当前最稳妥的 C++ 边缘部署方式是：

- Python 侧保持原模型结构和权重加载逻辑。
- 导出固定帧数、固定分辨率、固定辅助输入索引的 TorchScript。
- C++ 侧只负责前处理、调用 LibTorch、后处理和保存结果。

ONNX/TensorRT 后续可以继续做，但应作为单独验证链路。当前 C++ 可运行版本以 TorchScript 为主。

## 当前已验证的 Windows GPU 环境

当前机器已完成并验证的路径如下：

```text
项目路径:
C:\Users\30738\Desktop\project\3dre\VGGT\OmniVGGT

LibTorch GPU:
C:\Dev\libtorch\2.7.0-cu128

OpenCV C++:
C:\Dev\opencv\4.10.0

OpenCV_DIR:
C:\Dev\opencv\4.10.0\build

CUDA:
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8

C++ 编译器:
Visual Studio 2019 Community MSVC 14.29

GPU:
NVIDIA GeForce RTX 5060 Laptop GPU
```

已确认存在：

```text
C:\Dev\libtorch\2.7.0-cu128\lib\torch_cuda.dll
C:\Dev\libtorch\2.7.0-cu128\share\cmake\Torch\TorchConfig.cmake
C:\Dev\opencv\4.10.0\build\OpenCVConfig.cmake
```

已成功编译出的 C++ 程序：

```text
setc\build_manual_libtorch\omnivggt_edge.exe
setc\build_manual_libtorch\omnivggt_stream.exe
```

## setc 目录文件说明

```text
setc/
  CMakeLists.txt
  README.md
  local_env.ps1
  scripts/
    export_torchscript.py
    install_windows_gpu_deps.ps1
    setup_export_venv_torch270.ps1
    build_windows_gpu_manual.bat
    run_windows_gpu_s1.bat
    run_windows_gpu_stream.bat
    build_windows_gpu.bat
    build_windows_gpu_nmake.bat
    check_vs_toolchain.bat
    check_cuda_nvcc.bat
  src/
    omnivggt_edge.cpp
    omnivggt_stream.cpp
```

核心文件：

- `scripts/export_torchscript.py`：导出 C++ 可加载的 TorchScript `.pt`。
- `src/omnivggt_edge.cpp`：C++ 单次窗口推理程序。
- `src/omnivggt_stream.cpp`：C++ 流式推理程序，包含外部对齐、变化检测和增量融合。
- `CMakeLists.txt`：C++ 构建配置。
- `scripts/build_windows_gpu_manual.bat`：当前 Windows GPU 环境实际编译成功的构建脚本。
- `scripts/run_windows_gpu_s1.bat`：当前已验证的 1 张图 CUDA 推理复现脚本。
- `scripts/run_windows_gpu_stream.bat`：当前已验证的 C++ 流式 CUDA 推理复现脚本。
- `local_env.ps1`：记录本机 LibTorch/OpenCV 路径。

## 依赖安装过程记录

### 1. LibTorch GPU

目标版本按项目 README 的 PyTorch/CUDA 组合对齐：

```text
LibTorch 2.7.0 + CUDA 12.8
```

安装目标：

```text
C:\Dev\libtorch\2.7.0-cu128
```

下载地址使用 PyTorch 的 LibTorch CUDA 12.8 包：

```text
https://download-r2.pytorch.org/libtorch/cu128/libtorch-win-shared-with-deps-2.7.0%2Bcu128.zip
```

由于该文件很大，下载时可能中断。脚本 `install_windows_gpu_deps.ps1` 使用 `curl -C -` 支持续传。

执行方式：

```powershell
cd C:\Users\30738\Desktop\project\3dre\VGGT\OmniVGGT
powershell -NoProfile -ExecutionPolicy Bypass -File setc\scripts\install_windows_gpu_deps.ps1
```

### 2. OpenCV C++

使用 OpenCV Windows 预编译包：

```text
OpenCV 4.10.0
```

安装目标：

```text
C:\Dev\opencv\4.10.0
```

有效 CMake 配置目录：

```text
C:\Dev\opencv\4.10.0\build
```

### 3. Visual Studio / MSVC

VS Code 只是编辑器，不提供 C++ 编译器。当前机器使用已有的：

```text
C:\Program Files (x86)\Microsoft Visual Studio\2019\Community
```

实际编译器：

```text
C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64\cl.exe
```

检查脚本：

```powershell
cmd /c setc\scripts\check_vs_toolchain.bat
```

## 为什么当前不用 find_package(Torch)

最开始的标准 CMake 写法是：

```cmake
find_package(Torch REQUIRED)
```

但在当前 Windows 环境中，`find_package(Torch)` 会触发 LibTorch 的 CUDA 检测，进而调用 CMake 的 CUDA language enable 流程。当前 VS2019 + CUDA 12.8 组合会遇到这些问题：

- Visual Studio 生成器报 `No CUDA toolset found`。
- NMake 生成器继续后，CMake CUDA compiler identification 阶段失败或卡住。
- 最小 `nvcc` 编译测试也会卡住。

关键点是：我们的 `omnivggt_edge.cpp` 并不编译任何自定义 CUDA kernel，它只是加载 LibTorch 的 CUDA DLL 并在运行时调用 GPU 推理。因此没有必要让 CMake 编译 CUDA 代码。

当前成功方案是在 `CMakeLists.txt` 中手动链接 LibTorch GPU 库：

```text
torch.lib
torch_cpu.lib
torch_cuda.lib
c10.lib
c10_cuda.lib
caffe2_nvrtc.lib
```

运行时通过 PATH 找到：

```text
torch_cuda.dll
c10_cuda.dll
```

Windows 下 `torch_cuda.dll` 不一定会因为链接 import lib 自动加载。当前 `omnivggt_edge.cpp` 在 `--device cuda` 时会先显式 `LoadLibraryA("c10_cuda.dll")` 和 `LoadLibraryA("torch_cuda.dll")`，再调用 `torch::cuda::is_available()`。如果 CUDA 后端不可用，程序直接报错退出，不会切到 CPU。

这仍然是 GPU 推理，不是 CPU 版本。

## 外部对齐和流式模块的 C++ 覆盖情况

原项目中的外部流式模块位于：

```text
stream_omnivggt/
  align/
  detect/
  window/
  map/
  pipeline/
```

这些模块不是 OmniVGGT 模型 forward 的一部分，因此不会自动进入 TorchScript `.pt`。如果只运行 `omnivggt_edge.exe`，它只会执行一次固定窗口模型推理和基础点云导出，不会执行 `stream_omnivggt` 的流式状态、对齐和增量融合。

当前已新增 C++ 版本：

```text
setc\src\omnivggt_stream.cpp
setc\build_manual_libtorch\omnivggt_stream.exe
setc\scripts\run_windows_gpu_stream.bat
```

`omnivggt_stream.exe` 已迁移并实现的外部模块能力：

- 对应 `stream_omnivggt.pipeline.aligned_canvas_stream` 的 anchor canvas 流式框架。
- 对应 `estimate_homography_to_anchor` 的 SIFT/ORB + RANSAC homography 2D 对齐。
- 对应 fallback translation 的 phase correlation 平移兜底。
- 对应 `_change_mask` 的 support change + photometric change 检测。
- 对应 `dilate_change_mask` 的变化区域膨胀。
- 对应 `_align_depth` 的 overlap depth affine 对齐。
- 对应 `_fuse` 的单层 depth/color/confidence 加权融合。
- 对应 `export_pointcloud` 的 canonical canvas 点云导出。
- 对应 `timings.md` 的逐帧耗时、changed ratio、homography inliers、fallback、fused pixels 记录。











当前 C++ 版本和 Python `stream_omnivggt` 仍不完全等价的部分：

- Python 的通用 `StreamEngine` 中 active window selector、keyframe list、HybridMap block/cold-store/TSDF 还没有完整移植为同名结构。
- Python `AlignedCanvasStream` 的后续帧 2-frame ROI 模型路径需要 S2 TorchScript；当前 C++ 版本使用已验证的 S1 TorchScript，逐帧推理后再做 C++ 对齐和融合。
- Python 外部 `rotation_aligner`、`height_aligner` 是 callable 插件机制；C++ 版本当前没有插件 ABI，只实现了实际用到的 2D/heightfield 对齐融合路径。

因此结论是：

```text
单次推理：运行 omnivggt_edge.exe
流式推理 + 外部对齐/融合：运行 omnivggt_stream.exe
```

如果迁移目标只需要当前已验证的 C++ 流式推理，不需要携带 Python 的 `stream_omnivggt/` 目录；需要携带的是 `omnivggt_stream.exe`、TorchScript 模型、LibTorch/OpenCV DLL 和输入图片。

## 设置环境变量

PowerShell 当前可能禁止执行 `.ps1`，因此最稳妥方式是在当前终端内直接设置：

```powershell
$env:LIBTORCH="C:\Dev\libtorch\2.7.0-cu128"
$env:OpenCV_DIR="C:\Dev\opencv\4.10.0\build"
$env:CUDA_PATH="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
$env:Path="$env:LIBTORCH\lib;$env:OpenCV_DIR\x64\vc16\bin;$env:OpenCV_DIR\x64\vc15\bin;$env:CUDA_PATH\bin;$env:Path"
```

如果你的 PowerShell 允许执行脚本，也可以：

```powershell
. .\setc\local_env.ps1
```

如果提示 execution policy 禁止执行，不需要改系统策略，使用上面的手动环境变量方式即可。

## 编译 C++ 推理程序

当前已验证成功的编译方式：

```powershell
cd C:\Users\30738\Desktop\project\3dre\VGGT\OmniVGGT
cmd /c setc\scripts\build_windows_gpu_manual.bat
```

成功后生成：

```text
setc\build_manual_libtorch\omnivggt_edge.exe
```

当前已实测成功的模型产物：

```text
setc\artifacts\omnivggt_s1_518x518_torch270.pt
setc\artifacts\omnivggt_s1_518x518_torch270.json
```

说明：该模型由 `setc\venv_torch270_cu128` 中的 `torch 2.7.0+cu128` 导出，并已用 `LibTorch 2.7.0+cu128` C++ 程序加载执行 CUDA 推理。当前正式验证链路已经做到 PyTorch 导出版本和 LibTorch C++ 运行版本一致。

可以用下面命令检查程序是否能启动：

```powershell
$env:LIBTORCH="C:\Dev\libtorch\2.7.0-cu128"
$env:OpenCV_DIR="C:\Dev\opencv\4.10.0\build"
$env:CUDA_PATH="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
$env:Path="$env:LIBTORCH\lib;$env:OpenCV_DIR\x64\vc16\bin;$env:OpenCV_DIR\x64\vc15\bin;$env:CUDA_PATH\bin;$env:Path"

.\setc\build_manual_libtorch\omnivggt_edge.exe --help
```

预期输出会显示：

```text
Usage:
  omnivggt_edge --model model.pt --image_dir images --output_dir out --num_images N --height H --width W [--device cpu|cuda]
```

实际推理时必须使用：

```text
--device cuda
```

## 导出 TorchScript 模型

C++ 程序需要加载 TorchScript `.pt`，不能直接加载 `.safetensors`。

推荐让导出端 PyTorch 版本和 C++ LibTorch 版本一致：

```text
PyTorch 2.7.0 + cu128
LibTorch 2.7.0 + cu128
```

当前全局 Python 曾检测到：

```text
torch 2.11.0.dev20260206+cu128
```

这和 LibTorch 2.7.0 不一致。TorchScript 跨大版本加载存在风险，因此最终使用隔离 venv 安装 PyTorch 2.7.0。

### 创建导出 venv

脚本：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File setc\scripts\setup_export_venv_torch270.ps1
```

该 venv 目标位置：

```text
setc\venv_torch270_cu128
```

注意：PyTorch GPU wheel 很大，网络不稳定时可能中断。中断后重新运行脚本即可继续尝试。

本机最终实际安装过程是：

```powershell
curl.exe -L -C - --retry 30 --retry-all-errors --connect-timeout 30 `
  --output C:\Dev\downloads\torch-2.7.0+cu128-cp311-cp311-win_amd64.whl `
  https://download.pytorch.org/whl/cu128/torch-2.7.0%2Bcu128-cp311-cp311-win_amd64.whl

setc\venv_torch270_cu128\Scripts\python.exe -m pip install sympy jinja2 networkx
setc\venv_torch270_cu128\Scripts\python.exe -m pip install C:\Dev\downloads\torch-2.7.0+cu128-cp311-cp311-win_amd64.whl
setc\venv_torch270_cu128\Scripts\python.exe -m pip install scipy opencv-python evo
```

验证结果：

```text
torch=2.7.0+cu128
torch.version.cuda=12.8
torch.cuda.is_available()=True
```

### 执行导出

如果 venv 安装成功，使用：

```powershell
setc\venv_torch270_cu128\Scripts\python.exe setc\scripts\export_torchscript.py `
  --checkpoint checkpoints\OmniVGGT.safetensors `
  --output setc\artifacts\omnivggt_s2_518x518.pt `
  --num-images 2 `
  --height 518 `
  --width 518 `
  --device cuda `
  --dtype float32
```

输出：

```text
setc\artifacts\omnivggt_s2_518x518.pt
setc\artifacts\omnivggt_s2_518x518.json
```

### 固定输入规格

导出的 TorchScript 是固定输入规格的。也就是说：

- `--num-images 2` 导出的模型，C++ 运行时也必须 `--num_images 2`。
- `--height 518 --width 518` 导出的模型，C++ 运行时也必须传相同的 `--height 518 --width 518`。
- 如果边缘端要跑 4 帧或更低分辨率，需要重新导出一个对应规格的 `.pt`。

例如 4 帧：

```powershell
setc\venv_torch270_cu128\Scripts\python.exe setc\scripts\export_torchscript.py `
  --checkpoint checkpoints\OmniVGGT.safetensors `
  --output setc\artifacts\omnivggt_s4_518x518.pt `
  --num-images 4 `
  --height 518 `
  --width 518 `
  --device cuda `
  --dtype float32
```

### 带相机/深度辅助输入的导出

如果需要让模型使用相机或深度辅助输入，必须在导出时固定索引：

```powershell
setc\venv_torch270_cu128\Scripts\python.exe setc\scripts\export_torchscript.py `
  --checkpoint checkpoints\OmniVGGT.safetensors `
  --output setc\artifacts\omnivggt_s4_cam0_depth13.pt `
  --num-images 4 `
  --height 518 `
  --width 518 `
  --camera-indices 0 `
  --depth-indices 1,3 `
  --device cuda `
  --dtype float32
```

C++ runner 每次都会传入 `images/extrinsics/intrinsics/depth/mask` 五个 tensor，但 TorchScript 内部只会使用导出时固定的 camera/depth indices。

## 运行 C++ GPU 推理

先设置环境变量：

```powershell
cd C:\Users\30738\Desktop\project\3dre\VGGT\OmniVGGT

$env:LIBTORCH="C:\Dev\libtorch\2.7.0-cu128"
$env:OpenCV_DIR="C:\Dev\opencv\4.10.0\build"
$env:CUDA_PATH="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
$env:Path="$env:LIBTORCH\lib;$env:OpenCV_DIR\x64\vc16\bin;$env:OpenCV_DIR\x64\vc15\bin;$env:CUDA_PATH\bin;$env:Path"
```

当前已验证的 1 张图 CUDA 推理可以直接运行：

```powershell
cmd /c setc\scripts\run_windows_gpu_s1.bat
```

本机实测输出：

```text
Loaded 1 frames
Running OmniVGGT TorchScript inference
Wrote cameras to "setc\output_s1_torch270\cameras.txt"
Wrote 201244 points to "setc\output_s1_torch270\pointcloud.ply"
Done. Outputs are in "setc\output_s1_torch270"
```

对应摘要文件确认实际设备为 CUDA：

```text
device=cuda
outputs:
  pose_enc=[1, 1, 9]
  depth=[1, 1, 518, 518, 1]
  depth_conf=[1, 1, 518, 518]
  world_points=[1, 1, 518, 518, 3]
  world_points_conf=[1, 1, 518, 518]
```

运行：

```powershell
.\setc\build_manual_libtorch\omnivggt_edge.exe `
  --model setc\artifacts\omnivggt_s2_518x518.pt `
  --image_dir example\office\images `
  --output_dir setc\output `
  --num_images 2 `
  --height 518 `
  --width 518 `
  --device cuda `
  --dtype float32 `
  --conf_percentile 25
```

输出文件：

```text
setc\output\pointcloud.ply
setc\output\cameras.txt
setc\output\summary.txt
```

## 运行 C++ 流式 GPU 推理

流式推理使用新的可执行文件：

```text
setc\build_manual_libtorch\omnivggt_stream.exe
```

当前已验证的复现命令：

```powershell
cmd /c setc\scripts\run_windows_gpu_stream.bat
```

该脚本默认使用：

```text
model=setc\artifacts\omnivggt_s1_518x518_torch270.pt
image_dir=example\office\images
output_dir=setc\stream_output_torch270
num_images=3
device=cuda
```

本机实测输出：

```text
frame=0 changed=0.7568 model_ms=212330.91 fused=203056 points=203056 fallback=scene_jump
frame=1 changed=0.7107 model_ms=413956.64 fused=190699 points=203056 fallback=bad_homography
frame=2 changed=0.2053 model_ms=1464.57 fused=55094 points=210949 fallback=None
Wrote stream pointcloud with 158212 points to "setc\stream_output_torch270\stream_pointcloud.ply"
Done. Outputs are in "setc\stream_output_torch270"
```

输出文件：

```text
setc\stream_output_torch270\stream_pointcloud.ply
setc\stream_output_torch270\timings.md
setc\stream_output_torch270\canvas_rgb.png
setc\stream_output_torch270\debug\*_warped.png
setc\stream_output_torch270\debug\*_change_mask.png
setc\stream_output_torch270\debug\*_support.png
```

`omnivggt_stream.exe` 也是强制 GPU 路径；传入非 `--device cuda` 会直接报错。

## 独立 C++ 流式观察与交互式回放插件

方案中的实时观察功能已作为 `setc` 内的独立、可选 C++ 组件实现。原有的
`omnivggt_edge` 和 `omnivggt_stream` 源码、目标和运行方式不变；只有显式打开
`OMNIVGGT_ENABLE_LIVE_OBSERVER=ON` 时，才会额外构建以下程序：

```text
omnivggt_stream_server.exe   目录监听、TorchScript 推理、唯一提交者和实时发布
omnivggt_live_viewer.exe     独立窗口、三维软件投影、旋转/缩放/平移和时间轴
omnivggt_replay_log.exe      从历史帧重建并导出 PLY
omnivggt_validate_history.exe校验 Delta 前进/后退、快照和索引
omnivggt_observer_core_smoke.exe 纯 C++ 核心往返冒烟测试
```

插件只使用现有的 C++ OpenCV/LibTorch 环境和标准 TCP，不修改 Python，也不把
gRPC、GLFW 或 Zstd 变成原项目的强制依赖。实时协议是版本化的小端二进制包；磁盘
历史位于每次运行独立的 `run_YYYYMMDD_HHMMSS/` 目录，包含 `run_meta.bin`、
`frame_index.bin`、`deltas.bin`、`deltas.idx`、`snapshots/` 和 `metrics.csv`。

在 Windows GPU 环境构建：

```powershell
cmd /c setc\scripts\build_windows_live_observer.bat
```

启动服务端和查看器：

```powershell
setc\scripts\run_windows_live_observer.bat
setc\build_live_observer\Release\omnivggt_live_viewer.exe --host 127.0.0.1 --port 37651
```

服务端持续扫描 `--image_dir`；写入中的文件需要连续两次扫描保持稳定才会进入
队列。队列满时保留最新输入，并将被合并帧记录为 `Coalesced`。每个输入仍获得
单调递增的 `frame_seq`，只有真正应用 Patch 才增加 `commit_version`；无变化帧
记录为 `NoChange`。服务端的推理线程不直接修改权威 Canvas，所有点槽位、支持域、
快照和 Delta 由提交线程串行维护。

查看器的 LiveState 与 PlaybackState 是两份独立状态。鼠标左键拖动旋转，
Shift+左键或中键平移，滚轮缩放；`R` 返回实时，空格播放/暂停，时间轴点击或
方向键请求历史帧。历史 seek 使用 generation 取消旧请求，因此快速拖动不会让
旧回放任务无限积压。当前 Delta 的新点以黄色高亮，被替换的旧点以红色 ghost
显示。

校验和导出：

```powershell
setc\build_live_observer\Release\omnivggt_validate_history.exe --run_dir setc\observer_output\run_YYYYMMDD_HHMMSS
setc\build_live_observer\Release\omnivggt_replay_log.exe --run_dir setc\observer_output\run_YYYYMMDD_HHMMSS --frame 120 --output_ply frame120.ply
setc\build_live_observer\Release\omnivggt_observer_core_smoke.exe
```

### 三图单次批推理（B=3,S=1）

`setc\scripts\start_cpp_live_replay.bat` 默认将稳定图片按步长 1 组成
`123`、`234`、… 的三图滑窗。三张图在 C++ 端构造成一个 batch，只调用一次
CUDA TorchScript forward；中间图是锚点，组内深度先做鲁棒标定，几何提交以锚图
 support 为所有权掩码，侧图只允许在锚图掩码内部补充可信深度，颜色不把旋转后的
 侧图黑边写进 Canvas；组模式也不复用单图 RGB bridge，避免在旋转组边缘生成长方形色缝。
 导出/viewer 只填充封闭的内部 support 孔洞，不填充外部黑边。
这样不会把 OmniVGGT 原生 `S=3` 输出直接拼成三层。

需要的模型是：

```text
setc\artifacts\omnivggt_observer_b3s1_406x252_bf16_unfrozen_torch270.pt
```

模型导出命令：

```powershell
setc\venv_torch270_cu128\Scripts\python.exe setc\scripts\export_torchscript.py `
  --output setc\artifacts\omnivggt_observer_b3s1_406x252_bf16_unfrozen_torch270.pt `
  --batch-size 3 --num-images 1 --height 252 --width 406 `
  --dtype bfloat16 --autocast-dtype bfloat16 --observer-depth-only --no-freeze
```

每次运行的 `input_groups.csv` 保存实际滑窗和锚点，`metrics.csv` 的
`forward_calls`、`forward_batch_size`、`forward_sequence_size` 用于确认没有退化为
三次串行单图推理。设置 `$env:INPUT_GROUP_SIZE="1"` 可回到原有单图/双图路径。

如果不传 `-DOMNIVGGT_ENABLE_LIVE_OBSERVER=ON`，CMake 仍只生成原有的两个 C++
推理目标，因此该功能不会改变原项目的正常运行路径。

## C++ 前处理约定

C++ 端尽量复刻 Python `visual_util.load_images_and_cameras` 的输入逻辑：

- 读取 RGB 图片。
- 宽度 resize 到 `--width`。
- 高度按比例缩放后 round 到 14 的倍数。
- 如果缩放后高度大于 `--height`，从中间裁剪。
- 如果缩放后高度小于 `--height`，上下黑边 padding。
- 像素转 float，范围 `[0, 1]`。
- 不在 C++ 端做 ImageNet mean/std。

原因：OmniVGGT 的 image mean/std normalize 在模型内部 aggregator 中完成。如果 C++ 端重复做 mean/std，会导致输入分布错误。

## C++ 输出说明

模型 TorchScript 输出五个 tensor：

```text
pose_enc
depth
depth_conf
world_points
world_points_conf
```

C++ 后处理：

- `world_points + world_points_conf + 原图颜色` 保存为 `pointcloud.ply`。
- `pose_enc` 解码为 predicted world-to-camera extrinsic 和 intrinsic，保存为 `cameras.txt`。
- 运行参数、输入图片、输出 tensor shape 保存到 `summary.txt`。

## 海康 MVS 采集与图片/内参伴随输出

仓库新增了一个可选的 MVS 采集组件：`omnivggt_hikvision_capture`。它使用
`MvCameraControl.h` 完成枚举、打开、取流和像素格式转换，并为每一帧保存：

```text
capture_output/
├── images/frame_000000.png
├── cameras/frame_000000.json
└── frames.csv
```

`frame_XXXXXXXX.json` 内含 `frame_id`、设备帧号、主机时间戳、相机 ID、图像尺寸、
3×3 `intrinsic`、可选五参数 `distortion` 以及内参来源。图片和 JSON 使用相同的
stem，`frames.csv` 再提供一份可批量遍历的索引。

如果需要嵌入现有实时程序而不是使用命令行，可直接使用同一个 C++ 接口：

```cpp
omnivggt::hikvision::HikvisionCamera camera(camera_options);
camera.open();
camera.start();
const auto frame = camera.grab();
if (frame) {
    // frame->bgr 与 frame->intrinsics 属于同一次采集。
    cv::Mat K = frame->intrinsics.camera_matrix();
    omnivggt::hikvision::save_captured_frame(*frame, output_options);
}
```

### 内参来源的边界

MVS 工业相机 SDK 的通用接口是按 MVS 客户端里显示的 GenICam 节点类型读取参数：
整型使用 `MV_CC_GetIntValueEx`，浮点型使用 `MV_CC_GetFloatValue`，字符串使用
`MV_CC_GetStringValue`。普通 2D 工业相机通常没有一个统一的“读取已标定 K 矩阵”
接口；`Width`、`Height`、像元尺寸或镜头标称焦距不能替代经过标定的像素焦距和主点。

因此采集器支持两种明确模式：

1. 推荐模式：用 MVS/ OpenCV 完成一次棋盘格或圆点板标定，把
   `camera_matrix`、可选 `distortion_coefficients` 和 `image_width`/
   `image_height` 写入 OpenCV YAML/XML，然后用 `--calibration-file` 传入。
2. 设备节点模式：如果具体相机确实暴露了标定后的 `fx/fy/cx/cy` 节点，则通过
   `--fx-node`、`--fy-node`、`--cx-node`、`--cy-node` 指定节点名；畸变节点必须
   一次性指定 `k1/k2/p1/p2/k3` 五个。

节点名称和节点类型应以 MVS 客户端当前连接设备的参数页为准；可参考海康官方的
[工业相机参数设置获取说明](https://www.hikrobotics.com/cn2/source/vision/video/2021/6/25/20210625082558512.pdf)。

示例标定文件：

```yaml
%YAML:1.0
image_width: 2448
image_height: 2048
camera_matrix: !!opencv-matrix
   rows: 3
   cols: 3
   dt: d
   data: [ 1800.0, 0.0, 1223.5, 0.0, 1802.0, 1019.5, 0.0, 0.0, 1.0 ]
distortion_coefficients: !!opencv-matrix
   rows: 1
   cols: 5
   dt: d
   data: [ -0.1, 0.02, 0.0, 0.0, -0.01 ]
```

### 编译与运行

MVS 依赖默认关闭，不会影响原有目标。安装海康 MVS（包含 Development/Includes
和 Libraries）后，在仓库根目录执行：

```powershell
cmake -S setc -B setc/build_hikvision -G "Visual Studio 16 2019" -A x64 `
  -DLIBTORCH_ROOT="C:/Dev/libtorch/2.7.0-cu128" `
  -DOpenCV_DIR="C:/Dev/opencv/4.10.0/build" `
  -DHIKVISION_MVS_ROOT="C:/Program Files (x86)/MVS/Development" `
  -DOMNIVGGT_ENABLE_HIKVISION_MVS=ON
cmake --build setc/build_hikvision --config Release --target omnivggt_hikvision_capture
```

推荐使用已标定文件：

```powershell
setc/build_hikvision/Release/omnivggt_hikvision_capture.exe `
  --device-index 0 `
  --calibration-file camera_calibration.yml `
  --output-dir hikvision_capture `
  --frames 100
```

如果设备型号的 MVS 参数页确认存在标定节点，则改用：

```powershell
setc/build_hikvision/Release/omnivggt_hikvision_capture.exe `
  --device-index 0 `
  --fx-node FX_NODE_FROM_MVS --fy-node FY_NODE_FROM_MVS `
  --cx-node CX_NODE_FROM_MVS --cy-node CY_NODE_FROM_MVS `
  --output-dir hikvision_capture
```

上面四个 `*_FROM_MVS` 只是占位符，不能直接照抄；请替换成你的相机在 MVS 参数页
中显示的实际节点名。

采集输出可以直接给 Python 流程读取，JSON sidecar 会按图片 stem 自动匹配：

```powershell
python -m stream_omnivggt.cli.run_stream_demo `
  --image-dir hikvision_capture/images `
  --camera-dir hikvision_capture/cameras `
  --mock-backend
```

注意：采集器保存的是当前 MVS 输出分辨率下的原始 BGR 图像，改变 ROI、Offset、
binning、decimation 或后续 resize 后，必须同步变换 K，最稳妥的做法是按最终输出
分辨率重新标定。MVS DLL 需要在运行时 PATH 中可见；Windows 下通常由 MVS 安装器
放入 `Common Files/MVS/Runtime`。

## 常见问题和处理

### 1. PowerShell 不允许执行 local_env.ps1

错误类似：

```text
running scripts is disabled on this system
```

处理：不要改系统策略，直接在当前 PowerShell 输入：

```powershell
$env:LIBTORCH="C:\Dev\libtorch\2.7.0-cu128"
$env:OpenCV_DIR="C:\Dev\opencv\4.10.0\build"
$env:Path="$env:LIBTORCH\lib;$env:OpenCV_DIR\x64\vc16\bin;$env:Path"
```

### 2. 找不到 DLL

如果运行 `omnivggt_edge.exe` 时提示缺 DLL，通常是 PATH 没设置。确认：

```powershell
$env:Path
```

里面应包含：

```text
C:\Dev\libtorch\2.7.0-cu128\lib
C:\Dev\opencv\4.10.0\build\x64\vc16\bin
```

### 3. find_package(Torch) 触发 CUDA 错误

不要走旧的标准 CMake 脚本路径。当前 Windows 环境使用：

```powershell
cmd /c setc\scripts\build_windows_gpu_manual.bat
```

这个脚本配合当前 `CMakeLists.txt` 手动链接 LibTorch GPU 库，避免 CMake 调 nvcc。

### 4. PyTorch 2.7.0 wheel 下载中断

这是网络问题。重新运行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File setc\scripts\setup_export_venv_torch270.ps1
```

如果仍然失败，可以手动下载对应 wheel 后在 venv 里本地安装。但原则不变：导出 TorchScript 的 PyTorch 版本最好和 LibTorch C++ 版本一致。

### 5. 不得退化为 CPU

运行命令必须使用：

```text
--device cuda
```

不要把 `--device` 改成 `cpu`。如果 CUDA 加载失败，应修复 DLL、驱动、LibTorch 版本或模型导出问题，而不是切到 CPU。

## 迁移到边缘设备

Windows x64 边缘设备需要带上：

```text
omnivggt_edge.exe
omnivggt_stream.exe
TorchScript 模型 .pt
LibTorch 2.7.0-cu128 的运行时 DLL
OpenCV 4.10.0 的运行时 DLL
输入图片目录
```

如果只需要单次推理，可以不带 `omnivggt_stream.exe`。如果需要你的外部对齐/流式融合模块，必须带 `omnivggt_stream.exe`。

当前 C++ 推理阶段不再需要携带：

```text
omnivggt/
stream_omnivggt/
checkpoints/
setc\venv_torch270_cu128/
Python 解释器和 Python 包
```

但如果目标设备需要重新导出 TorchScript，则仍然需要原项目 Python 源码、`checkpoints/OmniVGGT.safetensors` 和导出 venv。

最简单方式是在目标机保持相同目录：

```text
C:\Dev\libtorch\2.7.0-cu128
C:\Dev\opencv\4.10.0
```

然后设置 PATH 后运行。

Linux/Jetson 不能直接使用 Windows 的 LibTorch/OpenCV。需要重新准备对应平台的 LibTorch、OpenCV 和 C++ 编译产物。Jetson 上长期更建议在 TorchScript 验证后继续做 TensorRT 部署。
