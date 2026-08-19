# OmniVGGT C++ Linux 版

这是一个可以单独下载的 Linux C++ 发布目录。目录内已经包含 README、依赖说明、CMake 配置、C++ 源码和 Linux 脚本；不需要再下载仓库外的 README、`requirements.txt` 或 C++ 文件。

为了避免提交大文件，本目录没有包含输入数据、checkpoint、TorchScript 模型、Python 模型源码或构建产物。它们由使用者按下面的约定自行放置，或通过环境变量指定。已经拥有 `.pt` 模型时，不需要安装 Python，也不需要模型源码。

## 1. 下载后的目录和文件放置

进入本目录后，可以按下面的结构准备运行文件：

```text
setc_linux/
├── CMakeLists.txt
├── README.md
├── requirements.txt                         # 仅重新导出模型时需要
├── scripts/
│   ├── build_linux.sh
│   ├── build_linux_live_observer.sh
│   ├── start_cpp_live_replay.sh
│   ├── export_torchscript.py
│   └── export_pair_bucket_family.py
├── src/                                      # C++ 源码
├── data1/                                    # 使用者自行放入 PNG/JPG 图片
├── models/                                   # 使用者自行放入 .pt 和可选 checkpoint
│   ├── omnivggt_observer_s1_700x434_bf16_unfrozen_torch270.pt
│   ├── omnivggt_observer_s2_700x700_bf16_unfrozen_torch270.pt
│   └── omnivggt_full_b1s3_406x252_bf16_unfrozen_torch270.pt
├── outputs/                                  # 运行时自动生成
└── build_live_observer/                      # 编译时自动生成
```

默认三图运行只需要
`models/omnivggt_full_b1s3_406x252_bf16_unfrozen_torch270.pt`；仅在设置
`INPUT_GROUP_SIZE=1` 使用旧单图/双图路径时，才需要前两个 S1/S2 模型。

默认输入目录是 `data1/`，默认输出目录是 `outputs/data1_cpp_linux_live_replay/`。模型和数据也可以完全放在目录外：

```bash
GROUP_MODEL=/data/models/group_b1s3.pt \
IMAGE_DIR=/data/scene_images \
OUTPUT_DIR=/data/omnivggt_output \
bash scripts/start_cpp_live_replay.sh
```

输入图片支持 `.png`、`.jpg` 和 `.jpeg`。目录内只放程序和配置，不会自动下载或复制数据、模型。

## 2. Linux 依赖

C++ 编译和运行需要在 Linux 机器上准备：

- CMake 3.18 或更高版本、支持 C++17 的编译器和 pthread；
- 与 TorchScript 模型版本匹配的 Linux LibTorch。当前模型按 LibTorch/PyTorch 2.7.0 + cu128 导出；你的 NVIDIA 595.84 驱动报告 CUDA 13.2，向后兼容运行 cu128，不需要安装 `nvcc`；
- OpenCV 开发包，至少包含 `core`、`imgcodecs`、`imgproc`、`features2d`、`calib3d`；live viewer 还需要 `highgui`；
- NVIDIA 驱动和 CUDA 运行环境。C++ observer 当前只支持 `--device cuda`；本项目没有 `.cu` 自定义 kernel，因此不要求 `nvcc`；
- 只有在重新导出 TorchScript 模型时，才需要 Python、CUDA 版 PyTorch、`safetensors` 和外部 OmniVGGT Python 模型源码。

本目录的脚本不会安装依赖。编译时设置 LibTorch：

```bash
export LIBTORCH=/opt/libtorch/2.7.0-cu128
# Ubuntu 24.04 + apt OpenCV 4.6 通常会被自动找到；自定义安装时再设置：
# export OpenCV_DIR=/opt/opencv/lib/cmake/opencv4
# 构建脚本默认使用 PATH 中的 g++（当前设备为 g++ 13.3）。
```

如果自定义 OpenCV 的动态库不在系统搜索路径，还可以补充：

```bash
export LD_LIBRARY_PATH=/opt/opencv/lib:${LD_LIBRARY_PATH:-}
```

## 3. 可选 Python 导出环境

`requirements.txt` 是本目录自己的文件，只服务于可选的模型导出脚本，不是 C++ 运行时依赖。已有 `.pt` 文件时可以跳过这一节。

```bash
python3 -m venv .venv-export
source .venv-export/bin/activate

# 先按本机 CUDA 版本安装匹配的 PyTorch；项目当前使用 2.7.0 + cu128 示例：
python3 -m pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu128
python3 -m pip install -r requirements.txt
```

导出脚本不包含 Python 版 OmniVGGT 模型实现。重新导出时，通过 `--model-source` 指向一个包含 `omnivggt/` 包的外部目录；这只是导出输入，不是 C++ 运行时依赖。

## 4. 编译

在 `setc_linux` 目录中执行：

```bash
cd /path/to/setc_linux
export LIBTORCH=/opt/libtorch/2.7.0-cu128
bash scripts/build_linux_live_observer.sh
```

编译结果位于 `build_live_observer/bin/`，包括：

- `omnivggt_stream_server`
- `omnivggt_live_viewer`
- `omnivggt_edge`
- `omnivggt_stream`
- history/replay 工具和 `omnivggt_observer_core_smoke`

只编译两个离线 CLI 时可以运行：

```bash
bash scripts/build_linux.sh
```

也可以手动指定构建目录或并行度：

```bash
BUILD_DIR=/tmp/omnivggt-linux-build \
CMAKE_BUILD_PARALLEL_LEVEL=8 \
bash scripts/build_linux_live_observer.sh
```

## 5. 可选模型导出

如果模型 checkpoint 放在本目录的 `models/OmniVGGT.safetensors`，外部 Python 模型源码位于 `/path/to/OmniVGGT-Python/`，可以这样生成默认运行所需的单帧模型：

```bash
python3 scripts/export_torchscript.py \
  --model-source /path/to/OmniVGGT-Python \
  --checkpoint models/OmniVGGT.safetensors \
  --output models/omnivggt_observer_s1_700x434_bf16_unfrozen_torch270.pt \
  --num-images 1 --height 434 --width 700 \
  --dtype bfloat16 --autocast-dtype bfloat16 \
  --observer-depth-only --no-freeze
```

固定 700x700 pair 模型：

```bash
python3 scripts/export_torchscript.py \
  --model-source /path/to/OmniVGGT-Python \
  --checkpoint models/OmniVGGT.safetensors \
  --output models/omnivggt_observer_s2_700x700_bf16_unfrozen_torch270.pt \
  --num-images 2 --height 700 --width 700 \
  --dtype bfloat16 --autocast-dtype bfloat16 \
  --observer-depth-only --no-freeze
```

三图 B=1,S=3 模型：

```bash
python3 scripts/export_torchscript.py \
  --model-source /path/to/OmniVGGT-Python \
  --checkpoint models/OmniVGGT.safetensors \
  --output models/omnivggt_full_b1s3_406x252_bf16_unfrozen_torch270.pt \
  --batch-size 1 --num-images 3 --height 252 --width 406 \
  --dtype bfloat16 --autocast-dtype bfloat16 \
  --no-freeze
```

动态 pair bucket 可使用：

```bash
python3 scripts/export_pair_bucket_family.py \
  --model-source /path/to/OmniVGGT-Python \
  --checkpoint models/OmniVGGT.safetensors \
  --output-dir models
```

## 6. 运行 live replay

模型、数据和程序准备好后，在 `setc_linux` 目录中运行：

```bash
bash scripts/start_cpp_live_replay.sh
```

默认配置与 Windows C++ 流程保持一致：每个逻辑帧使用一个不重叠的三相机组
（`1-1`、`1-2`、`1-3`，然后 `2-1`、`2-2`、`2-3`、…），组内锚点为中图，一次
 `B=1,S=3` CUDA BF16 forward。三相机在模型内共享世界坐标；下方低纹理区域经过稳健曲面拟合后独立做 XY 栅格压缩，锚相机拥有重叠区域，侧相机只扩展空缺。机械臂和其他非平面点使用独立槽位并只保留锚相机观测，避免与底面竞争或产生多机械臂重影。组模型输入 `406x252`，700x700 对齐画布、队列
容量 1024、端口 37651。服务器日志写入输出目录的 `server.log`。关闭 viewer 后，
启动脚本只会清理它自己启动的 server PID，不会按进程名结束其他任务。

对于 `data2` 这类旧的时序数据，可设置 `INPUT_GROUP_STRIDE=1` 恢复滑动窗口
（`123`、`234`、…）。

若要兼容原有单图/双图路径，可设置：

```bash
INPUT_GROUP_SIZE=1 bash scripts/start_cpp_live_replay.sh
```

直接启动 `omnivggt_stream_server` 时同样默认使用不重叠三图组；只有显式传入
`--input-group-size 1` 才会切换回单图/双图路径。

三图运行目录中的 `input_groups.csv` 记录每个三相机组，`metrics.csv` 中的
`forward_calls=1`、`forward_batch_size=1`、`forward_sequence_size=3` 是速度和形状验收字段。

如果 LibTorch 或 CUDA 动态库不在系统搜索路径，启动脚本会自动把 `${LIBTORCH}/lib` 和 `${CUDA_HOME}/lib64`（如果存在）加入 `LD_LIBRARY_PATH`：

```bash
export LIBTORCH=/opt/libtorch/2.7.0-cu128
# CUDA_HOME 仅在你有 CUDA toolkit/runtime 目录时设置；没有 nvcc 也可以运行：
# export CUDA_HOME=/usr/local/cuda
bash scripts/start_cpp_live_replay.sh
```

live viewer 需要 Linux 图形会话（X11/Wayland）。如果只需要服务器或离线工具，可以直接运行 `build_live_observer/bin/` 下对应的可执行文件并查看 `--help`。

## 7. 基础测试和当前验证范围

不需要数据或模型即可运行 history smoke test：

```bash
ctest --test-dir build_live_observer --output-on-failure
```

也可以单独执行：

```bash
build_live_observer/bin/omnivggt_observer_core_smoke
```

当前 Windows 电脑没有 Linux 编译环境，本次只做了 Linux CMake/路径审查、shell 语法检查和 Python 导出脚本语法检查，没有重新安装环境或冒充完成 CUDA/GUI 实机测试。请在 Linux 机器上按本文档完成首次构建和运行验证。
