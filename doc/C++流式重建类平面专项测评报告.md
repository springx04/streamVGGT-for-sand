# C++ 流式重建类平面专项测评报告

日期：2026-08-07
测试对象：setc/build_live_observer/Release/omnivggt_stream_server.exe
测试类型：单个开源类平面表面 + 已知平面单应变换的短流式回放

## 背景：现有流程的缺陷与本项目已解决点

| 原有流程问题 | 本项目的解决方式 | 当前项目依据 |
|---|---|---|
| 直接依赖多帧 VGGT 位姿融合，容易出现多层重影 | 以单个锚帧建立规范二维画布，其他帧用单应变换对齐；OmniVGGT 只提供深度和置信度，不直接承担跨帧几何融合 | 当前 C++ 流式链路记录了稳定的单应内点和像素误差 |
| 逐帧深度尺度不一致，容易形成上下分层点云 | 使用全局仿射深度对齐、重叠区域局部校正和锚定深度锁定 | 动态主方案参考深度对齐后 AbsRel 2.410%，点到点 RMSE 约为平均深度的 3.00% |
| 各帧边界直接拼接，容易出现直线接缝 | 将深度、置信度、边界羽化和跨帧一致性结合为软表面融合，形成单一高度场 | 最终输出使用连续画布和清洗导出，不采用逐帧独立点云拼接 |
| 观测孔遮挡、后续删点会留下内部孔洞 | 采用多帧支撑约束、支撑闭运算、内部孔洞填补和不可靠外边界裁剪 | 旧报告的几何优先诊断中内部孔洞为 0；当前 C++ PLY 也经过窄孔洞填补和边界清洗 |
| 孔径黑边被误当成真实表面边界 | 使用掩膜支撑和保守边界裁剪，把几何支撑作为边界依据 | C++ 导出阶段包含边界裁剪、有效性过滤和表面正则化 |
| 每一帧都完整执行模型，流式处理重复计算严重 | 引入变化区域检测、局部 ROI 推理和 no-change 跳帧 | 本次 5 帧中 2 帧跳过模型，跳帧率 40% |
| 直接删除深度离群点会同时删除有效 XY 支撑 | 对近似平面拟合残差进行裁剪、填补和正则化，尽量保留表面支撑 | 最终 VersionStore 保留 92,685 个有效点，清洗 PLY 导出 84,289 个点 |

因此，本项目的改进重点不是单纯更换一个深度模型，而是把“锚定配准—尺度对齐—软融合—支撑修复—增量更新”串成一条针对类平面表面的流式重建链路。

## 1. 结论摘要

本次测试没有使用任意室内序列，而是选取 TUM Photometric Depth Super-Resolution 数据集中的 blanket 子场景。该场景是带纹理的织物表面，适合检验项目针对“沙面/平面/缓变表面”的图像配准、变化区域更新、深度融合和跳帧逻辑。

为得到严格的类平面流，先从该子场景的一张公开 RGB 图像生成 5 帧已知单应变换回放。结果如下：

- 主方案采用动态 bucket：5 帧逐帧计时总和为 4355.319 ms，平均 871.064 ms/frame，按该回放折算为 1.148 FPS。
- 动态方案中 3 帧执行模型、2 帧跳过模型，跳帧率 40%；后续模型执行帧的平均模型耗时为 1037.469 ms，平均总耗时为 1381.687 ms，跳过模型的平均帧耗时为 165.656 ms。
- 后续帧的单应配准平均内点数为 393.8，平均重投影误差为 0.1128 px，最大为 0.121 px；动态 ROI 实际宽度为 490/518 px，部署的上界 bucket 为 518×700，相对于固定 700×700 输入减少约 26% 的模型像素量。
- 动态方案最终历史状态包含 92,685 个有效点；回放导出的清洗 PLY 包含 84,289 个点。
- 在 61,687 个共同有效点上进行同像素、尺度/偏置对齐后的三维点到点比较，MAE 为 150.8 mm、RMSE 为 181.2 mm、P95 为 339.4 mm；对应参考平均深度 6.036 m，RMSE 约为 3.00%。
- 动态方案参考深度经过全局尺度/偏置对齐后，AbsRel 为 2.410%，RMSE 为 174.9 mm，δ<1.05 为 92.14%，δ<1.10 和 δ<1.25 均为 100%。这属于参考表面的相对深度指标，不是独立激光测量意义下的绝对精度。
- 备选方案采用固定 pair-letterbox：同一序列总耗时为 5166.974 ms，平均 1033.395 ms/frame，约 0.968 FPS；重复运行约为 0.92–0.97 FPS。该方案不依赖动态模型目录，作为部署稳定性优先时的后备路径。
- 动态方案 GPU 峰值显存为 7454 MiB / 8151 MiB，约占 91.4%；C++ 进程工作集峰值约 3917.6 MB。固定备选方案已记录峰值约为 7801 MiB / 1824.1 MB。

综合判断：动态 bucket 已经完成从 Python 版到 C++ 流式路径的可测迁移。相对于固定备选方案，动态方案在精度基本一致的情况下，将完整回放吞吐从 0.968 FPS 提升到 1.148 FPS，后续实际执行模型帧的平均总耗时下降约 34.1%；固定方案则具有输入形状固定、artifact 管理简单、主机内存更低的工程优势。若用于论文最终表格，还需要在真实多帧平面序列或带独立测量真值的数据上重复测试。

## 2. 数据集和类平面测试协议

### 2.1 开源数据

数据来源：TUM Photometric Depth Super-Resolution 数据集的 xtion_blanket_sf2_sfs 子场景。官方数据页：

https://cvg.cit.tum.de/data/datasets/photometricdepthsr

本次使用的直接下载文件：

https://cvg.cit.tum.de/_media/data/datasets/photometricdepthsr/xtion_blanket_sf2_sfs.zip

本次主测实际需要的压缩数据为 xtion_blanket_sf2_sfs.zip，大小 15,074,916 bytes（约 15.1 MB），没有下载完整的大型 RGB-D 序列。MAT 文件中使用了：

| 内容 | 尺寸/用途 |
|---|---:|
| I_noise | 1280×960 RGB 输入 |
| mask_sr | 1280×960 类平面表面掩膜 |
| z_est | 1280×960 参考表面深度 |
| K_sr | 高分辨率参考内参 |

该数据是带颜色条纹纹理的 blanket 表面，不是数学意义上的无限平面；它具有连续、缓变的表面形状，因此比随机室内场景更贴合本项目的特化目标。

### 2.2 流式序列生成

为了将单张公开 RGB 图像变成可控的类平面流，对表面区域施加已知单应变换并保存为 5 张 PNG：

| 帧 | 缩放/平移/透视特征 | 目的 |
|---:|---|---|
| 0 | identity | 初始化锚帧 |
| 1 | 轻微放大，向左上移动 | 小幅平面运动 |
| 2 | 继续放大，向左上移动 | 增大变化区域 |
| 3 | 缩小，向右下移动 | 反向运动 |
| 4 | 缩小、平移并加入弱透视项 | 检查单应模型稳定性 |

生成序列的完整矩阵保存在：

stream_omnivggt_outputs/cpp_benchmark/dataset_cache/tum_photometric_blanket/planar_stream_input/sequence_metadata.json

这是一项“受控单平面单应回放”，不是原始 5 帧视频。因此它适合测配准、变化检测、局部推理、融合和历史回放一致性，但不能单独证明真实相机运动、遮挡或跨场景泛化能力。

## 3. C++ 配置和硬件

### 3.1 模型和运行参数

两种方案共用首帧模型、CUDA/BF16、对齐画布和跳帧阈值；区别只在后续 pair 模型的输入形状管理：

| 方案 | 后续模型路径 | 输入形状 | 定位 |
|---|---|---|---|
| 方案 A：动态 bucket | `--model-pair-dir .\setc\artifacts\dynamic_pair_518` | 按实际 ROI 选择 bucket；本次使用 518×700 上界 bucket | 主方案，利用变化区域减少模型输入面积 |
| 方案 B：固定 pair-letterbox | `--model-pair .\setc\artifacts\omnivggt_observer_s2_700x700_bf16_unfrozen_torch270.pt --pair-letterbox` | 始终 700×700 | 备选方案，形状固定、部署简单 |

共同参数如下：

- 首帧模型：`setc/artifacts/omnivggt_observer_s1_700x434_bf16_unfrozen_torch270.pt`
- 推理设备：CUDA；数据类型：bf16
- 首帧模型尺寸：700×434；固定备选 pair 尺寸：700×700
- 对齐画布：770×630；`min_conf=0`；queue capacity=1024；输入帧数=5

动态方案正式回放使用的命令等价于：

~~~powershell
$env:Path = 'C:\Dev\libtorch\2.7.0-cu128\lib;C:\Dev\opencv\4.10.0\build\x64\vc16\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin;' + $env:Path
& .\setc\build_live_observer\Release\omnivggt_stream_server.exe --model .\setc\artifacts\omnivggt_observer_s1_700x434_bf16_unfrozen_torch270.pt --model-pair .\setc\artifacts\omnivggt_observer_s2_700x700_bf16_unfrozen_torch270.pt --model-pair-dir .\setc\artifacts\dynamic_pair_518 --image_dir .\stream_omnivggt_outputs\cpp_benchmark\dataset_cache\tum_photometric_blanket\planar_stream_input --output_dir .\stream_omnivggt_outputs\cpp_benchmark\blanket_server_dynamic_518 --run_dir .\stream_omnivggt_outputs\cpp_benchmark\blanket_server_dynamic_518\run_20260807 --num_images 5 --once --queue_capacity 1024 --target-size 700 --target-width 700 --canvas-width 770 --canvas-height 630 --first-model-width 700 --first-model-height 434 --device cuda --dtype bf16 --min_conf 0 --no-save-debug
~~~

固定备选方案正式回放使用的命令等价于：

~~~powershell
& .\setc\build_live_observer\Release\omnivggt_stream_server.exe --model .\setc\artifacts\omnivggt_observer_s1_700x434_bf16_unfrozen_torch270.pt --model-pair .\setc\artifacts\omnivggt_observer_s2_700x700_bf16_unfrozen_torch270.pt --pair-letterbox --image_dir .\stream_omnivggt_outputs\cpp_benchmark\dataset_cache\tum_photometric_blanket\planar_stream_input --output_dir .\stream_omnivggt_outputs\cpp_benchmark\blanket_server --run_dir .\stream_omnivggt_outputs\cpp_benchmark\blanket_server\run_20260807 --num_images 5 --once --queue_capacity 1024 --target-size 700 --target-width 700 --canvas-width 770 --canvas-height 630 --first-model-width 700 --first-model-height 434 --device cuda --dtype bf16 --min_conf 0 --no-save-debug
~~~

动态 bucket 由 `setc/scripts/export_pair_bucket_family.py` 生成；本次低显存复现实验只部署了 518×700 一个上界 bucket。若后续需要更接近每个 ROI 的精确输入，可额外生成 490×700 等 bucket，C++ 侧按当前 ROI 选择并缓存。

### 3.2 硬件

| 项目 | 测试环境 |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU |
| GPU 显存 | nvidia-smi 报告 8,151 MiB |
| C++ 运行时 | LibTorch 2.7.0 CUDA 12.8 |
| OpenCV | 4.10.0 |
| 编译器 | MSVC 19.29.30159 |

## 4. 精度指标定义

### 4.1 参考深度

将 MAT 中的 z_est 与 K_sr 投影为参考相机坐标点。z_est 按本次评测的米制换算除以 1000。由于该参考来自数据集的表面重建/估计文件，而不是独立激光扫描，本报告称其为 reference surface，不把它写成独立 ground truth。

输出 PLY 的导出器会对深度做中值平移和 base-depth 归一化，且 x/y 使用画布归一化坐标。因此原始 PLY 坐标不能直接当成米制坐标。精度统计分成两部分：

1. 输出点云的归一化平面残差，用于检查是否保持连续类平面形状。
2. 将输出深度与参考 z_est 做全局尺度/偏置拟合后，计算相对深度误差。

$$
\mathrm{AbsRel} = \frac{1}{N}\sum_i\frac{|d_i-\hat d_i|}{d_i},\qquad
\mathrm{RMSE} = \sqrt{\frac{1}{N}\sum_i(d_i-\hat d_i)^2}
$$

三维点到点误差使用相同的像素对应关系：将 PLY 画布坐标映射回原始图像像素，以该像素的参考深度和相机内参分别反投影预测点与参考点，再计算两点的欧氏距离。预测深度使用上面的全局尺度/偏置关系；因此它评价的是同一表面位置的相对三维几何一致性。

同时报告 δ<1.05、δ<1.10、δ<1.25，其中 δ 指预测值和参考值的最大比值小于阈值。

### 4.2 配准精度

第 0 帧没有单应估计；第 1–4 帧统计 OpenCV 单应匹配内点数和 homography_error_px。误差单位为像素。

## 5. 测试结果

### 5.1 类平面/参考深度精度

主方案（动态 bucket）结果如下：

| 指标 | 结果 | 说明 |
|---|---:|---|
| 有效深度对应点 | 61,687 | 输出清洗点与参考表面投影后的共同有效点 |
| 同像素三维点到点 MAE（尺度/偏置对齐） | 150.8 mm | 61,687 个对应点 |
| 同像素三维点到点 RMSE（尺度/偏置对齐） | 181.2 mm | 相对于参考平均深度 6.036 m 约 3.00% |
| 同像素三维点到点 P95 | 339.4 mm | 相对于参考平均深度约 5.62% |
| 同像素三维点到点 P99 | 407.3 mm | 用于观察少量较大误差 |
| C++ 输出最佳平面 RMSE | 0.021592 | PLY 归一化坐标单位 |
| C++ 输出最佳平面 MAE | 0.019106 | PLY 归一化坐标单位 |
| C++ 输出最佳平面 P95 | 0.034874 | PLY 归一化坐标单位 |
| C++ 输出最佳平面 P99 | 0.037695 | PLY 归一化坐标单位 |
| 参考表面最佳平面 RMSE | 0.229197 m | 参考表面本身不是严格数学平面 |
| 参考表面最佳平面 P95 | 0.435715 m | 用于说明测试表面仍有缓变起伏 |
| 参考深度 AbsRel（尺度/偏置对齐） | 2.410% | 非独立绝对真值 |
| 参考深度 MAE（尺度/偏置对齐） | 145.6 mm | 61,687 个对应点；按深度定义 |
| 参考深度 RMSE（尺度/偏置对齐） | 174.9 mm | 61,687 个对应点；按深度定义 |
| 参考深度 P95 | 330.0 mm | 61,687 个对应点 |
| 参考深度 P99 | 388.6 mm | 61,687 个对应点 |
| δ<1.05 | 92.14% | 参考深度相对指标 |
| δ<1.10 | 100.00% | 参考深度相对指标 |
| δ<1.25 | 100.00% | 参考深度相对指标 |

动态方案尺度/偏置拟合得到的参考深度关系为：

$$
d_\mathrm{ref}=-2.57369\,z_\mathrm{ply}+6.03753
$$

其中负号来自当前 PLY 深度轴的归一化方向，不表示模型产生了负物理深度。这个结果更适合表述为“相对表面深度一致性”，不适合表述为绝对尺度精度。

作为固定备选的原始 `pair-letterbox` 基线为：共同有效点 62,347 个，三维 MAE/RMSE/P95/P99 分别为 150.5/182.5/346.8/430.2 mm，参考深度 AbsRel 为 2.405%，δ<1.05、δ<1.10、δ<1.25 分别为 91.30%、100.00%、100.00%，归一化平面 RMSE 为 0.022753。两种方案的误差处于同一水平，动态输入没有引入可见的精度退化。

从点云角度看，两种方案都可以评价为较好：动态方案在约 5.56–6.50 m 的参考深度范围内，经过一次全局尺度/偏置对齐后，三维点到点 RMSE 约为平均深度的 3.00%，P95 约为 5.62%；同时 92.14% 的点落在 5% 相对深度误差内，全部点落在 10% 内。这说明在当前类平面流中，点云表面形状和深度连续性保持得比较好，未出现大范围的深度发散。这里的“较好”针对相对/归一化点云精度，不能替代独立测量真值下的绝对点云误差。

### 5.2 单应配准和流式更新

| 帧 | total_ms | model_ms | 是否跳过模型 | changed_ratio | 内点数 | 误差 px | 状态有效点/更新点 |
|---:|---:|---:|---|---:|---:|---:|---:|
| 0 | 1260.635 | 416.184 | 否 | 0.0950 | 0 | -1.000 | 46,098 / 46,098 |
| 1 | 1566.407 | 1052.994 | 否 | 0.0030 | 399 | 0.101 | 46,873 / 7,013 |
| 2 | 1196.966 | 1021.943 | 否 | 0.0940 | 334 | 0.117 | 92,685 / 45,812 |
| 3 | 166.549 | 0 | 是 | 0.0010 | 445 | 0.112 | 92,685 / 0 |
| 4 | 164.762 | 0 | 是 | 0.0010 | 397 | 0.121 | 92,685 / 0 |

后续帧（1–4）统计：

- 单应内点数：平均 393.75，范围 334–445。
- 单应误差：平均 0.11275 px，范围 0.101–0.121 px。
- 后续帧中 2/4 帧跳过模型，跳帧率 50%；全序列跳帧率为 40%。
- 最终 VersionStore 历史状态为 5 帧、5 个 delta、latest_version=5。

固定备选方案的对应逐帧结果如下，保留用于无动态 bucket 部署时的复核：

| 帧 | total_ms | model_ms | 是否跳过模型 | changed_ratio | 内点数 | 误差 px | 状态有效点/更新点 |
|---:|---:|---:|---|---:|---:|---:|---:|
| 0 | 632.062 | 537.515 | 否 | 0.0950 | 0 | -1.000 | 46,098 / 46,098 |
| 1 | 2210.946 | 1698.894 | 否 | 0.0025 | 399 | 0.101 | 46,873 / 7,013 |
| 2 | 1980.283 | 1788.603 | 否 | 0.0944 | 334 | 0.117 | 92,685 / 45,812 |
| 3 | 171.722 | 0 | 是 | 0.0006 | 445 | 0.112 | 92,685 / 0 |
| 4 | 171.961 | 0 | 是 | 0.0007 | 397 | 0.121 | 92,685 / 0 |

### 5.3 时间分解

下表使用 metrics.csv 的逐帧计时；model_ms 在跳过模型的帧上为 0。P90 使用 5 个样本的线性分位数计算。

| 阶段 | 动态方案 5 帧平均 ms/frame | 动态方案总计 ms | 备注 |
|---|---:|---:|---|
| read | 29.929 | 149.645 | 图像读取 |
| align2d | 92.971 | 464.857 | 单应/局部对齐 |
| diff | 7.449 | 37.247 | 变化检测 |
| model | 498.224 | 2491.121 | 含 3 个执行模型的帧 |
| model（执行帧平均） | 830.374 | 2491.121 / 3 | 仅统计帧 0–2 |
| model（后续执行帧平均） | 1037.469 | 2074.937 / 2 | 仅统计帧 1–2 |
| depth_align | 9.916 | 49.582 | 深度对齐 |
| patch | 55.063 | 275.313 | 增量补丁/融合 |
| total | 871.064 | 4355.319 | 逐帧处理时间 |

动态方案 total 的中位数为 1196.966 ms，P90 为 1444.098 ms。去掉首帧后，帧 1–4 的平均处理时间为 773.671 ms/frame，对应 1.293 FPS；该值包含两帧动态模型推理和两帧跳帧。

固定备选方案作为对照：总计 5166.974 ms，平均 1033.395 ms/frame；其中 read/align2d/diff/model/depth_align/patch 分别为 34.904/97.977/8.507/805.002/9.523/54.672 ms/frame，模型执行帧平均 1341.671 ms，total 中位数 632.062 ms，P90 为 2118.681 ms。

#### 5.3.1 速度结论和瓶颈

从工程速度角度，动态方案和固定备选都应区分“完整流式吞吐”和“单个模型执行帧耗时”：

| 速度指标 | 动态主方案 | 固定备选方案 | 解释 |
|---|---:|---:|---|
| 5 帧完整逐帧吞吐 | 1.148 FPS | 0.968 FPS | 包含首帧、推理帧和跳帧 |
| 去掉首帧后的流式吞吐 | 1.293 FPS | 0.882 FPS | 帧 1–4 平均处理时间分别为 773.671/1133.728 ms |
| 模型执行帧平均耗时 | 830.374 ms/frame | 1341.671 ms/frame | 3 个执行模型帧 |
| 后续模型执行帧平均耗时 | 1381.687 ms/frame | 2095.615 ms/frame | 帧 1–2，包含对齐、ROI 推理和融合 |
| 后续模型执行帧平均模型耗时 | 1037.469 ms/frame | 1743.749 ms/frame | 只统计帧 1–2 的 model_ms |
| 跳过模型帧平均耗时 | 165.656 ms/frame | 171.842 ms/frame | 帧 3–4，仅做对齐、变化检测和状态维护 |
| 跳帧相对后续推理帧节省 | 88.0% | 91.8% | 动态/固定分别与各自后续推理帧比较 |

动态方案的模型阶段占本次总处理时间约 57.20%，固定方案约 77.90%；动态输入把后续模型平均耗时从 1743.749 ms 降到 1037.469 ms，下降约 40.5%。完整后续帧总耗时则从 2095.615 ms 降到 1381.687 ms，下降约 34.1%，说明 ROI 缩小确实已经传递到 C++ pair 模型执行。

需要注意首帧和后续帧不是同一种计算：首帧是单帧 700×434 模型，后续是当前帧与 anchor 的双帧 pair 模型，还要支付单应对齐、ROI 构造、深度对齐和融合。因此动态 ROI 能保证后续的 model_ms 明显下降，但不能仅凭“重建点数少了”推出每个后续 total_ms 都一定低于首帧。

对于旋转观测孔这类相对缓慢的采集过程，两种方案配合 40% 跳帧都能有效降低重复计算；它们都不应表述为 30 FPS 实时视频速度。动态方案适合追求吞吐和显存余量的部署，固定方案适合作为模型文件管理更简单、主机内存更紧张时的后备实现。

#### 5.3.2 两种方案的实现状态和选择建议

当前 C++ 已同时支持两种后续 pair 推理路径：

| 项目 | 方案 A：动态 bucket（主方案） | 方案 B：固定 pair-letterbox（备选） |
|---|---|---|
| 后续帧 ROI | 帧 1 为 490×700，帧 2 为 518×700 | 同样先计算真实 ROI |
| 实际模型输入 | 本次统一落到 518×700 上界 bucket；可扩展为多个精确 bucket | 始终 700×700 |
| 启用方式 | `--model-pair-dir .\setc\artifacts\dynamic_pair_518` | `--model-pair ... --pair-letterbox` |
| 模型面积 | 518×700，相比 700×700 少约 26% | 700×700 |
| 模型管理 | 需要准备与缓存动态形状 artifact；当前实现支持 bucket 选择和有限 LRU | 单个固定 pair artifact，部署最简单 |
| 本次结果 | 已完成 C++ 实测：1.148 FPS | 已完成 C++ 实测：0.92–0.97 FPS |

动态方案使用 `--model-pair-dir` 按当前 ROI 选择可容纳的 bucket；本次为了控制显存和测试时间，只部署了 518×700 一个上界 bucket，小于该尺寸的 ROI 通过边缘复制补齐，几何更新仍按真实 ROI 尺寸进行。若部署时准备 490×700、518×700 等多个 bucket，可进一步减少 padding，但会增加模型文件占用和显存缓存压力。

固定方案保留的意义是提供一个不依赖动态 artifact 的确定性后备路径：当设备磁盘/显存不适合管理多个 bucket，或需要最少的模型部署文件时，直接使用 700×700 `pair-letterbox` 即可。它的点云精度与动态方案处于同一水平，但后续模型执行帧较慢，且显存峰值更接近 8 GB 上限。

因此，报告中的推荐顺序是“动态 bucket 为默认，固定 pair-letterbox 为备选”。Python 附录中的动态 ROI 结果用于说明迁移方向；正文中的动态 C++ 结果则是已经在同一类平面 5 帧回放上完成的速度和精度实测。

### 5.4 点数、历史和输出

| 项目 | 结果 |
|---|---:|
| 最终 VersionStore 有效点 | 92,685 |
| 最终清洗 PLY 点数 | 84,289 |
| 5 帧更新点数累计 | 98,923 |
| 清洗 PLY 大小 | 3,606,109 bytes |
| run history 大小 | 16,918,349 bytes |

PLY 点数少于 VersionStore 有效点，是因为回放导出阶段执行了窄孔洞填补、边界裁剪、局部异常清理和表面正则化；它不是历史丢帧。

## 6. 资源占用

显存和主机工作集均使用 nvidia-smi/进程工作集采样，采样周期 250 ms：

| 资源 | 动态主方案 | 固定备选方案 |
|---|---:|---:|
| GPU baseline used | 约 642 MiB | 约 642 MiB |
| GPU peak used | 7,454 MiB | 7,801 MiB |
| GPU total | 8,151 MiB | 8,151 MiB |
| GPU peak / total | 91.4% | 95.7% |
| C++ 进程工作集峰值 | 约 3,917.6 MB | 约 1,824.1 MB |

动态方案将模型输入面积压缩到 518×700，GPU 峰值比固定方案低约 4.3 个百分点；但由于动态 bucket 的 TorchScript/JIT 缓存和 C++ 运行时状态，当前一次性部署的主机工作集较高。固定方案主机内存更低，但显存峰值更接近设备上限。不同驱动、桌面占用和后台进程会影响 nvidia-smi 的 used 值，因此这里报告的是设备级显存占用，不等同于 PyTorch allocator 的 allocated/reserved。

## 7. 可复核文件和验证

动态主方案原始结果：

- metrics.csv：`stream_omnivggt_outputs/cpp_benchmark/blanket_server_dynamic_518/run_20260807/metrics.csv`
- 历史文件夹：`stream_omnivggt_outputs/cpp_benchmark/blanket_server_dynamic_518/run_20260807/`
- 最终 PLY：`stream_omnivggt_outputs/cpp_benchmark/blanket_server_dynamic_518/final_pointcloud.ply`
- 输入序列：stream_omnivggt_outputs/cpp_benchmark/dataset_cache/tum_photometric_blanket/planar_stream_input/

固定备选方案原始结果：

- metrics.csv：`stream_omnivggt_outputs/cpp_benchmark/blanket_server/run_20260807/metrics.csv`
- 历史文件夹：`stream_omnivggt_outputs/cpp_benchmark/blanket_server/run_20260807/`
- 最终 PLY：`stream_omnivggt_outputs/cpp_benchmark/blanket_server/final_pointcloud.ply`

历史校验命令：

~~~powershell
& .\setc\build_live_observer\Release\omnivggt_validate_history.exe --run_dir .\stream_omnivggt_outputs\cpp_benchmark\blanket_server_dynamic_518\run_20260807
~~~

实际校验结果：

~~~text
history valid: frames=5 deltas=5 latest_frame=4 latest_version=5 hash=16618848639565797231
~~~

固定备选方案的历史校验结果为：

~~~powershell
& .\setc\build_live_observer\Release\omnivggt_validate_history.exe --run_dir .\stream_omnivggt_outputs\cpp_benchmark\blanket_server\run_20260807
~~~

~~~text
history valid: frames=5 deltas=5 latest_frame=4 latest_version=5 hash=4536601731698089429
~~~

最终 PLY 由回放程序生成：

~~~powershell
& .\setc\build_live_observer\Release\omnivggt_replay_log.exe --run_dir .\stream_omnivggt_outputs\cpp_benchmark\blanket_server_dynamic_518\run_20260807 --output_ply .\stream_omnivggt_outputs\cpp_benchmark\blanket_server_dynamic_518\final_pointcloud.ply
~~~

成功回放结果为 frame=4、version=5、state points=92,685，PLY vertex=84,289。

## 8. 论文使用时的限制和后续建议

1. 当前 5 帧来自单张真实表面图像的已知单应变换，不是原始真实相机视频；因此本实验重点是类平面配准、更新和跳帧机制。
2. z_est 是数据集提供的 reference surface，不是独立的激光/结构光真值；AbsRel、RMSE 和 δ 指标经过全局尺度/偏置对齐，只能作为相对表面深度一致性。
3. 当前 PLY 导出器对深度做归一化，原始 PLY 坐标没有可直接解释的米制尺度；如果要报告绝对 Chamfer、绝对点到面误差或真实尺度法向误差，需要让 C++ 输出相机内参/尺度并使用独立测量真值。
4. 5 帧样本量不足以给出稳定的均值/方差；最终论文建议至少使用多个不同纹理、不同倾角和不同运动幅度的平面序列，并报告 mean±std、P50/P90 和显存峰值。
5. 动态主方案显存峰值约占 8 GB 的 91.4%，固定备选约占 95.7%；后续若要提高分辨率、增加模型分支、部署更多动态 bucket 或扩大队列，都需要先做显存预算。

本报告因此可以作为当前 C++ 流式重建链路的专项基线，而不是最终的通用三维重建榜单结果。

## 附录 A：Python 版流式重建测试结果

本附录单独记录已有的 Python 版 live replay 结果，用来说明动态 ROI 和模型推理时间的参考上限。数据来源为 `stream_omnivggt_outputs/data2_python_live_replay/timings.md`，对应 12 帧序列；它与正文中的 C++ 动态主方案、固定备选方案分别统计，不混合计算。

### A.1 测试配置和总体结果

| 项目 | Python 结果 |
|---|---:|
| 后端 | `omnivggt-pytorch` |
| 图像帧数 | 12 |
| 后端加载时间 | 13,836.832 ms |
| 首次输入到点云（含后端加载） | 15,124.111 ms |
| 首帧总时间 | 1,222.280 ms |
| 首帧模型时间 | 1,140.262 ms |
| 后续有模型执行帧平均总时间 | 1,462.262 ms |
| 后续有模型执行帧 P90 总时间 | 1,859.015 ms |
| 后续有模型执行帧平均模型时间 | 814.393 ms |
| 最终点数（timings summary） | 186,011 |

后续模型执行帧的模型时间相比首帧减少约 28.6%（1,140.262 → 814.393 ms），说明 Python 版确实把动态 ROI 迁移到了较小的模型输入上；但后续整帧平均总时间为首帧的约 1.20 倍，原因是后续帧还包含当前帧与 anchor 的单应对齐、变化检测、ROI 构造、深度对齐和点云融合。因而“动态尺寸使模型更快”与“整帧总耗时必然比首帧更短”是两个不同结论。

Python 版的输入形状不是固定的：项目记录显示 frame 1 使用 `700×672` 的双帧输入，frame 4 使用 `560×700` bucket。正文的 C++ 动态主方案已经完成对应的 TorchScript pair 路径实测；本次为控制显存，采用 518×700 上界 bucket，而不是为每个 ROI 准备完整 bucket 家族。

### A.2 逐帧结果

| 帧 | total (ms) | model (ms) | changed ratio | delta pixels | anchor pixels | points |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1,222.28 | 1,140.26 | 0.3360 | 163,010 | 0 | 163,010 |
| 1 | 2,106.99 | 1,296.62 | 0.0212 | 20,720 | 29,627 | 171,301 |
| 2 | 1,859.01 | 1,102.41 | 0.0102 | 6,083 | 18,295 | 174,152 |
| 3 | 1,812.22 | 992.98 | 0.0115 | 7,362 | 20,030 | 178,327 |
| 4 | 1,515.49 | 776.67 | 0.0506 | 25,798 | 20,004 | 192,840 |
| 5 | 1,553.86 | 827.92 | 0.0120 | 5,537 | 19,009 | 193,695 |
| 6 | 1,467.92 | 732.67 | 0.0115 | 1,042 | 18,782 | 193,708 |
| 7 | 1,613.45 | 879.02 | 0.0089 | 1,360 | 14,853 | 193,737 |
| 8 | 1,683.14 | 892.09 | 0.0066 | 1,081 | 13,167 | 193,764 |
| 9 | 1,160.44 | 733.22 | 0.0064 | 3,086 | 13,961 | 193,764 |
| 10 | 1,173.49 | 724.72 | 0.0047 | 2,286 | 5,624 | 193,764 |
| 11 | 138.87 | 0.00 | 0.0000 | 0 | 0 | 193,764 |

frame 11 未执行模型，仅维护流式状态，因此它的 `model_ms=0`，不能作为动态 ROI 推理速度样本。Python 结果的正确解读是：动态 ROI 让模型阶段明显缩短；整帧吞吐仍由几何对齐、融合和跳帧策略共同决定。该结果构成正文 C++ 动态 bucket 结果的迁移前参考基线。
