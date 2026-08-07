# C++ 流式重建类平面专项测评报告

日期：2026-08-07
测试对象：setc/build_live_observer/Release/omnivggt_stream_server.exe
测试类型：单个开源类平面表面 + 已知平面单应变换的短流式回放

## 背景：现有流程的缺陷与本项目已解决点

本节参考项目已有的《沙面点云重建报告》，只整理“开发过程中已经观察到、且当前项目已经通过实现处理”的问题，不把尚未解决的沙粒级细节恢复、绝对尺度标定等内容写成已解决。

| 原有流程问题 | 本项目的解决方式 | 当前项目依据 |
|---|---|---|
| 直接依赖多帧 VGGT 位姿融合，容易出现多层重影 | 以单个锚帧建立规范二维画布，其他帧用单应变换对齐；OmniVGGT 只提供深度和置信度，不直接承担跨帧几何融合 | 当前 C++ 流式链路记录了稳定的单应内点和像素误差 |
| 逐帧深度尺度不一致，容易形成上下分层点云 | 使用全局仿射深度对齐、重叠区域局部校正和锚定深度锁定 | 本次参考深度对齐后 AbsRel 2.405%，点到点 RMSE 约为平均深度的 3.02% |
| 各帧边界直接拼接，容易出现直线接缝 | 将深度、置信度、边界羽化和跨帧一致性结合为软表面融合，形成单一高度场 | 最终输出使用连续画布和清洗导出，不采用逐帧独立点云拼接 |
| 观测孔遮挡、后续删点会留下内部孔洞 | 采用多帧支撑约束、支撑闭运算、内部孔洞填补和不可靠外边界裁剪 | 旧报告的几何优先诊断中内部孔洞为 0；当前 C++ PLY 也经过窄孔洞填补和边界清洗 |
| 孔径黑边被误当成真实表面边界 | 使用掩膜支撑和保守边界裁剪，把几何支撑作为边界依据 | C++ 导出阶段包含边界裁剪、有效性过滤和表面正则化 |
| 每一帧都完整执行模型，流式处理重复计算严重 | 引入变化区域检测、局部 ROI 推理和 no-change 跳帧 | 本次 5 帧中 2 帧跳过模型，跳帧率 40% |
| 直接删除深度离群点会同时删除有效 XY 支撑 | 对近似平面拟合残差进行裁剪、填补和正则化，尽量保留表面支撑 | 最终 VersionStore 保留 92,685 个有效点，清洗 PLY 导出 84,289 个点 |

因此，本项目的改进重点不是单纯更换一个深度模型，而是把“锚定配准—尺度对齐—软融合—支撑修复—增量更新”串成一条针对类平面表面的流式重建链路。

## 1. 结论摘要

本次测试没有使用任意室内序列，而是选取 TUM Photometric Depth Super-Resolution 数据集中的 blanket 子场景。该场景是带纹理的织物表面，适合检验项目针对“沙面/平面/缓变表面”的图像配准、变化区域更新、深度融合和跳帧逻辑。

为得到严格的类平面流，先从该子场景的一张公开 RGB 图像生成 5 帧已知单应变换回放。结果如下：

- 5 帧逐帧计时总和为 5166.974 ms，平均 1033.395 ms/frame，按该回放折算为 0.968 FPS。
- 5 帧中 3 帧执行模型、2 帧跳过模型，跳帧率 40%；跳过模型的平均帧耗时为 171.842 ms。
- 后续帧的单应配准平均内点数为 393.8，平均重投影误差为 0.1128 px，最大为 0.121 px。
- 最终历史状态包含 92,685 个有效点；回放导出的清洗 PLY 包含 84,289 个点。
- 在 62,347 个共同有效点上进行同像素、尺度/偏置对齐后的三维点到点比较，MAE 为 150.5 mm、RMSE 为 182.5 mm、P95 为 346.8 mm；对应参考平均深度 6.033 m，RMSE 约为 3.02%。
- 在成功复跑中，RTX 5060 Laptop GPU 的 nvidia-smi 显存峰值为 7,801 MiB / 8,151 MiB，约占 95.7%；C++ 进程工作集峰值约 1,824 MB。
- 参考深度经过全局尺度/偏置对齐后，AbsRel 为 2.405%，RMSE 为 175.8 mm，δ<1.05 为 91.30%，δ<1.10 和 δ<1.25 均为 100%。这属于参考表面的相对深度指标，不是独立激光测量意义下的绝对精度。

综合判断：当前 C++ 流式链路的类平面配准和跳帧机制工作正常，但模型实际执行帧约 1.34 s/frame，且显存已经接近 8 GB 上限。若用于论文最终表格，还需要在真实多帧平面序列或带独立测量真值的数据上重复测试。

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

- 首帧模型：setc/artifacts/omnivggt_observer_s1_700x434_bf16_unfrozen_torch270.pt
- 后续帧模型：setc/artifacts/omnivggt_observer_s2_700x700_bf16_unfrozen_torch270.pt
- 推理设备：CUDA
- 数据类型：bf16
- 目标尺寸：700×700
- 首帧模型尺寸：700×434
- 对齐画布：770×630
- 后续帧：pair-letterbox
- min_conf：0
- queue capacity：1024
- 输入帧数：5

正式回放使用的命令等价于：

~~~powershell
$env:Path = 'C:\Dev\libtorch\2.7.0-cu128\lib;C:\Dev\opencv\4.10.0\build\x64\vc16\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin;' + $env:Path
& .\setc\build_live_observer\Release\omnivggt_stream_server.exe --model .\setc\artifacts\omnivggt_observer_s1_700x434_bf16_unfrozen_torch270.pt --model-pair .\setc\artifacts\omnivggt_observer_s2_700x700_bf16_unfrozen_torch270.pt --pair-letterbox --image_dir .\stream_omnivggt_outputs\cpp_benchmark\dataset_cache\tum_photometric_blanket\planar_stream_input --output_dir .\stream_omnivggt_outputs\cpp_benchmark\blanket_server --run_dir .\stream_omnivggt_outputs\cpp_benchmark\blanket_server\run_20260807 --num_images 5 --once --queue_capacity 1024 --target-size 700 --target-width 700 --canvas-width 770 --canvas-height 630 --first-model-width 700 --first-model-height 434 --device cuda --dtype bf16 --min_conf 0 --no-save-debug
~~~

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

| 指标 | 结果 | 说明 |
|---|---:|---|
| 有效深度对应点 | 62,347 | 输出清洗点与参考表面投影后的共同有效点 |
| 同像素三维点到点 MAE（尺度/偏置对齐） | 150.5 mm | 62,347 个对应点 |
| 同像素三维点到点 RMSE（尺度/偏置对齐） | 182.5 mm | 相对于参考平均深度 6.033 m 约 3.02% |
| 同像素三维点到点 P95 | 346.8 mm | 相对于参考平均深度约 5.75% |
| 同像素三维点到点 P99 | 430.2 mm | 用于观察少量较大误差 |
| C++ 输出最佳平面 RMSE | 0.022753 | PLY 归一化坐标单位 |
| C++ 输出最佳平面 MAE | 0.020613 | PLY 归一化坐标单位 |
| C++ 输出最佳平面 P95 | 0.032463 | PLY 归一化坐标单位 |
| 参考表面最佳平面 RMSE | 0.229197 m | 参考表面本身不是严格数学平面 |
| 参考表面最佳平面 P95 | 0.435715 m | 用于说明测试表面仍有缓变起伏 |
| 参考深度 AbsRel（尺度/偏置对齐） | 2.405% | 非独立绝对真值 |
| 参考深度 MAE（尺度/偏置对齐） | 145.2 mm | 62,347 个对应点 |
| 参考深度 RMSE（尺度/偏置对齐） | 175.8 mm | 62,347 个对应点 |
| 参考深度 P95 | 336.4 mm | 62,347 个对应点 |
| δ<1.05 | 91.30% | 参考深度相对指标 |
| δ<1.10 | 100.00% | 参考深度相对指标 |
| δ<1.25 | 100.00% | 参考深度相对指标 |

尺度/偏置拟合得到的参考深度关系为：

$$
d_\mathrm{ref}=-2.62085\,z_\mathrm{ply}+6.03289
$$

其中负号来自当前 PLY 深度轴的归一化方向，不表示模型产生了负物理深度。这个结果更适合表述为“相对表面深度一致性”，不适合表述为绝对尺度精度。

从点云角度看，本次结果可以评价为较好：在约 5.56–6.50 m 的参考深度范围内，经过一次全局尺度/偏置对齐后，三维点到点 RMSE 约为平均深度的 3.02%，P95 约为 5.75%；同时 91.30% 的点落在 5% 相对深度误差内，全部点落在 10% 内。这说明在当前类平面流中，点云表面形状和深度连续性保持得比较好，未出现大范围的深度发散。这里的“较好”针对相对/归一化点云精度，不能替代独立测量真值下的绝对点云误差。

### 5.2 单应配准和流式更新

| 帧 | total_ms | model_ms | 是否跳过模型 | changed_ratio | 内点数 | 误差 px | 状态有效点/更新点 |
|---:|---:|---:|---|---:|---:|---:|---:|
| 0 | 632.062 | 537.515 | 否 | 0.0950 | 0 | -1.000 | 46,098 / 46,098 |
| 1 | 2210.946 | 1698.894 | 否 | 0.0025 | 399 | 0.101 | 46,873 / 7,013 |
| 2 | 1980.283 | 1788.603 | 否 | 0.0944 | 334 | 0.117 | 92,685 / 45,812 |
| 3 | 171.722 | 0 | 是 | 0.0006 | 445 | 0.112 | 92,685 / 0 |
| 4 | 171.961 | 0 | 是 | 0.0007 | 397 | 0.121 | 92,685 / 0 |

后续帧（1–4）统计：

- 单应内点数：平均 393.75，范围 334–445。
- 单应误差：平均 0.11275 px，范围 0.101–0.121 px。
- 后续帧中 2/4 帧跳过模型，跳帧率 50%；全序列跳帧率为 40%。
- 最终 VersionStore 历史状态为 5 帧、5 个 delta、latest_version=5。

### 5.3 时间分解

下表使用 metrics.csv 的逐帧计时；model_ms 在跳过模型的帧上为 0。P90 使用 5 个样本的线性分位数计算。

| 阶段 | 5 帧平均 ms/frame | 总计 ms | 备注 |
|---|---:|---:|---|
| read | 34.904 | 174.522 | 图像读取 |
| align2d | 97.977 | 489.887 | 单应/局部对齐 |
| diff | 8.507 | 42.534 | 变化检测 |
| model | 805.002 | 4025.012 | 含 3 个执行模型的帧 |
| model（执行帧平均） | 1341.671 | 4025.012 / 3 | 仅统计帧 0–2 |
| depth_align | 9.523 | 47.615 | 深度对齐 |
| patch | 54.672 | 273.362 | 增量补丁/融合 |
| total | 1033.395 | 5166.974 | 逐帧处理时间 |

total 的中位数为 632.062 ms，P90 为 2118.681 ms。去掉首帧后，帧 1–4 的平均处理时间为 1133.728 ms/frame，对应 0.882 FPS；该值包含两帧模型推理和两帧跳帧。

#### 5.3.1 速度结论和瓶颈

从工程速度角度，本次结果应区分“完整流式吞吐”和“单个模型执行帧耗时”：

| 速度指标 | 结果 | 解释 |
|---|---:|---|
| 5 帧完整逐帧吞吐 | 0.968 FPS | 5,166.974 ms / 5 帧，包含首帧、推理帧和跳帧 |
| 去掉首帧后的流式吞吐 | 0.882 FPS | 帧 1–4 的平均处理时间 1,133.728 ms |
| 模型执行帧平均耗时 | 1,341.671 ms/frame | 3 个执行模型帧，折算模型执行能力约 0.745 FPS |
| 后续模型执行帧平均耗时 | 2,095.615 ms/frame | 帧 1–2，包含对齐、ROI 推理和融合 |
| 跳过模型帧平均耗时 | 171.842 ms/frame | 帧 3–4，仅做对齐、变化检测和状态维护 |
| 跳帧相对后续推理帧节省 | 91.8% | 每个跳帧约节省 1,923.773 ms |

时间瓶颈主要集中在模型执行：模型阶段占本次总处理时间约 77.90%，其次为二维对齐 9.48% 和增量 patch/融合 5.29%。这说明当前项目的流式优化已经有效减少了重复推理，但在 700×700、BF16 和当前双模型配置下，速度上限仍主要由模型推理决定。

资源采样的重复运行使用相同配置，第二次 5 帧总计为 5,446.351 ms，即 0.918 FPS；两次吞吐相差约 5.13%，且跳帧行为和点数结果一致。因此当前机器上的可复现实用速度可写成约 0.92–0.97 FPS，而不是只引用一次运行的单点值。对于旋转观测孔这类相对缓慢的采集过程，该速度配合 40% 跳帧可以有效降低重复计算；它不应表述为 30 FPS 实时视频速度。

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

显存和主机工作集使用成功复跑 blanket_server_mem3 采样，采样周期 250 ms：

| 资源 | 结果 |
|---|---:|
| GPU baseline used | 约 642 MiB |
| GPU peak used | 7,801 MiB |
| GPU total | 8,151 MiB |
| GPU peak / total | 95.7% |
| C++ 进程工作集峰值 | 约 1,824.1 MB |

显存峰值接近设备上限，论文或产品测试中应把这一项作为部署约束报告；不同驱动、桌面占用和后台进程会影响 nvidia-smi 的 used 值，因此这里报告的是设备级显存占用，不等同于 PyTorch allocator 的 allocated/reserved。

## 7. 可复核文件和验证

主测原始结果：

- metrics.csv：stream_omnivggt_outputs/cpp_benchmark/blanket_server/run_20260807/metrics.csv
- 历史文件夹：stream_omnivggt_outputs/cpp_benchmark/blanket_server/run_20260807/
- 最终 PLY：stream_omnivggt_outputs/cpp_benchmark/blanket_server/final_pointcloud.ply
- 输入序列：stream_omnivggt_outputs/cpp_benchmark/dataset_cache/tum_photometric_blanket/planar_stream_input/

历史校验命令：

~~~powershell
& .\setc\build_live_observer\Release\omnivggt_validate_history.exe --run_dir .\stream_omnivggt_outputs\cpp_benchmark\blanket_server\run_20260807
~~~

实际校验结果：

~~~text
history valid: frames=5 deltas=5 latest_frame=4 latest_version=5 hash=4536601731698089429
~~~

最终 PLY 由回放程序生成：

~~~powershell
& .\setc\build_live_observer\Release\omnivggt_replay_log.exe --run_dir .\stream_omnivggt_outputs\cpp_benchmark\blanket_server\run_20260807 --output_ply .\stream_omnivggt_outputs\cpp_benchmark\blanket_server\final_pointcloud.ply
~~~

成功回放结果为 frame=4、version=5、state points=92,685，PLY vertex=84,289。

## 8. 论文使用时的限制和后续建议

1. 当前 5 帧来自单张真实表面图像的已知单应变换，不是原始真实相机视频；因此本实验重点是类平面配准、更新和跳帧机制。
2. z_est 是数据集提供的 reference surface，不是独立的激光/结构光真值；AbsRel、RMSE 和 δ 指标经过全局尺度/偏置对齐，只能作为相对表面深度一致性。
3. 当前 PLY 导出器对深度做归一化，原始 PLY 坐标没有可直接解释的米制尺度；如果要报告绝对 Chamfer、绝对点到面误差或真实尺度法向误差，需要让 C++ 输出相机内参/尺度并使用独立测量真值。
4. 5 帧样本量不足以给出稳定的均值/方差；最终论文建议至少使用多个不同纹理、不同倾角和不同运动幅度的平面序列，并报告 mean±std、P50/P90 和显存峰值。
5. 当前显存峰值约占 8 GB 的 95.7%，后续若要提高分辨率、增加模型分支或扩大队列，需要先做显存预算。

本报告因此可以作为当前 C++ 流式重建链路的专项基线，而不是最终的通用三维重建榜单结果。
