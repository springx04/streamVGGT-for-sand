# 工控机 SSH 与真实 GUI 操作手册

本文记录本任务中已经实际使用过的工控机连接、文件传输、GUI 检查、自动点击和截图方法，供下一位 AI 直接继续工作。

## 1. 环境信息

- 工控机地址：`100.82.98.79`
- SSH 用户：`albert`
- 本机私钥：`C:\Users\30738\.ssh\codexkey`
- 工控机项目：`/home/albert/桌面/slave_robot_runtime_cpp./setc_linux`
- GUI 可执行文件：`/home/albert/桌面/slave_robot_runtime_cpp./slave_robot_runtime_cpp/build/slave_robot_gui`
- GUI 属于项目其它部分，只允许运行和只读观察，不允许修改。
- 工控机没有可用的项目 Git 工作流。修改 `setc_linux` 前应在其 `backups/` 下建立带时间戳的可恢复文件备份。
- GUI、相机和点云重建都是真实环境，禁止用模拟输入代替验收。

不要记录、输出或上传私钥内容。上述路径只表示使用已经配置好的本机密钥。

## 2. 从 Windows PowerShell 执行 SSH

推荐统一使用下面的形式：

```powershell
& 'C:\Windows\System32\OpenSSH\ssh.exe' `
  -T -n `
  -i 'C:\Users\30738\.ssh\codexkey' `
  -o IdentitiesOnly=yes `
  -o BatchMode=yes `
  -o ConnectTimeout=30 `
  albert@100.82.98.79 `
  '<Linux Bash 命令>'
```

例如检查 GUI 和重建服务：

```powershell
& 'C:\Windows\System32\OpenSSH\ssh.exe' -T -n `
  -i 'C:\Users\30738\.ssh\codexkey' `
  -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 `
  albert@100.82.98.79 `
  'ps -eo pid,lstart,args | grep -E "slave_robot_gui|omnivggt_stream_server" | grep -v grep || true'
```

注意：

- PowerShell 的 `$env:...`、`Get-Content`、`Test-Path`、调用运算符 `&` 只能在 Windows PowerShell 提示符执行，不能粘贴到登录后的 Linux Bash 提示符。
- 远端路径中包含中文和一个真实存在的结尾点：`slave_robot_runtime_cpp.`。不要擅自删掉这个点。
- 复杂远端命令尽量整体放在单引号中，远端路径用 Bash 双引号保护。
- SSH 偶尔需要 10～30 秒返回；不要因为短暂无输出马上重复启动同一服务。

## 3. 文件上传和下载

下载工控机文件到本地：

```powershell
& 'C:\Windows\System32\OpenSSH\scp.exe' `
  -i 'C:\Users\30738\.ssh\codexkey' `
  -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 `
  'albert@100.82.98.79:/home/albert/桌面/slave_robot_runtime_cpp./setc_linux/src/observer/inference_pipeline.cpp' `
  'C:\Users\30738\Desktop\project\3dre\VGGT\OmniVGGT\tmp\inference_pipeline.cpp'
```

上传本地文件到工控机：

```powershell
& 'C:\Windows\System32\OpenSSH\scp.exe' `
  -i 'C:\Users\30738\.ssh\codexkey' `
  -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 `
  'C:\本地路径\inference_pipeline.cpp' `
  'albert@100.82.98.79:/home/albert/桌面/slave_robot_runtime_cpp./setc_linux/src/observer/inference_pipeline.cpp'
```

模型文件很大，工控机已经具有所需 B1S3 模型，除非用户再次明确要求，否则不要重复上传：

```text
/home/albert/桌面/slave_robot_runtime_cpp./setc_linux/models/
omnivggt_full_b1s3_406x252_bf16_unfrozen_torch270.pt
```

## 4. 修改前建立恢复备份

工控机上没有可依赖的 Git，因此每次覆盖源文件前先备份明确目标：

```bash
p="/home/albert/桌面/slave_robot_runtime_cpp./setc_linux"
stamp=$(date +%Y%m%d_%H%M%S)
mkdir -p "$p/backups"
cp -- "$p/src/observer/inference_pipeline.cpp" \
  "$p/backups/inference_pipeline.pre_change_${stamp}.cpp"
```

多个文件分别复制并写清备份原因。不要递归覆盖或删除整个项目，也不要修改 `setc_linux` 以外的目录。

## 5. 找到 GUI 的 DISPLAY 和 XAUTHORITY

工控机重启或 GUI 重开后，先重新读取环境，不要永久假定显示号：

```bash
gpid=$(pgrep -n slave_robot_gui || true)
if [ -n "$gpid" ]; then
  tr '\0' '\n' < "/proc/$gpid/environ" \
    | grep -E '^(DISPLAY|XAUTHORITY)='
fi
```

此前实际得到：

```text
DISPLAY=:1
XAUTHORITY=/run/user/1000/gdm/Xauthority
```

这些值在重启后可能变化，必须以当前 GUI 进程环境为准。后续截图和点击命令都要带上这两个环境变量。

## 6. GUI 启动与真实重建流程

正常流程是：

1. 确认 `slave_robot_gui` 已运行。
2. 在 GUI 上点击“启动海康相机”，等待三路海康画面都持续更新。
3. 点击“启动 C++ 三维重建”。只有点击该按钮后，`omnivggt_stream_server` 和点云客户端流程才会真正开始。
4. 检查日志区是否显示点云流已连接，并用 `ps` 再确认服务进程。
5. GUI 是常驻实时程序，不会自动结束。不要等待 GUI 退出，也不要因为它不退出就宣布任务完成。

不要把“手动从终端启动了 server”误认为完整 GUI 流程已启动。终端直启只适合单独诊断；最终验收必须通过 GUI 按钮走真实相机和 GUI 客户端链路。

## 7. 通过 XTest 自动点击 GUI

工控机没有安装 Python Xlib，但可以由 Python `ctypes` 直接调用系统 `libX11` 和 `libXtst`。

下面示例点击全局坐标 `(516, 940)`：

```bash
DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority python3 -c \
'from ctypes import *
x=CDLL("libX11.so.6")
t=CDLL("libXtst.so.6")
x.XOpenDisplay.restype=c_void_p
d=x.XOpenDisplay(None)
assert d, "cannot open X display"
t.XTestFakeMotionEvent.argtypes=[c_void_p,c_int,c_int,c_int,c_ulong]
t.XTestFakeButtonEvent.argtypes=[c_void_p,c_uint,c_int,c_ulong]
x.XFlush.argtypes=[c_void_p]
t.XTestFakeMotionEvent(d,-1,516,940,0)
t.XTestFakeButtonEvent(d,1,1,0)
t.XTestFakeButtonEvent(d,1,0,0)
x.XFlush(d)'
```

在此前的 2560×1440 GUI 布局中，曾实际使用：

- “启动 C++ 三维重建”附近：`(516, 940)`
- “切换视角”附近：`(516, 1037)`

坐标不是稳定 API。GUI 移动、缩放、重启或分辨率变化后，必须先抓取整屏截图重新确认按钮位置，禁止盲点。相机启动按钮没有保留下可靠固定坐标，应先看当前截图再点击。

切换视角按钮会循环不同预设。历史 GUI 源码检查得到的顺序是：斜视、顶视、侧视、正视；但 widget 当前状态会延续，因此不能只按点击次数推断当前视角，应查看实际点云画面。

## 8. 抓取真实 GUI 截图

工控机已有 Pillow，可用 `ImageGrab` 抓取当前桌面：

```bash
DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority python3 -c \
'from PIL import ImageGrab; ImageGrab.grab().save("/tmp/setc_gui_full.jpg", quality=94)'
```

然后用 SCP 下载：

```powershell
& 'C:\Windows\System32\OpenSSH\scp.exe' `
  -i 'C:\Users\30738\.ssh\codexkey' `
  -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 `
  'albert@100.82.98.79:/tmp/setc_gui_full.jpg' `
  'C:\Users\30738\Desktop\project\3dre\VGGT\OmniVGGT\tmp\setc_gui_full.jpg'
```

此前 GUI 布局中常用的大致裁剪区域：

- 中央 OpenGL 点云：`(600, 250, 1450, 950)`
- 右侧相机区域：`(1370, 150, 2560, 1250)`

裁剪坐标同样只供参考，应先保存整屏，再根据当前分辨率调整。不要只截顶视：至少连续抓取顶视、45 度斜视和侧视，以检查覆盖范围、机械臂高度和异常散点。

项目中还保留了辅助脚本：

```text
setc_linux/focus_gui_capture_sequence.py
setc_linux/capture_above_oblique_sequence.py
```

运行前先阅读脚本。`focus_gui_capture_sequence.py` 会发送 Alt+Tab，若当前窗口顺序改变，可能切到错误窗口，因此不能无条件执行。

## 9. 构建和基础测试

在工控机执行：

```bash
cd "/home/albert/桌面/slave_robot_runtime_cpp./setc_linux"
cmake --build build_live_observer --parallel 4
cd build_live_observer
ctest --output-on-failure
```

可执行文件通常位于：

```text
setc_linux/build_live_observer/bin/omnivggt_stream_server
```

编译通过和 `ctest` 通过只说明基础代码可运行，不能证明点云正确。最终必须重新点击 GUI 的 C++ 重建按钮并检查真实 OpenGL 画面。

## 10. 运行状态检查

```bash
ps -eo pid,lstart,args \
  | grep -E 'slave_robot_gui|omnivggt_stream_server' \
  | grep -v grep || true
```

若 GUI 日志持续显示“点云流连接失败”：

1. 先确认 server 是否真的存在，而不是连续重复点击。
2. 检查 server 是否启动后立即退出，并读取它自己的日志。
3. 检查 GUI 是否已启动相机，以及 C++ 重建按钮当前是“启动”还是“已在运行”。
4. 必要时只结束明确 PID 的旧 server，再通过 GUI 重启。不要按模糊进程名杀死 GUI 或其它用户程序。

## 11. 三相机输入核对

GUI 右侧通常有三路海康和一路内窥镜：

- 视角 1：海康 GigE
- 视角 2：海康 GigE
- 视角 3：海康 GigE
- 视角 4：内窥镜

重建只应使用前三路海康。一次抓取中，应同时保存：

- GUI 三路海康画面；
- 实际送入模型的三个输入槽；
- 当前 frame/group 序号；
- OpenGL 点云画面。

不能只看文件名判断相机顺序。当前 `live_frame_source.cpp` 仍通过 `source_seq % 3` 猜测槽位，相位在重启后可能变化。接手者应先实拍核对图像特征，再改成明确相机 ID 加时间戳的组帧。

## 12. 真实 GUI 验收方式

每次修改后必须完成以下循环，发现问题继续修，不能以日志正常提前结束：

1. 备份目标源文件。
2. 修改并上传，仅限 `setc_linux`。
3. 编译并运行基础测试。
4. 确认三台海康相机在 GUI 中实时更新。
5. 通过 GUI 按钮启动 C++ 三维重建。
6. 连续观察至少 30 秒。
7. 抓取相机画面和点云顶视。
8. 切到 45 度斜视和侧视，检查机械臂是否真正高于平面。
9. 检查三相机联合范围、机械臂下方补地面、颜色接缝、散点、跳变和场景更新延迟。
10. 任一项不符合目标就继续修改、编译和复查。

验收重点：

- 顶视联合范围不能仍是单一长方形。
- 机械臂下方在其它相机可见的地面必须被补齐。
- 机械臂不能只是平面贴图，侧视必须显示连续真实高度。
- 不得存在大片平面上方或下方异常点。
- 点云颜色和结构必须能对应三路真实图像。
- 连续帧不得整体跳变，新放入物体应及时显示。

## 13. 操作安全提醒

- 只修改 `setc_linux`；GUI 和其它项目目录只能运行或只读检查。
- 不使用模拟输入替代真实相机。
- 不等待常驻 GUI 自动结束。
- 不因编译、日志或点云文件正常就宣布完成。
- 不频繁计算 SHA-256。
- 不重复上传已有的大模型。
- 不执行递归删除、`git reset --hard` 或其它不可恢复操作。
- 自动点击前先截图确认坐标；结束旧服务时先解析准确 PID。
