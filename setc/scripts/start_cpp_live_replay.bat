@echo off
setlocal

rem One-click C++ equivalent of stream_omnivggt\start_python_live_replay.bat.
rem Keep the input/output/device/dtype controls in sync with the Python
rem launcher; the optional observer remains outside the normal project run.

set "REPO=%~dp0..\.."
for %%I in ("%REPO%") do set "REPO=%%~fI"
if not defined LIBTORCH set "LIBTORCH=C:\Dev\libtorch\2.7.0-cu128"
if not defined OpenCV_DIR set "OpenCV_DIR=C:\Dev\opencv\4.10.0\build"
if not defined CUDA_PATH set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "PATH=%LIBTORCH%\lib;%OpenCV_DIR%\x64\vc16\bin;%OpenCV_DIR%\x64\vc15\bin;%CUDA_PATH%\bin;%PATH%"

set "SERVER=%REPO%\setc\build_live_observer\Release\omnivggt_stream_server.exe"
set "VIEWER=%REPO%\setc\build_live_observer\Release\omnivggt_live_viewer.exe"
set "MODEL=%REPO%\setc\artifacts\omnivggt_observer_s1_700x434_bf16_unfrozen_torch270.pt"
rem Use the depth-only pair graph.  The fixed graph is loaded once; each
rem Python-sized ROI is kept aspect-preserving inside its 700x700 border.
set "PAIR_MODEL=%REPO%\setc\artifacts\omnivggt_observer_s2_700x700_bf16_unfrozen_torch270.pt"
set "IMAGE_DIR=%REPO%\data2"
set "OUTPUT_DIR=%REPO%\stream_omnivggt_outputs\data2_python_live_replay"
set "TARGET_WIDTH=700"
set "TARGET_SIZE=700"
set "CANVAS_WIDTH=770"
set "CANVAS_HEIGHT=630"
set "FIRST_MODEL_WIDTH=700"
set "FIRST_MODEL_HEIGHT=434"
rem Offline replay must retain every frame while the first CUDA model call warms up.
rem The normal external stream keeps the latest-frame policy; this launcher is
rem deliberately lossless so its frame history matches Python live replay.
set "QUEUE_CAPACITY=1024"
set "PORT=37651"

rem Always start a fresh external observer.  Without this cleanup, a second
rem click can connect to the old server still listening on the fixed port and
rem the window then shows the previous run's snapshot/state.
taskkill /F /IM omnivggt_live_viewer.exe >nul 2>&1
taskkill /F /IM omnivggt_stream_server.exe >nul 2>&1
ping 127.0.0.1 -n 2 >nul

if not exist "%SERVER%" (
  echo [ERROR] C++ observer server not found:
  echo         %SERVER%
  echo Build with setc\scripts\build_windows_live_observer.bat
  pause
  exit /b 1
)
if not exist "%VIEWER%" (
  echo [ERROR] C++ live viewer not found:
  echo         %VIEWER%
  echo Build with setc\scripts\build_windows_live_observer.bat
  pause
  exit /b 1
)
if not exist "%MODEL%" (
  echo [ERROR] Observer depth TorchScript model not found:
  echo         %MODEL%
  echo Export it with setc\scripts\export_torchscript.py using --no-freeze.
  pause
  exit /b 1
)
if not exist "%PAIR_MODEL%" (
  echo [ERROR] Observer two-frame depth TorchScript model not found:
  echo         %PAIR_MODEL%
  echo Export it with --num-images 2 and --no-freeze.
  pause
  exit /b 1
)
if not exist "%IMAGE_DIR%" (
  echo [ERROR] data2 directory not found: %IMAGE_DIR%
  pause
  exit /b 1
)

echo Starting C++ OmniVGGT live replay...
echo Dataset: %IMAGE_DIR%
echo Output:  %OUTPUT_DIR%
echo Target:  %TARGET_WIDTH%x%TARGET_SIZE%  device=cuda  dtype=bf16  display_max_points=0
echo Queue:   %QUEUE_CAPACITY%  (lossless offline replay)
echo Canvas:  %CANVAS_WIDTH%x%CANVAS_HEIGHT%  first_model=%FIRST_MODEL_WIDTH%x%FIRST_MODEL_HEIGHT%
echo Pair:    fixed 700x700 observer graph, aspect-preserving ROI letterbox
echo          (loaded and warmed once; avoids per-ROI TorchScript reloads)
echo Viewer:  %VIEWER%
echo Close the viewer with q or Esc. The server stays independent until stopped.
echo.

start "OmniVGGT C++ Stream Server" /b "%SERVER%" ^
  --model "%MODEL%" ^
  --model-pair "%PAIR_MODEL%" ^
  --pair-letterbox ^
  --image-dir "%IMAGE_DIR%" ^
  --output-dir "%OUTPUT_DIR%" ^
  --target-size %TARGET_SIZE% --target-width %TARGET_WIDTH% ^
  --canvas-width %CANVAS_WIDTH% --canvas-height %CANVAS_HEIGHT% ^
  --first-model-width %FIRST_MODEL_WIDTH% --first-model-height %FIRST_MODEL_HEIGHT% ^
  --device cuda --dtype bf16 --min_conf 0.0 ^
  --queue-capacity %QUEUE_CAPACITY% ^
  --port %PORT% --no-save-debug

rem The fixed TorchScript model is loaded before the server can accept the
rem viewer connection.  Allow that one-time CUDA load to finish first.
rem ``timeout`` reads console input and exits immediately when this launcher
rem is started from an IDE/automation host with redirected stdin.  ping is a
rem stdin-independent delay and keeps the one-click launcher reliable there.
ping 127.0.0.1 -n 7 >nul
"%VIEWER%" --host 127.0.0.1 --port %PORT% --display-max-points 0

endlocal
