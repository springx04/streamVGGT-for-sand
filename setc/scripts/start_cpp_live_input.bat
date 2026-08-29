@echo off
setlocal

rem Windows equivalent of setc_linux/scripts/start_cpp_live_input.sh.
rem The input directory is expected to already receive image files from the
rem upstream producer; camera acquisition is intentionally outside this launcher.

set "REPO=%~dp0..\.."
for %%I in ("%REPO%") do set "REPO=%%~fI"
if not defined LIBTORCH set "LIBTORCH=C:\Dev\libtorch\2.7.0-cu128"
if not defined OpenCV_DIR set "OpenCV_DIR=C:\Dev\opencv\4.10.0\build"
if not defined CUDA_PATH set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "PATH=%LIBTORCH%\lib;%OpenCV_DIR%\x64\vc16\bin;%OpenCV_DIR%\x64\vc15\bin;%CUDA_PATH%\bin;%PATH%"

if not defined SERVER set "SERVER=%REPO%\setc\build_live_observer\Release\omnivggt_stream_server.exe"
if not defined IMAGE_DIR set "IMAGE_DIR=%REPO%\data2"
if not defined OUTPUT_DIR set "OUTPUT_DIR=%REPO%\stream_omnivggt_outputs\data2_cpp_live_input"
if not defined MODEL set "MODEL=%REPO%\setc\artifacts\omnivggt_observer_s1_700x434_bf16_unfrozen_torch270.pt"
if not defined PAIR_MODEL set "PAIR_MODEL=%REPO%\setc\artifacts\omnivggt_observer_s2_700x700_bf16_unfrozen_torch270.pt"
if not defined MAX_INFLIGHT_GROUPS set "MAX_INFLIGHT_GROUPS=3"

if not "%MAX_INFLIGHT_GROUPS%"=="3" (
  echo [ERROR] MAX_INFLIGHT_GROUPS must be 3.
  exit /b 1
)
if not defined HISTORY_KEEP_GROUPS (
  echo [ERROR] HISTORY_KEEP_GROUPS is required.
  exit /b 1
)

if not exist "%SERVER%" (
  echo [ERROR] C++ observer server not found:
  echo         %SERVER%
  echo Build with setc\scripts\build_windows_live_observer.bat
  exit /b 1
)
if not exist "%MODEL%" (
  echo [ERROR] Observer depth TorchScript model not found:
  echo         %MODEL%
  exit /b 1
)
if not exist "%PAIR_MODEL%" (
  echo [ERROR] Observer two-frame depth TorchScript model not found:
  echo         %PAIR_MODEL%
  exit /b 1
)
if not exist "%IMAGE_DIR%" (
  echo [ERROR] Image directory not found:
  echo         %IMAGE_DIR%
  exit /b 1
)

echo Starting C++ OmniVGGT live input...
echo Image dir:        %IMAGE_DIR%
echo Output dir:       %OUTPUT_DIR%
echo Max inflight:     %MAX_INFLIGHT_GROUPS% (fixed C++ contract)
echo History keep:     %HISTORY_KEEP_GROUPS%
echo Input group:      size=3 stride=1 anchor=1
echo Models:           S1=%MODEL%
echo                   S2=%PAIR_MODEL%
echo Device/dtype:     cuda/bf16
echo.

"%SERVER%" ^
  --image-dir "%IMAGE_DIR%" ^
  --output-dir "%OUTPUT_DIR%" ^
  --model "%MODEL%" ^
  --model-pair "%PAIR_MODEL%" ^
  --pair-letterbox ^
  --input-group-size 3 ^
  --input-group-stride 1 ^
  --group-anchor-index 1 ^
  --history-keep-groups %HISTORY_KEEP_GROUPS% ^
  --device cuda ^
  --dtype bf16

endlocal
