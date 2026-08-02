@echo off
setlocal

rem One-click launcher for the optional Python OmniVGGT live replay viewer.
rem The launcher lives with the Python streaming module; it does not change
rem the normal project entry points.

set "ROOT=%~dp0.."
pushd "%ROOT%"

rem Reuse the existing GPU Python environment. This is runtime infrastructure,
rem not the location of the Python feature code.
set "PYTHON_EXE=%ROOT%\setc\venv_torch270_cu128\Scripts\python.exe"
set "IMAGE_DIR=%ROOT%\data2"
set "OUTPUT_DIR=%ROOT%\stream_omnivggt_outputs\data2_python_live_replay"

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Python environment not found:
    echo         %PYTHON_EXE%
    echo.
    pause
    set "EXIT_CODE=1"
    goto :finish
)

if not exist "%IMAGE_DIR%" (
    echo [ERROR] data2 directory not found:
    echo         %IMAGE_DIR%
    echo.
    pause
    set "EXIT_CODE=1"
    goto :finish
)

echo Starting OmniVGGT Python Live Replay...
echo Dataset: %IMAGE_DIR%
echo Close the viewer with q or Esc.
echo.

"%PYTHON_EXE%" -m stream_omnivggt.cli.run_data2_live_replay ^
    --image-dir "%IMAGE_DIR%" ^
    --output-dir "%OUTPUT_DIR%" ^
    --target-width 700 ^
    --target-size 700 ^
    --display-max-points 0 ^
    --device cuda ^
    --dtype bf16 ^
    --no-save-debug

set "EXIT_CODE=%ERRORLEVEL%"

:finish
popd
if not "%EXIT_CODE%"=="0" (
    echo.
    echo Live replay exited with code %EXIT_CODE%.
    pause
)
endlocal & exit /b %EXIT_CODE%
