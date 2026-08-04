@echo off
setlocal

set "REPO=%~dp0..\.."
for %%I in ("%REPO%") do set "REPO=%%~fI"

set "VSDEVCMD=C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\Common7\Tools\VsDevCmd.bat"
set "VSCMAKE=C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"

if defined LIBTORCH goto have_libtorch
set "LIBTORCH=C:\Dev\libtorch\2.7.0-cu128"
:have_libtorch

if defined OpenCV_DIR goto have_opencv
set "OpenCV_DIR=C:\Dev\opencv\4.10.0\build"
:have_opencv

if exist "%VSDEVCMD%" goto have_vsdevcmd
echo Missing VsDevCmd: %VSDEVCMD%
exit /b 1
:have_vsdevcmd

call "%VSDEVCMD%" -arch=x64 -host_arch=x64 >nul
set "PATH=%LIBTORCH%\lib;%OpenCV_DIR%\x64\vc16\bin;%OpenCV_DIR%\x64\vc15\bin;%PATH%"

if exist "%VSCMAKE%" goto use_vs_cmake
set "CMAKE=cmake"
goto configure

:use_vs_cmake
set "CMAKE=%VSCMAKE%"

:configure
"%CMAKE%" -S "%REPO%\setc" -B "%REPO%\setc\build_manual_libtorch" -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DLIBTORCH_ROOT="%LIBTORCH%" -DOpenCV_DIR="%OpenCV_DIR%"
if errorlevel 1 exit /b %ERRORLEVEL%

"%CMAKE%" --build "%REPO%\setc\build_manual_libtorch" --config Release
if errorlevel 1 exit /b %ERRORLEVEL%

echo Built: %REPO%\setc\build_manual_libtorch\omnivggt_edge.exe
echo Built: %REPO%\setc\build_manual_libtorch\omnivggt_stream.exe
