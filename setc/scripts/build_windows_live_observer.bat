@echo off
setlocal

set "REPO=%~dp0..\.."
for %%I in ("%REPO%") do set "REPO=%%~fI"

set "VSCMAKE=C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
if not defined LIBTORCH set "LIBTORCH=C:\Dev\libtorch\2.7.0-cu128"
if not defined OpenCV_DIR set "OpenCV_DIR=C:\Dev\opencv\4.10.0\build"

rem The Codex desktop shell can provide both Path and PATH.  MSBuild treats
rem those as duplicate keys, so normalize the child environment first.
set "PATH="
set "Path="
set "PATH=%LIBTORCH%\lib;%OpenCV_DIR%\x64\vc16\bin;%OpenCV_DIR%\x64\vc15\bin;%SystemRoot%\System32;%SystemRoot%"
if exist "%VSCMAKE%" goto use_vs_cmake
set "CMAKE=cmake"
goto configure
:use_vs_cmake
set "CMAKE=%VSCMAKE%"
:configure

"%CMAKE%" -S "%REPO%\setc" -B "%REPO%\setc\build_live_observer" -G "Visual Studio 16 2019" -A x64 ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DLIBTORCH_ROOT="%LIBTORCH%" ^
  -DOpenCV_DIR="%OpenCV_DIR%" ^
  -DOMNIVGGT_ENABLE_LIVE_OBSERVER=ON
if errorlevel 1 exit /b %ERRORLEVEL%

"%CMAKE%" --build "%REPO%\setc\build_live_observer" --config Release
if errorlevel 1 exit /b %ERRORLEVEL%

echo Built independent observer targets under %REPO%\setc\build_live_observer
