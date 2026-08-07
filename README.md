# OmniVGGT Main Workflows

This repository currently has two production workflows for live replay on Windows:

1. Python main workflow (one-click): `stream_omnivggt\start_python_live_replay.bat`
2. C++ main workflow (one-click): `setc\scripts\start_cpp_live_replay.bat`

Both workflows use the `data2` image directory by default and write outputs to:

- `stream_omnivggt_outputs\data2_python_live_replay`

## 1) Python Main Workflow

### 1.1 Prepare runtime environment

From repository root:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File setc\scripts\setup_export_venv_torch270.ps1
```

This creates/reuses:

- `setc\venv_torch270_cu128`

### 1.2 One-click start

```powershell
cmd /c stream_omnivggt\start_python_live_replay.bat
```

The launcher runs:

- `python -m stream_omnivggt.cli.run_data2_live_replay`
- `--image-dir data2`
- `--output-dir stream_omnivggt_outputs\data2_python_live_replay`
- `--target-width 700 --target-size 700 --device cuda --dtype bf16`
- `--display-max-points 0 --no-save-debug`

### 1.3 Expected outputs

- `stream_omnivggt_outputs\data2_python_live_replay\pointcloud_final.ply`
- `stream_omnivggt_outputs\data2_python_live_replay\timings.json`
- `stream_omnivggt_outputs\data2_python_live_replay\timings.md`

## 2) C++ Main Workflow

### 2.1 Install C++ dependencies (first machine setup)

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File setc\scripts\install_windows_gpu_deps.ps1
```

Default install targets used by current scripts:

- `LIBTORCH=C:\Dev\libtorch\2.7.0-cu128`
- `OpenCV_DIR=C:\Dev\opencv\4.10.0\build`
- `CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8`

If your local paths differ, set environment variables before build/run.

### 2.2 Build live observer binaries

```powershell
cmd /c setc\scripts\build_windows_live_observer.bat
```

Expected binaries:

- `setc\build_live_observer\Release\omnivggt_stream_server.exe`
- `setc\build_live_observer\Release\omnivggt_live_viewer.exe`

### 2.3 Build required TorchScript artifacts (if missing)

If these files already exist, skip this section:

- `setc\artifacts\omnivggt_observer_s1_700x434_bf16_unfrozen_torch270.pt`
- `setc\artifacts\omnivggt_observer_s2_700x700_bf16_unfrozen_torch270.pt`

Export commands (from repository root):

```powershell
setc\venv_torch270_cu128\Scripts\python.exe setc\scripts\export_torchscript.py --output setc\artifacts\omnivggt_observer_s1_700x434_bf16_unfrozen_torch270.pt --num-images 1 --height 434 --width 700 --dtype bfloat16 --autocast-dtype bfloat16 --observer-depth-only --no-freeze

setc\venv_torch270_cu128\Scripts\python.exe setc\scripts\export_torchscript.py --output setc\artifacts\omnivggt_observer_s2_700x700_bf16_unfrozen_torch270.pt --num-images 2 --height 700 --width 700 --dtype bfloat16 --autocast-dtype bfloat16 --observer-depth-only --no-freeze
```

### 2.4 One-click start

```powershell
cmd /c setc\scripts\start_cpp_live_replay.bat
```

The launcher starts `omnivggt_stream_server.exe` in background, then launches `omnivggt_live_viewer.exe`.

## Notes

- Python and C++ launchers are aligned to the same replay target (`data2`, `700x700`, `cuda`, `bf16`).
- The C++ launcher always kills old viewer/server processes first, then starts a fresh run on port `37651`.
- For C++ offline replay fidelity, queue capacity is fixed to `1024` in the launcher.

