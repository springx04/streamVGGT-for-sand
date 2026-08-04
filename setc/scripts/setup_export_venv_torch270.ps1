param(
    [string]$VenvPath = "setc\venv_torch270_cu128",
    [string]$TorchWheel = "C:\Dev\downloads\torch-2.7.0+cu128-cp311-cp311-win_amd64.whl"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $VenvPath)) {
    python -m venv $VenvPath
}

$python = Join-Path $VenvPath "Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Venv python not found: $python"
}

& $python -m pip install --upgrade pip

& $python -m pip install safetensors huggingface_hub einops numpy scipy opencv-python evo sympy jinja2 networkx

if (Test-Path -LiteralPath $TorchWheel) {
    & $python -m pip install $TorchWheel
} else {
    & $python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
}

& $python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
