param(
    [string]$DevRoot = "C:\Dev",
    [string]$LibtorchVersion = "2.7.0",
    [string]$CudaTag = "cu128",
    [string]$OpenCvVersion = "4.10.0"
)

$ErrorActionPreference = "Stop"

function New-DirectoryIfMissing {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        New-Item -ItemType Directory -Path $Path | Out-Null
    }
}

function Get-UniqueDirectory {
    param([string]$BasePath)
    if (-not (Test-Path -LiteralPath $BasePath)) {
        return $BasePath
    }
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    return "$BasePath-$stamp"
}

function Invoke-Download {
    param(
        [string]$Url,
        [string]$OutputPath
    )
    $resume = $false
    if (Test-Path -LiteralPath $OutputPath) {
        $existing = Get-Item -LiteralPath $OutputPath
        if ($existing.Length -gt 1024) {
            Write-Host "Resuming existing download: $OutputPath"
            $resume = $true
        } else {
            Write-Host "Existing download is too small and will be overwritten: $OutputPath"
        }
    }
    Write-Host "Downloading:"
    Write-Host "  $Url"
    Write-Host "to:"
    Write-Host "  $OutputPath"
    if ($resume) {
        & curl.exe -L --fail --retry 5 --retry-delay 5 -C - -o $OutputPath $Url
    } else {
        & curl.exe -L --fail --retry 5 --retry-delay 5 -o $OutputPath $Url
    }
    if ($LASTEXITCODE -ne 0) {
        throw "curl failed with exit code $LASTEXITCODE"
    }
}

$downloadDir = Join-Path $DevRoot "downloads"
$libtorchParent = Join-Path $DevRoot "libtorch"
$opencvParent = Join-Path $DevRoot "opencv"

New-DirectoryIfMissing $DevRoot
New-DirectoryIfMissing $downloadDir
New-DirectoryIfMissing $libtorchParent
New-DirectoryIfMissing $opencvParent

$libtorchInstall = Join-Path $libtorchParent "$LibtorchVersion-$CudaTag"
$libtorchConfig = Join-Path $libtorchInstall "share\cmake\Torch\TorchConfig.cmake"
$libtorchZipName = "libtorch-win-shared-with-deps-$LibtorchVersion+$CudaTag.zip"
$libtorchZip = Join-Path $downloadDir $libtorchZipName
$libtorchUrl = "https://download-r2.pytorch.org/libtorch/$CudaTag/libtorch-win-shared-with-deps-$LibtorchVersion%2B$CudaTag.zip"

if (Test-Path -LiteralPath $libtorchConfig) {
    Write-Host "LibTorch already installed: $libtorchInstall"
} else {
    Invoke-Download -Url $libtorchUrl -OutputPath $libtorchZip
    $extractRoot = Get-UniqueDirectory (Join-Path $libtorchParent "extract-$LibtorchVersion-$CudaTag")
    New-DirectoryIfMissing $extractRoot
    Write-Host "Extracting LibTorch to $extractRoot"
    Expand-Archive -LiteralPath $libtorchZip -DestinationPath $extractRoot
    $nested = Join-Path $extractRoot "libtorch"
    if (-not (Test-Path -LiteralPath $nested)) {
        throw "Expected extracted LibTorch directory was not found: $nested"
    }
    if (Test-Path -LiteralPath $libtorchInstall) {
        throw "Target already exists but TorchConfig.cmake was not found: $libtorchInstall"
    }
    Move-Item -LiteralPath $nested -Destination $libtorchInstall
    Write-Host "LibTorch installed: $libtorchInstall"
}

$opencvInstall = Join-Path $opencvParent $OpenCvVersion
$opencvConfig = Join-Path $opencvInstall "build\OpenCVConfig.cmake"
$opencvExe = Join-Path $downloadDir "opencv-$OpenCvVersion-windows.exe"
$opencvUrl = "https://github.com/opencv/opencv/releases/download/$OpenCvVersion/opencv-$OpenCvVersion-windows.exe"

if (Test-Path -LiteralPath $opencvConfig) {
    Write-Host "OpenCV already installed: $opencvInstall"
} else {
    Invoke-Download -Url $opencvUrl -OutputPath $opencvExe
    $extractRoot = Get-UniqueDirectory (Join-Path $opencvParent "extract-$OpenCvVersion")
    New-DirectoryIfMissing $extractRoot
    Write-Host "Extracting OpenCV to $extractRoot"
    & $opencvExe "-o$extractRoot" -y
    if ($LASTEXITCODE -ne 0) {
        throw "OpenCV extractor failed with exit code $LASTEXITCODE"
    }

    $candidate = Get-ChildItem -LiteralPath $extractRoot -Recurse -Filter OpenCVConfig.cmake |
        Select-Object -First 1
    if ($null -eq $candidate) {
        throw "OpenCVConfig.cmake was not found under $extractRoot"
    }
    $buildDir = Split-Path -Parent $candidate.FullName
    $rootCandidate = Split-Path -Parent $buildDir
    if (Test-Path -LiteralPath $opencvInstall) {
        throw "Target already exists but OpenCVConfig.cmake was not found: $opencvInstall"
    }
    Move-Item -LiteralPath $rootCandidate -Destination $opencvInstall
    Write-Host "OpenCV installed: $opencvInstall"
}

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$envFile = Join-Path $repoRoot "setc\local_env.ps1"
$envText = @"
`$env:LIBTORCH = "$libtorchInstall"
`$env:OpenCV_DIR = "$opencvInstall\build"
`$env:Path = "`$env:LIBTORCH\lib;`$env:OpenCV_DIR\x64\vc16\bin;`$env:OpenCV_DIR\x64\vc15\bin;`$env:Path"
"@
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($envFile, $envText, $utf8NoBom)

Write-Host ""
Write-Host "Done."
Write-Host "LibTorch:   $libtorchInstall"
Write-Host "OpenCV_DIR: $opencvInstall\build"
Write-Host "Env file:   $envFile"
Write-Host ""
Write-Host "Next in PowerShell:"
Write-Host "  . .\setc\local_env.ps1"
Write-Host "  cmd /c setc\scripts\build_windows_gpu.bat"
