param(
    [string]$BindHost = "",
    [int]$Port = 0,
    [string]$PythonVersion = "",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"

$ProjectRoot = $PSScriptRoot
$EnvName = "AnatoMask"
$EnvDir = Join-Path $ProjectRoot ".uv-envs\$EnvName"
$EnvPython = Join-Path $EnvDir "Scripts\python.exe"
$DefaultHost = if ($env:ANATOMASK_HOST) { $env:ANATOMASK_HOST } else { "0.0.0.0" }
$DefaultPort = if ($env:ANATOMASK_PORT) { [int]$env:ANATOMASK_PORT } else { 7860 }
$ResolvedHost = if ($BindHost) { $BindHost } else { $DefaultHost }
$ResolvedPort = if ($Port -gt 0) { $Port } else { $DefaultPort }
$ResolvedPythonVersion = if ($PythonVersion) { $PythonVersion } elseif ($env:ANATOMASK_PYTHON_VERSION) { $env:ANATOMASK_PYTHON_VERSION } else { "3.10" }
$UvHome = Join-Path $HOME ".local\bin"
$DetectCudaScript = Join-Path $ProjectRoot "scripts\detect_cuda.ps1"

Set-Location $ProjectRoot

function Write-Log {
    param([string]$Message)
    Write-Host "[AnatoMask] $Message"
}

function Fail {
    param([string]$Message)
    throw "[AnatoMask] ERROR: $Message"
}

function Get-FriendlyUrl {
    param(
        [string]$BindHost,
        [int]$Port
    )

    $DisplayHost = if (-not $BindHost -or $BindHost -eq "0.0.0.0" -or $BindHost -eq "::" -or $BindHost -eq "[::]") {
        "127.0.0.1"
    } else {
        $BindHost
    }

    return "http://${DisplayHost}:$Port"
}

function Get-UvCommandPath {
    $candidates = @(
        (Get-Command uv -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source -ErrorAction SilentlyContinue),
        (Join-Path $UvHome "uv.exe"),
        (Join-Path $UvHome "uv")
    ) | Where-Object { $_ }

    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return (Resolve-Path $candidate).Path
        }
    }

    return $null
}

function Install-Uv {
    Write-Log "uv was not found. Installing it now."
    powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
}

function Ensure-Uv {
    $script:UvBin = Get-UvCommandPath
    if (-not $script:UvBin) {
        Install-Uv
        $script:UvBin = Get-UvCommandPath
    }

    if (-not $script:UvBin) {
        Fail "uv installation completed but uv.exe was still not found."
    }

    if (-not ($env:Path -split ";" | Where-Object { $_ -eq $UvHome })) {
        $env:Path = "$UvHome;$env:Path"
    }

    Write-Log "Using uv: $script:UvBin"
}

function Get-CudaInfo {
    if (-not (Test-Path $DetectCudaScript)) {
        Fail "Missing CUDA detection script: $DetectCudaScript"
    }

    try {
        $channel = (& powershell.exe -NoProfile -ExecutionPolicy Bypass -File $DetectCudaScript -Channel).Trim()
        $driverMajor = (& powershell.exe -NoProfile -ExecutionPolicy Bypass -File $DetectCudaScript -DriverMajor).Trim()
    } catch {
        & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $DetectCudaScript -Summary
        Fail "This machine does not meet the GPU runtime requirements. Install a supported NVIDIA driver first. CPU is not supported."
    }

    if (-not $channel) {
        & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $DetectCudaScript -Summary
        Fail "Failed to determine the supported PyTorch CUDA channel."
    }

    return @{
        Channel = $channel
        DriverMajor = $driverMajor
    }
}

function Ensure-Environment {
    if (Test-Path $EnvPython) {
        Write-Log "Found existing uv environment: $EnvDir"
        return
    }

    Write-Log "Creating uv environment $EnvName -> $EnvDir"
    & $script:UvBin python install $ResolvedPythonVersion
    & $script:UvBin venv --python $ResolvedPythonVersion $EnvDir
}

function Invoke-RuntimeCheck {
    if (-not (Test-Path $EnvPython)) {
        return @{
            ExitCode = 1
            Output = "Environment python executable was not found."
        }
    }

    $code = @'
import importlib
import os
import sys
import traceback

project_root = r"__PROJECT_ROOT__"
if project_root and project_root not in sys.path:
    sys.path.insert(0, project_root)

errors = []
modules = [
    "gradio",
    "monai",
    "nibabel",
    "SimpleITK",
    "tensorboardX",
    "PIL",
    "scipy",
    "timm",
    "torch",
    "launcher.webui",
]
for name in modules:
    try:
        importlib.import_module(name)
    except Exception as exc:
        errors.append(f"{name}: {type(exc).__name__}: {exc}")

if not errors:
    import torch
    if not torch.cuda.is_available():
        errors.append("torch.cuda.is_available(): False")
    else:
        print(f"torch={torch.__version__}")
        print(f"torch_cuda={torch.version.cuda}")
        print(f"gpu_count={torch.cuda.device_count()}")

if errors:
    print("RUNTIME_CHECK_FAILED")
    for item in errors:
        print(item)
    raise SystemExit(1)
print("RUNTIME_CHECK_OK")
'@
    $code = $code.Replace("__PROJECT_ROOT__", $ProjectRoot.Replace("\", "\\"))

    $tempScript = [System.IO.Path]::Combine($env:TEMP, "anatomask_runtime_check_$PID.py")
    $stdoutPath = [System.IO.Path]::GetTempFileName()
    $stderrPath = [System.IO.Path]::GetTempFileName()

    try {
        Set-Content -LiteralPath $tempScript -Value $code -Encoding ASCII
        $proc = Start-Process -FilePath $EnvPython `
            -ArgumentList @($tempScript) `
            -RedirectStandardOutput $stdoutPath `
            -RedirectStandardError $stderrPath `
            -NoNewWindow `
            -Wait `
            -PassThru
        $stdoutText = ""
        $stderrText = ""
        if (Test-Path $stdoutPath) {
            $stdoutText = (Get-Content -LiteralPath $stdoutPath -Raw -ErrorAction SilentlyContinue)
        }
        if (Test-Path $stderrPath) {
            $stderrText = (Get-Content -LiteralPath $stderrPath -Raw -ErrorAction SilentlyContinue)
        }
        return @{
            ExitCode = $proc.ExitCode
            Output = (($stdoutText, $stderrText) | Where-Object { $_ }) -join "`n"
        }
    } finally {
        foreach ($path in @($tempScript, $stdoutPath, $stderrPath)) {
            if ($path -and (Test-Path $path)) {
                Remove-Item -LiteralPath $path -Force -ErrorAction SilentlyContinue
            }
        }
    }
}

function Test-RuntimeReady {
    $result = Invoke-RuntimeCheck
    $script:LastRuntimeCheckOutput = $result.Output
    return ($result.ExitCode -eq 0)
}

function Install-BasePackages {
    Write-Log "Installing base dependencies from requirements.web.txt"
    & $script:UvBin pip install --python $EnvPython --link-mode copy -r (Join-Path $ProjectRoot "requirements.web.txt")
}

function Install-GpuPyTorch {
    param([string]$TorchChannel)
    Write-Log "Installing GPU PyTorch build ($TorchChannel)"
    & $script:UvBin pip install --python $EnvPython --link-mode copy --upgrade --index-url "https://download.pytorch.org/whl/$TorchChannel" torch torchvision torchaudio
}

function Recreate-Environment {
    if (Test-Path $EnvDir) {
        Write-Log "Removing broken environment: $EnvDir"
        Remove-Item -LiteralPath $EnvDir -Recurse -Force
    }
    Ensure-Environment
}

function Ensure-Runtime {
    param([string]$TorchChannel)

    if (Test-RuntimeReady) {
        Write-Log "AnatoMask environment is ready. Launching the Web UI."
        return
    }

    if ($script:LastRuntimeCheckOutput) {
        Write-Log "Runtime check reported:"
        Write-Host $script:LastRuntimeCheckOutput
    }

    Write-Log "Runtime check failed or dependencies are incomplete. Installing required packages."
    Install-GpuPyTorch -TorchChannel $TorchChannel
    Install-BasePackages

    if (-not (Test-RuntimeReady)) {
        if ($script:LastRuntimeCheckOutput) {
            Write-Log "Runtime check still failed after install:"
            Write-Host $script:LastRuntimeCheckOutput
        }

        Write-Log "Recreating the uv environment and reinstalling once."
        Recreate-Environment
        Install-GpuPyTorch -TorchChannel $TorchChannel
        Install-BasePackages
        if (Test-RuntimeReady) {
            Write-Log "Environment rebuild succeeded."
            return
        }
        if ($script:LastRuntimeCheckOutput) {
            Write-Log "Runtime check after rebuild still failed:"
            Write-Host $script:LastRuntimeCheckOutput
        }

        Fail "Environment setup finished, but runtime validation still failed. Check the diagnostics above."
    }
}

function Launch-WebUi {
    $argsToUse = @()
    if ($ExtraArgs -and $ExtraArgs.Count -gt 0) {
        $argsToUse = $ExtraArgs
    } else {
        $argsToUse = @("--host", $ResolvedHost, "--port", "$ResolvedPort")
    }

    $FriendlyUrl = Get-FriendlyUrl -BindHost $ResolvedHost -Port $ResolvedPort
    Write-Log "Starting Web UI. Open $FriendlyUrl if the browser does not appear automatically."
    & $EnvPython (Join-Path $ProjectRoot "launch_webui.py") @argsToUse
}

Ensure-Uv
$cudaInfo = Get-CudaInfo
Write-Log "Detected NVIDIA driver major version: $($cudaInfo.DriverMajor)"
Write-Log "Selected PyTorch channel: $($cudaInfo.Channel)"
Ensure-Environment
Ensure-Runtime -TorchChannel $cudaInfo.Channel
Launch-WebUi
