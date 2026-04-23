param(
    [switch]$Channel,
    [switch]$DriverMajor,
    [switch]$Quiet,
    [switch]$Summary
)

$modeFlags = @($Channel, $DriverMajor, $Quiet, $Summary) | Where-Object { $_ }
if ($modeFlags.Count -gt 1) {
    throw "Use only one of -Channel, -DriverMajor, -Quiet, or -Summary."
}

if (-not $Channel -and -not $DriverMajor -and -not $Quiet -and -not $Summary) {
    $Summary = $true
}

function Write-Failure {
    param([string]$Message)
    if (-not $Quiet -and -not $Channel -and -not $DriverMajor) {
        Write-Error $Message
    }
    exit 1
}

function Get-NvidiaSmiCommand {
    foreach ($candidate in @("nvidia-smi.exe", "nvidia-smi")) {
        $command = Get-Command $candidate -ErrorAction SilentlyContinue
        if ($command) {
            return $command.Source
        }
    }
    return $null
}

$nvidiaSmi = Get-NvidiaSmiCommand
if (-not $nvidiaSmi) {
    Write-Failure "[AnatoMask] nvidia-smi was not found. This build requires an NVIDIA GPU and CUDA-capable driver."
}

$driverVersion = & $nvidiaSmi --query-gpu=driver_version --format=csv,noheader 2>$null | Select-Object -First 1
if (-not $driverVersion) {
    Write-Failure "[AnatoMask] Failed to read the NVIDIA driver version from nvidia-smi."
}

$driverVersion = $driverVersion.Trim()
$driverMajorText = $driverVersion.Split(".", 2)[0]
$driverMajorValue = 0
if (-not [int]::TryParse($driverMajorText, [ref]$driverMajorValue)) {
    Write-Failure "[AnatoMask] Failed to parse the NVIDIA driver version: $driverVersion"
}

$torchChannel = $null
if ($driverMajorValue -ge 580) {
    $torchChannel = "cu130"
} elseif ($driverMajorValue -ge 525) {
    $torchChannel = "cu126"
}

if (-not $torchChannel) {
    Write-Failure "[AnatoMask] NVIDIA driver version is too old: $driverVersion. This launcher only supports PyTorch CUDA 12.6 or 13.0 channels."
}

if ($Channel) {
    Write-Output $torchChannel
} elseif ($DriverMajor) {
    Write-Output $driverMajorValue
} elseif ($Summary) {
    Write-Output "[AnatoMask] Detected NVIDIA driver version: $driverVersion"
    Write-Output "[AnatoMask] Recommended PyTorch CUDA channel: $torchChannel"
    Write-Output "[AnatoMask] Recommended install command: python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/$torchChannel"
}
