# MTMCT-Vision Environment Setup Script
# Auto-detects NVIDIA GPU (CUDA 12.1) or CPU, creates .venv and installs dependencies.
param(
    [ValidateSet('Auto','CPU','CUDA121')][string]$Backend = 'Auto'
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location -LiteralPath $repoRoot

Write-Host "==========================================================================" -ForegroundColor Cyan
Write-Host " MTMCT-Vision | Automated Environment Setup (Windows 10/11 x64)" -ForegroundColor Cyan
Write-Host "==========================================================================" -ForegroundColor Cyan

# 1. Check Python & Create .venv
$pythonExe = Join-Path $repoRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) {
    Write-Host "[1/4] Creating virtual environment (.venv)..." -ForegroundColor Yellow
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        throw "Error: Python not found in PATH. Please install Python 3.10/3.11 and check 'Add python to PATH'."
    }
} else {
    Write-Host "[1/4] Virtual environment (.venv) already exists." -ForegroundColor Green
}

# Upgrade pip safely
& $pythonExe -m pip install --upgrade pip

# 2. Detect GPU / CPU
if ($Backend -eq 'Auto') {
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        $Backend = 'CUDA121'
        Write-Host "[2/4] NVIDIA GPU detected -> Selecting PyTorch CUDA 12.1" -ForegroundColor Green
    } else {
        $Backend = 'CPU'
        Write-Host "[2/4] No NVIDIA GPU detected -> Selecting PyTorch CPU" -ForegroundColor Yellow
    }
}

# 3. Install PyTorch
Write-Host "[3/4] Installing PyTorch ($Backend)..." -ForegroundColor Yellow
if ($Backend -eq 'CPU') {
    $torchIndex = 'https://download.pytorch.org/whl/cpu'
} else {
    $torchIndex = 'https://download.pytorch.org/whl/cu121'
}
& $pythonExe -m pip install torch torchvision --index-url $torchIndex
if ($LASTEXITCODE -ne 0) {
    throw "Error: Failed to install PyTorch."
}

# 4. Install runtime requirements
Write-Host "[4/4] Installing packages from requirements-runtime.txt..." -ForegroundColor Yellow
& $pythonExe -m pip install -r requirements-runtime.txt
if ($LASTEXITCODE -ne 0) {
    throw "Error: Failed to install dependencies from requirements-runtime.txt."
}

Write-Host ""
Write-Host "==========================================================================" -ForegroundColor Green
Write-Host " SETUP COMPLETED SUCCESSFULLY! SYSTEM IS READY." -ForegroundColor Green
Write-Host " You can now run:" -ForegroundColor Cyan
Write-Host "   - Run Live Cam.bat            : Single Camera / Webcam Tracking & ReID"
Write-Host "   - Run Multi Cam Demo.bat      : Multi-Camera Tracking + 2D Floor Plan"
Write-Host "   - Run Web Dashboard.bat       : Streamlit Analytics Dashboard"
Write-Host "==========================================================================" -ForegroundColor Green
