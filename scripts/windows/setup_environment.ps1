# MTMCT-Vision Environment Setup Script
# Tự động phát hiện GPU NVIDIA (CUDA 12.1) hoặc CPU, tạo môi trường ảo .venv và cài đặt dependencies.
param(
    [ValidateSet('Auto','CPU','CUDA121')][string]$Backend = 'Auto'
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location -LiteralPath $repoRoot

Write-Host "==========================================================================" -ForegroundColor Cyan
Write-Host " MTMCT-Vision | Bộ Cài Đặt Môi Trường Tự Động (Windows 10/11 x64)" -ForegroundColor Cyan
Write-Host "==========================================================================" -ForegroundColor Cyan

# 1. Kiểm tra Python
$pythonExe = Join-Path $repoRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) {
    Write-Host "[1/4] Đang khởi tạo môi trường ảo .venv..." -ForegroundColor Yellow
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        throw "Lỗi: Không tìm thấy python trong PATH. Hãy cài đặt Python 3.10/3.11 và tích chọn 'Add python to PATH'."
    }
} else {
    Write-Host "[1/4] Môi trường ảo .venv đã tồn tại sẵn sàng." -ForegroundColor Green
}

# 2. Xác định cấu hình GPU / CPU
if ($Backend -eq 'Auto') {
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        $Backend = 'CUDA121'
        Write-Host "[2/4] Phát hiện Card đồ họa rời NVIDIA -> Chọn PyTorch CUDA 12.1" -ForegroundColor Green
    } else {
        $Backend = 'CPU'
        Write-Host "[2/4] Không phát hiện NVIDIA GPU -> Chọn PyTorch bản CPU" -ForegroundColor Yellow
    }
}

# 3. Cài đặt PyTorch
Write-Host "[3/4] Cài đặt PyTorch ($Backend)..." -ForegroundColor Yellow
$torchIndex = if ($Backend -eq 'CPU') { 'https://download.pytorch.org/whl/cpu' } else { 'https://download.pytorch.org/whl/cu121' }
& $pythonExe -m pip install torch torchvision --index-url $torchIndex
if ($LASTEXITCODE -ne 0) { throw "Cài đặt PyTorch thất bại." }

# 4. Cài đặt các thư viện Runtime
Write-Host "[4/4] Cài đặt các thư viện từ requirements-runtime.txt..." -ForegroundColor Yellow
& $pythonExe -m pip install -r requirements-runtime.txt
if ($LASTEXITCODE -ne 0) { throw "Cài đặt thư viện dependencies thất bại." }

Write-Host ""
Write-Host "==========================================================================" -ForegroundColor Green
Write-Host " CÀI ĐẶT THÀNH CÔNG 100%! HỆ THỐNG ĐÃ SẴN SÀNG HOẠT ĐỘNG." -ForegroundColor Green
Write-Host " Bạn có thể chạy:" -ForegroundColor Cyan
Write-Host "   - Run Live Cam.bat            : Giám sát 1 camera (Webcam / IP Cam)"
Write-Host "   - Run Multi Cam Demo.bat      : Giám sát đa camera + Sơ đồ 2D Floor Plan"
Write-Host "   - Run Web Dashboard.bat       : Mở Dashboard phân tích trên trình duyệt"
Write-Host "==========================================================================" -ForegroundColor Green
