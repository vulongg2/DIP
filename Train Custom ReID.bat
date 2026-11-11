@echo off
setlocal
cd /d "%~dp0"

echo ======================================================================
echo  HUAN LUYEN / FINE-TUNE MO HINH CUSTOM REID (METRIC LEARNING)
echo ======================================================================

set "PYTHON_EXE=.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
    set "PYTHON_EXE=python"
)

"%PYTHON_EXE%" scripts\train_reid.py %*

echo.
echo ======================================================================
echo  HOAN TAT! Nhan phim bat ky de thoat...
echo ======================================================================
pause
