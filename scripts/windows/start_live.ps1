$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location -LiteralPath $repoRoot

$sysPy = "C:\Users\Windows\AppData\Local\Programs\Python\Python311\python.exe"
if (Test-Path -LiteralPath $sysPy) {
    $pythonExe = $sysPy
} else {
    $pythonExe = "python"
}

& $pythonExe "$repoRoot\scripts\run_live_cam.py" @args
