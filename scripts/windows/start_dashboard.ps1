$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location -LiteralPath $repoRoot

$streamlitExe = "C:\Users\Windows\AppData\Local\Programs\Python\Python311\Scripts\streamlit.exe"
if (Test-Path -LiteralPath $streamlitExe) {
    & $streamlitExe run "$repoRoot\scripts\run_dashboard.py"
} else {
    streamlit run "$repoRoot\scripts\run_dashboard.py"
}
