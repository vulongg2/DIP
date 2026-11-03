$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location -LiteralPath $repoRoot

$streamlitExe = Join-Path $repoRoot '.venv\Scripts\streamlit.exe'
if (-not (Test-Path -LiteralPath $streamlitExe)) {
    streamlit run "$repoRoot\scripts\run_dashboard.py"
} else {
    & $streamlitExe run "$repoRoot\scripts\run_dashboard.py"
}
