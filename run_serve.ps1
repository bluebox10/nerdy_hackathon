Write-Host "===================================================" -ForegroundColor Cyan
Write-Host "Starting WhyWrong Server on http://127.0.0.1:8077" -ForegroundColor Green
Write-Host "Misconception-level diagnosis for adaptive practice" -ForegroundColor Yellow
Write-Host "===================================================" -ForegroundColor Cyan

$serveDir = Join-Path $PSScriptRoot "serve"
Set-Location $serveDir

python -m uvicorn app:app --host 0.0.0.0 --port 8077
