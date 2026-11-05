# Launch Forward Scholar in Frontend Window Mode
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "Forward Scholar - Research Paper Search Engine (Frontend Mode)" -ForegroundColor Green
Write-Host "URL: http://127.0.0.1:8000" -ForegroundColor Yellow
Write-Host "======================================================================" -ForegroundColor Cyan

Start-Sleep -Seconds 1
if (Test-Path "C:\Program Files\Google\Chrome\Application\chrome.exe") {
    Start-Process "C:\Program Files\Google\Chrome\Application\chrome.exe" -ArgumentList "--app=http://127.0.0.1:8000"
} elseif (Test-Path "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe") {
    Start-Process "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" -ArgumentList "--app=http://127.0.0.1:8000"
} else {
    Start-Process "http://127.0.0.1:8000"
}

python app.py
