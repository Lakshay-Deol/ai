@echo off
title Forward Scholar - Research Paper Search Engine (Frontend Mode)
echo ======================================================================
echo Launching Forward Scholar in Interactive Frontend Mode...
echo Server URL: http://127.0.0.1:8000
echo ======================================================================

timeout /t 1 /nobreak >nul
if exist "C:\Program Files\Google\Chrome\Application\chrome.exe" (
    start "" "C:\Program Files\Google\Chrome\Application\chrome.exe" --app=http://127.0.0.1:8000
) else if exist "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" (
    start "" "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" --app=http://127.0.0.1:8000
) else (
    start "" http://127.0.0.1:8000
)

python app.py
pause
