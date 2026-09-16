@echo off
echo ===================================================
echo Starting WhyWrong Server on http://127.0.0.1:8077
echo Misconception-level diagnosis for adaptive practice
echo ===================================================

cd /d "%~dp0serve"
python -m uvicorn app:app --host 0.0.0.0 --port 8077
pause
