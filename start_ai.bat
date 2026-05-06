@echo off
cd /d "C:\Users\Nate\Desktop\aiproject"

echo Starting Ollama...
start "Ollama" cmd /k "ollama run qwen2.5:7b" 

timeout /t 3 >nul

echo Starting FastAPI...
start "FastAPI" cmd /k "python -m uvicorn main:app --reload"

timeout /t 3 >nul

echo Opening browser...
start "" "http://127.0.0.1:8000"

pause
