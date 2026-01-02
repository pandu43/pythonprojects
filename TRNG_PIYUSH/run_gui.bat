@echo off
REM Activate virtual environment
call .venv\Scripts\activate

REM Set PYTHONPATH to include src so imports work
set "PYTHONPATH=%~dp0src;%PYTHONPATH%"

REM Run the harness GUI
python scripts/harness_gui.py

pause
