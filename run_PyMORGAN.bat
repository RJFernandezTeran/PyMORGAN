@echo off
setlocal enabledelayedexpansion

title PyMORGAN

cd /d "%~dp0"

:: Check for existing Python environment (.venv, uv, python, py)
set "PYTHON="

if exist ".venv\Scripts\python.exe" (
    set "PYTHON=.venv\Scripts\python.exe"
    goto :run
)

where uv >nul 2>&1
if not errorlevel 1 (
    set "USE_UV=1"
    goto :run
)

where python >nul 2>&1
if not errorlevel 1 (
    set "PYTHON=python"
    goto :run
)

where py >nul 2>&1
if not errorlevel 1 (
    set "PYTHON=py"
    goto :run
)

echo [ERROR] No Python interpreter was found on your system.
echo Please install Python (>= 3.12) from https://www.python.org/ or install uv.
echo.
pause
exit /b 1

:run
if defined USE_UV (
    uv run python -m pymorgan %*
) else (
    "%PYTHON%" -m pymorgan %*
)

if errorlevel 1 (
    echo.
    echo [PyMORGAN exited with error code %errorlevel%]
    pause
)
