@echo off
setlocal
title Qwen3-TTS Studio
cd /d "%~dp0"
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

echo ============================================
echo   Qwen3-TTS Studio
echo ============================================

rem --- 1. Python 3.11 ---------------------------------------------------------
py -3.11 --version >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [setup] Python 3.11 no encontrado. Intentando instalarlo con winget...
    where winget >nul 2>nul
    if %ERRORLEVEL% NEQ 0 (
        echo [error] No hay winget. Instala Python 3.11 desde https://www.python.org/downloads/ y vuelve a ejecutar.
        pause
        exit /b 1
    )
    winget install -e --id Python.Python.3.11 --silent --accept-package-agreements --accept-source-agreements
    rem refrescar PATH de esta sesion
    set "PATH=%LOCALAPPDATA%\Programs\Python\Launcher;%PATH%"
    py -3.11 --version >nul 2>nul
    if %ERRORLEVEL% NEQ 0 (
        echo [error] Python 3.11 se instalo pero no se detecta. Cierra esta ventana y ejecuta iniciar.bat de nuevo.
        pause
        exit /b 1
    )
)

rem --- 2. Entorno virtual -----------------------------------------------------
if not exist "venv\Scripts\python.exe" (
    echo [setup] Creando entorno virtual...
    py -3.11 -m venv venv
    if %ERRORLEVEL% NEQ 0 (
        echo [error] No se pudo crear el venv.
        pause
        exit /b 1
    )
)

rem --- 3. Instalar si falta y arrancar ---------------------------------------
"venv\Scripts\python.exe" launcher.py %*
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
    echo.
    echo [error] La aplicacion termino con codigo %RC%. Revisa los mensajes anteriores.
    echo         Para reinstalar: iniciar.bat --reinstall    Para forzar CPU: iniciar.bat --cpu
    pause
)
exit /b %RC%
