@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

set "PS_SCRIPT=%SCRIPT_DIR%start_anatomask.ps1"
if not exist "%PS_SCRIPT%" (
  echo [AnatoMask] ERROR: Missing PowerShell launcher: "%PS_SCRIPT%"
  exit /b 1
)

where powershell.exe >nul 2>nul
if %ERRORLEVEL%==0 (
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%PS_SCRIPT%" %*
  if not %ERRORLEVEL%==0 (
    echo.
    echo [AnatoMask] Launcher failed with exit code %ERRORLEVEL%.
    pause
  )
  exit /b %ERRORLEVEL%
)

where pwsh.exe >nul 2>nul
if %ERRORLEVEL%==0 (
  pwsh.exe -NoProfile -ExecutionPolicy Bypass -File "%PS_SCRIPT%" %*
  if not %ERRORLEVEL%==0 (
    echo.
    echo [AnatoMask] Launcher failed with exit code %ERRORLEVEL%.
    pause
  )
  exit /b %ERRORLEVEL%
)

echo [AnatoMask] ERROR: Neither powershell.exe nor pwsh.exe was found.
pause
exit /b 1
