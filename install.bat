@echo off
setlocal
set "MOMINER_DIR=%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%MOMINER_DIR%install.ps1" %*
exit /b %ERRORLEVEL%
