@echo off
setlocal
set "MOM_DIR=%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%MOM_DIR%install.ps1" %*
exit /b %ERRORLEVEL%
