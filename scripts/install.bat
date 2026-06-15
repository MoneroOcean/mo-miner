@echo off
rem %~dp0 = this script's directory (trailing backslash); run the sibling installer, forwarding all args.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
exit /b %ERRORLEVEL%
