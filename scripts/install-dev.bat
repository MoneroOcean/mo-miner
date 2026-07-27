@echo off
rem Canonical Windows mom development environment installer. GPU display drivers are prerequisites
rem and are intentionally never installed here. PowerShell owns MSI/VS/toolchain error handling;
rem this batch file is the stable user, VM, and GitHub Actions entry point.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0install-dev.ps1" %*
exit /b %ERRORLEVEL%
