@echo off
rem Canonical Windows mom development environment installer. GPU display drivers are prerequisites
rem and are intentionally never installed here. PowerShell owns MSI/VS/toolchain error handling;
rem this batch file is the stable user, VM, and GitHub Actions entry point.
setlocal
rem Keep Windows PowerShell 5.1 isolated from PowerShell 7 modules in its parent (pwsh would
rem otherwise hand it 7.x module paths and cmdlets such as Get-FileHash fail to load).
set "PSModulePath=%SystemRoot%\System32\WindowsPowerShell\v1.0\Modules"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0install-dev.ps1" %*
exit /b %ERRORLEVEL%
