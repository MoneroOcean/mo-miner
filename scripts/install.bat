@echo off
rem %~dp0 = this script's directory (trailing backslash); run the sibling installer, forwarding all args.
setlocal
rem Keep Windows PowerShell 5.1 isolated from PowerShell 7 modules in its parent (pwsh would
rem otherwise hand it 7.x module paths and cmdlets such as Get-FileHash fail to load).
set "PSModulePath=%SystemRoot%\System32\WindowsPowerShell\v1.0\Modules"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
exit /b %ERRORLEVEL%
