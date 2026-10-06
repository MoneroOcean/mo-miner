@echo off
rem Install Intel oneAPI components from a web installer.
rem   %1 URL        download page (or direct .exe) to resolve the installer from
rem   %2 COMPONENTS comma-separated component list passed to the bootstrapper
rem   %3 PACKAGE    package name used to match the installer URL (required)
setlocal
set "URL=%~1"
set "COMPONENTS=%~2"
set "PACKAGE=%~3"
set "SCRIPT_DIR=%~dp0"

if "%PACKAGE%"=="" (
  echo Package name is required.
  exit /b 1
)

set "RUN_ID="
for /f "delims=" %%G in ('powershell -NoProfile -Command "[Guid]::NewGuid().ToString()"') do set "RUN_ID=%%G"
if not defined RUN_ID (
  echo Unable to create a unique oneAPI installer workspace.
  exit /b 1
)
set "INSTALLER=%TEMP%\mom-oneapi-%RUN_ID%.exe"
set "EXTRACT_DIR=%TEMP%\mom-oneapi-%RUN_ID%"
set "EXTRACT_LOG=%TEMP%\mom-oneapi-%RUN_ID%-extract.log"
set "EXIT_CODE=1"

set "RESOLVED_URL="
for /f "delims=" %%U in ('powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%resolve-oneapi-url.ps1" "%URL%" "%PACKAGE%"') do set "RESOLVED_URL=%%U"
if not defined RESOLVED_URL goto cleanup
set "URL=%RESOLVED_URL%"

curl.exe --fail --location --show-error --retry 5 --retry-delay 5 --output "%INSTALLER%" --url "%URL%"
if not errorlevel 1 goto download_ok
set "EXIT_CODE=%ERRORLEVEL%"
goto cleanup
:download_ok
powershell -NoProfile -Command "$s=Get-AuthenticodeSignature -LiteralPath $env:INSTALLER; if($s.Status -ne 'Valid'){Write-Error ('Authenticode verification failed: '+$s.Status); exit 1}"
if not errorlevel 1 goto installer_signature_ok
set "EXIT_CODE=%ERRORLEVEL%"
goto cleanup
:installer_signature_ok

start "" /b /wait "%INSTALLER%" -s -x -f "%EXTRACT_DIR%" --log "%EXTRACT_LOG%"
set "EXTRACT_EXIT_CODE=%ERRORLEVEL%"
if "%EXTRACT_EXIT_CODE%"=="0" goto extraction_ok
set "EXIT_CODE=%EXTRACT_EXIT_CODE%"
goto cleanup
:extraction_ok

set "BOOTSTRAPPER=%EXTRACT_DIR%\bootstrapper.exe"
if not exist "%BOOTSTRAPPER%" goto cleanup
powershell -NoProfile -Command "$s=Get-AuthenticodeSignature -LiteralPath $env:BOOTSTRAPPER; if($s.Status -ne 'Valid'){Write-Error ('Authenticode verification failed: '+$s.Status); exit 1}"
if not errorlevel 1 goto bootstrapper_signature_ok
set "EXIT_CODE=%ERRORLEVEL%"
goto cleanup
:bootstrapper_signature_ok
"%BOOTSTRAPPER%" -s --action install --components=%COMPONENTS% --eula=accept -p=NEED_VS2017_INTEGRATION=0 -p=NEED_VS2019_INTEGRATION=0 -p=NEED_VS2022_INTEGRATION=1 --log-dir="%EXTRACT_DIR%\logs"
set "EXIT_CODE=%ERRORLEVEL%"

:cleanup
del /f /q "%INSTALLER%" >nul 2>nul
del /f /q "%EXTRACT_LOG%" >nul 2>nul
rd /s /q "%EXTRACT_DIR%" >nul 2>nul
exit /b %EXIT_CODE%
