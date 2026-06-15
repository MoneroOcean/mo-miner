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
set "INSTALLER=%TEMP%\oneapi-webimage.exe"
set "EXTRACT_DIR=oneapi_webimage_extracted"

if "%PACKAGE%"=="" (
  echo Package name is required.
  exit /b 1
)

for /f "delims=" %%U in ('powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%resolve-oneapi-url.ps1" "%URL%" "%PACKAGE%"') do set "URL=%%U"

curl.exe --fail --location --show-error --retry 5 --retry-delay 5 --output "%INSTALLER%" --url "%URL%"
if errorlevel 1 exit /b 1

rd /s /q "%EXTRACT_DIR%" >nul 2>nul
start "" /b /wait "%INSTALLER%" -s -x -f "%EXTRACT_DIR%" --log extract.log
rem Save the exit code before cleanup (del/rd) clobbers ERRORLEVEL.
set "EXTRACT_EXIT_CODE=%ERRORLEVEL%"
del "%INSTALLER%" >nul 2>nul
if not "%EXTRACT_EXIT_CODE%"=="0" exit /b %EXTRACT_EXIT_CODE%

"%EXTRACT_DIR%\bootstrapper.exe" -s --action install --components=%COMPONENTS% --eula=accept -p=NEED_VS2017_INTEGRATION=0 -p=NEED_VS2019_INTEGRATION=0 -p=NEED_VS2022_INTEGRATION=1 --log-dir=.
rem Save the exit code before cleanup (rd) clobbers ERRORLEVEL.
set "INSTALLER_EXIT_CODE=%ERRORLEVEL%"
rd /s /q "%EXTRACT_DIR%" >nul 2>nul
exit /b %INSTALLER_EXIT_CODE%
