#requires -Version 5.1
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $repo

& "$repo\.github\workflows\scripts\test-powershell-syntax.ps1"

& node.exe --test tests/windows_command.js tests/windows_script_safety.js
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
