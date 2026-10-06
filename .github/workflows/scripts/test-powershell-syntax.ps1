$ErrorActionPreference = 'Stop'

$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$gitDir = Join-Path $repo '.git'
if (Test-Path $gitDir) {
  $relativePaths = @(& git.exe -C $repo ls-files --cached --others --exclude-standard -- '*.ps1')
  if ($LASTEXITCODE -ne 0) { throw 'Unable to enumerate repository PowerShell files.' }
  $paths = @($relativePaths | ForEach-Object { Join-Path $repo $_ })
} else {
  # run.sh and release-source exports deliberately omit .git. PowerShell sources live at the root
  # or under these two script trees; enumerate them directly without descending into uploaded
  # build/toolchain caches or node_modules.
  $paths = @(
    Get-ChildItem $repo -File -Filter '*.ps1'
    Get-ChildItem (Join-Path $repo 'scripts') -File -Filter '*.ps1' -Recurse
    Get-ChildItem (Join-Path $repo '.github\workflows\scripts') -File -Filter '*.ps1' -Recurse
  )
}

$failed = $false
foreach ($entry in $paths) {
  $path = if ($entry -is [IO.FileInfo]) { $entry } else { Get-Item -LiteralPath $entry }
  $tokens = $null
  $errors = $null
  $scriptAst = [System.Management.Automation.Language.Parser]::ParseFile(
    $path.FullName, [ref]$tokens, [ref]$errors)
  foreach ($error in @($errors)) {
    Write-Error "$($path.FullName)`:$($error.Extent.StartLineNumber): $($error.Message)"
    $failed = $true
  }
  if ($null -ne $scriptAst) {
    foreach ($command in $scriptAst.FindAll({
        param($node)
        $node -is [System.Management.Automation.Language.CommandAst]
      }, $true)) {
      $hasLiteralPath = @($command.CommandElements | Where-Object {
        $_ -is [System.Management.Automation.Language.CommandParameterAst] -and
        $_.ParameterName -ieq 'LiteralPath'
      }).Count -gt 0
      if ($command.GetCommandName() -ieq 'New-Item' -and $hasLiteralPath) {
        Write-Error "$($path.FullName)`:$($command.Extent.StartLineNumber): New-Item does not support -LiteralPath in Windows PowerShell 5.1."
        $failed = $true
      }
    }
  }
}
if ($failed) { throw 'PowerShell syntax validation failed.' }

$windirWasPresent = $null -ne (Get-Item Env:WINDIR -ErrorAction SilentlyContinue)
$originalWindir = $env:WINDIR
try {
  . (Join-Path $PSScriptRoot 'windows-dll-deps.ps1')
  $env:WINDIR = 'C:\Windows'
  if (-not (Test-MominerWindowsPath 'C:\Windows')) {
    throw 'Test-MominerWindowsPath rejected C:\Windows.'
  }
  if (-not (Test-MominerWindowsPath 'C:\Windows\System32')) {
    throw 'Test-MominerWindowsPath rejected C:\Windows\System32.'
  }
  if (Test-MominerWindowsPath 'C:\WindowsEvil') {
    throw 'Test-MominerWindowsPath accepted C:\WindowsEvil.'
  }
}
finally {
  if ($windirWasPresent) {
    $env:WINDIR = $originalWindir
  } else {
    Remove-Item Env:WINDIR -ErrorAction SilentlyContinue
  }
}

Write-Host 'PowerShell syntax validation and helper checks passed.'
