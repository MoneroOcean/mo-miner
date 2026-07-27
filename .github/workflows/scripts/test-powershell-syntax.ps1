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
  [System.Management.Automation.Language.Parser]::ParseFile(
    $path.FullName, [ref]$tokens, [ref]$errors) | Out-Null
  foreach ($error in @($errors)) {
    Write-Error "$($path.FullName)`:$($error.Extent.StartLineNumber): $($error.Message)"
    $failed = $true
  }
}
if ($failed) { throw 'PowerShell syntax validation failed.' }
Write-Host 'PowerShell syntax validation passed.'
