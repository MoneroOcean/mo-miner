# Run a command as a child process tree, kill it after -Minutes and retry up to -Attempts times.
# For idempotent CI setup only: the caller keeps no in-process environment changes, so exports must
# go through GITHUB_ENV/GITHUB_PATH or the machine. Run it as the same user as the command (sudo pwsh
# for root installers) so the whole tree can be killed.
# Usage: retry.ps1 [-Minutes 10] [-Attempts 3] [--] command args...
# Raw $args, not param(): the binder would claim the command's own -options and mishandle "--".
$ErrorActionPreference = 'Stop'
$options = @{Minutes = 10; Attempts = 3}
$Command = [Collections.Generic.List[string]]::new([string[]]$args)
while ($Command.Count -ge 2 -and $Command[0].TrimStart('-') -in $options.Keys) {
  $options[$Command[0].TrimStart('-')] = [int]$Command[1]
  $Command.RemoveRange(0, 2)
}
if ($Command.Count -and $Command[0] -eq '--') { $Command.RemoveAt(0) }
if (-not $Command.Count) { throw 'Usage: retry.ps1 [-Minutes N] [-Attempts N] [--] command args...' }
# PATH/PATHEXT resolution starts npm.cmd and friends without a shell; ArgumentList needs no quoting.
$info = [Diagnostics.ProcessStartInfo]::new((Get-Command $Command[0] -CommandType Application)[0].Source)
$Command | Select-Object -Skip 1 | ForEach-Object { $info.ArgumentList.Add($_) }
foreach ($attempt in 1..$options.Attempts) {
  $process = [Diagnostics.Process]::Start($info)
  $deadline = [DateTime]::UtcNow.AddMinutes($options.Minutes)
  # Heartbeat every 5 min: if the log stops without one, the runner died rather than the command hung.
  while (-not $process.WaitForExit([int][Math]::Max(0, [Math]::Min(300000, ($deadline - [DateTime]::UtcNow).TotalMilliseconds)))) {
    if ([DateTime]::UtcNow -ge $deadline) { break }
    Write-Host "[retry $(Get-Date -Format HH:mm:ss)] attempt $attempt still running: $($Command[0])"
  }
  if ($process.HasExited -and $process.ExitCode -eq 0) { exit 0 }
  if (-not $process.HasExited) { $process.Kill($true) } # whole tree, on every platform
  Write-Warning "attempt $attempt of $($options.Attempts) failed or exceeded $($options.Minutes) min: $Command"
}
exit 1
