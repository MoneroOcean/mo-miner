param(
  [Parameter(Mandatory = $true)]
  [string]$DownloadPage,

  [Parameter(Mandatory = $true)]
  [string]$PackageName
)

$ErrorActionPreference = "Stop"

$escapedPackage = [regex]::Escape($PackageName)
$installerPattern = "https://registrationcenter-download\.intel\.com/[^`"'<>\s]+/$escapedPackage-[0-9][^`"'<>\s]*_offline\.exe"

if ($DownloadPage -match "\.exe($|\?)") {
  Write-Output ($DownloadPage -replace "_offline(?=\.exe($|\?))", "")
  exit 0
}

$page = Invoke-WebRequest `
  -UseBasicParsing `
  -Uri $DownloadPage `
  -UserAgent "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/125.0 Safari/537.36"

$match = [regex]::Match($page.Content, $installerPattern)
if (-not $match.Success) {
  throw "Unable to resolve Intel oneAPI download URL for $PackageName from $DownloadPage"
}

# Intel publishes the offline URL in static docs. The matching online
# bootstrapper URL uses the same path without the "_offline" suffix.
Write-Output ($match.Value -replace "_offline\.exe$", ".exe")
