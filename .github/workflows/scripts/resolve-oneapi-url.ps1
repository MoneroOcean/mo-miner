param(
  [Parameter(Mandatory = $true)]
  [string]$DownloadPage,

  [Parameter(Mandatory = $true)]
  [string]$PackageName
)

$ErrorActionPreference = "Stop"

# Intel publishes offline installer URLs; the matching online bootstrapper is
# the same URL with the "_offline" suffix dropped from before ".exe".
function ConvertTo-OnlineUrl([string]$url) { $url -replace "_offline(?=\.exe($|\?))", "" }

$escapedPackage = [regex]::Escape($PackageName)
$installerPattern = "https://registrationcenter-download\.intel\.com/[^`"'<>\s]+/$escapedPackage-[0-9][^`"'<>\s]*_offline\.exe"

# Caller may pass a direct .exe link (possibly with a query string); if so,
# skip the page scrape and convert it in place.
if ($DownloadPage -match "\.exe($|\?)") {
  Write-Output (ConvertTo-OnlineUrl $DownloadPage)
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

Write-Output (ConvertTo-OnlineUrl $match.Value)
