$ErrorActionPreference = "Stop"

$Repo = "aep/moxgo"
$Binary = "moxgo.exe"
$InstallDir = "$env:LOCALAPPDATA\moxgo"

$release = Invoke-RestMethod -Uri "https://api.github.com/repos/$Repo/releases/latest"
$tag = $release.tag_name

$arch = if ([Environment]::Is64BitOperatingSystem) { "amd64" } else {
    Write-Error "Unsupported architecture: 32-bit Windows is not supported."
    exit 1
}

$asset = "moxgo-${tag}-windows_${arch}.exe"
$url = "https://github.com/$Repo/releases/download/$tag/$asset"

Write-Host "Downloading moxgo $tag for windows/$arch..."

if (-not (Test-Path $InstallDir)) {
    New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
}

$dest = Join-Path $InstallDir $Binary
Invoke-WebRequest -Uri $url -OutFile $dest -UseBasicParsing

$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($userPath -notlike "*$InstallDir*") {
    [Environment]::SetEnvironmentVariable("Path", "$userPath;$InstallDir", "User")
    $env:Path = "$env:Path;$InstallDir"
    Write-Host "Added $InstallDir to your PATH."
}

Write-Host "Installed moxgo $tag to $dest"
Write-Host "Restart your terminal, then run: moxgo"
