# deploy.ps1 — copy project files from this Windows machine to the Raspberry Pi
#
# Usage (from the project root):
#   .\scripts\pi\deploy.ps1
#   .\scripts\pi\deploy.ps1 -PiHost pi@192.168.1.42
#
# Prerequisites on this machine:
#   - OpenSSH client installed (built into Windows 10/11)
#   - SSH key auth set up OR you will be prompted for the Pi password
#
# Run this every time you update templates, config, core, or audio_responses.

param(
    [string]$PiHost = "pi@raspberrypi.local"
)

$PI_DIR = "~/speaker-verification"

Write-Host "Deploying to $PiHost ..." -ForegroundColor Cyan

# Helper: scp a local directory to the Pi (creates remote dir if needed)
function Copy-Dir {
    param([string]$LocalPath, [string]$RemotePath)
    Write-Host "  -> $LocalPath"
    scp -r $LocalPath "${PiHost}:${RemotePath}"
    if ($LASTEXITCODE -ne 0) { throw "scp failed for $LocalPath" }
}

# Helper: scp a single file
function Copy-File {
    param([string]$LocalFile, [string]$RemotePath)
    Write-Host "  -> $LocalFile"
    scp $LocalFile "${PiHost}:${RemotePath}"
    if ($LASTEXITCODE -ne 0) { throw "scp failed for $LocalFile" }
}

# Ensure remote directories exist
Write-Host "`n[1/5] Creating remote directories"
ssh $PiHost "mkdir -p $PI_DIR/core $PI_DIR/templates/custom_dataset $PI_DIR/config $PI_DIR/audio_responses $PI_DIR/scripts/pi"

# Core modules
Write-Host "`n[2/5] Copying core/ modules"
Copy-Dir "core" "$PI_DIR"

# Templates
Write-Host "`n[3/5] Copying templates"
Copy-Dir "templates/custom_dataset" "$PI_DIR/templates"

# Config (threshold JSON only — other dataset configs not needed on the Pi)
Write-Host "`n[4/5] Copying config"
Copy-File "config/custom_threshold.json" "$PI_DIR/config/"

# Pi scripts
Write-Host "`n[5a/5] Copying Pi scripts"
Copy-Dir "scripts/pi" "$PI_DIR/scripts"

# Audio responses (only if the directory is populated)
$responseFiles = Get-ChildItem "audio_responses" -Filter "*.wav" -ErrorAction SilentlyContinue
if ($responseFiles) {
    Write-Host "`n[5b/5] Copying audio responses ($($responseFiles.Count) files)"
    Copy-Dir "audio_responses" "$PI_DIR"
} else {
    Write-Host "`n[5b/5] audio_responses/ is empty — skipping." -ForegroundColor Yellow
    Write-Host "       Run 'python scripts\pi\record_responses.py' first, then redeploy."
}

Write-Host "`nDeploy complete." -ForegroundColor Green
Write-Host "SSH into the Pi and run:"
Write-Host "  ~/speaker-verification/scripts/pi/setup_pi.sh   (first time only)"
Write-Host "  python3 ~/speaker-verification/scripts/pi/main.py"
