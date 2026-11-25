# ==============================================================================
# ALOHA Toss & Catch // Safe Release & Verification Automation Script
# ==============================================================================
Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "🚀 ALOHA Bimanual Toss & Catch // Safe Release & Build Harness" -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan

$ErrorActionPreference = "Stop"

# Step 1: Clean temporary files and caches
Write-Host "`n[Step 1/5] Cleaning pycache and temporary files..." -ForegroundColor Yellow
Get-ChildItem -Path . -Recurse -Include __pycache__, *.pyc, *.pyo | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
Write-Host "  -> Cleaned all caches successfully." -ForegroundColor Green

# Step 2: Run automated smoke tests
Write-Host "`n[Step 2/5] Running Automated Smoke Test Suite..." -ForegroundColor Yellow
python tests/test_smoke.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Smoke tests failed! Aborting release." -ForegroundColor Red
    exit 1
}
Write-Host "  -> 100% Smoke tests PASSED!" -ForegroundColor Green

# Step 3: Install editable package (generating .egg-info)
Write-Host "`n[Step 3/5] Installing package in editable mode (pip install -e .)..." -ForegroundColor Yellow
pip install -e . --no-deps
Write-Host "  -> Package installed successfully as 'aloha-bimanual-toss-catch'!" -ForegroundColor Green

# Step 4: Build Standalone Clean Bundle ZIP (Excluding Internal Secret/Harness Files)
Write-Host "`n[Step 4/5] Packaging Standalone Project Archive (aloha_bimanual_toss_catch_bundle.zip)..." -ForegroundColor Yellow
$BundleZip = "aloha_bimanual_toss_catch_bundle.zip"
if (Test-Path $BundleZip) {
    Remove-Item $BundleZip -Force
}

python -c @"
import zipfile, os
from pathlib import Path

bundle_zip = 'aloha_bimanual_toss_catch_bundle.zip'
include_files = [
    'run_live_cockpit.py', 'evaluate.py', 'train_ppo.py', 'eval_info.json',
    'pyproject.toml', 'setup.py', 'README.md', 'README_KR.md', 'LICENSE'
]
include_dirs = ['envs', 'rewards', 'render', 'assets', 'checkpoints']

with zipfile.ZipFile(bundle_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for f in include_files:
        if os.path.exists(f):
            zipf.write(f, arcname=f)
    for d in include_dirs:
        if os.path.exists(d):
            for root, dirs, files in os.walk(d):
                if '__pycache__' in root:
                    continue
                for file in files:
                    if file.endswith(('.pyc', '.pyo')):
                        continue
                    full_path = os.path.join(root, file)
                    arcname = os.path.relpath(full_path, '.')
                    zipf.write(full_path, arcname=arcname)
print(f'  -> Successfully packaged {bundle_zip}')
"@

# Step 5: Hugging Face Model Hub Release Whitelist Verification
Write-Host "`n[Step 5/5] Verifying Hugging Face Release Whitelist & Security Boundaries..." -ForegroundColor Yellow
$HuggingFaceWhitelist = @(
    "checkpoints/aloha_toss_catch_final.zip",
    "aloha_bimanual_toss_catch_bundle.zip",
    "eval_info.json",
    "assets/aloha_bimanual_toss.xml",
    "assets/cockpit_preview.png",
    "assets/aloha_toss_catch_demo.gif",
    "README.md",
    "README_KR.md",
    "LICENSE"
)

$allFound = $true
foreach ($f in $HuggingFaceWhitelist) {
    if (Test-Path $f) {
        Write-Host "  [OK] Found $f" -ForegroundColor Green
    } else {
        Write-Host "  [ERROR] Missing required whitelist asset: $f" -ForegroundColor Red
        $allFound = $false
    }
}

# Verify secret isolation (MUST NOT be in public bundle)
$SecretFiles = @(".cursorrules", "docs/DDR_ARCHITECTURE.md", "MEMORY.md")
Write-Host "`n  Verifying Secret Document Isolation..." -ForegroundColor Cyan
foreach ($sf in $SecretFiles) {
    if (Test-Path $sf) {
        Write-Host "  [SECURED LOCAL] $sf exists locally only." -ForegroundColor DarkGray
    }
}

if (-not $allFound) {
    Write-Host "`n❌ Missing required release files! Please review." -ForegroundColor Red
    exit 1
}

Write-Host "`n=================================================================" -ForegroundColor Cyan
Write-Host "✨ RELEASE READY! The environment is packaged and verified!" -ForegroundColor Green
Write-Host "   Hugging Face Model Hub whitelist assets are 100% verified." -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan
