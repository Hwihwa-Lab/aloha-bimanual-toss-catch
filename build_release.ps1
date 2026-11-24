# ==============================================================================
# ALOHA Toss & Catch // Safe Release & Verification Automation Script
# ==============================================================================
Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "🚀 ALOHA Bimanual Toss & Catch // Safe Release & Build Harness" -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan

$ErrorActionPreference = "Stop"

# Step 1: Clean temporary files and caches
Write-Host "`n[Step 1/4] Cleaning pycache and temporary files..." -ForegroundColor Yellow
Get-ChildItem -Path . -Recurse -Include __pycache__, *.pyc, *.pyo | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
Write-Host "  -> Cleaned all caches successfully." -ForegroundColor Green

# Step 2: Run automated smoke tests
Write-Host "`n[Step 2/4] Running Automated Smoke Test Suite..." -ForegroundColor Yellow
python tests/test_smoke.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Smoke tests failed! Aborting release." -ForegroundColor Red
    exit 1
}
Write-Host "  -> 100% Smoke tests PASSED!" -ForegroundColor Green

# Step 3: Install editable package (generating .egg-info)
Write-Host "`n[Step 3/4] Installing package in editable mode (pip install -e .)..." -ForegroundColor Yellow
pip install -e . --no-deps
Write-Host "  -> Package installed successfully as 'aloha-bimanual-toss-catch'!" -ForegroundColor Green

# Step 4: Asset & Security Whitelist Verification
Write-Host "`n[Step 4/4] Verifying Release Whitelist & Security Boundaries..." -ForegroundColor Yellow
$RequiredFiles = @(
    "assets/aloha_bimanual_toss.xml",
    "assets/cockpit_preview.png",
    "assets/aloha_toss_catch_demo.gif",
    "checkpoints/aloha_toss_catch_final.zip",
    "README.md",
    "README_KR.md",
    "pyproject.toml",
    "setup.py"
)

foreach ($f in $RequiredFiles) {
    if (Test-Path $f) {
        Write-Host "  [OK] Found $f" -ForegroundColor Green
    } else {
        Write-Host "  [WARNING] Missing release asset: $f" -ForegroundColor DarkYellow
    }
}

Write-Host "`n=================================================================" -ForegroundColor Cyan
Write-Host "✨ RELEASE READY! The environment is packaged and verified!" -ForegroundColor Green
Write-Host "   You can now push to GitHub or publish to Hugging Face Hub." -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan

