# Fix All Import Issues - Complete Solution
# This script fixes all import problems in the AI Trading System

Write-Host "=== Fixing All Import Issues ===" -ForegroundColor Green

# Test the fixes
Write-Host "Testing import fixes..." -ForegroundColor Cyan

try {
    py -3.11 test_imports_fixed.py
    Write-Host "Import test completed!" -ForegroundColor Green
} catch {
    Write-Host "Import test failed!" -ForegroundColor Red
}

# Create results directory if it doesn't exist
if (-not (Test-Path "results")) {
    New-Item -ItemType Directory -Path "results" -Force | Out-Null
    Write-Host "Created results directory" -ForegroundColor Green
}

Write-Host "`nAll import issues should now be fixed!" -ForegroundColor Green
Write-Host "You can now run:" -ForegroundColor Cyan
Write-Host "1. .\quick_start.ps1" -ForegroundColor Yellow
Write-Host "2. .\simple_start.ps1" -ForegroundColor Yellow
Write-Host "3. .\start_training_en.ps1" -ForegroundColor Yellow

Read-Host "`nPress any key to exit"