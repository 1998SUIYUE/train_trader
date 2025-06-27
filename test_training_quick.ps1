# Quick Test for Training Scripts
# Tests if the logging issue is fixed

Write-Host "=== Testing Training Script Fix ===" -ForegroundColor Green

# Ensure results directory exists
if (-not (Test-Path "results")) {
    New-Item -ItemType Directory -Path "results" -Force | Out-Null
    Write-Host "Created results directory" -ForegroundColor Green
}

# Test GPU training with very small parameters
Write-Host "Testing GPU training with minimal parameters..." -ForegroundColor Cyan

try {
    py -3.11 core/main_gpu.py --data_file "XAUUSD_M1_202503142037_202506261819.csv" --population_size 10 --generations 2
    Write-Host "SUCCESS: GPU training test completed!" -ForegroundColor Green
} catch {
    Write-Host "FAILED: GPU training test failed" -ForegroundColor Red
    Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Yellow
}

Write-Host "`nTest completed!" -ForegroundColor Green
Write-Host "If successful, you can now run the full training scripts" -ForegroundColor White

Read-Host "`nPress any key to exit"