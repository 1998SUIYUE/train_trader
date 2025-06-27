# Basic Package Installation - Just the essentials to get started
# Installs only numpy and pandas needed for sample data generation

Write-Host "=== Installing Basic Packages ===" -ForegroundColor Green
Write-Host "Installing only numpy and pandas to get started quickly" -ForegroundColor Yellow

# Check Python 3.11
$pythonCmd = ""
try {
    py -3.11 --version | Out-Null
    $pythonCmd = "py -3.11"
    Write-Host "Found Python 3.11" -ForegroundColor Green
} catch {
    try {
        & "C:\Python311\python.exe" --version | Out-Null
        $pythonCmd = "C:\Python311\python.exe"
        Write-Host "Found Python 3.11 (direct path)" -ForegroundColor Green
    } catch {
        Write-Host "Python 3.11 not found!" -ForegroundColor Red
        Read-Host "Press any key to exit"
        exit 1
    }
}

# Install essential packages
$essentialPackages = @("numpy", "pandas")

foreach ($pkg in $essentialPackages) {
    Write-Host "Installing $pkg..." -ForegroundColor Cyan
    try {
        if ($pythonCmd -eq "py -3.11") {
            py -3.11 -m pip install $pkg
        } else {
            & $pythonCmd -m pip install $pkg
        }
        Write-Host "$pkg installed!" -ForegroundColor Green
    } catch {
        Write-Host "Failed to install $pkg" -ForegroundColor Red
    }
}

# Test
Write-Host "`nTesting..." -ForegroundColor Cyan
try {
    if ($pythonCmd -eq "py -3.11") {
        py -3.11 -c "import numpy, pandas; print('Basic packages working!')"
    } else {
        & $pythonCmd -c "import numpy, pandas; print('Basic packages working!')"
    }
    Write-Host "Ready to generate sample data!" -ForegroundColor Green
} catch {
    Write-Host "Test failed" -ForegroundColor Red
}

Read-Host "Press any key to continue"