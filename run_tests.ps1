# run_tests.ps1  —  Wrapper to run the regression suite with correct env vars
# Usage:  .\run_tests.ps1
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
Write-Host "Running 9-point regression suite..." -ForegroundColor Cyan
.\venv\Scripts\python.exe run_test_suite.py
if ($LASTEXITCODE -eq 0) {
    Write-Host "`nAll tests passed!" -ForegroundColor Green
} else {
    Write-Host "`nTest suite failed with exit code $LASTEXITCODE" -ForegroundColor Red
}
