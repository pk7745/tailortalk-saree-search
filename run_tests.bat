@echo off
REM Wrapper batch file to run the regression test suite with correct env vars
SET KMP_DUPLICATE_LIB_OK=TRUE
SET OMP_NUM_THREADS=1
SET MKL_NUM_THREADS=1
echo Running 9-point regression suite...
.\venv\Scripts\python.exe run_test_suite.py
