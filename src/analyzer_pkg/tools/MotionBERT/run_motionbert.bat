@echo off
rem ---------------------------------------------------------------------------
rem   run_motionbert.bat  –  Windows port of run_motionbert.sh
rem ---------------------------------------------------------------------------
rem   Usage:  run_motionbert.bat  <job_dir>
rem ---------------------------------------------------------------------------

setlocal enabledelayedexpansion

:: ----- argument check -------------------------------------------------------
if "%~1"=="" (
    echo Usage: %~nx0 ^<job_dir^>
    exit /b 1
)
set "JOB_DIR=%~1"

:: ----- paths ---------------------------------------------------------------
set "VIDEO=%JOB_DIR%\src.mp4"
set "JSON=%JOB_DIR%\alphapose\alphapose-results.json"
if not exist "%JSON%" (
    echo ❌  2D json missing: "%JSON%"
    exit /b 1
)

set "ROOT=%MOTIONBERT_ROOT%"
set "CFG=%ROOT%\configs\pose3d\MB_ft_h36m_global_lite.yaml"
set "CKPT=%ROOT%\checkpoint\pose3d\FT_MB_lite_MB_ft_h36m_global_lite\best_epoch.bin"
set "OUT_DIR=%JOB_DIR%\motionbert"
if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"

:: ----- banner --------------------------------------------------------------
echo ---- MotionBERT -----------------------------------------------------
echo  video : %VIDEO%
echo  json  : %JSON%
echo  out   : %OUT_DIR%
echo --------------------------------------------------------------------

:: ----- run MotionBERT ------------------------------------------------------
pushd "%ROOT%"
set "QT_QPA_PLATFORM=offscreen"
python infer_wild.py ^
    --config "%CFG%" ^
    -e      "%CKPT%" ^
    -v      "%VIDEO%" ^
    -j      "%JSON%" ^
    -o      "%OUT_DIR%\3d-pose-results.npz" ^
    --pixel
if errorlevel 1 (
    popd
    exit /b 1
)
popd

echo ✅  MotionBERT done – results in %OUT_DIR%
endlocal
