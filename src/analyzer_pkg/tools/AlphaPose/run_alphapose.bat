@echo off
rem ---------------------------------------------------------------------------
rem  run_alphapose.bat  – Windows, v10 (minimal: no tracker, no debug frames)
rem ---------------------------------------------------------------------------

if not defined ALPHAPOSE_ROOT (
    echo ❌  ALPHAPOSE_ROOT env var not set
    exit /b 1
)

rem ── fixed batch sizes; tweak if VRAM is tiny ───────────────────────────────
set "DET=32"
set "POSE=64"
set "Q=32"
echo ℹ️  detbatch=%DET%  posebatch=%POSE%  qsize=%Q%

rem ── parse args (same three modes as before) ────────────────────────────────
if "%~2" NEQ "" (
    set "JOB_DIR=%~f1" & set "VIDEO=%~f2"
) else if exist "%~1\" (
    set "JOB_DIR=%~f1" & set "VIDEO=%~f1\src.mp4"
) else if not "%~1"=="" (
    set "VIDEO=%~f1"
    set "JOB_DIR=%TEMP%\video_meta_%RANDOM%"
    mkdir "%JOB_DIR%"
) else (
    echo Usage: %~nx0 ^<job_dir^> [video]
    exit /b 1
)

if not exist "%JOB_DIR%\alphapose" mkdir "%JOB_DIR%\alphapose"

rem ── run AlphaPose (**--sp** = single‑person, so no tracker/no cython_bbox) ─
set "ROOT=%ALPHAPOSE_ROOT%"
set "CFG=%ROOT%\configs\halpe_26\resnet\256x192_res50_lr1e-3_1x.yaml"
set "CKPT=%ROOT%\pretrained_models\halpe26_fast_res50_256x192.pth"

pushd "%ROOT%"
python scripts\demo_inference.py ^
        --cfg        "%CFG%" ^
        --checkpoint "%CKPT%" ^
        --video      "%VIDEO%" ^
        --outdir     "%JOB_DIR%\alphapose" ^
        --detbatch   %DET% ^
        --posebatch  %POSE% ^
        --qsize      %Q% ^
        --sp
if errorlevel 1 (popd & exit /b 1)
popd

echo ✅  AlphaPose done → %JOB_DIR%\alphapose
endlocal
