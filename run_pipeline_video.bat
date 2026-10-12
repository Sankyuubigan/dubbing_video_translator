@echo off
setlocal enableextensions
REM Полный прогон пайплайна на произвольном видео.
REM Запуск: run_pipeline_video.bat "<путь к видео>" [1=с дубляжом]
REM Пример: run_pipeline_video.bat "E:\Downloads\20mintest - Melanie Marie [2023].mp4"
for %%I in ("%~dp0.") do set "PROJ=%%~fI"

if "%~1"=="" (
    echo [ERROR] Нужен путь к видео первым аргументом.
    pause
    exit /b 1
)

REM --- Find VS and init MSVC environment ---
set "VS_INIT_OK="
for /f "usebackq delims=" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -property installationPath 2^>nul`) do (
    if exist "%%i\VC\Auxiliary\Build\vcvarsall.bat" (
        call "%%i\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul 2>&1
        set "VS_INIT_OK=1"
    )
)
if not defined VS_INIT_OK (
  echo [ERROR] Visual Studio not found. Install VS with C++ workload.
  pause
  exit /b 1
)

REM --- Reset wrappers so they do not leak into cargo ---
set "CC="
set "CXX="
set "CMAKE_C_COMPILER_LAUNCHER="
set "CMAKE_CXX_COMPILER_LAUNCHER="
set "RUSTC_WRAPPER="
set "CARGO_BUILD_RUSTC_WRAPPER="
set "CARGO_PROFILE_RELEASE_LTO="
set "CARGO_PROFILE_RELEASE_CODEGEN_UNITS="
set "CARGO_PROFILE_RELEASE_STRIP="

REM --- Ensure Rust/Cargo on PATH (vcvarsall may reset it) ---
set "PATH=%USERPROFILE%\.cargo\bin;%USERPROFILE%\.rustup\toolchains\stable-x86_64-pc-windows-msvc\bin;%PATH%"

cd /d "%PROJ%\src-tauri"

if /i "%~2"=="1" (
    set DEEDUB_KEEP_TTS_WAV=1
    set DUB_FLAG=--dub
) else (
    set DUB_FLAG=
)

echo Running pipeline on %~1 ...
cargo run --bin test-pipeline --release -- --video "%~1" %DUB_FLAG%

if errorlevel 1 (
    echo PIPELINE FAILED
    pause
    exit /b 1
)
echo PIPELINE OK
endlocal