@echo off
setlocal enableextensions

REM ===================================================================
REM  Полный headless-прогон пайплайна с переопределением модели перевода.
REM  Без DEEDUB_LLM_MODEL / MT_MODEL работает как обычный run_pipeline.bat,
REM  только на test/test_TTS_dubbing.mp4.
REM
REM  Использование:
REM     run_pipeline_mt.bat                              - модель из настроек
REM     run_pipeline_mt.bat <path-to.gguf> [temp] [1=с дубляжом] [chat^|insttrans]
REM ===================================================================

for %%I in ("%~dp0.") do set "PROJ=%%~fI"

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

if not "%~1"=="" (
    if not exist "%~1" (
        echo [ERROR] GGUF not found: %~1
        exit /b 1
    )
    set "DEEDUB_LLM_MODEL=%~1"
    for %%F in ("%~1") do echo MODEL: %%~nF
)
if not "%~2"=="" set "DEEDUB_LLM_TEMP=%~2"
if not "%~4"=="" set "DEEDUB_LLM_PROMPT=%~4"
if "%~3"=="1" set DEEDUB_KEEP_TTS_WAV=1

echo VIDEO : test/test_TTS_dubbing.mp4
echo.

if "%~3"=="1" (
    cargo run --bin test-pipeline --release -- --video test/test_TTS_dubbing.mp4 --dub
) else (
    cargo run --bin test-pipeline --release -- --video test/test_TTS_dubbing.mp4
)

if errorlevel 1 (
    echo PIPELINE FAILED
    exit /b 1
)
echo PIPELINE OK
endlocal