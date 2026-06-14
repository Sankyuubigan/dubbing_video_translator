@echo off
cd /d "%~dp0"

REM Setup MSVC
call "D:\Programs\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
if %ERRORLEVEL% neq 0 exit /b 1

REM Ensure Rust/Cargo on PATH (vcvarsall may reset it)
set PATH=C:\Users\user\.cargo\bin;C:\Users\user\.rustup\toolchains\stable-x86_64-pc-windows-msvc\bin;%PATH%

REM Disable sccache
set RUSTC_WRAPPER=
set CC=cl
set CXX=cl

cd src-tauri

echo Building and running pipeline...
cargo run --bin test-pipeline --release --config "rustc-wrapper = ''" -- --video test/for_test.mp4

if %ERRORLEVEL% neq 0 (
    echo ERROR: Pipeline failed
    pause
    exit /b 1
)

pause
