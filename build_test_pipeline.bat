@echo off
cd /d "%~dp0"

call "D:\Programs\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
if %ERRORLEVEL% neq 0 exit /b 1

set PATH=C:\Users\user\.cargo\bin;C:\Users\user\.rustup\toolchains\stable-x86_64-pc-windows-msvc\bin;%PATH%
set RUSTC_WRAPPER=
set CC=cl
set CXX=cl

cd src-tauri

echo Building test-pipeline...
cargo build --bin test-pipeline --release --config "rustc-wrapper = ''"

if %ERRORLEVEL% neq 0 (
    echo BUILD FAILED
    pause
    exit /b 1
)
echo BUILD OK