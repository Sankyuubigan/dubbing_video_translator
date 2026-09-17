@echo off
cd /d "%~dp0"

call "D:\Programs\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
if %ERRORLEVEL% neq 0 exit /b 1

set PATH=C:\Users\user\.cargo\bin;C:\Users\user\.rustup\toolchains\stable-x86_64-pc-windows-msvc\bin;%PATH%
set RUSTC_WRAPPER=
set CC=cl
set CXX=cl

cd src-tauri

REM Keep raw/stretched/final WAV for ASR audit
set DUBVID_KEEP_TTS_WAV=1

echo Running pipeline...
cargo run --bin test-pipeline --release --config "rustc-wrapper = ''" -- --video test/test_TTS_dubbing.mp4 --dub

if %ERRORLEVEL% neq 0 (
    echo PIPELINE FAILED
    pause
    exit /b 1
)
echo PIPELINE OK