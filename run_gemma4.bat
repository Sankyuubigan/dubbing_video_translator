@echo off
call "D:\Programs\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
set RUSTC_WRAPPER=
cd /d "%~dp0src-tauri"
cargo run --bin test-pipeline --release --config "rustc-wrapper = ''" -- --video test/for_test.mp4
pause
