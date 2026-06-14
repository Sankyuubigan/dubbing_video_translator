@echo off
cd /d "%~dp0"

echo ========================================
echo  DubVidTra2 - Run Unit Tests
echo ========================================
echo.

REM Setup MSVC environment if not already available
where cl >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [0/1] Setting up MSVC compiler environment...
    for %%P in (
        "D:\Programs\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"
        "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat"
        "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat"
        "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvarsall.bat"
        "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat"
        "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"
    ) do (
        if exist "%%~P" (
            echo Found: %%~P
            call "%%~P" x64
            goto :msvc_done
        )
    )
    echo ERROR: Visual Studio C++ tools not found! Install "Desktop development with C++" workload.
    pause
    exit /b 1
)
:msvc_done

echo [1/2] Compiling tests...
cd src-tauri

REM disable sccache wrapper from global cargo config
set RUSTC_WRAPPER=

call cargo test --lib --no-run --config "rustc-wrapper = ''"
if %ERRORLEVEL% neq 0 (
    echo ERROR: Compilation failed
    pause
    exit /b 1
)

echo.
echo [2/2] Running unit tests...
echo.

call cargo test --lib --config "rustc-wrapper = ''" -- translation --nocapture
if %ERRORLEVEL% neq 0 (
    echo.
    echo WARNING: Tests failed to run (may be DLL/env issue).
    echo Compilation succeeded though.
    echo.
    pause
    exit /b 1
)

echo.
echo ========================================
echo  All tests passed!
echo ========================================
pause
