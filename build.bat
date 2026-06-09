@echo off
cd /d "%~dp0"

echo ========================================
echo  DubVidTra2 - Build and Run
echo ========================================
echo.

REM Setup MSVC environment if not already available
where cl >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [0/3] Setting up MSVC compiler environment...
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

echo [1/3] Installing frontend dependencies...
call npm install
if %ERRORLEVEL% neq 0 (
    echo ERROR: npm install failed
    pause
    exit /b 1
)

echo.
echo [2/3] Building Rust backend...
cd src-tauri
call cargo build --release
if %ERRORLEVEL% neq 0 (
    echo ERROR: Rust build failed
    pause
    exit /b 1
)

echo.
echo [3/3] Building frontend and bundling...
cd ..
call npx tauri build
if %ERRORLEVEL% neq 0 (
    echo ERROR: tauri build failed
    pause
    exit /b 1
)

echo.
echo ========================================
echo  Build successful! Starting application...
echo ========================================

start "" "src-tauri\target\release\app.exe"
