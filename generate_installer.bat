@echo off
setlocal enableextensions

set "PROJ=%~dp0"
if not exist "%PROJ%src-tauri\tauri.conf.json" (
  echo [ERROR] This script must live in the project root, next to src-tauri\tauri.conf.json
  pause
  exit /b 1
)

set "TOOLKIT=%TAURI_BUILD_TOOLKIT%"
if "%TOOLKIT%"=="" set "TOOLKIT=%~dp0..\my-tauri-plugins\tauri-build-toolkit\cli.cjs"
if not exist "%TOOLKIT%" (
  echo [ERROR] Tauri build toolkit not found: "%TOOLKIT%"
  echo Set env TAURI_BUILD_TOOLKIT to the toolkit cli.cjs, or place the
  echo toolkit folder at: my-tauri-plugins\tauri-build-toolkit
  pause
  exit /b 1
)

set "VS_INIT_OK="
for /f "usebackq delims=" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -property installationPath 2^>nul`) do (
    if exist "%%i\VC\Auxiliary\Build\vcvarsall.bat" (
        call "%%i\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul 2>&1
        set "VS_INIT_OK=1"
    )
)
if not defined VS_INIT_OK (
  echo [ERROR] Visual Studio not found. vswhere. Install VS with C++ workload.
  pause
  exit /b 1
)

set "CC="
set "CXX="
set "CMAKE_C_COMPILER_LAUNCHER="
set "CMAKE_CXX_COMPILER_LAUNCHER="
set "RUSTC_WRAPPER="
set "CARGO_BUILD_RUSTC_WRAPPER="

set "CARGO_PROFILE_RELEASE_LTO="
set "CARGO_PROFILE_RELEASE_CODEGEN_UNITS="
set "CARGO_PROFILE_RELEASE_STRIP="

node "%TOOLKIT%" installer --project "%PROJ%"
if errorlevel 1 (
  echo [ERROR] Installer build failed.
  pause
  exit /b 1
)

echo [+DONE] generate_installer.bat finished.
endlocal