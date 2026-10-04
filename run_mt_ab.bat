@echo off
setlocal enableextensions

REM ===================================================================
REM  A/B перевода на ФИКСИРОВАННОМ входе: обе модели получают один и тот
REM  список чанков (temp\mt_ab_input.json), один и тот же промпт и один и
REM  тот же production-код translation::translate_with_session.
REM  Диаризация и TTS не запускаются - иначе вход гуляет между прогонами.
REM
REM  Использование:
REM     run_mt_ab.bat ^<path-to.gguf^> [temperature] [prompt-style]
REM
REM  prompt-style: chat (дефолт) | insttrans (родной формат Index-Translate)
REM
REM  Результат каждого прогона: temp\mt_ab_out.json (+ копия в temp\mt_ab_out_<метка>.json)
REM ===================================================================

if "%~1"=="" (
    echo Usage: run_mt_ab.bat ^<path-to.gguf^> [temperature] [prompt-style: chat^|insttrans]
    exit /b 1
)

set "MT_MODEL=%~1"
if not exist "%MT_MODEL%" (
    echo [ERROR] GGUF not found: %MT_MODEL%
    exit /b 1
)

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

set "DEEDUB_LLM_MODEL=%MT_MODEL%"
if not "%~2"=="" set "DEEDUB_LLM_TEMP=%~2"
if not "%~3"=="" set "DEEDUB_LLM_PROMPT=%~3"

REM Метка прогона = имя файла модели без пути и расширения + стиль промпта,
REM иначе два формата одной модели пишут в один файл и второй затирает первый.
for %%F in ("%MT_MODEL%") do set "MT_LABEL=%%~nF"
if defined DEEDUB_LLM_PROMPT set "MT_LABEL=%MT_LABEL%_%DEEDUB_LLM_PROMPT%"

echo MODEL : %MT_MODEL%
if defined DEEDUB_LLM_TEMP (echo TEMP  : %DEEDUB_LLM_TEMP%) else (echo TEMP  : default 0.6)
if defined DEEDUB_LLM_PROMPT (echo PROMPT: %DEEDUB_LLM_PROMPT%) else (echo PROMPT: default chat)
echo INPUT : temp\mt_ab_input.json
echo.

cargo run --bin test-mt-ab --release -- --video "%PROJ%\test\test_TTS_dubbing.mp4" --chunks "%PROJ%\temp\mt_ab_input.json"

if errorlevel 1 (
    echo MT-AB FAILED
    exit /b 1
)

REM Пути к temp/ - абсолютные: мы стоим в src-tauri, а temp/ лежит в корне
if not exist "%PROJ%\temp\mt_ab_out.json" (
    echo [ERROR] test-mt-ab не создал %PROJ%\temp\mt_ab_out.json
    exit /b 1
)
copy /y "%PROJ%\temp\mt_ab_out.json" "%PROJ%\temp\mt_ab_out_%MT_LABEL%.json" >nul
echo SAVED : temp\mt_ab_out_%MT_LABEL%.json
endlocal