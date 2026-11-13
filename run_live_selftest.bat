@echo off
setlocal enableextensions
REM E2E-селфтест live-режима (шаги 2-5 PRP): поднимает live-сессию, работает
REM DURATION секунд, пишет счётчики LIVE-SELFTEST в test\last_logs.txt и
REM выходит (exit(0)). Вызов: run_live_selftest.bat [source] [target] [duration]
REM   source: twitch | url | file  (по умолчанию twitch)
REM   target: канал | ссылка | путь к файлу (по умолчанию bogur для twitch)
REM Перед запуском нужна собранная версия: build.bat.

for %%I in ("%~dp0.") do set "PROJ=%%~fI"
cd /d "%PROJ%"

set "EXE=%PROJ%\src-tauri\target\release\deedub.exe"
if not exist "%EXE%" (
    echo [ERROR] Нет %EXE% - сначала запусти build.bat
    pause
    exit /b 1
)

REM --- Голос-референс для дубляжа: 10 с из тестового ролика (шаг 4).
set "VOICE=%PROJ%\temp\live_selftest_voice.wav"
if not exist "%VOICE%" (
    if not exist "%PROJ%\test\test_TTS_dubbing.mp4" (
        echo [WARN] Нет test\test_TTS_dubbing.mp4 - селфтест пойдёт без дубляжа
    ) else (
        echo Generating voice reference...
        ffmpeg -y -hide_banner -loglevel error -ss 30 -t 10 -i "%PROJ%\test\test_TTS_dubbing.mp4" -vn -ac 1 -ar 16000 "%VOICE%"
        if errorlevel 1 (
            echo [WARN] ffmpeg не смог собрать голос - селфтест пойдёт без дубляжа
        )
    )
)

REM --- Параметры ---
set "DEEDUB_LIVE_SELFTEST=1"
if "%~1"=="" (set "SOURCE=twitch") else set "SOURCE=%~1"
if "%~2"=="" (
    if /i "%SOURCE%"=="file" (
        set "TARGET=%PROJ%\test\test_TTS_dubbing.mp4"
    ) else (
        set "TARGET=bogur"
    )
) else set "TARGET=%~2"
if "%~3"=="" (set "DURATION=120") else set "DURATION=%~3"
set "DEEDUB_LIVE_SELFTEST_SOURCE=%SOURCE%"
set "DEEDUB_LIVE_SELFTEST_TARGET=%TARGET%"
set "DEEDUB_LIVE_SELFTEST_DURATION=%DURATION%"

echo LIVE selftest: source=%SOURCE% target=%TARGET% duration=%DURATION%s
echo (результат ищи в test\last_logs.txt: строка LIVE-SELFTEST)
"%EXE%"
endlocal
