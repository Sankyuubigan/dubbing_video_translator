@echo off
REM Unit-tests in RELEASE profile.
REM Needed on machines where the debug test harness fails to start with
REM 0xc0000139 STATUS_ENTRYPOINT_NOT_FOUND (desktop rules §6.2, AGENTS.md).
REM Everything else is identical to test.bat — only the cargo profile differs.
setlocal
call "%~dp0test.bat" --release %*
exit /b %ERRORLEVEL%
