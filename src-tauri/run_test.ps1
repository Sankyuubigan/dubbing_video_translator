param(
    [string]$Video = "",
    [switch]$All,
    [switch]$AudioOnly
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$TestVideo = if ($Video) { $Video } else { Join-Path $ProjectRoot "for_test.mp4" }

Write-Host "=== DubVidTra2 Test Runner ===" -ForegroundColor Cyan
Write-Host ""

if ($AudioOnly) {
    Write-Host "--- Audio Extraction Test ---" -ForegroundColor Yellow
    cd $PSScriptRoot
    cargo test test_extract_audio -- --nocapture 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED: Audio extraction test" -ForegroundColor Red
        exit 1
    }
    Write-Host "PASSED" -ForegroundColor Green
    exit 0
}

if ($All) {
    Write-Host "--- Audio Extraction Test ---" -ForegroundColor Yellow
    cd $PSScriptRoot
    cargo test test_extract_audio -- --nocapture 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED: Audio extraction test" -ForegroundColor Red
        exit 1
    }
    Write-Host "PASSED: audio extraction" -ForegroundColor Green
    Write-Host ""
}

Write-Host "--- Build Check ---" -ForegroundColor Yellow
cd $PSScriptRoot
cargo build 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "FAILED: Build" -ForegroundColor Red
    exit 1
}
Write-Host "PASSED: build" -ForegroundColor Green
Write-Host ""

Write-Host "--- Full Pipeline (Tauri auto-test mode) ---" -ForegroundColor Yellow
Write-Host "Note: full pipeline requires Tauri GUI and may have native crashes in llama.cpp eval_chunks" -ForegroundColor DarkYellow
Write-Host "Test video: $TestVideo" -ForegroundColor White

if (Test-Path $TestVideo) {
    Write-Host "Video file exists: $TestVideo" -ForegroundColor Green
} else {
    Write-Host "Video file NOT FOUND: $TestVideo" -ForegroundColor Red
    exit 1
}

# Check config
$ConfigPath = Join-Path $env:USERPROFILE ".dubvidtra2" "config.toml"
if (Test-Path $ConfigPath) {
    Write-Host "Config found: $ConfigPath" -ForegroundColor Green
    Get-Content $ConfigPath
} else {
    Write-Host "Config NOT FOUND at $ConfigPath" -ForegroundColor Yellow
    Write-Host "Run the app first to create config, or create it manually." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=== To run the full pipeline in GUI mode: ===" -ForegroundColor Cyan
Write-Host "  cd src-tauri && cargo tauri dev" -ForegroundColor White
Write-Host ""
Write-Host "=== To run with auto-test (pipeline runs on app start): ===" -ForegroundColor Cyan
Write-Host '  $env:DUBVID_TEST_VIDEO = "path\to\video.mp4"' -ForegroundColor White
Write-Host "  cd src-tauri && cargo tauri dev" -ForegroundColor White
Write-Host ""
Write-Host "=== Run Rust unit tests: ===" -ForegroundColor Cyan
Write-Host "  cd src-tauri && cargo test" -ForegroundColor White
