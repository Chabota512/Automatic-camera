$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "Python 3.11+ is required. Install Python from python.org and rerun this script."
}

$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Host "Creating the project virtual environment..."
    python -m venv .venv
}

Write-Host "Installing Python dependencies..."
& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -r requirements.txt pyinstaller

Write-Host "Preparing the application icon..."
& $VenvPython assets\create_icon.py

Write-Host "Building dist\AutomaticCamera.exe..."
$PyInstallerArgs = @(
    "--noconfirm",
    "--clean",
    "--onefile",
    "--windowed",
    "--name", "AutomaticCamera",
    "--icon", "assets\automatic-camera.ico",
    "--add-data", "assets;assets",
    "windows_camera_app.py"
)
& $VenvPython -m PyInstaller @PyInstallerArgs

Write-Host ""
Write-Host "Build complete: $ProjectRoot\dist\AutomaticCamera.exe" -ForegroundColor Green
Write-Host "Run the source application with: .\.venv\Scripts\python.exe windows_camera_app.py"