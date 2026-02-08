# RunApp.ps1
# Helper script to ensure environment is ready and start the application.

$VENV_PATH = ".venv"
$PYTHON_EXE = if ($IsWindows) { "$VENV_PATH\Scripts\python.exe" } else { "$VENV_PATH/bin/python" }

if (-not (Test-Path $VENV_PATH)) {
    Write-Host "Virtual environment not found. Creating one..." -ForegroundColor Cyan
    python -m venv $VENV_PATH
}

Write-Host "Ensuring dependencies are up to date..." -ForegroundColor Cyan
& $PYTHON_EXE -m pip install -q -r requirements.txt

Write-Host "Starting the AI Pronunciation Trainer..." -ForegroundColor Green
& $PYTHON_EXE webApp.py
