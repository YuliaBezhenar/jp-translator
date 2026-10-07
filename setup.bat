@echo off
cd /d "%~dp0"
echo === Live Translator setup ===
where python >nul 2>nul
if errorlevel 1 (
  echo Python was not found. Install Python 3.11 or newer from python.org and tick "Add python.exe to PATH".
  pause
  exit /b 1
)
if not exist venv (
  python -m venv venv
  if errorlevel 1 (
    echo Could not create the virtual environment.
    pause
    exit /b 1
  )
)
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install "typing-extensions>=4.10.0"

pip show torch >nul 2>nul
if not errorlevel 1 (
  echo PyTorch is already installed - skipping.
  goto :deps
)
echo.
echo Which PyTorch build do you want?
echo   [1] NVIDIA GPU (CUDA 13.0) - recommended, much faster
echo   [2] CPU only - works everywhere, but is slow
choice /c 12 /n /m "Choose 1 or 2: "
if errorlevel 2 goto :cpu
pip install torch --index-url https://download.pytorch.org/whl/cu130
goto :torchdone
:cpu
pip install torch
:torchdone
if errorlevel 1 goto :failed

:deps
pip install -r requirements.txt
if errorlevel 1 goto :failed
echo.
echo Optional packages (the app also works without them):
pip install python-docx || echo python-docx failed - transcripts will be saved as TXT only.
pip install pywebview || echo pywebview failed - the app will open in an Edge window instead.

echo.
echo Creating the "Live Translator" shortcut in this folder...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$s=(New-Object -ComObject WScript.Shell).CreateShortcut('%~dp0Live Translator.lnk'); $s.TargetPath='%~dp0Live Translator.bat'; $s.WorkingDirectory='%~dp0'; $s.WindowStyle=7; $s.Description='Live Translator'; if (Test-Path '%~dp0icon.ico') { $s.IconLocation='%~dp0icon.ico' }; $s.Save()"
if errorlevel 1 echo Could not create the shortcut - you can still start the app with "Live Translator.bat".

echo.
echo Setup finished.
echo Start the app with the "Live Translator" shortcut in this folder (you can copy it to your Desktop).
echo The first start downloads the AI models (about 9 GB) - this can take a while.
pause
exit /b 0

:failed
echo.
echo Installation failed. See the messages above.
pause
exit /b 1