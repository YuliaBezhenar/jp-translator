# Live Translator

Local, offline **Japanese <-> English live speech translator** with an interface.
Speech recognition and translation run on your own computer using Liquid AI's LFM2.5 models.

## Requirements
- Windows 10/11, a microphone
- Python 3.11 or newer (tested with 3.14) - tick "Add python.exe to PATH" when installing
- An NVIDIA GPU with 8 GB+ VRAM is strongly recommended (CPU-only works, but slowly)
- About 9 GB of free disk space for the models, and internet access on the first start

## Install
1. Download or clone this repository.
2. Double-click **`setup.bat`** - it creates a virtual environment, installs everything and
   creates a **`Live Translator`** shortcut (with the app icon) in this folder.
3. Start the app by double-clicking that shortcut (or `Live Translator.bat`). You can copy the
   shortcut to your Desktop. If you move the project folder, run `setup.bat` again to refresh it.

The first start downloads the models from Hugging Face (~9 GB) into your user cache
(`C:\Users\<you>\.cache\huggingface`). The window shows "Loading ..." until they are ready.

## Run from source
```
venv\Scripts\activate
python translator.py
```
(this shows the console output, e.g. timings and GPU memory)

## Troubleshooting
- **Nothing happens / error box on start** - details are in `app.log`.
- **`WinError 4551 ... Application Control policy has blocked this file`** - Windows Smart App Control
  blocks PyTorch DLLs. Check Windows Security -> App & browser control -> Smart App Control.
- **Slow translation** - make sure PyTorch uses your GPU (`python -c "import torch; print(torch.cuda.is_available())"`).
  8 GB of VRAM is enough because only one speech model is kept on the GPU at a time.
- **Text is not recognised** - adjust the volume threshold in the settings (gear icon) and speak closer to the microphone.

## Models and licences
Models are **not** included in this repository; they are downloaded from Hugging Face on first run.

Models used:
- [LFM2.5-1.2B-JP](https://docs.liquid.ai/lfm/models/lfm25-1.2b-jp#transformers) - translation
- [LFM2.5-Audio-1.5B-JP](https://docs.liquid.ai/lfm/models/lfm25-audio-1.5b-jp) - Japanese speech recognition
- [LFM2.5-Audio-1.5B](https://docs.liquid.ai/lfm/models/lfm25-audio-1.5b) - English speech recognition
- [Whisper base](https://huggingface.co/openai/whisper-base) - spoken-language detection

Check each model's licence before any commercial use (LFM models use the LFM Open License).