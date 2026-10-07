"""Starts Live Translator without a console window and shows a message box if startup fails."""
import os
import sys
import traceback

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

try:
    import translator
    translator.main()
except SystemExit:
    raise
except BaseException:  # noqa: BLE001
    msg = traceback.format_exc()
    with open("app.log", "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(
            0, "Live Translator could not start:\n\n" + msg[-1500:] + "\n(Details: app.log)",
            "Live Translator", 0x10)
    except Exception:  # noqa: BLE001
        pass
