import sys


def main() -> None:
    if sys.platform == "darwin":
        from dictate.menubar import main as run
    else:
        from dictate.tray import main as run
    run()
