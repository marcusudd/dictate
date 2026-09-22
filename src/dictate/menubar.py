import argparse

import rumps
from PyObjCTools import AppHelper

from dictate.supervisor import MODES, Supervisor

TITLE_OFF = "◯"
TITLE_LOADING = "⏳"
TITLE_ON = "🎙"
TITLE_RECORDING = "🔴"
TITLE_ERROR = "⚠️"


class DictateApp(rumps.App):
    def __init__(self) -> None:
        super().__init__("Dictate", title=TITLE_OFF, quit_button=None)
        # supervisor callbacks come from its reader thread; AppKit wants the main one
        self.sup = Supervisor(lambda: AppHelper.callAfter(self.refresh))
        self.toggle_item = rumps.MenuItem("Diktering", callback=self.toggle)
        self.mode_items = {
            mode: rumps.MenuItem(label, callback=self.pick_mode)
            for mode, (label, _) in MODES.items()
        }
        self.stream_item = rumps.MenuItem(
            "Streaming (text medan du pratar)", callback=self.toggle_streaming
        )
        self.error_item = rumps.MenuItem("")
        self.error_item.set_callback(None)
        self.menu = [
            self.toggle_item,
            None,
            *self.mode_items.values(),
            None,
            self.stream_item,
            None,
            self.error_item,
            rumps.MenuItem("Avsluta", callback=self.quit),
        ]
        self.timer = rumps.Timer(self.tick, 0.5)
        self.timer.start()

    def toggle(self, sender: rumps.MenuItem) -> None:
        self.sup.toggle()

    def pick_mode(self, sender: rumps.MenuItem) -> None:
        mode = next(m for m, item in self.mode_items.items() if item is sender)
        self.sup.set_mode(mode)

    def toggle_streaming(self, sender: rumps.MenuItem) -> None:
        self.sup.set_streaming(not self.sup.streaming)

    def tick(self, timer: rumps.Timer) -> None:
        self.sup.poll()
        self.refresh()

    def refresh(self) -> None:
        state = self.sup.state
        suffix = MODES[self.sup.mode][1]
        self.title = {
            "off": TITLE_OFF,
            "loading": TITLE_LOADING,
            "on": TITLE_ON + suffix,
            "recording": TITLE_RECORDING + suffix,
            "error": TITLE_ERROR,
        }[state]
        self.toggle_item.state = self.sup.worker is not None
        for mode, item in self.mode_items.items():
            item.state = mode == self.sup.mode
        # English is always live, the switch only means something for Whisper
        self.stream_item.state = self.sup.streaming or self.sup.mode == "en"
        self.stream_item.set_callback(
            None if self.sup.mode == "en" else self.toggle_streaming
        )
        self.error_item.title = self.sup.error or ""
        self.error_item.hidden = self.sup.error is None

    def quit(self, sender: rumps.MenuItem) -> None:
        self.sup.stop()
        rumps.quit_application()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--on", action="store_true", help="turn dictation on right away"
    )
    args = parser.parse_args()

    app = DictateApp()
    if args.on:
        app.sup.start()
    app.run()
