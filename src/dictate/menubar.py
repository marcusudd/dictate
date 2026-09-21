import argparse
import subprocess
import sys
import threading

import rumps
from PyObjCTools import AppHelper

TITLE_OFF = "◯"
TITLE_LOADING = "⏳"
TITLE_ON = "🎙"
TITLE_RECORDING = "🔴"

# mode -> (menu label, shown next to the mic when on)
MODES = {
    "auto": ("Auto (svenska + engelska)", ""),
    "sv": ("Svenska", " SV"),
    "en": ("English (live)", " EN"),
}


class DictateApp(rumps.App):
    def __init__(self) -> None:
        super().__init__("Dictate", title=TITLE_OFF, quit_button=None)
        self.toggle_item = rumps.MenuItem("Diktering", callback=self.toggle)
        self.mode = "auto"
        self.mode_items = {
            mode: rumps.MenuItem(label, callback=self.pick_mode)
            for mode, (label, _) in MODES.items()
        }
        self.streaming = False
        self.stream_item = rumps.MenuItem(
            "Streaming (text medan du pratar)", callback=self.toggle_streaming
        )
        self.menu = [
            self.toggle_item,
            None,
            *self.mode_items.values(),
            None,
            self.stream_item,
            None,
            rumps.MenuItem("Avsluta", callback=self.quit),
        ]
        self.worker: subprocess.Popen | None = None
        self.ready = threading.Event()
        self.recording = False
        self.timer = rumps.Timer(self.refresh, 0.5)
        self.timer.start()

    def toggle(self, sender: rumps.MenuItem) -> None:
        if self.worker is None:
            self.start_worker()
        else:
            self.stop_worker()

    def pick_mode(self, sender: rumps.MenuItem) -> None:
        mode = next(m for m, item in self.mode_items.items() if item is sender)
        if mode == self.mode:
            return
        self.mode = mode
        # each mode is a different model, so switching means a fresh worker
        if self.worker is not None:
            self.stop_worker()
            self.start_worker()

    def toggle_streaming(self, sender: rumps.MenuItem) -> None:
        self.streaming = not self.streaming
        # same model either way, so the worker just gets told — no reload
        if self.worker is not None:
            self.worker.stdin.write(f"STREAM {'ON' if self.streaming else 'OFF'}\n")
            self.worker.stdin.flush()

    def start_worker(self) -> None:
        self.ready = threading.Event()
        self.worker = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "dictate.worker",
                "--mode",
                self.mode,
                *(["--stream"] if self.streaming else []),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        threading.Thread(
            target=self.read_worker, args=(self.worker, self.ready), daemon=True
        ).start()

    def read_worker(self, worker: subprocess.Popen, ready: threading.Event) -> None:
        # everything else on stdout is thrown away — r2t2_asr prints the
        # transcribed text there
        for line in worker.stdout:
            line = line.strip()
            if line == "READY":
                ready.set()
            elif line in ("REC", "IDLE") and worker is self.worker:
                self.recording = line == "REC"
                # straight to the main thread, the 0.5 s timer is too slow for this
                AppHelper.callAfter(self.refresh)

    def stop_worker(self) -> None:
        # killing the process is what hands the ~4.5 GB back
        self.worker.terminate()
        try:
            self.worker.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.worker.kill()
        self.worker = None
        self.recording = False

    def refresh(self, timer: rumps.Timer | None = None) -> None:
        if self.worker is not None and self.worker.poll() is not None:
            self.worker = None
            self.recording = False
        if self.worker is None:
            self.title = TITLE_OFF
        elif self.recording:
            self.title = TITLE_RECORDING + MODES[self.mode][1]
        elif self.ready.is_set():
            self.title = TITLE_ON + MODES[self.mode][1]
        else:
            self.title = TITLE_LOADING
        self.toggle_item.state = self.worker is not None
        for mode, item in self.mode_items.items():
            item.state = mode == self.mode
        # English is always live, the switch only means something for Whisper
        self.stream_item.state = self.streaming or self.mode == "en"
        self.stream_item.set_callback(
            None if self.mode == "en" else self.toggle_streaming
        )

    def quit(self, sender: rumps.MenuItem) -> None:
        if self.worker is not None:
            self.stop_worker()
        rumps.quit_application()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--on", action="store_true", help="turn dictation on right away"
    )
    args = parser.parse_args()

    app = DictateApp()
    if args.on:
        app.start_worker()
    app.run()
