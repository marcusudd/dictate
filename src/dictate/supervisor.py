import subprocess
import sys
import threading
from collections.abc import Callable

# mode -> (menu label, shown next to the mic on macOS when on)
MODES = {
    "auto": ("Auto (svenska + engelska)", ""),
    "sv": ("Svenska", " SV"),
    "en": ("English (live)", " EN"),
}


class Supervisor:
    """Owns the worker process. The tray views only draw what's in here."""

    def __init__(self, on_change: Callable[[], None]) -> None:
        self.on_change = on_change
        self.mode = "auto"
        self.streaming = False
        self.error: str | None = None
        self.worker: subprocess.Popen | None = None
        self.ready = threading.Event()
        self.recording = False
        # on Windows poll() runs on a background thread while menu clicks
        # call start()/stop() on another, so these must not interleave
        self.lock = threading.Lock()

    @property
    def state(self) -> str:
        if self.worker is None:
            return "error" if self.error else "off"
        if self.recording:
            return "recording"
        return "on" if self.ready.is_set() else "loading"

    def toggle(self) -> None:
        if self.worker is None:
            self.start()
        else:
            self.stop()

    def start(self) -> None:
        with self.lock:
            self._start()
        self.on_change()

    def _start(self) -> None:
        self.error = None
        self.recording = False
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
            # Windows would otherwise decode with the ANSI code page
            encoding="utf-8",
            errors="replace",
        )
        threading.Thread(
            target=self.read, args=(self.worker, self.ready), daemon=True
        ).start()

    def read(self, worker: subprocess.Popen, ready: threading.Event) -> None:
        # everything else on stdout is thrown away — r2t2_asr prints the
        # transcribed text there
        for line in worker.stdout:
            line = line.strip()
            if line == "READY":
                ready.set()
            elif line in ("REC", "IDLE") and worker is self.worker:
                self.recording = line == "REC"
            elif line.startswith("ERROR ") and worker is self.worker:
                self.error = line.removeprefix("ERROR ")
            else:
                continue
            self.on_change()

    def stop(self) -> None:
        with self.lock:
            self._stop()
        self.on_change()

    def _stop(self) -> None:
        if self.worker is None:
            return
        # killing the process is what hands the ~4.5 GB back
        self.worker.terminate()
        try:
            self.worker.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.worker.kill()
        self.worker = None
        self.recording = False

    def set_mode(self, mode: str) -> None:
        with self.lock:
            if mode == self.mode:
                return
            self.mode = mode
            # each mode is a different model, so switching means a fresh worker
            if self.worker is not None:
                self._stop()
                self._start()
        self.on_change()

    def set_streaming(self, on: bool) -> None:
        with self.lock:
            self.streaming = on
            # same model either way, so the worker just gets told — no reload
            if self.worker is not None:
                self.worker.stdin.write(f"STREAM {'ON' if on else 'OFF'}\n")
                self.worker.stdin.flush()
        self.on_change()

    def poll(self) -> None:
        """Call from a timer: notices a worker that died on its own."""
        with self.lock:
            worker = self.worker
            if worker is None or worker.poll() is None:
                return
            self.worker = None
            self.recording = False
            # stop() clears self.worker before this can see it, so an exit
            # here was unexpected. A crash without an ERROR line (OOM, a
            # failed download) would otherwise just turn the icon grey.
            if worker.returncode != 0 and self.error is None:
                self.error = (
                    f"worker stopped (exit {worker.returncode}), see the terminal"
                )
        self.on_change()
