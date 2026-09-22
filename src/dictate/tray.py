import argparse
import sys
import threading
import time
from collections.abc import Callable

import pystray
from PIL import Image, ImageDraw

from dictate.supervisor import MODES, Supervisor

COLORS = {
    "off": (140, 140, 140),
    "loading": (240, 190, 40),
    "on": (60, 180, 90),
    "recording": (220, 50, 50),
    "error": (220, 50, 50),
}
STREAM_LABEL = "Streaming (text medan du pratar)"


def dot(color: tuple[int, int, int]) -> Image.Image:
    image = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    ImageDraw.Draw(image).ellipse((8, 8, 56, 56), fill=color)
    return image


def gui_thread_runner() -> Callable[[Callable[[], None]], None]:
    """pystray's GTK/AppIndicator backends aren't thread-safe; Windows' is.
    On Linux, hop to the GLib main loop."""
    if sys.platform.startswith("linux"):
        try:
            from gi.repository import GLib
        except ImportError:
            raise SystemExit(
                "dictate needs PyGObject for the tray icon. Install python3-gi "
                "and gir1.2-ayatanaappindicator3-0.1 (Debian/Ubuntu) or "
                "python3-gobject and libayatana-appindicator-gtk3 (Fedora), then "
                "create the venv with `uv venv --system-site-packages` before "
                "`uv sync`."
            )
        return lambda fn: GLib.idle_add(fn)
    return lambda fn: fn()


class Tray:
    def __init__(self) -> None:
        self.run_on_gui = gui_thread_runner()
        self.sup = Supervisor(lambda: self.run_on_gui(self.refresh))
        self.icon = pystray.Icon(
            "dictate", icon=dot(COLORS["off"]), title="Dictate", menu=self.menu()
        )

    def menu(self) -> pystray.Menu:
        sup = self.sup
        return pystray.Menu(
            pystray.MenuItem(
                "Diktering",
                lambda: sup.toggle(),
                checked=lambda item: sup.worker is not None,
            ),
            pystray.Menu.SEPARATOR,
            *[
                pystray.MenuItem(
                    label,
                    lambda *, m=mode: sup.set_mode(m),
                    checked=lambda item, m=mode: sup.mode == m,
                )
                for mode, (label, _) in MODES.items()
            ],
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(
                STREAM_LABEL,
                lambda: sup.set_streaming(not sup.streaming),
                # English is always live, the switch only means something for Whisper
                checked=lambda item: sup.streaming or sup.mode == "en",
                enabled=lambda item: sup.mode != "en",
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(
                lambda item: sup.error or "",
                None,
                enabled=False,
                visible=lambda item: sup.error is not None,
            ),
            pystray.MenuItem("Avsluta", self.quit),
        )

    def refresh(self) -> None:
        state = self.sup.state
        self.icon.icon = dot(COLORS[state])
        label = MODES[self.sup.mode][0]
        # the full error is a row in the menu; the tooltip only says where to
        # look, since Windows raises ValueError on titles over 127 chars
        title = f"Dictate – {label}" if state != "error" else "Dictate – fel, se menyn"
        self.icon.title = title[:127]
        self.icon.update_menu()

    def poll_forever(self) -> None:
        # pystray has no timer, so a thread notices a worker that died. The
        # poll itself must run on the GUI thread too, so hop to it here
        # rather than calling self.sup.poll() straight from this thread.
        while True:
            time.sleep(0.5)
            self.run_on_gui(self.sup.poll)

    def quit(self) -> None:
        self.sup.stop()
        self.icon.stop()

    def run(self, on: bool) -> None:
        threading.Thread(target=self.poll_forever, daemon=True).start()
        if on:
            self.sup.start()
        self.icon.run()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--on", action="store_true", help="turn dictation on right away"
    )
    args = parser.parse_args()
    Tray().run(args.on)
