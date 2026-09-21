# Clicks through the menu bar logic against fake_worker.py — no model, no typing.
# Run from this folder.
import subprocess
import sys

import rumps

import dictate.menubar as mb

real_popen = subprocess.Popen
mb.subprocess.Popen = lambda cmd, **kw: real_popen([sys.executable, "fake_worker.py", *cmd[3:]], **kw)
app = mb.DictateApp()
step = 0


def tick(timer):
    global step
    step += 1
    if step == 2:
        app.start_worker()
    elif step == 3:
        app.toggle_streaming(app.stream_item)   # on, worker running -> stdin
    elif step == 4:
        print("auto: stream item checked", bool(app.stream_item.state), flush=True)
        app.pick_mode(app.mode_items["sv"])      # restart must keep --stream
    elif step == 5:
        app.toggle_streaming(app.stream_item)   # off
    elif step == 6:
        app.pick_mode(app.mode_items["en"])
    elif step == 7:
        print("en: stream item checked", bool(app.stream_item.state), "clickable",
              app.stream_item.callback is not None, flush=True)
        app.pick_mode(app.mode_items["auto"])
    elif step == 8:
        print("back to auto: checked", bool(app.stream_item.state), "clickable",
              app.stream_item.callback is not None, flush=True)
        app.quit(None)


rumps.Timer(tick, 0.8).start()
app.run()
