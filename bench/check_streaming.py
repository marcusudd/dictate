# usage: check_streaming.py <auto|sv>   (run from this folder, after fetch.py)
# Feeds audio through the real callback at real-time pace; fake keyboard logs when text lands
import io
import sys
import threading
import time

import numpy as np
import soundfile as sf

import dictate.worker as worker
from dictate.config import Settings
from dictate.worker import BLOCK_SAMPLES, PushToTalk

mode = sys.argv[1]
read = lambda p: sf.read(p, dtype="int16")[0]
clips = {"12s sv": read("data/sv_se_01.wav"),
         "50s sv": np.concatenate([read(f"data/sv_se_{i:02d}.wav") for i in (0, 3, 4, 5)])}
if mode == "auto":
    clips["en"] = read("data/en_us_04.wav")

ptt = PushToTalk(Settings(), mode, streaming=False)

# the menu bar's STREAM ON line, without a real parent process
exits = []
worker.os._exit = exits.append
sys.stdin = io.StringIO("STREAM ON\n")
worker.listen_to_parent(ptt)
print(f"[{mode}] after 'STREAM ON' on stdin: streaming={ptt.streaming}, exit-on-EOF called={exits == [0]}", flush=True)


class TimedKeys:
    def __init__(self, t0: float) -> None:
        self.t0, self.log, self.out = t0, [], ""

    def type(self, text: str) -> None:
        self.out += text
        if text.strip():
            self.log.append((round(time.monotonic() - self.t0, 1), text.strip()))


for name, wav in clips.items():
    dur = len(wav) / 16000
    t0 = time.monotonic()
    ptt.keys = TimedKeys(t0)

    def feed() -> None:
        for n, i in enumerate(range(0, len(wav), BLOCK_SAMPLES)):
            time.sleep(max(0, t0 + (n + 1) * BLOCK_SAMPLES / 16000 - time.monotonic()))
            ptt.on_audio(wav[i:i + BLOCK_SAMPLES].reshape(-1, 1), BLOCK_SAMPLES, None, None)
        ptt.audio.put(None)

    threading.Thread(target=feed).start()
    ptt.transcribe_streaming()
    end = time.monotonic() - t0
    log = ptt.keys.log
    before = sum(len(t.split()) for ts, t in log if ts <= dur)
    total = len(ptt.keys.out.split())
    print(f"[{mode}] {name} ({dur:.0f}s): first text at {log[0][0]}s, {len(log)} bursts, "
          f"{before}/{total} words typed before release, done {end - dur:.1f}s after release", flush=True)
    print(f"    bursts: {[(ts, t[:28]) for ts, t in log[:5]]} ...", flush=True)
    print(f"    text: {ptt.keys.out[:150]!r}", flush=True)
