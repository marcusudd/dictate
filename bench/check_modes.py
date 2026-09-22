# usage: check_modes.py <auto|sv|en>   (run from this folder, after fetch.py)
# Drives PushToTalk through its real audio callback in 512-sample blocks, fake keyboard
import sys
import time

import numpy as np
import soundfile as sf

from dictate.config import Settings
from dictate.worker import BLOCK_SAMPLES, PushToTalk


class FakeKeys:
    def __init__(self) -> None:
        self.out, self.events = "", 0

    def type(self, text: str) -> None:
        self.out += text
        self.events += bool(text)

    def tap(self, key) -> None:
        self.out = self.out[:-1]


mode = sys.argv[1]
ptt = PushToTalk(Settings(), mode, streaming=False)
wav = sf.read("data/en_us_04.wav", dtype="int16")[0]
silent_start = np.concatenate([np.zeros(BLOCK_SAMPLES * 3, dtype=np.int16), wav])
ptt.keys = FakeKeys()
ptt.hearing = False
print(f"[{mode}] feeding 3 all-zero blocks, then speech", flush=True)
for n, i in enumerate(range(0, len(silent_start), BLOCK_SAMPLES)):
    if n == 3:
        print(f"[{mode}] (first real block goes in now; hearing so far = {ptt.hearing})", flush=True)
    ptt.on_block(silent_start[i:i + BLOCK_SAMPLES])
ptt.audio.put(None)
t0 = time.perf_counter()
ptt.transcribe_live() if mode == "en" else ptt.transcribe_on_release()
print(f"[{mode}] {time.perf_counter() - t0:.1f}s, {ptt.keys.events} type events -> {ptt.keys.out[:80]!r}", flush=True)
