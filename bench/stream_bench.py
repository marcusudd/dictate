# Simulates dictate's Whisper streaming on the FLEURS clips, using the same
# word-agreement functions as the worker. Run from this folder, after fetch.py.
# The clock is simulated from measured compute time, so a pass that takes 0.4 s
# really delays the next one by 0.4 s, like it would live.
# usage: stream_bench.py <key> <model> <langs> <interval_sec>
import json
import re
import sys
import time

import numpy as np
import soundfile as sf

from dictate import device as device_mod
from dictate.whisper import Whisper
from dictate.worker import same_word, untyped_start

key, model_id, langs, interval = sys.argv[1], sys.argv[2], sys.argv[3].split(","), float(sys.argv[4])
SR = 16000
manifest = json.load(open("data/manifest.json"))
if langs == ["sv"]:
    manifest = [m for m in manifest if m["lang"] == "sv_se"]


def norm_word(w: str) -> str:
    return re.sub(r"[^\w]", "", w.lower())


def norm(text: str) -> list[str]:
    return re.sub(r"[^\w\s]", " ", text.lower().replace("-", " ")).split()


def edits(ref: list[str], hyp: list[str]) -> int:
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
    return d[-1]


dev, dtype = device_mod.pick("auto")
whisper = Whisper(model_id, langs, dev, dtype)
rows = []
for m in manifest:
    wav, _ = sf.read(m["path"], dtype="float32")
    dur = len(wav) / SR
    committed: list[str] = []
    commit_time: list[float] = []
    prev: list[str] = []
    t = interval  # audio clock at the start of each pass
    passes = []
    while t < dur:
        t0 = time.perf_counter()
        hyp = whisper.transcribe(wav[: int(t * SR)]).split()
        d = time.perf_counter() - t0
        passes.append(d)
        untyped = hyp[untyped_start(hyp, committed):]
        agreed = 0
        while agreed < min(len(untyped), len(prev)) and same_word(untyped[agreed], prev[agreed]):
            agreed += 1
        for w in untyped[:agreed]:
            committed.append(w)
            commit_time.append(t + d)
        prev = untyped[agreed:]
        t = max(t + interval, t + d)
    n_before_release = len(committed)
    t0 = time.perf_counter()
    final = whisper.transcribe(wav).split()
    wait = time.perf_counter() - t0
    text = committed + final[untyped_start(final, committed):]
    ref = norm(m["ref"])
    rows.append({
        "lang": m["lang"], "err": edits(ref, norm(" ".join(text))), "n": len(ref),
        "err_offline": edits(ref, norm(" ".join(final))),
        "first_text": commit_time[0] if commit_time else None,
        "before_release": n_before_release / max(1, len(text)),
        "pass_s": passes, "wait": wait, "dur": dur,
    })

json.dump(rows, open(f"results/stream-{key}-{interval}.json", "w"))
for lang in sorted({r["lang"] for r in rows}):
    rs = [r for r in rows if r["lang"] == lang]
    wer = 100 * sum(r["err"] for r in rs) / sum(r["n"] for r in rs)
    wer_off = 100 * sum(r["err_offline"] for r in rs) / sum(r["n"] for r in rs)
    ps = np.concatenate([r["pass_s"] for r in rs])
    print(f"{key} every {interval}s {lang}: WER {wer:.1f}% (on-release {wer_off:.1f}%)  "
          f"first text after {np.median([r['first_text'] for r in rs if r['first_text']]):.1f}s  "
          f"typed before release {100 * np.mean([r['before_release'] for r in rs]):.0f}%  "
          f"pass median {np.median(ps):.2f}s max {ps.max():.2f}s  wait after release {np.median([r['wait'] for r in rs]):.2f}s",
          flush=True)
