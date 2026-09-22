# Word error rate + timing on the FLEURS clips. Run from this folder, after fetch.py.
# usage: bench.py <key> <hf model id> <qwen-stream|qwen-offline|whisper> [forced]
#        bench.py <key> <hf model id> direct <langs, e.g. sv,en>
# direct = dictate's own Whisper wrapper, whisper = the transformers pipeline,
# forced = tell the model the language instead of letting it detect
import json
import re
import sys
import time

import numpy as np
import soundfile as sf
import torch

from dictate import device as device_mod

key, model_id, mode = sys.argv[1:4]
forced = len(sys.argv) > 4 and mode != "direct"
LANG = {"sv_se": ("Swedish", "swedish"), "en_us": ("English", "english")}
CHUNK = 0.32

manifest = json.load(open("data/manifest.json"))
dev, dtype = device_mod.pick("auto")


def norm(text: str) -> list[str]:
    text = re.sub(r"[^\w\s]", " ", text.lower().replace("-", " "))
    return text.split()


def edits(ref: list[str], hyp: list[str]) -> int:
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
    return d[-1]


if mode.startswith("qwen"):
    from dictate.asr import load_model
    from dictate.config import Settings

    asr = load_model(Settings(en_model=model_id), dev, dtype)
    if mode == "qwen-offline":
        asr.model.max_new_tokens = 512
elif mode == "direct":
    from dictate.whisper import Whisper

    whisper = Whisper(model_id, sys.argv[4].split(","), dev, dtype)
else:
    from transformers import pipeline

    pipe = pipeline("automatic-speech-recognition", model=model_id, device=dev,
                    dtype=torch.float16)
    pipe(np.zeros(16000, dtype=np.float32))  # warmup

rows = []
for m in manifest:
    wav, sr = sf.read(m["path"], dtype="float32")
    lang = LANG[m["lang"]]
    chunk_ms: list[float] = []
    t0 = time.perf_counter()
    if mode == "qwen-stream":
        state = asr.init_streaming_state(language=lang[0] if forced else None,
                                         chunk_size_sec=CHUNK)
        step = int(CHUNK * sr)
        for i in range(0, len(wav), step):
            t1 = time.perf_counter()
            asr.streaming_transcribe(wav[i:i + step], state)
            chunk_ms.append((time.perf_counter() - t1) * 1000)
        hyp = asr.finish_streaming_transcribe(state)
    elif mode == "direct":
        hyp = whisper.transcribe(wav)
    elif mode == "qwen-offline":
        hyp = asr.transcribe(audio=(wav, sr), language=lang[0] if forced else None)[0].text
    else:
        kw = {"language": lang[1], "task": "transcribe"} if forced else {"task": "transcribe"}
        hyp = pipe(wav, generate_kwargs=kw)["text"]
    total = time.perf_counter() - t0
    ref_w, hyp_w = norm(m["ref"]), norm(hyp)
    rows.append({**m, "hyp": hyp, "err": edits(ref_w, hyp_w), "n": len(ref_w),
                 "total_s": total, "chunk_ms": chunk_ms})

name = f"{key}-{mode}{'-forced' if forced else ''}"
json.dump(rows, open(f"results/results-{name}.json", "w"), ensure_ascii=False, indent=1)
for lang in LANG:
    rs = [r for r in rows if r["lang"] == lang]
    wer = 100 * sum(r["err"] for r in rs) / sum(r["n"] for r in rs)
    line = f"{name:34s} {lang}  WER {wer:5.1f}%"
    if mode == "qwen-stream":
        ms = np.concatenate([r["chunk_ms"] for r in rs])
        rtf = sum(r["total_s"] for r in rs) / sum(r["sec"] for r in rs)
        late = np.mean([r["total_s"] > r["sec"] for r in rs]) * 100
        line += (f"  chunk ms median {np.median(ms):4.0f} p90 {np.percentile(ms, 90):4.0f}"
                 f"  RTF {rtf:.2f}  clips falling behind {late:3.0f}%")
    else:
        t = [r["total_s"] for r in rs]
        line += f"  wait after release: median {np.median(t):.2f}s max {max(t):.2f}s"
    print(line, flush=True)
