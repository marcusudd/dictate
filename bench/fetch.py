# Pulls 25 real-speech clips per language from FLEURS (test split) with reference text
import io
import json
import urllib.request

import pyarrow.parquet as pq
import soundfile as sf

N = 25
manifest = []
for lang in ["sv_se", "en_us"]:
    url = f"https://huggingface.co/api/datasets/google/fleurs/parquet/{lang}/test/0.parquet"
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    table = pq.read_table(io.BytesIO(urllib.request.urlopen(req).read()))
    print(lang, table.num_rows, table.column_names)
    rows = table.slice(0, N).to_pylist()
    for i, r in enumerate(rows):
        wav, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
        assert sr == 16000, sr
        path = f"data/{lang}_{i:02d}.wav"
        sf.write(path, wav, sr)
        manifest.append({"lang": lang, "path": path, "ref": r["raw_transcription"],
                         "sec": round(len(wav) / sr, 1)})
json.dump(manifest, open("data/manifest.json", "w"), ensure_ascii=False, indent=1)
for lang in ["sv_se", "en_us"]:
    secs = [m["sec"] for m in manifest if m["lang"] == lang]
    print(lang, len(secs), "clips,", round(sum(secs)), "s total, max", max(secs))
print(manifest[0]["ref"])
