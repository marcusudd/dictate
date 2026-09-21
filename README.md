# dictate

Hold right Option, talk, and the text lands wherever the cursor is. Everything
runs locally on the Mac GPU (MPS) — no server, nothing leaves the machine.

## Run

```
uv run dictate
```

A `◯` shows up in the menu bar. Click it → **Diktering** to load the model
(`⏳`, 6–9 s). When it turns into `🎙` you're live: hold right Option while you
talk, release when you're done.

The mic turns into `🔴` while it's actually recording. That's not tied to the
key press: it lights when the first real samples come in from the microphone
(about 0.1 s after the key goes down), so if it's red, what you say is being
captured. If it never turns red, the mic is delivering silence — usually a
missing Microphone permission.

Click **Diktering** again to turn it off. The model lives in its own process,
and turning it off kills that process — that's what gives the RAM back. The
menu bar app itself is tiny.

`uv run dictate --on` skips the click and loads the model right away. I have
a `dictate` shell function in `~/.zshrc` built on that: `dictate start` runs
it in the background with dictation on (log in `~/Library/Logs/dictate.log`),
and `dictate stop` quits the whole thing.

Models are downloaded to `~/.cache/huggingface` the first time a mode is used.

## Modes

Picked in the menu. Only the selected model is loaded; switching while
dictation is on restarts the worker with the other model.

| Mode | Model | Text appears | Word errors sv / en | RAM |
|---|---|---|---|---|
| **Auto** (default) | Whisper large-v3-turbo | on release, ~0.4 s | 10 % / 5 % | 2.7 GB |
| **Svenska** `🎙 SV` | KB-Whisper large | on release, ~1 s | 5 % / – | 3.7 GB |
| **English (live)** `🎙 EN` | Confucius4-R2T2 | while you talk | – / 4 % | 5.2 GB |

Auto decides between Swedish and English for each utterance. Error rates are
from 25 + 25 read-aloud FLEURS clips on an M4 Max, so treat them as a ranking,
not a promise. R2T2 gets ~30 % of Swedish words wrong, which is why it only
does English here.

### Streaming

**Streaming (text medan du pratar)** in the menu switches the two Whisper modes
from typing on release to typing while you talk. No reload — it applies from
the next time you press the key. English is always live, so it's greyed out
there.

Whisper can't actually stream, so this re-transcribes everything said so far
once a second and types the words two passes in a row agree on. Typed words are
never taken back. What that costs, same clips as above:

| | On release | Streaming |
|---|---|---|
| Auto, English | 4.8 % | 5.2 % |
| Auto, Swedish | 10.4 % | 11.8 % |
| Svenska | 5.1 % | 6.4 % |

The first words show up after ~3.5 s, then it trails you by 1–2 s. So it's for
longer stretches of text; for a sentence or two, off is both faster and more
accurate.

Speak the wrong language into a single-language mode and it translates:
Svenska turns English into Swedish, English turns Swedish into English.

## Permissions

macOS asks for three things, all under System Settings → Privacy & Security:

- **Accessibility** — to type the text
- **Input Monitoring** — to see the right Option key
- **Microphone**

They're granted to whatever app launched `uv run dictate` (Terminal, VS Code,
iTerm...), not to dictate itself. Without Accessibility the hotkey silently
does nothing — the worker logs "This process is not trusted!".

## Settings

Environment variables, all optional:

| Variable | Default | |
|---|---|---|
| `DICTATE_AUTO_MODEL` | `openai/whisper-large-v3-turbo` | HF repo id or local path |
| `DICTATE_SV_MODEL` | `KBLab/kb-whisper-large` | |
| `DICTATE_EN_MODEL` | `netease-youdao/Confucius4-R2T2` | |
| `DICTATE_STREAM_INTERVAL_SEC` | `1.0` | Whisper streaming pass interval. 0.5 types sooner, more errors |
| `DICTATE_CHUNK_SEC` | `0.32` | R2T2 chunk size. 0.16 doesn't keep up on an M4 Max |
| `DICTATE_MAX_NEW_TOKENS` | `32` | R2T2 token cap per chunk |

## How it works

- `menubar.py` — the menu bar app. Starts and kills the worker.
- `worker.py` — hotkey, microphone, and typing. In English mode text is typed
  as it settles and never revised while you talk; the tail gets flushed when
  you release. The Whisper modes record until release, then type — or, with streaming
  on, re-transcribe once a second and type what's stable.
- `whisper.py` — runs Whisper directly, one 30 s window at a time, instead of
  through the transformers pipeline. The pipeline's long-form loop never
  exits when the model returns nothing, which KB-Whisper does on noise — one
  slip on the key would freeze dictation. Also drops low-confidence output,
  since Whisper invents a "Thank you." for silence.
- `asr.py` — loads R2T2 on MPS. R2T2's streaming code only supports vLLM,
  which doesn't run on a Mac, so this swaps in the transformers model behind
  the one vLLM call the streaming code makes.
- `r2t2_asr.py` — copied unchanged from the
  [Confucius4-R2T2 repo](https://github.com/netease-youdao/Confucius4-R2T2)
  (Apache-2.0). The package there depends on vLLM, so it can't be installed
  here as-is.
  Its license is in `LICENSE-r2t2`.

## Benchmarks

`bench/` has what the numbers above come from. `fetch.py` downloads 25 Swedish
and 25 English FLEURS clips (needs `uv run --with pyarrow`), `bench.py` and
`stream_bench.py` measure word error rate and timing, and the raw per-clip
output is in `bench/results/`. The `check_*.py` scripts run the worker and the
menu with a fake keyboard, so changes can be tried without anything being typed
into whatever app has focus. All of them run from inside `bench/`.
