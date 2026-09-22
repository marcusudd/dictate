# dictate

Hold a key, talk, and the text lands wherever the cursor is. Swedish and
English. Everything runs on your own machine — no server, no account,
nothing leaves the computer.

Built on a Mac, being ported to Linux and Windows. Here's where that stands:

| | Status |
|---|---|
| macOS, Apple Silicon (MPS) | Used daily. All modes, streaming, menu bar. |
| Linux, CPU, X11 | Installs and transcribes. Tray icon only tried under Xvfb. |
| Linux, NVIDIA (CUDA) | Should work, untested. |
| Windows, NVIDIA (CUDA) | Written, never run. |
| Wayland | Not supported — see Permissions. |

If you try it on something in the lower rows, please open an issue with what
happened (OS, GPU, Python, and the terminal output), or send a PR.

## What you need

- Python 3.12 and [uv](https://docs.astral.sh/uv/)
- RAM for the model: 3–6 GB, see Modes
- A GPU is what makes it usable. Apple Silicon or an NVIDIA card. It runs on
  CPU too, but expect several seconds per sentence.

### macOS

Nothing extra. Three permissions are asked for on first run, see below.

### Linux

An X11 session (or XWayland, see Permissions), and these packages:

```
# Debian / Ubuntu
sudo apt install libportaudio2 python3-gi gir1.2-ayatanaappindicator3-0.1
# Fedora
sudo dnf install portaudio python3-gobject libayatana-appindicator-gtk3
```

PyGObject has no wheels, so the venv has to see the system copy:

```
uv venv --system-site-packages
uv sync
```

GNOME hides tray icons by default; install the AppIndicator shell extension.
KDE and most others show them out of the box.

NVIDIA: driver 580 or newer (CUDA 13). GTX 10xx and older aren't supported
by CUDA 13 — see Settings for the cu126 switch.

### Windows

An NVIDIA driver 580 or newer if you have an NVIDIA card. `uv sync` fetches the
CUDA build of PyTorch from pytorch.org (PyPI's Windows build is CPU-only).
Nothing else.

## Run

```
git clone https://github.com/marcusudd/dictate
cd dictate
uv sync            # Linux: after `uv venv --system-site-packages`
uv run dictate
```

An icon shows up in the menu bar / tray. Click it → **Diktering** to load the
model (6–9 s on a Mac, longer the first time while it downloads). When it
shows a mic (macOS) or turns green (Linux/Windows) you're live: hold the
hotkey while you talk, release when you're done.

The hotkey is **right Option** on a Mac and **right Ctrl** elsewhere (right Alt
is AltGr on Swedish PC keyboards). `DICTATE_HOTKEY` changes it.

While it's actually recording the icon goes red. That's not tied to the key:
it lights when the first real samples arrive from the microphone, about 0.1 s
after the key goes down. If it never goes red, the mic is delivering silence —
usually a missing permission.

Click **Diktering** again to turn it off. The model lives in its own process
and turning it off kills that process — that's what gives the RAM back.

`uv run dictate --on` loads the model right away. A shell function like this
makes it a background service:

```zsh
dictate() {
  case "$1" in
    start) pgrep -f "dictate --on" >/dev/null ||
           (nohup uv run --project ~/projects/dictate dictate --on >>~/dictate.log 2>&1 &) ;;
    stop)  pkill -f "dictate --on" ;;
  esac
}
```

Models download to `~/.cache/huggingface` the first time a mode is used.

## Modes

Picked in the menu. Only the selected model is loaded; switching while
dictation is on restarts the worker with the other model.

| Mode | Model | Text appears | Word errors sv / en | RAM |
|---|---|---|---|---|
| **Auto** (default) | Whisper large-v3-turbo | on release, ~0.4 s | 10 % / 5 % | 2.7 GB |
| **Svenska** | KB-Whisper large | on release, ~1 s | 5 % / – | 3.7 GB |
| **English (live)** | Confucius4-R2T2 | while you talk | – / 4 % | 5.2 GB |

Auto decides between Swedish and English for each utterance. Error rates are
from 25 + 25 read-aloud FLEURS clips on an M4 Max, so treat them as a ranking,
not a promise. R2T2 gets ~30 % of Swedish words wrong, which is why it only
does English here. RAM is for MPS/CUDA in half precision; CPU runs in float32
and needs roughly double.

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

**macOS** asks for three things, all under System Settings → Privacy & Security:

- **Accessibility** — to type the text
- **Input Monitoring** — to see the hotkey
- **Microphone**

They're granted to whatever app launched `uv run dictate` (Terminal, VS Code,
iTerm...), not to dictate itself. Without Accessibility the hotkey silently
does nothing — the worker logs "This process is not trusted!".

**Linux** needs an X server: the hotkey listener and the typing go through X11.
Under Wayland only apps running through XWayland can be typed into, and the
hotkey is only seen while such an app has focus. A proper Wayland path
(evdev for the key, ydotool for typing) would be a welcome PR.

**Windows** needs nothing, except that a normal (non-admin) dictate can't type
into windows running as administrator.

## Settings

Environment variables, all optional:

| Variable | Default | |
|---|---|---|
| `DICTATE_DEVICE` | `auto` | `cuda`, `mps` or `cpu`. auto tries them in that order |
| `DICTATE_HOTKEY` | `alt_r` on macOS, `ctrl_r` elsewhere | a [pynput key name](https://pynput.readthedocs.io/en/latest/keyboard.html#pynput.keyboard.Key) |
| `DICTATE_AUTO_MODEL` | `openai/whisper-large-v3-turbo` | any Whisper checkpoint |
| `DICTATE_SV_MODEL` | `KBLab/kb-whisper-large` | any Whisper checkpoint |
| `DICTATE_EN_MODEL` | `netease-youdao/Confucius4-R2T2` | any Qwen3-ASR checkpoint |
| `DICTATE_STREAM_INTERVAL_SEC` | `1.0` | Whisper streaming pass interval. 0.5 types sooner, more errors |
| `DICTATE_CHUNK_SEC` | `0.32` | R2T2 chunk size. 0.16 doesn't keep up on an M4 Max |
| `DICTATE_MAX_NEW_TOKENS` | `32` | R2T2 token cap per chunk |

### Swapping models

The three slots each expect an architecture, and the worker checks the
model's config before downloading anything. Put the wrong kind in and the
menu shows the error instead of loading.

- **Auto** and **Svenska** take any Whisper model (Hugging Face id or local
  path). Smaller ones that work: `KBLab/kb-whisper-medium` (1.5 GB, a bit
  worse on Swedish), `openai/whisper-large-v3` (slower, slightly better).
- **English** takes any Qwen3-ASR model. `Qwen/Qwen3-ASR-1.7B` also works
  but was worse on English than R2T2 in the benchmarks.

### Pascal / older NVIDIA cards

CUDA 13 dropped GTX 10xx (Pascal) and older. In `pyproject.toml`, change
`cu130` to `cu126` in both the `[tool.uv.sources]` and `[[tool.uv.index]]`
entries, then `uv lock && uv sync`. On Linux you'll also need that index, since
PyPI's torch is CUDA 13 too — add `sys_platform == 'linux'` to the marker.

## Troubleshooting

- **Icon never leaves "loading"** — look at the terminal. The worker logs which
  device it picked (`device cuda dtype bfloat16`) and any Python error.
- **`device cpu` on a machine with a GPU** — CUDA: check `nvidia-smi` and the
  driver version. MPS: `uv run python -c "import torch; print(torch.backends.mps.is_available())"`.
- **Nothing typed, icon went red** — Accessibility permission (macOS), or
  you're in a native Wayland window (Linux).
- **`PortAudioError`** on Linux — `libportaudio2` isn't installed.
- **Tray icon missing** on GNOME — install the AppIndicator extension.

Errors show up in the menu, not just the terminal: on macOS the menu bar icon
turns into ⚠️ and the error is a row in the menu; on Linux and Windows the dot
turns red and the error is a row in the menu. On Windows the tray icon also has
a hover tooltip with the current mode or the error — AppIndicator on Linux
doesn't support one, so the menu is the only place to see it there.

## How it works

- `app.py` — picks the tray implementation for the platform.
- `config.py` — the `DICTATE_*` settings, see Settings.
- `menubar.py` (rumps, macOS) and `tray.py` (pystray, elsewhere) — thin
  views over `supervisor.py`, which starts and kills the worker and reads
  its status lines.
- `worker.py` — hotkey, microphone, and typing. In English mode text is typed
  as it settles and never revised while you talk; the tail gets flushed when
  you release. The Whisper modes record until release, then type — or, with
  streaming on, re-transcribe once a second and type what's stable.
- `sources.py` — the microphone as 16 kHz blocks, resampling if the OS
  won't open the device at that rate.
- `device.py` — cuda → mps → cpu, and which dtype to load in.
- `models.py` — checks a slot's model is the right architecture.
- `whisper.py` — runs Whisper directly, one 30 s window at a time, instead of
  through the transformers pipeline. The pipeline's long-form loop never
  exits when the model returns nothing, which KB-Whisper does on noise — one
  slip on the key would freeze dictation. Also drops low-confidence output,
  since Whisper invents a "Thank you." for silence.
- `asr.py` — loads R2T2. R2T2's streaming code only supports vLLM, which
  doesn't run on a Mac, so this swaps in the transformers model behind the
  one vLLM call the streaming code makes.
- `r2t2_asr.py` — copied unchanged from the
  [Confucius4-R2T2 repo](https://github.com/netease-youdao/Confucius4-R2T2).
  The package there depends on vLLM, so it can't be installed here as-is.

## Benchmarks

`bench/` has what the numbers above come from. `fetch.py` downloads 25 Swedish
and 25 English FLEURS clips (needs `uv run --with pyarrow`), `bench.py` and
`stream_bench.py` measure word error rate and timing, and the raw per-clip
output is in `bench/results/`. The `check_*.py` scripts run the worker and the
menu with a fake keyboard, so changes can be tried without anything being typed
into whatever app has focus. All of them run from inside `bench/`.

## Licences

- This code: MIT, see `LICENSE`.
- `src/dictate/r2t2_asr.py`: Apache-2.0, NetEase Youdao, see `LICENSE-r2t2`.
- The models are downloaded from Hugging Face on first use and aren't part of
  this repo. Whisper large-v3-turbo is MIT, KB-Whisper is Apache-2.0, and the
  Confucius4-R2T2 weights are under NetEase's
  [model licence](https://github.com/netease-youdao/Confucius4-R2T2/blob/master/MODEL_LICENSE),
  which you accept by downloading them.
