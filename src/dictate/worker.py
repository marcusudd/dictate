import argparse
import os
import queue
import re
import sys
import threading
import time

import numpy as np
import sounddevice as sd
from pynput import keyboard

from dictate import device as device_mod
from dictate import models
from dictate.asr import SAMPLE_RATE, load_model
from dictate.config import Settings
from dictate.whisper import Whisper

HOTKEY = keyboard.Key.alt_r
MODES = ["auto", "sv", "en"]
# shorter than this is a slip on the key, not speech
MIN_UTTERANCE_SEC = 0.4
# small blocks so the first one shows up ~80 ms after the key goes down
# instead of after a full 320 ms chunk
BLOCK_SAMPLES = 512


class PushToTalk:
    def __init__(self, settings: Settings, mode: str, streaming: bool) -> None:
        self.settings = settings
        self.mode = mode
        # Whisper modes only, flipped from the menu bar between utterances
        self.streaming = streaming
        dev, dtype = device_mod.pick(settings.device)
        print(
            f"device {dev.type} dtype {str(dtype).removeprefix('torch.')}",
            file=sys.stderr,
            flush=True,
        )
        models.check(mode, getattr(settings, f"{mode}_model"))
        if mode == "en":
            self.asr = load_model(settings, dev, dtype)
        elif mode == "sv":
            self.whisper = Whisper(settings.sv_model, ["sv"], dev, dtype)
        else:
            self.whisper = Whisper(settings.auto_model, ["sv", "en"], dev, dtype)
        self.keys = keyboard.Controller()
        self.audio: queue.Queue[np.ndarray | None] = queue.Queue()
        self.stream: sd.InputStream | None = None
        self.hearing = False

    def on_press(self, key: keyboard.Key | keyboard.KeyCode | None) -> None:
        if key != HOTKEY or self.stream is not None:
            return
        self.hearing = False
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=BLOCK_SAMPLES,
            callback=self.on_audio,
        )
        self.stream.start()

    def on_release(self, key: keyboard.Key | keyboard.KeyCode | None) -> None:
        if key != HOTKEY or self.stream is None:
            return
        self.stream.stop()
        self.stream.close()
        self.stream = None
        self.audio.put(None)
        print("IDLE", flush=True)

    def on_audio(
        self, indata: np.ndarray, frames: int, time: object, status: sd.CallbackFlags
    ) -> None:
        # REC is what lights the red dot in the menu bar, so it only goes out
        # once real samples arrive. A mic that's still starting up or has no
        # permission delivers pure zeros.
        if not self.hearing and indata.any():
            self.hearing = True
            print("REC", flush=True)
        self.audio.put(indata[:, 0].copy())

    def transcribe_forever(self) -> None:
        while True:
            if self.mode == "en":
                self.transcribe_live()
            elif self.streaming:
                self.transcribe_streaming()
            else:
                self.transcribe_on_release()

    def transcribe_on_release(self) -> None:
        chunks = []
        while (chunk := self.audio.get()) is not None:
            chunks.append(chunk)
        if sum(len(chunk) for chunk in chunks) < MIN_UTTERANCE_SEC * SAMPLE_RATE:
            return
        audio = np.concatenate(chunks).astype(np.float32) / 32768
        text = self.whisper.transcribe(audio)
        if text:
            self.keys.type(text + " ")

    def transcribe_streaming(self) -> None:
        """Whisper can't stream, so this re-transcribes everything said so far
        about once a second and types the words two passes in a row agree on.
        Typed words are never taken back."""
        chunks = []
        finished: list[str] = []
        previous: list[str] = []
        typed: list[str] = []
        next_pass = time.monotonic() + self.settings.stream_interval_sec
        while (chunk := self.audio.get()) is not None:
            chunks.append(chunk)
            # catch up on blocks that piled up during the last pass first
            if time.monotonic() < next_pass or not self.audio.empty():
                continue
            next_pass = time.monotonic() + self.settings.stream_interval_sec
            audio = np.concatenate(chunks).astype(np.float32) / 32768
            words = self.whisper.transcribe(audio, finished).split()
            untyped = words[untyped_start(words, typed) :]
            agreed = 0
            while agreed < min(len(untyped), len(previous)) and same_word(
                untyped[agreed], previous[agreed]
            ):
                agreed += 1
            self.type_words(untyped[:agreed], typed)
            previous = untyped[agreed:]

        if sum(len(chunk) for chunk in chunks) < MIN_UTTERANCE_SEC * SAMPLE_RATE:
            return
        audio = np.concatenate(chunks).astype(np.float32) / 32768
        words = self.whisper.transcribe(audio, finished).split()
        self.type_words(words[untyped_start(words, typed) :], typed)
        if typed:
            self.keys.type(" ")

    def type_words(self, words: list[str], typed: list[str]) -> None:
        if not words:
            return
        self.keys.type((" " if typed else "") + " ".join(words))
        typed.extend(words)

    def transcribe_live(self) -> None:
        state = self.asr.init_streaming_state(
            language="English", chunk_size_sec=self.settings.chunk_sec
        )
        typed = ""
        while (chunk := self.audio.get()) is not None:
            _, fixed = self.asr.streaming_transcribe(chunk, state)
            # the first unfixed_chunk_num chunks are re-decoded from scratch,
            # so their text isn't settled yet
            if state.chunk_id < state.unfixed_chunk_num:
                continue
            if state.text.startswith(fixed) and fixed.startswith(typed):
                self.keys.type(fixed[len(typed) :])
                typed = fixed

        final = self.asr.finish_streaming_transcribe(state)
        if not final:
            return
        final += " "
        common = os.path.commonprefix([typed, final])
        for _ in range(len(typed) - len(common)):
            self.keys.tap(keyboard.Key.backspace)
        self.keys.type(final[len(common) :])


def untyped_start(words: list[str], typed: list[str]) -> int:
    """Where in a fresh transcription the not-yet-typed part begins.

    Counting words isn't enough: a later pass can merge or split an earlier
    word, and everything after it shifts. So find the last few typed words in
    the new transcription and carry on from there."""
    if not typed:
        return 0
    # a typed word the model has since changed its mind about won't be found,
    # so fall back to matching fewer words before giving up and counting
    for n in (3, 2, 1):
        tail = typed[-n:]
        ends = [
            end
            for end in range(len(tail), len(words) + 1)
            if all(same_word(a, b) for a, b in zip(words[end - len(tail) : end], tail))
        ]
        if ends:
            return min(ends, key=lambda end: abs(end - len(typed)))
    return min(len(typed), len(words))


def same_word(a: str, b: str) -> bool:
    # "Hej," in one pass and "hej" in the next is still agreement
    return re.sub(r"\W", "", a.lower()) == re.sub(r"\W", "", b.lower())


def listen_to_parent(ptt: PushToTalk) -> None:
    for line in sys.stdin:
        if line.strip() in ("STREAM ON", "STREAM OFF"):
            ptt.streaming = line.strip() == "STREAM ON"
    # the menu bar app holds our stdin open; EOF means it's gone
    os._exit(0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES, default="auto")
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()

    try:
        ptt = PushToTalk(Settings(), args.mode, args.stream)
    except ValueError as err:
        print(f"ERROR {err}", flush=True)
        sys.exit(1)
    threading.Thread(target=listen_to_parent, args=(ptt,), daemon=True).start()
    threading.Thread(target=ptt.transcribe_forever, daemon=True).start()
    print("READY", flush=True)
    with keyboard.Listener(
        on_press=ptt.on_press, on_release=ptt.on_release
    ) as listener:
        listener.join()


if __name__ == "__main__":
    main()
