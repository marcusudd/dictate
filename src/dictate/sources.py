from collections.abc import Callable

import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly

SAMPLE_RATE = 16000
# small blocks so the first one shows up ~80 ms after the key goes down
# instead of after a full 320 ms chunk
BLOCK_SAMPLES = 512


class Microphone:
    """The default input device as 16 kHz mono int16 blocks.

    Asks for 16 kHz first. Windows' WASAPI shared mode only hands out the
    device's own rate, so when that's refused the stream opens at the native
    rate and every block is resampled on the way out."""

    def __init__(self, on_block: Callable[[np.ndarray], None]) -> None:
        self.on_block = on_block
        self.stream: sd.InputStream | None = None
        self.rate = SAMPLE_RATE

    def start(self) -> None:
        try:
            self.stream = self.open(SAMPLE_RATE)
        except sd.PortAudioError:
            self.rate = int(sd.query_devices(kind="input")["default_samplerate"])
            self.stream = self.open(self.rate)
        self.stream.start()

    def open(self, rate: int) -> sd.InputStream:
        blocksize = round(BLOCK_SAMPLES * rate / SAMPLE_RATE)
        return sd.InputStream(
            samplerate=rate,
            channels=1,
            dtype="int16",
            blocksize=blocksize,
            callback=self.callback,
        )

    def callback(
        self, indata: np.ndarray, frames: int, time: object, status: sd.CallbackFlags
    ) -> None:
        block = indata[:, 0]
        if self.rate != SAMPLE_RATE:
            g = np.gcd(self.rate, SAMPLE_RATE)
            block = resample_poly(
                block.astype(np.float32), SAMPLE_RATE // g, self.rate // g
            )
            block = np.clip(block, -32768, 32767).astype(np.int16)
        self.on_block(block.copy())

    def stop(self) -> None:
        if self.stream is None:
            return
        self.stream.stop()
        self.stream.close()
        self.stream = None
