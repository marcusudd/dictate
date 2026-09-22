import numpy as np
import torch
from transformers import (
    GenerationMixin,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

SAMPLE_RATE = 16000
WINDOW_SEC = 30
# Whisper makes up a "Thank you." or "." when there's no speech in the clip.
# Real speech scores around -0.3 to 0 here, noise around -1, so anything the
# model is this unsure about gets dropped. Its own no-speech token is useless
# for this: it stays at 0 even on pure silence.
MIN_AVG_LOGPROB = -0.7


class Whisper:
    """Runs Whisper one 30 s window at a time with plain greedy decoding.

    Deliberately not the transformers ASR pipeline or Whisper's own generate():
    their long-form loop never exits when the model returns nothing for a
    clip, which KB-Whisper does on noise without speech.
    """

    def __init__(
        self,
        model_id: str,
        languages: list[str],
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        # float16 is what the benchmarks ran with on MPS; CPU has no half kernels
        self.device = device
        self.dtype = torch.float32 if device.type == "cpu" else torch.float16
        self.processor = WhisperProcessor.from_pretrained(model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_id, dtype=self.dtype
        ).to(device)
        ids = self.processor.tokenizer.convert_tokens_to_ids
        self.sot = ids("<|startoftranscript|>")
        self.task = [ids("<|transcribe|>"), ids("<|notimestamps|>")]
        self.languages = [ids(f"<|{lang}|>") for lang in languages]
        self.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32))  # warmup

    @torch.no_grad()
    def transcribe(self, audio: np.ndarray, finished: list[str] | None = None) -> str:
        """finished: pass the same list on every call when re-transcribing
        audio that keeps growing. Full 30 s windows land there once and are
        reused, so only the last window costs anything."""
        if not audio.any():
            return ""
        windows = split_windows(audio)
        texts = []
        for i, window in enumerate(windows):
            if finished is not None and i < len(finished):
                texts.append(finished[i])
                continue
            text = self.transcribe_window(window)
            if finished is not None and i < len(windows) - 1:
                finished.append(text)
            texts.append(text)
        return " ".join(text for text in texts if text)

    def transcribe_window(self, audio: np.ndarray) -> str:
        features = self.processor(
            audio, sampling_rate=SAMPLE_RATE, return_tensors="pt"
        ).input_features.to(self.device, self.dtype)
        encoded = self.model.get_encoder()(features)

        # picks between our languages only, so Swedish can't come out as
        # Norwegian or Danish
        sot = torch.tensor([[self.sot]], device=self.device)
        logits = self.model(encoder_outputs=encoded, decoder_input_ids=sot).logits
        language = max(self.languages, key=lambda token: logits[0, -1, token].item())

        prompt = torch.tensor([[self.sot, language, *self.task]], device=self.device)
        # GenerationMixin.generate skips Whisper's long-form loop
        out = GenerationMixin.generate(
            self.model,
            encoder_outputs=encoded,
            decoder_input_ids=prompt,
            max_new_tokens=440,
            do_sample=False,
            num_beams=1,
            output_scores=True,
            return_dict_in_generate=True,
        )
        tokens = out.sequences[0, prompt.shape[1] :]
        logprobs = [
            step[0].float().log_softmax(-1)[token].item()
            for step, token in zip(out.scores, tokens)
        ]
        if not logprobs or sum(logprobs) / len(logprobs) < MIN_AVG_LOGPROB:
            return ""
        return self.processor.tokenizer.decode(tokens, skip_special_tokens=True).strip()


def split_windows(audio: np.ndarray) -> list[np.ndarray]:
    """Cuts audio longer than 30 s at the quietest spot in the last 5 s of
    each window, so words don't get chopped in half."""
    window = WINDOW_SEC * SAMPLE_RATE
    search = 5 * SAMPLE_RATE
    hop = SAMPLE_RATE // 10
    windows = []
    while len(audio) > window:
        tail = audio[window - search : window]
        energy = [np.abs(tail[i : i + hop]).mean() for i in range(0, search - hop, hop)]
        cut = window - search + int(np.argmin(energy)) * hop + hop // 2
        windows.append(audio[:cut])
        audio = audio[cut:]
    windows.append(audio)
    return windows
