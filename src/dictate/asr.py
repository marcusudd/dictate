import types

import numpy as np
import torch

from dictate.config import Settings
from dictate.r2t2_asr import R2T2ASRModel

SAMPLE_RATE = 16000


class HFGenerate:
    """Takes the vLLM-style generate call r2t2_asr makes and runs it on the
    transformers model instead — vLLM doesn't run on a Mac."""

    def __init__(self, model: torch.nn.Module, processor, max_new_tokens: int) -> None:
        self.model = model
        self.processor = processor
        self.max_new_tokens = max_new_tokens

    @torch.no_grad()
    def generate(
        self, inputs: list[dict], sampling_params: object = None, use_tqdm: bool = False
    ) -> list[types.SimpleNamespace]:
        item = inputs[0]
        audio = item["multi_modal_data"]["audio"][0]
        enc = self.processor(
            text=[item["prompt"]], audio=[audio], return_tensors="pt", padding=True
        )
        enc = enc.to(self.model.device).to(self.model.dtype)
        out = self.model.generate(
            **enc,
            max_new_tokens=self.max_new_tokens,
            # set explicitly, otherwise transformers logs a line per chunk
            pad_token_id=self.processor.tokenizer.pad_token_id,
        )
        text = self.processor.batch_decode(
            out.sequences[:, enc["input_ids"].shape[1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return [types.SimpleNamespace(outputs=[types.SimpleNamespace(text=text)])]


def load_model(settings: Settings) -> R2T2ASRModel:
    asr = R2T2ASRModel.from_pretrained(
        settings.en_model, dtype=torch.bfloat16, device_map="mps"
    )
    # r2t2_asr refuses to stream unless backend is "vllm", but the only vLLM
    # thing it actually touches is model.generate
    asr.model = HFGenerate(asr.model, asr.processor, settings.max_new_tokens)
    asr.backend = "vllm"

    # the first generate call is several seconds slower than the rest
    state = asr.init_streaming_state(chunk_size_sec=1.0)
    asr.streaming_transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32), state)
    return asr
