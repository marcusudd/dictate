import torch

ORDER = ("cuda", "mps", "cpu")


def pick(name: str) -> tuple[torch.device, torch.dtype]:
    """Resolves DICTATE_DEVICE to a torch device and the dtype models load in.

    bfloat16 where the hardware does it natively, float16 on older CUDA cards,
    float32 on CPU because float16 kernels don't exist there."""
    if name == "auto":
        name = next(dev for dev in ORDER if available(dev))
    elif name not in ORDER:
        raise ValueError(f"DICTATE_DEVICE={name!r}, expected one of {', '.join(ORDER)}")
    elif not available(name):
        raise ValueError(f"DICTATE_DEVICE={name!r} but this machine has no {name}")

    if name == "cuda":
        native_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
        return torch.device("cuda"), torch.bfloat16 if native_bf16 else torch.float16
    if name == "mps":
        return torch.device("mps"), torch.bfloat16
    return torch.device("cpu"), torch.float32


def available(name: str) -> bool:
    if name == "cuda":
        return torch.cuda.is_available()
    if name == "mps":
        return torch.backends.mps.is_available()
    return True
