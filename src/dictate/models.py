from transformers import PretrainedConfig

# what each menu mode's slot must be. Auto/Svenska run through whisper.py,
# English through the vendored R2T2 streaming code (any Qwen3-ASR checkpoint)
EXPECTED = {"auto": "whisper", "sv": "whisper", "en": "qwen3_asr"}
SLOT_VAR = {"auto": "DICTATE_AUTO_MODEL", "sv": "DICTATE_SV_MODEL", "en": "DICTATE_EN_MODEL"}


def check(slot: str, model_id: str) -> None:
    """Reads only config.json, so a wrong model fails before the weights download."""
    try:
        config, _ = PretrainedConfig.get_config_dict(model_id)
    except (OSError, ValueError) as err:
        raise ValueError(f"{SLOT_VAR[slot]}={model_id!r}: can't load it ({err})") from err
    model_type = config.get("model_type")
    if model_type != EXPECTED[slot]:
        raise ValueError(
            f"{SLOT_VAR[slot]}={model_id!r} is a {model_type!r} model, "
            f"this slot needs {EXPECTED[slot]!r}"
        )
