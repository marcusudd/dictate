from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DICTATE_")

    # auto: Whisper turbo, picks Swedish or English per utterance, types on release
    auto_model: str = "openai/whisper-large-v3-turbo"
    # sv: KB-Whisper, types on release
    sv_model: str = "KBLab/kb-whisper-large"
    # en: R2T2, types while you talk
    en_model: str = "netease-youdao/Confucius4-R2T2"

    # Whisper streaming: how often what you've said so far gets re-transcribed.
    # 0.5 types sooner but commits more wrong words
    stream_interval_sec: float = 1.0

    # R2T2 only. 160 ms chunks don't keep up on MPS, 320 ms do
    chunk_sec: float = 0.32
    max_new_tokens: int = 32
