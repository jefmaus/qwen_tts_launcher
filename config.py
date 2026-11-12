"""Rutas, catálogo de modelos y valores por defecto de Qwen3-TTS Studio."""
from __future__ import annotations

from pathlib import Path

APP_NAME = "Qwen3-TTS Studio"
APP_VERSION = "2.0.0"

BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"
HISTORY_DIR = BASE_DIR / "history"
HISTORY_FILE = HISTORY_DIR / "history.json"
VOICES_DIR = BASE_DIR / "voices"
SETTINGS_FILE = BASE_DIR / "settings.json"

for _d in (OUTPUTS_DIR, HISTORY_DIR, VOICES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Modos y modelos
# ---------------------------------------------------------------------------
MODE_DESIGN = "design"
MODE_CUSTOM = "custom"
MODE_CLONE = "clone"

MODE_LABELS = {
    MODE_DESIGN: "Diseño de voz",
    MODE_CUSTOM: "Voz predefinida",
    MODE_CLONE: "Clonación",
}
LABEL_TO_MODE = {v: k for k, v in MODE_LABELS.items()}

SIZE_LABELS = {
    "1.7B": "1.7B · calidad",
    "0.6B": "0.6B · rápido",
}
LABEL_TO_SIZE = {v: k for k, v in SIZE_LABELS.items()}

# Catálogo de modelos oficiales en Hugging Face.
MODELS = [
    {"id": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", "mode": MODE_DESIGN, "size": "1.7B", "gb": 4.2},
    {"id": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", "mode": MODE_CUSTOM, "size": "1.7B", "gb": 4.2},
    {"id": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", "mode": MODE_CUSTOM, "size": "0.6B", "gb": 2.0},
    {"id": "Qwen/Qwen3-TTS-12Hz-1.7B-Base", "mode": MODE_CLONE, "size": "1.7B", "gb": 4.2},
    {"id": "Qwen/Qwen3-TTS-12Hz-0.6B-Base", "mode": MODE_CLONE, "size": "0.6B", "gb": 2.0},
]


def models_for_mode(mode: str) -> list[dict]:
    return [m for m in MODELS if m["mode"] == mode]


def sizes_for_mode(mode: str) -> list[str]:
    return [m["size"] for m in models_for_mode(mode)]


def model_id_for(mode: str, size: str) -> str:
    for m in models_for_mode(mode):
        if m["size"] == size:
            return m["id"]
    # Si el tamaño no existe para ese modo (VoiceDesign solo 1.7B) usa el primero.
    return models_for_mode(mode)[0]["id"]


def model_info(model_id: str) -> dict | None:
    for m in MODELS:
        if m["id"] == model_id:
            return m
    return None


# ---------------------------------------------------------------------------
# Listas de respaldo (se sustituyen por las del modelo cuando está cargado)
# ---------------------------------------------------------------------------
FALLBACK_SPEAKERS = [
    "aiden", "dylan", "eric", "ono_anna", "ryan", "serena", "sohee", "uncle_fu", "vivian",
]
FALLBACK_LANGUAGES = [
    "auto", "spanish", "english", "chinese", "french", "german", "italian",
    "japanese", "korean", "portuguese", "russian",
]
LANGUAGE_LABELS = {
    "auto": "Auto",
    "spanish": "Español",
    "english": "Inglés",
    "chinese": "Chino",
    "french": "Francés",
    "german": "Alemán",
    "italian": "Italiano",
    "japanese": "Japonés",
    "korean": "Coreano",
    "portuguese": "Portugués",
    "russian": "Ruso",
}


def language_label(code: str) -> str:
    return LANGUAGE_LABELS.get(code, code.capitalize())


def speaker_label(code: str) -> str:
    return " ".join(w.capitalize() for w in code.split("_"))


# ---------------------------------------------------------------------------
# Valores por defecto
# ---------------------------------------------------------------------------
DEFAULT_VOICE_DESIGN_PROMPT = (
    "A calm, warm male narrator in his 30s, Colombian Spanish, steady storytelling pace, "
    "audiobook style, consistent tone."
)
DEFAULT_INSTRUCT = ""

DEFAULTS = {
    "mode": MODE_DESIGN,
    "size": "1.7B",
    "language": "spanish",
    "speaker": "ryan",
    "voice_design_prompt": DEFAULT_VOICE_DESIGN_PROMPT,
    "instruct": DEFAULT_INSTRUCT,
    "x_vector_only": False,
    # Muestreo (hard defaults de qwen_tts/inference/qwen3_tts_model.py)
    "temperature": 0.9,
    "top_p": 1.0,
    "top_k": 50,
    "repetition_penalty": 1.05,
    "max_new_tokens": 2048,
    "seed": -1,
    # Texto largo
    "chunk_chars": 300,
    "pause_sentence": 0.25,
    "pause_paragraph": 0.7,
    # Post-proceso
    "silence_start": 0.5,
    "silence_end": 1.0,
    "target_lufs": -16.0,
    "normalize": True,
    "export_mp3": True,
    # Hardware
    "device": "auto",
    "dtype": "bfloat16",
    "attn": "eager",
}

HISTORY_MAX = 200
MP3_BITRATE = "192k"
