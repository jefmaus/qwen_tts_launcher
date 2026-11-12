"""Motor de síntesis: gestor de modelo, fragmentación de texto, generación y post-proceso."""
from __future__ import annotations

import contextlib
import gc
import os
import random
import re
import shutil
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import soundfile as sf

import config
import storage


# ---------------------------------------------------------------------------
# Importación silenciosa de torch / qwen_tts (sox y flash-attn imprimen avisos)
# ---------------------------------------------------------------------------
class _NoSox:
    """Sustituto de os.popen('sox -h'): evita lanzar cmd (y su error en consola) al importar `sox`."""

    def readlines(self):
        return []

    def read(self):
        return ""

    def close(self):
        return None


@contextlib.contextmanager
def _quiet_import():
    """Silencia los avisos de importación (sox ausente, flash-attn ausente) sin tocar descriptores
    del sistema: en una consola real de Windows redirigir fd 1/2 invalida el handle de stdout."""
    import io
    import logging

    logging.getLogger("sox").setLevel(logging.ERROR)
    real_popen = os.popen

    def fake_popen(cmd, *a, **k):
        if isinstance(cmd, str) and cmd.strip().startswith("sox"):
            return _NoSox()
        return real_popen(cmd, *a, **k)

    os.popen = fake_popen
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            yield
    finally:
        os.popen = real_popen


import torch  # noqa: E402

with _quiet_import():
    import warnings

    warnings.filterwarnings("ignore")
    try:
        from qwen_tts import Qwen3TTSModel, VoiceClonePromptItem  # noqa: F401

        HAS_QWEN = True
        QWEN_IMPORT_ERROR = ""
    except Exception as _e:  # pragma: no cover
        Qwen3TTSModel = None
        HAS_QWEN = False
        QWEN_IMPORT_ERROR = f"{type(_e).__name__}: {_e}"

try:
    from transformers.utils import logging as _hf_logging

    _hf_logging.set_verbosity_error()  # silencia "Setting pad_token_id..." en cada fragmento
except Exception:
    pass

try:
    import pyloudnorm as pyln

    HAS_LOUDNORM = True
except Exception:
    HAS_LOUDNORM = False

try:
    from pydub import AudioSegment

    HAS_PYDUB = True
except Exception:
    HAS_PYDUB = False


# ---------------------------------------------------------------------------
# ffmpeg
# ---------------------------------------------------------------------------
def find_ffmpeg() -> str | None:
    env = os.environ.get("QWEN_TTS_FFMPEG")
    if env and Path(env).exists():
        return env
    found = shutil.which("ffmpeg")
    if found:
        return found
    for p in (r"C:\ffmpeg\bin\ffmpeg.exe", r"C:\ffmpeg-8.1.2-essentials_build\bin\ffmpeg.exe"):
        if Path(p).exists():
            return p
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


FFMPEG_PATH = find_ffmpeg()
if HAS_PYDUB and FFMPEG_PATH:
    AudioSegment.converter = FFMPEG_PATH
    ffprobe = str(Path(FFMPEG_PATH).with_name("ffprobe" + Path(FFMPEG_PATH).suffix))
    if Path(ffprobe).exists():
        AudioSegment.ffprobe = ffprobe
HAS_MP3 = HAS_PYDUB and FFMPEG_PATH is not None


# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------
def cuda_available() -> bool:
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


def resolve_device(device: str) -> str:
    if device in (None, "", "auto"):
        return "cuda:0" if cuda_available() else "cpu"
    if device.startswith("cuda") and not cuda_available():
        return "cpu"
    return device


def gpu_name() -> str | None:
    if cuda_available():
        try:
            return torch.cuda.get_device_name(0)
        except Exception:
            return None
    return None


def vram_info() -> tuple[float, float] | None:
    """(usada_GB, total_GB) o None si no hay CUDA."""
    if not cuda_available():
        return None
    try:
        free, total = torch.cuda.mem_get_info()
        return (total - free) / 1e9, total / 1e9
    except Exception:
        return None


_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def is_downloaded(model_id: str) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache

        r = try_to_load_from_cache(model_id, "model.safetensors")
        return isinstance(r, str) and Path(r).exists()
    except Exception:
        return False


def download_model(model_id: str) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(model_id)


# ---------------------------------------------------------------------------
# Gestor de modelo (uno solo residente)
# ---------------------------------------------------------------------------
class ModelManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.model = None
        self.model_id: str | None = None
        self.device: str | None = None
        self.dtype: str | None = None
        self.attn: str | None = None
        self.load_seconds: float | None = None

    @property
    def lock(self) -> threading.RLock:
        return self._lock

    def is_loaded(self, model_id: str | None = None) -> bool:
        if self.model is None:
            return False
        return model_id is None or model_id == self.model_id

    def unload(self) -> None:
        with self._lock:
            if self.model is not None:
                try:
                    del self.model.model
                except Exception:
                    pass
                self.model = None
                self.model_id = None
            gc.collect()
            if cuda_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

    def load(self, model_id: str, device: str = "auto", dtype: str = "bfloat16", attn: str = "eager",
             log: Optional[Callable[[str], None]] = None):
        log = log or (lambda m: None)
        if not HAS_QWEN:
            raise RuntimeError(f"No se pudo importar qwen_tts: {QWEN_IMPORT_ERROR}")
        device = resolve_device(device)
        if device == "cpu" and dtype != "float32":
            log("CPU detectada: se usa float32 (bfloat16/float16 son muy lentos en CPU).")
            dtype = "float32"
        attn = attn or "eager"
        with self._lock:
            if self.model is not None and (self.model_id, self.device, self.dtype, self.attn) == (model_id, device, dtype, attn):
                return self.model
            if self.model is not None:
                log(f"Descargando {self.model_id} de memoria...")
                self.unload()
            if not is_downloaded(model_id):
                log(f"Descargando {model_id} desde Hugging Face (solo la primera vez)...")
            log(f"Cargando {model_id} en {device} ({dtype}, {attn})...")
            t0 = time.time()
            kwargs = dict(device_map=device, dtype=_DTYPES[dtype])
            try:
                model = Qwen3TTSModel.from_pretrained(model_id, attn_implementation=attn, **kwargs)
            except Exception as e:
                if attn != "eager":
                    log(f"'{attn}' no disponible ({type(e).__name__}); usando eager.")
                    attn = "eager"
                    model = Qwen3TTSModel.from_pretrained(model_id, attn_implementation="eager", **kwargs)
                else:
                    raise
            self.model = model
            self.model_id = model_id
            self.device = device
            self.dtype = dtype
            self.attn = attn
            self.load_seconds = time.time() - t0
            log(f"Modelo listo en {self.load_seconds:.1f} s.")
            return model

    def supported_speakers(self) -> list[str]:
        if self.model is not None:
            try:
                s = self.model.get_supported_speakers()
                if s:
                    return list(s)
            except Exception:
                pass
        return list(config.FALLBACK_SPEAKERS)

    def supported_languages(self) -> list[str]:
        if self.model is not None:
            try:
                s = self.model.get_supported_languages()
                if s:
                    langs = list(s)
                    # 'auto' primero, español segundo
                    ordered = [l for l in ("auto", "spanish", "english") if l in langs]
                    return ordered + [l for l in langs if l not in ordered]
            except Exception:
                pass
        return list(config.FALLBACK_LANGUAGES)

    def status_text(self) -> str:
        parts = []
        g = gpu_name()
        v = vram_info()
        if g:
            parts.append(f"GPU: {g}")
            if v:
                parts.append(f"VRAM {v[0]:.1f}/{v[1]:.1f} GB")
        else:
            parts.append("CPU (sin CUDA)")
        if self.model is not None:
            short = self.model_id.split("/")[-1].replace("Qwen3-TTS-12Hz-", "")
            parts.append(f"Modelo: {short} · {self.device} · {self.dtype}")
        else:
            parts.append("Modelo: ninguno cargado")
        return "  |  ".join(parts)


MANAGER = ModelManager()


# ---------------------------------------------------------------------------
# Fragmentación de texto
# ---------------------------------------------------------------------------
_SENT_RE = re.compile(r"(?<=[.!?…;:])\s+|(?<=[。！？])")


def _split_sentences(paragraph: str) -> list[str]:
    parts = [p.strip() for p in _SENT_RE.split(paragraph) if p and p.strip()]
    return parts or ([paragraph.strip()] if paragraph.strip() else [])


def _split_long(sentence: str, max_chars: int) -> list[str]:
    """Trocea una frase demasiado larga por comas o espacios sin cortar palabras."""
    if len(sentence) <= max_chars:
        return [sentence]
    out, cur = [], ""
    for word in re.split(r"(?<=,)\s+|\s+", sentence):
        if cur and len(cur) + 1 + len(word) > max_chars:
            out.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    if cur:
        out.append(cur)
    return out


def split_text(text: str, max_chars: int = 300) -> list[tuple[str, bool]]:
    """Devuelve [(fragmento, es_fin_de_parrafo), ...]."""
    max_chars = max(40, int(max_chars))
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[tuple[str, bool]] = []
    for para in paragraphs:
        para = re.sub(r"\s*\n\s*", " ", para)
        sentences: list[str] = []
        for s in _split_sentences(para):
            sentences.extend(_split_long(s, max_chars))
        cur = ""
        for s in sentences:
            if cur and len(cur) + 1 + len(s) > max_chars:
                chunks.append((cur, False))
                cur = s
            else:
                cur = f"{cur} {s}".strip()
        if cur:
            chunks.append((cur, False))
        if chunks:
            chunks[-1] = (chunks[-1][0], True)
    return chunks


# ---------------------------------------------------------------------------
# Post-proceso de audio
# ---------------------------------------------------------------------------
def normalize_loudness(wav: np.ndarray, sr: int, target_lufs: float) -> np.ndarray:
    if not HAS_LOUDNORM or wav.size < sr * 0.5:
        return wav
    try:
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(wav)
        if not np.isfinite(loudness):
            return wav
        gain_db = target_lufs - loudness
        out = wav * (10 ** (gain_db / 20))
        peak = float(np.max(np.abs(out))) if out.size else 0.0
        if peak > 0.99:
            out = out * (0.99 / peak)
        return out.astype(np.float32)
    except Exception:
        return wav


def finalize_audio(wav: np.ndarray, sr: int, target_lufs: float | None, silence_start: float,
                   silence_end: float) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=-1)
    if target_lufs is not None:
        wav = normalize_loudness(wav, sr, target_lufs)
    wav = np.clip(wav, -0.99, 0.99)
    head = np.zeros(int(sr * max(0.0, silence_start)), dtype=np.float32)
    tail = np.zeros(int(sr * max(0.0, silence_end)), dtype=np.float32)
    return np.concatenate([head, wav, tail])


def export_mp3(wav_path: Path, bitrate: str = config.MP3_BITRATE) -> Path | None:
    if not HAS_MP3:
        return None
    mp3_path = wav_path.with_suffix(".mp3")
    try:
        AudioSegment.from_wav(str(wav_path)).export(str(mp3_path), format="mp3", bitrate=bitrate)
        return mp3_path
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Generación
# ---------------------------------------------------------------------------
class GenerationCancelled(Exception):
    pass


@dataclass
class GenRequest:
    text: str
    mode: str
    model_id: str
    language: str = "auto"
    # design
    voice_design_prompt: str = ""
    # custom
    speaker: str = "ryan"
    instruct: str = ""
    # clone
    ref_audio: str | None = None
    ref_text: str = ""
    x_vector_only: bool = False
    # muestreo
    temperature: float = 0.9
    top_p: float = 1.0
    top_k: int = 50
    repetition_penalty: float = 1.05
    max_new_tokens: int = 2048
    seed: int = -1
    # texto largo
    chunk_chars: int = 300
    pause_sentence: float = 0.25
    pause_paragraph: float = 0.7
    # post-proceso
    silence_start: float = 0.5
    silence_end: float = 1.0
    target_lufs: float = -16.0
    normalize: bool = True
    export_mp3: bool = True
    # hardware
    device: str = "auto"
    dtype: str = "bfloat16"
    attn: str = "eager"
    extra: dict = field(default_factory=dict)


@dataclass
class GenResult:
    wav_path: Path
    mp3_path: Path | None
    sr: int
    duration_s: float
    seed: int
    chunks: int
    elapsed_s: float
    history_entry: dict


def _set_seed(seed: int) -> int:
    if seed is None or int(seed) < 0:
        seed = random.randint(0, 2**31 - 1)
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if cuda_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def validate_request(req: GenRequest) -> None:
    if not req.text or not req.text.strip():
        raise ValueError("Escribe el texto que quieres convertir a voz.")
    if req.mode == config.MODE_DESIGN and not (req.voice_design_prompt or "").strip():
        raise ValueError("Describe la voz en el campo 'Descripción de la voz' (en inglés).")
    if req.mode == config.MODE_CUSTOM and not req.speaker:
        raise ValueError("Elige un hablante.")
    if req.mode == config.MODE_CLONE:
        if not req.ref_audio or not Path(req.ref_audio).exists():
            raise ValueError("Sube un audio de referencia (3 a 15 segundos, una sola voz, sin ruido).")
        if not req.x_vector_only and not (req.ref_text or "").strip():
            raise ValueError(
                "Escribe la transcripción exacta del audio de referencia, o activa "
                "'Solo timbre (x-vector)' para clonar sin transcripción (menor fidelidad)."
            )


def generate(req: GenRequest, progress: Optional[Callable[[float, str], None]] = None,
             cancel: Optional[threading.Event] = None) -> GenResult:
    progress = progress or (lambda p, m: None)
    cancel = cancel or threading.Event()
    validate_request(req)

    t_start = time.time()
    progress(0.0, "Preparando modelo...")
    model = MANAGER.load(req.model_id, req.device, req.dtype, req.attn, log=lambda m: progress(0.02, m))

    chunks = split_text(req.text, req.chunk_chars)
    if not chunks:
        raise ValueError("El texto está vacío.")
    n = len(chunks)
    seed = _set_seed(req.seed)

    gen_kwargs = dict(
        temperature=float(req.temperature),
        top_p=float(req.top_p),
        top_k=int(req.top_k),
        repetition_penalty=float(req.repetition_penalty),
        max_new_tokens=int(req.max_new_tokens),
    )
    language = (req.language or "auto").lower()

    with MANAGER.lock:
        clone_prompt = None
        if req.mode == config.MODE_CLONE:
            progress(0.05, "Analizando audio de referencia...")
            clone_prompt = model.create_voice_clone_prompt(
                ref_audio=req.ref_audio,
                ref_text=(req.ref_text or "").strip() or None,
                x_vector_only_mode=bool(req.x_vector_only),
            )

        pieces: list[np.ndarray] = []
        sr = 24000
        for i, (chunk, end_para) in enumerate(chunks):
            if cancel.is_set():
                raise GenerationCancelled()
            preview = chunk if len(chunk) <= 60 else chunk[:57] + "..."
            progress(0.05 + 0.85 * (i / n), f"Fragmento {i + 1}/{n}: {preview}")
            try:
                if req.mode == config.MODE_DESIGN:
                    wavs, sr = model.generate_voice_design(
                        text=chunk, instruct=req.voice_design_prompt.strip(), language=language, **gen_kwargs
                    )
                elif req.mode == config.MODE_CUSTOM:
                    wavs, sr = model.generate_custom_voice(
                        text=chunk, speaker=req.speaker, language=language,
                        instruct=(req.instruct or "").strip() or None, **gen_kwargs
                    )
                else:
                    wavs, sr = model.generate_voice_clone(
                        text=chunk, language=language, voice_clone_prompt=clone_prompt, **gen_kwargs
                    )
            except torch.cuda.OutOfMemoryError as e:
                if cuda_available():
                    torch.cuda.empty_cache()
                raise RuntimeError(
                    "Sin memoria en la GPU. Prueba fragmentos más cortos, el modelo 0.6B, "
                    "dtype float16 o cierra otros programas que usen la GPU."
                ) from e
            w = np.asarray(wavs[0], dtype=np.float32).squeeze()
            if w.ndim > 1:
                w = w.mean(axis=-1)
            pieces.append(w)
            if i < n - 1:
                pause = req.pause_paragraph if end_para else req.pause_sentence
                if pause > 0:
                    pieces.append(np.zeros(int(sr * pause), dtype=np.float32))

    if cancel.is_set():
        raise GenerationCancelled()

    progress(0.92, "Normalizando y guardando...")
    wav = np.concatenate(pieces) if len(pieces) > 1 else pieces[0]
    wav = finalize_audio(
        wav, sr,
        req.target_lufs if req.normalize else None,
        req.silence_start, req.silence_end,
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    uid = uuid.uuid4().hex[:6]
    wav_path = config.OUTPUTS_DIR / f"{ts}_{req.mode}_{uid}.wav"
    sf.write(str(wav_path), wav, sr, subtype="PCM_16")
    mp3_path = export_mp3(wav_path) if req.export_mp3 else None

    duration = len(wav) / sr
    elapsed = time.time() - t_start
    entry = {
        "timestamp": ts,
        "mode": req.mode,
        "model": req.model_id,
        "text": req.text.strip()[:500],
        "seed": seed,
        "wav": str(wav_path),
        "mp3": str(mp3_path) if mp3_path else None,
        "duration_s": round(duration, 2),
        "elapsed_s": round(elapsed, 1),
        "chunks": n,
        "params": {
            "language": language,
            "speaker": req.speaker if req.mode == config.MODE_CUSTOM else None,
            "voice_design_prompt": req.voice_design_prompt if req.mode == config.MODE_DESIGN else None,
            "instruct": req.instruct if req.mode == config.MODE_CUSTOM else None,
            "ref_text": req.ref_text if req.mode == config.MODE_CLONE else None,
            "x_vector_only": req.x_vector_only if req.mode == config.MODE_CLONE else None,
            "temperature": req.temperature,
            "top_p": req.top_p,
            "top_k": req.top_k,
            "repetition_penalty": req.repetition_penalty,
            "max_new_tokens": req.max_new_tokens,
            "chunk_chars": req.chunk_chars,
            "target_lufs": req.target_lufs if req.normalize else None,
        },
    }
    storage.add_history(entry)
    progress(1.0, "Listo.")
    return GenResult(
        wav_path=wav_path, mp3_path=mp3_path, sr=sr, duration_s=duration, seed=seed,
        chunks=n, elapsed_s=elapsed, history_entry=entry,
    )
