"""Historial, voces guardadas y ajustes persistentes."""
from __future__ import annotations

import json
import re
import shutil
import threading
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any

import config

_lock = threading.Lock()


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    raw = path.read_bytes()
    # Archivos antiguos pudieron escribirse con la codificación local de Windows (cp1252).
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return json.loads(raw.decode(enc))
        except Exception:
            continue
    return default


def _write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def slugify(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-").lower()
    return s or "voz"


# ---------------------------------------------------------------------------
# Historial
# ---------------------------------------------------------------------------
def list_history() -> list[dict]:
    hist = _read_json(config.HISTORY_FILE, [])
    if not isinstance(hist, list):
        return []
    out = []
    for e in hist:
        if not isinstance(e, dict):
            continue
        # Compatibilidad con entradas del app antiguo.
        e.setdefault("mode", "")
        e.setdefault("model", "")
        e.setdefault("seed", None)
        e.setdefault("duration_s", None)
        e.setdefault("wav", e.get("file"))
        e.setdefault("mp3", None)
        out.append(e)
    return out


def add_history(entry: dict) -> list[dict]:
    with _lock:
        hist = list_history()
        hist.insert(0, entry)
        hist = hist[: config.HISTORY_MAX]
        _write_json(config.HISTORY_FILE, hist)
    return hist


def delete_history(index: int, delete_files: bool = False) -> list[dict]:
    with _lock:
        hist = list_history()
        if 0 <= index < len(hist):
            e = hist.pop(index)
            if delete_files:
                for k in ("wav", "mp3"):
                    p = e.get(k)
                    if p:
                        try:
                            Path(p).unlink(missing_ok=True)
                        except Exception:
                            pass
        _write_json(config.HISTORY_FILE, hist)
    return hist


def clear_history() -> list[dict]:
    with _lock:
        _write_json(config.HISTORY_FILE, [])
    return []


def history_rows(hist: list[dict]) -> list[list]:
    rows = []
    for e in hist:
        ts = e.get("timestamp", "")
        try:
            ts_fmt = datetime.strptime(ts, "%Y%m%d_%H%M%S").strftime("%d/%m/%Y %H:%M")
        except Exception:
            ts_fmt = ts
        dur = e.get("duration_s")
        dur_fmt = f"{dur:.1f} s" if isinstance(dur, (int, float)) else ""
        text = (e.get("text") or "").replace("\n", " ")
        if len(text) > 90:
            text = text[:87] + "..."
        rows.append([
            ts_fmt,
            config.MODE_LABELS.get(e.get("mode", ""), e.get("mode", "") or "—"),
            text,
            dur_fmt,
            "" if e.get("seed") is None else str(e.get("seed")),
        ])
    return rows


# ---------------------------------------------------------------------------
# Voces guardadas
# ---------------------------------------------------------------------------
def _voice_dir(slug: str) -> Path:
    return config.VOICES_DIR / slug


def list_voices(mode: str | None = None) -> list[dict]:
    voices = []
    for d in sorted(config.VOICES_DIR.iterdir() if config.VOICES_DIR.exists() else []):
        meta = _read_json(d / "voice.json", None)
        if not isinstance(meta, dict):
            continue
        meta["slug"] = d.name
        if meta.get("ref_audio"):
            meta["ref_audio"] = str(d / meta["ref_audio"])
        if mode is None or meta.get("mode") == mode:
            voices.append(meta)
    return voices


def load_voice(slug: str) -> dict | None:
    for v in list_voices():
        if v["slug"] == slug:
            return v
    return None


def save_voice(name: str, mode: str, data: dict, ref_audio_path: str | None = None) -> dict:
    name = (name or "").strip()
    if not name:
        raise ValueError("Pon un nombre a la voz.")
    slug = slugify(name)
    d = _voice_dir(slug)
    d.mkdir(parents=True, exist_ok=True)
    meta = {
        "name": name,
        "mode": mode,
        "created": datetime.now().isoformat(timespec="seconds"),
        **data,
    }
    if mode == config.MODE_CLONE:
        if not ref_audio_path or not Path(ref_audio_path).exists():
            raise ValueError("Una voz clonada necesita audio de referencia.")
        ext = Path(ref_audio_path).suffix.lower() or ".wav"
        dst = d / f"ref{ext}"
        if Path(ref_audio_path).resolve() != dst.resolve():
            shutil.copyfile(ref_audio_path, dst)
        meta["ref_audio"] = dst.name
    _write_json(d / "voice.json", meta)
    out = dict(meta)
    out["slug"] = slug
    if out.get("ref_audio"):
        out["ref_audio"] = str(d / out["ref_audio"])
    return out


def delete_voice(slug: str) -> None:
    d = _voice_dir(slug)
    if d.exists() and d.is_dir() and d.parent == config.VOICES_DIR:
        shutil.rmtree(d, ignore_errors=True)


def voice_choices(mode: str) -> list[tuple[str, str]]:
    """Pares (etiqueta, slug) para un Dropdown."""
    return [(v["name"], v["slug"]) for v in list_voices(mode)]


# ---------------------------------------------------------------------------
# Ajustes
# ---------------------------------------------------------------------------
def load_settings() -> dict:
    s = dict(config.DEFAULTS)
    saved = _read_json(config.SETTINGS_FILE, {})
    if isinstance(saved, dict):
        for k, v in saved.items():
            if k in s:
                s[k] = v
    return s


def save_settings(values: dict) -> None:
    with _lock:
        current = load_settings()
        for k, v in values.items():
            if k in current:
                current[k] = v
        _write_json(config.SETTINGS_FILE, current)
