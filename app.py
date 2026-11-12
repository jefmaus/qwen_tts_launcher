"""Interfaz Gradio de Qwen3-TTS Studio."""
from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

import gradio as gr

import config
import storage
import tts_engine as E

# ---------------------------------------------------------------------------
# Estado global de generación (una a la vez)
# ---------------------------------------------------------------------------
_CANCEL: dict[str, threading.Event] = {}

CSS = """
.gradio-container { max-width: none !important; width: 100% !important; padding: 0 28px 24px 28px !important; }
#header { display:flex; align-items:center; justify-content:space-between; gap:16px; flex-wrap:wrap;
          padding: 10px 2px 2px 2px; }
#header h1 { margin:0; font-size:1.7rem; letter-spacing:-0.01em; }
#header .sub { opacity:.65; font-size:.9rem; margin-top:2px; }
#chip { font-size:.84rem; padding:7px 14px; border-radius:999px; border:1px solid var(--border-color-primary);
        background: var(--background-fill-secondary); }
#gen_btn { height:56px; font-size:1.1rem; font-weight:600; }
#stop_btn { height:56px; }
#status_box textarea { font-family: var(--font-mono); font-size:.86rem; }
.card { border:1px solid var(--border-color-primary); border-radius:12px; padding:12px 14px !important;
        background: var(--background-fill-secondary); }
.card-title { font-weight:600; font-size:.95rem; margin: 2px 0 2px 0; }
.small-note { font-size:.82rem; opacity:.7; }
#counter { font-size:.8rem; opacity:.7; text-align:right; margin-top:-4px; }
.icon-btn { min-width: 46px !important; max-width: 46px; padding: 0 !important; }
footer { display:none !important; }
"""

FORCE_DARK_JS = """
() => {
  const url = new URL(window.location);
  if (url.searchParams.get('__theme') !== 'dark') {
    url.searchParams.set('__theme', 'dark');
    window.location.href = url.href;
  }
}
"""

THEME = gr.themes.Soft(
    primary_hue="violet",
    secondary_hue="indigo",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "Segoe UI", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "Consolas", "monospace"],
).set(
    button_primary_background_fill="linear-gradient(90deg, #7c3aed, #4f46e5)",
    button_primary_background_fill_hover="linear-gradient(90deg, #8b5cf6, #6366f1)",
    block_radius="12px",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def chip_html() -> str:
    return f'<span id="chip">{E.MANAGER.status_text()}</span>'


def lang_choices() -> list[tuple[str, str]]:
    return [(config.language_label(c), c) for c in E.MANAGER.supported_languages()]


def speaker_choices() -> list[tuple[str, str]]:
    return [(config.speaker_label(s), s) for s in E.MANAGER.supported_speakers()]


def size_choices(mode: str) -> list[tuple[str, str]]:
    return [(config.SIZE_LABELS[s], s) for s in config.sizes_for_mode(mode)]


def download_note(mode: str, size: str) -> str:
    sizes = config.sizes_for_mode(mode)
    if size not in sizes:
        size = sizes[0]
    mid = config.model_id_for(mode, size)
    info = config.model_info(mid) or {}
    short = mid.split("/")[-1]
    if E.is_downloaded(mid):
        return f"✅ `{short}` descargado"
    return f"⬇️ `{short}` se descargará la primera vez (~{info.get('gb', '?')} GB)"


def text_counter(text: str, chunk_chars: float) -> str:
    text = text or ""
    n = len(E.split_text(text, int(chunk_chars))) if text.strip() else 0
    frag = "fragmento" if n == 1 else "fragmentos"
    return f"{len(text)} caracteres · {n} {frag}"


def read_txt(file) -> str:
    if not file:
        return gr.update()
    path = getattr(file, "name", None) or str(file)
    try:
        return Path(path).read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        raise gr.Error(f"No se pudo leer el archivo: {e}")


def open_outputs_folder() -> str:
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(config.OUTPUTS_DIR))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(config.OUTPUTS_DIR)])
        else:
            subprocess.Popen(["xdg-open", str(config.OUTPUTS_DIR)])
        return f"Carpeta abierta: {config.OUTPUTS_DIR}"
    except Exception as e:
        return f"No se pudo abrir la carpeta: {e}"


def models_table() -> list[list]:
    rows = []
    for m in config.MODELS:
        loaded = E.MANAGER.is_loaded(m["id"])
        state = "🟢 cargado" if loaded else ("✅ descargado" if E.is_downloaded(m["id"]) else "⬇️ no descargado")
        rows.append([m["id"].split("/")[-1], config.MODE_LABELS[m["mode"]], m["size"], f"~{m['gb']} GB", state])
    return rows


# ---------------------------------------------------------------------------
# Generación
# ---------------------------------------------------------------------------
GEN_INPUTS = [
    "mode_label", "size", "text", "language", "voice_design_prompt", "speaker", "instruct",
    "ref_audio", "ref_text", "x_vector_only",
    "temperature", "top_p", "top_k", "repetition_penalty", "max_new_tokens", "seed",
    "chunk_chars", "pause_sentence", "pause_paragraph",
    "silence_start", "silence_end", "target_lufs", "normalize", "export_mp3",
    "device", "dtype", "attn",
]


def on_generate(*values, progress=gr.Progress()):
    kw = dict(zip(GEN_INPUTS, values))
    mode = config.LABEL_TO_MODE.get(kw["mode_label"], kw["mode_label"])
    size = kw["size"] if kw["size"] in config.sizes_for_mode(mode) else config.sizes_for_mode(mode)[0]
    req = E.GenRequest(
        text=kw["text"] or "",
        mode=mode,
        model_id=config.model_id_for(mode, size),
        language=kw["language"] or "auto",
        voice_design_prompt=kw["voice_design_prompt"] or "",
        speaker=kw["speaker"] or "",
        instruct=kw["instruct"] or "",
        ref_audio=kw["ref_audio"],
        ref_text=kw["ref_text"] or "",
        x_vector_only=bool(kw["x_vector_only"]),
        temperature=float(kw["temperature"]),
        top_p=float(kw["top_p"]),
        top_k=int(kw["top_k"]),
        repetition_penalty=float(kw["repetition_penalty"]),
        max_new_tokens=int(kw["max_new_tokens"]),
        seed=int(kw["seed"]) if kw["seed"] is not None else -1,
        chunk_chars=int(kw["chunk_chars"]),
        pause_sentence=float(kw["pause_sentence"]),
        pause_paragraph=float(kw["pause_paragraph"]),
        silence_start=float(kw["silence_start"]),
        silence_end=float(kw["silence_end"]),
        target_lufs=float(kw["target_lufs"]),
        normalize=bool(kw["normalize"]),
        export_mp3=bool(kw["export_mp3"]),
        device=kw["device"] or "auto",
        dtype=kw["dtype"] or "bfloat16",
        attn=kw["attn"] or "eager",
    )
    try:
        E.validate_request(req)
    except ValueError as e:
        raise gr.Error(str(e))

    # Recordar ajustes del usuario.
    storage.save_settings({
        "mode": mode, "size": size, "language": req.language, "speaker": req.speaker,
        "voice_design_prompt": req.voice_design_prompt, "instruct": req.instruct,
        "x_vector_only": req.x_vector_only,
        "temperature": req.temperature, "top_p": req.top_p, "top_k": req.top_k,
        "repetition_penalty": req.repetition_penalty, "max_new_tokens": req.max_new_tokens,
        "seed": req.seed, "chunk_chars": req.chunk_chars, "pause_sentence": req.pause_sentence,
        "pause_paragraph": req.pause_paragraph, "silence_start": req.silence_start,
        "silence_end": req.silence_end, "target_lufs": req.target_lufs, "normalize": req.normalize,
        "export_mp3": req.export_mp3, "device": req.device, "dtype": req.dtype, "attn": req.attn,
    })

    cancel = threading.Event()
    _CANCEL["ev"] = cancel
    q: queue.Queue = queue.Queue()
    result: dict = {}

    def work():
        try:
            result["ok"] = E.generate(req, lambda p, m: q.put((p, m)), cancel)
        except BaseException as e:  # noqa: BLE001
            result["err"] = e

    th = threading.Thread(target=work, daemon=True)
    th.start()

    def partial(msg: str):
        return (gr.update(), gr.update(), msg, gr.update(), chip_html(), gr.update(), gr.update())

    progress(0.0, desc="Iniciando...")
    yield partial("Iniciando...")
    while th.is_alive():
        try:
            p, m = q.get(timeout=0.3)
        except queue.Empty:
            continue
        progress(p, desc=m)
        yield partial(m)
    th.join()
    while not q.empty():
        p, m = q.get_nowait()
        progress(p, desc=m)

    if "err" in result:
        err = result["err"]
        if isinstance(err, E.GenerationCancelled):
            yield partial("⏹️ Generación cancelada.")
            return
        raise gr.Error(f"{type(err).__name__}: {err}")

    r: E.GenResult = result["ok"]
    status = (
        f"✅ {r.wav_path.name}\n"
        f"Duración {r.duration_s:.1f} s · {r.chunks} fragmento(s) · {r.elapsed_s:.1f} s de proceso · seed {r.seed}"
        + (f"\nMP3: {r.mp3_path.name}" if r.mp3_path else "")
    )
    yield (
        str(r.wav_path),
        gr.update(value=str(r.mp3_path), visible=True) if r.mp3_path else gr.update(value=None, visible=False),
        status,
        storage.history_rows(storage.list_history()),
        chip_html(),
        gr.update(choices=lang_choices()),
        gr.update(choices=speaker_choices()),
    )


def on_stop():
    ev = _CANCEL.get("ev")
    if ev is not None:
        ev.set()
        return "⏳ Deteniendo al terminar el fragmento actual..."
    return "No hay ninguna generación en curso."


# ---------------------------------------------------------------------------
# Voces guardadas
# ---------------------------------------------------------------------------
def on_mode_change(mode_label: str, size: str):
    mode = config.LABEL_TO_MODE.get(mode_label, mode_label)
    sizes = config.sizes_for_mode(mode)
    size = size if size in sizes else sizes[0]
    return (
        gr.update(visible=mode == config.MODE_DESIGN),
        gr.update(visible=mode == config.MODE_CUSTOM),
        gr.update(visible=mode == config.MODE_CLONE),
        gr.update(choices=size_choices(mode), value=size, interactive=len(sizes) > 1),
        download_note(mode, size),
        gr.update(choices=storage.voice_choices(mode), value=None),
    )


def on_size_change(mode_label: str, size: str):
    mode = config.LABEL_TO_MODE.get(mode_label, mode_label)
    return download_note(mode, size)


def on_save_voice(name, mode_label, voice_design_prompt, speaker, instruct, ref_audio, ref_text, x_vector_only):
    mode = config.LABEL_TO_MODE.get(mode_label, mode_label)
    try:
        if mode == config.MODE_DESIGN:
            data = {"voice_design_prompt": (voice_design_prompt or "").strip()}
            if not data["voice_design_prompt"]:
                raise ValueError("La descripción de la voz está vacía.")
            v = storage.save_voice(name, mode, data)
        elif mode == config.MODE_CUSTOM:
            v = storage.save_voice(name, mode, {"speaker": speaker, "instruct": instruct or ""})
        else:
            v = storage.save_voice(
                name, mode, {"ref_text": ref_text or "", "x_vector_only": bool(x_vector_only)}, ref_audio_path=ref_audio
            )
    except ValueError as e:
        raise gr.Error(str(e))
    gr.Info(f"Voz '{v['name']}' guardada.")
    return gr.update(choices=storage.voice_choices(mode), value=v["slug"]), ""


def on_load_voice(slug: str):
    v = storage.load_voice(slug) if slug else None
    if not v:
        return (gr.update(),) * 5
    return (
        gr.update(value=v.get("voice_design_prompt")) if v["mode"] == config.MODE_DESIGN else gr.update(),
        gr.update(value=v.get("speaker")) if v["mode"] == config.MODE_CUSTOM else gr.update(),
        gr.update(value=v.get("instruct", "")) if v["mode"] == config.MODE_CUSTOM else gr.update(),
        gr.update(value=v.get("ref_audio")) if v["mode"] == config.MODE_CLONE else gr.update(),
        gr.update(value=v.get("ref_text", "")) if v["mode"] == config.MODE_CLONE else gr.update(),
    )


def on_load_voice_xvec(slug: str):
    v = storage.load_voice(slug) if slug else None
    if not v or v["mode"] != config.MODE_CLONE:
        return gr.update()
    return gr.update(value=bool(v.get("x_vector_only", False)))


def on_delete_voice(slug: str, mode_label: str):
    mode = config.LABEL_TO_MODE.get(mode_label, mode_label)
    if slug:
        storage.delete_voice(slug)
        gr.Info("Voz eliminada.")
    return gr.update(choices=storage.voice_choices(mode), value=None)


# ---------------------------------------------------------------------------
# Historial
# ---------------------------------------------------------------------------
def on_history_select(evt: gr.SelectData):
    hist = storage.list_history()
    idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    if idx is None or idx >= len(hist):
        return None, "", -1
    e = hist[idx]
    wav = e.get("wav")
    params = e.get("params") or {}
    lines = [
        f"**Texto:** {e.get('text', '')}",
        f"**Modo:** {config.MODE_LABELS.get(e.get('mode'), e.get('mode') or '—')} · **Modelo:** {e.get('model') or '—'}",
        f"**Seed:** {e.get('seed')} · **Fragmentos:** {e.get('chunks', '—')} · **Proceso:** {e.get('elapsed_s', '—')} s",
    ]
    detail_keys = ("voice_design_prompt", "speaker", "instruct", "ref_text", "temperature", "top_p", "top_k")
    extras = [f"{k}: {params[k]}" for k in detail_keys if params.get(k) not in (None, "")]
    if extras:
        lines.append("**Parámetros:** " + " · ".join(str(x) for x in extras))
    lines.append(f"**Archivo:** `{wav}`" + (f" · MP3: `{e.get('mp3')}`" if e.get("mp3") else ""))
    audio = wav if wav and Path(wav).exists() else None
    if wav and not audio:
        lines.append("⚠️ El archivo ya no existe en disco.")
    return audio, "\n\n".join(lines), idx


def on_history_delete(idx: int, delete_files: bool):
    if idx is None or idx < 0:
        raise gr.Error("Selecciona una fila del historial.")
    hist = storage.delete_history(int(idx), delete_files=bool(delete_files))
    return storage.history_rows(hist), None, "", -1


def on_history_clear():
    storage.clear_history()
    return [], None, "", -1


# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------
def on_download_model(model_short: str, progress=gr.Progress()):
    mid = next((m["id"] for m in config.MODELS if m["id"].endswith(model_short)), None)
    if not mid:
        raise gr.Error("Modelo desconocido.")
    if E.is_downloaded(mid):
        return models_table(), f"✅ {model_short} ya estaba descargado."
    progress(0.1, desc=f"Descargando {model_short}... (ver consola para el progreso)")
    try:
        E.download_model(mid)
    except Exception as e:
        raise gr.Error(f"Fallo al descargar: {e}")
    return models_table(), f"✅ {model_short} descargado."


def on_unload_model():
    E.MANAGER.unload()
    return models_table(), "Modelo descargado de memoria.", chip_html()


def on_refresh_models():
    return models_table(), chip_html()


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def build_demo() -> gr.Blocks:
    s = storage.load_settings()
    mode0 = s["mode"] if s["mode"] in config.MODE_LABELS else config.MODE_DESIGN
    sizes0 = config.sizes_for_mode(mode0)
    size0 = s["size"] if s["size"] in sizes0 else sizes0[0]
    device_choices = ["auto"] + (["cuda:0"] if E.cuda_available() else []) + ["cpu"]

    with gr.Blocks(title=config.APP_NAME, theme=THEME, css=CSS, js=FORCE_DARK_JS) as demo:
        with gr.Row(elem_id="header"):
            gr.HTML(
                f"<div><h1>🎙️ {config.APP_NAME}</h1>"
                f"<div class='sub'>Texto a voz local con Qwen3-TTS · v{config.APP_VERSION}</div></div>"
            )
            chip = gr.HTML(chip_html())

        with gr.Tabs():
            # ------------------------------------------------------------ Generar
            with gr.Tab("🎧 Generar"):
                with gr.Row(equal_height=False):
                    # ------------------------------------------------ Columna izquierda: entrada
                    with gr.Column(scale=6):
                        with gr.Group(elem_classes="card"):
                            mode_radio = gr.Radio(
                                choices=list(config.MODE_LABELS.values()),
                                value=config.MODE_LABELS[mode0],
                                label="Modo",
                                info="Diseño: describe la voz · Predefinida: voces de fábrica · Clonación: a partir de un audio",
                            )
                            with gr.Row():
                                size_dd = gr.Dropdown(
                                    choices=size_choices(mode0), value=size0, label="Modelo",
                                    interactive=len(sizes0) > 1, scale=1,
                                )
                                language_dd = gr.Dropdown(
                                    choices=lang_choices(), value=s["language"], label="Idioma", scale=1,
                                )
                            download_md = gr.Markdown(download_note(mode0, size0), elem_classes="small-note")

                        with gr.Group(elem_classes="card"):
                            with gr.Row():
                                voice_dd = gr.Dropdown(
                                    choices=storage.voice_choices(mode0), value=None, label="Voz guardada",
                                    info="Carga una voz guardada para este modo", scale=1,
                                )
                                voice_name = gr.Textbox(
                                    label="Guardar voz actual como", placeholder="Nombre de la voz", scale=1,
                                    info="Guarda los campos de este modo con un nombre",
                                )
                            with gr.Row():
                                voice_del_btn = gr.Button("🗑️ Borrar voz seleccionada", size="sm", scale=1)
                                voice_save_btn = gr.Button("💾 Guardar voz actual", size="sm", variant="secondary", scale=1)

                        with gr.Group(visible=mode0 == config.MODE_DESIGN, elem_classes="card") as grp_design:
                            voice_design_prompt = gr.Textbox(
                                label="Descripción de la voz (en inglés)", lines=3,
                                value=s["voice_design_prompt"],
                                info="Edad, género, timbre, acento, ritmo y estilo. Mantenla idéntica para conservar la misma voz.",
                            )
                        with gr.Group(visible=mode0 == config.MODE_CUSTOM, elem_classes="card") as grp_custom:
                            with gr.Row():
                                speaker_dd = gr.Dropdown(choices=speaker_choices(), value=s["speaker"], label="Hablante", scale=1)
                                instruct_tb = gr.Textbox(
                                    label="Instrucción (opcional, en inglés)", value=s["instruct"], scale=2,
                                    placeholder="p. ej. calm audiobook narration / whisper / angry and fast",
                                    info="El modelo 0.6B ignora la instrucción.",
                                )
                        with gr.Group(visible=mode0 == config.MODE_CLONE, elem_classes="card") as grp_clone:
                            with gr.Row():
                                ref_audio = gr.Audio(
                                    label="Audio de referencia (3 a 15 s, una sola voz, sin ruido)",
                                    type="filepath", sources=["upload", "microphone"], scale=1,
                                )
                                with gr.Column(scale=1):
                                    ref_text = gr.Textbox(
                                        label="Transcripción exacta del audio de referencia", lines=4,
                                        placeholder="Escribe literalmente lo que se dice en el audio",
                                    )
                                    x_vector_cb = gr.Checkbox(
                                        value=bool(s["x_vector_only"]), label="Solo timbre (x-vector)",
                                        info="No necesita transcripción; clona peor la entonación.",
                                    )

                        with gr.Group(elem_classes="card"):
                            text_tb = gr.Textbox(
                                label="Texto", lines=12, max_lines=40,
                                placeholder="Escribe o pega el texto. Separa párrafos con una línea en blanco.",
                            )
                            with gr.Row(equal_height=True):
                                txt_file = gr.UploadButton("📄 Cargar .txt", file_types=[".txt", ".md"], size="sm", scale=0)
                                counter_md = gr.Markdown("0 caracteres · 0 fragmentos", elem_id="counter")

                        with gr.Row():
                            gen_btn = gr.Button("🚀 Generar audio", variant="primary", elem_id="gen_btn", scale=4)
                            stop_btn = gr.Button("⏹️ Detener", variant="stop", elem_id="stop_btn", scale=1)

                    # ------------------------------------------------ Columna derecha: salida y parámetros
                    with gr.Column(scale=5):
                        with gr.Group(elem_classes="card"):
                            audio_out = gr.Audio(label="Resultado", type="filepath", autoplay=True, show_download_button=True)
                            mp3_file = gr.File(label="MP3", visible=False, height=60)
                            status_tb = gr.Textbox(label="Estado / progreso", lines=4, interactive=False, elem_id="status_box")
                            open_btn = gr.Button("📂 Abrir carpeta de salidas", size="sm")

                        with gr.Group(elem_classes="card"):
                            gr.Markdown("🎛️ **Muestreo**", elem_classes="card-title")
                            with gr.Row():
                                temperature = gr.Slider(0.1, 1.5, value=s["temperature"], step=0.05, label="Temperature",
                                                        info="Bajo = estable, alto = expresivo")
                                top_p = gr.Slider(0.1, 1.0, value=s["top_p"], step=0.05, label="Top-p")
                                top_k = gr.Slider(1, 100, value=s["top_k"], step=1, label="Top-k")
                            with gr.Row():
                                repetition_penalty = gr.Slider(1.0, 1.5, value=s["repetition_penalty"], step=0.01, label="Repetition penalty")
                                max_new_tokens = gr.Slider(256, 4096, value=s["max_new_tokens"], step=64, label="Max tokens por fragmento",
                                                           info="12 tokens ≈ 1 s de audio")
                                seed = gr.Number(value=s["seed"], precision=0, label="Seed", info="-1 = aleatoria")

                        with gr.Group(elem_classes="card"):
                            gr.Markdown("📝 **Texto largo**", elem_classes="card-title")
                            with gr.Row():
                                chunk_chars = gr.Slider(80, 800, value=s["chunk_chars"], step=10, label="Caracteres por fragmento")
                                pause_sentence = gr.Slider(0.0, 1.5, value=s["pause_sentence"], step=0.05, label="Pausa entre fragmentos (s)")
                                pause_paragraph = gr.Slider(0.0, 3.0, value=s["pause_paragraph"], step=0.05, label="Pausa entre párrafos (s)")

                        with gr.Group(elem_classes="card"):
                            gr.Markdown("🔊 **Post-proceso**", elem_classes="card-title")
                            with gr.Row():
                                target_lufs = gr.Slider(-24, -10, value=s["target_lufs"], step=0.5, label="LUFS objetivo",
                                                        info="-16 Spotify/YouTube · -19 ACX")
                                silence_start = gr.Slider(0.0, 3.0, value=s["silence_start"], step=0.1, label="Silencio inicial (s)")
                                silence_end = gr.Slider(0.0, 5.0, value=s["silence_end"], step=0.1, label="Silencio final (s)")
                            with gr.Row():
                                normalize = gr.Checkbox(value=bool(s["normalize"]), label="Normalizar volumen")
                                export_mp3 = gr.Checkbox(
                                    value=bool(s["export_mp3"]) and E.HAS_MP3, label="Exportar MP3",
                                    interactive=E.HAS_MP3, info=None if E.HAS_MP3 else "ffmpeg no encontrado",
                                )

                        with gr.Accordion("🖥️ Hardware (se aplica al cargar el modelo)", open=False):
                            with gr.Row():
                                device = gr.Dropdown(choices=device_choices, value=s["device"] if s["device"] in device_choices else "auto", label="Dispositivo")
                                dtype = gr.Dropdown(choices=["bfloat16", "float16", "float32"], value=s["dtype"], label="Precisión")
                                attn = gr.Dropdown(choices=["eager", "sdpa"], value=s["attn"], label="Atención")

            # ------------------------------------------------------------ Historial
            with gr.Tab("🕘 Historial"):
                with gr.Row():
                    with gr.Column(scale=3):
                        history_df = gr.Dataframe(
                            headers=["Fecha", "Modo", "Texto", "Duración", "Seed"],
                            datatype=["str", "str", "str", "str", "str"],
                            value=storage.history_rows(storage.list_history()),
                            interactive=False, wrap=True, column_widths=["14%", "14%", "52%", "10%", "10%"],
                        )
                    with gr.Column(scale=2):
                        hist_audio = gr.Audio(label="Reproducir", type="filepath")
                        hist_md = gr.Markdown("Selecciona una fila para ver los detalles.")
                        hist_idx = gr.State(-1)
                        with gr.Row():
                            hist_del_files = gr.Checkbox(value=False, label="Borrar también los archivos")
                        with gr.Row():
                            hist_del_btn = gr.Button("🗑️ Borrar entrada", size="sm")
                            hist_clear_btn = gr.Button("🧹 Vaciar historial", size="sm", variant="stop")
                            hist_open_btn = gr.Button("📂 Abrir carpeta", size="sm")

            # ------------------------------------------------------------ Modelos
            with gr.Tab("📦 Modelos"):
                gr.Markdown(
                    "Los modelos se descargan de Hugging Face a la caché del usuario la primera vez que se usan. "
                    "Solo un modelo permanece cargado en memoria; al cambiar de modo o tamaño se sustituye automáticamente."
                )
                models_df = gr.Dataframe(
                    headers=["Modelo", "Modo", "Tamaño", "Disco", "Estado"],
                    datatype=["str"] * 5, value=models_table(), interactive=False,
                )
                with gr.Row():
                    model_pick = gr.Dropdown(
                        choices=[m["id"].split("/")[-1] for m in config.MODELS],
                        value=config.MODELS[0]["id"].split("/")[-1], label="Descargar modelo", scale=3,
                    )
                    dl_btn = gr.Button("⬇️ Descargar ahora", scale=1)
                    unload_btn = gr.Button("🧹 Liberar memoria", scale=1)
                    refresh_btn = gr.Button("🔄 Actualizar", scale=1)
                models_status = gr.Textbox(label="Estado", interactive=False)
                gr.Markdown(
                    "**Requisitos orientativos:** 1.7B necesita ~5 GB de VRAM en bfloat16; 0.6B ~2.5 GB. "
                    "En CPU funciona con float32 pero es mucho más lento.",
                    elem_classes="small-note",
                )

        # ---------------------------------------------------------------- Eventos
        mode_radio.change(
            on_mode_change, inputs=[mode_radio, size_dd],
            outputs=[grp_design, grp_custom, grp_clone, size_dd, download_md, voice_dd],
        )
        size_dd.change(on_size_change, inputs=[mode_radio, size_dd], outputs=[download_md])
        text_tb.change(text_counter, inputs=[text_tb, chunk_chars], outputs=[counter_md], show_progress="hidden")
        chunk_chars.change(text_counter, inputs=[text_tb, chunk_chars], outputs=[counter_md], show_progress="hidden")
        txt_file.upload(read_txt, inputs=[txt_file], outputs=[text_tb])

        voice_save_btn.click(
            on_save_voice,
            inputs=[voice_name, mode_radio, voice_design_prompt, speaker_dd, instruct_tb, ref_audio, ref_text, x_vector_cb],
            outputs=[voice_dd, voice_name],
        )
        voice_dd.change(
            on_load_voice, inputs=[voice_dd],
            outputs=[voice_design_prompt, speaker_dd, instruct_tb, ref_audio, ref_text],
        ).then(on_load_voice_xvec, inputs=[voice_dd], outputs=[x_vector_cb])
        voice_del_btn.click(on_delete_voice, inputs=[voice_dd, mode_radio], outputs=[voice_dd])

        gen_inputs = [
            mode_radio, size_dd, text_tb, language_dd, voice_design_prompt, speaker_dd, instruct_tb,
            ref_audio, ref_text, x_vector_cb,
            temperature, top_p, top_k, repetition_penalty, max_new_tokens, seed,
            chunk_chars, pause_sentence, pause_paragraph,
            silence_start, silence_end, target_lufs, normalize, export_mp3,
            device, dtype, attn,
        ]
        assert len(gen_inputs) == len(GEN_INPUTS)
        gen_btn.click(
            on_generate, inputs=gen_inputs,
            outputs=[audio_out, mp3_file, status_tb, history_df, chip, language_dd, speaker_dd],
            show_progress="full", show_progress_on=[status_tb],
        )
        stop_btn.click(on_stop, outputs=[status_tb])
        open_btn.click(open_outputs_folder, outputs=[status_tb])

        history_df.select(on_history_select, outputs=[hist_audio, hist_md, hist_idx])
        hist_del_btn.click(on_history_delete, inputs=[hist_idx, hist_del_files], outputs=[history_df, hist_audio, hist_md, hist_idx])
        hist_clear_btn.click(on_history_clear, outputs=[history_df, hist_audio, hist_md, hist_idx])
        hist_open_btn.click(open_outputs_folder, outputs=[hist_md])

        dl_btn.click(on_download_model, inputs=[model_pick], outputs=[models_df, models_status],
                     show_progress_on=[models_status])
        unload_btn.click(on_unload_model, outputs=[models_df, models_status, chip])
        refresh_btn.click(on_refresh_models, outputs=[models_df, chip])

    demo.queue(default_concurrency_limit=1)
    return demo


def main(port: int = 7860, share: bool = False, inbrowser: bool = True, host: str = "127.0.0.1") -> None:
    demo = build_demo()
    demo.launch(
        server_name=host, server_port=port, share=share, inbrowser=False,
        show_api=False, show_error=True, prevent_thread_lock=True, quiet=True,
    )
    url = f"http://{host}:{port}/?__theme=dark"
    print(f"\n  {config.APP_NAME} listo en  {url}\n  (Ctrl+C para cerrar)\n", flush=True)
    if inbrowser:
        import webbrowser

        webbrowser.open(url)
    try:
        demo.block_thread()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
