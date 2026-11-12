"""Instalación idempotente y arranque de Qwen3-TTS Studio.

Se ejecuta con el Python del venv (lo hace iniciar.bat). Solo usa la librería estándar
hasta que las dependencias están instaladas y verificadas.

Uso: python launcher.py [--cpu | --gpu] [--reinstall] [--port N] [--no-browser] [--share]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
VENV_DIR = BASE_DIR / "venv"
REQ_FILE = BASE_DIR / "requirements.txt"
STAMP_FILE = VENV_DIR / ".qwen_tts_stamp.json"

TORCH_VERSION = "2.5.1"
TORCH_INDEX = {
    "cu121": "https://download.pytorch.org/whl/cu121",
    "cpu": "https://download.pytorch.org/whl/cpu",
}
FFMPEG_DIRS = [r"C:\ffmpeg\bin", r"C:\ffmpeg-8.1.2-essentials_build\bin"]


def log(msg: str) -> None:
    print(f"[qwen-tts] {msg}", flush=True)


def run(cmd: list[str], **kw) -> int:
    log("> " + " ".join(cmd))
    return subprocess.call(cmd, **kw)


def has_nvidia() -> bool:
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        return subprocess.call(["nvidia-smi", "-L"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0
    except Exception:
        return False


def requirements_hash() -> str:
    return hashlib.sha256(REQ_FILE.read_bytes()).hexdigest()[:16]


def read_stamp() -> dict:
    try:
        return json.loads(STAMP_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_stamp(flavor: str) -> None:
    STAMP_FILE.write_text(json.dumps({
        "requirements": requirements_hash(),
        "torch_flavor": flavor,
        "torch_version": TORCH_VERSION,
        "python": sys.version.split()[0],
    }, indent=2), encoding="utf-8")


def verify_imports() -> tuple[bool, str]:
    code = (
        "import torch, gradio, transformers, soundfile, pyloudnorm, pydub, imageio_ffmpeg, qwen_tts;"
        "print(torch.__version__, torch.cuda.is_available())"
    )
    try:
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
    except Exception as e:
        return False, str(e)
    if out.returncode != 0:
        return False, (out.stderr or out.stdout).strip().splitlines()[-1] if (out.stderr or out.stdout).strip() else "error"
    return True, out.stdout.strip().splitlines()[-1]


def install(flavor: str) -> None:
    log(f"Instalando dependencias (torch {TORCH_VERSION} {flavor}). Puede tardar varios minutos...")
    pip = [sys.executable, "-m", "pip"]
    if run(pip + ["install", "--upgrade", "pip", "--quiet"]) != 0:
        log("Aviso: no se pudo actualizar pip, se continúa.")
    torch_cmd = pip + [
        "install", f"torch=={TORCH_VERSION}", f"torchaudio=={TORCH_VERSION}",
        "--index-url", TORCH_INDEX[flavor],
    ]
    if run(torch_cmd) != 0:
        raise SystemExit("ERROR: fallo al instalar PyTorch. Revisa tu conexión y vuelve a ejecutar iniciar.bat.")
    if run(pip + ["install", "-r", str(REQ_FILE)]) != 0:
        raise SystemExit("ERROR: fallo al instalar requirements.txt. Vuelve a ejecutar iniciar.bat (o con --reinstall).")


def ensure_installed(args: argparse.Namespace) -> None:
    if not REQ_FILE.exists():
        raise SystemExit(f"No se encuentra {REQ_FILE}")
    flavor = "cpu" if args.cpu else ("cu121" if (args.gpu or has_nvidia()) else "cpu")
    stamp = read_stamp()
    up_to_date = (
        stamp.get("requirements") == requirements_hash()
        and stamp.get("torch_flavor") == flavor
        and stamp.get("torch_version") == TORCH_VERSION
    )
    if up_to_date and not args.reinstall:
        ok, info = verify_imports()
        if ok:
            log(f"Entorno listo ({flavor}) · torch {info}")
            return
        log(f"El entorno no pasa la verificación ({info}); se reinstala.")
    elif not args.reinstall:
        if stamp:
            log("Cambiaron las dependencias o el tipo de PyTorch; se actualiza el entorno.")
        else:
            log("Primera ejecución: se instala el entorno.")
    if stamp.get("torch_flavor") not in (None, flavor):
        # Cambiar cu121 <-> cpu exige forzar la reinstalación de torch.
        run([sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchaudio"])
    install(flavor)
    ok, info = verify_imports()
    if not ok:
        raise SystemExit(f"ERROR: la verificación falló tras instalar: {info}")
    write_stamp(flavor)
    log(f"Instalación completada ({flavor}) · torch {info}")


def setup_ffmpeg() -> None:
    if shutil.which("ffmpeg"):
        return
    for d in FFMPEG_DIRS:
        if Path(d, "ffmpeg.exe").exists():
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
            return
    try:
        import imageio_ffmpeg

        os.environ["QWEN_TTS_FFMPEG"] = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        log("Aviso: ffmpeg no encontrado; la exportación a MP3 estará desactivada.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Qwen3-TTS Studio launcher")
    parser.add_argument("--cpu", action="store_true", help="Forzar PyTorch CPU")
    parser.add_argument("--gpu", action="store_true", help="Forzar PyTorch CUDA 12.1")
    parser.add_argument("--reinstall", action="store_true", help="Reinstalar dependencias")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--share", action="store_true", help="Crear enlace público de Gradio")
    parser.add_argument("--install-only", action="store_true", help="Instalar/verificar sin arrancar")
    args = parser.parse_args()

    if not Path(sys.prefix).resolve() == VENV_DIR.resolve():
        log(f"Aviso: no se está usando el venv del proyecto ({sys.prefix}).")

    os.chdir(BASE_DIR)
    ensure_installed(args)
    if args.install_only:
        return 0
    setup_ffmpeg()

    sys.path.insert(0, str(BASE_DIR))
    import app  # noqa: E402  (importa torch/gradio; solo tras verificar el entorno)

    app.main(port=args.port, share=args.share, inbrowser=not args.no_browser, host=args.host)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(0)
    except SystemExit:
        raise
    except BaseException:
        import traceback

        err_file = BASE_DIR / "launcher_error.log"
        try:
            err_file.write_text(traceback.format_exc(), encoding="utf-8")
        except Exception:
            pass
        traceback.print_exc()
        log(f"Error inesperado. Detalle guardado en {err_file}")
        raise SystemExit(1)
