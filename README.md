# Qwen3-TTS Studio

Interfaz local para generar voz con los modelos **Qwen3-TTS** (Alibaba) en Windows, sin servicios externos.
Tres modos: **Diseño de voz** (describes la voz en texto), **Voz predefinida** (9 hablantes de fábrica con
instrucción de estilo) y **Clonación** (a partir de un audio de 3 a 15 segundos).

## Arranque

Doble clic en **`iniciar.bat`**. La primera vez instala todo (Python 3.11 si falta, entorno virtual,
PyTorch con CUDA si hay GPU NVIDIA, dependencias). Las siguientes veces arranca en segundos.
Se abre el navegador en `http://127.0.0.1:7860`.

Opciones (desde una consola en esta carpeta):

```
iniciar.bat --cpu          fuerza PyTorch CPU (y lo reinstala si hacía falta)
iniciar.bat --gpu          fuerza PyTorch CUDA 12.1
iniciar.bat --reinstall    reinstala las dependencias
iniciar.bat --port 8000    otro puerto
iniciar.bat --no-browser   no abre el navegador
iniciar.bat --share        enlace público temporal de Gradio
```

Los modelos se descargan de Hugging Face la primera vez que se usan (1.7B ≈ 4.2 GB, 0.6B ≈ 2 GB) y quedan
en la caché del usuario (`%USERPROFILE%\.cache\huggingface\hub`). Desde la pestaña **Modelos** se pueden
descargar por adelantado.

## Requisitos

- Windows 10/11, 16 GB de RAM recomendados.
- GPU NVIDIA con 6 GB o más de VRAM para los modelos 1.7B (probado en RTX 3060 Ti 8 GB). Sin GPU funciona
  en CPU con float32, mucho más lento.
- ffmpeg para exportar MP3: se usa el del sistema si existe (PATH o `C:\ffmpeg\bin`); si no, se usa el
  binario que trae el paquete `imageio-ffmpeg`, sin instalar nada.

## Carpetas

| Carpeta / archivo | Contenido |
|---|---|
| `outputs/` | WAV (y MP3) generados, con fecha, modo e identificador en el nombre |
| `history/history.json` | Historial de generaciones (texto, modo, seed, parámetros, rutas) |
| `voices/` | Voces guardadas: descripción, hablante+instrucción o audio de referencia+transcripción |
| `settings.json` | Últimos ajustes usados en la interfaz |
| `venv/` | Entorno virtual de Python (se puede borrar; `iniciar.bat` lo recrea) |

## Archivos del proyecto

- `iniciar.bat`: detecta Python, crea el venv y llama a `launcher.py`.
- `launcher.py`: instalación idempotente (marca de versión en `venv/.qwen_tts_stamp.json`), detección de GPU, ffmpeg y arranque.
- `app.py`: interfaz Gradio.
- `tts_engine.py`: gestor de modelo (uno solo en memoria), fragmentación de texto largo, generación y post-proceso (LUFS, silencios, MP3).
- `storage.py`: historial, voces guardadas y ajustes.
- `config.py`: catálogo de modelos y valores por defecto.
- `MANUAL.md`: explicación de cada control de la interfaz.

## Solución de problemas

- **"Sin memoria en la GPU"**: usa fragmentos más cortos (Avanzado → Caracteres por fragmento), el modelo 0.6B,
  precisión float16, o cierra otros programas que usen la GPU. En Modelos → "Liberar memoria" descarga el modelo actual.
- **Arranque lento o pip falla**: ejecuta `iniciar.bat --reinstall`. Si el venv está corrupto, borra la carpeta `venv` y vuelve a ejecutar.
- **No detecta la GPU**: comprueba que `nvidia-smi` funciona en una consola. Si instalaste los drivers después del
  venv, ejecuta `iniciar.bat --gpu`.
- **MP3 desactivado**: no se encontró ffmpeg; `iniciar.bat --reinstall` instala `imageio-ffmpeg`, que lo incluye.
- **Voz cambia entre fragmentos**: en Diseño de voz mantén la descripción idéntica y fija una seed; en Clonación la
  referencia se analiza una sola vez y se reutiliza en todos los fragmentos.
