# Manual de la interfaz

Describe cada control tal como aparece en la ventana, de arriba abajo.

## Cabecera

A la derecha hay un indicador con la GPU detectada, la VRAM usada y el modelo cargado en memoria.
Se actualiza al terminar cada generación.

## Pestaña Generar

### Modo
- **Diseño de voz**: creas una voz nueva describiéndola en inglés. Solo existe en tamaño 1.7B.
- **Voz predefinida**: usas una de las nueve voces de fábrica y, opcionalmente, una instrucción de estilo.
- **Clonación**: imitas la voz de un audio de referencia.

Al cambiar de modo se muestran solo los campos que aplican; el modelo correcto se elige solo.

### Modelo
`1.7B · calidad` o `0.6B · rápido` (este último solo en Voz predefinida y Clonación). Debajo se indica si ya
está descargado. Al generar, si el modelo cargado no es el que toca, se descarga de memoria y se carga el nuevo
(unos 10 segundos).

### Idioma
`Auto` deja que el modelo decida. Para audiolibros en español conviene forzar **Español** y evitar acentos
cruzados. La lista viene del propio modelo.

### Voz guardada
Desplegable con las voces guardadas para el modo actual. Al elegir una se rellenan sus campos
(descripción, hablante e instrucción, o audio de referencia y transcripción). El botón 🗑️ la borra.
Para guardar la voz actual escribe un nombre y pulsa **Guardar**. Las voces viven en la carpeta `voices/`.

### Descripción de la voz (solo Diseño de voz)
Identidad vocal, en inglés: edad, género, grosor, altura tonal, textura (rasposa, ahumada, clara), acento,
ritmo y estabilidad. Ejemplo: `A calm, warm male narrator in his 30s, Colombian Spanish, steady storytelling
pace, audiobook style`. Debe ser **exactamente la misma** en todo un proyecto: si la cambias, creas otra persona.

### Hablante e Instrucción (solo Voz predefinida)
Hablante es una identidad fija de fábrica (Ryan, Vivian, Serena, Aiden, Dylan, Eric, Ono Anna, Sohee, Uncle Fu).
La instrucción, en inglés y corta, dice **cómo** leer este texto: `calm audiobook`, `sad whisper`, `angry and fast`.
El modelo 0.6B ignora la instrucción.

### Audio de referencia, Transcripción y Solo timbre (solo Clonación)
Sube o graba un clip de 3 a 15 segundos de una sola persona, sin música ni ruido, con frases completas.
En **Transcripción** escribe literalmente lo que se dice: así el modelo imita también la entonación.
Si no tienes la transcripción marca **Solo timbre (x-vector)**: clona el color de la voz pero con menos fidelidad.
La referencia se analiza una sola vez por generación y se reutiliza en todos los fragmentos.

### Texto
El texto completo a convertir. Separa párrafos con una línea en blanco. Debajo se muestra el número de
caracteres y en cuántos **fragmentos** se dividirá. También puedes cargar un `.txt`.

Consejos: termina las frases con punto y espacio; evita pegar texto con saltos de línea a mitad de frase;
escribe los números como palabras si quieres controlar cómo se leen.

### Generar / Detener
**Generar** carga el modelo si hace falta, divide el texto en fragmentos, los sintetiza uno a uno con la misma
voz, los une con las pausas configuradas, normaliza el volumen y guarda WAV (y MP3). La barra de progreso
muestra el fragmento en curso. **Detener** cancela al terminar el fragmento actual.

### Resultado, MP3 y Estado
Reproductor con el WAV final (se reproduce solo al terminar) y botón de descarga; enlace al MP3 si está activado;
y un cuadro de estado con el nombre del archivo, duración, fragmentos, tiempo de proceso y seed usada.
**Abrir carpeta de salidas** abre `outputs/` en el Explorador.

## Avanzado

### Muestreo
- **Temperature**: variabilidad. Bajo (0.6–0.8) = lectura estable y repetible; alto = más expresivo pero con
  cambios de ritmo inesperados. Por defecto 0.9.
- **Top-p**: filtro por probabilidad acumulada. Bajo cierra el abanico y contiene la voz; alto permite más inflexiones.
- **Top-k**: filtro por cantidad de opciones. Bajo = conservador; alto = más margen actoral.
- **Repetition penalty**: penaliza repetir códigos de audio; súbelo un poco (1.1) si notas sonidos que se atascan.
- **Max tokens por fragmento**: límite de duración por fragmento (12 tokens ≈ 1 segundo). Si un fragmento se
  corta, súbelo o reduce los caracteres por fragmento.
- **Seed**: con el mismo texto, voz y seed el resultado es idéntico. `-1` elige una al azar y la muestra en Estado
  y en el historial para poder repetirla.

### Texto largo
- **Caracteres por fragmento**: tamaño máximo de cada trozo (se corta por frases, nunca por palabras).
  300 es un buen equilibrio; menos = más estable, más = menos cortes.
- **Pausa entre fragmentos** y **Pausa entre párrafos**: silencios insertados al unir.

### Post-proceso
- **Normalizar volumen** y **LUFS objetivo**: -16 es el estándar de Spotify/YouTube; -19 el de ACX/Audible.
  Nunca se recorta la señal: si el objetivo pediría superar el pico máximo, se baja la ganancia.
- **Exportar MP3**: además del WAV genera un MP3 a 192 kbps (requiere ffmpeg; se desactiva si no hay).
- **Silencio inicial / final**: los audiolibros suelen exigir 0,5 s al inicio y 1–2 s al final.

### Hardware
- **Dispositivo**: `auto` usa la GPU si existe. `cpu` fuerza procesador (float32, lento).
- **Precisión**: `bfloat16` recomendado en GPU RTX; `float16` usa algo menos de memoria; `float32` es la
  precisión completa (más memoria y tiempo).
- **Atención**: `eager` es compatible con todo; `sdpa` puede ser algo más rápido y cae a `eager` si falla.
Los cambios de hardware se aplican la próxima vez que se carga el modelo.

## Pestaña Historial

Tabla con las últimas 200 generaciones. Al pulsar una fila se carga en el reproductor y se muestran sus
detalles (modelo, seed, parámetros, ruta). **Borrar entrada** quita la fila (y los archivos si marcas la casilla),
**Vaciar historial** borra solo la lista, **Abrir carpeta** abre `outputs/`.

## Pestaña Modelos

Lista los cinco modelos oficiales con su estado (no descargado / descargado / cargado). **Descargar ahora**
baja uno por adelantado, **Liberar memoria** descarga el modelo actual de la GPU y **Actualizar** refresca la tabla.
