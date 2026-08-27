# Analizador del Polarímetro de Lámina Rotante

Herramienta en Python para analizar capturas CSV exportadas desde **WaveForms** (Analog Discovery 3) de un polarímetro casero de lámina rotante.

Calcula el **vector de Stokes completo** (S0, S1, S2, S3), el **grado de polarización (DOP)**, el **azimut ψ** y la **elipticidad χ** a partir de un ajuste de Fourier de la señal del fotodiodo.

Basado en:
> Wilkinson, Maurer, Flood, Lander, Chafin & Flagg, *"Complete Stokes vector
> analysis with a compact, portable rotating waveplate polarimeter"*,
> Rev. Sci. Instrum. 92, 093101 (2021).

---

## Tabla de contenidos

1. [Requisitos](#requisitos)
2. [Instalación](#instalación)
3. [Cómo correr el proyecto](#cómo-correr-el-proyecto)
   - [Opción A — App interactiva (recomendada)](#opción-a--app-interactiva-recomendada)
   - [Opción B — Script de consola](#opción-b--script-de-consola)
4. [Formato del archivo CSV de entrada](#formato-del-archivo-csv-de-entrada)
5. [Parámetros de calibración](#parámetros-de-calibración)
6. [Descripción de los archivos](#descripción-de-los-archivos)
7. [Cómo se calcula el vector de Stokes](#cómo-se-calcula-el-vector-de-stokes)
8. [Solución de problemas comunes](#solución-de-problemas-comunes)

---

## Requisitos

- Python 3.9 o superior
- Las librerías listadas en `requirements.txt`:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `streamlit`

---

## Instalación

**1. Clona o descarga este repositorio** en tu máquina.

**2. (Recomendado) Crea un entorno virtual:**

```bash
python -m venv venv
```

Actívalo:

- Windows:
  ```bash
  venv\Scripts\activate
  ```
- macOS / Linux:
  ```bash
  source venv/bin/activate
  ```

**3. Instala las dependencias:**

```bash
pip install -r requirements.txt
```

---

## Cómo correr el proyecto

### Opción A — App interactiva (recomendada)

Ideal para uso diario en el laboratorio. Abre una interfaz web donde puedes subir archivos, ajustar la calibración con sliders y ver los resultados al instante.

```bash
streamlit run app_polarimetro.py
```

Se abrirá automáticamente en tu navegador en `http://localhost:8501`.

**Flujo de uso en la app:**

1. **Sube** uno o varios archivos CSV con el botón de carga (o arrastra y suelta).
2. **Ajusta los parámetros de calibración** en la barra lateral izquierda:
   - **δ (delta)** — retardancia real de tu QWP en grados. Usa `90` para una lámina ideal; si ya calibraste tu QWP según el artículo, ingresa el valor medido (p. ej. `93.0`).
   - **Umbral de disparo** — en modo automático se calcula solo; activa el modo manual solo si la señal de disparo es muy ruidosa.
   - **Invertir canales** — actívalo si conectaste el fotodiodo al canal 2 y el disparo al canal 1 (al revés del estándar).
3. **Revisa los resultados** para cada archivo:
   - Vector de Stokes normalizado (S1, S2, S3)
   - Grado de polarización (DOP)
   - Azimut ψ y elipticidad χ
   - Elipse de polarización y esfera de Poincaré
   - Traza cruda con el ajuste de Fourier superpuesto
4. **Descarga el historial** de la sesión (tabla + Poincaré acumulada) como CSV con el botón de descarga al final de la página.

---

### Opción B — Script de consola

Ideal para procesar múltiples archivos en lote o automatizar el análisis.

**Uso básico (un solo archivo):**

```bash
python analizar_polarimetro.py captura1.csv
```

**Varios archivos a la vez:**

```bash
python analizar_polarimetro.py captura1.csv captura2.csv captura3.csv
```

**Con parámetros de calibración:**

```bash
python analizar_polarimetro.py *.csv --delta 93.0 --umbral 1.0 --salida resultados/
```

**Todos los argumentos disponibles:**

| Argumento | Descripción | Valor por defecto |
|---|---|---|
| `--delta` | Retardancia del QWP en grados | `90.0` |
| `--umbral` | Umbral de voltaje (V) para detección del disparo | automático |
| `--salida` | Carpeta donde se guardan los reportes PNG y CSV | directorio actual |
| `--historial` | Ruta del CSV de historial acumulado | `historial_stokes.csv` |
| `--canal-fotodiodo` | Número de canal del fotodiodo (`1` o `2`) | `1` |
| `--canal-disparo` | Número de canal del disparo (`1` o `2`) | `2` |

**Salidas generadas por archivo:**

- `resultados/<nombre>_reporte.png` — figura completa con traza cruda, ajuste, elipse de polarización y posición en la esfera de Poincaré.
- `resultados/<nombre>_periodos.csv` — tabla con los coeficientes de Fourier y el vector de Stokes de cada periodo analizado.
- `resultados/historial_stokes.csv` — fila acumulada en el historial de sesiones.

---

## Formato del archivo CSV de entrada

El CSV debe ser exportado desde **WaveForms** con dos canales:

| Canal | Señal |
|---|---|
| Channel 1 (`ch1`) | Fotodiodo (intensidad óptica) |
| Channel 2 (`ch2`) | Pulso de disparo (un flanco por revolución de la lámina) |

La primera fila puede ser de encabezado o datos numéricos directos; el script detecta ambos formatos automáticamente.

---

## Parámetros de calibración

| Parámetro | Descripción |
|---|---|
| **δ (delta)** | Retardancia real de la lámina de cuarto de onda (QWP) en grados. Una lámina ideal tiene δ = 90°; la tuya puede variar. Mídela con luz X-polarizada conocida como describe la sección III del artículo. |
| **Umbral de disparo** | Nivel de voltaje (V) para detectar el flanco de subida del canal de disparo. El modo automático usa el punto medio entre el mínimo y el máximo de la señal, lo que funciona bien en la mayoría de los casos. |
| **Canales** | Por defecto: fotodiodo en CH1, disparo en CH2. Cámbialos si tu montaje está al revés. |

---

## Descripción de los archivos

| Archivo | Función |
|---|---|
| `polarimeter_lib.py` | Núcleo del análisis: carga el CSV, detecta el disparo, ajusta el modelo de Fourier y calcula el vector de Stokes. Sin dependencias gráficas (importable como librería). |
| `polarimeter_plots.py` | Módulo de visualización: señales crudas, ajuste por periodo, elipse de polarización y esfera de Poincaré. |
| `analizar_polarimetro.py` | Script de línea de comandos para procesar uno o varios CSV y guardar reportes. |
| `app_polarimetro.py` | Aplicación web interactiva construida con Streamlit. |
| `requirements.txt` | Lista de dependencias de Python. |

---

## Cómo se calcula el vector de Stokes

1. **Detección del disparo:** se identifican los flancos de subida del canal 2 → cada flanco define θ = 0 al inicio de una rotación completa de la lámina.

2. **Ajuste de Fourier por periodo:** con la frecuencia de giro ω conocida, se ajusta por mínimos cuadrados la señal del fotodiodo de cada periodo al modelo:

   $$I(\theta) = \tfrac{1}{2}\bigl[A + B\sin(2\theta) + C\cos(4\theta) + D\sin(4\theta)\bigr]$$

3. **Conversión a Stokes** usando la retardancia real δ de la QWP (Ec. 1–2 del artículo):

   $$S_0 = A - \frac{C}{\tan^2(\delta/2)}, \quad S_1 = \frac{C}{\sin^2(\delta/2)}, \quad S_2 = \frac{D}{\sin^2(\delta/2)}, \quad S_3 = \frac{B}{\sin\delta}$$

4. **Promedio entre periodos:** si la captura contiene varios periodos, se promedian los coeficientes y se reporta la desviación estándar entre periodos (como en la Fig. 3 del artículo).

---

## Solución de problemas comunes

**DOP > 1 (vector de Stokes no físico)**
Indica ruido relativo alto o desfase de disparo sin calibrar. Revisa:
- Ganancia y offset del canal del fotodiodo en WaveForms (el offset de 50 mV del canal 1 puede afectar el análisis).
- Iluminación: asegúrate de que la señal del fotodiodo tenga buena amplitud.
- Calibra el desfase φ con luz linealmente polarizada conocida (sección III del artículo).

**Solo un periodo en la captura**
Si la ventana de captura en WaveForms es muy corta (< 100 ms a ~13 Hz), el script analiza solo un periodo y no puede reportar incertidumbre. Aumenta la duración de la captura a varias decenas de periodos para obtener estadísticas.

**Error al cargar el CSV**
Verifica que el archivo fue exportado desde WaveForms con dos canales activos y que el separador de decimales es el punto (`.`), no la coma.

**La app Streamlit no abre el navegador**
Abre manualmente `http://localhost:8501` en tu navegador, o ejecuta:

```bash
streamlit run app_polarimetro.py --server.headless false
```
