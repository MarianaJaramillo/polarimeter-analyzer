"""
polarimeter_lib.py
===================

Libreria de analisis para el polarimetro de lamina rotante (rotating waveplate
polarimeter), basada en:

    Wilkinson, Maurer, Flood, Lander, Chafin & Flagg,
    "Complete Stokes vector analysis with a compact, portable rotating
    waveplate polarimeter", Rev. Sci. Instrum. 92, 093101 (2021).
    https://doi.org/10.1063/5.0052835

Modelo fisico (Eq. 1 del articulo):

    I(theta) = 1/2 * [ A + B*sin(2*theta) + C*cos(4*theta) + D*sin(4*theta) ]
    theta    = omega*t + phi

donde omega es la frecuencia angular de giro de la lamina de cuarto de onda
(QWP) y phi el desfase (fijado a 0 por el disparo del foto-interruptor al
inicio de cada periodo).

Relacion con el vector de Stokes (Eq. 2), permitiendo retardancia real
delta != pi/2:

    S0 = A - C/tan^2(delta/2)
    S1 = C/sin^2(delta/2)
    S2 = D/sin^2(delta/2)
    S3 = B/sin(delta)

Este modulo NO depende de ningun entorno grafico: solo numpy/pandas/scipy.
Es usado tanto por el script de linea de comandos (analizar_polarimetro.py)
como por la app interactiva (app_polarimetro.py).
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Carga de archivos CSV exportados por Digilent WaveForms
# ---------------------------------------------------------------------------

@dataclass
class MetadatosCaptura:
    dispositivo: Optional[str] = None
    numero_serie: Optional[str] = None
    fecha_hora: Optional[str] = None
    fs: Optional[float] = None          # frecuencia de muestreo (Hz)
    n_muestras: Optional[int] = None
    canal1_info: Optional[str] = None
    canal2_info: Optional[str] = None
    disparo_info: Optional[str] = None
    columnas: Optional[list] = None
    crudo: str = ""                     # todas las lineas de comentario, por si acaso


def cargar_csv_waveforms(ruta_o_buffer) -> tuple[pd.DataFrame, MetadatosCaptura]:
    """
    Carga un CSV exportado por WaveForms (osciloscopio Analog Discovery 3).

    Acepta una ruta de archivo (str/Path) o un objeto tipo archivo/bytes
    (util para `st.file_uploader` de Streamlit).

    Devuelve
    --------
    df : DataFrame con columnas ['t', 'ch1', 'ch2'] (unicamente las 2
         primeras columnas de canal, sin importar sus nombres originales).
    meta : MetadatosCaptura con la info de las lineas '#...' del encabezado.
    """
    if hasattr(ruta_o_buffer, "read"):
        contenido = ruta_o_buffer.read()
        if isinstance(contenido, bytes):
            contenido = contenido.decode("utf-8", errors="replace")
        texto = contenido
    else:
        with open(ruta_o_buffer, "r", encoding="utf-8", errors="replace") as f:
            texto = f.read()

    lineas = texto.splitlines()
    lineas_comentario = [l for l in lineas if l.strip().startswith("#")]
    meta = MetadatosCaptura(crudo="\n".join(lineas_comentario))

    for l in lineas_comentario:
        l = l.lstrip("#").strip()
        if l.lower().startswith("device name"):
            meta.dispositivo = l.split(":", 1)[1].strip()
        elif l.lower().startswith("serial number"):
            meta.numero_serie = l.split(":", 1)[1].strip()
        elif l.lower().startswith("date time"):
            meta.fecha_hora = l.split(":", 1)[1].strip()
        elif l.lower().startswith("sample rate"):
            m = re.search(r"([\d.]+)\s*Hz", l, re.IGNORECASE)
            if m:
                meta.fs = float(m.group(1))
        elif l.lower().startswith("samples"):
            m = re.search(r"(\d+)", l)
            if m:
                meta.n_muestras = int(m.group(1))
        elif l.lower().startswith("channel 1"):
            meta.canal1_info = l
        elif l.lower().startswith("channel 2"):
            meta.canal2_info = l
        elif l.lower().startswith("trigger"):
            meta.disparo_info = l

    # Buscar la linea de encabezado real (la que no empieza con '#' y tiene comas)
    df = pd.read_csv(io.StringIO(texto), comment="#", skip_blank_lines=True)
    df = df.dropna(how="all")
    meta.columnas = list(df.columns)

    # Renombrar las 3 primeras columnas a nombres estandar: t, ch1, ch2
    cols = list(df.columns)
    ren = {}
    if len(cols) >= 1:
        ren[cols[0]] = "t"
    if len(cols) >= 2:
        ren[cols[1]] = "ch1"
    if len(cols) >= 3:
        ren[cols[2]] = "ch2"
    df = df.rename(columns=ren)[["t", "ch1", "ch2"]].reset_index(drop=True)
    df = df.astype(float)

    if meta.fs is None and len(df) > 1:
        meta.fs = 1.0 / np.median(np.diff(df["t"].values))
    if meta.n_muestras is None:
        meta.n_muestras = len(df)

    return df, meta


# ---------------------------------------------------------------------------
# 2. Deteccion del disparo (trigger) y frecuencia de rotacion
# ---------------------------------------------------------------------------

def detectar_disparos(t: np.ndarray, trigger: np.ndarray,
                       umbral: Optional[float] = None,
                       fraccion_umbral: float = 0.5) -> np.ndarray:
    """
    Detecta los flancos de subida de la senal de disparo (foto-interruptor),
    que marcan el inicio de cada periodo de rotacion de la lamina (theta=0).

    Parametros
    ----------
    t, trigger : arreglos de tiempo y voltaje del canal de disparo.
    umbral : nivel de voltaje para considerar "alto". Si es None, se calcula
        automaticamente como min + fraccion_umbral*(max-min).
    fraccion_umbral : fraccion (0-1) usada para el umbral automatico.

    Devuelve
    --------
    t_disparos : arreglo con los tiempos de cada flanco de subida.
    """
    trigger = np.asarray(trigger, dtype=float)
    t = np.asarray(t, dtype=float)

    if umbral is None:
        lo, hi = np.min(trigger), np.max(trigger)
        umbral = lo + fraccion_umbral * (hi - lo)

    por_encima = trigger > umbral
    idx_subida = np.where((~por_encima[:-1]) & (por_encima[1:]))[0] + 1

    if len(idx_subida) == 0:
        return np.array([])

    # Interpolacion lineal para refinar el instante exacto del cruce
    t_disparos = []
    for i in idx_subida:
        t0, t1 = t[i - 1], t[i]
        v0, v1 = trigger[i - 1], trigger[i]
        if v1 == v0:
            t_cruce = t1
        else:
            t_cruce = t0 + (umbral - v0) * (t1 - t0) / (v1 - v0)
        t_disparos.append(t_cruce)

    return np.array(t_disparos)


def frecuencia_rotacion(t_disparos: np.ndarray) -> dict:
    """
    Calcula la frecuencia angular de rotacion a partir de los tiempos de
    disparo consecutivos.

    Devuelve un dict con: n_periodos, periodo_medio, periodo_std, f_rot,
    omega, periodos (arreglo de duraciones individuales).
    """
    if len(t_disparos) < 2:
        return dict(n_periodos=0, periodo_medio=np.nan, periodo_std=np.nan,
                    f_rot=np.nan, omega=np.nan, periodos=np.array([]))

    periodos = np.diff(t_disparos)
    periodo_medio = float(np.mean(periodos))
    periodo_std = float(np.std(periodos))
    f_rot = 1.0 / periodo_medio
    omega = 2 * np.pi * f_rot

    return dict(n_periodos=len(periodos), periodo_medio=periodo_medio,
                periodo_std=periodo_std, f_rot=f_rot, omega=omega,
                periodos=periodos)


# ---------------------------------------------------------------------------
# 3. Ajuste de los coeficientes de Fourier (A, B, C, D) por minimos cuadrados
# ---------------------------------------------------------------------------

@dataclass
class ResultadoAjuste:
    A: float
    B: float
    C: float
    D: float
    r2: float
    n_muestras: int


def ajustar_coeficientes(t: np.ndarray, señal: np.ndarray, t0: float,
                          omega: float) -> ResultadoAjuste:
    """
    Ajusta la senal del fotodiodo al modelo (Eq. 1 del articulo):

        I(theta) = 1/2*[A + B*sin(2theta) + C*cos(4theta) + D*sin(4theta)]
        theta = omega*(t - t0)

    mediante minimos cuadrados lineales.

    t0 es el tiempo del disparo que define theta=0 (phi=0).
    """
    t = np.asarray(t, dtype=float)
    señal = np.asarray(señal, dtype=float)
    theta = omega * (t - t0)

    M = np.column_stack([
        np.ones_like(theta),
        np.sin(2 * theta),
        np.cos(4 * theta),
        np.sin(4 * theta),
    ])

    coef, *_ = np.linalg.lstsq(M, señal, rcond=None)
    a, b, c, d = coef
    A, B, C, D = 2 * a, 2 * b, 2 * c, 2 * d

    ajuste = M @ coef
    ss_res = np.sum((señal - ajuste) ** 2)
    ss_tot = np.sum((señal - np.mean(señal)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return ResultadoAjuste(A=A, B=B, C=C, D=D, r2=float(r2), n_muestras=len(t))


# ---------------------------------------------------------------------------
# 4. Coeficientes -> vector de Stokes
# ---------------------------------------------------------------------------

@dataclass
class VectorStokes:
    S0: float
    S1: float
    S2: float
    S3: float

    @property
    def normalizado(self):
        s0 = self.S0 if self.S0 != 0 else np.nan
        return (self.S1 / s0, self.S2 / s0, self.S3 / s0)


def stokes_desde_coeficientes(A: float, B: float, C: float, D: float,
                               delta: float = np.pi / 2) -> VectorStokes:
    """
    Calcula el vector de Stokes (Eq. 2 del articulo) a partir de los
    coeficientes de Fourier A, B, C, D y la retardancia real `delta`
    (en radianes) de la lamina de cuarto de onda. Por defecto delta=pi/2
    (lamina ideal).
    """
    S0 = A - C / np.tan(delta / 2) ** 2
    S1 = C / np.sin(delta / 2) ** 2
    S2 = D / np.sin(delta / 2) ** 2
    S3 = B / np.sin(delta)
    return VectorStokes(S0=S0, S1=S1, S2=S2, S3=S3)


def calibrar_delta_optimo(A: float, B: float, C: float, D: float,
                           delta_min_deg: float = 60.0,
                           delta_max_deg: float = 120.0) -> dict:
    """
    Encuentra la retardancia delta (en grados) que produce DOP = 1 para
    los coeficientes de Fourier dados, asumiendo que la fuente es
    completamente polarizada (DOP_real = 1).

    Util para calibrar la QWP: mide con una fuente de polarizacion conocida
    (p. ej., laser + polarizador lineal), luego llama esta funcion con los
    coeficientes promedio y usa el delta_opt_deg resultante en la app.

    Devuelve
    --------
    delta_opt_deg : retardancia optima en grados.
    delta_opt_rad : idem en radianes.
    dop_en_opt    : DOP calculado con ese delta (deberia ser ~1.0).
    """
    deltas = np.linspace(np.radians(delta_min_deg), np.radians(delta_max_deg), 10_000)

    with np.errstate(divide='ignore', invalid='ignore'):
        tan2 = np.tan(deltas / 2) ** 2
        sin2 = np.sin(deltas / 2) ** 2
        sind = np.sin(deltas)
        S0 = A - C / tan2
        S1 = C / sin2
        S2 = D / sin2
        S3 = B / sind
        dop = np.where(S0 > 0, np.sqrt(S1**2 + S2**2 + S3**2) / S0, np.inf)

    idx = int(np.argmin(np.abs(dop - 1.0)))
    return dict(
        delta_opt_deg=float(np.degrees(deltas[idx])),
        delta_opt_rad=float(deltas[idx]),
        dop_en_opt=float(dop[idx]),
    )


def parametros_derivados(s: VectorStokes) -> dict:
    """
    Cantidades derivadas del vector de Stokes normalizado:
      - grado de polarizacion (DOP)
      - azimut de la elipse de polarizacion, psi (grados, en [-90,90))
      - angulo de elipticidad, chi (grados, en [-45,45])
      - helicidad (R = dextrogira / L = levogira / lineal)
      - advertencia_no_fisico : True si DOP > 1.02
    """
    S0, S1, S2, S3 = s.S0, s.S1, s.S2, s.S3
    if S0 == 0 or np.isnan(S0):
        return dict(DOP=np.nan, psi_deg=np.nan, chi_deg=np.nan, helicidad="?",
                    advertencia_no_fisico=True)

    s1, s2, s3 = S1 / S0, S2 / S0, S3 / S0
    dop = np.sqrt(s1 ** 2 + s2 ** 2 + s3 ** 2)

    psi = 0.5 * np.degrees(np.arctan2(s2, s1))          # longitud / azimut
    denom = np.sqrt(s1 ** 2 + s2 ** 2)
    chi = 0.5 * np.degrees(np.arctan2(s3, denom)) if denom > 0 else np.sign(s3) * 45.0

    if s3 > 1e-3:
        helicidad = "dextrogira (R)"
    elif s3 < -1e-3:
        helicidad = "levogira (L)"
    else:
        helicidad = "lineal"

    return dict(DOP=dop, psi_deg=psi, chi_deg=chi, helicidad=helicidad,
                advertencia_no_fisico=bool(dop > 1.02))


# ---------------------------------------------------------------------------
# 5. Analisis de un periodo y de la traza completa (multiples periodos)
# ---------------------------------------------------------------------------

@dataclass
class ResultadoPeriodo:
    idx: int
    t_inicio: float
    t_fin: float
    ajuste: ResultadoAjuste
    stokes: VectorStokes
    derivados: dict


@dataclass
class ResultadoAnalisis:
    periodos: list                 # lista de ResultadoPeriodo
    f_rot: float
    omega: float
    periodo_medio: float
    periodo_std: float
    stokes_medio: VectorStokes
    stokes_std: tuple              # (S0,S1,S2,S3) desviaciones estandar normalizadas
    derivados_medio: dict
    tabla: pd.DataFrame             # una fila por periodo, para exportar/graficar


def analizar_periodo(t: np.ndarray, señal: np.ndarray, t_ini: float, t_fin: float,
                      omega: float, delta: float = np.pi / 2, idx: int = 0) -> ResultadoPeriodo:
    mascara = (t >= t_ini) & (t < t_fin)
    ajuste = ajustar_coeficientes(t[mascara], señal[mascara], t0=t_ini, omega=omega)
    stokes = stokes_desde_coeficientes(ajuste.A, ajuste.B, ajuste.C, ajuste.D, delta=delta)
    derivados = parametros_derivados(stokes)
    return ResultadoPeriodo(idx=idx, t_inicio=t_ini, t_fin=t_fin, ajuste=ajuste,
                             stokes=stokes, derivados=derivados)


def analizar_traza_completa(t: np.ndarray, ch1: np.ndarray, ch2: np.ndarray,
                             delta: float = np.pi / 2,
                             umbral_disparo: Optional[float] = None,
                             fraccion_umbral: float = 0.5) -> ResultadoAnalisis:
    """
    Flujo completo: detecta disparos en ch2, calcula la frecuencia de
    rotacion, y ajusta el vector de Stokes en cada periodo delimitado por
    dos disparos consecutivos (theta=0 al inicio de cada uno). Si solo hay
    un periodo completo, se reporta ese unico resultado sin estadistica de
    dispersion (igual que si hubiera varios, pero con std = NaN).
    """
    t = np.asarray(t, dtype=float)
    ch1 = np.asarray(ch1, dtype=float)
    ch2 = np.asarray(ch2, dtype=float)

    t_disparos = detectar_disparos(t, ch2, umbral=umbral_disparo, fraccion_umbral=fraccion_umbral)
    if len(t_disparos) < 2:
        raise ValueError(
            "No se detectaron al menos 2 disparos en el canal de trigger; "
            "no es posible delimitar un periodo completo. Revisa el umbral "
            "de disparo o el canal seleccionado."
        )

    info_f = frecuencia_rotacion(t_disparos)
    omega = info_f["omega"]

    periodos = []
    for i in range(len(t_disparos) - 1):
        r = analizar_periodo(t, ch1, t_disparos[i], t_disparos[i + 1], omega=omega,
                              delta=delta, idx=i)
        periodos.append(r)

    filas = []
    for r in periodos:
        filas.append(dict(
            periodo=r.idx, t_inicio=r.t_inicio, t_fin=r.t_fin,
            A=r.ajuste.A, B=r.ajuste.B, C=r.ajuste.C, D=r.ajuste.D, R2=r.ajuste.r2,
            S0=r.stokes.S0, S1=r.stokes.S1, S2=r.stokes.S2, S3=r.stokes.S3,
            s1=r.stokes.S1 / r.stokes.S0 if r.stokes.S0 else np.nan,
            s2=r.stokes.S2 / r.stokes.S0 if r.stokes.S0 else np.nan,
            s3=r.stokes.S3 / r.stokes.S0 if r.stokes.S0 else np.nan,
            DOP=r.derivados["DOP"], psi_deg=r.derivados["psi_deg"],
            chi_deg=r.derivados["chi_deg"], helicidad=r.derivados["helicidad"],
        ))
    tabla = pd.DataFrame(filas)

    s1_m, s2_m, s3_m = tabla["s1"].mean(), tabla["s2"].mean(), tabla["s3"].mean()
    s1_s, s2_s, s3_s = tabla["s1"].std(), tabla["s2"].std(), tabla["s3"].std()
    stokes_medio = VectorStokes(S0=1.0, S1=s1_m, S2=s2_m, S3=s3_m)
    derivados_medio = parametros_derivados(stokes_medio)

    return ResultadoAnalisis(
        periodos=periodos, f_rot=info_f["f_rot"], omega=info_f["omega"],
        periodo_medio=info_f["periodo_medio"], periodo_std=info_f["periodo_std"],
        stokes_medio=stokes_medio, stokes_std=(np.nan, s1_s, s2_s, s3_s),
        derivados_medio=derivados_medio, tabla=tabla,
    )
