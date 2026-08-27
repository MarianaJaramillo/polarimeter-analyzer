#!/usr/bin/env python3
"""
analizar_polarimetro.py
========================

Script de linea de comandos para analizar capturas CSV exportadas desde
WaveForms (Analog Discovery 3) del polarimetro de lamina rotante.

USO BASICO
----------
    python analizar_polarimetro.py captura1.csv

    python analizar_polarimetro.py captura1.csv captura2.csv captura3.csv \
        --delta 91.5 --salida resultados/ --historial resultados/historial.csv

Cada archivo genera:
  - Una figura PNG con el reporte completo (traza cruda, ajuste, elipse,
    y su punto en la esfera de Poincare) en la carpeta de salida.
  - Una fila nueva agregada al CSV de historial (creado si no existe),
    para que puedas ir acumulando mediciones de distintas sesiones.

Parametros de calibracion
-------------------------
--delta   : retardancia real de la lamina de cuarto de onda, EN GRADOS.
            Por defecto 90 (lamina ideal). Si calibraste tu QWP como en el
            articulo de referencia (Wilkinson et al. 2021, Rev. Sci.
            Instrum. 92, 093101), pon aqui el valor medido (p.ej. 93.0
            si mediste delta=0.5167*pi ~ 93.0 grados).
--umbral  : nivel de voltaje (en V) para detectar el flanco de subida del
            disparo en el canal 2. Si no se indica, se calcula
            automaticamente como el punto medio entre el minimo y el
            maximo de esa senal.
--canal-fotodiodo / --canal-disparo : por si en tu montaje el fotodiodo y
            el disparo estan en canales opuestos a los esperados (por
            defecto: fotodiodo=ch1, disparo=ch2, tal como los exporta
            WaveForms con Channel 1 / Channel 2).
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

from polarimeter_lib import cargar_csv_waveforms, analizar_traza_completa, detectar_disparos
from polarimeter_plots import reporte_completo


def analizar_un_archivo(ruta: str, delta_rad: float, umbral, canales_invertidos: bool,
                         carpeta_salida: str) -> dict:
    nombre = os.path.splitext(os.path.basename(ruta))[0]
    df, meta = cargar_csv_waveforms(ruta)
    t = df["t"].values
    ch1 = df["ch1"].values
    ch2 = df["ch2"].values
    if canales_invertidos:
        ch1, ch2 = ch2, ch1

    resultado = analizar_traza_completa(t, ch1, ch2, delta=delta_rad, umbral_disparo=umbral)
    t_disparos = detectar_disparos(t, ch2, umbral=umbral)

    d = resultado.derivados_medio
    s = resultado.stokes_medio

    print(f"\n=== {os.path.basename(ruta)} ===")
    if meta.fecha_hora:
        print(f"  Fecha de captura : {meta.fecha_hora}")
    print(f"  Frecuencia de giro: {resultado.f_rot:.4f} Hz  "
          f"(periodo {resultado.periodo_medio*1000:.3f} ms, {resultado.tabla.shape[0]} periodo(s) analizados)")
    print(f"  Vector de Stokes normalizado (S0=1):")
    print(f"    S1 = {s.S1:+.4f}   S2 = {s.S2:+.4f}   S3 = {s.S3:+.4f}")
    print(f"  DOP = {d['DOP']:.4f}   azimut psi = {d['psi_deg']:+.2f} deg   "
          f"elipticidad chi = {d['chi_deg']:+.2f} deg   ({d['helicidad']})")
    if d["advertencia_no_fisico"]:
        print("  ADVERTENCIA: DOP > 1.02 -> vector de Stokes no fisico.")
        print("     Posibles causas: baja relacion senal/ruido, mala fase de disparo")
        print("     (delta/offset del foto-interruptor), o delta de la QWP mal calibrado.")
    if resultado.tabla.shape[0] > 1:
        print(f"  Dispersion entre periodos (std): "
              f"s1={resultado.stokes_std[1]:.4f}  s2={resultado.stokes_std[2]:.4f}  "
              f"s3={resultado.stokes_std[3]:.4f}")

    os.makedirs(carpeta_salida, exist_ok=True)
    ruta_png = os.path.join(carpeta_salida, f"{nombre}_reporte.png")
    fig = reporte_completo(t, ch1, ch2, t_disparos, resultado)
    fig.savefig(ruta_png, dpi=130)
    print(f"  Reporte guardado en: {ruta_png}")

    ruta_tabla = os.path.join(carpeta_salida, f"{nombre}_periodos.csv")
    resultado.tabla.to_csv(ruta_tabla, index=False)

    return dict(
        archivo=os.path.basename(ruta),
        fecha_analisis=datetime.now().isoformat(timespec="seconds"),
        fecha_captura=meta.fecha_hora,
        f_rot_Hz=resultado.f_rot,
        n_periodos=resultado.tabla.shape[0],
        delta_grados=np.degrees(delta_rad),
        S1=s.S1, S2=s.S2, S3=s.S3,
        DOP=d["DOP"], psi_deg=d["psi_deg"], chi_deg=d["chi_deg"],
        helicidad=d["helicidad"], advertencia_no_fisico=d["advertencia_no_fisico"],
    )


def main():
    ap = argparse.ArgumentParser(description="Analiza capturas del polarimetro de lamina rotante.")
    ap.add_argument("archivos", nargs="+", help="Uno o mas archivos CSV exportados por WaveForms")
    ap.add_argument("--delta", type=float, default=90.0,
                    help="Retardancia real de la QWP en GRADOS (por defecto 90 = ideal)")
    ap.add_argument("--umbral", type=float, default=None,
                    help="Umbral de voltaje para el disparo (por defecto: automatico)")
    ap.add_argument("--canales-invertidos", action="store_true",
                    help="Usar si el fotodiodo esta en el canal 2 y el disparo en el canal 1")
    ap.add_argument("--salida", default="resultados_polarimetro",
                    help="Carpeta donde guardar los reportes PNG y tablas por periodo")
    ap.add_argument("--historial", default=None,
                    help="CSV donde acumular una fila de resumen por archivo analizado "
                         "(por defecto: <salida>/historial_stokes.csv)")
    args = ap.parse_args()

    delta_rad = np.radians(args.delta)
    historial = args.historial or os.path.join(args.salida, "historial_stokes.csv")

    filas = []
    for ruta in args.archivos:
        try:
            fila = analizar_un_archivo(ruta, delta_rad, args.umbral, args.canales_invertidos, args.salida)
            filas.append(fila)
        except Exception as e:
            print(f"\n!! Error analizando {ruta}: {e}", file=sys.stderr)

    if filas:
        nuevo = pd.DataFrame(filas)
        os.makedirs(os.path.dirname(historial) or ".", exist_ok=True)
        if os.path.exists(historial):
            previo = pd.read_csv(historial)
            combinado = pd.concat([previo, nuevo], ignore_index=True)
        else:
            combinado = nuevo
        combinado.to_csv(historial, index=False)
        print(f"\nHistorial actualizado ({len(combinado)} mediciones en total): {historial}")


if __name__ == "__main__":
    main()
