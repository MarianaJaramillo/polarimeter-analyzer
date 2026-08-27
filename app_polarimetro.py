"""
app_polarimetro.py
====================

App interactiva (Streamlit) para el analisis del polarimetro de lamina
rotante. Pensada para un flujo de trabajo tipo:

    1. Capturas con WaveForms -> exportas CSV.
    2. Subes el CSV aqui (o varios a la vez).
    3. Ajustas la calibracion (delta de la QWP, umbral de disparo) si hace falta.
    4. Ves el vector de Stokes, la elipse de polarizacion y la esfera de
       Poincare al instante.
    5. La app va acumulando un historial de todas las mediciones de la
       sesion, descargable como CSV, y las muestra juntas en la esfera de
       Poincare para comparar mediciones/calibraciones.

Para correrla:
    pip install streamlit
    streamlit run app_polarimetro.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from polarimeter_lib import (cargar_csv_waveforms, analizar_traza_completa,
                             detectar_disparos, calibrar_delta_optimo)
from polarimeter_plots import (
    graficar_traza_cruda,
    graficar_ajuste_periodo,
    graficar_elipse_polarizacion,
    graficar_poincare,
)

st.set_page_config(page_title="Polarímetro - Stokes", layout="wide")

if "historial" not in st.session_state:
    st.session_state.historial = pd.DataFrame()

st.title("🔬 Analizador de polarímetro de lámina rotante")
st.caption(
    "Basado en Wilkinson et al., *Rev. Sci. Instrum.* 92, 093101 (2021) — "
    "análisis de Fourier de la señal del fotodiodo sincronizado con el disparo "
    "del foto-interruptor."
)

# ------------------------------------------------------------------ Sidebar
with st.sidebar:
    st.header("Calibración")
    delta_grados = st.number_input(
        "Retardancia real δ de la QWP (grados)", min_value=1.0, max_value=179.0,
        value=90.0, step=0.1,
        help="90° = lámina ideal. Usa el valor medido si calibraste tu QWP "
             "siguiendo el procedimiento del artículo de referencia.",
    )
    modo_umbral = st.radio("Umbral de disparo (canal 2)", ["Automático", "Manual"], horizontal=True)
    umbral_manual = None
    if modo_umbral == "Manual":
        umbral_manual = st.number_input("Umbral (V)", value=1.0, step=0.1)

    canales_invertidos = st.checkbox(
        "Fotodiodo en canal 2 / disparo en canal 1 (invertir)", value=False
    )

    st.divider()
    st.header("Captura(s)")
    archivos = st.file_uploader(
        "Sube uno o varios CSV exportados de WaveForms",
        type=["csv"], accept_multiple_files=True,
    )

    st.divider()
    if st.button("🗑️ Borrar historial de esta sesión"):
        st.session_state.historial = pd.DataFrame()
        st.rerun()

if not archivos:
    st.info("⬅️ Sube una o más capturas CSV desde la barra lateral para comenzar.")
    st.stop()

delta_rad = np.radians(delta_grados)

# ------------------------------------------------------------------ Analisis
tabs = st.tabs([a.name for a in archivos])

nuevas_filas = []

for tab, archivo in zip(tabs, archivos):
    with tab:
        try:
            archivo.seek(0)
            df, meta = cargar_csv_waveforms(archivo)
        except Exception as e:
            st.error(f"No se pudo leer el archivo: {e}")
            continue

        t = df["t"].values
        ch1 = df["ch1"].values
        ch2 = df["ch2"].values
        if canales_invertidos:
            ch1, ch2 = ch2, ch1

        try:
            resultado = analizar_traza_completa(
                t, ch1, ch2, delta=delta_rad,
                umbral_disparo=umbral_manual if modo_umbral == "Manual" else None,
            )
        except ValueError as e:
            st.error(str(e))
            continue

        t_disparos = detectar_disparos(
            t, ch2, umbral=umbral_manual if modo_umbral == "Manual" else None
        )
        s = resultado.stokes_medio
        d = resultado.derivados_medio

        col_meta, col_stokes = st.columns([1, 2])
        with col_meta:
            st.markdown("**Metadatos de la captura**")
            st.write(f"Dispositivo: {meta.dispositivo or '—'}")
            st.write(f"Fecha: {meta.fecha_hora or '—'}")
            st.write(f"Frecuencia de muestreo: {meta.fs:,.0f} Hz" if meta.fs else "")
            st.write(f"Frecuencia de giro medida: **{resultado.f_rot:.4f} Hz**")
            st.write(f"Periodos analizados: **{resultado.tabla.shape[0]}**")

        with col_stokes:
            st.markdown("**Vector de Stokes normalizado (S0 = 1)**")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("S1", f"{s.S1:+.3f}")
            c2.metric("S2", f"{s.S2:+.3f}")
            c3.metric("S3", f"{s.S3:+.3f}")
            c4.metric("DOP", f"{d['DOP']:.3f}")
            st.write(
                f"Azimut ψ = **{d['psi_deg']:+.2f}°**  ·  "
                f"Elipticidad χ = **{d['chi_deg']:+.2f}°**  ·  "
                f"Helicidad: **{d['helicidad']}**"
            )
            if resultado.tabla.shape[0] > 1:
                st.caption(
                    f"Dispersión entre periodos (±1σ): "
                    f"s1={resultado.stokes_std[1]:.3f}  "
                    f"s2={resultado.stokes_std[2]:.3f}  "
                    f"s3={resultado.stokes_std[3]:.3f}"
                )
            if d["advertencia_no_fisico"]:
                r_ref = resultado.periodos[0].ajuste
                cal = calibrar_delta_optimo(r_ref.A, r_ref.B, r_ref.C, r_ref.D)
                st.warning(
                    f"⚠️ DOP = {d['DOP']:.3f} > 1 — el vector de Stokes cae fuera de la "
                    f"esfera de Poincaré.  \n"
                    f"**Causa más probable**: la retardancia real de tu QWP no es exactamente 90°.  \n"
                    f"**Prueba**: cambia δ en la barra lateral a **{cal['delta_opt_deg']:.1f}°** "
                    f"(estimado desde esta medición)."
                )

        st.divider()
        g1, g2, g3 = st.columns(3)
        with g1:
            fig = graficar_traza_cruda(t, ch1, ch2, t_disparos=t_disparos)
            st.pyplot(fig, clear_figure=True)
        with g2:
            fig = graficar_ajuste_periodo(t, ch1, resultado.periodos[0], resultado.omega)
            st.pyplot(fig, clear_figure=True)
        with g3:
            fig = graficar_elipse_polarizacion(d["psi_deg"], d["chi_deg"], dop=min(d["DOP"], 1.0))
            st.pyplot(fig, clear_figure=True)

        st.markdown("**Detalle por periodo**")
        st.dataframe(resultado.tabla, use_container_width=True, hide_index=True)

        nuevas_filas.append(dict(
            archivo=archivo.name, fecha_captura=meta.fecha_hora,
            f_rot_Hz=resultado.f_rot, n_periodos=resultado.tabla.shape[0],
            delta_grados=delta_grados,
            S1=s.S1, S2=s.S2, S3=s.S3, DOP=d["DOP"],
            psi_deg=d["psi_deg"], chi_deg=d["chi_deg"], helicidad=d["helicidad"],
            advertencia_no_fisico=d["advertencia_no_fisico"],
        ))

# ------------------------------------------------------------------ Historial + Poincare conjunta
if nuevas_filas:
    nuevo_df = pd.DataFrame(nuevas_filas)
    existentes = set(st.session_state.historial.get("archivo", pd.Series(dtype=str)))
    a_agregar = nuevo_df[~nuevo_df["archivo"].isin(existentes)]
    if not a_agregar.empty:
        st.session_state.historial = pd.concat(
            [st.session_state.historial, a_agregar], ignore_index=True
        )

st.divider()
st.header("📈 Historial de la sesión")
if st.session_state.historial.empty:
    st.write("Todavía no hay mediciones acumuladas.")
else:
    st.dataframe(st.session_state.historial, use_container_width=True, hide_index=True)

    puntos = st.session_state.historial[["S1", "S2", "S3"]].values
    fig_poincare = graficar_poincare(puntos)
    st.pyplot(fig_poincare, clear_figure=True)

    csv_bytes = st.session_state.historial.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Descargar historial (CSV)", data=csv_bytes,
        file_name="historial_stokes.csv", mime="text/csv",
    )
