"""
app_polarimeter.py
====================

Interactive (Streamlit) app for analyzing the rotating-waveplate
polarimeter. Intended for a workflow such as:

    1. Capture with WaveForms -> export CSV.
    2. Upload the CSV here (one or several at once).
    3. Adjust the calibration (QWP delta, trigger threshold) if needed.
    4. View the Stokes vector, the polarization ellipse, and the Poincare
       sphere instantly.
    5. The app accumulates a history of all measurements in the session,
       downloadable as CSV, and shows them together on the Poincare
       sphere to compare measurements/calibrations.

To run it:
    pip install streamlit
    streamlit run app_polarimeter.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from polarimeter_lib import (load_waveforms_csv, analyze_full_trace,
                             detect_triggers, calibrate_optimal_delta)
from polarimeter_plots import (
    plot_raw_trace,
    plot_period_fit,
    plot_polarization_ellipse,
    plot_poincare_sphere,
)

st.set_page_config(page_title="Polarimeter - Stokes", layout="wide")

if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame()

st.title("🔬 Rotating-waveplate polarimeter analyzer")
st.caption(
    "Based on Wilkinson et al., *Rev. Sci. Instrum.* 92, 093101 (2021) — "
    "Fourier analysis of the photodiode signal synchronized with the "
    "photo-interrupter trigger."
)

# ------------------------------------------------------------------ Sidebar
with st.sidebar:
    st.header("Calibration")
    delta_deg = st.number_input(
        "Real QWP retardance δ (degrees)", min_value=1.0, max_value=179.0,
        value=90.0, step=0.1,
        help="90° = ideal waveplate. Use the measured value if you "
             "calibrated your QWP following the reference paper's procedure.",
    )
    threshold_mode = st.radio("Trigger threshold (channel 2)", ["Automatic", "Manual"], horizontal=True)
    manual_threshold = None
    if threshold_mode == "Manual":
        manual_threshold = st.number_input("Threshold (V)", value=1.0, step=0.1)

    channels_swapped = st.checkbox(
        "Photodiode on channel 2 / trigger on channel 1 (swap)", value=False
    )

    st.divider()
    st.header("Capture(s)")
    files = st.file_uploader(
        "Upload one or more CSV files exported from WaveForms",
        type=["csv"], accept_multiple_files=True,
    )

    st.divider()
    if st.button("🗑️ Clear session history"):
        st.session_state.history = pd.DataFrame()
        st.rerun()

if not files:
    st.info("⬅️ Upload one or more CSV captures from the sidebar to get started.")
    st.stop()

delta_rad = np.radians(delta_deg)

# ------------------------------------------------------------------ Analysis
tabs = st.tabs([f.name for f in files])

new_rows = []

for tab, file in zip(tabs, files):
    with tab:
        try:
            file.seek(0)
            df, meta = load_waveforms_csv(file)
        except Exception as e:
            st.error(f"Could not read the file: {e}")
            continue

        t = df["t"].values
        ch1 = df["ch1"].values
        ch2 = df["ch2"].values
        if channels_swapped:
            ch1, ch2 = ch2, ch1

        try:
            result = analyze_full_trace(
                t, ch1, ch2, delta=delta_rad,
                trigger_threshold=manual_threshold if threshold_mode == "Manual" else None,
            )
        except ValueError as e:
            st.error(str(e))
            continue

        t_triggers = detect_triggers(
            t, ch2, threshold=manual_threshold if threshold_mode == "Manual" else None
        )
        s = result.mean_stokes
        d = result.mean_derived

        col_meta, col_stokes = st.columns([1, 2])
        with col_meta:
            st.markdown("**Capture metadata**")
            st.write(f"Device: {meta.device or '—'}")
            st.write(f"Date: {meta.datetime or '—'}")
            st.write(f"Sampling rate: {meta.fs:,.0f} Hz" if meta.fs else "")
            st.write(f"Measured rotation frequency: **{result.f_rot:.4f} Hz**")
            st.write(f"Periods analyzed: **{result.table.shape[0]}**")

        with col_stokes:
            st.markdown("**Normalized Stokes vector (S0 = 1)**")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("S1", f"{s.S1:+.3f}")
            c2.metric("S2", f"{s.S2:+.3f}")
            c3.metric("S3", f"{s.S3:+.3f}")
            c4.metric("DOP", f"{d['DOP']:.3f}")
            st.write(
                f"Azimuth ψ = **{d['psi_deg']:+.2f}°**  ·  "
                f"Ellipticity χ = **{d['chi_deg']:+.2f}°**  ·  "
                f"Handedness: **{d['handedness']}**"
            )
            if result.table.shape[0] > 1:
                st.caption(
                    f"Dispersion across periods (±1σ): "
                    f"s1={result.stokes_std[1]:.3f}  "
                    f"s2={result.stokes_std[2]:.3f}  "
                    f"s3={result.stokes_std[3]:.3f}"
                )
            if d["non_physical_warning"]:
                r_ref = result.periods[0].fit
                cal = calibrate_optimal_delta(r_ref.A, r_ref.B, r_ref.C, r_ref.D)
                st.warning(
                    f"⚠️ DOP = {d['DOP']:.3f} > 1 — the Stokes vector falls outside the "
                    f"Poincaré sphere.  \n"
                    f"**Most likely cause**: the real retardance of your QWP is not exactly 90°.  \n"
                    f"**Try**: change δ in the sidebar to **{cal['delta_opt_deg']:.1f}°** "
                    f"(estimated from this measurement)."
                )

        st.divider()
        g1, g2, g3 = st.columns(3)
        with g1:
            fig = plot_raw_trace(t, ch1, ch2, t_triggers=t_triggers)
            st.pyplot(fig, clear_figure=True)
        with g2:
            fig = plot_period_fit(t, ch1, result.periods[0], result.omega)
            st.pyplot(fig, clear_figure=True)
        with g3:
            fig = plot_polarization_ellipse(d["psi_deg"], d["chi_deg"], dop=min(d["DOP"], 1.0))
            st.pyplot(fig, clear_figure=True)

        st.markdown("**Per-period detail**")
        st.dataframe(result.table, use_container_width=True, hide_index=True)

        new_rows.append(dict(
            file=file.name, capture_datetime=meta.datetime,
            f_rot_Hz=result.f_rot, n_periods=result.table.shape[0],
            delta_deg=delta_deg,
            S1=s.S1, S2=s.S2, S3=s.S3, DOP=d["DOP"],
            psi_deg=d["psi_deg"], chi_deg=d["chi_deg"], handedness=d["handedness"],
            non_physical_warning=d["non_physical_warning"],
        ))

# ------------------------------------------------------------------ History + combined Poincare sphere
if new_rows:
    new_df = pd.DataFrame(new_rows)
    existing = set(st.session_state.history.get("file", pd.Series(dtype=str)))
    to_add = new_df[~new_df["file"].isin(existing)]
    if not to_add.empty:
        st.session_state.history = pd.concat(
            [st.session_state.history, to_add], ignore_index=True
        )

st.divider()
st.header("📈 Session history")
if st.session_state.history.empty:
    st.write("No measurements accumulated yet.")
else:
    st.dataframe(st.session_state.history, use_container_width=True, hide_index=True)

    points = st.session_state.history[["S1", "S2", "S3"]].values
    fig_poincare = plot_poincare_sphere(points)
    st.pyplot(fig_poincare, clear_figure=True)

    csv_bytes = st.session_state.history.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download history (CSV)", data=csv_bytes,
        file_name="stokes_history.csv", mime="text/csv",
    )
