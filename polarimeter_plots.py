"""
polarimeter_plots.py
=====================

Funciones de visualizacion para el analisis del polarimetro. Se basan
unicamente en matplotlib, para que funcionen igual desde un script de
consola, un notebook o una app Streamlit (con st.pyplot).
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (necesario para proj='3d')

from polarimeter_lib import ResultadoAnalisis, ajustar_coeficientes


# ---------------------------------------------------------------------------
def graficar_traza_cruda(t, ch1, ch2, t_disparos=None, ax=None):
    """Grafica las dos senales crudas (fotodiodo + disparo) contra el tiempo."""
    if ax is None:
        fig, ax = plt.subplots(2, 1, figsize=(9, 4.5), sharex=True)
    else:
        fig = ax[0].figure

    ax[0].plot(t, ch1, lw=0.6, color="#1f77b4")
    ax[0].set_ylabel("Fotodiodo\n(Canal 1) [V]")
    ax[0].grid(alpha=0.3)

    ax[1].plot(t, ch2, lw=0.6, color="#d62728")
    ax[1].set_ylabel("Disparo\n(Canal 2) [V]")
    ax[1].set_xlabel("Tiempo [s]")
    ax[1].grid(alpha=0.3)

    if t_disparos is not None:
        for td in t_disparos:
            ax[0].axvline(td, color="k", ls="--", lw=0.8, alpha=0.6)
            ax[1].axvline(td, color="k", ls="--", lw=0.8, alpha=0.6)

    fig.suptitle("Senales crudas capturadas")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
def graficar_ajuste_periodo(t, ch1, resultado_periodo, omega, ax=None, n_periodo=200):
    """
    Grafica los datos de un periodo (en funcion de theta, en grados) junto
    con el modelo ajustado I(theta), y muestra el residual.
    """
    r = resultado_periodo
    mascara = (t >= r.t_inicio) & (t < r.t_fin)
    tp = t[mascara]
    Ip = ch1[mascara]
    theta = np.degrees(omega * (tp - r.t_inicio)) % 360

    A, B, C, D = r.ajuste.A, r.ajuste.B, r.ajuste.C, r.ajuste.D
    theta_modelo = np.linspace(0, 360, 400)
    th_rad = np.radians(theta_modelo)
    modelo = 0.5 * (A + B * np.sin(2 * th_rad) + C * np.cos(4 * th_rad) + D * np.sin(4 * th_rad))

    if ax is None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True,
                                        gridspec_kw={"height_ratios": [3, 1]})
    else:
        ax1, ax2 = ax
        fig = ax1.figure

    orden = np.argsort(theta)
    ax1.scatter(theta[orden], Ip[orden], s=4, alpha=0.35, label="Datos", color="#1f77b4")
    ax1.plot(theta_modelo, modelo, color="#d62728", lw=2,
             label=f"Ajuste (R²={r.ajuste.r2:.3f})")
    ax1.set_ylabel("Intensidad [V]")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(alpha=0.3)
    ax1.set_title(f"Periodo {r.idx}: ajuste del modelo de Fourier")

    modelo_en_datos = 0.5 * (A + B * np.sin(2 * np.radians(theta)) +
                              C * np.cos(4 * np.radians(theta)) + D * np.sin(4 * np.radians(theta)))
    residual = Ip - modelo_en_datos
    ax2.scatter(theta[orden], residual[orden], s=4, alpha=0.4, color="gray")
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_ylabel("Residual [V]")
    ax2.set_xlabel("θ = ωt (grados desde el disparo)")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
def graficar_elipse_polarizacion(psi_deg, chi_deg, dop=1.0, ax=None):
    """
    Dibuja la elipse de polarizacion (vista de frente, en el plano
    transversal al haz) dados el azimut psi y la elipticidad chi (grados).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
    else:
        fig = ax.figure

    psi = np.radians(psi_deg)
    chi = np.radians(chi_deg)

    tau = np.linspace(0, 2 * np.pi, 300)
    # Elipse canonica con semiejes a=cos(chi), b=sin(chi), luego rotada por psi
    a, b = np.cos(chi), np.sin(chi)
    x0 = a * np.cos(tau)
    y0 = b * np.sin(tau)
    x = x0 * np.cos(psi) - y0 * np.sin(psi)
    y = x0 * np.sin(psi) + y0 * np.cos(psi)

    escala = max(dop, 0.05)
    ax.plot(escala * x, escala * y, color="#1f77b4", lw=2)
    ax.axhline(0, color="gray", lw=0.6)
    ax.axvline(0, color="gray", lw=0.6)
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect("equal")
    ax.set_title(f"Elipse de polarización\nψ={psi_deg:.1f}°  χ={chi_deg:.1f}°")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
def graficar_poincare(puntos, ax=None):
    """
    Dibuja la esfera de Poincare (unitaria) y uno o varios puntos
    normalizados (s1, s2, s3).

    puntos: lista de tuplas (s1, s2, s3) o arreglo Nx3.
    """
    puntos = np.atleast_2d(np.asarray(puntos, dtype=float))

    if ax is None:
        fig = plt.figure(figsize=(5.5, 5))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(xs, ys, zs, color="lightgray", linewidth=0.4, alpha=0.6)

    # Ejes principales
    ax.plot([-1, 1], [0, 0], [0, 0], color="gray", lw=0.8)
    ax.plot([0, 0], [-1, 1], [0, 0], color="gray", lw=0.8)
    ax.plot([0, 0], [0, 0], [-1, 1], color="gray", lw=0.8)
    ax.text(1.15, 0, 0, "S1", fontsize=8)
    ax.text(0, 1.15, 0, "S2", fontsize=8)
    ax.text(0, 0, 1.15, "S3", fontsize=8)

    colores = plt.cm.viridis(np.linspace(0, 1, len(puntos)))
    ax.scatter(puntos[:, 0], puntos[:, 1], puntos[:, 2], color=colores, s=45,
               edgecolor="k", depthshade=True)

    normas = np.linalg.norm(puntos, axis=1)
    lim = max(1.1, float(np.max(normas)) * 1.15) if len(normas) else 1.1
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    titulo = "Esfera de Poincaré"
    if np.any(normas > 1.02):
        titulo += "\n(⚠ punto(s) fuera de la esfera: DOP>1, vector no físico)"
    ax.set_title(titulo, fontsize=10)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
def reporte_completo(t, ch1, ch2, t_disparos, resultado: ResultadoAnalisis):
    """
    Arma una figura tipo "reporte" de una sola pagina con: traza cruda,
    ajuste del (primer) periodo, elipse de polarizacion y esfera de
    Poincare del promedio de todos los periodos analizados.
    """
    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.3])

    ax_ch1 = fig.add_subplot(gs[0, :])
    ax_ch2 = fig.add_subplot(gs[1, :], sharex=ax_ch1)
    graficar_traza_cruda(t, ch1, ch2, t_disparos=t_disparos, ax=[ax_ch1, ax_ch2])
    ax_ch1.set_title("")
    fig.suptitle("Reporte de análisis del polarímetro", fontsize=13, fontweight="bold")

    r0 = resultado.periodos[0]
    ax_ajuste = fig.add_subplot(gs[2, 0])
    ax_res = ax_ajuste.inset_axes([0, -0.4, 1, 0.3])
    graficar_ajuste_periodo(t, ch1, r0, resultado.omega, ax=[ax_ajuste, ax_res])

    ax_elipse = fig.add_subplot(gs[2, 1])
    d = resultado.derivados_medio
    graficar_elipse_polarizacion(d["psi_deg"], d["chi_deg"], dop=min(d["DOP"], 1.0), ax=ax_elipse)

    fig.tight_layout()
    return fig
