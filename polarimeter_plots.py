"""
polarimeter_plots.py
=====================

Visualization functions for the polarimeter analysis. They rely only on
matplotlib, so they work the same way from a command-line script, a
notebook, or a Streamlit app (with st.pyplot).
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (required for proj='3d')

from polarimeter_lib import AnalysisResult


# ---------------------------------------------------------------------------
def plot_raw_trace(t, ch1, ch2, t_triggers=None, ax=None):
    """Plot the two raw signals (photodiode + trigger) against time."""
    if ax is None:
        fig, ax = plt.subplots(2, 1, figsize=(9, 4.5), sharex=True)
    else:
        fig = ax[0].figure

    ax[0].plot(t, ch1, lw=0.6, color="#1f77b4")
    ax[0].set_ylabel("Photodiode\n(Channel 1) [V]")
    ax[0].grid(alpha=0.3)

    ax[1].plot(t, ch2, lw=0.6, color="#d62728")
    ax[1].set_ylabel("Trigger\n(Channel 2) [V]")
    ax[1].set_xlabel("Time [s]")
    ax[1].grid(alpha=0.3)

    if t_triggers is not None:
        for tt in t_triggers:
            ax[0].axvline(tt, color="k", ls="--", lw=0.8, alpha=0.6)
            ax[1].axvline(tt, color="k", ls="--", lw=0.8, alpha=0.6)

    fig.suptitle("Raw captured signals")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
def plot_period_fit(t, ch1, period_result, omega, ax=None, n_period=200):
    """
    Plot the data of a single period (as a function of theta, in degrees)
    together with the fitted model I(theta), and show the residual.
    """
    r = period_result
    mask = (t >= r.t_start) & (t < r.t_end)
    tp = t[mask]
    Ip = ch1[mask]
    theta = np.degrees(omega * (tp - r.t_start)) % 360

    A, B, C, D = r.fit.A, r.fit.B, r.fit.C, r.fit.D
    theta_model = np.linspace(0, 360, 400)
    th_rad = np.radians(theta_model)
    model = 0.5 * (A + B * np.sin(2 * th_rad) + C * np.cos(4 * th_rad) + D * np.sin(4 * th_rad))

    if ax is None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True,
                                        gridspec_kw={"height_ratios": [3, 1]})
    else:
        ax1, ax2 = ax
        fig = ax1.figure

    order = np.argsort(theta)
    ax1.scatter(theta[order], Ip[order], s=4, alpha=0.35, label="Data", color="#1f77b4")
    ax1.plot(theta_model, model, color="#d62728", lw=2,
             label=f"Fit (R²={r.fit.r2:.3f})")
    ax1.set_ylabel("Intensity [V]")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(alpha=0.3)
    ax1.set_title(f"Period {r.idx}: Fourier model fit")

    model_at_data = 0.5 * (A + B * np.sin(2 * np.radians(theta)) +
                           C * np.cos(4 * np.radians(theta)) + D * np.sin(4 * np.radians(theta)))
    residual = Ip - model_at_data
    ax2.scatter(theta[order], residual[order], s=4, alpha=0.4, color="gray")
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_ylabel("Residual [V]")
    ax2.set_xlabel("θ = ωt (degrees from trigger)")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
def plot_polarization_ellipse(psi_deg, chi_deg, dop=1.0, ax=None):
    """
    Draw the polarization ellipse (front view, in the plane transverse to
    the beam) given the azimuth psi and the ellipticity chi (degrees).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
    else:
        fig = ax.figure

    psi = np.radians(psi_deg)
    chi = np.radians(chi_deg)

    tau = np.linspace(0, 2 * np.pi, 300)
    # Canonical ellipse with semi-axes a=cos(chi), b=sin(chi), then rotated by psi
    a, b = np.cos(chi), np.sin(chi)
    x0 = a * np.cos(tau)
    y0 = b * np.sin(tau)
    x = x0 * np.cos(psi) - y0 * np.sin(psi)
    y = x0 * np.sin(psi) + y0 * np.cos(psi)

    scale = max(dop, 0.05)
    ax.plot(scale * x, scale * y, color="#1f77b4", lw=2)
    ax.axhline(0, color="gray", lw=0.6)
    ax.axvline(0, color="gray", lw=0.6)
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect("equal")
    ax.set_title(f"Polarization ellipse\nψ={psi_deg:.1f}°  χ={chi_deg:.1f}°")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
def plot_poincare_sphere(points, ax=None):
    """
    Draw the (unit) Poincare sphere and one or more normalized points
    (s1, s2, s3).

    points: list of (s1, s2, s3) tuples or an Nx3 array.
    """
    points = np.atleast_2d(np.asarray(points, dtype=float))

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

    # Principal axes
    ax.plot([-1, 1], [0, 0], [0, 0], color="gray", lw=0.8)
    ax.plot([0, 0], [-1, 1], [0, 0], color="gray", lw=0.8)
    ax.plot([0, 0], [0, 0], [-1, 1], color="gray", lw=0.8)
    ax.text(1.15, 0, 0, "S1", fontsize=8)
    ax.text(0, 1.15, 0, "S2", fontsize=8)
    ax.text(0, 0, 1.15, "S3", fontsize=8)

    colors = plt.cm.viridis(np.linspace(0, 1, len(points)))
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color=colors, s=45,
               edgecolor="k", depthshade=True)

    norms = np.linalg.norm(points, axis=1)
    lim = max(1.1, float(np.max(norms)) * 1.15) if len(norms) else 1.1
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    title = "Poincaré sphere"
    if np.any(norms > 1.02):
        title += "\n(⚠ point(s) outside the sphere: DOP>1, non-physical vector)"
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
def full_report(t, ch1, ch2, t_triggers, result: AnalysisResult):
    """
    Build a single-page "report" figure with: raw trace, fit of the
    (first) period, polarization ellipse, and Poincare-sphere position of
    the average of all analyzed periods.
    """
    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.3])

    ax_ch1 = fig.add_subplot(gs[0, :])
    ax_ch2 = fig.add_subplot(gs[1, :], sharex=ax_ch1)
    plot_raw_trace(t, ch1, ch2, t_triggers=t_triggers, ax=[ax_ch1, ax_ch2])
    ax_ch1.set_title("")
    fig.suptitle("Polarimeter analysis report", fontsize=13, fontweight="bold")

    r0 = result.periods[0]
    ax_fit = fig.add_subplot(gs[2, 0])
    ax_res = ax_fit.inset_axes([0, -0.4, 1, 0.3])
    plot_period_fit(t, ch1, r0, result.omega, ax=[ax_fit, ax_res])

    ax_ellipse = fig.add_subplot(gs[2, 1])
    d = result.mean_derived
    plot_polarization_ellipse(d["psi_deg"], d["chi_deg"], dop=min(d["DOP"], 1.0), ax=ax_ellipse)

    fig.tight_layout()
    return fig
