"""
polarimeter_lib.py
===================

Analysis library for a rotating-waveplate polarimeter, based on:

    Wilkinson, Maurer, Flood, Lander, Chafin & Flagg,
    "Complete Stokes vector analysis with a compact, portable rotating
    waveplate polarimeter", Rev. Sci. Instrum. 92, 093101 (2021).
    https://doi.org/10.1063/5.0052835

Physical model (Eq. 1 in the reference paper):

    I(theta) = 1/2 * [ A + B*sin(2*theta) + C*cos(4*theta) + D*sin(4*theta) ]
    theta    = omega*t + phi

where omega is the angular rotation frequency of the quarter-wave plate
(QWP) and phi is the phase offset (set to 0 at the trigger pulse emitted
by the photo-interrupter at the start of each revolution).

Relation to the Stokes vector (Eq. 2), allowing for a real retardance
delta != pi/2:

    S0 = A - C/tan^2(delta/2)
    S1 = C/sin^2(delta/2)
    S2 = D/sin^2(delta/2)
    S3 = B/sin(delta)

This module has no graphical dependencies (only numpy/pandas). It is used
both by the command-line script (analyze_polarimeter.py) and by the
interactive app (app_polarimeter.py).
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Loading CSV files exported by Digilent WaveForms
# ---------------------------------------------------------------------------

@dataclass
class CaptureMetadata:
    device: Optional[str] = None
    serial_number: Optional[str] = None
    datetime: Optional[str] = None
    fs: Optional[float] = None          # sampling rate (Hz)
    n_samples: Optional[int] = None
    channel1_info: Optional[str] = None
    channel2_info: Optional[str] = None
    trigger_info: Optional[str] = None
    columns: Optional[list] = None
    raw_header: str = ""                # all comment lines, kept for reference


def load_waveforms_csv(path_or_buffer) -> tuple[pd.DataFrame, CaptureMetadata]:
    """
    Load a CSV file exported by WaveForms (Analog Discovery 3 oscilloscope).

    Accepts a file path (str/Path) or a file-like/bytes object (useful for
    Streamlit's `st.file_uploader`).

    Returns
    -------
    df : DataFrame with columns ['t', 'ch1', 'ch2'] (only the first two
         channel columns, regardless of their original names).
    meta : CaptureMetadata with the information parsed from the '#...'
           header lines.
    """
    if hasattr(path_or_buffer, "read"):
        content = path_or_buffer.read()
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="replace")
        text = content
    else:
        with open(path_or_buffer, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()

    lines = text.splitlines()
    comment_lines = [l for l in lines if l.strip().startswith("#")]
    meta = CaptureMetadata(raw_header="\n".join(comment_lines))

    for l in comment_lines:
        l = l.lstrip("#").strip()
        if l.lower().startswith("device name"):
            meta.device = l.split(":", 1)[1].strip()
        elif l.lower().startswith("serial number"):
            meta.serial_number = l.split(":", 1)[1].strip()
        elif l.lower().startswith("date time"):
            meta.datetime = l.split(":", 1)[1].strip()
        elif l.lower().startswith("sample rate"):
            m = re.search(r"([\d.]+)\s*Hz", l, re.IGNORECASE)
            if m:
                meta.fs = float(m.group(1))
        elif l.lower().startswith("samples"):
            m = re.search(r"(\d+)", l)
            if m:
                meta.n_samples = int(m.group(1))
        elif l.lower().startswith("channel 1"):
            meta.channel1_info = l
        elif l.lower().startswith("channel 2"):
            meta.channel2_info = l
        elif l.lower().startswith("trigger"):
            meta.trigger_info = l

    # Locate the real header row (the one that is not a '#' comment and has commas)
    df = pd.read_csv(io.StringIO(text), comment="#", skip_blank_lines=True)
    df = df.dropna(how="all")
    meta.columns = list(df.columns)

    # Rename the first 3 columns to the standard names: t, ch1, ch2
    cols = list(df.columns)
    rename_map = {}
    if len(cols) >= 1:
        rename_map[cols[0]] = "t"
    if len(cols) >= 2:
        rename_map[cols[1]] = "ch1"
    if len(cols) >= 3:
        rename_map[cols[2]] = "ch2"
    df = df.rename(columns=rename_map)[["t", "ch1", "ch2"]].reset_index(drop=True)
    df = df.astype(float)

    if meta.fs is None and len(df) > 1:
        meta.fs = 1.0 / np.median(np.diff(df["t"].values))
    if meta.n_samples is None:
        meta.n_samples = len(df)

    return df, meta


# ---------------------------------------------------------------------------
# 2. Trigger detection and rotation frequency
# ---------------------------------------------------------------------------

def detect_triggers(t: np.ndarray, trigger: np.ndarray,
                     threshold: Optional[float] = None,
                     threshold_fraction: float = 0.5) -> np.ndarray:
    """
    Detect the rising edges of the trigger signal (photo-interrupter),
    which mark the start of each waveplate rotation period (theta=0).

    Parameters
    ----------
    t, trigger : time and voltage arrays of the trigger channel.
    threshold : voltage level considered "high". If None, it is computed
        automatically as min + threshold_fraction*(max-min).
    threshold_fraction : fraction (0-1) used for the automatic threshold.

    Returns
    -------
    t_triggers : array with the time of each rising edge.
    """
    trigger = np.asarray(trigger, dtype=float)
    t = np.asarray(t, dtype=float)

    if threshold is None:
        lo, hi = np.min(trigger), np.max(trigger)
        threshold = lo + threshold_fraction * (hi - lo)

    above = trigger > threshold
    rising_idx = np.where((~above[:-1]) & (above[1:]))[0] + 1

    if len(rising_idx) == 0:
        return np.array([])

    # Linear interpolation to refine the exact crossing instant
    t_triggers = []
    for i in rising_idx:
        t0, t1 = t[i - 1], t[i]
        v0, v1 = trigger[i - 1], trigger[i]
        if v1 == v0:
            t_cross = t1
        else:
            t_cross = t0 + (threshold - v0) * (t1 - t0) / (v1 - v0)
        t_triggers.append(t_cross)

    return np.array(t_triggers)


def rotation_frequency(t_triggers: np.ndarray) -> dict:
    """
    Compute the angular rotation frequency from consecutive trigger times.

    Returns a dict with: n_periods, mean_period, period_std, f_rot, omega,
    periods (array of individual period durations).
    """
    if len(t_triggers) < 2:
        return dict(n_periods=0, mean_period=np.nan, period_std=np.nan,
                    f_rot=np.nan, omega=np.nan, periods=np.array([]))

    periods = np.diff(t_triggers)
    mean_period = float(np.mean(periods))
    period_std = float(np.std(periods))
    f_rot = 1.0 / mean_period
    omega = 2 * np.pi * f_rot

    return dict(n_periods=len(periods), mean_period=mean_period,
                period_std=period_std, f_rot=f_rot, omega=omega,
                periods=periods)


# ---------------------------------------------------------------------------
# 3. Least-squares fit of the Fourier coefficients (A, B, C, D)
# ---------------------------------------------------------------------------

@dataclass
class FitResult:
    A: float
    B: float
    C: float
    D: float
    r2: float
    n_samples: int


def fit_coefficients(t: np.ndarray, signal: np.ndarray, t0: float,
                      omega: float) -> FitResult:
    """
    Fit the photodiode signal to the model (Eq. 1 in the reference paper):

        I(theta) = 1/2*[A + B*sin(2theta) + C*cos(4theta) + D*sin(4theta)]
        theta = omega*(t - t0)

    using linear least squares.

    t0 is the trigger time that defines theta=0 (phi=0).
    """
    t = np.asarray(t, dtype=float)
    signal = np.asarray(signal, dtype=float)
    theta = omega * (t - t0)

    M = np.column_stack([
        np.ones_like(theta),
        np.sin(2 * theta),
        np.cos(4 * theta),
        np.sin(4 * theta),
    ])

    coef, *_ = np.linalg.lstsq(M, signal, rcond=None)
    a, b, c, d = coef
    A, B, C, D = 2 * a, 2 * b, 2 * c, 2 * d

    fitted = M @ coef
    ss_res = np.sum((signal - fitted) ** 2)
    ss_tot = np.sum((signal - np.mean(signal)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return FitResult(A=A, B=B, C=C, D=D, r2=float(r2), n_samples=len(t))


# ---------------------------------------------------------------------------
# 4. Fourier coefficients -> Stokes vector
# ---------------------------------------------------------------------------

@dataclass
class StokesVector:
    S0: float
    S1: float
    S2: float
    S3: float

    @property
    def normalized(self):
        s0 = self.S0 if self.S0 != 0 else np.nan
        return (self.S1 / s0, self.S2 / s0, self.S3 / s0)


def stokes_from_coefficients(A: float, B: float, C: float, D: float,
                              delta: float = np.pi / 2) -> StokesVector:
    """
    Compute the Stokes vector (Eq. 2 in the reference paper) from the
    Fourier coefficients A, B, C, D and the real retardance `delta` (in
    radians) of the quarter-wave plate. Defaults to delta=pi/2 (ideal
    waveplate).
    """
    S0 = A - C / np.tan(delta / 2) ** 2
    S1 = C / np.sin(delta / 2) ** 2
    S2 = D / np.sin(delta / 2) ** 2
    S3 = B / np.sin(delta)
    return StokesVector(S0=S0, S1=S1, S2=S2, S3=S3)


def calibrate_optimal_delta(A: float, B: float, C: float, D: float,
                             delta_min_deg: float = 60.0,
                             delta_max_deg: float = 120.0) -> dict:
    """
    Find the retardance delta (in degrees) that yields DOP = 1 for the
    given Fourier coefficients, assuming the source is fully polarized
    (true DOP = 1).

    Useful for calibrating the QWP: measure with a source of known
    polarization (e.g., laser + linear polarizer), then call this function
    with the average coefficients and use the resulting delta_opt_deg in
    the app.

    Returns
    -------
    delta_opt_deg : optimal retardance, in degrees.
    delta_opt_rad : same, in radians.
    dop_at_opt    : DOP computed with that delta (should be ~1.0).
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
        dop_at_opt=float(dop[idx]),
    )


def derived_parameters(s: StokesVector) -> dict:
    """
    Quantities derived from the normalized Stokes vector:
      - degree of polarization (DOP)
      - azimuth of the polarization ellipse, psi (degrees, in [-90,90))
      - ellipticity angle, chi (degrees, in [-45,45])
      - handedness (R = right-handed / L = left-handed / linear)
      - non_physical_warning : True if DOP > 1.02
    """
    S0, S1, S2, S3 = s.S0, s.S1, s.S2, s.S3
    if S0 == 0 or np.isnan(S0):
        return dict(DOP=np.nan, psi_deg=np.nan, chi_deg=np.nan, handedness="?",
                    non_physical_warning=True)

    s1, s2, s3 = S1 / S0, S2 / S0, S3 / S0
    dop = np.sqrt(s1 ** 2 + s2 ** 2 + s3 ** 2)

    psi = 0.5 * np.degrees(np.arctan2(s2, s1))          # azimuth
    denom = np.sqrt(s1 ** 2 + s2 ** 2)
    chi = 0.5 * np.degrees(np.arctan2(s3, denom)) if denom > 0 else np.sign(s3) * 45.0

    if s3 > 1e-3:
        handedness = "right-handed (R)"
    elif s3 < -1e-3:
        handedness = "left-handed (L)"
    else:
        handedness = "linear"

    return dict(DOP=dop, psi_deg=psi, chi_deg=chi, handedness=handedness,
                non_physical_warning=bool(dop > 1.02))


# ---------------------------------------------------------------------------
# 5. Analysis of a single period and of the full trace (multiple periods)
# ---------------------------------------------------------------------------

@dataclass
class PeriodResult:
    idx: int
    t_start: float
    t_end: float
    fit: FitResult
    stokes: StokesVector
    derived: dict


@dataclass
class AnalysisResult:
    periods: list                  # list of PeriodResult
    f_rot: float
    omega: float
    mean_period: float
    period_std: float
    mean_stokes: StokesVector
    stokes_std: tuple              # (S0,S1,S2,S3) normalized standard deviations
    mean_derived: dict
    table: pd.DataFrame            # one row per period, for export/plotting


def analyze_period(t: np.ndarray, signal: np.ndarray, t_start: float, t_end: float,
                    omega: float, delta: float = np.pi / 2, idx: int = 0) -> PeriodResult:
    mask = (t >= t_start) & (t < t_end)
    fit = fit_coefficients(t[mask], signal[mask], t0=t_start, omega=omega)
    stokes = stokes_from_coefficients(fit.A, fit.B, fit.C, fit.D, delta=delta)
    derived = derived_parameters(stokes)
    return PeriodResult(idx=idx, t_start=t_start, t_end=t_end, fit=fit,
                         stokes=stokes, derived=derived)


def analyze_full_trace(t: np.ndarray, ch1: np.ndarray, ch2: np.ndarray,
                        delta: float = np.pi / 2,
                        trigger_threshold: Optional[float] = None,
                        threshold_fraction: float = 0.5) -> AnalysisResult:
    """
    Full pipeline: detect triggers on ch2, compute the rotation frequency,
    and fit the Stokes vector in each period delimited by two consecutive
    triggers (theta=0 at the start of each one). If only one full period is
    available, that single result is reported without dispersion
    statistics (same structure as with several periods, but with std =
    NaN).
    """
    t = np.asarray(t, dtype=float)
    ch1 = np.asarray(ch1, dtype=float)
    ch2 = np.asarray(ch2, dtype=float)

    t_triggers = detect_triggers(t, ch2, threshold=trigger_threshold,
                                  threshold_fraction=threshold_fraction)
    if len(t_triggers) < 2:
        raise ValueError(
            "Fewer than 2 triggers were detected on the trigger channel; "
            "a full period cannot be delimited. Check the trigger "
            "threshold or the selected channel."
        )

    f_info = rotation_frequency(t_triggers)
    omega = f_info["omega"]

    periods = []
    for i in range(len(t_triggers) - 1):
        r = analyze_period(t, ch1, t_triggers[i], t_triggers[i + 1], omega=omega,
                            delta=delta, idx=i)
        periods.append(r)

    rows = []
    for r in periods:
        rows.append(dict(
            period=r.idx, t_start=r.t_start, t_end=r.t_end,
            A=r.fit.A, B=r.fit.B, C=r.fit.C, D=r.fit.D, R2=r.fit.r2,
            S0=r.stokes.S0, S1=r.stokes.S1, S2=r.stokes.S2, S3=r.stokes.S3,
            s1=r.stokes.S1 / r.stokes.S0 if r.stokes.S0 else np.nan,
            s2=r.stokes.S2 / r.stokes.S0 if r.stokes.S0 else np.nan,
            s3=r.stokes.S3 / r.stokes.S0 if r.stokes.S0 else np.nan,
            DOP=r.derived["DOP"], psi_deg=r.derived["psi_deg"],
            chi_deg=r.derived["chi_deg"], handedness=r.derived["handedness"],
        ))
    table = pd.DataFrame(rows)

    s1_m, s2_m, s3_m = table["s1"].mean(), table["s2"].mean(), table["s3"].mean()
    s1_s, s2_s, s3_s = table["s1"].std(), table["s2"].std(), table["s3"].std()
    mean_stokes = StokesVector(S0=1.0, S1=s1_m, S2=s2_m, S3=s3_m)
    mean_derived = derived_parameters(mean_stokes)

    return AnalysisResult(
        periods=periods, f_rot=f_info["f_rot"], omega=f_info["omega"],
        mean_period=f_info["mean_period"], period_std=f_info["period_std"],
        mean_stokes=mean_stokes, stokes_std=(np.nan, s1_s, s2_s, s3_s),
        mean_derived=mean_derived, table=table,
    )
