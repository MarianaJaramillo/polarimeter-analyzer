#!/usr/bin/env python3
"""
analyze_polarimeter.py
========================

Command-line script to analyze CSV captures exported from WaveForms
(Analog Discovery 3) of the rotating-waveplate polarimeter.

BASIC USAGE
-----------
    python analyze_polarimeter.py capture1.csv

    python analyze_polarimeter.py capture1.csv capture2.csv capture3.csv \
        --delta 91.5 --output results/ --history results/history.csv

Each file produces:
  - A PNG figure with the full report (raw trace, fit, ellipse, and its
    point on the Poincare sphere) in the output folder.
  - A new row appended to the history CSV (created if it does not exist),
    so measurements from different sessions can be accumulated.

Calibration parameters
-----------------------
--delta   : real retardance of the quarter-wave plate, IN DEGREES.
            Defaults to 90 (ideal waveplate). If you calibrated your QWP as
            described in the reference paper (Wilkinson et al. 2021, Rev.
            Sci. Instrum. 92, 093101), set this to the measured value
            (e.g., 93.0 if you measured delta=0.5167*pi ~ 93.0 degrees).
--threshold : voltage level (in V) used to detect the rising edge of the
            trigger on channel 2. If not given, it is computed
            automatically as the midpoint between the minimum and maximum
            of that signal.
--photodiode-channel / --trigger-channel : use these if, in your setup,
            the photodiode and the trigger are on channels opposite to the
            expected ones (default: photodiode=ch1, trigger=ch2, matching
            how WaveForms exports Channel 1 / Channel 2).
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

from polarimeter_lib import load_waveforms_csv, analyze_full_trace, detect_triggers
from polarimeter_plots import full_report


def analyze_one_file(path: str, delta_rad: float, threshold, channels_swapped: bool,
                      output_dir: str) -> dict:
    name = os.path.splitext(os.path.basename(path))[0]
    df, meta = load_waveforms_csv(path)
    t = df["t"].values
    ch1 = df["ch1"].values
    ch2 = df["ch2"].values
    if channels_swapped:
        ch1, ch2 = ch2, ch1

    result = analyze_full_trace(t, ch1, ch2, delta=delta_rad, trigger_threshold=threshold)
    t_triggers = detect_triggers(t, ch2, threshold=threshold)

    d = result.mean_derived
    s = result.mean_stokes

    print(f"\n=== {os.path.basename(path)} ===")
    if meta.datetime:
        print(f"  Capture date/time : {meta.datetime}")
    print(f"  Rotation frequency: {result.f_rot:.4f} Hz  "
          f"(period {result.mean_period*1000:.3f} ms, {result.table.shape[0]} period(s) analyzed)")
    print(f"  Normalized Stokes vector (S0=1):")
    print(f"    S1 = {s.S1:+.4f}   S2 = {s.S2:+.4f}   S3 = {s.S3:+.4f}")
    print(f"  DOP = {d['DOP']:.4f}   azimuth psi = {d['psi_deg']:+.2f} deg   "
          f"ellipticity chi = {d['chi_deg']:+.2f} deg   ({d['handedness']})")
    if d["non_physical_warning"]:
        print("  WARNING: DOP > 1.02 -> non-physical Stokes vector.")
        print("     Possible causes: low signal-to-noise ratio, incorrect trigger phase")
        print("     (photo-interrupter delay/offset), or a mis-calibrated QWP delta.")
    if result.table.shape[0] > 1:
        print(f"  Dispersion across periods (std): "
              f"s1={result.stokes_std[1]:.4f}  s2={result.stokes_std[2]:.4f}  "
              f"s3={result.stokes_std[3]:.4f}")

    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, f"{name}_report.png")
    fig = full_report(t, ch1, ch2, t_triggers, result)
    fig.savefig(png_path, dpi=130)
    print(f"  Report saved to: {png_path}")

    table_path = os.path.join(output_dir, f"{name}_periods.csv")
    result.table.to_csv(table_path, index=False)

    return dict(
        file=os.path.basename(path),
        analysis_datetime=datetime.now().isoformat(timespec="seconds"),
        capture_datetime=meta.datetime,
        f_rot_Hz=result.f_rot,
        n_periods=result.table.shape[0],
        delta_deg=np.degrees(delta_rad),
        S1=s.S1, S2=s.S2, S3=s.S3,
        DOP=d["DOP"], psi_deg=d["psi_deg"], chi_deg=d["chi_deg"],
        handedness=d["handedness"], non_physical_warning=d["non_physical_warning"],
    )


def main():
    ap = argparse.ArgumentParser(description="Analyze rotating-waveplate polarimeter captures.")
    ap.add_argument("files", nargs="+", help="One or more CSV files exported by WaveForms")
    ap.add_argument("--delta", type=float, default=90.0,
                    help="Real retardance of the QWP in DEGREES (default 90 = ideal)")
    ap.add_argument("--threshold", type=float, default=None,
                    help="Voltage threshold for the trigger (default: automatic)")
    ap.add_argument("--channels-swapped", action="store_true",
                    help="Use if the photodiode is on channel 2 and the trigger on channel 1")
    ap.add_argument("--output", default="polarimeter_results",
                    help="Folder where PNG reports and per-period tables are saved")
    ap.add_argument("--history", default=None,
                    help="CSV file where a summary row per analyzed file is accumulated "
                         "(default: <output>/stokes_history.csv)")
    args = ap.parse_args()

    delta_rad = np.radians(args.delta)
    history = args.history or os.path.join(args.output, "stokes_history.csv")

    rows = []
    for path in args.files:
        try:
            row = analyze_one_file(path, delta_rad, args.threshold, args.channels_swapped, args.output)
            rows.append(row)
        except Exception as e:
            print(f"\n!! Error analyzing {path}: {e}", file=sys.stderr)

    if rows:
        new_rows = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(history) or ".", exist_ok=True)
        if os.path.exists(history):
            previous = pd.read_csv(history)
            combined = pd.concat([previous, new_rows], ignore_index=True)
        else:
            combined = new_rows
        combined.to_csv(history, index=False)
        print(f"\nHistory updated ({len(combined)} total measurements): {history}")


if __name__ == "__main__":
    main()
