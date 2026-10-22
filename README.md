# Rotating-Waveplate Polarimeter Analyzer

Python toolkit to analyze CSV captures exported from **WaveForms** (Digilent Analog Discovery 3) of a home-built rotating-waveplate polarimeter.

It computes the **full Stokes vector** (S0, S1, S2, S3), the **degree of polarization (DOP)**, the **azimuth ψ**, and the **ellipticity χ** from a Fourier fit of the photodiode signal.

This repository is provided as **supplementary software** for the associated manuscript. It reproduces the data-reduction pipeline used to obtain the polarization measurements reported in the paper, so that results can be independently verified and reused.

Method based on:
> Wilkinson, Maurer, Flood, Lander, Chafin & Flagg, *"Complete Stokes vector
> analysis with a compact, portable rotating waveplate polarimeter"*,
> Rev. Sci. Instrum. 92, 093101 (2021). https://doi.org/10.1063/5.0052835

---

## Table of contents

1. [Requirements](#requirements)
2. [Installation](#installation)
3. [How to run](#how-to-run)
   - [Option A — Interactive app (recommended)](#option-a--interactive-app-recommended)
   - [Option B — Command-line script](#option-b--command-line-script)
4. [Input CSV file format](#input-csv-file-format)
5. [Calibration parameters](#calibration-parameters)
6. [Repository structure](#repository-structure)
7. [Stokes vector computation](#stokes-vector-computation)
8. [Troubleshooting](#troubleshooting)
9. [Citation](#citation)

---

## Requirements

- Python 3.9 or newer
- The packages listed in [requirements.txt](requirements.txt):
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `streamlit`

---

## Installation

**1. Clone or download this repository.**

**2. (Recommended) Create a virtual environment:**

```bash
python -m venv venv
```

Activate it:

- Windows:
  ```bash
  venv\Scripts\activate
  ```
- macOS / Linux:
  ```bash
  source venv/bin/activate
  ```

**3. Install the dependencies:**

```bash
pip install -r requirements.txt
```

---

## How to run

### Option A — Interactive app (recommended)

Ideal for everyday lab use. Opens a web interface where you can upload files, tune the calibration with sliders, and see the results instantly.

```bash
streamlit run app_polarimeter.py
```

It opens automatically in your browser at `http://localhost:8501`.

**Workflow in the app:**

1. **Upload** one or more CSV files with the file uploader (or drag and drop).
2. **Adjust the calibration parameters** in the left sidebar:
   - **δ (delta)** — real retardance of your QWP, in degrees. Use `90` for an ideal waveplate; if you already calibrated your QWP as described in the reference paper, enter the measured value (e.g., `93.0`).
   - **Trigger threshold** — computed automatically by default; switch to manual mode only if the trigger signal is very noisy.
   - **Swap channels** — enable if the photodiode is wired to channel 2 and the trigger to channel 1 (opposite of the standard wiring).
3. **Review the results** for each file:
   - Normalized Stokes vector (S1, S2, S3)
   - Degree of polarization (DOP)
   - Azimuth ψ and ellipticity χ
   - Polarization ellipse and Poincaré sphere
   - Raw trace with the overlaid Fourier fit
4. **Download the session history** (summary table + accumulated Poincaré sphere) as a CSV with the download button at the bottom of the page.

---

### Option B — Command-line script

Ideal for batch-processing multiple files or automating the analysis.

**Basic usage (single file):**

```bash
python analyze_polarimeter.py capture1.csv
```

**Multiple files at once:**

```bash
python analyze_polarimeter.py capture1.csv capture2.csv capture3.csv
```

**With calibration parameters:**

```bash
python analyze_polarimeter.py *.csv --delta 93.0 --threshold 1.0 --output results/
```

**All available arguments:**

| Argument | Description | Default |
|---|---|---|
| `--delta` | QWP retardance, in degrees | `90.0` |
| `--threshold` | Voltage threshold (V) for trigger detection | automatic |
| `--output` | Folder where PNG/CSV reports are saved | current directory |
| `--history` | Path to the cumulative history CSV | `stokes_history.csv` |
| `--photodiode-channel` | Photodiode channel number (`1` or `2`) | `1` |
| `--trigger-channel` | Trigger channel number (`1` or `2`) | `2` |

**Outputs generated per file:**

- `results/<name>_report.png` — a full figure with the raw trace, fit, polarization ellipse, and the point's position on the Poincaré sphere.
- `results/<name>_periods.csv` — table with the Fourier coefficients and Stokes vector for each analyzed period.
- `results/stokes_history.csv` — an accumulated row added to the session history.

---

## Input CSV file format

The CSV must be exported from **WaveForms** with two channels:

| Channel | Signal |
|---|---|
| Channel 1 (`ch1`) | Photodiode (optical intensity) |
| Channel 2 (`ch2`) | Trigger pulse (one edge per waveplate revolution) |

The first row may be a header or raw numeric data; the script automatically detects both formats.

---

## Calibration parameters

| Parameter | Description |
|---|---|
| **δ (delta)** | Real retardance of the quarter-wave plate (QWP), in degrees. An ideal waveplate has δ = 90°; yours may differ. Measure it with light of known linear polarization, as described in Section III of the reference paper. |
| **Trigger threshold** | Voltage level (V) used to detect the rising edge of the trigger channel. Automatic mode uses the midpoint between the minimum and maximum of the signal, which works well in most cases. |
| **Channels** | Default: photodiode on CH1, trigger on CH2. Swap them if your setup is wired the other way around. |

---

## Repository structure

| File | Purpose |
|---|---|
| [polarimeter_lib.py](polarimeter_lib.py) | Core analysis: loads the CSV, detects the trigger, fits the Fourier model, and computes the Stokes vector. No graphical dependencies (importable as a library). |
| [polarimeter_plots.py](polarimeter_plots.py) | Visualization module: raw signals, per-period fit, polarization ellipse, and Poincaré sphere. |
| [analyze_polarimeter.py](analyze_polarimeter.py) | Command-line script to batch-process one or more CSV files and save reports. |
| [app_polarimeter.py](app_polarimeter.py) | Interactive web application built with Streamlit. |
| [requirements.txt](requirements.txt) | Python dependency list. |

---

## Stokes vector computation

1. **Trigger detection:** the rising edges of channel 2 are identified → each edge defines θ = 0 at the start of a full waveplate rotation.

2. **Per-period Fourier fit:** with the known rotation angular frequency ω, the photodiode signal of each period is fit by least squares to the model:

   $$I(\theta) = \tfrac{1}{2}\bigl[A + B\sin(2\theta) + C\cos(4\theta) + D\sin(4\theta)\bigr]$$

3. **Conversion to Stokes parameters** using the real QWP retardance δ (Eq. 1–2 of the reference paper):

   $$S_0 = A - \frac{C}{\tan^2(\delta/2)}, \quad S_1 = \frac{C}{\sin^2(\delta/2)}, \quad S_2 = \frac{D}{\sin^2(\delta/2)}, \quad S_3 = \frac{B}{\sin\delta}$$

4. **Averaging across periods:** if the capture contains several periods, the coefficients are averaged and the standard deviation between periods is reported (as in Fig. 3 of the reference paper).

---

## Troubleshooting

**DOP > 1 (non-physical Stokes vector)**
Indicates a high relative noise level or an uncalibrated trigger phase. Check:
- Gain and offset of the photodiode channel in WaveForms (the 50 mV offset on channel 1 can affect the analysis).
- Illumination: make sure the photodiode signal has adequate amplitude.
- Calibrate the phase offset φ using light with known linear polarization (Section III of the reference paper).

**Only one period in the capture**
If the WaveForms capture window is too short (< 100 ms at ~13 Hz), the script analyzes only one period and cannot report uncertainty. Increase the capture duration to several tens of periods to obtain statistics.

**Error loading the CSV**
Verify that the file was exported from WaveForms with two active channels and that the decimal separator is a dot (`.`), not a comma.

**The Streamlit app does not open a browser**
Open `http://localhost:8501` manually in your browser, or run:

```bash
streamlit run app_polarimeter.py --server.headless false
```

---

## Citation

If you use this code, please cite the manuscript it accompanies, and the original polarimeter method:

> Wilkinson, Maurer, Flood, Lander, Chafin & Flagg, *"Complete Stokes vector
> analysis with a compact, portable rotating waveplate polarimeter"*,
> Rev. Sci. Instrum. 92, 093101 (2021). https://doi.org/10.1063/5.0052835

