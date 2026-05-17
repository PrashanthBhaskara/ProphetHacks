#!/usr/bin/env python3
"""
Solve ECON 31730 PSET 2, Question 5.

The script:
1. loads monthly unemployment data from cached FRED CSV files,
2. estimates AR(p) spectra with p selected by BIC over p = 1, ..., 50,
3. estimates kernel spectra using the Epanechnikov / quadratic spectral kernel
   with bandwidth 10 periodogram ordinates,
4. saves clean figures for both estimators, and
5. writes machine-generated summary files for the LaTeX answer.

The formulas follow the "Spectral analysis II" lecture notes:
- AR spectral estimator and delta-method band: slides 2-4.
- Kernel spectral estimator and asymptotic band: slides 15-18.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(BASE_DIR / ".cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "question5_output"
FIG_DIR = OUTPUT_DIR / "figures"

START_DATE = "1948-01-01"
END_DATE = "2019-12-01"
MAX_LAGS = 50
KERNEL_BANDWIDTH = 10
AR_GRID_SIZE = 1500
CRIT_95 = 1.959963984540054
SEASONAL_OMEGA = 2.0 * math.pi / 12.0

SERIES_INFO = {
    "UNRATENSA": "Not seasonally adjusted",
    "UNRATE": "Seasonally adjusted",
}


@dataclass
class ARFit:
    p: int
    intercept: float
    phi: np.ndarray
    sigma2: float
    bic: float
    residuals: np.ndarray
    cov_beta: np.ndarray
    var_sigma2: float
    n_obs: int


@dataclass
class SeriesSummary:
    series_id: str
    label: str
    n_obs: int
    ar_bic_lag: int
    ar_log_spectrum_at_pi_over_6: float
    ar_local_seasonal_contrast: float
    kernel_log_spectrum_at_pi_over_6: float
    kernel_local_seasonal_contrast: float


def load_series(series_id: str) -> pd.Series:
    path = DATA_DIR / f"{series_id}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing data file {path}. Download the FRED CSV before running the script."
        )

    df = pd.read_csv(path, parse_dates=["observation_date"])
    mask = (df["observation_date"] >= START_DATE) & (df["observation_date"] <= END_DATE)
    sample = df.loc[mask, ["observation_date", series_id]].copy()
    sample[series_id] = sample[series_id].astype(float)
    return sample.set_index("observation_date")[series_id]


def fit_ar_order_common_sample(y: np.ndarray, p: int, p_max: int) -> ARFit:
    y = np.asarray(y, dtype=float)
    Y = y[p_max:]
    n_obs = len(Y)

    X = np.ones((n_obs, p + 1))
    for lag in range(1, p + 1):
        X[:, lag] = y[p_max - lag : len(y) - lag]

    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    residuals = Y - X @ beta
    sigma2 = float(np.mean(residuals**2))
    bic = float(np.log(sigma2) + p * np.log(n_obs) / n_obs)

    cov_beta = sigma2 * np.linalg.inv(X.T @ X)
    resid_sq = residuals**2
    denom = max(n_obs - p, 1)
    var_sigma2 = float(np.var(resid_sq, ddof=1) / denom)

    return ARFit(
        p=p,
        intercept=float(beta[0]),
        phi=beta[1:].copy(),
        sigma2=sigma2,
        bic=bic,
        residuals=residuals,
        cov_beta=cov_beta,
        var_sigma2=var_sigma2,
        n_obs=n_obs,
    )


def select_bic_ar(y: np.ndarray, p_max: int) -> tuple[ARFit, list[float]]:
    fits = [fit_ar_order_common_sample(y, p, p_max) for p in range(1, p_max + 1)]
    best_fit = min(fits, key=lambda fit: fit.bic)
    return best_fit, [fit.bic for fit in fits]


def ar_log_spectrum_and_se(fit: ARFit, omega: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lags = np.arange(1, fit.p + 1)
    cos_terms = np.cos(np.outer(omega, lags))
    sin_terms = np.sin(np.outer(omega, lags))

    c_term = cos_terms @ fit.phi
    s_term = sin_terms @ fit.phi
    denom = (1.0 - c_term) ** 2 + s_term**2

    log_spectrum = np.log(fit.sigma2) - np.log(2.0 * np.pi) - np.log(denom)

    grad_phi = 2.0 * ((1.0 - c_term)[:, None] * cos_terms - s_term[:, None] * sin_terms)
    grad_phi /= denom[:, None]

    cov_phi = fit.cov_beta[1:, 1:]
    var_from_phi = np.einsum("ij,jk,ik->i", grad_phi, cov_phi, grad_phi)
    var_from_sigma2 = fit.var_sigma2 / (fit.sigma2**2)
    se = np.sqrt(np.maximum(var_from_phi + var_from_sigma2, 0.0))

    return log_spectrum, se


def epanechnikov_weights(bandwidth: int) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.arange(-bandwidth, bandwidth + 1)
    raw_weights = 1.0 - (offsets / bandwidth) ** 2
    weights = raw_weights / raw_weights.sum()
    return offsets, weights


def kernel_log_spectrum(y: np.ndarray, bandwidth: int) -> tuple[np.ndarray, np.ndarray, float]:
    y = np.asarray(y, dtype=float)
    n_obs = len(y)

    dft = np.fft.fft(y) / math.sqrt(n_obs)
    periodogram = np.abs(dft) ** 2
    periodogram[0] = periodogram[1]

    offsets, weights = epanechnikov_weights(bandwidth)
    positive_idx = np.arange(n_obs // 2 + 1)

    spectrum = np.empty_like(positive_idx, dtype=float)
    for pos, freq_idx in enumerate(positive_idx):
        spectrum[pos] = np.sum(weights * periodogram[(freq_idx + offsets) % n_obs]) / (
            2.0 * np.pi
        )

    omega = 2.0 * np.pi * positive_idx / n_obs
    log_spectrum = np.log(spectrum)
    log_se = float(math.sqrt(np.sum(weights**2)))
    return omega, log_spectrum, log_se


def closest_index(grid: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(grid - target)))


def local_contrast(log_spectrum: np.ndarray, idx: int) -> float:
    if idx <= 0 or idx >= len(log_spectrum) - 1:
        return float("nan")
    return float(log_spectrum[idx] - 0.5 * (log_spectrum[idx - 1] + log_spectrum[idx + 1]))


def build_series_summary(
    series_id: str,
    label: str,
    n_obs: int,
    ar_fit: ARFit,
    ar_omega: np.ndarray,
    ar_log_spectrum: np.ndarray,
    kernel_omega: np.ndarray,
    kernel_log_spectrum: np.ndarray,
) -> SeriesSummary:
    ar_idx = closest_index(ar_omega, SEASONAL_OMEGA)
    kernel_idx = closest_index(kernel_omega, SEASONAL_OMEGA)

    return SeriesSummary(
        series_id=series_id,
        label=label,
        n_obs=n_obs,
        ar_bic_lag=ar_fit.p,
        ar_log_spectrum_at_pi_over_6=float(ar_log_spectrum[ar_idx]),
        ar_local_seasonal_contrast=local_contrast(ar_log_spectrum, ar_idx),
        kernel_log_spectrum_at_pi_over_6=float(kernel_log_spectrum[kernel_idx]),
        kernel_local_seasonal_contrast=local_contrast(kernel_log_spectrum, kernel_idx),
    )


def setup_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.15,
            "grid.linewidth": 0.6,
        }
    )


def plot_two_panel(
    save_stem: str,
    title: str,
    x_values: dict[str, np.ndarray],
    y_values: dict[str, np.ndarray],
    lower_band: dict[str, np.ndarray],
    upper_band: dict[str, np.ndarray],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=True)
    panel_order = ["UNRATENSA", "UNRATE"]

    band_min = min(float(lower_band[s].min()) for s in panel_order)
    band_max = max(float(upper_band[s].max()) for s in panel_order)
    y_pad = 0.08 * (band_max - band_min)

    for ax, series_id in zip(axes, panel_order):
        x_axis = x_values[series_id] / (2.0 * np.pi)
        ax.fill_between(
            x_axis,
            lower_band[series_id],
            upper_band[series_id],
            color="#d9e8f5",
            alpha=0.9,
            linewidth=0,
        )
        ax.plot(x_axis, y_values[series_id], color="#124c7c", linewidth=2.0)
        ax.set_title(SERIES_INFO[series_id], fontsize=12)
        ax.set_xlim(0.0, 0.5)
        ax.set_xlabel("frequency/(2*pi)")
        ax.set_ylim(band_min - y_pad, band_max + y_pad)

    axes[0].set_ylabel("log spectrum")
    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()

    for extension in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{save_stem}.{extension}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_summary_files(summary: dict[str, object]) -> None:
    json_path = OUTPUT_DIR / "question5_summary.json"
    tex_path = OUTPUT_DIR / "question5_generated.tex"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    sa = summary["series"]["UNRATE"]
    nsa = summary["series"]["UNRATENSA"]

    tex_lines = [
        "% Auto-generated by question5_unemployment_spectra.py",
        rf"\newcommand{{\QFiveObs}}{{{summary['n_obs']}}}",
        rf"\newcommand{{\QFiveLagNSA}}{{{nsa['ar_bic_lag']}}}",
        rf"\newcommand{{\QFiveLagSA}}{{{sa['ar_bic_lag']}}}",
        rf"\newcommand{{\QFiveARAnnualNSA}}{{{nsa['ar_log_spectrum_at_pi_over_6']:.2f}}}",
        rf"\newcommand{{\QFiveARAnnualSA}}{{{sa['ar_log_spectrum_at_pi_over_6']:.2f}}}",
        rf"\newcommand{{\QFiveKernelAnnualNSA}}{{{nsa['kernel_log_spectrum_at_pi_over_6']:.2f}}}",
        rf"\newcommand{{\QFiveKernelAnnualSA}}{{{sa['kernel_log_spectrum_at_pi_over_6']:.2f}}}",
        rf"\newcommand{{\QFiveARAnnualGap}}{{{nsa['ar_log_spectrum_at_pi_over_6'] - sa['ar_log_spectrum_at_pi_over_6']:.2f}}}",
        rf"\newcommand{{\QFiveKernelAnnualGap}}{{{nsa['kernel_log_spectrum_at_pi_over_6'] - sa['kernel_log_spectrum_at_pi_over_6']:.2f}}}",
        rf"\newcommand{{\QFiveARContrastNSA}}{{{nsa['ar_local_seasonal_contrast']:.2f}}}",
        rf"\newcommand{{\QFiveARContrastSA}}{{{sa['ar_local_seasonal_contrast']:.2f}}}",
        rf"\newcommand{{\QFiveKernelContrastNSA}}{{{nsa['kernel_local_seasonal_contrast']:.2f}}}",
        rf"\newcommand{{\QFiveKernelContrastSA}}{{{sa['kernel_local_seasonal_contrast']:.2f}}}",
        rf"\newcommand{{\QFiveKernelBandwidth}}{{{summary['kernel_bandwidth']}}}",
    ]

    tex_path.write_text("\n".join(tex_lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    setup_plot_style()

    ar_x: dict[str, np.ndarray] = {}
    ar_y: dict[str, np.ndarray] = {}
    ar_lower: dict[str, np.ndarray] = {}
    ar_upper: dict[str, np.ndarray] = {}

    kernel_x: dict[str, np.ndarray] = {}
    kernel_y: dict[str, np.ndarray] = {}
    kernel_lower: dict[str, np.ndarray] = {}
    kernel_upper: dict[str, np.ndarray] = {}

    series_summaries: dict[str, dict[str, object]] = {}

    for series_id, label in SERIES_INFO.items():
        series = load_series(series_id)
        y = series.to_numpy()

        ar_fit, bic_values = select_bic_ar(y, MAX_LAGS)
        ar_omega = np.linspace(0.0, np.pi, AR_GRID_SIZE)
        ar_log_spectrum, ar_se = ar_log_spectrum_and_se(ar_fit, ar_omega)

        kernel_omega, kernel_log_spectrum_vals, kernel_log_se = kernel_log_spectrum(
            y, KERNEL_BANDWIDTH
        )

        ar_x[series_id] = ar_omega
        ar_y[series_id] = ar_log_spectrum
        ar_lower[series_id] = ar_log_spectrum - CRIT_95 * ar_se
        ar_upper[series_id] = ar_log_spectrum + CRIT_95 * ar_se

        kernel_x[series_id] = kernel_omega
        kernel_y[series_id] = kernel_log_spectrum_vals
        kernel_lower[series_id] = kernel_log_spectrum_vals - CRIT_95 * kernel_log_se
        kernel_upper[series_id] = kernel_log_spectrum_vals + CRIT_95 * kernel_log_se

        summary_obj = build_series_summary(
            series_id=series_id,
            label=label,
            n_obs=len(y),
            ar_fit=ar_fit,
            ar_omega=ar_omega,
            ar_log_spectrum=ar_log_spectrum,
            kernel_omega=kernel_omega,
            kernel_log_spectrum=kernel_log_spectrum_vals,
        )

        series_summaries[series_id] = {
            **asdict(summary_obj),
            "bic_values": bic_values,
        }

    plot_two_panel(
        save_stem="q5_ar_log_spectrum",
        title="AR(BIC) log spectral density with pointwise 95% band",
        x_values=ar_x,
        y_values=ar_y,
        lower_band=ar_lower,
        upper_band=ar_upper,
    )

    plot_two_panel(
        save_stem="q5_kernel_log_spectrum",
        title="Kernel log spectral density with pointwise 95% band",
        x_values=kernel_x,
        y_values=kernel_y,
        lower_band=kernel_lower,
        upper_band=kernel_upper,
    )

    summary = {
        "sample_start": "1948:1",
        "sample_end": "2019:12",
        "n_obs": int(next(iter(series_summaries.values()))["n_obs"]),
        "max_lags": MAX_LAGS,
        "kernel_bandwidth": KERNEL_BANDWIDTH,
        "annual_frequency_over_2pi": 1.0 / 12.0,
        "series": series_summaries,
    }
    write_summary_files(summary)

    print("Question 5 summary")
    print("-" * 72)
    print(f"Sample: {summary['sample_start']} to {summary['sample_end']} ({summary['n_obs']} months)")
    for series_id in ("UNRATENSA", "UNRATE"):
        stats = summary["series"][series_id]
        print(f"{stats['label']}:")
        print(f"  BIC AR lag length: {stats['ar_bic_lag']}")
        print(
            f"  AR log spectrum at annual frequency (pi/6): "
            f"{stats['ar_log_spectrum_at_pi_over_6']:.3f}"
        )
        print(
            f"  Kernel log spectrum at annual frequency (pi/6): "
            f"{stats['kernel_log_spectrum_at_pi_over_6']:.3f}"
        )
        print(
            f"  AR local seasonal contrast: {stats['ar_local_seasonal_contrast']:.3f}"
        )
        print(
            f"  Kernel local seasonal contrast: "
            f"{stats['kernel_local_seasonal_contrast']:.3f}"
        )


if __name__ == "__main__":
    main()
