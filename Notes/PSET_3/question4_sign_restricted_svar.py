#!/usr/bin/env python3
"""
Solve ECON 31730 PSET 3, Question 4.

The script loads cached FRED data, estimates a three-variable VAR(12), and
approximates the sign-restricted identified set by drawing Haar-distributed
orthogonal matrices using QR decompositions with positive R diagonals.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "question4_output"
FIGURE_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(OUTPUT_DIR / ".cache"))
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


Y_COLS = ["output_growth", "inflation", "real_rate"]
P = 12
HORIZON = 36
SIGN_HORIZONS = [0, 1, 2]
N_ROTATION_DRAWS = 1_000_000
HAAR_SEED = 317304
CHUNK_SIZE = 100_000


@dataclass
class VARResult:
    beta: np.ndarray
    intercept: np.ndarray
    a_mats: list[np.ndarray]
    residuals: np.ndarray
    sigma: np.ndarray
    dates: pd.DatetimeIndex


def month_label(ts: pd.Timestamp) -> str:
    return f"{ts.year}M{ts.month}"


def format_float(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def load_fred_series(series_id: str) -> pd.Series:
    path = DATA_DIR / f"{series_id}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Download it from FRED before running this script."
        )

    df = pd.read_csv(path, parse_dates=["observation_date"], na_values=["."])
    return df.set_index("observation_date")[series_id].astype(float).sort_index()


def prepare_sample() -> tuple[pd.DataFrame, dict[str, object]]:
    indpro = load_fred_series("INDPRO")
    pcepi = load_fred_series("PCEPI")
    gs1 = load_fred_series("GS1")

    # Annualized monthly log growth rates in percentage points.
    output_growth = 1200.0 * np.log(indpro).diff()
    inflation = 1200.0 * np.log(pcepi).diff()
    real_rate = gs1 - inflation

    df = pd.concat(
        [
            output_growth.rename("output_growth"),
            inflation.rename("inflation"),
            real_rate.rename("real_rate"),
        ],
        axis=1,
        join="inner",
    )
    df = df.loc["1984-01-01":"2019-12-01"].dropna().copy()

    meta = {
        "sample_start": month_label(df.index[0]),
        "sample_end": month_label(df.index[-1]),
        "sample_n_obs": int(len(df)),
        "raw_source": "FRED graph CSV files cached in PSET_3/data",
        "transformation": "1200 times monthly log growth for INDPRO and PCEPI; GS1 minus inflation for real_rate",
    }
    return df, meta


def build_var_xy(values: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
    n_obs, n_vars = values.shape
    y = values[p:]
    x_parts = [np.ones((n_obs - p, 1))]
    for lag in range(1, p + 1):
        x_parts.append(values[p - lag : n_obs - lag])
    return y, np.column_stack(x_parts)


def fit_var(df: pd.DataFrame, p: int) -> VARResult:
    values = df[Y_COLS].to_numpy()
    y, x = build_var_xy(values, p)
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    fitted = x @ beta
    residuals = y - fitted
    sigma = residuals.T @ residuals / len(residuals)

    n_vars = len(Y_COLS)
    a_mats = []
    for lag in range(p):
        block = beta[1 + lag * n_vars : 1 + (lag + 1) * n_vars]
        a_mats.append(block.T)

    return VARResult(
        beta=beta,
        intercept=beta[0],
        a_mats=a_mats,
        residuals=residuals,
        sigma=sigma,
        dates=df.index[p:],
    )


def companion_ma_matrices(a_mats: list[np.ndarray], horizon: int) -> list[np.ndarray]:
    n_vars = a_mats[0].shape[0]
    p = len(a_mats)
    state_dim = n_vars * p

    transition = np.zeros((state_dim, state_dim))
    transition[:n_vars, : n_vars * p] = np.hstack(a_mats)
    if p > 1:
        transition[n_vars:, :-n_vars] = np.eye(n_vars * (p - 1))

    shock_selector = np.zeros((state_dim, n_vars))
    shock_selector[:n_vars, :] = np.eye(n_vars)
    measurement = np.zeros((n_vars, state_dim))
    measurement[:, :n_vars] = np.eye(n_vars)

    psi = []
    transition_power = np.eye(state_dim)
    for _ in range(horizon + 1):
        psi.append(measurement @ transition_power @ shock_selector)
        transition_power = transition @ transition_power
    return psi


def haar_first_columns(
    rng: np.random.Generator,
    n_draws: int,
    n_vars: int,
    chunk_size: int,
):
    drawn = 0
    while drawn < n_draws:
        batch = min(chunk_size, n_draws - drawn)
        x = rng.standard_normal((batch, n_vars, n_vars))
        q, r = np.linalg.qr(x)
        signs = np.sign(np.diagonal(r, axis1=1, axis2=2))
        signs[signs == 0.0] = 1.0
        q = q * signs[:, None, :]
        drawn += batch
        yield q[:, :, 0]


def sign_restricted_draws(
    psi: list[np.ndarray],
    sigma: np.ndarray,
    n_draws: int,
    seed: int,
    chunk_size: int,
) -> tuple[np.ndarray, dict[str, float]]:
    output_idx = Y_COLS.index("output_growth")
    inflation_idx = Y_COLS.index("inflation")
    real_rate_idx = Y_COLS.index("real_rate")
    n_vars = len(Y_COLS)

    chol = np.linalg.cholesky(sigma)
    real_loadings = np.vstack([psi[h][real_rate_idx, :] @ chol for h in SIGN_HORIZONS])
    inflation_loadings = np.vstack([psi[h][inflation_idx, :] @ chol for h in SIGN_HORIZONS])
    output_loadings = np.vstack([psi[h][output_idx, :] @ chol for h in range(HORIZON + 1)])

    rng = np.random.default_rng(seed)
    accepted_irfs = []
    n_accepted = 0
    for q_first in haar_first_columns(rng, n_draws, n_vars, chunk_size):
        real_responses = q_first @ real_loadings.T
        inflation_responses = q_first @ inflation_loadings.T
        keep = np.all(real_responses >= -1e-12, axis=1) & np.all(
            inflation_responses <= 1e-12, axis=1
        )
        if np.any(keep):
            q_keep = q_first[keep]
            accepted_irfs.append(q_keep @ output_loadings.T)
            n_accepted += int(np.sum(keep))

    if not accepted_irfs:
        raise RuntimeError("No Haar draws satisfied the sign restrictions.")

    irfs = np.vstack(accepted_irfs)
    meta = {
        "rotation_draws": int(n_draws),
        "accepted_draws": int(n_accepted),
        "acceptance_rate": float(n_accepted / n_draws),
        "haar_seed": int(seed),
    }
    return irfs, meta


def write_outputs(
    df: pd.DataFrame,
    var: VARResult,
    irf_draws: np.ndarray,
    sample_meta: dict[str, object],
    draw_meta: dict[str, float],
) -> None:
    horizons = np.arange(HORIZON + 1)
    lower = np.min(irf_draws, axis=0)
    upper = np.max(irf_draws, axis=0)
    median = np.quantile(irf_draws, 0.50, axis=0)
    cred_lower = np.quantile(irf_draws, 0.16, axis=0)
    cred_upper = np.quantile(irf_draws, 0.84, axis=0)
    midpoint = 0.5 * (lower + upper)
    median_minus_midpoint = median - midpoint

    summary_df = pd.DataFrame(
        {
            "horizon": horizons,
            "identified_lower": lower,
            "identified_upper": upper,
            "identified_midpoint": midpoint,
            "posterior_median": median,
            "credible_68_lower": cred_lower,
            "credible_68_upper": cred_upper,
            "median_minus_midpoint": median_minus_midpoint,
        }
    )
    summary_df.to_csv(OUTPUT_DIR / "question4_irf_summary.csv", index=False)

    selected_h = [0, 6, 12, 24, 36]
    selected = {
        str(h): {
            "identified_lower": float(lower[h]),
            "identified_upper": float(upper[h]),
            "identified_midpoint": float(midpoint[h]),
            "posterior_median": float(median[h]),
            "credible_68_lower": float(cred_lower[h]),
            "credible_68_upper": float(cred_upper[h]),
            "median_minus_midpoint": float(median_minus_midpoint[h]),
        }
        for h in selected_h
    }
    max_abs_gap_idx = int(np.argmax(np.abs(median_minus_midpoint)))
    summary = {
        **sample_meta,
        **draw_meta,
        "lags": P,
        "horizon": HORIZON,
        "sign_horizons": SIGN_HORIZONS,
        "var_start": month_label(var.dates[0]),
        "var_end": month_label(var.dates[-1]),
        "var_n_obs": int(len(var.dates)),
        "var_max_abs_companion_eigenvalue": float(
            np.max(np.abs(np.linalg.eigvals(companion_matrix(var.a_mats))))
        ),
        "residual_covariance": var.sigma.tolist(),
        "selected_irfs": selected,
        "max_abs_median_midpoint_gap": float(np.max(np.abs(median_minus_midpoint))),
        "max_abs_median_midpoint_gap_horizon": max_abs_gap_idx,
    }
    (OUTPUT_DIR / "question4_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    save_figure(summary_df)
    write_latex_macros(summary)


def companion_matrix(a_mats: list[np.ndarray]) -> np.ndarray:
    n_vars = a_mats[0].shape[0]
    p = len(a_mats)
    state_dim = n_vars * p
    transition = np.zeros((state_dim, state_dim))
    transition[:n_vars, : n_vars * p] = np.hstack(a_mats)
    if p > 1:
        transition[n_vars:, :-n_vars] = np.eye(n_vars * (p - 1))
    return transition


def save_figure(summary_df: pd.DataFrame) -> None:
    horizons = summary_df["horizon"].to_numpy()
    lower = summary_df["identified_lower"].to_numpy()
    upper = summary_df["identified_upper"].to_numpy()
    midpoint = summary_df["identified_midpoint"].to_numpy()
    median = summary_df["posterior_median"].to_numpy()
    cred_lower = summary_df["credible_68_lower"].to_numpy()
    cred_upper = summary_df["credible_68_upper"].to_numpy()

    plt.figure(figsize=(7.2, 4.5))
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.fill_between(
        horizons,
        lower,
        upper,
        color="#d9d9d9",
        alpha=0.75,
        label="Identified set",
    )
    plt.fill_between(
        horizons,
        cred_lower,
        cred_upper,
        color="#6baed6",
        alpha=0.45,
        label="68% credible set",
    )
    plt.plot(horizons, median, color="#08519c", linewidth=2.0, label="Posterior median")
    plt.plot(
        horizons,
        midpoint,
        color="#7f2704",
        linewidth=1.6,
        linestyle="--",
        label="Identified-set midpoint",
    )
    plt.xlabel("Months after shock")
    plt.ylabel("Annualized percentage points")
    plt.title("Output-growth response under sign restrictions")
    plt.legend(frameon=False, ncol=2)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "q4_sign_restricted_output_irf.pdf")
    plt.close()


def write_latex_macros(summary: dict[str, object]) -> None:
    suffixes = {
        0: "Zero",
        6: "Six",
        12: "Twelve",
        24: "TwentyFour",
        36: "ThirtySix",
    }
    lines = [
        "% Auto-generated by question4_sign_restricted_svar.py",
        rf"\newcommand{{\QFourSampleStart}}{{{summary['sample_start']}}}",
        rf"\newcommand{{\QFourSampleEnd}}{{{summary['sample_end']}}}",
        rf"\newcommand{{\QFourSampleN}}{{{summary['sample_n_obs']}}}",
        rf"\newcommand{{\QFourVARStart}}{{{summary['var_start']}}}",
        rf"\newcommand{{\QFourVAREnd}}{{{summary['var_end']}}}",
        rf"\newcommand{{\QFourVARN}}{{{summary['var_n_obs']}}}",
        rf"\newcommand{{\QFourLags}}{{{summary['lags']}}}",
        rf"\newcommand{{\QFourHorizon}}{{{summary['horizon']}}}",
        rf"\newcommand{{\QFourRotationDraws}}{{{summary['rotation_draws']:,}}}",
        rf"\newcommand{{\QFourAcceptedDraws}}{{{summary['accepted_draws']:,}}}",
        rf"\newcommand{{\QFourAcceptanceRate}}{{{100.0 * summary['acceptance_rate']:.2f}}}",
        rf"\newcommand{{\QFourHaarSeed}}{{{summary['haar_seed']}}}",
        rf"\newcommand{{\QFourMaxEigen}}{{{format_float(summary['var_max_abs_companion_eigenvalue'], 4)}}}",
        rf"\newcommand{{\QFourMaxMedianGap}}{{{format_float(summary['max_abs_median_midpoint_gap'], 3)}}}",
        rf"\newcommand{{\QFourMaxMedianGapHorizon}}{{{summary['max_abs_median_midpoint_gap_horizon']}}}",
    ]
    for h, suffix in suffixes.items():
        vals = summary["selected_irfs"][str(h)]
        lines.extend(
            [
                rf"\newcommand{{\QFourLower{suffix}}}{{{format_float(vals['identified_lower'])}}}",
                rf"\newcommand{{\QFourUpper{suffix}}}{{{format_float(vals['identified_upper'])}}}",
                rf"\newcommand{{\QFourMid{suffix}}}{{{format_float(vals['identified_midpoint'])}}}",
                rf"\newcommand{{\QFourMedian{suffix}}}{{{format_float(vals['posterior_median'])}}}",
                rf"\newcommand{{\QFourCredLower{suffix}}}{{{format_float(vals['credible_68_lower'])}}}",
                rf"\newcommand{{\QFourCredUpper{suffix}}}{{{format_float(vals['credible_68_upper'])}}}",
            ]
        )

    (OUTPUT_DIR / "question4_generated.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    df, sample_meta = prepare_sample()
    var = fit_var(df, P)
    psi = companion_ma_matrices(var.a_mats, HORIZON)
    irf_draws, draw_meta = sign_restricted_draws(
        psi, var.sigma, N_ROTATION_DRAWS, HAAR_SEED, CHUNK_SIZE
    )
    write_outputs(df, var, irf_draws, sample_meta, draw_meta)

    summary = json.loads((OUTPUT_DIR / "question4_summary.json").read_text())
    print("Question 4 summary")
    print("-" * 72)
    print(
        f"Sample: {summary['sample_start']} to {summary['sample_end']} "
        f"(T = {summary['sample_n_obs']})"
    )
    print(
        f"VAR observations after {P} lags: {summary['var_n_obs']}; "
        f"max companion eigenvalue = {summary['var_max_abs_companion_eigenvalue']:.4f}"
    )
    print(
        f"Haar draws: {summary['rotation_draws']:,}; accepted: "
        f"{summary['accepted_draws']:,} ({100.0 * summary['acceptance_rate']:.2f}%)"
    )
    print("Selected output-growth IRFs:")
    for h in [0, 6, 12, 24, 36]:
        vals = summary["selected_irfs"][str(h)]
        print(
            f"  h={h:2d}: set=[{vals['identified_lower']: .3f}, "
            f"{vals['identified_upper']: .3f}], median={vals['posterior_median']: .3f}, "
            f"68%=[{vals['credible_68_lower']: .3f}, {vals['credible_68_upper']: .3f}]"
        )
    print(
        "Largest absolute median-midpoint gap: "
        f"{summary['max_abs_median_midpoint_gap']:.3f} at h="
        f"{summary['max_abs_median_midpoint_gap_horizon']}"
    )


if __name__ == "__main__":
    main()
