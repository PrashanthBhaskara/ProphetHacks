#!/usr/bin/env python3
"""
Solve ECON 31730 PSET 3, Question 3.

The script estimates:
1. SVAR-IV impulse responses and the EBP forecast variance decomposition
   using ff4_tc as an external instrument for the monetary policy shock;
2. LP-IV responses from a recursive VAR with the IV ordered first;
3. LP-IV responses from horizon-by-horizon 2SLS regressions; and
4. the Granger-causality implication of invertibility;
5. recursive-residual bootstrap confidence intervals for the optional
   confidence-interval question.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, f


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "question3_output"
FIGURE_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(OUTPUT_DIR / ".cache"))
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATA_PATH = BASE_DIR / "gk_data.csv"
P = 4
HORIZON = 36
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 317303
Y_COLS = ["ipgr", "infl", "gs1", "ebp"]
W_COLS = ["ff4_tc", *Y_COLS]
POLICY_VAR = "gs1"
OUTCOME_VAR = "ebp"
IV_VAR = "ff4_tc"


@dataclass
class VARResult:
    columns: list[str]
    beta: np.ndarray
    intercept: np.ndarray
    a_mats: list[np.ndarray]
    residuals: np.ndarray
    sigma: np.ndarray
    fitted: np.ndarray
    x: np.ndarray
    y: np.ndarray
    dates: pd.DatetimeIndex


def month_label(ts: pd.Timestamp) -> str:
    return f"{ts.year}M{ts.month}"


def format_float(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(
        {"year": df["year"].astype(int), "month": df["month"].astype(int), "day": 1}
    )
    df = df.set_index("date").sort_index()
    return df[[*Y_COLS, IV_VAR]].astype(float)


def build_var_xy(values: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
    n_obs, n_vars = values.shape
    y = values[p:]
    x_parts = [np.ones((n_obs - p, 1))]
    for lag in range(1, p + 1):
        x_parts.append(values[p - lag : n_obs - lag])
    x = np.column_stack(x_parts)
    return y, x


def fit_var(df: pd.DataFrame, columns: list[str], p: int) -> VARResult:
    values = df[columns].to_numpy()
    y, x = build_var_xy(values, p)
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    fitted = x @ beta
    residuals = y - fitted
    sigma = residuals.T @ residuals / len(residuals)
    n_vars = len(columns)
    a_mats = []
    for lag in range(p):
        block = beta[1 + lag * n_vars : 1 + (lag + 1) * n_vars]
        a_mats.append(block.T)
    return VARResult(
        columns=columns,
        beta=beta,
        intercept=beta[0],
        a_mats=a_mats,
        residuals=residuals,
        sigma=sigma,
        fitted=fitted,
        x=x,
        y=y,
        dates=df.index[p:],
    )


def ma_matrices(a_mats: list[np.ndarray], horizon: int) -> list[np.ndarray]:
    n_vars = a_mats[0].shape[0]
    p = len(a_mats)
    psi = [np.eye(n_vars)]
    for h in range(1, horizon + 1):
        mat = np.zeros((n_vars, n_vars))
        for lag in range(1, min(p, h) + 1):
            mat += a_mats[lag - 1] @ psi[h - lag]
        psi.append(mat)
    return psi


def svar_iv(df: pd.DataFrame) -> dict[str, object]:
    var = fit_var(df, Y_COLS, P)
    z = df[IV_VAR].to_numpy()[P:]
    gamma = var.residuals.T @ z / len(z)
    sigma_inv = np.linalg.inv(var.sigma)
    h_unit = gamma / np.sqrt(float(gamma.T @ sigma_inv @ gamma))

    policy_idx = Y_COLS.index(POLICY_VAR)
    outcome_idx = Y_COLS.index(OUTCOME_VAR)
    if h_unit[policy_idx] < 0:
        h_unit = -h_unit
    h_norm = h_unit / h_unit[policy_idx]

    psi = ma_matrices(var.a_mats, HORIZON)
    irfs = np.array([mat @ h_norm for mat in psi])
    unit_irfs = np.array([mat @ h_unit for mat in psi])

    fvd = []
    numerator = 0.0
    denominator = 0.0
    e = np.zeros(len(Y_COLS))
    e[outcome_idx] = 1.0
    for h in range(1, HORIZON + 1):
        response_h_minus_1 = float(e @ psi[h - 1] @ h_unit)
        numerator += response_h_minus_1**2
        denominator += float(e @ psi[h - 1] @ var.sigma @ psi[h - 1].T @ e)
        fvd.append(numerator / denominator)

    first_stage = first_stage_regression(var.residuals[:, policy_idx], z)
    return {
        "var": var,
        "impact_vector_unit": h_unit,
        "impact_vector_normalized": h_norm,
        "irfs": irfs,
        "unit_irfs": unit_irfs,
        "fvd": np.array(fvd),
        "first_stage": first_stage,
    }


def recursive_var_lpiv(df: pd.DataFrame) -> dict[str, object]:
    var = fit_var(df, W_COLS, P)
    chol = np.linalg.cholesky(var.sigma)
    shock = chol[:, 0].copy()
    policy_idx = W_COLS.index(POLICY_VAR)
    outcome_idx = W_COLS.index(OUTCOME_VAR)
    if shock[policy_idx] < 0:
        shock = -shock
    shock_norm = shock / shock[policy_idx]
    psi = ma_matrices(var.a_mats, HORIZON)
    irfs = np.array([mat @ shock_norm for mat in psi])
    return {"var": var, "impact_vector_normalized": shock_norm, "irfs": irfs}


def residualize(v: np.ndarray, controls: np.ndarray) -> np.ndarray:
    coef = np.linalg.lstsq(controls, v, rcond=None)[0]
    return v - controls @ coef


def lpiv_2sls(df: pd.DataFrame) -> dict[str, object]:
    y_values = df[Y_COLS].to_numpy()
    z_values = df[IV_VAR].to_numpy()
    policy_idx = Y_COLS.index(POLICY_VAR)
    outcome_idx = Y_COLS.index(OUTCOME_VAR)
    n_obs = len(df)

    coefs = []
    first_stage_f = []
    first_stage_partial_r2 = []
    n_by_h = []
    for horizon in range(HORIZON + 1):
        rows = np.arange(P, n_obs - horizon)
        outcome = y_values[rows + horizon, outcome_idx]
        endogenous = y_values[rows, policy_idx]
        instrument = z_values[rows]
        controls_parts = [np.ones((len(rows), 1))]
        for lag in range(1, P + 1):
            controls_parts.append(y_values[rows - lag])
            controls_parts.append(z_values[rows - lag, None])
        controls = np.column_stack(controls_parts)

        y_tilde = residualize(outcome, controls)
        x_tilde = residualize(endogenous, controls)
        z_tilde = residualize(instrument, controls)
        denom = float(z_tilde @ x_tilde)
        beta = float(z_tilde @ y_tilde / denom)
        coefs.append(beta)

        f_stat, partial_r2 = first_stage_with_controls(endogenous, instrument, controls)
        first_stage_f.append(f_stat)
        first_stage_partial_r2.append(partial_r2)
        n_by_h.append(len(rows))

    return {
        "irf": np.array(coefs),
        "first_stage_f": np.array(first_stage_f),
        "first_stage_partial_r2": np.array(first_stage_partial_r2),
        "n_by_h": np.array(n_by_h),
    }


def first_stage_regression(y: np.ndarray, z: np.ndarray) -> dict[str, float]:
    x = np.column_stack([np.ones(len(z)), z])
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    resid = y - x @ beta
    sse = float(resid @ resid)
    centered = y - y.mean()
    sst = float(centered @ centered)
    sigma2 = sse / (len(y) - x.shape[1])
    vcov = sigma2 * np.linalg.inv(x.T @ x)
    t_stat = float(beta[1] / np.sqrt(vcov[1, 1]))
    return {
        "coef": float(beta[1]),
        "f_stat": t_stat**2,
        "p_value": float(1.0 - f.cdf(t_stat**2, 1, len(y) - x.shape[1])),
        "r2": 1.0 - sse / sst,
    }


def first_stage_with_controls(
    endogenous: np.ndarray,
    instrument: np.ndarray,
    controls: np.ndarray,
) -> tuple[float, float]:
    x_unrestricted = np.column_stack([controls, instrument])
    beta_u = np.linalg.lstsq(x_unrestricted, endogenous, rcond=None)[0]
    resid_u = endogenous - x_unrestricted @ beta_u
    sse_u = float(resid_u @ resid_u)

    beta_r = np.linalg.lstsq(controls, endogenous, rcond=None)[0]
    resid_r = endogenous - controls @ beta_r
    sse_r = float(resid_r @ resid_r)

    df_denom = len(endogenous) - x_unrestricted.shape[1]
    f_stat = ((sse_r - sse_u) / 1.0) / (sse_u / df_denom)
    partial_r2 = (sse_r - sse_u) / sse_r
    return float(f_stat), float(partial_r2)


def invertibility_test(df: pd.DataFrame) -> dict[str, float]:
    values_y = df[Y_COLS].to_numpy()
    values_w = df[W_COLS].to_numpy()
    y_dep = values_y[P:]
    _, x_unrestricted = build_var_xy(values_w, P)

    x_parts = [np.ones((len(df) - P, 1))]
    for lag in range(1, P + 1):
        x_parts.append(values_y[P - lag : len(df) - lag])
    x_restricted = np.column_stack(x_parts)

    beta_u = np.linalg.lstsq(x_unrestricted, y_dep, rcond=None)[0]
    resid_u = y_dep - x_unrestricted @ beta_u
    sigma_u = resid_u.T @ resid_u / len(resid_u)

    beta_r = np.linalg.lstsq(x_restricted, y_dep, rcond=None)[0]
    resid_r = y_dep - x_restricted @ beta_r
    sigma_r = resid_r.T @ resid_r / len(resid_r)

    stat = len(resid_u) * (np.linalg.slogdet(sigma_r)[1] - np.linalg.slogdet(sigma_u)[1])
    df_restr = P * len(Y_COLS)
    p_value = 1.0 - chi2.cdf(stat, df_restr)
    return {"lr_stat": float(stat), "df": int(df_restr), "p_value": float(p_value)}


def generate_var_bootstrap_sample(
    base_df: pd.DataFrame,
    var: VARResult,
    rng: np.random.Generator,
) -> pd.DataFrame:
    values = base_df[var.columns].to_numpy()
    n_obs, n_vars = values.shape
    residuals = var.residuals - var.residuals.mean(axis=0, keepdims=True)
    draw_idx = rng.integers(0, len(residuals), size=n_obs - P)

    boot_values = np.empty_like(values)
    boot_values[:P] = values[:P]
    for t in range(P, n_obs):
        x_parts = [1.0]
        for lag in range(1, P + 1):
            x_parts.extend(boot_values[t - lag])
        x_row = np.asarray(x_parts)
        boot_values[t] = x_row @ var.beta + residuals[draw_idx[t - P]]

    return pd.DataFrame(boot_values, index=base_df.index, columns=var.columns)


def bootstrap_confidence_intervals(
    df: pd.DataFrame,
    reps: int = BOOTSTRAP_REPS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    joint_var = fit_var(df, W_COLS, P)

    svar_ebp = []
    fvd = []
    lpvar_ebp = []
    lp2sls_ebp = []
    outcome_idx_y = Y_COLS.index(OUTCOME_VAR)
    outcome_idx_w = W_COLS.index(OUTCOME_VAR)

    for _ in range(reps):
        boot_df = generate_var_bootstrap_sample(df[W_COLS], joint_var, rng)
        try:
            svar_b = svar_iv(boot_df)
            lpvar_b = recursive_var_lpiv(boot_df)
            lp2sls_b = lpiv_2sls(boot_df)
        except (np.linalg.LinAlgError, FloatingPointError, ValueError):
            continue

        svar_path = svar_b["irfs"][:, outcome_idx_y]
        fvd_path = svar_b["fvd"]
        lpvar_path = lpvar_b["irfs"][:, outcome_idx_w]
        lp2sls_path = lp2sls_b["irf"]
        if (
            np.all(np.isfinite(svar_path))
            and np.all(np.isfinite(fvd_path))
            and np.all(np.isfinite(lpvar_path))
            and np.all(np.isfinite(lp2sls_path))
        ):
            svar_ebp.append(svar_path)
            fvd.append(fvd_path)
            lpvar_ebp.append(lpvar_path)
            lp2sls_ebp.append(lp2sls_path)

    if len(svar_ebp) == 0:
        raise RuntimeError("No valid bootstrap repetitions were produced.")

    draws = {
        "svar_ebp": np.asarray(svar_ebp),
        "fvd": np.asarray(fvd),
        "lpvar_ebp": np.asarray(lpvar_ebp),
        "lp2sls_ebp": np.asarray(lp2sls_ebp),
    }
    ci = {}
    for key, value in draws.items():
        ci[key] = {
            "lower": np.quantile(value, 0.05, axis=0),
            "upper": np.quantile(value, 0.95, axis=0),
        }
    return {
        "draws": draws,
        "ci": ci,
        "requested_reps": int(reps),
        "used_reps": int(len(svar_ebp)),
        "seed": int(seed),
    }


def save_figures(
    svar: dict[str, object],
    lpvar: dict[str, object],
    lp2sls: dict[str, object],
    bootstrap: dict[str, object] | None = None,
) -> None:
    horizons = np.arange(HORIZON + 1)
    outcome_idx_y = Y_COLS.index(OUTCOME_VAR)
    policy_idx_y = Y_COLS.index(POLICY_VAR)
    outcome_idx_w = W_COLS.index(OUTCOME_VAR)

    plt.figure(figsize=(6.8, 4.2))
    plt.axhline(0, color="black", linewidth=0.8)
    plt.plot(horizons, svar["irfs"][:, outcome_idx_y], color="#1f77b4", linewidth=2)
    plt.xlabel("Months after shock")
    plt.ylabel("Percentage points")
    plt.title("SVAR-IV response of EBP to a 100 bp GS1 shock")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "q3_svar_iv_ebp_irf.pdf")
    plt.close()

    fvd_horizons = np.arange(1, HORIZON + 1)
    plt.figure(figsize=(6.8, 4.2))
    plt.plot(fvd_horizons, 100.0 * svar["fvd"], color="#d62728", linewidth=2)
    plt.xlabel("Forecast horizon in months")
    plt.ylabel("Percent of forecast-error variance")
    plt.title("SVAR-IV FVD of EBP")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "q3_svar_iv_ebp_fvd.pdf")
    plt.close()

    plt.figure(figsize=(6.8, 4.2))
    plt.axhline(0, color="black", linewidth=0.8)
    plt.plot(horizons, svar["irfs"][:, outcome_idx_y], label="SVAR-IV", linewidth=2)
    plt.plot(
        horizons,
        lpvar["irfs"][:, outcome_idx_w],
        label="LP-IV recursive VAR",
        linewidth=2,
        linestyle="--",
    )
    plt.plot(horizons, lp2sls["irf"], label="LP-IV 2SLS", linewidth=2, linestyle=":")
    plt.xlabel("Months after shock")
    plt.ylabel("Percentage points")
    plt.title("EBP response to a 100 bp GS1 shock")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "q3_ebp_irf_comparison.pdf")
    plt.close()

    plt.figure(figsize=(6.8, 4.2))
    plt.axhline(0, color="black", linewidth=0.8)
    plt.plot(horizons, svar["irfs"][:, policy_idx_y], label="SVAR-IV", linewidth=2)
    plt.plot(
        horizons,
        lpvar["irfs"][:, W_COLS.index(POLICY_VAR)],
        label="LP-IV recursive VAR",
        linewidth=2,
        linestyle="--",
    )
    plt.xlabel("Months after shock")
    plt.ylabel("Percentage points")
    plt.title("GS1 response after 100 bp impact normalization")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "q3_gs1_irf_comparison.pdf")
    plt.close()

    if bootstrap is None:
        return

    ci = bootstrap["ci"]
    fvd_horizons = np.arange(1, HORIZON + 1)
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.9), sharex=False)
    panels = [
        (
            axes[0, 0],
            horizons,
            svar["irfs"][:, outcome_idx_y],
            ci["svar_ebp"]["lower"],
            ci["svar_ebp"]["upper"],
            "SVAR-IV EBP IRF",
            "Percentage points",
            "#1f77b4",
        ),
        (
            axes[0, 1],
            fvd_horizons,
            100.0 * svar["fvd"],
            100.0 * ci["fvd"]["lower"],
            100.0 * ci["fvd"]["upper"],
            "SVAR-IV EBP FVD",
            "Percent",
            "#d62728",
        ),
        (
            axes[1, 0],
            horizons,
            lpvar["irfs"][:, outcome_idx_w],
            ci["lpvar_ebp"]["lower"],
            ci["lpvar_ebp"]["upper"],
            "LP-IV recursive VAR EBP IRF",
            "Percentage points",
            "#2ca02c",
        ),
        (
            axes[1, 1],
            horizons,
            lp2sls["irf"],
            ci["lp2sls_ebp"]["lower"],
            ci["lp2sls_ebp"]["upper"],
            "LP-IV 2SLS EBP IRF",
            "Percentage points",
            "#9467bd",
        ),
    ]
    for ax, x, point, lower, upper, title, ylabel, color in panels:
        ax.axhline(0, color="black", linewidth=0.7)
        ax.fill_between(x, lower, upper, color=color, alpha=0.18, linewidth=0)
        ax.plot(x, point, color=color, linewidth=1.9)
        ax.set_title(title)
        ax.set_xlabel("Months after shock")
        ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "q3_optional_ci_panels.pdf")
    plt.close(fig)


def write_outputs(
    df: pd.DataFrame,
    svar: dict[str, object],
    lpvar: dict[str, object],
    lp2sls: dict[str, object],
    inv_test: dict[str, float],
    bootstrap: dict[str, object] | None = None,
) -> None:
    horizons = np.arange(HORIZON + 1)
    outcome_idx_y = Y_COLS.index(OUTCOME_VAR)
    outcome_idx_w = W_COLS.index(OUTCOME_VAR)
    policy_idx_y = Y_COLS.index(POLICY_VAR)

    irf_df = pd.DataFrame(
        {
            "horizon": horizons,
            "svar_iv_ebp": svar["irfs"][:, outcome_idx_y],
            "svar_iv_gs1": svar["irfs"][:, policy_idx_y],
            "lpiv_recursive_var_ebp": lpvar["irfs"][:, outcome_idx_w],
            "lpiv_2sls_ebp": lp2sls["irf"],
        }
    )
    irf_df.to_csv(OUTPUT_DIR / "question3_irfs.csv", index=False)

    fvd_df = pd.DataFrame(
        {
            "horizon": np.arange(1, HORIZON + 1),
            "svar_iv_ebp_fvd": svar["fvd"],
        }
    )
    fvd_df.to_csv(OUTPUT_DIR / "question3_fvd.csv", index=False)

    if bootstrap is not None:
        ci = bootstrap["ci"]
        ci_irf_df = pd.DataFrame(
            {
                "horizon": horizons,
                "svar_iv_ebp": svar["irfs"][:, outcome_idx_y],
                "svar_iv_ebp_ci90_lower": ci["svar_ebp"]["lower"],
                "svar_iv_ebp_ci90_upper": ci["svar_ebp"]["upper"],
                "lpiv_recursive_var_ebp": lpvar["irfs"][:, outcome_idx_w],
                "lpiv_recursive_var_ebp_ci90_lower": ci["lpvar_ebp"]["lower"],
                "lpiv_recursive_var_ebp_ci90_upper": ci["lpvar_ebp"]["upper"],
                "lpiv_2sls_ebp": lp2sls["irf"],
                "lpiv_2sls_ebp_ci90_lower": ci["lp2sls_ebp"]["lower"],
                "lpiv_2sls_ebp_ci90_upper": ci["lp2sls_ebp"]["upper"],
            }
        )
        ci_irf_df.to_csv(OUTPUT_DIR / "question3_bootstrap_ci_irfs.csv", index=False)

        ci_fvd_df = pd.DataFrame(
            {
                "horizon": np.arange(1, HORIZON + 1),
                "svar_iv_ebp_fvd": svar["fvd"],
                "svar_iv_ebp_fvd_ci90_lower": ci["fvd"]["lower"],
                "svar_iv_ebp_fvd_ci90_upper": ci["fvd"]["upper"],
            }
        )
        ci_fvd_df.to_csv(OUTPUT_DIR / "question3_bootstrap_ci_fvd.csv", index=False)

    selected_h = [0, 6, 12, 24, 36]
    selected_fvd_h = [1, 6, 12, 24, 36]
    macro_suffix = {
        0: "Zero",
        1: "One",
        6: "Six",
        12: "Twelve",
        24: "TwentyFour",
        36: "ThirtySix",
    }
    summary = {
        "sample_start": month_label(df.index[0]),
        "sample_end": month_label(df.index[-1]),
        "raw_n_obs": int(len(df)),
        "lags": P,
        "horizon": HORIZON,
        "var_n_obs": int(len(svar["var"].residuals)),
        "svar_first_stage": svar["first_stage"],
        "lpiv_2sls_first_stage_h0": {
            "f_stat": float(lp2sls["first_stage_f"][0]),
            "partial_r2": float(lp2sls["first_stage_partial_r2"][0]),
        },
        "invertibility_test": inv_test,
        "bootstrap": None
        if bootstrap is None
        else {
            "requested_reps": bootstrap["requested_reps"],
            "used_reps": bootstrap["used_reps"],
            "seed": bootstrap["seed"],
        },
        "selected_irfs": {
            str(h): {
                "svar_iv_ebp": float(svar["irfs"][h, outcome_idx_y]),
                "lpiv_recursive_var_ebp": float(lpvar["irfs"][h, outcome_idx_w]),
                "lpiv_2sls_ebp": float(lp2sls["irf"][h]),
            }
            for h in selected_h
        },
        "selected_fvd": {
            str(h): float(svar["fvd"][h - 1]) for h in selected_fvd_h
        },
    }
    if bootstrap is not None:
        ci = bootstrap["ci"]
        selected_ci_h = [0, 12, 36]
        selected_fvd_ci_h = [1, 12, 36]
        summary["selected_ci"] = {
            str(h): {
                "svar_iv_ebp_lower": float(ci["svar_ebp"]["lower"][h]),
                "svar_iv_ebp_upper": float(ci["svar_ebp"]["upper"][h]),
                "lpiv_recursive_var_ebp_lower": float(ci["lpvar_ebp"]["lower"][h]),
                "lpiv_recursive_var_ebp_upper": float(ci["lpvar_ebp"]["upper"][h]),
                "lpiv_2sls_ebp_lower": float(ci["lp2sls_ebp"]["lower"][h]),
                "lpiv_2sls_ebp_upper": float(ci["lp2sls_ebp"]["upper"][h]),
            }
            for h in selected_ci_h
        }
        summary["selected_fvd_ci"] = {
            str(h): {
                "lower": float(ci["fvd"]["lower"][h - 1]),
                "upper": float(ci["fvd"]["upper"][h - 1]),
            }
            for h in selected_fvd_ci_h
        }
    (OUTPUT_DIR / "question3_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    lines = [
        "% Auto-generated by question3_gk_svar_lpiv.py",
        rf"\newcommand{{\QThreeSampleStart}}{{{summary['sample_start']}}}",
        rf"\newcommand{{\QThreeSampleEnd}}{{{summary['sample_end']}}}",
        rf"\newcommand{{\QThreeRawN}}{{{summary['raw_n_obs']}}}",
        rf"\newcommand{{\QThreeLags}}{{{P}}}",
        rf"\newcommand{{\QThreeHorizon}}{{{HORIZON}}}",
        rf"\newcommand{{\QThreeVARN}}{{{summary['var_n_obs']}}}",
        rf"\newcommand{{\QThreeSVARFirstStageF}}{{{format_float(summary['svar_first_stage']['f_stat'], 2)}}}",
        rf"\newcommand{{\QThreeSVARFirstStageRTwo}}{{{format_float(100.0 * summary['svar_first_stage']['r2'], 2)}}}",
        rf"\newcommand{{\QThreeLPFirstStageF}}{{{format_float(summary['lpiv_2sls_first_stage_h0']['f_stat'], 2)}}}",
        rf"\newcommand{{\QThreeLPPartialRTwo}}{{{format_float(100.0 * summary['lpiv_2sls_first_stage_h0']['partial_r2'], 2)}}}",
        rf"\newcommand{{\QThreeInvertLR}}{{{format_float(inv_test['lr_stat'], 2)}}}",
        rf"\newcommand{{\QThreeInvertDF}}{{{inv_test['df']}}}",
        rf"\newcommand{{\QThreeInvertPValue}}{{{format_float(inv_test['p_value'], 3)}}}",
    ]
    if bootstrap is not None:
        lines.extend(
            [
                rf"\newcommand{{\QThreeBootstrapReps}}{{{bootstrap['requested_reps']:,}}}",
                rf"\newcommand{{\QThreeBootstrapUsed}}{{{bootstrap['used_reps']:,}}}",
                rf"\newcommand{{\QThreeBootstrapSeed}}{{{bootstrap['seed']}}}",
            ]
        )
    for h in selected_h:
        vals = summary["selected_irfs"][str(h)]
        suffix = macro_suffix[h]
        lines.extend(
            [
                rf"\newcommand{{\QThreeSVAREBPH{suffix}}}{{{format_float(vals['svar_iv_ebp'], 3)}}}",
                rf"\newcommand{{\QThreeLPVAREBPH{suffix}}}{{{format_float(vals['lpiv_recursive_var_ebp'], 3)}}}",
                rf"\newcommand{{\QThreeLPTwoSLSEBPH{suffix}}}{{{format_float(vals['lpiv_2sls_ebp'], 3)}}}",
            ]
        )
    for h in selected_fvd_h:
        suffix = macro_suffix[h]
        lines.append(
            rf"\newcommand{{\QThreeFVDH{suffix}}}{{{format_float(100.0 * summary['selected_fvd'][str(h)], 2)}}}"
        )
    if bootstrap is not None:
        ci_suffix_h = [0, 12, 36]
        for h in ci_suffix_h:
            vals = summary["selected_irfs"][str(h)]
            ci_vals = summary["selected_ci"][str(h)]
            suffix = macro_suffix[h]
            lines.extend(
                [
                    rf"\newcommand{{\QThreeCISVAREBPLower{suffix}}}{{{format_float(ci_vals['svar_iv_ebp_lower'], 3)}}}",
                    rf"\newcommand{{\QThreeCISVAREBPUpper{suffix}}}{{{format_float(ci_vals['svar_iv_ebp_upper'], 3)}}}",
                    rf"\newcommand{{\QThreeCILPVAREBPLower{suffix}}}{{{format_float(ci_vals['lpiv_recursive_var_ebp_lower'], 3)}}}",
                    rf"\newcommand{{\QThreeCILPVAREBPUpper{suffix}}}{{{format_float(ci_vals['lpiv_recursive_var_ebp_upper'], 3)}}}",
                    rf"\newcommand{{\QThreeCILPTwoSLSEBPLower{suffix}}}{{{format_float(ci_vals['lpiv_2sls_ebp_lower'], 3)}}}",
                    rf"\newcommand{{\QThreeCILPTwoSLSEBPUpper{suffix}}}{{{format_float(ci_vals['lpiv_2sls_ebp_upper'], 3)}}}",
                    rf"\newcommand{{\QThreeCIIRFTableSVAR{suffix}}}{{{format_float(vals['svar_iv_ebp'], 3)} [{format_float(ci_vals['svar_iv_ebp_lower'], 3)}, {format_float(ci_vals['svar_iv_ebp_upper'], 3)}]}}",
                    rf"\newcommand{{\QThreeCIIRFTableLPVAR{suffix}}}{{{format_float(vals['lpiv_recursive_var_ebp'], 3)} [{format_float(ci_vals['lpiv_recursive_var_ebp_lower'], 3)}, {format_float(ci_vals['lpiv_recursive_var_ebp_upper'], 3)}]}}",
                    rf"\newcommand{{\QThreeCIIRFTableLPTwoSLS{suffix}}}{{{format_float(vals['lpiv_2sls_ebp'], 3)} [{format_float(ci_vals['lpiv_2sls_ebp_lower'], 3)}, {format_float(ci_vals['lpiv_2sls_ebp_upper'], 3)}]}}",
                ]
            )
        for h in [1, 12, 36]:
            suffix = macro_suffix[h]
            fvd_point = 100.0 * summary["selected_fvd"][str(h)]
            fvd_ci = summary["selected_fvd_ci"][str(h)]
            lines.extend(
                [
                    rf"\newcommand{{\QThreeCIFVDLower{suffix}}}{{{format_float(100.0 * fvd_ci['lower'], 2)}}}",
                    rf"\newcommand{{\QThreeCIFVDUpper{suffix}}}{{{format_float(100.0 * fvd_ci['upper'], 2)}}}",
                    rf"\newcommand{{\QThreeCIFVDTable{suffix}}}{{{format_float(fvd_point, 2)} [{format_float(100.0 * fvd_ci['lower'], 2)}, {format_float(100.0 * fvd_ci['upper'], 2)}]}}",
                ]
            )
    (OUTPUT_DIR / "question3_generated.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    df = load_data()
    svar = svar_iv(df)
    lpvar = recursive_var_lpiv(df)
    lp2sls = lpiv_2sls(df)
    inv_test = invertibility_test(df)
    bootstrap = bootstrap_confidence_intervals(df)

    save_figures(svar, lpvar, lp2sls, bootstrap)
    write_outputs(df, svar, lpvar, lp2sls, inv_test, bootstrap)

    print("Question 3 summary")
    print("-" * 72)
    print(f"Sample: {month_label(df.index[0])} to {month_label(df.index[-1])} (T = {len(df)})")
    print(f"VAR observations after {P} lags: {len(svar['var'].residuals)}")
    print(
        "SVAR-IV first-stage F for GS1 residual on FF4: "
        f"{svar['first_stage']['f_stat']:.2f}"
    )
    print(
        "LP-IV 2SLS first-stage F at h=0: "
        f"{lp2sls['first_stage_f'][0]:.2f}"
    )
    print(
        "Invertibility LR test: "
        f"stat={inv_test['lr_stat']:.2f}, df={inv_test['df']}, p={inv_test['p_value']:.3f}"
    )
    print(
        "Bootstrap repetitions: "
        f"{bootstrap['used_reps']} valid out of {bootstrap['requested_reps']} "
        f"(seed {bootstrap['seed']})"
    )
    print("Selected EBP IRFs:")
    for h in [0, 6, 12, 24, 36]:
        print(
            f"  h={h:2d}: SVAR-IV={svar['irfs'][h, Y_COLS.index(OUTCOME_VAR)]: .3f}, "
            f"LP-VAR={lpvar['irfs'][h, W_COLS.index(OUTCOME_VAR)]: .3f}, "
            f"2SLS={lp2sls['irf'][h]: .3f}"
        )


if __name__ == "__main__":
    main()
