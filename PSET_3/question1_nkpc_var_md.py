#!/usr/bin/env python3
"""
Solve ECON 31730 PSET 3, Question 1.

The script uses the same cached FRED series as PSET 2, Question 6, estimates
the reduced-form VAR(1) for inflation and unemployment, computes the
minimum-distance NKPC parameters implied by the VAR coefficients, and reports
recursive residual bootstrap confidence intervals.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
REPO_DIR = BASE_DIR.parent
PSET2_DATA_DIR = REPO_DIR / "PSET_2" / "data"
OUTPUT_DIR = BASE_DIR / "question1_output"

DROP_FROM_DATE = "2020-01-01"
PARAMETER_NAMES = ["gamma_f", "lambda", "rho_pi", "rho_x"]
VAR_COEFFICIENT_NAMES = ["pi_lag_pi", "pi_lag_x", "x_lag_pi", "x_lag_x"]
BOOTSTRAP_REPS = 10000
BOOTSTRAP_SEED = 31730

# This matches the professor's Matlab solution convention:
# inflation is a quarter-over-quarter log change in percentage points.
INFLATION_SCALE = 100.0


@dataclass
class VARResult:
    intercept: np.ndarray
    a_matrix: np.ndarray
    residuals: np.ndarray
    fitted: np.ndarray
    response_dates: pd.DatetimeIndex


def quarter_label(timestamp: pd.Timestamp) -> str:
    quarter = (timestamp.month - 1) // 3 + 1
    return f"{timestamp.year}Q{quarter}"


def format_float(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}"


def latex_sci(value: float, digits: int = 3) -> str:
    if value == 0:
        return "0"
    if 1e-3 <= abs(value) < 1e4:
        return f"{value:.6f}"
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / (10.0**exponent)
    return rf"{mantissa:.{digits}f}\times 10^{{{exponent}}}"


def load_csv_series(series_id: str) -> pd.Series:
    path = PSET2_DATA_DIR / f"{series_id}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing required data file: {path}")

    df = pd.read_csv(path, parse_dates=["observation_date"])
    return df.set_index("observation_date")[series_id].astype(float)


def prepare_sample() -> tuple[pd.DataFrame, dict[str, object]]:
    gdpdef = load_csv_series("GDPDEF")
    unrate_m = load_csv_series("UNRATE")

    inflation = INFLATION_SCALE * np.log(gdpdef).diff()
    unrate_q = unrate_m.resample("QS").mean()

    df = pd.concat(
        [inflation.rename("pi"), unrate_q.rename("x")],
        axis=1,
        join="inner",
    )
    df = df[df.index < DROP_FROM_DATE].dropna().copy()

    meta = {
        "raw_start": quarter_label(df.index[0]),
        "raw_end": quarter_label(df.index[-1]),
        "raw_n_obs": int(len(df)),
        "inflation_scale": INFLATION_SCALE,
    }
    return df, meta


def fit_var1(df: pd.DataFrame) -> VARResult:
    y = df[["pi", "x"]].to_numpy()
    x_reg = np.column_stack([np.ones(len(y) - 1), y[:-1]])
    y_resp = y[1:]

    beta = np.linalg.solve(x_reg.T @ x_reg, x_reg.T @ y_resp)
    fitted = x_reg @ beta
    residuals = y_resp - fitted

    return VARResult(
        intercept=beta[0],
        a_matrix=beta[1:].T,
        residuals=residuals,
        fitted=fitted,
        response_dates=df.index[1:],
    )


def theta_from_var_coefficients(a_matrix: np.ndarray) -> np.ndarray:
    rho_pi = float(a_matrix[0, 0])
    rho_x = float(a_matrix[0, 1])
    a21 = float(a_matrix[1, 0])
    a22 = float(a_matrix[1, 1])

    gamma_den = a21 * rho_pi * rho_x + a22 * (1.0 - rho_pi**2)
    gamma_num = a22 * (1.0 - rho_pi) + a21 * rho_x
    if abs(gamma_den) < 1e-12:
        raise FloatingPointError("Minimum-distance mapping is nearly singular.")

    gamma_f = gamma_num / gamma_den

    if abs(a22) >= 1e-12:
        denominator = (1.0 - gamma_f * rho_pi) * rho_x / a22
    elif abs(a21) >= 1e-12:
        denominator = (rho_pi - 1.0 + gamma_f * (1.0 - rho_pi**2)) / a21
    else:
        raise FloatingPointError("Cannot recover lambda from zero second VAR row.")

    lam = denominator - gamma_f * rho_x
    return np.array([gamma_f, lam, rho_pi, rho_x], dtype=float)


def var_coefficients_from_theta(theta: np.ndarray) -> np.ndarray:
    gamma_f, lam, rho_pi, rho_x = theta
    denominator = lam + gamma_f * rho_x
    if abs(denominator) < 1e-12:
        raise FloatingPointError("Structural denominator is nearly zero.")

    a21 = ((1.0 - gamma_f * rho_pi) * rho_pi - (1.0 - gamma_f)) / denominator
    a22 = ((1.0 - gamma_f * rho_pi) * rho_x) / denominator
    return np.array([rho_pi, rho_x, a21, a22], dtype=float)


def recursive_residual_bootstrap(
    df: pd.DataFrame,
    var_result: VARResult,
    reps: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = df[["pi", "x"]].to_numpy()
    n_obs = len(y)
    residuals = var_result.residuals - var_result.residuals.mean(axis=0, keepdims=True)
    n_resid = residuals.shape[0]

    draws = np.full((reps, len(PARAMETER_NAMES)), np.nan)
    for b in range(reps):
        boot_resid = residuals[rng.integers(0, n_resid, size=n_resid)]
        y_star = np.empty_like(y)
        y_star[0] = y[0]
        for t in range(1, n_obs):
            y_star[t] = var_result.intercept + var_result.a_matrix @ y_star[t - 1] + boot_resid[t - 1]

        df_star = pd.DataFrame(y_star, index=df.index, columns=["pi", "x"])
        try:
            draws[b] = theta_from_var_coefficients(fit_var1(df_star).a_matrix)
        except FloatingPointError:
            continue

    good = np.all(np.isfinite(draws), axis=1)
    return draws[good]


def write_outputs(summary: dict[str, object], bootstrap_draws: np.ndarray) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    (OUTPUT_DIR / "question1_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    var_records = []
    a = np.array(summary["var_a_matrix"], dtype=float)
    row_names = ["pi_t", "x_t"]
    col_names = ["pi_t_minus_1", "x_t_minus_1"]
    for i, row in enumerate(row_names):
        for j, col in enumerate(col_names):
            var_records.append({"equation": row, "regressor": col, "estimate": a[i, j]})
    pd.DataFrame.from_records(var_records).to_csv(
        OUTPUT_DIR / "question1_var_coefficients.csv", index=False
    )

    theta_records = []
    theta = np.array(summary["theta_hat"], dtype=float)
    ci_lower = np.array(summary["ci_lower"], dtype=float)
    ci_upper = np.array(summary["ci_upper"], dtype=float)
    for i, name in enumerate(PARAMETER_NAMES):
        theta_records.append(
            {
                "parameter": name,
                "estimate": theta[i],
                "ci_lower": ci_lower[i],
                "ci_upper": ci_upper[i],
            }
        )
    pd.DataFrame.from_records(theta_records).to_csv(
        OUTPUT_DIR / "question1_parameter_estimates.csv", index=False
    )

    pd.DataFrame(bootstrap_draws, columns=PARAMETER_NAMES).to_csv(
        OUTPUT_DIR / "question1_bootstrap_draws.csv", index=False
    )

    lines = [
        "% Auto-generated by question1_nkpc_var_md.py",
        rf"\newcommand{{\QOneRawStart}}{{{summary['raw_start']}}}",
        rf"\newcommand{{\QOneRawEnd}}{{{summary['raw_end']}}}",
        rf"\newcommand{{\QOneRawN}}{{{summary['raw_n_obs']}}}",
        rf"\newcommand{{\QOneVARStart}}{{{summary['var_start']}}}",
        rf"\newcommand{{\QOneVAREnd}}{{{summary['var_end']}}}",
        rf"\newcommand{{\QOneVARN}}{{{summary['var_n_obs']}}}",
        rf"\newcommand{{\QOneBootstrapReps}}{{{summary['bootstrap_reps']}}}",
        rf"\newcommand{{\QOneBootstrapSeed}}{{{summary['bootstrap_seed']}}}",
        rf"\newcommand{{\QOneBootstrapUsed}}{{{summary['bootstrap_used_reps']}}}",
        rf"\newcommand{{\QOneMaxEigen}}{{{format_float(summary['max_abs_eigenvalue'], 4)}}}",
        rf"\newcommand{{\QOneInterceptPi}}{{{latex_sci(summary['var_intercept'][0])}}}",
        rf"\newcommand{{\QOneInterceptX}}{{{format_float(summary['var_intercept'][1], 4)}}}",
        rf"\newcommand{{\QOneAOneOne}}{{{format_float(a[0, 0], 4)}}}",
        rf"\newcommand{{\QOneAOneTwo}}{{{latex_sci(a[0, 1])}}}",
        rf"\newcommand{{\QOneATwoOne}}{{{format_float(a[1, 0], 4)}}}",
        rf"\newcommand{{\QOneATwoTwo}}{{{format_float(a[1, 1], 4)}}}",
    ]

    macro_prefixes = ["GammaF", "Lambda", "RhoPi", "RhoX"]
    for i, prefix in enumerate(macro_prefixes):
        formatter = latex_sci if prefix in {"Lambda", "RhoX"} else format_float
        lines.extend(
            [
                rf"\newcommand{{\QOne{prefix}}}{{{formatter(theta[i])}}}",
                rf"\newcommand{{\QOne{prefix}CILower}}{{{formatter(ci_lower[i])}}}",
                rf"\newcommand{{\QOne{prefix}CIUpper}}{{{formatter(ci_upper[i])}}}",
            ]
        )

    (OUTPUT_DIR / "question1_generated.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    df, meta = prepare_sample()
    var_result = fit_var1(df)
    theta_hat = theta_from_var_coefficients(var_result.a_matrix)
    implied_a = var_coefficients_from_theta(theta_hat).reshape(2, 2)
    bootstrap_draws = recursive_residual_bootstrap(
        df, var_result, BOOTSTRAP_REPS, BOOTSTRAP_SEED
    )
    ci_lower, ci_upper = np.quantile(bootstrap_draws, [0.025, 0.975], axis=0)
    eigenvalues = np.linalg.eigvals(var_result.a_matrix)

    summary = {
        **meta,
        "var_start": quarter_label(var_result.response_dates[0]),
        "var_end": quarter_label(var_result.response_dates[-1]),
        "var_n_obs": int(len(var_result.response_dates)),
        "var_intercept": var_result.intercept.tolist(),
        "var_a_matrix": var_result.a_matrix.tolist(),
        "theta_hat": theta_hat.tolist(),
        "implied_var_coefficients": implied_a.ravel().tolist(),
        "ci_method": "recursive residual bootstrap, Efron percentile quantiles",
        "ci_lower": ci_lower.tolist(),
        "ci_upper": ci_upper.tolist(),
        "bootstrap_reps": BOOTSTRAP_REPS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_used_reps": int(len(bootstrap_draws)),
        "max_abs_eigenvalue": float(np.max(np.abs(eigenvalues))),
    }
    write_outputs(summary, bootstrap_draws)

    print("Question 1 summary")
    print("-" * 72)
    print(f"Raw sample: {summary['raw_start']} to {summary['raw_end']} (T = {summary['raw_n_obs']})")
    print(f"VAR sample: {summary['var_start']} to {summary['var_end']} (T = {summary['var_n_obs']})")
    print(f"Max absolute VAR eigenvalue: {summary['max_abs_eigenvalue']:.4f}")
    print()
    print("VAR(1) slope matrix A:")
    print(var_result.a_matrix)
    print()
    print("Minimum-distance estimates and 95% bootstrap CIs:")
    for i, name in enumerate(PARAMETER_NAMES):
        print(
            f"  {name:>8s} = {theta_hat[i]: .8g} "
            f"[{ci_lower[i]: .8g}, {ci_upper[i]: .8g}]"
        )
    print(f"\nBootstrap draws used: {len(bootstrap_draws)} / {BOOTSTRAP_REPS}")


if __name__ == "__main__":
    main()
