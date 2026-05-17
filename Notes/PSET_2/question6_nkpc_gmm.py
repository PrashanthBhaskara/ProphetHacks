#!/usr/bin/env python3
"""
Solve ECON 31730 PSET 2, Question 6.

This script:
1. loads cached FRED data for GDPDEF and UNRATE,
2. constructs quarterly inflation and quarterly-average unemployment,
3. estimates the NKPC by 2SLS / one-step GMM and efficient two-step GMM,
4. computes HAC-robust standard errors with a Bartlett/Newey-West LRV estimator,
5. reports Hansen's J-test, and
6. writes machine-generated summary files for the LaTeX answer.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "question6_output"

DROP_FROM_DATE = "2020-01-01"
PARAMETER_NAMES = ["c", "gamma_f", "gamma_b", "lambda"]


@dataclass
class GMMResult:
    method: str
    theta: np.ndarray
    se: np.ndarray
    objective: float


def load_csv_series(series_id: str) -> pd.Series:
    path = DATA_DIR / f"{series_id}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Download the FRED CSV into PSET_2/data before running."
        )

    df = pd.read_csv(path, parse_dates=["observation_date"])
    return df.set_index("observation_date")[series_id].astype(float)


def quarter_label(timestamp: pd.Timestamp) -> str:
    quarter = (timestamp.month - 1) // 3 + 1
    return f"{timestamp.year}Q{quarter}"


def prepare_sample() -> tuple[pd.DataFrame, dict[str, object]]:
    gdpdef = load_csv_series("GDPDEF")
    unrate_m = load_csv_series("UNRATE")

    unrate_q = unrate_m.resample("QS").mean()
    # The assignment defines inflation as quarter-over-quarter log growth,
    # not annualized inflation.
    inflation = np.log(gdpdef).diff()

    overlap = pd.concat(
        [inflation.rename("pi"), unrate_q.rename("x")],
        axis=1,
        join="inner",
    )
    overlap = overlap[overlap.index < DROP_FROM_DATE].dropna().copy()

    df = overlap.copy()
    for lag in (1, 2, 3):
        df[f"pi_lag{lag}"] = df["pi"].shift(lag)
        df[f"x_lag{lag}"] = df["x"].shift(lag)
    df["pi_lead1"] = df["pi"].shift(-1)
    df = df.dropna().copy()

    meta = {
        "raw_start": quarter_label(overlap.index[0]),
        "raw_end": quarter_label(overlap.index[-1]),
        "raw_n_obs": int(len(overlap)),
        "estimation_start": quarter_label(df.index[0]),
        "estimation_end": quarter_label(df.index[-1]),
        "estimation_n_obs": int(len(df)),
    }
    return df, meta


def build_matrices(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = df["pi"].to_numpy()

    x = np.column_stack(
        [
            np.ones(len(df)),
            df["pi_lead1"].to_numpy(),
            df["pi_lag1"].to_numpy(),
            df["x"].to_numpy(),
        ]
    )

    z = np.column_stack(
        [
            np.ones(len(df)),
            df["pi_lag1"].to_numpy(),
            df["pi_lag2"].to_numpy(),
            df["pi_lag3"].to_numpy(),
            df["x_lag1"].to_numpy(),
            df["x_lag2"].to_numpy(),
            df["x_lag3"].to_numpy(),
        ]
    )

    return y, x, z


def automatic_bandwidth(n_obs: int) -> int:
    return max(int(np.floor(4.0 * (n_obs / 100.0) ** (2.0 / 9.0))), 1)


def moment_matrix(y: np.ndarray, x: np.ndarray, z: np.ndarray, theta: np.ndarray) -> np.ndarray:
    residual = y - x @ theta
    return z * residual[:, None]


def hac_long_run_variance(g: np.ndarray, bandwidth: int) -> np.ndarray:
    n_obs = g.shape[0]
    centered = g - g.mean(axis=0, keepdims=True)

    omega = centered.T @ centered / n_obs
    for lag in range(1, bandwidth + 1):
        weight = 1.0 - lag / (bandwidth + 1.0)
        gamma = centered[lag:].T @ centered[:-lag] / n_obs
        omega += weight * (gamma + gamma.T)
    return omega


def linear_gmm_estimate(y: np.ndarray, x: np.ndarray, z: np.ndarray, weight: np.ndarray) -> np.ndarray:
    n_obs = len(y)
    zx = z.T @ x / n_obs
    zy = z.T @ y / n_obs
    return np.linalg.solve(zx.T @ weight @ zx, zx.T @ weight @ zy)


def sandwich_vcov(x: np.ndarray, z: np.ndarray, weight: np.ndarray, omega: np.ndarray) -> np.ndarray:
    n_obs = x.shape[0]
    derivative = -(z.T @ x / n_obs)
    middle = derivative.T @ weight @ derivative
    inv_middle = np.linalg.inv(middle)
    asymptotic = inv_middle @ (derivative.T @ weight @ omega @ weight @ derivative) @ inv_middle
    return asymptotic / n_obs


def gmm_objective(g_bar: np.ndarray, weight: np.ndarray) -> float:
    return float(g_bar @ weight @ g_bar)


def format_float(value: float) -> str:
    if abs(value) < 0.01:
        return f"{value:.6f}"
    return f"{value:.4f}"


def write_outputs(summary: dict[str, object]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    json_path = OUTPUT_DIR / "question6_summary.json"
    csv_path = OUTPUT_DIR / "question6_estimates.csv"
    tex_path = OUTPUT_DIR / "question6_generated.tex"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    records: list[dict[str, object]] = []
    for result_key in ("two_sls", "efficient_gmm"):
        result = summary[result_key]
        for name, estimate, se in zip(PARAMETER_NAMES, result["theta"], result["se"]):
            records.append(
                {
                    "method": result["method"],
                    "parameter": name,
                    "estimate": estimate,
                    "std_error": se,
                }
            )
    pd.DataFrame.from_records(records).to_csv(csv_path, index=False)

    two_sls = summary["two_sls"]
    eff = summary["efficient_gmm"]
    j_test = summary["j_test"]

    tex_lines = [
        "% Auto-generated by question6_nkpc_gmm.py",
        rf"\newcommand{{\QSixRawStart}}{{{summary['raw_start']}}}",
        rf"\newcommand{{\QSixRawEnd}}{{{summary['raw_end']}}}",
        rf"\newcommand{{\QSixRawN}}{{{summary['raw_n_obs']}}}",
        rf"\newcommand{{\QSixEstStart}}{{{summary['estimation_start']}}}",
        rf"\newcommand{{\QSixEstEnd}}{{{summary['estimation_end']}}}",
        rf"\newcommand{{\QSixEstN}}{{{summary['estimation_n_obs']}}}",
        rf"\newcommand{{\QSixBandwidth}}{{{summary['hac_bandwidth']}}}",
        rf"\newcommand{{\QSixCOne}}{{{format_float(two_sls['theta'][0])}}}",
        rf"\newcommand{{\QSixGFOne}}{{{format_float(two_sls['theta'][1])}}}",
        rf"\newcommand{{\QSixGBOne}}{{{format_float(two_sls['theta'][2])}}}",
        rf"\newcommand{{\QSixLamOne}}{{{format_float(two_sls['theta'][3])}}}",
        rf"\newcommand{{\QSixCOneSE}}{{{format_float(two_sls['se'][0])}}}",
        rf"\newcommand{{\QSixGFOneSE}}{{{format_float(two_sls['se'][1])}}}",
        rf"\newcommand{{\QSixGBOneSE}}{{{format_float(two_sls['se'][2])}}}",
        rf"\newcommand{{\QSixLamOneSE}}{{{format_float(two_sls['se'][3])}}}",
        rf"\newcommand{{\QSixCTwo}}{{{format_float(eff['theta'][0])}}}",
        rf"\newcommand{{\QSixGFTwo}}{{{format_float(eff['theta'][1])}}}",
        rf"\newcommand{{\QSixGBTwo}}{{{format_float(eff['theta'][2])}}}",
        rf"\newcommand{{\QSixLamTwo}}{{{format_float(eff['theta'][3])}}}",
        rf"\newcommand{{\QSixCTwoSE}}{{{format_float(eff['se'][0])}}}",
        rf"\newcommand{{\QSixGFTwoSE}}{{{format_float(eff['se'][1])}}}",
        rf"\newcommand{{\QSixGBTwoSE}}{{{format_float(eff['se'][2])}}}",
        rf"\newcommand{{\QSixLamTwoSE}}{{{format_float(eff['se'][3])}}}",
        rf"\newcommand{{\QSixJStat}}{{{j_test['statistic']:.4f}}}",
        rf"\newcommand{{\QSixJPValue}}{{{j_test['p_value']:.4f}}}",
    ]
    tex_path.write_text("\n".join(tex_lines) + "\n", encoding="utf-8")


def main() -> None:
    df, meta = prepare_sample()
    y, x, z = build_matrices(df)
    n_obs = len(y)

    weight_one = np.linalg.inv(z.T @ z / n_obs)
    theta_one = linear_gmm_estimate(y, x, z, weight_one)
    g_one = moment_matrix(y, x, z, theta_one)

    bandwidth = automatic_bandwidth(n_obs)
    omega_one = hac_long_run_variance(g_one, bandwidth)
    vcov_one = sandwich_vcov(x, z, weight_one, omega_one)

    weight_two = np.linalg.inv(omega_one)
    theta_two = linear_gmm_estimate(y, x, z, weight_two)
    g_two = moment_matrix(y, x, z, theta_two)
    omega_two = hac_long_run_variance(g_two, bandwidth)
    vcov_two = sandwich_vcov(x, z, weight_two, omega_two)

    j_stat = n_obs * g_two.mean(axis=0) @ np.linalg.inv(omega_two) @ g_two.mean(axis=0)
    j_df = z.shape[1] - x.shape[1]
    j_p_value = float(1.0 - chi2.cdf(j_stat, j_df))

    result_one = GMMResult(
        method="2SLS / one-step GMM",
        theta=theta_one,
        se=np.sqrt(np.diag(vcov_one)),
        objective=gmm_objective(g_one.mean(axis=0), weight_one),
    )
    result_two = GMMResult(
        method="Efficient two-step GMM",
        theta=theta_two,
        se=np.sqrt(np.diag(vcov_two)),
        objective=gmm_objective(g_two.mean(axis=0), weight_two),
    )

    summary = {
        **meta,
        "hac_bandwidth": bandwidth,
        "two_sls": {
            "method": result_one.method,
            "theta": result_one.theta.tolist(),
            "se": result_one.se.tolist(),
            "objective": result_one.objective,
        },
        "efficient_gmm": {
            "method": result_two.method,
            "theta": result_two.theta.tolist(),
            "se": result_two.se.tolist(),
            "objective": result_two.objective,
        },
        "j_test": {
            "statistic": float(j_stat),
            "df": int(j_df),
            "p_value": j_p_value,
        },
    }
    write_outputs(summary)

    print("Question 6 summary")
    print("-" * 72)
    print(
        f"Raw overlap sample: {summary['raw_start']} to {summary['raw_end']} "
        f"(T = {summary['raw_n_obs']})"
    )
    print(
        f"Estimation sample: {summary['estimation_start']} to {summary['estimation_end']} "
        f"(T = {summary['estimation_n_obs']})"
    )
    print(f"HAC bandwidth: L = {bandwidth}")
    print()

    for label, result in (
        ("2SLS / one-step GMM", result_one),
        ("Efficient two-step GMM", result_two),
    ):
        print(label)
        for name, estimate, se in zip(PARAMETER_NAMES, result.theta, result.se):
            print(f"  {name:>7s} = {format_float(float(estimate))}  (se {format_float(float(se))})")
        print()

    print(f"Hansen J-statistic: {j_stat:.4f}")
    print(f"J-test p-value:     {j_p_value:.4f}")


if __name__ == "__main__":
    main()
