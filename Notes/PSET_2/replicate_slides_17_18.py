#!/usr/bin/env python3
r"""
Replicate slides 17 and 18 from "Stationary models III":
- Simulation study: sampling density of \hat{rho}_1
- Simulation study: sampling distribution of \hat{p}

The simulation setup in the slides is:
    y_t = nu + rho_1 y_{t-1} + rho_2 y_{t-2} + u_t,
    u_t ~ iid N(0, sigma^2),
with true parameters
    p = 2, nu = 0, rho_1 = 0.6, rho_2 = 0.2, sigma^2 = 1,
sample size T = 200, candidate lag orders p in {1,2,3},
lag order selected by BIC or AIC, and repeated 10,000 times.

This script reproduces the figures closely while keeping the code transparent.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


# -----------------------------
# User-facing configuration
# -----------------------------
N_REPS = 10_000
T = 200
P_MAX = 3
BURN_IN = 500
SEED = 12345
SAVE_DIR = Path(__file__).resolve().parent

# True DGP from the slides
NU_TRUE = 0.0
RHO_TRUE = np.array([0.6, 0.2])
SIGMA2_TRUE = 1.0


@dataclass
class SimulationResults:
    phat_bic: np.ndarray
    phat_aic: np.ndarray
    rho1hat_bic: np.ndarray
    rho1hat_aic: np.ndarray


def simulate_true_ar2(
    rng: np.random.Generator,
    T: int,
    p_max: int,
    burn_in: int,
    nu: float,
    rho: np.ndarray,
    sigma2: float,
) -> np.ndarray:
    """
    Simulate from the true AR(2) model and return exactly T + p_max observations.

    We keep p_max pre-sample observations so that AR(1), AR(2), and AR(3)
    can all be estimated on the same effective sample size T, matching the
    convention in the slides.
    """
    total_len = burn_in + T + p_max
    y = np.zeros(total_len)
    shocks = rng.normal(loc=0.0, scale=np.sqrt(sigma2), size=total_len)

    for t in range(2, total_len):
        y[t] = nu + rho[0] * y[t - 1] + rho[1] * y[t - 2] + shocks[t]

    return y[-(T + p_max):]


def fit_ar_common_sample(y: np.ndarray, p: int, p_max: int) -> Dict[str, np.ndarray | float]:
    """
    Fit AR(p) with intercept by OLS using a common effective sample size T.

    y has length T + p_max, where the first p_max entries are used as pre-sample.
    The dependent sample is y[p_max:], so every candidate model uses the same T rows.

    Returns:
        beta       : OLS coefficients [intercept, rho_1, ..., rho_p]
        sigma2_hat : SSR / T   (the variance proxy used in the slide IC formulas)
        bic        : log(sigma2_hat) + p * log(T) / T   for n = 1
        aic        : log(sigma2_hat) + 2 * p / T        for n = 1
    """
    T_eff = len(y) - p_max
    Y = y[p_max:]

    X = np.ones((T_eff, p + 1))
    for lag in range(1, p + 1):
        X[:, lag] = y[p_max - lag : p_max - lag + T_eff]

    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ beta
    sigma2_hat = np.mean(resid**2)  # SSR / T_eff

    bic = np.log(sigma2_hat) + p * np.log(T_eff) / T_eff
    aic = np.log(sigma2_hat) + 2.0 * p / T_eff

    return {
        "beta": beta,
        "sigma2_hat": sigma2_hat,
        "bic": bic,
        "aic": aic,
    }


def run_simulation(
    n_reps: int = N_REPS,
    T: int = T,
    p_max: int = P_MAX,
    burn_in: int = BURN_IN,
    seed: int = SEED,
    nu: float = NU_TRUE,
    rho: np.ndarray = RHO_TRUE,
    sigma2: float = SIGMA2_TRUE,
) -> SimulationResults:
    """Run the full Monte Carlo experiment."""
    rng = np.random.default_rng(seed)

    phat_bic = np.empty(n_reps, dtype=int)
    phat_aic = np.empty(n_reps, dtype=int)
    rho1hat_bic = np.empty(n_reps)
    rho1hat_aic = np.empty(n_reps)

    for r in range(n_reps):
        y = simulate_true_ar2(
            rng=rng,
            T=T,
            p_max=p_max,
            burn_in=burn_in,
            nu=nu,
            rho=rho,
            sigma2=sigma2,
        )

        fits: Dict[int, Dict[str, np.ndarray | float]] = {
            p: fit_ar_common_sample(y=y, p=p, p_max=p_max)
            for p in range(1, p_max + 1)
        }

        p_bic = min(fits, key=lambda p: float(fits[p]["bic"]))
        p_aic = min(fits, key=lambda p: float(fits[p]["aic"]))

        phat_bic[r] = p_bic
        phat_aic[r] = p_aic
        rho1hat_bic[r] = float(fits[p_bic]["beta"][1])
        rho1hat_aic[r] = float(fits[p_aic]["beta"][1])

    return SimulationResults(
        phat_bic=phat_bic,
        phat_aic=phat_aic,
        rho1hat_bic=rho1hat_bic,
        rho1hat_aic=rho1hat_aic,
    )


def plot_density_rho1(results: SimulationResults, save_path: Path | None = None) -> None:
    r"""Replicate slide 17: sampling density of post-selection \hat{rho}_1."""
    grid = np.linspace(0.3, 1.0, 500)
    kde_bic = gaussian_kde(results.rho1hat_bic)
    kde_aic = gaussian_kde(results.rho1hat_aic)

    fig, ax = plt.subplots(figsize=(7.8, 5.7))
    ax.plot(grid, kde_bic(grid), color="black", linewidth=1.8, label="BIC")
    ax.plot(grid, kde_aic(grid), color="red", linewidth=1.8, linestyle=(0, (4, 4)), label="AIC")
    ax.axvline(0.6, color="black", linewidth=0.8, linestyle="--")

    ax.set_title(r"Simulation study: sampling density of $\hat{\rho}_1$", fontsize=16, color="#2b36b4", pad=22)
    ax.set_xlim(0.3, 1.0)
    ax.set_ylim(0.0, 6.0)
    ax.set_xticks(np.arange(0.3, 1.01, 0.1))
    ax.set_yticks(np.arange(0, 6.1, 1.0))
    ax.legend(loc="upper right", frameon=True, framealpha=1.0)
    ax.tick_params(direction="in", top=True, right=True, length=4, width=0.6)

    for spine in ax.spines.values():
        spine.set_linewidth(0.6)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_distribution_phat(results: SimulationResults, save_path: Path | None = None) -> None:
    r"""Replicate slide 18: sampling distribution of selected lag order \hat{p}."""
    p_grid = np.array([1, 2, 3])
    bic_probs = np.array([np.mean(results.phat_bic == p) for p in p_grid])
    aic_probs = np.array([np.mean(results.phat_aic == p) for p in p_grid])

    fig, axes = plt.subplots(1, 2, figsize=(7.8, 5.7), sharey=False)
    fig.suptitle(r"Simulation study: sampling distribution of $\hat{p}$", fontsize=16, color="#2b36b4", y=0.96)

    for ax, probs, title in zip(axes, [bic_probs, aic_probs], ["BIC", "AIC"]):
        ax.bar(p_grid, probs, width=0.95, color="#67a2cc", edgecolor="#4d6a7a", linewidth=0.7)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlim(0.3, 3.7)
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks([1, 2, 3])
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.tick_params(direction="in", top=True, right=True, length=3, width=0.6)
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def print_summary(results: SimulationResults) -> None:
    """Print numerical summaries so you can check the simulation against the slides."""
    bic_probs = {p: np.mean(results.phat_bic == p) for p in [1, 2, 3]}
    aic_probs = {p: np.mean(results.phat_aic == p) for p in [1, 2, 3]}

    print("Monte Carlo summary")
    print("-" * 60)
    print(f"Repetitions: {len(results.phat_bic):,}")
    print(f"Sample size T: {T}")
    print(f"True model: AR(2), rho1 = {RHO_TRUE[0]}, rho2 = {RHO_TRUE[1]}, sigma^2 = {SIGMA2_TRUE}")
    print()
    print("P_hat distribution")
    print("BIC:", bic_probs)
    print("AIC:", aic_probs)
    print()
    print("Post-selection rho1-hat")
    print(f"BIC mean: {np.mean(results.rho1hat_bic):.4f}, std: {np.std(results.rho1hat_bic, ddof=1):.4f}")
    print(f"AIC mean: {np.mean(results.rho1hat_aic):.4f}, std: {np.std(results.rho1hat_aic, ddof=1):.4f}")


def main() -> None:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    results = run_simulation()
    print_summary(results)

    plot_density_rho1(results, save_path=SAVE_DIR / "slide17_sampling_density_rho1.png")
    plot_distribution_phat(results, save_path=SAVE_DIR / "slide18_sampling_distribution_phat.png")


if __name__ == "__main__":
    main()
