#!/usr/bin/env python3
"""
ECON 31730 – Problem Set 1, Questions 3 & 4
Bayesian VAR with lag selection via BIC/AIC and marginal likelihood.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import multigammaln
from scipy.stats import invwishart, gaussian_kde
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12,
    'figure.figsize': (10, 6), 'figure.dpi': 150
})

OUTPUT_DIR = "/Users/dhruvkohli/Desktop/UChicago_Class/Masters/Spring 2026/Advanced Time Series Analysis"


# ---------- Data ----------

def load_and_prepare_data():
    """Pull GDP, deflator, and fed funds from FRED; return VAR-ready array."""
    try:
        from pandas_datareader import data as pdr
        gdp   = pdr.DataReader('GDPC1',    'fred', '1954-01-01', '2020-12-31')
        defl  = pdr.DataReader('GDPDEF',   'fred', '1954-01-01', '2020-12-31')
        ffr_m = pdr.DataReader('FEDFUNDS', 'fred', '1954-01-01', '2020-12-31')
    except Exception:
        from fredapi import Fred
        fred = Fred()
        gdp   = fred.get_series('GDPC1',    observation_start='1954-01-01', observation_end='2020-12-31').to_frame('GDPC1')
        defl  = fred.get_series('GDPDEF',   observation_start='1954-01-01', observation_end='2020-12-31').to_frame('GDPDEF')
        ffr_m = fred.get_series('FEDFUNDS', observation_start='1954-01-01', observation_end='2020-12-31').to_frame('FEDFUNDS')

    # quarterly fed funds
    ffr_q = ffr_m.resample('QS').mean()
    ffr_q.columns = ['FEDFUNDS']

    # annualised quarter-on-quarter log growth
    gdp_g  = (400 * np.log(gdp).diff()).rename(columns={gdp.columns[0]: 'RGDP_growth'})
    defl_g = (400 * np.log(defl).diff()).rename(columns={defl.columns[0]: 'GDPDEF_infl'})

    df = gdp_g.join(defl_g, how='inner').join(ffr_q, how='inner').dropna()
    df['r_spread'] = df['FEDFUNDS'] - df['GDPDEF_infl']
    df = df.loc['1954-07-01':'2019-12-31']

    Y = df[['RGDP_growth', 'GDPDEF_infl', 'r_spread']].values
    print(f"Sample: {df.index[0]:%Y-Q}{(df.index[0].month-1)//3+1} to "
          f"{df.index[-1]:%Y-Q}{(df.index[-1].month-1)//3+1},  T = {len(Y)}")
    return Y, df.index, df


def var_matrices(Y, p, p_max=None):
    """Set up the y (T x n) and x (T x k) matrices for VAR(p).
    We condition on the first p_max observations so that the effective
    sample size stays the same across different lag orders."""
    if p_max is None:
        p_max = p
    n = Y.shape[1]
    T = Y.shape[0] - p_max
    k = n * p + 1
    y = Y[p_max:]
    x = np.ones((T, k))
    for s in range(1, p + 1):
        x[:, 1+(s-1)*n : 1+s*n] = Y[p_max-s : Y.shape[0]-s]
    return y, x, T, n, k


# ---------- Q3: BIC / AIC ----------

def question3(Y):
    pmax, n = 20, Y.shape[1]
    bic, aic = np.full(pmax, np.nan), np.full(pmax, np.nan)

    for p in range(1, pmax + 1):
        y, x, T, _, k = var_matrices(Y, p, p_max=pmax)
        B = np.linalg.solve(x.T @ x, x.T @ y)
        e = y - x @ B
        ld = np.log(np.linalg.det(e.T @ e / T))
        bic[p-1] = ld + n**2 * p * np.log(T) / T
        aic[p-1] = ld + n**2 * p * 2 / T

    p_bic, p_aic = bic.argmin() + 1, aic.argmin() + 1
    print(f"\n--- Q3: Lag selection (T = {Y.shape[0] - pmax}) ---")
    print(f"  BIC -> p = {p_bic},  AIC -> p = {p_aic}")

    # plot
    ps = np.arange(1, pmax + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for ax, vals, name, best, col in [
        (ax1, bic, 'BIC', p_bic, 'steelblue'),
        (ax2, aic, 'AIC', p_aic, 'darkorange'),
    ]:
        ax.plot(ps, vals, 'o-', color=col)
        ax.axvline(best, ls='--', color='red', label=f'p* = {best}')
        ax.set(xlabel='Lag order p', ylabel=f'{name}(p)', title=name)
        ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/q3_bic_aic.pdf", bbox_inches='tight')
    plt.close()
    print("  -> q3_bic_aic.pdf")
    return p_bic, p_aic


# ---------- Q4(i): Marginal likelihood ----------

def prior_Omega(n, p):
    """Diagonal prior covariance Omega for the VAR coefficients.
    Intercept gets variance 100; lag-s coefficients get 100/s^2."""
    k = n * p + 1
    diag = np.zeros(k)
    diag[0] = 100.0
    for s in range(1, p + 1):
        diag[1+(s-1)*n : 1+s*n] = 100.0 / s**2
    return np.diag(diag)


def log_marginal_likelihood(Y, p, p_max=20):
    """Log marginal likelihood using the GLP (2015) formula (appendix A.9)."""
    n = Y.shape[1]
    d = n + 2
    Psi = (d - n - 1) * 0.02**2 * np.eye(n)

    if p == 0:
        T = Y.shape[0] - p_max
        y, x = Y[p_max:], np.ones((T, 1))
        k = 1
        Om = np.diag([100.0])
    else:
        y, x, T, _, k = var_matrices(Y, p, p_max=p_max)
        Om = prior_Omega(n, p)

    Om_inv = np.diag(1.0 / np.diag(Om))
    XtX = x.T @ x

    # posterior mean of B and residuals
    B = np.linalg.solve(XtX + Om_inv, x.T @ y)
    e = y - x @ B
    S = e.T @ e + B.T @ Om_inv @ B

    # five terms of eq. A.9
    t1 = -n * T / 2 * np.log(np.pi)
    t2 = multigammaln((T + d) / 2, n) - multigammaln(d / 2, n)
    t3 = -T / 2 * np.sum(np.log(np.diag(Psi)))

    # t4: log det of (D_Om' X'X D_Om + I)
    d_om = np.sqrt(np.diag(Om))
    eig4 = np.linalg.eigvalsh(np.diag(d_om) @ XtX @ np.diag(d_om))
    t4 = -n / 2 * np.sum(np.log(1 + eig4))

    # t5: log det of (D_Psi' S D_Psi + I)
    d_psi = 1.0 / np.sqrt(np.diag(Psi))
    eig5 = np.linalg.eigvalsh(np.diag(d_psi) @ S @ np.diag(d_psi))
    t5 = -(T + d) / 2 * np.sum(np.log(1 + eig5))

    return t1 + t2 + t3 + t4 + t5


def question4i(Y):
    pmax = 20
    lml = np.array([log_marginal_likelihood(Y, p, p_max=pmax) for p in range(pmax + 1)])
    p_star = lml.argmax()

    print(f"\n--- Q4(i): Marginal likelihood (T = {Y.shape[0] - pmax}) ---")
    print(f"  Best p = {p_star}")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(pmax + 1), lml, 'o-', color='darkgreen', ms=5)
    ax.axvline(p_star, ls='--', color='red', label=f'p* = {p_star}')
    ax.set(xlabel='Lag order p', ylabel='Log marginal likelihood',
           title='Log Marginal Likelihood vs Lag Order')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/q4i_logml.pdf", bbox_inches='tight')
    plt.close()
    print("  -> q4i_logml.pdf")
    return p_star


# ---------- Q4(ii): Predictive density for 2020-Q2 ----------

def question4ii(Y, p_star, p_max=20):
    n, d = 3, 5
    Psi = (d - n - 1) * 0.02**2 * np.eye(n)

    y, x, T, _, k = var_matrices(Y, p_star, p_max=p_max)
    Om = prior_Omega(n, p_star)
    Om_inv = np.diag(1.0 / np.diag(Om))
    XtX = x.T @ x

    # posterior parameters
    Om_post = np.linalg.inv(XtX + Om_inv)
    B_post  = Om_post @ (x.T @ y)
    e = y - x @ B_post
    Psi_post = Psi + e.T @ e + B_post.T @ Om_inv @ B_post
    d_post = T + d

    L_Om = np.linalg.cholesky(Om_post)
    tail = Y[-p_star:]   # last p observations for building the forecast

    n_draws = 20_000
    gdp_pred = np.zeros(n_draws)
    rng = np.random.default_rng(42)

    for i in range(n_draws):
        # draw Sigma ~ IW, then B | Sigma ~ MatrixNormal
        Sig = invwishart.rvs(df=d_post, scale=Psi_post, random_state=rng)
        L_Sig = np.linalg.cholesky(Sig)
        B_draw = B_post + L_Om @ rng.standard_normal((k, n)) @ L_Sig.T

        # iterate forward two quarters (2020-Q1, 2020-Q2)
        hist = list(tail.copy())
        for _ in range(2):
            xnew = np.ones(k)
            for s in range(1, p_star + 1):
                xnew[1+(s-1)*n : 1+s*n] = hist[-s]
            ynew = B_draw.T @ xnew + rng.multivariate_normal(np.zeros(n), Sig)
            hist.append(ynew)

        gdp_pred[i] = hist[-1][0]

    # plot
    fig, ax = plt.subplots(figsize=(10, 5))
    kde = gaussian_kde(gdp_pred, bw_method='silverman')
    grid = np.linspace(np.percentile(gdp_pred, 0.5), np.percentile(gdp_pred, 99.5), 500)
    ax.fill_between(grid, kde(grid), alpha=0.3, color='steelblue')
    ax.plot(grid, kde(grid), color='steelblue', lw=2)
    ax.axvline(np.median(gdp_pred), ls='--', color='red', lw=1.5,
               label=f'Median = {np.median(gdp_pred):.2f}')
    ax.set(xlabel='Annualised real GDP growth (%)',
           ylabel='Density',
           title=f'Predictive Density of Real GDP Growth, 2020-Q2 (p = {p_star})')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/q4ii_pred_density.pdf", bbox_inches='tight')
    plt.close()

    print(f"\n--- Q4(ii): Predictive density for 2020-Q2 (p = {p_star}) ---")
    print(f"  Mean = {gdp_pred.mean():.2f},  Median = {np.median(gdp_pred):.2f},  "
          f"Std = {gdp_pred.std():.2f}")
    print(f"  90% interval: [{np.percentile(gdp_pred, 5):.2f}, {np.percentile(gdp_pred, 95):.2f}]")
    print("  -> q4ii_pred_density.pdf")


# ---------- Run ----------

if __name__ == '__main__':
    Y, dates, df = load_and_prepare_data()
    p_bic, p_aic = question3(Y)
    p_star = question4i(Y)
    question4ii(Y, p_star)
    print("\nDone.")
