"""
Sampling-based verification of an alternative condensed formula for phi.

The current low-dimensional model in agent_model.py represents phi using
eight auxiliary random variables:
    S_X, S_Y, S_A, Z_XX, U_XY, V_XY, U_XA, V_XA

This script compares that construction to a proposed condensed formula that
uses only:
    S_X, Z_XX, g

No PyMC inference is run here. We only sample the auxiliary variables from
their known distributions, compute phi deterministically, and compare the
resulting empirical distributions.
"""
from dataclasses import dataclass
import math

import numpy as np
from scipy.stats import ks_2samp


N_SAMPLES = 100_000
KS_PVALUE_THRESHOLD = 1e-3
QUANTILE_LEVELS = (0.05, 0.25, 0.50, 0.75, 0.95)


@dataclass(frozen=True)
class ParameterSetting:
    """
    Parameters needed to evaluate phi under both formulas.
    """
    name: str
    alpha_mu_X: float
    alpha_sigma_X: float
    alpha_mu_A: float
    alpha_sigma_A: float
    alpha_mu_Y: float
    alpha_sigma_Y: float
    N_E: float
    a: float
    b: float
    d: float
    plate_size: int = 1000


def sample_phi_with_eight_variables(setting, n_samples, rng):
    """
    Sample phi using the current eight-variable formula in agent_model.py.
    """
    n = setting.plate_size
    S_X = rng.normal(loc=0.0, scale=math.sqrt(n), size=n_samples)
    S_Y = rng.normal(loc=0.0, scale=math.sqrt(n), size=n_samples)
    S_A = rng.normal(loc=0.0, scale=math.sqrt(n), size=n_samples)
    Z_XX = rng.chisquare(df=n - 1, size=n_samples)
    U_XY = rng.chisquare(df=n - 1, size=n_samples)
    V_XY = rng.chisquare(df=n - 1, size=n_samples)
    U_XA = rng.chisquare(df=n - 1, size=n_samples)
    V_XA = rng.chisquare(df=n - 1, size=n_samples)

    S_9 = S_X * S_A / n + (U_XA - V_XA) / 2.0
    S_8 = S_X * S_Y / n + (U_XY - V_XY) / 2.0
    S_7 = Z_XX + S_X**2 / n
    S_6 = (
        n * setting.alpha_mu_X * setting.alpha_mu_Y
        + setting.alpha_mu_X * setting.alpha_sigma_Y * S_Y
        + setting.alpha_mu_Y * setting.alpha_sigma_X * S_X
        + setting.alpha_sigma_X * setting.alpha_sigma_Y * S_8
    )
    S_5 = setting.N_E * (
        n * setting.alpha_mu_X * setting.alpha_mu_A
        + setting.alpha_mu_X * setting.alpha_sigma_A * S_A
        + setting.alpha_mu_A * setting.alpha_sigma_X * S_X
        + setting.alpha_sigma_X * setting.alpha_sigma_A * S_9
    )
    S_4 = setting.a * (n * setting.alpha_mu_X + setting.alpha_sigma_X * S_X)
    S_123 = (
        n * setting.alpha_mu_X**2
        + 2.0 * setting.alpha_mu_X * setting.alpha_sigma_X * S_X
        + setting.alpha_sigma_X**2 * S_7
    )
    phi = setting.b + setting.d * setting.N_E + (S_4 + S_5 + S_6) / S_123
    return phi, S_X, Z_XX, S_123


def sample_phi_with_condensed_formula(setting, S_X, Z_XX, n_samples, rng):
    """
    Sample phi using the proposed three-variable condensed formula.
    """
    n = setting.plate_size
    g = rng.normal(loc=0.0, scale=1.0, size=n_samples)

    beta = setting.b + setting.N_E * setting.d
    c = setting.a + setting.N_E * setting.alpha_mu_A + setting.alpha_mu_Y
    tau_sq = setting.N_E**2 * setting.alpha_sigma_A**2 + setting.alpha_sigma_Y**2
    tau = math.sqrt(tau_sq)

    S_7 = Z_XX + S_X**2 / n
    S_123 = (
        n * setting.alpha_mu_X**2
        + 2.0 * setting.alpha_mu_X * setting.alpha_sigma_X * S_X
        + setting.alpha_sigma_X**2 * S_7
    )
    phi = (
        beta
        + c * (n * setting.alpha_mu_X + setting.alpha_sigma_X * S_X) / S_123
        + tau * g / np.sqrt(S_123)
    )
    return phi


def summarize(values):
    """
    Return mean, std, and selected quantiles.
    """
    quantiles = np.quantile(values, QUANTILE_LEVELS)
    return {
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'quantiles': {level: float(value) for level, value in zip(QUANTILE_LEVELS, quantiles)},
    }


def print_summary_comparison(eight_summary, condensed_summary):
    """
    Print side-by-side summary statistics.
    """
    print("  mean:", f"{eight_summary['mean']:.10f}", "|", f"{condensed_summary['mean']:.10f}")
    print("  std: ", f"{eight_summary['std']:.10f}", "|", f"{condensed_summary['std']:.10f}")
    print("  quantiles:")
    for level in QUANTILE_LEVELS:
        print(
            f"    q{int(level * 100):02d}:",
            f"{eight_summary['quantiles'][level]:.10f}",
            "|",
            f"{condensed_summary['quantiles'][level]:.10f}",
        )


def run_setting(setting, n_samples=N_SAMPLES, seed=4):
    """
    Run one comparison setting and print diagnostics.
    """
    print(f"=== Setting: {setting.name} ===")
    print(setting)
    rng_eight = np.random.default_rng(seed)
    rng_condensed = np.random.default_rng(seed + 1)

    phi_eight, S_X, Z_XX, _ = sample_phi_with_eight_variables(setting, n_samples, rng_eight)
    phi_condensed = sample_phi_with_condensed_formula(
        setting, S_X, Z_XX, n_samples, rng_condensed
    )

    eight_summary = summarize(phi_eight)
    condensed_summary = summarize(phi_condensed)
    print_summary_comparison(eight_summary, condensed_summary)

    ks_stat, ks_pvalue = ks_2samp(phi_eight, phi_condensed)
    print(f"  KS statistic: {ks_stat:.10f}")
    print(f"  KS p-value:   {ks_pvalue:.10f}")

    verdict = "PASS" if ks_pvalue > KS_PVALUE_THRESHOLD else "FAIL"
    print(f"  verdict: {verdict}")
    print()
    return {
        'setting': setting.name,
        'ks_stat': float(ks_stat),
        'ks_pvalue': float(ks_pvalue),
        'verdict': verdict,
        'eight_summary': eight_summary,
        'condensed_summary': condensed_summary,
    }


def main():
    """
    Compare the empirical phi distributions across several parameter settings.
    """
    settings = [
        ParameterSetting(
            name='codebase_defaults',
            alpha_mu_X=80.0,
            alpha_sigma_X=10.0,
            alpha_mu_A=2.0,
            alpha_sigma_A=1.0,
            alpha_mu_Y=0.0,
            alpha_sigma_Y=0.1,
            N_E=1.0,
            a=12.0,
            b=-0.1,
            d=0.125,
        ),
        ParameterSetting(
            name='custom_NE_0_5',
            alpha_mu_X=80.0,
            alpha_sigma_X=10.0,
            alpha_mu_A=5.0,
            alpha_sigma_A=2.0,
            alpha_mu_Y=3.0,
            alpha_sigma_Y=1.0,
            N_E=0.5,
            a=1.0,
            b=0.3,
            d=0.2,
        ),
        ParameterSetting(
            name='custom_NE_0_0',
            alpha_mu_X=80.0,
            alpha_sigma_X=10.0,
            alpha_mu_A=5.0,
            alpha_sigma_A=2.0,
            alpha_mu_Y=3.0,
            alpha_sigma_Y=1.0,
            N_E=0.0,
            a=1.0,
            b=0.3,
            d=0.2,
        ),
        ParameterSetting(
            name='custom_NE_2_0',
            alpha_mu_X=80.0,
            alpha_sigma_X=10.0,
            alpha_mu_A=5.0,
            alpha_sigma_A=2.0,
            alpha_mu_Y=3.0,
            alpha_sigma_Y=1.0,
            N_E=2.0,
            a=1.0,
            b=0.3,
            d=0.2,
        ),
    ]

    results = [run_setting(setting) for setting in settings]
    all_pass = all(result['verdict'] == 'PASS' for result in results)
    print("=== Overall verdict ===")
    print("PASS" if all_pass else "FAIL")
    if not all_pass:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
