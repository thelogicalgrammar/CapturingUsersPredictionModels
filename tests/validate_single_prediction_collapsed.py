"""
Monte Carlo validation of the single-prediction collapsed reduction.

The reduction (the "Single-prediction collapsed reduction" document of the
paper derivations) marginalizes the structural coefficient vectors N_b and
N_d and replaces the training-data plate with two scalar auxiliaries (R, T),
so that conditionally on (R, T, N_E, the population-level parameters, and
X_o) the observed prediction is Gaussian: \n
:math:`Pred^o \\sim \\mathcal{N}(\\mu_{red}, V_{red})` \n
:math:`\\mu_{red} = c + (m_b + N_E m_d) S_o` \n
:math:`V_{red} = Q_o (v_b + N_E^2 v_d) + \\tau^2 (1/n + R/T)`.

This script validates the three distributional claims behind the
implementation in ``single_prediction_collapsed_agent_model.py`` against
brute force, using only NumPy/SciPy sampling (no NUTS):

1. R construction (Cochran): the central-chi-squared construction \n
   :math:`R = (\\Delta_X + G_R/\\sqrt{n})^2 + C_R/n`,
   :math:`G_R \\sim \\mathcal{N}(0,1)`, :math:`C_R \\sim \\chi^2_{k-1}`, \n
   used in the PyMC model must reproduce both the native non-central
   chi-squared :math:`\\tfrac{1}{n}\\chi'^2_k(n\\Delta_X^2)` and the direct
   k-coordinate Gaussian construction :math:`R = \\lVert r \\rVert^2`.

2. Collapsed likelihood: predictions sampled from the collapsed likelihood
   (with the Cochran R) must reproduce the prediction distribution of the
   explicit reference: a full historical plate sampled from the SCM equations
   with random coefficient vectors, plus an OLS regression with intercept.

3. Coefficient reconstruction: the conditional Gaussian (Matheron) draws of
   (N_b, N_d) produced by
   ``SinglePredictionCollapsedAgentModel._sample_conditional_coefficients()``
   - the actual implementation, not a re-derivation - must reproduce the
   brute-force conditional distribution of the projections
   :math:`B_o = X_o^\\top N_b` and :math:`D_o = X_o^\\top N_d` given the
   observed prediction. The brute-force side conditions by ABC windowing of
   explicit-plate simulations; the collapsed side conditions exactly by
   importance-resampling (R, T) prior draws with the collapsed likelihood.

Distributional agreement is measured with the normalized Wasserstein
distance from ``tests.statistical_tests_for_similarity`` (pass/fail) and,
as a secondary indication only, the two-sample Kolmogorov-Smirnov p-value
(its scale depends strongly on the number of repetitions, see the notes in
``statistical_tests_for_similarity``).

Run from the repository root:

.. code-block:: bash

    python -m tests.validate_single_prediction_collapsed

The defaults run in a few minutes on a laptop. Reference cluster-scale runs
(plate_size=1000, repetitions=20000, k up to 100, reconstruction with
200000 plates): all checks passed with maximum normalized Wasserstein
distance 0.024.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp

import single_prediction_collapsed_agent_model as spcam
from tests.statistical_tests_for_similarity import (
    normalized_wasserstein_distance)


DEFAULT_K_VALUES = (1, 2, 5, 10)
DEFAULT_RECONSTRUCTION_K_VALUES = (2, 5)


def parse_args():
    """
    Parse the command line arguments of the validation run.
    """
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--k-values', type=int, nargs='+',
                        default=list(DEFAULT_K_VALUES),
                        help='Covariate counts for checks 1 and 2.')
    parser.add_argument('--reconstruction-k-values', type=int, nargs='+',
                        default=list(DEFAULT_RECONSTRUCTION_K_VALUES),
                        help='Covariate counts for check 3.')
    parser.add_argument('--plate-size', type=int, default=1000,
                        help='Number of historical patients in the plate.')
    parser.add_argument('--repetitions', type=int, default=5000,
                        help='Monte Carlo repetitions for checks 1 and 2.')
    parser.add_argument('--reconstruction-repetitions', type=int,
                        default=50000,
                        help='Explicit plates for the ABC reference of '
                             'check 3.')
    parser.add_argument('--importance-draws', type=int, default=50000,
                        help='(R, T) prior draws for the importance '
                             'resampling of check 3.')
    parser.add_argument('--resample-size', type=int, default=2000,
                        help='Resampled Matheron draws of check 3.')
    parser.add_argument('--abc-window-fraction', type=float, default=0.05,
                        help='ABC window of check 3, as a fraction of the '
                             'predictive standard deviation.')
    parser.add_argument('--seed', type=int, default=20260610,
                        help='Seed for the random number generator.')
    parser.add_argument('--threshold', type=float, default=0.05,
                        help='Normalized-Wasserstein pass threshold for '
                             'checks 1 and 2.')
    parser.add_argument('--reconstruction-threshold', type=float,
                        default=0.08,
                        help='Pass threshold for check 3 (looser because '
                             'the ABC window adds a small smoothing bias).')
    parser.add_argument('--output-dir', type=Path, default=None,
                        help='If given, write a summary CSV and a metadata '
                             'JSON to this directory.')
    return parser.parse_args()


def make_parameters(k):
    """
    Return the fixed population-level parameters used in the validation.

    These match the default experiment configuration: the coefficient means
    are scaled by 1 / k so the aggregate signal stays close to the
    univariate case, and the observed patient is one standard deviation
    above the population mean on every covariate.

    Args:
        k (int): Number of covariates.

    Returns:
        dict: Parameter dictionary used by the samplers below.
    """
    return {
        'alpha_mu_X': np.full(k, 80.0),
        'alpha_sigma_X': np.full(k, 10.0),
        'alpha_mu_A': 2.0,
        'alpha_sigma_A': 1.0,
        'alpha_mu_Y': 12.0,
        'alpha_sigma_Y': 0.1,
        'N_E': 1.0,
        'N_b_mu': np.full(k, -0.1 / k),
        'N_b_sigma': np.full(k, 0.1),
        'N_d_mu': np.full(k, 0.125 / k),
        'N_d_sigma': np.full(k, 0.1),
        'X_o': np.full(k, 90.0),
    }


def compare_samples(reference, candidate, threshold):
    """
    Compare two groups of observations and return the comparison summary.

    The pass/fail decision uses the normalized Wasserstein distance from
    ``statistical_tests_for_similarity``. The Kolmogorov-Smirnov p-value is
    reported as a secondary indication only.

    Args:
        reference (np.ndarray): Observations from the brute-force reference.
        candidate (np.ndarray): Observations from the reduced construction.
        threshold (float): Maximum accepted normalized Wasserstein distance.

    Returns:
        dict: Comparison summary with the distances and the pass flag.
    """
    wasserstein = normalized_wasserstein_distance(
        reference, candidate, max_diff=threshold)
    ks = ks_2samp(reference, candidate)
    return {
        'reference_mean': float(np.mean(reference)),
        'candidate_mean': float(np.mean(candidate)),
        'reference_std': float(np.std(reference)),
        'candidate_std': float(np.std(candidate)),
        'normalized_wasserstein':
            wasserstein['normalized_wasserstein_distance'],
        'ks_pvalue': float(ks.pvalue),
        'passed': wasserstein['success'],
    }


def sample_R_cochran(rng, delta_squared, k, plate_size, size=None):
    """
    Sample R through the central-chi-squared (Cochran) construction.

    This mirrors the construction in the PyMC graph of
    ``SinglePredictionCollapsedAgentModel._build_model()``: \n
    :math:`R = (\\Delta_X + G_R/\\sqrt{n})^2 + C_R/n` \n
    with the convention :math:`\\chi^2_0 = 0` when k = 1.

    Args:
        rng (np.random.Generator): Initialized random number generator.
        delta_squared (float): Aggregate non-centrality Delta_X^2.
        k (int): Number of covariates.
        plate_size (int): Number of historical patients n.
        size (int or None): Number of draws (None for a single scalar).

    Returns:
        float or np.ndarray: Draw(s) of R.
    """
    G_R = rng.normal(size=size)
    C_R = rng.chisquare(df=k - 1, size=size) if k > 1 else 0.0
    return ((np.sqrt(delta_squared) + G_R / np.sqrt(plate_size))**2 +
            C_R / plate_size)


def validate_r_construction(k, args, rng):
    """
    Check 1: the Cochran construction of R against the alternatives.

    Compares the Cochran construction against (a) the direct k-coordinate
    Gaussian construction :math:`R = \\lVert r \\rVert^2` with
    :math:`r \\sim \\mathcal{N}_k(\\delta, n^{-1} I_k)`, and (b) the native
    non-central chi-squared sampler of NumPy.

    Args:
        k (int): Number of covariates.
        args (argparse.Namespace): Validation configuration.
        rng (np.random.Generator): Initialized random number generator.

    Returns:
        dict: Comparison summary (worst case over the two comparisons).
    """
    params = make_parameters(k)
    delta = (params['X_o'] - params['alpha_mu_X']) / params['alpha_sigma_X']
    delta_squared = float(delta @ delta)
    size = args.repetitions

    gaussian_vector = rng.normal(
        delta, 1.0 / np.sqrt(args.plate_size), size=(size, k))
    gaussian_construction = np.sum(gaussian_vector**2, axis=1)
    native = rng.noncentral_chisquare(
        df=k, nonc=args.plate_size * delta_squared,
        size=size) / args.plate_size
    cochran = sample_R_cochran(rng, delta_squared, k, args.plate_size, size)

    versus_gaussian = compare_samples(gaussian_construction, cochran,
                                      args.threshold)
    versus_native = compare_samples(native, cochran, args.threshold)
    worst = max(versus_gaussian, versus_native,
                key=lambda row: row['normalized_wasserstein'])
    return {
        'check': 'r_construction', 'k': k, **worst,
        'passed': versus_gaussian['passed'] and versus_native['passed'],
    }


def fit_ols_with_intercept(x, y):
    """
    Fit an OLS regression with intercept.

    Args:
        x (np.ndarray): Covariate matrix of shape (n, k).
        y (np.ndarray): Outcome vector of length n.

    Returns:
        tuple: ``(intercept, slope_vector)``.
    """
    design = np.column_stack([np.ones(x.shape[0]), x])
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    return float(coefficients[0]), coefficients[1:]


def explicit_prediction_sample(rng, params, plate_size,
                               return_projections=False):
    """
    Draw one prediction from the explicit full-plate reference.

    Samples fresh coefficient vectors from their priors, simulates the full
    historical plate from the SCM equations, fits the OLS model with
    intercept, and evaluates the prediction at the observed patient. This is
    the brute-force counterpart that the collapsed likelihood must match.

    Args:
        rng (np.random.Generator): Initialized random number generator.
        params (dict): Population-level parameters from
            :func:`make_parameters`.
        plate_size (int): Number of historical patients n.
        return_projections (bool): If True, also return the projections
            B_o = X_o . N_b and D_o = X_o . N_d of the sampled coefficient
            vectors (used by the reconstruction check).

    Returns:
        float or tuple: The prediction, or ``(prediction, B_o, D_o)``.
    """
    k = params['alpha_mu_X'].size
    N_b = rng.normal(params['N_b_mu'], params['N_b_sigma'])
    N_d = rng.normal(params['N_d_mu'], params['N_d_sigma'])

    X = rng.normal(params['alpha_mu_X'], params['alpha_sigma_X'],
                   size=(plate_size, k))
    eps_A = rng.normal(0.0, 1.0, size=plate_size)
    eps_Y = rng.normal(0.0, 1.0, size=plate_size)
    A = X @ N_d + params['alpha_mu_A'] + params['alpha_sigma_A'] * eps_A
    Y = (X @ N_b + params['N_E'] * A + params['alpha_mu_Y'] +
         params['alpha_sigma_Y'] * eps_Y)

    intercept, slope = fit_ols_with_intercept(X, Y)
    prediction = float(intercept + params['X_o'] @ slope)
    if return_projections:
        return (prediction, float(params['X_o'] @ N_b),
                float(params['X_o'] @ N_d))
    return prediction


def collapsed_likelihood_terms(params):
    """
    Return the (R, T)-free terms of the collapsed likelihood.

    Args:
        params (dict): Population-level parameters from
            :func:`make_parameters`.

    Returns:
        tuple: ``(mean, coefficient_variance, tau_squared)`` where the mean
        is mu_red, the coefficient variance is the Q_o-term of V_red, and
        tau_squared completes the OLS-noise term of V_red.
    """
    X_o = params['X_o']
    N_E = params['N_E']
    coefficient_mean = X_o @ (params['N_b_mu'] + N_E * params['N_d_mu'])
    coefficient_variance = np.sum((X_o * params['N_b_sigma'])**2)
    coefficient_variance += N_E**2 * np.sum((X_o * params['N_d_sigma'])**2)
    c = params['alpha_mu_Y'] + N_E * params['alpha_mu_A']
    tau_squared = (N_E**2 * params['alpha_sigma_A']**2 +
                   params['alpha_sigma_Y']**2)
    return c + coefficient_mean, coefficient_variance, tau_squared


def collapsed_prediction_sample(rng, params, plate_size):
    """
    Draw one prediction from the collapsed likelihood with the Cochran R.

    Args:
        rng (np.random.Generator): Initialized random number generator.
        params (dict): Population-level parameters from
            :func:`make_parameters`.
        plate_size (int): Number of historical patients n.

    Returns:
        float: One prediction draw.
    """
    k = params['alpha_mu_X'].size
    delta = (params['X_o'] - params['alpha_mu_X']) / params['alpha_sigma_X']
    R = sample_R_cochran(rng, float(delta @ delta), k, plate_size)
    T = rng.chisquare(df=plate_size - k)
    mean, coefficient_variance, tau_squared = collapsed_likelihood_terms(
        params)
    variance = coefficient_variance + tau_squared * (1.0 / plate_size +
                                                     R / T)
    return float(rng.normal(mean, np.sqrt(variance)))


def validate_collapsed_likelihood(k, args, rng):
    """
    Check 2: the collapsed likelihood against the explicit reference.

    Args:
        k (int): Number of covariates.
        args (argparse.Namespace): Validation configuration.
        rng (np.random.Generator): Initialized random number generator.

    Returns:
        dict: Comparison summary.
    """
    params = make_parameters(k)
    reference = np.empty(args.repetitions)
    reduced = np.empty(args.repetitions)
    for repetition in range(args.repetitions):
        reference[repetition] = explicit_prediction_sample(
            rng, params, args.plate_size)
        reduced[repetition] = collapsed_prediction_sample(
            rng, params, args.plate_size)
    return {
        'check': 'collapsed_likelihood', 'k': k,
        **compare_samples(reference, reduced, args.threshold),
    }


def validate_coefficient_reconstruction(k, args, rng):
    """
    Check 3: the conditional coefficient reconstruction of the agent model.

    The conditioning value is the collapsed predictive mean. The reference
    keeps the explicit-plate draws whose prediction falls in a small window
    around the conditioning value (ABC: conditioning by brute force). The
    collapsed side conditions exactly: (R, T) prior draws are resampled with
    weights proportional to the collapsed likelihood at the conditioning
    value, and for each resampled pair the coefficient vectors are drawn
    with ``SinglePredictionCollapsedAgentModel._sample_conditional_coefficients()``
    - so this check exercises the reconstruction code actually used for the
    posterior predictive quantities, not a copy of its equations.

    Args:
        k (int): Number of covariates.
        args (argparse.Namespace): Validation configuration.
        rng (np.random.Generator): Initialized random number generator.

    Returns:
        dict: Comparison summary (worst case over B_o and D_o).
    """
    params = make_parameters(k)
    mean, coefficient_variance, tau_squared = collapsed_likelihood_terms(
        params)
    target = mean

    # Brute-force reference: explicit plates, ABC window on the prediction.
    predictions = np.empty(args.reconstruction_repetitions)
    B_projections = np.empty(args.reconstruction_repetitions)
    D_projections = np.empty(args.reconstruction_repetitions)
    for repetition in range(args.reconstruction_repetitions):
        predictions[repetition], B_projections[repetition], \
            D_projections[repetition] = explicit_prediction_sample(
                rng, params, args.plate_size, return_projections=True)
    window = args.abc_window_fraction * np.std(predictions)
    accepted = np.abs(predictions - target) < window
    reference_B = B_projections[accepted]
    reference_D = D_projections[accepted]
    print('ABC window {:.3}, accepted {} of {} plates'.format(
        window, int(np.sum(accepted)), args.reconstruction_repetitions))

    # Collapsed side: importance resampling of (R, T), then the agent
    # model's own conditional (Matheron) coefficient draws.
    agent = spcam.SinglePredictionCollapsedAgentModel(
        n_covariates=k,
        plate_size=args.plate_size,
        prior_parameters={
            'N_b_mu': params['N_b_mu'], 'N_b_sigma': params['N_b_sigma'],
            'N_d_mu': params['N_d_mu'], 'N_d_sigma': params['N_d_sigma'],
        })
    agent.X_o = params['X_o']
    agent.pred_o = float(target)

    delta = (params['X_o'] - params['alpha_mu_X']) / params['alpha_sigma_X']
    R_draws = sample_R_cochran(rng, float(delta @ delta), k,
                               args.plate_size, args.importance_draws)
    T_draws = rng.chisquare(df=args.plate_size - k,
                            size=args.importance_draws)
    variance = coefficient_variance + tau_squared * (
        1.0 / args.plate_size + R_draws / T_draws)
    log_weights = -0.5 * ((target - mean)**2 / variance + np.log(variance))
    weights = np.exp(log_weights - np.max(log_weights))
    weights /= np.sum(weights)
    resampled = rng.choice(args.importance_draws, size=args.resample_size,
                           replace=True, p=weights)

    X_o = params['X_o']
    reconstructed_B = np.empty(args.resample_size)
    reconstructed_D = np.empty(args.resample_size)
    for i, index in enumerate(resampled):
        parameters = {
            'N_E': params['N_E'],
            'alpha_mu_A': params['alpha_mu_A'],
            'alpha_sigma_A': params['alpha_sigma_A'],
            'alpha_mu_Y': params['alpha_mu_Y'],
            'alpha_sigma_Y': params['alpha_sigma_Y'],
            'R': R_draws[index], 'T': T_draws[index],
        }
        N_b, N_d = agent._sample_conditional_coefficients(parameters, rng)
        reconstructed_B[i] = X_o @ N_b
        reconstructed_D[i] = X_o @ N_d

    comparison_B = compare_samples(reference_B, reconstructed_B,
                                   args.reconstruction_threshold)
    comparison_D = compare_samples(reference_D, reconstructed_D,
                                   args.reconstruction_threshold)
    worst = max(comparison_B, comparison_D,
                key=lambda row: row['normalized_wasserstein'])
    return {
        'check': 'coefficient_reconstruction', 'k': k, **worst,
        'passed': comparison_B['passed'] and comparison_D['passed'],
    }


def main():
    """
    Run all three checks and report the results.

    Exits with a non-zero status if any check fails, so the script can be
    used as a gate in batch runs.
    """
    args = parse_args()
    started_at = time.time()
    rng = np.random.default_rng(args.seed)

    rows = []
    for k in args.k_values:
        print('\n==== Check 1, R construction, k={} ===='.format(k))
        rows.append(validate_r_construction(k, args, rng))
    for k in args.k_values:
        print('\n==== Check 2, collapsed likelihood, k={} ===='.format(k))
        rows.append(validate_collapsed_likelihood(k, args, rng))
    for k in args.reconstruction_k_values:
        print('\n==== Check 3, coefficient reconstruction, k={} ===='.format(k))
        rows.append(validate_coefficient_reconstruction(k, args, rng))

    print('\n==== Summary ====')
    for row in rows:
        print('{:<28} k={:<4} normalized Wasserstein {:.4f}  {}'.format(
            row['check'], row['k'], row['normalized_wasserstein'],
            'passed' if row['passed'] else 'FAILED'))
    all_passed = all(row['passed'] for row in rows)
    max_distance = max(row['normalized_wasserstein'] for row in rows)
    runtime = time.time() - started_at
    print('All passed: {} (max normalized Wasserstein {:.4f}, '
          'runtime {:.1f}s)'.format(all_passed, max_distance, runtime))

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        fieldnames = list(rows[0].keys())
        with (args.output_dir / 'summary.csv').open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        metadata = {key: (str(value) if isinstance(value, Path) else value)
                    for key, value in vars(args).items()}
        metadata.update({'all_passed': all_passed,
                         'max_normalized_wasserstein': max_distance,
                         'runtime_seconds': runtime})
        with (args.output_dir / 'metadata.json').open('w') as file:
            json.dump(metadata, file, indent=2)
        print('Results written to {}'.format(args.output_dir))

    if not all_passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
