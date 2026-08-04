"""
Shared configuration for the dimensionality experiment.

The manifest generators, experiment driver, and plotting utilities use these
constants to describe the same data-generating SCM and parameter grid.
"""
import math


TRUE_SCM = {
    'alpha_mu_X': 80.0,
    'alpha_sigma_X': 10.0,
    'alpha_mu_A': 2.0,
    'alpha_sigma_A': 1.0,
    'alpha_mu_Y': 12.0,
    'alpha_sigma_Y': 0.1,
    'N_E': 1.0,
    'N_b': -0.1,
    'N_d': 0.125,
}

COEFFICIENT_PRIOR_REFERENCE = {
    'N_b_mu': -0.1,
    'N_b_sigma': 0.1,
    'N_d_mu': 0.125,
    'N_d_sigma': 0.1,
}

SCAN_CENTERS = {
    'mu_A': (TRUE_SCM['alpha_mu_A'], TRUE_SCM['alpha_sigma_A']),
    'mu_X': (TRUE_SCM['alpha_mu_X'], TRUE_SCM['alpha_sigma_X']),
    'mu_Y': (TRUE_SCM['alpha_mu_Y'], TRUE_SCM['alpha_sigma_Y']),
    'N_E_mu': (TRUE_SCM['N_E'], 0.05),
}

SCAN_DEVIATIONS = (-10, -3, -1, 0, 1, 3, 10)
EXPLORATORY_DEVIATIONS = (-30, -25, -20, 20, 25, 30)
N_COVARIATES = (1, 2, 4, 8, 16, 32)
N_REPLICATES = 50
WIDE_RESULTS_SHA256 = (
    'af83e70b7b92ba1d4b00eb6876b76ac007c52bc125fb321d1a40d7de4cb902df')

# Default sampling settings.
N_DRAWS = 1000
N_TUNE = 1000
N_CHAINS = 4
TARGET_ACCEPT = 0.85
AGENT_MODEL = 'collapsed_sign_marginalized'
PLATE_SIZE = 1000
A1_TREATMENT = 20
A0_TREATMENT = 10
DECISION_THRESHOLD = 5
FIT_INTERCEPT = True

# Default base random seed.
FIXED_RNG_SEED = 4
DIMENSIONALITY_EXPERIMENT_DESIGN = \
    'fixed_historical_data_iid_patients_dimensionality'
DIMENSIONALITY_PAIRING = 'independent_patient_per_manifest_row'
OBSERVATION_SEED_SCHEME = 'seedsequence_semantic_run_v2'
RUN_STREAM_VERSION = 2
RESULT_SCHEMA_VERSION = 4

SCALING_CHOICES = {
    '1k': '1/k',
    'sqrtk': '1/sqrt(k)',
}
DEFAULT_SCALING = '1k'

_PARAMETER_CODES = {
    'mu_A': 0,
    'mu_X': 1,
    'mu_Y': 2,
    'N_E_mu': 3,
}
_SCALING_CODES = {
    '1k': 0,
    'sqrtk': 1,
}
_STREAM_DOMAIN_CODES = {
    'observation': 1,
    'sampler': 2,
}


def scan_deviation(param_name, param_value):
    """Return the registered integer offset of a scanned prior value."""
    if param_name not in SCAN_CENTERS:
        raise ValueError('unknown scan parameter: {!r}'.format(param_name))
    center, step = SCAN_CENTERS[param_name]
    raw_deviation = (float(param_value) - center) / step
    deviation = int(round(raw_deviation))
    if not math.isclose(
            raw_deviation, deviation, rel_tol=0.0, abs_tol=1e-7):
        raise ValueError(
            '{} value {} is off the registered scan grid'.format(
                param_name, param_value))
    return deviation


def run_stream_key(
        domain, param_name, param_value, n_covariates, replicate, scaling):
    """Return an order-independent ``SeedSequence.spawn_key`` for one run."""
    if domain not in _STREAM_DOMAIN_CODES:
        raise ValueError('unknown random-stream domain: {!r}'.format(domain))
    if scaling not in _SCALING_CODES:
        raise ValueError('unknown coefficient scaling: {!r}'.format(scaling))
    if param_name not in _PARAMETER_CODES:
        raise ValueError('unknown scan parameter: {!r}'.format(param_name))
    if not isinstance(n_covariates, int) or n_covariates <= 0:
        raise ValueError('n_covariates must be a positive integer')
    if not isinstance(replicate, int) or replicate < 0:
        raise ValueError('replicate must be a non-negative integer')
    deviation = scan_deviation(param_name, param_value)
    encoded_deviation = (
        2 * deviation if deviation >= 0 else -2 * deviation - 1)
    return (
        RUN_STREAM_VERSION,
        _STREAM_DOMAIN_CODES[domain],
        _SCALING_CODES[scaling],
        _PARAMETER_CODES[param_name],
        encoded_deviation,
        n_covariates,
        replicate,
    )


def format_stream_key(stream_key):
    """Return the stable text representation stored with each result."""
    return ':'.join(str(value) for value in stream_key)
