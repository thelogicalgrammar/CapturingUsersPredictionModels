"""Random streams for experiments with independent historical datasets."""
import numpy as np


BASE_SEED = 6
SCHEME = 'seedsequence_semantic_run_v1'
STREAM_VERSION = 1
RESULT_SCHEMA_VERSION = 10

EXPERIMENT_CODES = {
    'univariate_no_intercept': 1,
    'univariate_intercept': 2,
    'multivariate_intercept': 3,
}
DOMAIN_CODES = {
    'historical_data': 1,
    'observation': 2,
    'sampler': 3,
}
PARAMETER_CODES = {
    'mu_A': 0,
    'mu_X': 1,
    'mu_Y': 2,
    'N_E_mu': 3,
}
SCALING_CODES = {
    '1k': 0,
    'sqrtk': 1,
}
NO_INTERCEPT_SCAN_CENTERS = {
    'mu_A': (2.0, 1.0),
    'mu_X': (80.0, 10.0),
    'mu_Y': (0.0, 0.1),
    'N_E_mu': (1.0, 0.05),
}


def run_stream_key(
        domain, experiment_design, param_name, deviation, n_covariates,
        replicate, scaling):
    """Return an order-independent stream key for one scientific row."""
    try:
        domain_code = DOMAIN_CODES[domain]
        experiment_code = EXPERIMENT_CODES[experiment_design]
        parameter_code = PARAMETER_CODES[param_name]
        scaling_code = SCALING_CODES[scaling]
    except KeyError as exc:
        raise ValueError('unknown stream-key value: {!r}'.format(
            exc.args[0])) from None
    if not isinstance(deviation, int):
        raise ValueError('deviation must be an integer')
    if not isinstance(n_covariates, int) or n_covariates <= 0:
        raise ValueError('n_covariates must be a positive integer')
    if not isinstance(replicate, int) or replicate < 0:
        raise ValueError('replicate must be a non-negative integer')
    encoded_deviation = (
        2 * deviation if deviation >= 0 else -2 * deviation - 1)
    return (
        STREAM_VERSION,
        domain_code,
        experiment_code,
        scaling_code,
        parameter_code,
        encoded_deviation,
        n_covariates,
        replicate,
    )


def run_rng(base_seed, *args, **kwargs):
    """Return the generator identified by a scientific row and domain."""
    key = run_stream_key(*args, **kwargs)
    sequence = np.random.SeedSequence(base_seed, spawn_key=key)
    return np.random.default_rng(sequence)


def format_stream_key(stream_key):
    """Return the stable text representation stored with each result."""
    return ':'.join(str(value) for value in stream_key)
