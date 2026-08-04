"""
Dimensionality experiment for a multivariate SCM with intercept.

The coefficient-prior scaling is either ``1k`` or ``sqrtk``. Sampler settings
and the selected model parameterization are written into every result row.

Run ``python experiment_dimensionality.py --help`` for the command-line
interface.
"""
import argparse
import hashlib
import os
import time
import uuid
from pathlib import Path

# ArviZ expects its cache directory to exist on compute nodes.
_cache_root = os.environ.get(
    'XDG_CACHE_HOME', os.path.expanduser('~/.cache'))
os.makedirs(os.path.join(_cache_root, 'arviz'), exist_ok=True)

import arviz as az
import numpy as np
from sklearn.linear_model import LinearRegression

import agent_model as am
import scm
import single_prediction_collapsed_agent_model as spcam
import utils
from experiment_randomness import (
    RESULT_SCHEMA_VERSION as INDEPENDENT_RESULT_SCHEMA_VERSION,
    SCHEME as INDEPENDENT_DATA_SEED_SCHEME,
    format_stream_key as format_independent_stream_key,
    run_rng as independent_run_rng,
    run_stream_key as independent_run_stream_key,
)
from dimensionality_config import (
    AGENT_MODEL as DEFAULT_AGENT_MODEL,
    A0_TREATMENT,
    A1_TREATMENT,
    COEFFICIENT_PRIOR_REFERENCE,
    DECISION_THRESHOLD,
    DEFAULT_SCALING,
    DIMENSIONALITY_EXPERIMENT_DESIGN,
    FIT_INTERCEPT,
    FIXED_RNG_SEED,
    N_CHAINS,
    N_DRAWS,
    N_TUNE,
    OBSERVATION_SEED_SCHEME,
    PLATE_SIZE,
    RESULT_SCHEMA_VERSION,
    SCALING_CHOICES,
    SCAN_CENTERS,
    TARGET_ACCEPT,
    TRUE_SCM,
    format_stream_key,
    run_stream_key,
)


plate_size = PLATE_SIZE
draws = N_DRAWS
A1_treatment = A1_TREATMENT
A0_treatment = A0_TREATMENT
threshold = DECISION_THRESHOLD
rng = FIXED_RNG_SEED

AGENT_MODEL = os.environ.get('DIM_MODEL', DEFAULT_AGENT_MODEL)
VALID_AGENT_MODELS = ('normal', 'collapsed_sign_marginalized')
if AGENT_MODEL not in VALID_AGENT_MODELS:
    raise ValueError(
        "DIM_MODEL must be one of {}, not {!r}".format(
            VALID_AGENT_MODELS, AGENT_MODEL))

parameter_descriptions = {
    'agent_priors':         'Default',
    'scm_noise_parameters': 'Default',
    'scm_constants':        'Default',
}

# Vector prior keys that are broadcast directly vs those scaled by 1/k.
_BROADCAST_VECTOR_PARAMS = {'mu_X', 'sigma_X', 'mu_sigma_X', 'sigma_sigma_X'}
_SCALED_VECTOR_PARAMS = {'N_b_mu', 'N_b_sigma', 'N_d_mu', 'N_d_sigma'}

def _make_priors(k, coef_sigma_scaling='1/k'):
    """
    Build agent prior dict for k covariates.

    N_b_mu[i] = -0.1/k and N_d_mu[i] = 0.125/k so that E[X^T N_b] and
    E[X^T N_d] match the scalar case (mu_X = 80). The mean scaling is
    forced: without it the total systematic effect grows with k.

    The sigma scaling is a free design choice, controlled by
    ``coef_sigma_scaling``:

    - ``'1/k'`` (default, original convention): preserves the coefficient
      of variation per dimension. The prior variance of the aggregate
      projections x^T N_b and x^T N_d then decays like 1/k
      (Q_o * (0.1/k)^2 with Q_o ~ 6500 k gives ~65/k), so the posterior
      can tighten with k because the prior itself becomes more concentrated.
    - ``'1/sqrt(k)'``: keeps the prior variance of the aggregate
      projections approximately constant across k for the baseline design.
      By the single-prediction reduction, the coefficients enter the
      likelihood only through those projections. This makes it a
      prior-variance-preserving sensitivity convention for comparing
      dimensionalities; it does not by itself imply equal total information
      or an invariant posterior across k.
    """
    if coef_sigma_scaling == '1/k':
        sigma_divisor = k
    elif coef_sigma_scaling == '1/sqrt(k)':
        sigma_divisor = np.sqrt(k)
    else:
        raise ValueError(
            "coef_sigma_scaling must be '1/k' or '1/sqrt(k)', not {!r}".format(
                coef_sigma_scaling))
    return {
        'mu_X':          np.full(k, TRUE_SCM['alpha_mu_X']),
        'sigma_X':       np.full(k, 0.1),
        'mu_sigma_X':    np.full(k, TRUE_SCM['alpha_sigma_X']),
        'sigma_sigma_X': np.full(k, 0.1),
        'mu_A':          TRUE_SCM['alpha_mu_A'],
        'sigma_A':       1.0,
        'mu_sigma_A':    TRUE_SCM['alpha_sigma_A'],
        'sigma_sigma_A': 0.1,
        'mu_Y':          TRUE_SCM['alpha_mu_Y'],
        'sigma_Y':       0.01,
        'mu_sigma_Y':    TRUE_SCM['alpha_sigma_Y'],
        'sigma_sigma_Y': 0.01,
        'N_E_mu':        TRUE_SCM['N_E'],
        'N_E_sigma':     1.0,
        'N_b_mu': np.full(
            k, COEFFICIENT_PRIOR_REFERENCE['N_b_mu'] / k),
        'N_b_sigma': np.full(
            k, COEFFICIENT_PRIOR_REFERENCE['N_b_sigma'] / sigma_divisor),
        'N_d_mu': np.full(
            k, COEFFICIENT_PRIOR_REFERENCE['N_d_mu'] / k),
        'N_d_sigma': np.full(
            k, COEFFICIENT_PRIOR_REFERENCE['N_d_sigma'] / sigma_divisor),
    }


def _apply_prior_modification(agent_priors, param_name, param_value, k,
                              coef_sigma_scaling='1/k'):
    """
    Apply a single-float CLI modification to agent_priors in-place.

    For broadcast vector params (mu_X, …): sets every dimension to param_value.
    For scaled vector params, param_value is the scalar (k = 1) equivalent:
    means are always divided by k; sigmas are divided by k or sqrt(k)
    according to ``coef_sigma_scaling``, consistently with ``_make_priors``.
    For scalar params: assigns directly.
    """
    if param_name in _BROADCAST_VECTOR_PARAMS:
        agent_priors[param_name] = np.full(k, param_value)
    elif param_name in {'N_b_sigma', 'N_d_sigma'}:
        divisor = k if coef_sigma_scaling == '1/k' else np.sqrt(k)
        agent_priors[param_name] = np.full(k, param_value / divisor)
    elif param_name in _SCALED_VECTOR_PARAMS:
        agent_priors[param_name] = np.full(k, param_value / k)
    else:
        agent_priors[param_name] = param_value


def _prepare_run_randomness(k, rng, observed_x=None, observation_rng=None,
                            data_rng=None):
    """Prepare the observed patient and random streams for one run.

    Args:
        k (int): Number of covariates.
        rng: Seed or NumPy Generator for the observed patient and sampler.
        observed_x (array-like or None): Optional externally supplied
            observation. Its shape must be ``(k,)``.
        observation_rng: Optional independent seed or NumPy Generator for the
            observed patient.
        data_rng: Optional independent seed or NumPy Generator for the
            historical dataset.

    Returns:
        tuple: ``(X_o, sampler_rng, historical_data_rng)``.
    """
    run_rng = np.random.default_rng(rng)
    if observed_x is None:
        patient_rng = (
            run_rng if observation_rng is None
            else np.random.default_rng(observation_rng)
        )
        X_o = patient_rng.normal(
            loc=TRUE_SCM['alpha_mu_X'],
            scale=TRUE_SCM['alpha_sigma_X'],
            size=k)
    else:
        X_o = np.asarray(observed_x, dtype=float)
        if X_o.shape != (k,):
            raise ValueError(
                "observed_x must have shape ({},), not {}".format(
                    k, X_o.shape))
        if not np.all(np.isfinite(X_o)):
            raise ValueError("observed_x must contain only finite values")

    historical_data_rng = (
        run_rng if data_rng is None else np.random.default_rng(data_rng))
    return X_o, run_rng, historical_data_rng


def _array_fingerprint(*arrays):
    """Return a stable SHA-256 fingerprint for typed NumPy arrays."""
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(array.dtype.str.encode('ascii'))
        digest.update(str(array.shape).encode('ascii'))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _publish_without_overwrite(tmp_path, final_path):
    """Atomically publish a staged file while preserving an existing result."""
    tmp_path = Path(tmp_path)
    final_path = Path(final_path)
    try:
        os.link(tmp_path, final_path)
    except FileExistsError:
        raise FileExistsError(
            "refusing to overwrite concurrently published result: {}".format(
                final_path))
    finally:
        tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Main experiment function
# ---------------------------------------------------------------------------

def main(agent_priors, modified_prior, k,
         A1_treatment=A1_treatment, A0_treatment=A0_treatment,
         plate_size=plate_size, draws=draws,
         parameter_descriptions=parameter_descriptions,
         sampler_settings=None, output_dir='./results/dim_runs', rng=rng,
         observed_x=None, observation_rng=None, data_rng=None,
         result_metadata=None):
    """
    Run one experiment trial with k-dimensional X.

    Args:
        agent_priors (dict): Prior parameter dict (already modified if needed).
        modified_prior (dict): ``{'modified_prior': name, 'modified_prior_value': value}``.
        k (int): Number of covariates (dimensionality of X).
        A1_treatment (float): Value of treatment 1.
        A0_treatment (float): Value of treatment 0.
        plate_size (int): Number of historical observations for the ML model.
        draws (int): posterior draw count per chain, also used for the
            predictive and CATE sample summaries.
        parameter_descriptions (dict): Labels written to the result row.
        sampler_settings (dict): ``draws``, ``tune``, ``chains`` and
            ``target_accept`` passed to the sampler.
        output_dir (str or Path): Directory for the per-run CSV file.
        rng: Seed or NumPy Generator for the observed patient, sampler, and
            predictive simulations.
        observed_x (array-like or None): Optional observed feature vector.
            When omitted, it is sampled from the true covariate distribution.
        observation_rng: Optional independent seed or NumPy Generator used
            only to generate ``X_o``.
        data_rng: Optional independent seed or NumPy Generator for the
            historical dataset.
        result_metadata (dict or None): Additional provenance fields written
            to the result row.
    """
    start_time = time.time()
    if sampler_settings is None:
        sampler_settings = {
            'draws': draws,
            'tune': N_TUNE,
            'chains': N_CHAINS,
            'target_accept': TARGET_ACCEPT,
        }
    else:
        sampler_settings = sampler_settings.copy()
    draws = sampler_settings['draws']
    is_collapsed = AGENT_MODEL == 'collapsed_sign_marginalized'
    seed_value = modified_prior.get('seed', 'NA')
    scaling_tag = ('_sqrtk'
                   if modified_prior.get('coef_sigma_scaling') == '1/sqrt(k)'
                   else '')
    run_file = 'results_{}_{:.6f}_k{}_seed{}{}.csv'.format(
        modified_prior['modified_prior'],
        modified_prior['modified_prior_value'],
        k, seed_value, scaling_tag)
    output_dir = Path(output_dir)
    final_path = output_dir / run_file
    if final_path.exists():
        raise FileExistsError(
            "refusing to overwrite existing result: {}".format(final_path))

    # 1. Data-generating SCM with true (fixed) parameters.
    data_scm = scm.MultivariateSCM(
        n_covariates=k,
        alpha_mu_X=np.full(k, TRUE_SCM['alpha_mu_X']),
        alpha_sigma_X=np.full(k, TRUE_SCM['alpha_sigma_X']),
        alpha_mu_A=TRUE_SCM['alpha_mu_A'],
        alpha_sigma_A=TRUE_SCM['alpha_sigma_A'],
        alpha_mu_Y=TRUE_SCM['alpha_mu_Y'],
        alpha_sigma_Y=TRUE_SCM['alpha_sigma_Y'],
        N_E=TRUE_SCM['N_E'],
        N_b=np.full(k, TRUE_SCM['N_b'] / k),
        N_d=np.full(k, TRUE_SCM['N_d'] / k),
    )

    # 2. Prepare the observed patient and historical-data random streams.
    X_o, rng_obj, historical_data_rng = _prepare_run_randomness(
        k, rng, observed_x=observed_x,
        observation_rng=observation_rng, data_rng=data_rng)

    # 3. Generate historical data and train prediction model.
    X, A, Y_historical, *_ = data_scm.sample(
        draws=plate_size, rng=historical_data_rng)
    pred_model = LinearRegression(
        fit_intercept=FIT_INTERCEPT).fit(X, Y_historical)
    pred_o = float(pred_model.predict(X_o.reshape(1, -1))[0])
    historical_data_sha256 = _array_fingerprint(X, A, Y_historical)

    # 4. Construct the selected agent model.
    if is_collapsed:
        agent_model = spcam.SinglePredictionCollapsedAgentModel(
            n_covariates=k,
            plate_size=plate_size,
            prior_parameters=agent_priors,
        )
    else:
        agent_model = am.AgentModelNormalPredictions(
            n_covariates=k,
            plate_size=plate_size,
            prior_parameters=agent_priors,
            fit_intercept=FIT_INTERCEPT,
        )

    # 5. Prior CATE and treatment decision.
    prior_CATE_mu = agent_model.get_prior_CATE_mu(A1_treatment, A0_treatment)
    if is_collapsed:
        prior_CATE_samples_, prior_CATE_mean, prior_CATE_std = \
            agent_model.sample_prior_predictive_CATE(A1_treatment, A0_treatment,
                                                     draws=draws, rng=rng_obj)
    else:
        prior_CATE_samples_, prior_CATE_mean, prior_CATE_std = \
            agent_model.sample_prior_predictive_CATE(X_o, A1_treatment, A0_treatment,
                                                     rng=rng_obj)
    prior_high_treatment = prior_CATE_mu > threshold
    prior_treatment = (
        A1_treatment if prior_high_treatment else A0_treatment)
    prior_predictive_summaries = agent_model.sample_prior_predictive_summaries(rng=rng_obj)

    # 6. Posterior sampling. All settings are explicit and recorded below.
    agent_model.sample(X_o=X_o, pred_o=pred_o, rng=rng_obj,
                       **sampler_settings)

    if is_collapsed:
        trace_diagnostics = agent_model.trace_diagnostics()
    else:
        rhat_values = az.rhat(agent_model.trace)
        ess_bulk_values = az.ess(agent_model.trace, method='bulk')
        ess_tail_values = az.ess(agent_model.trace, method='tail')
        trace_diagnostics = {
            'max_rhat': float(rhat_values.to_array().max().values),
            'min_ess_bulk': float(
                ess_bulk_values.to_array().min().values),
            'min_ess_tail': float(
                ess_tail_values.to_array().min().values),
            'n_divergences': int(
                agent_model.trace.sample_stats['diverging'].sum()),
        }

    # 7. Posterior CATE and treatment decision.
    if is_collapsed:
        posterior_CATE_samples_, posterior_CATE_mean, posterior_CATE_std = \
            agent_model.sample_posterior_predictive_CATE(A1_treatment, A0_treatment)
    else:
        posterior_CATE_samples_, posterior_CATE_mean, posterior_CATE_std = \
            agent_model.sample_posterior_predictive_CATE(X_o, A1_treatment, A0_treatment,
                                                         rng=rng_obj)
    posterior_high_treatment = posterior_CATE_mean > threshold
    posterior_treatment = (
        A1_treatment if posterior_high_treatment else A0_treatment)
    posterior_predictive_summaries  = agent_model.sample_posterior_predictive_summaries(rng=rng_obj)
    if is_collapsed:
        posterior_parameter_summaries = agent_model.posterior_parameter_summaries(rng=rng_obj)
    else:
        posterior_parameter_summaries = agent_model.posterior_parameter_summaries()

    # 8. Ground truth CATE and potential outcomes.
    true_CATE_mu = data_scm.get_CATE_mu(A1_treatment, A0_treatment)
    true_CATE_samples_, true_CATE_mean, true_CATE_std = \
        data_scm.sample_CATE(X_o, A1_treatment, A0_treatment, draws=draws, rng=rng_obj)
    potential_outcome_A1_samples_, potential_outcome_A1_mean, potential_outcome_A1_std = \
        data_scm.sample_do_A(A1_treatment, X_o, draws=draws, rng=rng_obj)
    potential_outcome_A0_samples_, potential_outcome_A0_mean, potential_outcome_A0_std = \
        data_scm.sample_do_A(A0_treatment, X_o, draws=draws, rng=rng_obj)

    # 9. Expected outcomes under prior and posterior beliefs.
    _, expected_prior_outcome_A1_mean, expected_prior_outcome_A1_std = \
        agent_model.sample_prior_predictive_do_A(X_o, A1_treatment, rng=rng_obj)
    _, expected_prior_outcome_A0_mean, expected_prior_outcome_A0_std = \
        agent_model.sample_prior_predictive_do_A(X_o, A0_treatment, rng=rng_obj)
    _, expected_posterior_outcome_A1_mean, expected_posterior_outcome_A1_std = \
        agent_model.sample_posterior_predictive_do_A(X_o, A1_treatment, rng=rng_obj)
    _, expected_posterior_outcome_A0_mean, expected_posterior_outcome_A0_std = \
        agent_model.sample_posterior_predictive_do_A(X_o, A0_treatment, rng=rng_obj)
    expected_outcomes = {
        'expected_prior_outcome_A1_mean':     expected_prior_outcome_A1_mean,
        'expected_prior_outcome_A1_std':      expected_prior_outcome_A1_std,
        'expected_prior_outcome_A0_mean':     expected_prior_outcome_A0_mean,
        'expected_prior_outcome_A0_std':      expected_prior_outcome_A0_std,
        'expected_posterior_outcome_A1_mean': expected_posterior_outcome_A1_mean,
        'expected_posterior_outcome_A1_std':  expected_posterior_outcome_A1_std,
        'expected_posterior_outcome_A0_mean': expected_posterior_outcome_A0_mean,
        'expected_posterior_outcome_A0_std':  expected_posterior_outcome_A0_std,
    }

    # 10. Decision correctness and outcome quality.
    true_high_treatment = true_CATE_mu > threshold
    true_sample_high_treatment = true_CATE_samples_ > threshold
    prior_decision_correct = prior_high_treatment == true_high_treatment
    posterior_decision_correct = \
        posterior_high_treatment == true_high_treatment
    true_best_treatment = (
        A1_treatment if true_high_treatment else A0_treatment)

    if prior_high_treatment != posterior_high_treatment:
        decision_changed = (
            1 if posterior_high_treatment == true_high_treatment else -1)
    else:
        decision_changed = 0

    frac_correct_prior_decision = np.mean(
        true_sample_high_treatment == prior_high_treatment)
    frac_correct_posterior_decision = np.mean(
        true_sample_high_treatment == posterior_high_treatment)
    frac_correct_true_best_treatment = np.mean(
        true_sample_high_treatment == true_high_treatment)

    Y_mean_prior_treatment = (
        potential_outcome_A1_mean
        if prior_high_treatment else potential_outcome_A0_mean)
    Y_mean_posterior_treatment = (
        potential_outcome_A1_mean
        if posterior_high_treatment else potential_outcome_A0_mean)
    Y_mean_true_best_treatment = (
        potential_outcome_A1_mean
        if true_high_treatment else potential_outcome_A0_mean)

    outputs = {
        'pred_o':                            pred_o,
        'threshold':                         threshold,
        'prior_treatment':                   prior_treatment,
        'posterior_treatment':               posterior_treatment,
        'true_best_treatment':               true_best_treatment,
        'prior_decision_correct':            prior_decision_correct,
        'posterior_decision_correct':        posterior_decision_correct,
        'decision_changed':                  decision_changed,
        'frac_correct_prior_decision':       frac_correct_prior_decision,
        'frac_correct_posterior_decision':   frac_correct_posterior_decision,
        'frac_correct_true_best_treatment':  frac_correct_true_best_treatment,
        'Y_mean_prior_treatment':            Y_mean_prior_treatment,
        'Y_mean_posterior_treatment':        Y_mean_posterior_treatment,
        'Y_mean_true_best_treatment':        Y_mean_true_best_treatment,
        'potential_outcome_A1_mean':         potential_outcome_A1_mean,
        'potential_outcome_A1_std':          potential_outcome_A1_std,
        'potential_outcome_A0_mean':         potential_outcome_A0_mean,
        'potential_outcome_A0_std':          potential_outcome_A0_std,
    }

    # 11. Assemble results.
    experiment_id = uuid.uuid4().hex

    # X_o: store as scalar for k=1, as bracketed list for k>1.
    # Individual components are always written as extra columns X_o_0, X_o_1, …
    x_o_scalar = float(X_o[0]) if k == 1 else str(X_o.tolist())

    compiled_results = {
        'experiment_id':       experiment_id,
        'X_o':                 x_o_scalar,
        'true_CATE_mu':        true_CATE_mu,
        'true_CATE_mean':      true_CATE_mean,
        'true_CATE_std':       true_CATE_std,
        'prior_CATE_mu':       prior_CATE_mu,
        'prior_CATE_mean':     prior_CATE_mean,
        'prior_CATE_std':      prior_CATE_std,
        'posterior_CATE_mean': posterior_CATE_mean,
        'posterior_CATE_std':  posterior_CATE_std,
        'plate_size':          plate_size,
        'A1_treatment':        A1_treatment,
        'A0_treatment':        A0_treatment,
        'historical_data_sha256': historical_data_sha256,
        'prediction_model_intercept': float(pred_model.intercept_),
    }
    for i, val in enumerate(X_o):
        compiled_results['X_o_{}'.format(i)] = float(val)
    for i, val in enumerate(np.ravel(pred_model.coef_)):
        compiled_results['prediction_model_coef_{}'.format(i)] = float(val)

    compiled_results.update(outputs)
    compiled_results.update(parameter_descriptions)

    end_time     = time.time()
    current_time = time.strftime("%H:%M:%S %d.%m.%Y")
    compiled_results['time_stamp']   = current_time
    compiled_results['running_time'] = end_time - start_time
    compiled_results['n_covariates'] = k

    compiled_results.update(modified_prior)
    if result_metadata:
        duplicate_metadata = set(result_metadata).intersection(
            compiled_results)
        if duplicate_metadata:
            raise ValueError(
                "result_metadata duplicates result fields: {}".format(
                    sorted(duplicate_metadata)))
        compiled_results.update(result_metadata)
    compiled_results.update(trace_diagnostics)
    compiled_results.update(prior_predictive_summaries)
    compiled_results.update(posterior_predictive_summaries)
    compiled_results.update(posterior_parameter_summaries)
    compiled_results.update(expected_outcomes)

    # One atomic file per configuration avoids mixed per-k schemas. Publishing
    # fails if a result with the same scientific key already exists.
    compiled_results['agent_model'] = AGENT_MODEL
    compiled_results['n_draws'] = sampler_settings['draws']
    compiled_results['n_tune'] = sampler_settings['tune']
    compiled_results['n_chains'] = sampler_settings['chains']
    compiled_results['target_accept'] = sampler_settings['target_accept']
    if is_collapsed:
        compiled_results['posterior_negative_sign_probability'] = \
            agent_model.posterior_negative_sign_probability()
    tmp_file = '{}.tmp.{}.{}'.format(run_file, os.getpid(), uuid.uuid4().hex)
    output_dir.mkdir(parents=True, exist_ok=True)
    utils.write_results_to_file(compiled_results,
                                file_name=tmp_file,
                                path=str(output_dir) + os.sep)
    tmp_path = output_dir / tmp_file
    _publish_without_overwrite(tmp_path, final_path)


def _positive_int(value):
    """Parse a positive integer for argparse."""
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return value


def _nonnegative_int(value):
    """Parse a non-negative integer for argparse."""
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("expected a non-negative integer")
    return value


def _target_accept(value):
    """Parse a valid NUTS target acceptance probability."""
    value = float(value)
    if not 0.0 < value < 1.0:
        raise argparse.ArgumentTypeError("expected a value strictly between 0 and 1")
    return value


def _parse_args():
    """Parse the dimensionality experiment command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('param_name')
    parser.add_argument('param_value', type=float)
    parser.add_argument('n_covariates', type=_positive_int)
    parser.add_argument(
        'seed', nargs='?', type=_nonnegative_int, default=FIXED_RNG_SEED,
        help='replicate identifier; used as the run seed unless '
             '--fixed-rng-seed is supplied')
    parser.add_argument(
        'sigma_scaling', nargs='?', choices=tuple(SCALING_CHOICES),
        default=DEFAULT_SCALING)
    parser.add_argument(
        'legacy_draws', nargs='?', type=_positive_int,
        help=argparse.SUPPRESS)
    parser.add_argument('--draws', type=_positive_int, default=None)
    parser.add_argument('--tune', type=_positive_int, default=N_TUNE)
    parser.add_argument('--chains', type=_positive_int, default=N_CHAINS)
    parser.add_argument(
        '--target-accept', type=_target_accept, default=TARGET_ACCEPT)
    parser.add_argument('--output-dir', default='./results/dim_runs')
    parser.add_argument(
        '--fixed-rng-seed', type=_nonnegative_int,
        help='base seed for historical data, observation and sampler streams')
    parser.add_argument(
        '--experiment-design',
        choices=('univariate_intercept', 'multivariate_intercept'),
        help='independent-data design for a registered scientific row')
    parser.add_argument('--base-seed', type=_nonnegative_int)
    parser.add_argument('--scan-deviation', type=int)
    parser.add_argument(
        '--x-o', type=float,
        help='externally supplied observation; supported only for k=1')
    parser.add_argument(
        '--source-row', type=_nonnegative_int,
        help='source-row identifier for an externally supplied observation')
    args = parser.parse_args()
    if args.draws is not None and args.legacy_draws is not None:
        parser.error("draws was supplied both positionally and with --draws")
    if args.x_o is not None:
        if args.n_covariates != 1:
            parser.error("--x-o is supported only when n_covariates is 1")
        if args.fixed_rng_seed is None or args.source_row is None:
            parser.error(
                "--x-o requires both --fixed-rng-seed and --source-row")
    elif args.source_row is not None:
        parser.error("--source-row requires --x-o")
    if args.experiment_design is not None:
        if args.base_seed is None or args.scan_deviation is None:
            parser.error(
                "--experiment-design requires --base-seed and "
                "--scan-deviation")
        if args.fixed_rng_seed is not None or args.x_o is not None:
            parser.error(
                "--experiment-design cannot be combined with "
                "--fixed-rng-seed or --x-o")
        if (args.experiment_design == 'univariate_intercept' and
                (args.n_covariates != 1 or args.sigma_scaling != '1k')):
            parser.error(
                "univariate_intercept requires n_covariates=1 and "
                "sigma_scaling=1k")
        if (args.experiment_design == 'multivariate_intercept' and
                args.sigma_scaling != 'sqrtk'):
            parser.error("multivariate_intercept requires sqrtk scaling")
    elif args.base_seed is not None or args.scan_deviation is not None:
        parser.error(
            "--base-seed and --scan-deviation require --experiment-design")
    args.draws = args.draws or args.legacy_draws or N_DRAWS
    return args


if __name__ == '__main__':
    args = _parse_args()
    param_name = args.param_name
    param_value = args.param_value
    k = args.n_covariates
    rng = args.seed
    coef_sigma_scaling = SCALING_CHOICES[args.sigma_scaling]

    agent_priors = _make_priors(k, coef_sigma_scaling=coef_sigma_scaling)

    if param_name not in agent_priors:
        raise ValueError(
            "'{}' is not a valid prior key. Valid keys: {}".format(
                param_name, sorted(agent_priors.keys()))
        )

    default_value = agent_priors[param_name]
    _apply_prior_modification(agent_priors, param_name, param_value, k,
                              coef_sigma_scaling=coef_sigma_scaling)

    # Compare element-wise for vector params, directly for scalars.
    is_default = np.all(agent_priors[param_name] == default_value)
    parameter_descriptions['agent_priors'] = 'Default' if is_default else 'Modified'

    modified_prior = {
        'modified_prior': param_name,
        'modified_prior_value': param_value,
        'seed': rng,
        'coef_sigma_scaling': coef_sigma_scaling,
    }
    observed_x = None
    result_metadata = None
    observation_rng = None
    data_rng = None
    run_rng = np.random.default_rng(args.seed)
    if args.experiment_design is not None:
        center, step = SCAN_CENTERS[param_name]
        expected_value = center + args.scan_deviation * step
        if not np.isclose(
                param_value, expected_value, rtol=0.0, atol=1e-7):
            raise ValueError(
                '{} value {} does not match scan deviation {:+d}'.format(
                    param_name, param_value, args.scan_deviation))
        key_args = (
            args.experiment_design,
            param_name,
            args.scan_deviation,
            args.n_covariates,
            args.seed,
            args.sigma_scaling,
        )
        data_key = independent_run_stream_key(
            'historical_data', *key_args)
        observation_key = independent_run_stream_key(
            'observation', *key_args)
        sampler_key = independent_run_stream_key('sampler', *key_args)
        data_rng = independent_run_rng(
            args.base_seed, 'historical_data', *key_args)
        observation_rng = independent_run_rng(
            args.base_seed, 'observation', *key_args)
        run_rng = independent_run_rng(
            args.base_seed, 'sampler', *key_args)
        result_metadata = {
            'result_schema_version': INDEPENDENT_RESULT_SCHEMA_VERSION,
            'experiment_design': args.experiment_design,
            'experimental_unit': 'historical_dataset_and_patient',
            'replicate_id': args.seed,
            'patient_id': format_independent_stream_key(observation_key),
            'data_seed': args.base_seed,
            'historical_data_stream_key':
                format_independent_stream_key(data_key),
            'sampler_seed': args.base_seed,
            'sampler_stream_key':
                format_independent_stream_key(sampler_key),
            'observation_seed_scheme': INDEPENDENT_DATA_SEED_SCHEME,
            'fit_intercept': FIT_INTERCEPT,
        }
    elif args.fixed_rng_seed is not None:
        data_rng = np.random.default_rng(args.fixed_rng_seed)
        if args.x_o is None:
            sampler_key = run_stream_key(
                'sampler', param_name, param_value, args.n_covariates,
                args.seed, args.sigma_scaling)
        else:
            sampler_key = (2, args.n_covariates, args.source_row)
        sampler_sequence = np.random.SeedSequence(
            args.fixed_rng_seed, spawn_key=sampler_key)
        run_rng = np.random.default_rng(sampler_sequence)
        result_metadata = {
            'result_schema_version': RESULT_SCHEMA_VERSION,
            'replicate_id': args.seed,
            'data_seed': args.fixed_rng_seed,
            'sampler_seed': args.fixed_rng_seed,
            'sampler_stream_key': format_stream_key(sampler_key),
            'fit_intercept': FIT_INTERCEPT,
            'experimental_unit': 'patient_observation',
        }
        if args.x_o is None:
            observation_key = run_stream_key(
                'observation', param_name, param_value, args.n_covariates,
                args.seed, args.sigma_scaling)
            observation_seed = np.random.SeedSequence(
                args.fixed_rng_seed, spawn_key=observation_key)
            observation_rng = np.random.default_rng(observation_seed)
            result_metadata.update({
                'experiment_design': DIMENSIONALITY_EXPERIMENT_DESIGN,
                'observation_seed_scheme': OBSERVATION_SEED_SCHEME,
                'patient_id': format_stream_key(observation_key),
            })
        else:
            observed_x = np.asarray([args.x_o])
            result_metadata.update({
                'experiment_design': 'univariate_intercept_wide',
                'observation_seed_scheme': 'archived_observation',
                'patient_id': args.source_row,
                'source_row': args.source_row,
            })

    print("Running experiment: param={}, value={}, k={}".format(
        param_name, param_value, k))

    sampler_settings = {
        'draws': args.draws,
        'tune': args.tune,
        'chains': args.chains,
        'target_accept': args.target_accept,
    }
    main(agent_priors, modified_prior, k,
         parameter_descriptions=parameter_descriptions,
         draws=args.draws,
         sampler_settings=sampler_settings,
         output_dir=args.output_dir,
         rng=run_rng,
         observed_x=observed_x,
         observation_rng=observation_rng,
         data_rng=data_rng,
         result_metadata=result_metadata)
