"""
Dimensionality experiment: runs AgentModelNormalPredictions with k-dimensional X.

CLI usage::

    python experiment_dimensionality.py <param_name> <param_value> <n_covariates>

``param_name`` is a key in ``agent_priors``; ``param_value`` is the new value
(scalar equivalent for vector-valued priors; see _apply_prior_modification).
``n_covariates`` sets the dimensionality k of X.

Example::

    python experiment_dimensionality.py N_E_mu 2.0 3
    python experiment_dimensionality.py N_b_mu -0.3 2

X_o is always sampled internally from the correct covariate distribution.
"""
import os
import sys
import time

# ArviZ tries to write ~/.cache/arviz/ on import; the directory may not
# exist on cluster nodes where the home directory is freshly mounted.
os.makedirs(os.path.expanduser('~/.cache/arviz'), exist_ok=True)

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

import scm
import utils
import agent_model as am


# ---------------------------------------------------------------------------
# Module-level defaults (identical treatment / threshold / draws as experiment.py)
# ---------------------------------------------------------------------------
plate_size  = 1000
draws       = 1000
A1_treatment = 20
A0_treatment = 10
threshold    = 5
rng          = 4   # Selected by fair die

parameter_descriptions = {
    'agent_priors':         'Default',
    'scm_noise_parameters': 'Default',
    'scm_constants':        'Default',
}

# Scalar equivalents (k = 1 reference values) used for CV-preserving scaling.
_NB_MU_1D   = -0.1
_NB_SIG_1D  =  0.1
_ND_MU_1D   =  0.125
_ND_SIG_1D  =  0.1

# Vector prior keys that are broadcast directly vs those scaled by 1/k.
_BROADCAST_VECTOR_PARAMS = {'mu_X', 'sigma_X', 'mu_sigma_X', 'sigma_sigma_X'}
_SCALED_VECTOR_PARAMS    = {'N_b_mu', 'N_b_sigma', 'N_d_mu', 'N_d_sigma'}
_VECTOR_PARAMS           = _BROADCAST_VECTOR_PARAMS | _SCALED_VECTOR_PARAMS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_priors(k):
    """
    Build agent prior dict for k covariates.

    N_b_mu[i] = -0.1/k and N_d_mu[i] = 0.125/k so that E[X^T N_b] and
    E[X^T N_d] match the scalar case (mu_X = 80).  N_b_sigma and N_d_sigma
    are scaled the same way to preserve the coefficient of variation per
    dimension.
    """
    return {
        'mu_X':          np.full(k, 80.0),
        'sigma_X':       np.full(k, 0.1),
        'mu_sigma_X':    np.full(k, 10.0),
        'sigma_sigma_X': np.full(k, 0.1),
        'mu_A':          2.0,
        'sigma_A':       1.0,
        'mu_sigma_A':    1.0,
        'sigma_sigma_A': 0.1,
        'mu_Y':          12.0,
        'sigma_Y':       0.01,
        'mu_sigma_Y':    0.1,
        'sigma_sigma_Y': 0.01,
        'N_E_mu':        1.0,
        'N_E_sigma':     1.0,
        'N_b_mu':        np.full(k, _NB_MU_1D  / k),
        'N_b_sigma':     np.full(k, _NB_SIG_1D / k),
        'N_d_mu':        np.full(k, _ND_MU_1D  / k),
        'N_d_sigma':     np.full(k, _ND_SIG_1D / k),
    }


def _apply_prior_modification(agent_priors, param_name, param_value, k):
    """
    Apply a single-float CLI modification to agent_priors in-place.

    For broadcast vector params (mu_X, …): sets every dimension to param_value.
    For scaled vector params (N_b_mu, …): interprets param_value as the scalar
    equivalent and divides by k per dimension (preserves E[X^T param] and CV).
    For scalar params: assigns directly.

    Returns the modified dict and a bool indicating whether anything changed.
    """
    if param_name in _BROADCAST_VECTOR_PARAMS:
        agent_priors[param_name] = np.full(k, param_value)
    elif param_name in _SCALED_VECTOR_PARAMS:
        agent_priors[param_name] = np.full(k, param_value / k)
    else:
        agent_priors[param_name] = param_value


# ---------------------------------------------------------------------------
# Main experiment function
# ---------------------------------------------------------------------------

def main(agent_priors, modified_prior, k,
         A1_treatment=A1_treatment, A0_treatment=A0_treatment,
         plate_size=plate_size, draws=draws,
         parameter_descriptions=parameter_descriptions,
         rng=rng):
    """
    Run one experiment trial with k-dimensional X.

    Args:
        agent_priors (dict): Prior parameter dict (already modified if needed).
        modified_prior (dict): ``{'modified_prior': name, 'modified_prior_value': value}``.
        k (int): Number of covariates (dimensionality of X).
        A1_treatment (float): Value of treatment 1.
        A0_treatment (float): Value of treatment 0.
        plate_size (int): Number of historical observations for the ML model.
        draws (int): MCMC draw count (tuning uses the same number).
        parameter_descriptions (dict): Labels written to the result row.
        rng: Seed or NumPy Generator for reproducibility.
    """
    start_time = time.time()

    # 1. Data-generating SCM with true (fixed) parameters.
    data_scm = scm.MultivariateSCM(
        n_covariates=k,
        alpha_mu_X=np.full(k, 80.0),
        alpha_sigma_X=np.full(k, 10.0),
        alpha_mu_A=2.0,
        alpha_sigma_A=1.0,
        alpha_mu_Y=12.0,
        alpha_sigma_Y=0.1,
        N_E=1.0,
        N_b=np.full(k, _NB_MU_1D / k),
        N_d=np.full(k, _ND_MU_1D / k),
    )

    # 2. Sample the observation X_o internally from the true covariate dist.
    rng_obj = np.random.default_rng(rng) if isinstance(rng, int) else rng
    X_o = rng_obj.normal(loc=80.0, scale=10.0, size=k)

    # 3. Generate historical data and train prediction model.
    X, A, Y_historical, *_ = data_scm.sample(draws=plate_size, rng=rng_obj)
    pred_model = LinearRegression(fit_intercept=True).fit(X, Y_historical)
    pred_o = float(pred_model.predict(X_o.reshape(1, -1))[0])

    # 4. Construct the agent model.
    agent_model = am.AgentModelNormalPredictions(
        n_covariates=k,
        plate_size=plate_size,
        prior_parameters=agent_priors,
        fit_intercept=True,
    )

    # 5. Prior CATE and treatment decision.
    prior_CATE_mu = agent_model.get_prior_CATE_mu(A1_treatment, A0_treatment)
    prior_CATE_samples_, prior_CATE_mean, prior_CATE_std = \
        agent_model.sample_prior_predictive_CATE(X_o, A1_treatment, A0_treatment)
    prior_treatment = A1_treatment if prior_CATE_mean > threshold else A0_treatment
    prior_predictive_summaries = agent_model.sample_prior_predictive_summaries(rng=rng_obj)

    # 6. Posterior sampling.
    agent_model.sample(X_o=X_o, pred_o=pred_o, draws=draws, rng=rng_obj)

    rhat_values     = az.rhat(agent_model.trace)
    ess_bulk_values = az.ess(agent_model.trace, method='bulk')
    ess_tail_values = az.ess(agent_model.trace, method='tail')
    trace_diagnostics = {
        'max_rhat':      float(rhat_values.to_array().max().values),
        'min_ess_bulk':  float(ess_bulk_values.to_array().min().values),
        'min_ess_tail':  float(ess_tail_values.to_array().min().values),
        'n_divergences': int(agent_model.trace.sample_stats['diverging'].sum()),
    }

    # 7. Posterior CATE and treatment decision.
    posterior_CATE_samples_, posterior_CATE_mean, posterior_CATE_std = \
        agent_model.sample_posterior_predictive_CATE(X_o, A1_treatment, A0_treatment,
                                                     rng=rng_obj)
    posterior_treatment = A1_treatment if posterior_CATE_mean > threshold else A0_treatment
    posterior_predictive_summaries  = agent_model.sample_posterior_predictive_summaries(rng=rng_obj)
    posterior_parameter_summaries   = agent_model.posterior_parameter_summaries()

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
    prior_decision_correct     = np.sign(true_CATE_mean - threshold) == np.sign(prior_CATE_mean - threshold)
    posterior_decision_correct = np.sign(true_CATE_mean - threshold) == np.sign(posterior_CATE_mean - threshold)
    true_best_treatment        = A1_treatment if np.sign(true_CATE_mean - threshold) > 0 else A0_treatment

    if np.sign(prior_CATE_mean - threshold) != np.sign(posterior_CATE_mean - threshold):
        decision_changed = 1 if np.sign(posterior_CATE_mean - threshold) == np.sign(true_CATE_mean - threshold) else -1
    else:
        decision_changed = 0

    frac_correct_prior_decision     = np.mean(np.sign(true_CATE_samples_ - threshold) == np.sign(prior_CATE_mean - threshold))
    frac_correct_posterior_decision = np.mean(np.sign(true_CATE_samples_ - threshold) == np.sign(posterior_CATE_mean - threshold))
    frac_correct_true_best_treatment = np.mean(np.sign(true_CATE_samples_ - threshold) == np.sign(true_CATE_mean - threshold))

    Y_mean_prior_treatment     = potential_outcome_A1_mean if np.sign(prior_CATE_mean - threshold) > 0 else potential_outcome_A0_mean
    Y_mean_posterior_treatment = potential_outcome_A1_mean if np.sign(posterior_CATE_mean - threshold) > 0 else potential_outcome_A0_mean
    Y_mean_true_best_treatment = potential_outcome_A1_mean if np.sign(true_CATE_mean - threshold) > 0 else potential_outcome_A0_mean

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
    current_time  = time.strftime("%Y%m%d%H%M%S")
    experiment_id = current_time + str(np.random.randint(1000, 9999))

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
    }
    for i, val in enumerate(X_o):
        compiled_results['X_o_{}'.format(i)] = float(val)

    compiled_results.update(outputs)
    compiled_results.update(parameter_descriptions)

    end_time     = time.time()
    current_time = time.strftime("%H:%M:%S %d.%m.%Y")
    compiled_results['time_stamp']   = current_time
    compiled_results['running_time'] = end_time - start_time
    compiled_results['n_covariates'] = k

    compiled_results.update(modified_prior)
    compiled_results.update(trace_diagnostics)
    compiled_results.update(prior_predictive_summaries)
    compiled_results.update(posterior_predictive_summaries)
    compiled_results.update(posterior_parameter_summaries)
    compiled_results.update(expected_outcomes)

    utils.write_results_to_file(compiled_results,
                                file_name='results_dimensionality.csv')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    try:
        param_name  = sys.argv[1]
        param_value = float(sys.argv[2])
        k           = int(sys.argv[3])
    except (IndexError, ValueError):
        raise ValueError(
            "Usage: python experiment_dimensionality.py <param_name> <param_value> <n_covariates>\n"
            "Example: python experiment_dimensionality.py N_E_mu 2.0 3"
        )

    agent_priors = _make_priors(k)

    if param_name not in agent_priors:
        raise ValueError(
            "'{}' is not a valid prior key. Valid keys: {}".format(
                param_name, sorted(agent_priors.keys()))
        )

    default_value = agent_priors[param_name]
    _apply_prior_modification(agent_priors, param_name, param_value, k)

    # Compare element-wise for vector params, directly for scalars.
    is_default = np.all(agent_priors[param_name] == default_value)
    parameter_descriptions['agent_priors'] = 'Default' if is_default else 'Modified'

    modified_prior = {
        'modified_prior':       param_name,
        'modified_prior_value': param_value,
    }

    print("Running experiment: param={}, value={}, k={}".format(
        param_name, param_value, k))

    rng_obj = np.random.default_rng(rng)
    main(agent_priors, modified_prior, k,
         parameter_descriptions=parameter_descriptions,
         rng=rng_obj)
