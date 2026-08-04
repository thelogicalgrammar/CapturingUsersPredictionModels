"""
Smoke tests for the multivariate model.
"""

import warnings

import arviz as az
import numpy as np
import pymc as pm
import pytest
from sklearn.linear_model import LinearRegression

import agent_model as am
import experiment
import scm


def synthetic_posterior():
    return {
        'alpha_mu_X': np.array([[[80.0, 80.0]]]),
        'alpha_sigma_X': np.array([[[10.0, 10.0]]]),
        'alpha_mu_A': np.array([[2.0]]),
        'alpha_sigma_A': np.array([[1.0]]),
        'alpha_mu_Y': np.array([[12.0]]),
        'alpha_sigma_Y': np.array([[0.1]]),
        'N_E': np.array([[1.0]]),
        'N_b': np.array([[[-0.05, -0.05]]]),
        'N_d': np.array([[[0.0625, 0.0625]]]),
    }


def test_multivariate_scm_shapes():
    model = scm.MultivariateSCM(
        n_covariates=2,
        alpha_mu_X=np.array([80.0, 120.0]),
        alpha_sigma_X=np.array([10.0, 15.0]),
        N_b=np.array([-0.05, -0.025]),
        N_d=np.array([0.0625, 0.03]),
    )
    X, A, Y, mu_Y_cond, std_Y_cond = model.sample(draws=5, rng=4)

    assert X.shape == (5, 2)
    assert A.shape == (5,)
    assert Y.shape == (5,)
    assert mu_Y_cond.shape == (5,)
    assert std_Y_cond.shape == (5,)

    X_o = np.array([80.0, 120.0])
    Y_do, Y_do_mean, Y_do_std = model.sample_do_A(20, X_o, draws=5, rng=4)
    CATE_samples, CATE_mean, CATE_std = model.sample_CATE(
        X_o, 20, 10, draws=5, rng=4)

    assert Y_do.shape == (5,)
    assert CATE_samples.shape == (5,)
    assert np.isfinite(Y_do_mean)
    assert np.isfinite(Y_do_std)
    assert np.isfinite(CATE_mean)
    assert np.all(CATE_samples == 10.0)
    assert CATE_std == 0


def test_single_covariate_observation_uses_requested_draws():
    models = (
        scm.SCM_WithConstants(),
        scm.MultivariateSCM(n_covariates=1),
    )
    X_o = np.array([80.0])

    for model in models:
        Y_do, _, Y_do_std = model.sample_do_A(
            20, X_o, draws=5, rng=4)
        scalar_Y_do, _, _ = model.sample_do_A(
            20, 80.0, draws=5, rng=4)
        CATE_samples, _, CATE_std = model.sample_CATE(
            X_o, 20, 10, draws=5, rng=4)
        scalar_CATE_samples, _, _ = model.sample_CATE(
            80.0, 20, 10, draws=5, rng=4)

        assert Y_do.shape == (5,)
        assert CATE_samples.shape == (5,)
        assert np.array_equal(Y_do, scalar_Y_do)
        assert np.array_equal(CATE_samples, scalar_CATE_samples)
        assert Y_do_std > 0
        assert np.all(CATE_samples == 10.0)
        assert CATE_std == 0


def test_single_covariate_batch_keeps_batch_shape():
    model = scm.MultivariateSCM(n_covariates=1)

    Y_do, _, _ = model.sample_do_A(
        20, np.array([[80.0]]), draws=5, rng=4)

    assert Y_do.shape == (1,)


def test_scalar_experiment_propagates_treatment_scale_uncertainty():
    model = am.ScalarAgentModel(
        plate_size=8,
        predictive_var_names=list(experiment.scm_noise_parameters))
    model.sample_prior_predictive(draws=3, rng=4)
    parameter_draws = model._parameter_draws(model.prior_samples.prior)

    assert 'alpha_sigma_A' in model.prior_samples.prior
    assert np.isclose(
        model.predictive_scm.alpha_sigma_A,
        parameter_draws[-1]['alpha_sigma_A'])


def test_trace_diagnostics_skip_only_exactly_constant_components():
    rng = np.random.default_rng(4)
    draws = 40
    trace = az.from_dict(
        posterior={
            'theta': rng.normal(size=(4, draws)),
            'constant': np.ones((4, draws)),
        },
        sample_stats={
            'diverging': np.zeros((4, draws), dtype=bool),
        })
    model = object.__new__(am.ScalarAgentModel)
    model.trace = trace

    diagnostics = model.trace_diagnostics()

    assert np.isfinite(diagnostics['max_rhat'])
    assert np.isfinite(diagnostics['min_ess_bulk'])
    assert np.isfinite(diagnostics['min_ess_tail'])
    assert diagnostics['n_divergences'] == 0


def test_trace_diagnostics_reject_undefined_nonconstant_components():
    rng = np.random.default_rng(4)
    values = rng.normal(size=(4, 40))
    values[0, 0] = np.nan
    trace = az.from_dict(
        posterior={'theta': values},
        sample_stats={
            'diverging': np.zeros((4, 40), dtype=bool),
        })
    model = object.__new__(am.ScalarAgentModel)
    model.trace = trace

    with pytest.raises(
            ValueError, match="Undefined diagnostic for non-constant"):
        model.trace_diagnostics()


def test_multivariate_agent_prior_predictive_with_intercept():
    model = am.MultivariateAgentModel(
        n_covariates=2,
        plate_size=8,
        fit_intercept=True)
    samples = model.sample_prior_predictive(draws=3, rng=4)

    assert len(samples) == 3
    assert samples[0][0].shape == (1, 2)


def test_multivariate_agent_prior_predictive_without_intercept():
    model = am.MultivariateAgentModel(
        n_covariates=2,
        plate_size=8,
        fit_intercept=False)
    samples = model.sample_prior_predictive(draws=3, rng=4)

    assert len(samples) == 3
    assert samples[0][0].shape == (1, 2)


def test_scalar_no_intercept_reduction_uses_exact_prediction_law():
    model = am.ScalarAgentModel(plate_size=8)

    old_auxiliaries = {
        'S_X', 'S_Y', 'S_A', 'Z_XX',
        'U_XY', 'V_XY', 'U_XA', 'V_XA',
    }
    assert {'s', 'W'}.issubset(model.model.named_vars)
    assert {'g', 'phi'}.isdisjoint(model.model.named_vars)
    assert old_auxiliaries.isdisjoint(model.model.named_vars)

    with model.model:
        pm.set_data({
            'X_o_': np.asarray(80.0),
            'pred_o_': np.asarray(0.0),
        })
        prior = pm.sample_prior_predictive(
            draws=20, random_seed=4)
    samples = prior.prior

    n = model.plate_size
    S_x = (
        n * samples['alpha_mu_X'] +
        samples['alpha_sigma_X'] * samples['s'])
    XTX = (
        S_x**2 / n +
        samples['alpha_sigma_X']**2 * samples['W'])
    beta = (
        model.scm_constants['b'] +
        model.scm_constants['d'] * samples['N_E'])
    c = (
        model.scm_constants['a'] +
        samples['alpha_mu_Y'] +
        samples['N_E'] * samples['alpha_mu_A'])
    tau = np.sqrt(
        samples['N_E']**2 * samples['alpha_sigma_A']**2 +
        samples['alpha_sigma_Y']**2)
    expected_phi_mean = beta + c * S_x / XTX
    expected_prediction_mu = 80.0 * expected_phi_mean
    expected_prediction_variance = tau**2 * 80.0**2 / XTX

    assert np.allclose(samples['S_x'], S_x)
    assert np.allclose(samples['XTX'], XTX)
    assert np.allclose(samples['phi_mean'], expected_phi_mean)
    assert np.allclose(samples['prediction_mu'], expected_prediction_mu)
    assert np.allclose(
        samples['prediction_variance'], expected_prediction_variance)


def test_scalar_no_intercept_prediction_variance_matches_explicit_plate():
    n = 8
    rng = np.random.default_rng(4)
    alpha_mu_X = 80.0
    alpha_sigma_X = 10.0
    alpha_mu_A = 2.0
    alpha_sigma_A = 1.0
    alpha_mu_Y = 0.0
    alpha_sigma_Y = 0.1
    N_E = 1.2
    a = 12.0
    b = -0.1
    d = 0.125

    epsilon_X = rng.normal(size=n)
    X = alpha_mu_X + alpha_sigma_X * epsilon_X
    tau = np.sqrt(
        N_E**2 * alpha_sigma_A**2 + alpha_sigma_Y**2)
    S_x = X.sum()
    XTX = np.dot(X, X)
    beta = b + d * N_E
    c = a + alpha_mu_Y + N_E * alpha_mu_A
    X_o = 75.0

    explicit_mean = X_o * (beta + c * S_x / XTX)
    residual_weights = X_o * X / XTX
    explicit_variance = tau**2 * np.dot(residual_weights, residual_weights)
    reduced_variance = tau**2 * X_o**2 / XTX

    assert np.allclose(explicit_mean, X_o * (beta + c * S_x / XTX))
    assert np.allclose(reduced_variance, explicit_variance)


def test_scalar_prediction_likelihood_uses_exact_training_variance():
    model = am.ScalarAgentModel(plate_size=8)
    with model.model:
        pm.set_data({
            'X_o_': np.asarray(40.0),
            'pred_o_': np.asarray(0.0),
        })
        point = model.model.initial_point()
        values = model.model.replace_rvs_by_values([
            model.model['prediction_mu'],
            model.model['prediction_variance'],
        ])
        moment_fn = model.model.compile_fn(values)
        logp_fn = model.model.compile_logp(
            vars=[model.model['prediction_obs']])
    observed = []
    for X_o in (40.0, 120.0):
        with model.model:
            pm.set_data({
                'X_o_': np.asarray(X_o),
                'pred_o_': np.asarray(0.0),
            })
        prediction_mu, prediction_variance = moment_fn(point)
        prediction_mu = float(prediction_mu)
        prediction_variance = float(prediction_variance)
        prediction_sigma = np.sqrt(prediction_variance)
        pred_o = prediction_mu + 2.0 * prediction_sigma
        with model.model:
            pm.set_data({'pred_o_': np.asarray(pred_o)})
        observed.append((float(logp_fn(point)), prediction_variance))

    expected = [
        -0.5 * 2.0**2 - 0.5 * np.log(2 * np.pi * variance)
        for _, variance in observed
    ]
    assert np.allclose(
        [logp for logp, _ in observed], expected, rtol=0, atol=1e-10)
    assert np.isclose(observed[1][1] / observed[0][1], 9.0)


@pytest.mark.parametrize('fit_intercept', [False, True])
def test_multivariate_prediction_likelihood_integrates_ols_noise(
        fit_intercept):
    model = am.MultivariateAgentModel(
        n_covariates=2,
        plate_size=8,
        fit_intercept=fit_intercept)

    assert {'g', 'h', 'phi', 'alpha'}.isdisjoint(model.model.named_vars)
    assert {'prediction_mu', 'prediction_variance'}.issubset(
        model.model.named_vars)

    with model.model:
        pm.set_data({
            'X_o_': np.asarray([80.0, 120.0]),
            'pred_o_': np.asarray(0.0),
        })
        point = model.model.initial_point()
        values = model.model.replace_rvs_by_values([
            model.model['prediction_mu'],
            model.model['prediction_variance'],
        ])
        moment_fn = model.model.compile_fn(values)
        logp_fn = model.model.compile_logp(
            vars=[model.model['prediction_obs']])
    prediction_mu, prediction_variance = moment_fn(point)
    prediction_mu = float(prediction_mu)
    prediction_variance = float(prediction_variance)
    assert prediction_variance > 0

    pred_o = prediction_mu - np.sqrt(prediction_variance)
    with model.model:
        pm.set_data({'pred_o_': np.asarray(pred_o)})
    expected = -0.5 - 0.5 * np.log(2 * np.pi * prediction_variance)
    assert np.isclose(float(logp_fn(point)), expected)


def test_scalar_prediction_likelihood_rejects_degenerate_observations():
    model = am.ScalarAgentModel(plate_size=8)

    with pytest.raises(ValueError, match="finite nonzero scalar"):
        model.sample(X_o=np.asarray(0.0), pred_o=np.asarray(1.0), draws=1)
    with pytest.raises(ValueError, match="one finite scalar prediction"):
        model.sample(
            X_o=np.asarray(80.0), pred_o=np.asarray([1.0]), draws=1)


def test_single_prediction_likelihood_keeps_prediction_scalar():
    data_scm = scm.MultivariateSCM(n_covariates=2)
    X, A, Y, *_ = data_scm.sample(draws=12, rng=4)
    pred_model = LinearRegression(fit_intercept=True).fit(X, Y)
    X_o = np.array([80.0, 120.0])
    pred_o = float(pred_model.predict(X_o.reshape(1, -1))[0])

    model = am.MultivariateAgentModel(
        n_covariates=2,
        plate_size=12,
        fit_intercept=True)
    pm.set_data({'X_o_': X_o, 'pred_o_': np.asarray(pred_o)},
                model=model.model)
    with model.model:
        logp = model.model.compile_logp()
        value = logp(model.model.initial_point())

    assert np.isfinite(value)


def test_posterior_predictive_do_A_extracts_scalar_outcome():
    model = am.MultivariateAgentModel(
        n_covariates=2,
        plate_size=8,
        fit_intercept=True)
    model.trace = az.from_dict(posterior=synthetic_posterior())

    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=DeprecationWarning)
        samples, mean, std = model.sample_posterior_predictive_do_A(
            np.array([80.0, 80.0]), 20, rng=4)

    assert samples.shape == (1,)
    assert np.isfinite(samples[0])
    assert np.isfinite(mean)
    assert np.isfinite(std)


def test_multivariate_agent_model_save_and_load(tmp_path):
    model = am.MultivariateAgentModel(
        n_covariates=2,
        plate_size=8,
        fit_intercept=True)
    model.trace = az.from_dict(posterior=synthetic_posterior())
    model.posterior_draws = 1
    model.X_o = np.array([80.0, 80.0])
    model.sample_prior_predictive(draws=1, rng=4)

    model.save_agent_model(file_name='saved_model', folder=tmp_path)
    loaded_model = am.MultivariateAgentModel.load_agent_model(
        file_name='saved_model',
        folder=tmp_path)

    assert loaded_model.n_covariates == model.n_covariates
    assert loaded_model.plate_size == model.plate_size
    assert loaded_model.fit_intercept == model.fit_intercept
    assert loaded_model.posterior_draws == model.posterior_draws
    assert loaded_model.prior_draws == model.prior_draws
    assert np.array_equal(loaded_model.X_o, model.X_o)
    assert loaded_model.model is not None
    assert loaded_model.trace is not None
    assert loaded_model.prior_samples is not None
