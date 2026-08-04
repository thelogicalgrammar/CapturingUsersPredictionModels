"""Tests for the collapsed no-intercept agent model."""

import numpy as np
import pymc as pm
import pytest

import single_prediction_no_intercept_agent_model as spniam


def test_no_intercept_likelihood_is_finite_and_sign_marginalized():
    model = spniam.SinglePredictionNoInterceptAgentModel(plate_size=12)

    pm.set_data(
        {'X_o_': np.asarray([80.0]), 'pred_o_': np.asarray(10.0)},
        model=model.model)
    with model.model:
        point = model.model.initial_point()
        logp = model.model.compile_logp()(point)
        gradient = model.model.compile_dlogp()(point)

    free_variables = {variable.name for variable in model.model.free_RVs}
    assert free_variables == {
        'alpha_mu_X', 'alpha_sigma_X', 'alpha_sigma_A',
        'alpha_sigma_Y', 'N_E_magnitude', 's', 'W',
    }
    assert 'N_E' not in model.model.named_vars
    assert 'N_E_magnitude_log_exp_m1__' in point
    assert np.isfinite(logp)
    assert np.all(np.isfinite(gradient))


def test_no_intercept_prediction_moments_match_reduced_ols_law():
    model = spniam.SinglePredictionNoInterceptAgentModel(plate_size=12)
    X_o = 75.0

    pm.set_data(
        {'X_o_': np.asarray([X_o]), 'pred_o_': np.asarray(10.0)},
        model=model.model)
    with model.model:
        point = model.model.initial_point()
        variables = model.model.replace_rvs_by_values([
            model.model[name] for name in (
                'alpha_mu_X', 'alpha_sigma_X', 'alpha_sigma_A',
                'alpha_sigma_Y', 'N_E_magnitude', 's', 'W',
                'prediction_mu', 'prediction_variance',
            )
        ])
        values = model.model.compile_fn(variables)(point)

    (alpha_mu_X, alpha_sigma_X, alpha_sigma_A, alpha_sigma_Y,
     magnitude, s, W, prediction_mu, prediction_variance) = values
    alpha_mu_X = float(alpha_mu_X[0])
    alpha_sigma_X = float(alpha_sigma_X[0])
    S_x = model.plate_size * alpha_mu_X + alpha_sigma_X * float(s)
    XTX = S_x**2 / model.plate_size + alpha_sigma_X**2 * float(W)
    mean_factor = S_x / XTX
    signs = np.asarray([1.0, -1.0])
    signed_effects = signs * float(magnitude)
    expected_mu = X_o * (
        model.scm_constants['b']
        + signed_effects * model.scm_constants['d']
        + mean_factor * (
            model.scm_constants['a']
            + model.prior_parameters['mu_Y']
            + signed_effects * model.prior_parameters['mu_A']))
    expected_variance = (
        (X_o * mean_factor)**2
        * (model.prior_parameters['sigma_Y']**2
           + float(magnitude)**2
           * model.prior_parameters['sigma_A']**2)
        + (float(magnitude)**2 * float(alpha_sigma_A)**2
           + float(alpha_sigma_Y)**2) * X_o**2 / XTX)

    assert np.allclose(prediction_mu, expected_mu)
    assert np.isclose(float(prediction_variance), expected_variance)


def test_no_intercept_conditional_reconstruction_matches_matheron_rule():
    model = spniam.SinglePredictionNoInterceptAgentModel(plate_size=12)
    model.X_o = np.asarray([75.0])
    model.pred_o = 10.0
    parameters = {
        'alpha_mu_X': np.asarray([80.0]),
        'alpha_sigma_X': np.asarray([10.0]),
        'alpha_sigma_A': 1.0,
        'alpha_sigma_Y': 0.1,
        's': -0.5,
        'W': 11.0,
    }
    N_E = -1.3

    actual = model._sample_conditional_parameters(
        parameters, N_E, np.random.default_rng(9))

    S_x = (
        model.plate_size * parameters['alpha_mu_X'][0]
        + parameters['alpha_sigma_X'][0] * parameters['s'])
    XTX = (
        S_x**2 / model.plate_size
        + parameters['alpha_sigma_X'][0]**2 * parameters['W'])
    mean_factor = S_x / XTX
    prior_mean = np.asarray([2.0, 0.0, -0.1, 0.125])
    prior_variance = np.asarray([1.0, 0.01**2, 0.0, 0.0])
    observation_vector = np.asarray([
        N_E * model.X_o[0] * mean_factor,
        model.X_o[0] * mean_factor,
        model.X_o[0],
        N_E * model.X_o[0],
    ])
    observation_offset = 12.0 * model.X_o[0] * mean_factor
    noise_variance = (
        N_E**2 * parameters['alpha_sigma_A']**2
        + parameters['alpha_sigma_Y']**2) * model.X_o[0]**2 / XTX
    rng = np.random.default_rng(9)
    prior_draw = rng.normal(prior_mean, np.sqrt(prior_variance))
    prediction_draw = (
        observation_offset + observation_vector @ prior_draw
        + rng.normal(0.0, np.sqrt(noise_variance)))
    gain = prior_variance * observation_vector / (
        observation_vector @ (prior_variance * observation_vector)
        + noise_variance)
    expected = prior_draw + gain * (model.pred_o - prediction_draw)
    actual_vector = np.concatenate([
        np.asarray(actual[:2]), actual[2], actual[3],
    ])

    np.testing.assert_allclose(actual_vector, expected)


def test_no_intercept_cate_uses_collapsed_api():
    model = spniam.SinglePredictionNoInterceptAgentModel(plate_size=12)

    samples, mean, std = model.sample_prior_predictive_CATE(
        20, 10, draws=4, rng=4)

    assert samples.shape == (4,)
    assert np.isfinite(mean)
    assert np.isfinite(std)


def test_no_intercept_rejects_invalid_observations():
    model = spniam.SinglePredictionNoInterceptAgentModel(plate_size=12)

    with pytest.raises(ValueError, match='finite nonzero scalar'):
        model.sample(X_o=0.0, pred_o=1.0, draws=1)
    with pytest.raises(ValueError, match='finite scalar prediction'):
        model.sample(X_o=80.0, pred_o=np.nan, draws=1)


def test_no_intercept_coefficients_cannot_drift_from_constants():
    with pytest.raises(ValueError, match='fixed structural coefficient'):
        spniam.SinglePredictionNoInterceptAgentModel(
            plate_size=12, prior_parameters={'N_b_sigma': 0.1})

    model = spniam.SinglePredictionNoInterceptAgentModel(plate_size=12)
    N_b, N_d = model._sample_prior_coefficients(
        np.random.default_rng(4))

    np.testing.assert_array_equal(N_b, np.asarray([-0.1]))
    np.testing.assert_array_equal(N_d, np.asarray([0.125]))


def test_no_intercept_agent_model_save_and_load(tmp_path):
    constants = {'a': 11.0, 'b': -0.2, 'd': 0.3}
    model = spniam.SinglePredictionNoInterceptAgentModel(
        plate_size=12, scm_constants=constants)

    model.save_agent_model(file_name='saved_no_intercept', folder=tmp_path)
    loaded = spniam.SinglePredictionNoInterceptAgentModel.load_agent_model(
        file_name='saved_no_intercept', folder=tmp_path)

    assert loaded.plate_size == model.plate_size
    assert loaded.scm_constants == constants
    assert loaded.model is not None


def test_no_intercept_predictive_scm_restores_structural_intercept():
    model = spniam.SinglePredictionNoInterceptAgentModel(plate_size=12)
    parameters = {
        'alpha_mu_X': np.asarray([80.0]),
        'alpha_sigma_X': np.asarray([10.0]),
        'alpha_mu_A': 2.0,
        'alpha_sigma_A': 1.0,
        'alpha_mu_Y': 0.2,
        'alpha_sigma_Y': 0.1,
        'N_E': 1.0,
    }

    model._update_predictive_scm(
        parameters, np.asarray([-0.1]), np.asarray([0.125]))

    assert model.predictive_scm.alpha_mu_Y == 12.2
    np.testing.assert_array_equal(
        model.predictive_scm.N_b, np.asarray([-0.1]))
    np.testing.assert_array_equal(
        model.predictive_scm.N_d, np.asarray([0.125]))
