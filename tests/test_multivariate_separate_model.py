"""
Smoke tests for the multivariate model.
"""

import warnings

import arviz as az
import numpy as np
import pymc as pm
from sklearn.linear_model import LinearRegression

import multivariate_agent_model as mam
import multivariate_scm as mscm


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
    model = mscm.MultivariateSCM(
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
    assert np.isfinite(CATE_std)


def test_multivariate_agent_prior_predictive_with_intercept():
    model = mam.MultivariateAgentModel(
        n_covariates=2,
        plate_size=8,
        fit_intercept=True)
    samples = model.sample_prior_predictive(draws=3, rng=4)

    assert len(samples) == 3
    assert samples[0][0].shape == (1, 2)


def test_multivariate_agent_prior_predictive_without_intercept():
    model = mam.MultivariateAgentModel(
        n_covariates=2,
        plate_size=8,
        fit_intercept=False)
    samples = model.sample_prior_predictive(draws=3, rng=4)

    assert len(samples) == 3
    assert samples[0][0].shape == (1, 2)


def test_single_prediction_likelihood_keeps_prediction_scalar():
    data_scm = mscm.MultivariateSCM(n_covariates=2)
    X, A, Y, *_ = data_scm.sample(draws=12, rng=4)
    pred_model = LinearRegression(fit_intercept=True).fit(X, Y)
    X_o = np.array([80.0, 120.0])
    pred_o = float(pred_model.predict(X_o.reshape(1, -1))[0])

    model = mam.MultivariateAgentModel(
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
    model = mam.MultivariateAgentModel(
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
    model = mam.MultivariateAgentModel(
        n_covariates=2,
        plate_size=8,
        fit_intercept=True)
    model.trace = az.from_dict(posterior=synthetic_posterior())
    model.posterior_draws = 1
    model.X_o = np.array([80.0, 80.0])
    model.sample_prior_predictive(draws=1, rng=4)

    model.save_agent_model(file_name='saved_model', folder=tmp_path)
    loaded_model = mam.MultivariateAgentModel.load_agent_model(
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
