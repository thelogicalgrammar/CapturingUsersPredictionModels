"""
Smoke tests for the single-prediction collapsed model.
"""

import arviz as az
import numpy as np
import pymc as pm
import pytensor.tensor as at
from scipy.stats import ks_2samp, ncx2

import single_prediction_collapsed_agent_model as spcam
from distributions import NoncentralChiSquared, ncx2_logp


def synthetic_posterior(n_covariates=2, n_draws=3, plate_size=12):
    """
    Build a minimal posterior dictionary with all predictive variables.
    """
    rng = np.random.default_rng(4)
    return {
        'alpha_mu_X': np.full((1, n_draws, n_covariates), 80.0),
        'alpha_sigma_X': np.full((1, n_draws, n_covariates), 10.0),
        'alpha_mu_A': np.full((1, n_draws), 2.0),
        'alpha_sigma_A': np.full((1, n_draws), 1.0),
        'alpha_mu_Y': np.full((1, n_draws), 12.0),
        'alpha_sigma_Y': np.full((1, n_draws), 0.1),
        'N_E': np.linspace(0.5, 1.5, n_draws).reshape(1, n_draws),
        'R': rng.chisquare(df=n_covariates,
                           size=(1, n_draws)) / plate_size,
        'T': rng.chisquare(df=plate_size - n_covariates,
                           size=(1, n_draws)),
    }


def test_collapsed_likelihood_keeps_prediction_scalar():
    model = spcam.SinglePredictionCollapsedAgentModel(
        n_covariates=2,
        plate_size=12)
    X_o = np.array([80.0, 120.0])
    pred_o = 30.0

    pm.set_data({'X_o_': X_o, 'pred_o_': np.asarray(pred_o)},
                model=model.model)
    with model.model:
        logp = model.model.compile_logp()
        value = logp(model.model.initial_point())

    assert np.isfinite(value)


def test_collapsed_model_marginalizes_structural_coefficients():
    model = spcam.SinglePredictionCollapsedAgentModel(
        n_covariates=3,
        plate_size=12)

    assert 'N_E' in model.model.named_vars
    assert 'R' in model.model.named_vars
    assert 'T' in model.model.named_vars
    assert 'NCX' in model.model.named_vars
    assert 'N_b' not in model.model.named_vars
    assert 'N_d' not in model.model.named_vars


def test_collapsed_model_samples_constant_number_of_plate_variables():
    # The plate is represented by the two scalar primitives (NCX, T) for
    # every k, so the set of sampled variables must not depend on k. This
    # includes k = 1, which needed a special case under the Cochran
    # construction (convention ChiSquared(0) = 0) but not under the native
    # non-central chi-squared.
    population_variables = {'alpha_mu_X', 'alpha_sigma_X', 'alpha_mu_A',
                            'alpha_sigma_A', 'alpha_mu_Y', 'alpha_sigma_Y',
                            'N_E'}
    for n_covariates in (1, 2, 5, 10):
        model = spcam.SinglePredictionCollapsedAgentModel(
            n_covariates=n_covariates,
            plate_size=12)
        free_variables = {variable.name
                          for variable in model.model.free_RVs}
        assert free_variables == population_variables | {'NCX', 'T'}


def test_noncentral_chi_squared_logp_matches_scipy():
    # The custom logp must reproduce scipy.stats.ncx2.logpdf, including the
    # negative half-integer Bessel order at nu = 1 and small non-centralities.
    value = at.dscalar('value')
    nu_ = at.dscalar('nu')
    lam_ = at.dscalar('lam')
    logp = pm.pytensorf.compile([value, nu_, lam_],
                                ncx2_logp(value, nu_, lam_))
    for nu in (1.0, 2.0, 3.0, 8.0):
        for lam in (1e-3, 0.5, 10.0, 50.0):
            for x in (0.1, 1.0, 8.0, 60.0):
                assert np.isclose(logp(x, nu, lam),
                                  ncx2.logpdf(x, df=nu, nc=lam),
                                  atol=1e-8)


def test_noncentral_chi_squared_forward_sampling_matches_scipy():
    # The random callable used for predictive sampling must draw from the
    # same law as scipy.stats.ncx2.
    with pm.Model():
        NoncentralChiSquared('NCX', nu=3.0, lam=7.5)
        draws = pm.sample_prior_predictive(draws=20000, random_seed=4)
    samples = draws.prior['NCX'].values.ravel()
    reference = ncx2.rvs(df=3.0, nc=7.5, size=20000,
                         random_state=np.random.default_rng(4))
    assert ks_2samp(samples, reference).pvalue > 0.01


def test_cochran_construction_matches_noncentral_chi_squared():
    # R = (Delta_X + G_R / sqrt(n))^2 + Z_R / n must reproduce
    # (1 / n) * noncentral_chisquare(k, n * Delta_X^2).
    rng = np.random.default_rng(4)
    n = 1000
    for k in (1, 2, 5):
        delta = rng.normal(size=k)
        delta_squared = float(delta @ delta)
        size = 20000

        reference = rng.noncentral_chisquare(
            df=k, nonc=n * delta_squared, size=size) / n

        G_R = rng.normal(size=size)
        Z_R = rng.chisquare(df=k - 1, size=size) if k > 1 else 0.0
        cochran = (np.sqrt(delta_squared) + G_R / np.sqrt(n))**2 + Z_R / n

        assert ks_2samp(reference, cochran).pvalue > 0.01


def test_conditional_coefficient_reconstruction_matches_analytic():
    # The Matheron draws must match the closed-form conditional mean and
    # covariance of theta = (N_b, N_d) given the observed prediction.
    model = spcam.SinglePredictionCollapsedAgentModel(
        n_covariates=2,
        plate_size=12)
    model.X_o = np.array([80.0, 120.0])
    model.pred_o = 30.0
    parameters = {
        'N_E': 1.0,
        'alpha_mu_A': 2.0, 'alpha_sigma_A': 1.0,
        'alpha_mu_Y': 12.0, 'alpha_sigma_Y': 0.1,
        'R': 0.5, 'T': 9.0,
    }
    rng = np.random.default_rng(4)
    draws = np.array([
        np.concatenate(model._sample_conditional_coefficients(parameters, rng))
        for _ in range(50000)])

    prior_mean = np.concatenate([model.prior_parameters['N_b_mu'],
                                 model.prior_parameters['N_d_mu']])
    prior_variance = np.concatenate([model.prior_parameters['N_b_sigma']**2,
                                     model.prior_parameters['N_d_sigma']**2])
    a = np.concatenate([model.X_o, parameters['N_E'] * model.X_o])
    c = parameters['alpha_mu_Y'] + parameters['N_E'] * parameters['alpha_mu_A']
    tau_squared = (parameters['N_E']**2 * parameters['alpha_sigma_A']**2 +
                   parameters['alpha_sigma_Y']**2)
    noise_variance = tau_squared * (1.0 / model.plate_size +
                                    parameters['R'] / parameters['T'])
    denominator = a @ (prior_variance * a) + noise_variance
    expected_mean = prior_mean + prior_variance * a * (
        model.pred_o - c - a @ prior_mean) / denominator
    expected_cov = np.diag(prior_variance) - np.outer(
        prior_variance * a, prior_variance * a) / denominator

    assert np.allclose(np.mean(draws, axis=0), expected_mean, atol=2e-3)
    assert np.allclose(np.cov(draws, rowvar=False), expected_cov, atol=2e-3)


def test_collapsed_prior_cate_uses_ne_prior():
    model = spcam.SinglePredictionCollapsedAgentModel(
        n_covariates=2,
        plate_size=12)
    samples, mean, std = model.sample_prior_predictive_CATE(
        20, 10, draws=4, rng=4)

    assert samples.shape == (4,)
    assert np.isfinite(mean)
    assert np.isfinite(std)


def test_collapsed_posterior_cate_uses_ne_trace():
    model = spcam.SinglePredictionCollapsedAgentModel(
        n_covariates=2,
        plate_size=12)
    model.trace = az.from_dict(posterior=synthetic_posterior())

    samples, mean, std = model.sample_posterior_predictive_CATE(20, 10)

    assert np.array_equal(samples, np.array([5.0, 10.0, 15.0]))
    assert mean == 10.0
    assert np.isclose(std, np.std(np.array([5.0, 10.0, 15.0])))


def test_collapsed_prior_predictive_summaries_and_do_A():
    model = spcam.SinglePredictionCollapsedAgentModel(
        n_covariates=2,
        plate_size=12)
    model.sample_prior_predictive(draws=8, rng=4)

    summaries = model.sample_prior_predictive_summaries(rng=4)
    do_A_samples, do_A_mean, do_A_std = model.sample_prior_predictive_do_A(
        np.array([80.0, 80.0]), 10, rng=4)

    assert all(np.isfinite(value) for value in summaries.values())
    assert set(summaries.keys()) >= {'pri_X_mean', 'pri_A_mean', 'pri_Y_mean'}
    assert do_A_samples.shape == (8,)
    assert np.isfinite(do_A_mean)
    assert np.isfinite(do_A_std)


def test_collapsed_posterior_predictive_summaries_and_do_A():
    model = spcam.SinglePredictionCollapsedAgentModel(
        n_covariates=2,
        plate_size=12)
    model.trace = az.from_dict(posterior=synthetic_posterior())
    model.X_o = np.array([80.0, 80.0])
    model.pred_o = 30.0

    summaries = model.sample_posterior_predictive_summaries(rng=4)
    do_A_samples, do_A_mean, do_A_std = model.sample_posterior_predictive_do_A(
        np.array([80.0, 80.0]), 10, rng=4)

    assert all(np.isfinite(value) for value in summaries.values())
    assert set(summaries.keys()) >= {'post_X_mean', 'post_A_mean', 'post_Y_mean'}
    assert do_A_samples.shape == (3,)
    assert np.isfinite(do_A_mean)
    assert np.isfinite(do_A_std)


def test_collapsed_agent_model_save_and_load(tmp_path):
    model = spcam.SinglePredictionCollapsedAgentModel(
        n_covariates=2,
        plate_size=12)
    model.trace = az.from_dict(posterior=synthetic_posterior())
    model.posterior_draws = 3
    model.X_o = np.array([80.0, 80.0])
    model.pred_o = 30.0
    model.sample_prior_predictive(draws=1, rng=4)

    model.save_agent_model(file_name='saved_collapsed_model', folder=tmp_path)
    loaded_model = spcam.SinglePredictionCollapsedAgentModel.load_agent_model(
        file_name='saved_collapsed_model',
        folder=tmp_path)

    assert loaded_model.n_covariates == model.n_covariates
    assert loaded_model.plate_size == model.plate_size
    assert loaded_model.posterior_draws == model.posterior_draws
    assert loaded_model.prior_draws == model.prior_draws
    assert np.array_equal(loaded_model.X_o, model.X_o)
    assert loaded_model.pred_o == model.pred_o
    assert loaded_model.model is not None
    assert loaded_model.trace is not None
    assert loaded_model.prior_samples is not None
