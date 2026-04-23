"""
Smoke test for the multivariate full-plate AgentModel path.

This test only checks that the multivariate code path runs end-to-end
with a small configuration and produces outputs with the expected shapes.
"""
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

import agent_model as am
import experiment
import scm


print("Testing only that the multivariate full-plate code runs, not validating results.")


def test_multivariate_full_plate():
    plate_size = 10
    draws = 20
    rng = 4

    scm_noise_parameters = {
        'alpha_mu_X': np.array([80.0, 120.0]),
        'alpha_sigma_X': np.array([10.0, 15.0]),
        'alpha_mu_A': 2.0,
        'alpha_sigma_A': 1.0,
        'alpha_mu_Y': 0.0,
        'alpha_sigma_Y': 0.1,
        'N_E': 1.0,
    }
    scm_constants = {'a': 12.0, 'b': np.array([-0.1, 0.05]), 'd': np.array([0.125, -0.05])}
    agent_priors = {
        'mu_X': np.array([80.0, 120.0]), 'sigma_X': np.array([0.1, 0.1]),
        'mu_sigma_X': np.array([10.0, 15.0]), 'sigma_sigma_X': np.array([0.1, 0.1]),
        'mu_A': 2.0, 'sigma_A': 1.0, 'mu_sigma_A': 1.0, 'sigma_sigma_A': 0.1,
        'mu_Y': 0.0, 'sigma_Y': 0.01, 'mu_sigma_Y': 0.1, 'sigma_sigma_Y': 0.01,
        'N_E_mu': 1.0, 'N_E_sigma': 1.0,
    }

    data_scm = scm.SCM(**scm_noise_parameters, **scm_constants)
    X, A, Y_historical, *_ = data_scm.sample(draws=plate_size, rng=rng)
    pred_model = LinearRegression(fit_intercept=False).fit(X, Y_historical)
    X_o = np.array([80.0, 120.0])
    pred_o = float(pred_model.predict(X_o.reshape(1, -1))[0])

    model = am.AgentModelWithFullPlateFixedCoefficients(
        plate_size=plate_size,
        prior_parameters=agent_priors,
        scm_constants=scm_constants,
        low_dim_aux_plate=False,
    )
    model.sample(X_o=X_o, pred_o=np.asarray(pred_o), draws=draws, tune=20,
                 chains=2, target_accept=.9, max_treedepth=10, rng=rng)
    prior_samples = model.sample_prior_predictive(draws=draws, rng=rng)
    posterior_samples = model.sample_posterior_predictive(rng=rng)
    prior_do, prior_do_mean, prior_do_std = model.sample_prior_predictive_do_A(X_o, 20, rng=rng)
    posterior_do, posterior_do_mean, posterior_do_std = model.sample_posterior_predictive_do_A(X_o, 20, rng=rng)
    prior_cate, prior_cate_mean, prior_cate_std = model.sample_prior_predictive_CATE(X_o, 20, 10, rng=rng)
    posterior_cate, posterior_cate_mean, posterior_cate_std = model.sample_posterior_predictive_CATE(X_o, 20, 10, rng=rng)
    prior_summary = model.sample_prior_predictive_summaries(rng=rng)
    posterior_summary = model.sample_posterior_predictive_summaries(rng=rng)

    assert model.n_covariates == 2
    assert len(prior_samples) == draws
    assert len(posterior_samples) == draws
    assert prior_do.shape == (draws,)
    assert posterior_do.shape == (draws,)
    assert prior_cate.shape == (draws,)
    assert posterior_cate.shape == (draws,)
    assert np.isfinite(prior_do_mean)
    assert np.isfinite(prior_do_std)
    assert np.isfinite(posterior_do_mean)
    assert np.isfinite(posterior_do_std)
    assert np.isfinite(prior_cate_mean)
    assert np.isfinite(prior_cate_std)
    assert np.isfinite(posterior_cate_mean)
    assert np.isfinite(posterior_cate_std)
    assert len(prior_summary) == 12
    assert len(posterior_summary) == 12

    print("Multivariate full-plate smoke test passed.")


def test_experiment_build_agent_model_uses_fixed_coefficient_subclass():
    agent_priors = {
        'mu_X': np.array([80.0, 120.0]), 'sigma_X': np.array([0.1, 0.1]),
        'mu_sigma_X': np.array([10.0, 15.0]), 'sigma_sigma_X': np.array([0.1, 0.1]),
        'mu_A': 2.0, 'sigma_A': 1.0, 'mu_sigma_A': 1.0, 'sigma_sigma_A': 0.1,
        'mu_Y': 0.0, 'sigma_Y': 0.01, 'mu_sigma_Y': 0.1, 'sigma_sigma_Y': 0.01,
        'N_E_mu': 1.0, 'N_E_sigma': 1.0,
    }
    scm_constants = {'a': 12.0, 'b': np.array([-0.1, 0.05]), 'd': np.array([0.125, -0.05])}

    model = experiment.build_agent_model(
        agent_priors=agent_priors,
        scm_constants=scm_constants,
        predictive_var_names=experiment.predictive_var_names,
        plate_size=10,
        low_dim_aux_plate=False,
    )

    assert isinstance(model, am.AgentModelWithFullPlateFixedCoefficients)


def test_experiment_build_agent_model_rejects_multivariate_reduced_fixed_coefficients():
    agent_priors = {
        'mu_X': np.array([80.0, 120.0]), 'sigma_X': np.array([0.1, 0.1]),
        'mu_sigma_X': np.array([10.0, 15.0]), 'sigma_sigma_X': np.array([0.1, 0.1]),
        'mu_A': 2.0, 'sigma_A': 1.0, 'mu_sigma_A': 1.0, 'sigma_sigma_A': 0.1,
        'mu_Y': 0.0, 'sigma_Y': 0.01, 'mu_sigma_Y': 0.1, 'sigma_sigma_Y': 0.01,
        'N_E_mu': 1.0, 'N_E_sigma': 1.0,
    }
    scm_constants = {'a': 12.0, 'b': np.array([-0.1, 0.05]), 'd': np.array([0.125, -0.05])}

    with pytest.raises(NotImplementedError):
        experiment.build_agent_model(
            agent_priors=agent_priors,
            scm_constants=scm_constants,
            predictive_var_names=experiment.predictive_var_names,
            plate_size=10,
            low_dim_aux_plate=True,
        )


if __name__ == '__main__':
    test_multivariate_full_plate()
