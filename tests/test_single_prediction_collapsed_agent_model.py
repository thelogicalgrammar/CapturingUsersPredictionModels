"""
Smoke tests for the single-prediction collapsed model.
"""

import arviz as az
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as at
from scipy.special import logsumexp
from scipy.stats import ks_2samp, ncx2, norm

import single_prediction_collapsed_agent_model as spcam
from distributions import (
    FoldedNormal,
    NoncentralChiSquared,
    folded_normal_logp,
    ncx2_logp,
)


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


def synthetic_marginal_posterior(n_covariates=2, n_draws=5, plate_size=12):
    """Build posterior draws before signed-parameter reconstruction."""
    rng = np.random.default_rng(8)
    magnitude = np.linspace(0.5, 1.5, n_draws).reshape(1, n_draws)
    offsets = np.linspace(-0.02, 0.02, n_draws).reshape(1, n_draws)
    return {
        'alpha_mu_X': (
            np.full((1, n_draws, n_covariates), 80.0)
            + offsets[:, :, np.newaxis]),
        'alpha_sigma_X': (
            np.full((1, n_draws, n_covariates), 10.0)
            + offsets[:, :, np.newaxis]),
        'alpha_sigma_A': np.full((1, n_draws), 1.0) + offsets,
        'alpha_sigma_Y': np.full((1, n_draws), 0.1) + offsets,
        'N_E_magnitude': magnitude,
        'N_E_positive_probability': np.linspace(
            0.2, 0.9, n_draws).reshape(1, n_draws),
        'R': rng.chisquare(df=n_covariates,
                           size=(1, n_draws)) / plate_size,
        'T': rng.chisquare(df=plate_size - n_covariates,
                           size=(1, n_draws)),
    }


def collapsed_model_with_synthetic_posterior(n_covariates, n_draws=5):
    """Build an observed collapsed model suitable for summary tests."""
    posterior = synthetic_posterior(
        n_covariates=n_covariates,
        n_draws=n_draws)
    # Give every directly summarized parameter non-constant draws, avoiding
    # undefined skew/kurtosis while preserving valid scale parameters.
    offsets = np.linspace(-0.02, 0.02, n_draws).reshape(1, n_draws)
    for name in ('alpha_mu_A', 'alpha_sigma_A',
                 'alpha_mu_Y', 'alpha_sigma_Y'):
        posterior[name] = posterior[name] + offsets
    for name in ('alpha_mu_X', 'alpha_sigma_X'):
        posterior[name] = posterior[name] + offsets[:, :, np.newaxis]

    model = spcam.SinglePredictionCollapsedAgentModel(
        n_covariates=n_covariates,
        plate_size=12)
    model.trace = az.from_dict(posterior=posterior)
    model.X_o = np.linspace(80.0, 120.0, n_covariates)
    model.pred_o = 30.0
    return model


def expected_parameter_summary_keys(n_covariates):
    """Return the exact AgentModel-compatible summary schema."""
    moments = ('mean', 'std', 'skew', 'kurtosis')
    scalar_parameters = ('alpha_mu_A', 'alpha_sigma_A',
                         'alpha_mu_Y', 'alpha_sigma_Y', 'N_E')
    vector_parameters = ('alpha_mu_X', 'alpha_sigma_X', 'N_b', 'N_d')
    prefixes = ['param_{}'.format(name) for name in scalar_parameters]
    for name in vector_parameters:
        if n_covariates == 1:
            prefixes.append('param_{}'.format(name))
        else:
            prefixes.extend('param_{}_{}'.format(name, dim)
                            for dim in range(n_covariates))
    return {'{}_{}'.format(prefix, moment)
            for prefix in prefixes for moment in moments}


def test_collapsed_likelihood_keeps_prediction_scalar():
    for n_covariates in (1, 2):
        model = spcam.SinglePredictionCollapsedAgentModel(
            n_covariates=n_covariates,
            plate_size=12)
        X_o = np.linspace(80.0, 120.0, n_covariates)
        pred_o = 30.0

        pm.set_data({'X_o_': X_o, 'pred_o_': np.asarray(pred_o)},
                    model=model.model)
        with model.model:
            logp = model.model.compile_logp()
            dlogp = model.model.compile_dlogp()
            initial_point = model.model.initial_point()
            value = logp(initial_point)
            gradient = dlogp(initial_point)

        assert np.isfinite(value)
        assert np.all(np.isfinite(gradient))


def test_collapsed_model_marginalizes_structural_coefficients():
    model = spcam.SinglePredictionCollapsedAgentModel(
        n_covariates=3,
        plate_size=12)

    assert 'N_E_magnitude' in model.model.named_vars
    assert 'N_E' not in model.model.named_vars
    assert 'alpha_mu_A' not in model.model.named_vars
    assert 'alpha_mu_Y' not in model.model.named_vars
    assert 'R' in model.model.named_vars
    assert 'T' in model.model.named_vars
    assert 'NCX' in model.model.named_vars
    assert 'N_b' not in model.model.named_vars
    assert 'N_d' not in model.model.named_vars


def test_collapsed_model_samples_constant_number_of_plate_variables():
    # The plate is represented by two scalar primitives for every k: (G_R, T)
    # for the signed k = 1 construction and (NCX, T) for k > 1.
    population_variables = {'alpha_mu_X', 'alpha_sigma_X', 'alpha_sigma_A',
                            'alpha_sigma_Y', 'N_E_magnitude'}
    for n_covariates in (1, 2, 5, 10):
        model = spcam.SinglePredictionCollapsedAgentModel(
            n_covariates=n_covariates,
            plate_size=12)
        free_variables = {variable.name
                          for variable in model.model.free_RVs}
        plate_variable = 'G_R' if n_covariates == 1 else 'NCX'
        assert free_variables == population_variables | {plate_variable, 'T'}
        assert ('G_R' in model.model.named_vars) == (n_covariates == 1)
        assert 'NCX' in model.model.named_vars
        assert ('NCX' in free_variables) == (n_covariates > 1)


def test_folded_normal_uses_softplus_transform():
    model = spcam.SinglePredictionCollapsedAgentModel(
        n_covariates=1,
        plate_size=12)

    assert 'N_E_magnitude_log_exp_m1__' in model.model.initial_point()
    assert 'N_E_magnitude_log__' not in model.model.initial_point()


def test_model_records_rao_blackwellized_sign_deterministics():
    model = spcam.SinglePredictionCollapsedAgentModel(
        n_covariates=1,
        plate_size=12)

    assert 'N_E_positive_probability' in model.model.named_vars
    assert 'N_E_conditional_mean' in model.model.named_vars


def test_folded_normal_logp_matches_both_normal_branches():
    value = at.dscalar('value')
    mu = at.dscalar('mu')
    sigma = at.dscalar('sigma')
    logp = pm.pytensorf.compile(
        [value, mu, sigma], folded_normal_logp(value, mu, sigma))
    for mean in (-2.0, 0.0, 1.5):
        for point in (1e-6, 0.2, 2.0, 8.0):
            expected = logsumexp((
                norm.logpdf(point, loc=mean, scale=1.3),
                norm.logpdf(-point, loc=mean, scale=1.3),
            ))
            assert np.isclose(logp(point, mean, 1.3), expected)
        assert logp(-0.1, mean, 1.3) == -np.inf


def test_folded_normal_reconstructs_asymmetric_normal_prior():
    mean = 1.2
    sigma = 0.8
    with pm.Model():
        FoldedNormal('magnitude', mean, sigma)
        samples = pm.sample_prior_predictive(draws=30000, random_seed=7)
    magnitude = samples.prior['magnitude'].values.reshape(-1)
    model = spcam.SinglePredictionCollapsedAgentModel(
        n_covariates=1,
        plate_size=12,
        prior_parameters={'N_E_mu': mean, 'N_E_sigma': sigma})
    positive_probability = model._prior_positive_sign_probability(magnitude)
    rng = np.random.default_rng(11)
    signed = np.where(
        rng.random(magnitude.size) < positive_probability,
        magnitude,
        -magnitude)
    reference = rng.normal(mean, sigma, size=magnitude.size)

    assert ks_2samp(signed, reference).pvalue > 0.01
    assert np.isclose(
        np.mean(signed > 0.0), norm.cdf(mean / sigma), atol=0.01)


def test_sign_marginalized_density_matches_explicit_signed_sum():
    N_E_mu = 1.3
    N_E_sigma = 0.7
    magnitude = 2.1
    prediction = 24.0
    branch_mu = np.asarray([21.0, 31.0])
    branch_sigma = np.asarray([1.8, 2.3])
    log_prior = np.asarray([
        norm.logpdf(magnitude, N_E_mu, N_E_sigma),
        norm.logpdf(-magnitude, N_E_mu, N_E_sigma),
    ])
    log_likelihood = norm.logpdf(
        prediction, branch_mu, branch_sigma)

    explicit = logsumexp(log_prior + log_likelihood)
    log_folded = logsumexp(log_prior)
    marginalized = (
        log_folded
        + logsumexp(log_prior - log_folded + log_likelihood))

    assert np.isclose(marginalized, explicit, atol=1e-12)


def test_signed_cochran_density_matches_noncentral_chi_squared():
    # If G is standard normal and Y = (sqrt(n) * delta + G)^2, the density
    # obtained from both inverse square-root branches must equal the ncx2
    # density with one degree of freedom and non-centrality n * delta**2.
    n = 1000
    positive_values = (1e-4, 0.1, 1.0, 10.0, 100.0)
    for signed_delta in (-0.4, -0.01, 0.0, 0.01, 0.4):
        shifted_mean = np.sqrt(n) * signed_delta
        for value in positive_values:
            root = np.sqrt(value)
            transformed_log_density = logsumexp((
                norm.logpdf(root - shifted_mean),
                norm.logpdf(-root - shifted_mean),
            )) - np.log(2.0 * root)
            reference_log_density = ncx2.logpdf(
                value, df=1, nc=n * signed_delta**2)
            assert np.isclose(
                transformed_log_density, reference_log_density,
                rtol=1e-11, atol=1e-11)


def test_signed_cochran_moments_match_noncentral_chi_squared():
    # Test negative, central, and positive signed offsets. The first two
    # moments are deterministic properties of the target law and complement
    # the pointwise density test above.
    rng = np.random.default_rng(341)
    n = 1000
    size = 200000
    for signed_delta in (-0.25, 0.0, 0.25):
        G_R = rng.normal(size=size)
        R = (signed_delta + G_R / np.sqrt(n))**2
        noncentrality = n * signed_delta**2
        expected_mean = (1.0 + noncentrality) / n
        expected_variance = 2.0 * (1.0 + 2.0 * noncentrality) / n**2
        assert np.isclose(np.mean(R), expected_mean, rtol=3e-3)
        assert np.isclose(np.var(R), expected_variance, rtol=1e-2)


def test_signed_cochran_k1_prior_predictive_keeps_public_variables():
    model = spcam.SinglePredictionCollapsedAgentModel(
        n_covariates=1,
        plate_size=12)
    model.sample_prior_predictive(draws=20, rng=4)

    assert set(model.prior_samples.prior.data_vars) == set(
        model.predictive_var_names)
    assert np.all(np.isfinite(model.prior_samples.prior['R'].values))
    assert np.all(model.prior_samples.prior['R'].values >= 0.0)


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


def test_noncentral_chi_squared_logp_is_stable_at_large_arguments():
    value = at.dscalar('large_value')
    nu_ = at.dscalar('large_nu')
    lam_ = at.dscalar('large_lam')
    logp = ncx2_logp(value, nu_, lam_)
    gradients = pytensor.grad(logp, (value, lam_))
    evaluate = pm.pytensorf.compile(
        [value, nu_, lam_], [logp, *gradients])
    references = [
        (
            2.0, 3.97e9, 3.970000002e9,
            -12.663101679807776,
            -1.8891687649896262e-10,
            6.297229217160822e-11,
        ),
        (
            16.0, 3.97e9, 3.32e9,
            -14517951.589032311,
            0.04675934761373097,
            -0.04276044657995294,
        ),
        (
            32.0, 1e12, 1.000000000032e12,
            -15.427596271737267,
            -7.499999999838125e-13,
            2.499999999921875e-13,
        ),
    ]
    for nu, lam, point, expected_logp, expected_dx, expected_dlam in \
            references:
        actual = evaluate(point, nu, lam)
        assert np.all(np.isfinite(actual))
        np.testing.assert_allclose(
            actual[0], expected_logp, rtol=2e-13, atol=2e-8)
        np.testing.assert_allclose(
            actual[1:], (expected_dx, expected_dlam),
            rtol=2e-10, atol=2e-12)


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


def test_joint_parameter_reconstruction_matches_analytic():
    model = spcam.SinglePredictionCollapsedAgentModel(
        n_covariates=2,
        plate_size=12)
    model.X_o = np.array([80.0, 120.0])
    model.pred_o = 30.0
    parameters = {
        'alpha_sigma_A': 1.0,
        'alpha_sigma_Y': 0.1,
        'R': 0.5,
        'T': 9.0,
    }
    N_E = -1.3
    rng = np.random.default_rng(9)
    draws = []
    for _ in range(50000):
        alpha_mu_A, alpha_mu_Y, N_b, N_d = (
            model._sample_conditional_parameters(parameters, N_E, rng))
        draws.append(np.concatenate(
            [np.asarray([alpha_mu_A, alpha_mu_Y]), N_b, N_d]))
    draws = np.asarray(draws)

    prior_mean = np.concatenate([
        np.asarray([model.prior_parameters['mu_A'],
                    model.prior_parameters['mu_Y']]),
        model.prior_parameters['N_b_mu'],
        model.prior_parameters['N_d_mu'],
    ])
    prior_variance = np.concatenate([
        np.asarray([model.prior_parameters['sigma_A']**2,
                    model.prior_parameters['sigma_Y']**2]),
        model.prior_parameters['N_b_sigma']**2,
        model.prior_parameters['N_d_sigma']**2,
    ])
    observation_vector = np.concatenate([
        np.asarray([N_E, 1.0]),
        model.X_o,
        N_E * model.X_o,
    ])
    noise_variance = (
        N_E**2 * parameters['alpha_sigma_A']**2
        + parameters['alpha_sigma_Y']**2
    ) * (1.0 / model.plate_size + parameters['R'] / parameters['T'])
    denominator = (
        observation_vector @ (prior_variance * observation_vector)
        + noise_variance)
    expected_mean = prior_mean + prior_variance * observation_vector * (
        model.pred_o - observation_vector @ prior_mean) / denominator
    expected_cov = np.diag(prior_variance) - np.outer(
        prior_variance * observation_vector,
        prior_variance * observation_vector) / denominator

    assert np.allclose(np.mean(draws, axis=0), expected_mean, atol=3e-3)
    assert np.allclose(np.cov(draws, rowvar=False), expected_cov, atol=1e-2)


def test_signed_reconstruction_is_cached_and_deterministic():
    first = spcam.SinglePredictionCollapsedAgentModel(
        n_covariates=2, plate_size=12)
    second = spcam.SinglePredictionCollapsedAgentModel(
        n_covariates=2, plate_size=12)
    for model in (first, second):
        model.trace = az.from_dict(
            posterior=synthetic_marginal_posterior())
        model.X_o = np.array([80.0, 120.0])
        model.pred_o = 30.0
        model._reconstruct_posterior_parameters(np.random.default_rng(17))

    for name in ('N_E', 'alpha_mu_A', 'alpha_mu_Y'):
        np.testing.assert_equal(
            first.trace.posterior[name].values,
            second.trace.posterior[name].values)
    for first_draw, second_draw in zip(
            first.posterior_coefficient_draws,
            second.posterior_coefficient_draws):
        np.testing.assert_equal(first_draw[0], second_draw[0])
        np.testing.assert_equal(first_draw[1], second_draw[1])
    assert np.isclose(
        first.posterior_negative_sign_probability(),
        np.mean(1.0 - synthetic_marginal_posterior()[
            'N_E_positive_probability']))
    first_summaries = first.posterior_parameter_summaries(rng=1)
    magnitude = synthetic_marginal_posterior()['N_E_magnitude'].reshape(-1)
    positive = synthetic_marginal_posterior()[
        'N_E_positive_probability'].reshape(-1)
    assert np.isclose(
        first_summaries['param_N_E_mean'],
        np.mean((2.0 * positive - 1.0) * magnitude))


def test_posterior_parameter_summaries_schema_and_determinism():
    # Equivalent model instances supplied with the same seed must reconstruct
    # exactly the same Matheron coefficient draws.  The returned key set must
    # also match AgentModel's scalar/k=1 and indexed/k>1 convention exactly.
    for n_covariates in (1, 2):
        first = collapsed_model_with_synthetic_posterior(n_covariates)
        second = collapsed_model_with_synthetic_posterior(n_covariates)

        first_summaries = first.posterior_parameter_summaries(rng=17)
        second_summaries = second.posterior_parameter_summaries(rng=17)

        assert set(first_summaries) == expected_parameter_summary_keys(
            n_covariates)
        for key in first_summaries:
            np.testing.assert_equal(first_summaries[key],
                                    second_summaries[key])

        N_E = first.trace.posterior['N_E'].values.reshape(-1)
        assert first_summaries['param_N_E_mean'] == np.mean(N_E)
        assert first_summaries['param_N_E_std'] == np.std(N_E)


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


def test_collapsed_posterior_cate_rao_blackwellizes_sign():
    posterior = synthetic_marginal_posterior(n_draws=5)
    magnitude = posterior['N_E_magnitude'].reshape(-1)
    positive = posterior['N_E_positive_probability'].reshape(-1)
    model = spcam.SinglePredictionCollapsedAgentModel(
        n_covariates=2,
        plate_size=12)
    model.trace = az.from_dict(posterior=posterior)
    model.X_o = np.array([80.0, 120.0])
    model.pred_o = 30.0
    model._reconstruct_posterior_parameters(np.random.default_rng(3))

    samples, mean, std = model.sample_posterior_predictive_CATE(20, 10)
    expected_N_E_mean = np.mean((2.0 * positive - 1.0) * magnitude)
    expected_N_E_variance = (
        np.mean(magnitude**2) - expected_N_E_mean**2)

    assert samples.shape == magnitude.shape
    assert np.isclose(mean, 10.0 * expected_N_E_mean)
    assert np.isclose(std, 10.0 * np.sqrt(expected_N_E_variance))


def test_collapsed_prior_predictive_summaries_and_do_A():
    model = spcam.SinglePredictionCollapsedAgentModel(
        n_covariates=2,
        plate_size=12)
    model.sample_prior_predictive(draws=8, rng=4)

    summaries = model.sample_prior_predictive_summaries(rng=4)
    do_A_samples, do_A_mean, do_A_std = model.sample_prior_predictive_do_A(
        np.array([80.0, 80.0]), 10, rng=4)

    assert all(np.isfinite(value) for value in summaries.values())
    expected_X_keys = {
        'pri_X_{}_{}'.format(dim, moment)
        for dim in range(2)
        for moment in ('mean', 'std', 'skew', 'kurtosis')
    }
    assert set(summaries) >= expected_X_keys | {'pri_A_mean', 'pri_Y_mean'}
    assert 'pri_X_mean' not in summaries
    assert do_A_samples.shape == (8,)
    assert np.isfinite(do_A_mean)
    assert np.isfinite(do_A_std)


def test_prior_do_A_advances_integer_seed_across_draws(monkeypatch):
    n_draws = 5
    model = spcam.SinglePredictionCollapsedAgentModel(
        n_covariates=1,
        plate_size=12)
    model.prior_samples = az.from_dict(
        prior=synthetic_posterior(
            n_covariates=1,
            n_draws=n_draws))
    model.prior_draws = n_draws
    model.prior_coefficient_draws = [
        (np.zeros(1), np.zeros(1)) for _ in range(n_draws)]

    def sample_do_A(_treatment, _X, rng=None):
        draw = np.random.default_rng(rng).normal(size=1)
        return draw, float(draw[0]), 0.0

    monkeypatch.setattr(model.predictive_scm, 'sample_do_A', sample_do_A)
    first = model.sample_prior_predictive_do_A(
        np.asarray([80.0]), 10, rng=17)[0]
    second = model.sample_prior_predictive_do_A(
        np.asarray([80.0]), 10, rng=17)[0]

    np.testing.assert_equal(first, second)
    assert np.unique(first).size == n_draws


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
    expected_X_keys = {
        'post_X_{}_{}'.format(dim, moment)
        for dim in range(2)
        for moment in ('mean', 'std', 'skew', 'kurtosis')
    }
    assert set(summaries) >= expected_X_keys | {'post_A_mean', 'post_Y_mean'}
    assert 'post_X_mean' not in summaries
    assert do_A_samples.shape == (3,)
    assert np.isfinite(do_A_mean)
    assert np.isfinite(do_A_std)


def test_collapsed_agent_model_save_and_load(tmp_path):
    model = spcam.SinglePredictionCollapsedAgentModel(
        n_covariates=2,
        plate_size=12)
    model.trace = az.from_dict(
        posterior=synthetic_marginal_posterior(n_draws=3))
    model.posterior_draws = 3
    model.X_o = np.array([80.0, 80.0])
    model.pred_o = 30.0
    model._reconstruct_posterior_parameters(np.random.default_rng(4))
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
    for name in ('N_E_magnitude', 'N_E_positive_probability', 'N_E',
                 'alpha_mu_A', 'alpha_mu_Y'):
        np.testing.assert_equal(
            loaded_model.trace.posterior[name].values,
            model.trace.posterior[name].values)
    assert (
        loaded_model.posterior_negative_sign_probability()
        == model.posterior_negative_sign_probability())
