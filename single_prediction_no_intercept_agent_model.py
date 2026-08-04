"""Collapsed scalar agent model for a prediction without an intercept.

The historical prediction is produced by a scalar OLS model without an
intercept.  Conditional on the reduced covariate statistics ``(s, W)``, its
likelihood is Gaussian.  The treatment effect sign and the treatment and
outcome means are integrated analytically, leaving NUTS to sample only the
magnitude of the treatment effect and the remaining continuous variables.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as at

from single_prediction_collapsed_agent_model import (
    SinglePredictionCollapsedAgentModel,
)


class SinglePredictionNoInterceptAgentModel(
        SinglePredictionCollapsedAgentModel):
    """Single-prediction model for scalar slope-only OLS.

    Args:
        plate_size (int): Number of historical observations.
        prior_parameters (dict): Scalar population-prior parameters.
        scm_constants (dict): Fixed structural coefficients ``a``, ``b`` and
            ``d``.
    """

    def __init__(self, plate_size, prior_parameters=None,
                 scm_constants=None):
        if scm_constants is None:
            scm_constants = {'a': 12.0, 'b': -0.1, 'd': 0.125}
        expected_constants = {'a', 'b', 'd'}
        if set(scm_constants) != expected_constants:
            raise ValueError(
                "scm_constants must contain exactly {}.".format(
                    sorted(expected_constants)))
        self.scm_constants = {
            name: float(value) for name, value in scm_constants.items()
        }
        super().__init__(
            n_covariates=1,
            plate_size=plate_size,
            prior_parameters=prior_parameters,
        )

    def _default_prior_parameters(self):
        """Return priors matching the scalar no-intercept experiment."""
        return {
            'mu_X': np.asarray([80.0]),
            'sigma_X': np.asarray([0.1]),
            'mu_sigma_X': np.asarray([10.0]),
            'sigma_sigma_X': np.asarray([0.1]),
            'mu_A': 2.0,
            'sigma_A': 1.0,
            'mu_sigma_A': 1.0,
            'sigma_sigma_A': 0.1,
            'mu_Y': 0.0,
            'sigma_Y': 0.01,
            'mu_sigma_Y': 0.1,
            'sigma_sigma_Y': 0.01,
            'N_E_mu': 1.0,
            'N_E_sigma': 1.0,
            'N_b_mu': np.asarray([self.scm_constants['b']]),
            'N_b_sigma': np.zeros(1),
            'N_d_mu': np.asarray([self.scm_constants['d']]),
            'N_d_sigma': np.zeros(1),
        }

    def _default_predictive_var_names(self):
        """Return sampled and reconstructed population variables."""
        return [
            'alpha_mu_X', 'alpha_sigma_X', 'alpha_mu_A', 'alpha_sigma_A',
            'alpha_mu_Y', 'alpha_sigma_Y', 'N_E', 's', 'W',
        ]

    def _initialize_prior_parameters(self, prior_parameters):
        """Validate that the slope coefficients remain fixed constants."""
        parameters = super()._initialize_prior_parameters(prior_parameters)
        expected = {
            'N_b_mu': np.asarray([self.scm_constants['b']]),
            'N_b_sigma': np.zeros(1),
            'N_d_mu': np.asarray([self.scm_constants['d']]),
            'N_d_sigma': np.zeros(1),
        }
        for name, value in expected.items():
            if not np.array_equal(parameters[name], value):
                raise ValueError(
                    '{} must match the fixed structural coefficient.'
                    .format(name))
        return parameters

    def _sampled_predictive_var_names(self):
        """Return graph variables needed for prior predictive draws."""
        return [
            'alpha_mu_X', 'alpha_sigma_X', 'alpha_sigma_A',
            'alpha_sigma_Y', 'N_E_magnitude', 's', 'W',
        ]

    def _build_model(self):
        """Build the exact sign-marginalized slope-only likelihood."""
        model = pm.Model()
        with model:
            X_o_ = pm.Data('X_o_', np.zeros(1))
            pred_o_ = pm.Data('pred_o_', np.asarray(0.0))

            parameters = self._build_population_parameters()
            alpha_mu_X = parameters['alpha_mu_X'][0]
            alpha_sigma_X = parameters['alpha_sigma_X'][0]
            alpha_sigma_A = parameters['alpha_sigma_A']
            alpha_sigma_Y = parameters['alpha_sigma_Y']
            N_E_magnitude = parameters['N_E_magnitude']

            s = pm.Normal('s', mu=0.0, sigma=np.sqrt(self.plate_size))
            W = pm.ChiSquared('W', nu=self.plate_size - 1)
            S_x = pm.Deterministic(
                'S_x', self.plate_size * alpha_mu_X + alpha_sigma_X * s)
            XTX = pm.Deterministic(
                'XTX', S_x**2 / self.plate_size + alpha_sigma_X**2 * W)
            mean_factor = pm.Deterministic('mean_factor', S_x / XTX)

            signs = at.as_tensor_variable(np.asarray([1.0, -1.0]))
            N_E_branches = signs * N_E_magnitude
            X_o_scalar = X_o_[0]
            prediction_base = X_o_scalar * (
                self.scm_constants['b']
                + mean_factor * (
                    self.scm_constants['a']
                    + self.prior_parameters['mu_Y']))
            prediction_effect = X_o_scalar * (
                self.scm_constants['d']
                + mean_factor * self.prior_parameters['mu_A'])
            prediction_mu = pm.Deterministic(
                'prediction_mu',
                prediction_base + N_E_branches * prediction_effect)

            mean_variance = (
                (X_o_scalar * mean_factor)**2
                * (self.prior_parameters['sigma_Y']**2
                   + N_E_magnitude**2
                   * self.prior_parameters['sigma_A']**2))
            noise_variance = (
                N_E_magnitude**2 * alpha_sigma_A**2 + alpha_sigma_Y**2
            ) * X_o_scalar**2 / XTX
            prediction_variance = pm.Deterministic(
                'prediction_variance', mean_variance + noise_variance)
            prediction_sigma = pm.Deterministic(
                'prediction_sigma', at.sqrt(prediction_variance))

            N_E_prior = pm.Normal.dist(
                self.prior_parameters['N_E_mu'],
                self.prior_parameters['N_E_sigma'])
            log_prior_branches = at.stack([
                pm.logp(N_E_prior, N_E_magnitude),
                pm.logp(N_E_prior, -N_E_magnitude),
            ])
            log_folded_prior = at.logsumexp(log_prior_branches)
            log_sign_weights = log_prior_branches - log_folded_prior
            prediction_logps = pm.logp(
                pm.Normal.dist(mu=prediction_mu, sigma=prediction_sigma),
                pred_o_)
            log_signed_branches = log_sign_weights + prediction_logps
            prediction_logp = at.logsumexp(log_signed_branches)
            pm.Mixture(
                'prediction_obs',
                w=at.exp(log_sign_weights),
                comp_dists=pm.Normal.dist(
                    mu=prediction_mu, sigma=prediction_sigma),
                observed=pred_o_)
            positive_probability = pm.Deterministic(
                'N_E_positive_probability',
                at.exp(log_signed_branches[0] - prediction_logp))
            pm.Deterministic(
                'N_E_conditional_mean',
                (2.0 * positive_probability - 1.0) * N_E_magnitude)

            pm.Normal(
                'X_o', parameters['alpha_mu_X'],
                parameters['alpha_sigma_X'], observed=X_o_, shape=1)
        return model

    def sample(self, X_o, pred_o, draws=1000, tune=1000,
               target_accept=.85, chains=4, max_treedepth=15,
               init="jitter+adapt_diag", rng=None):
        """Sample after validating the scalar observed prediction."""
        X_o = np.asarray(X_o, dtype=float)
        pred_o = np.asarray(pred_o, dtype=float)
        if X_o.size != 1 or not np.all(np.isfinite(X_o)) or X_o.item() == 0:
            raise ValueError("X_o must be one finite nonzero scalar.")
        if pred_o.size != 1 or not np.all(np.isfinite(pred_o)):
            raise ValueError("pred_o must be one finite scalar prediction.")
        return super().sample(
            np.asarray([X_o.item()]), pred_o.item(), draws=draws, tune=tune,
            target_accept=target_accept, chains=chains,
            max_treedepth=max_treedepth, init=init, rng=rng)

    def _sample_conditional_parameters(self, parameters, N_E, rng):
        """Draw marginalized scalar means given one posterior branch."""
        alpha_mu_X = float(np.asarray(parameters['alpha_mu_X'])[0])
        alpha_sigma_X = float(np.asarray(parameters['alpha_sigma_X'])[0])
        S_x = (self.plate_size * alpha_mu_X
               + alpha_sigma_X * parameters['s'])
        XTX = (S_x**2 / self.plate_size
               + alpha_sigma_X**2 * parameters['W'])
        mean_factor = S_x / XTX
        X_o = float(self.X_o[0])

        prior_mean = np.asarray([
            self.prior_parameters['mu_A'],
            self.prior_parameters['mu_Y'],
            self.scm_constants['b'],
            self.scm_constants['d'],
        ])
        prior_variance = np.asarray([
            self.prior_parameters['sigma_A']**2,
            self.prior_parameters['sigma_Y']**2,
            0.0,
            0.0,
        ])
        observation_vector = np.asarray([
            N_E * X_o * mean_factor,
            X_o * mean_factor,
            X_o,
            N_E * X_o,
        ])
        observation_offset = self.scm_constants['a'] * X_o * mean_factor
        noise_variance = (
            N_E**2 * parameters['alpha_sigma_A']**2
            + parameters['alpha_sigma_Y']**2
        ) * X_o**2 / XTX

        prior_draw = rng.normal(prior_mean, np.sqrt(prior_variance))
        prediction_draw = (
            observation_offset + observation_vector @ prior_draw
            + rng.normal(0.0, np.sqrt(noise_variance)))
        denominator = (
            observation_vector @ (prior_variance * observation_vector)
            + noise_variance)
        gain = prior_variance * observation_vector / denominator
        conditional_draw = prior_draw + gain * (
            self.pred_o - prediction_draw)
        return (
            float(conditional_draw[0]),
            float(conditional_draw[1]),
            conditional_draw[2:3],
            conditional_draw[3:4],
        )

    def _reconstruct_posterior_parameters(self, rng):
        """Attach signed and Gaussian reconstructions to the trace."""
        posterior = self.trace.posterior
        magnitudes = posterior['N_E_magnitude'].values
        positive_probabilities = posterior[
            'N_E_positive_probability'].values
        signs = np.where(
            rng.random(magnitudes.shape) < positive_probabilities, 1.0, -1.0)
        N_E = signs * magnitudes

        scalar_names = ('alpha_sigma_A', 'alpha_sigma_Y', 's', 'W')
        vector_names = ('alpha_mu_X', 'alpha_sigma_X')
        scalar_values = {
            name: posterior[name].values.reshape(-1)
            for name in scalar_names
        }
        vector_values = {
            name: posterior[name].values.reshape((-1, 1))
            for name in vector_names
        }

        alpha_mu_A = np.empty(N_E.size)
        alpha_mu_Y = np.empty(N_E.size)
        coefficient_draws = []
        for index, signed_effect in enumerate(N_E.reshape(-1)):
            parameters = {
                name: values[index]
                for name, values in scalar_values.items()
            }
            parameters.update({
                name: values[index]
                for name, values in vector_values.items()
            })
            reconstructed = self._sample_conditional_parameters(
                parameters, signed_effect, rng)
            alpha_mu_A[index], alpha_mu_Y[index], N_b, N_d = reconstructed
            coefficient_draws.append((N_b, N_d))

        scalar_dims = posterior['N_E_magnitude'].dims
        scalar_shape = magnitudes.shape
        self.trace.posterior = posterior.assign({
            'N_E': (scalar_dims, N_E),
            'alpha_mu_A': (scalar_dims, alpha_mu_A.reshape(scalar_shape)),
            'alpha_mu_Y': (scalar_dims, alpha_mu_Y.reshape(scalar_shape)),
        })
        self.posterior_sign_probabilities = positive_probabilities
        self.posterior_coefficient_draws = coefficient_draws

    def _sample_conditional_coefficients(self, parameters, rng):
        """Return the fixed scalar structural coefficients."""
        del parameters, rng
        return (
            np.asarray([self.scm_constants['b']]),
            np.asarray([self.scm_constants['d']]),
        )

    def _sample_prior_coefficients(self, rng):
        """Return the fixed scalar structural coefficients."""
        del rng
        return (
            np.asarray([self.scm_constants['b']]),
            np.asarray([self.scm_constants['d']]),
        )

    def _update_predictive_scm(self, parameters, N_b, N_d):
        """Update the predictive SCM while retaining its fixed intercept."""
        adjusted_parameters = parameters.copy()
        adjusted_parameters['alpha_mu_Y'] = (
            adjusted_parameters['alpha_mu_Y'] + self.scm_constants['a'])
        scm_parameters = {
            name: adjusted_parameters[name]
            for name in (
                'alpha_mu_X', 'alpha_sigma_X', 'alpha_mu_A',
                'alpha_sigma_A', 'alpha_mu_Y', 'alpha_sigma_Y', 'N_E',
            )
        }
        scm_parameters['N_b'] = N_b
        scm_parameters['N_d'] = N_d
        self.predictive_scm.update_parameters(scm_parameters)

    def _reload_kwargs(self):
        """Return constructor arguments used when loading a saved model."""
        return {
            'plate_size': self.plate_size,
            'prior_parameters': self.prior_parameters,
            'scm_constants': self.scm_constants,
        }
