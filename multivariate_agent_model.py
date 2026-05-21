"""
Multivariate Bayesian agent model.

The model uses reduced historical-plate representations:

    (s, W, g, h) for OLS with intercept
    (s, W, g) for OLS without intercept
"""

from pathlib import Path

import arviz as az
import joblib
import numpy as np
import pymc as pm
import pytensor.tensor as at
from scipy.stats import kurtosis, skew

from multivariate_scm import MultivariateSCM


DEBUGGING = False


def printing_callback(trace, draw):
    """
    Function for debugging pymc-sampling.
    """
    if DEBUGGING:
        print(f"Completed draw {draw.draw_idx} from chain {draw.chain}.")


class MultivariateAgentModel:
    """
    Bayesian agent model with k-dimensional covariates.

    If fit_intercept is True, the model uses the reduced variables
    (s, W, g, h). If fit_intercept is False, it uses (s, W, g).
    """
    def __init__(self, n_covariates, plate_size, prior_parameters=None,
                 predictive_var_names=None, fit_intercept=True):
        self.n_covariates = int(n_covariates)
        if self.n_covariates < 1:
            raise ValueError("n_covariates must be positive.")
        self.plate_size = int(plate_size)
        if self.plate_size <= self.n_covariates:
            raise ValueError("plate_size must be larger than n_covariates.")
        self.fit_intercept = fit_intercept

        self.prior_parameters = self._initialize_prior_parameters(prior_parameters)
        self.predictive_var_names = self._initialize_predictive_var_names(
            predictive_var_names)

        self.predictive_scm = MultivariateSCM(
            n_covariates=self.n_covariates,
            alpha_mu_X=self.prior_parameters['mu_X'],
            alpha_sigma_X=self.prior_parameters['mu_sigma_X'],
            alpha_mu_A=self.prior_parameters['mu_A'],
            alpha_sigma_A=self.prior_parameters['mu_sigma_A'],
            alpha_mu_Y=self.prior_parameters['mu_Y'],
            alpha_sigma_Y=self.prior_parameters['mu_sigma_Y'],
            N_E=self.prior_parameters['N_E_mu'],
            N_b=self.prior_parameters['N_b_mu'],
            N_d=self.prior_parameters['N_d_mu'],
        )

        self.trace = None
        self.prior_samples = None
        self.prior_predictive_samples = None
        self.posterior_predictive_samples = None
        self.posterior_draws = None
        self.prior_draws = None
        self.X_o = None

        self.model = self._build_model()

    def _normalize_vector_parameter(self, parameter, parameter_name):
        """
        Convert scalar or one-dimensional input to a vector of length k.
        """
        parameter = np.atleast_1d(np.asarray(parameter, dtype=float))
        if parameter.ndim != 1:
            raise ValueError("{} must be scalar or one-dimensional.".format(parameter_name))
        if parameter.size == 1:
            parameter = np.full(self.n_covariates, parameter.item(), dtype=float)
        elif parameter.size != self.n_covariates:
            text = "{} must have length {}, not {}.".format(
                parameter_name, self.n_covariates, parameter.size)
            raise ValueError(text)
        return parameter

    def _default_prior_parameters(self):
        """
        Return default prior parameters for the multivariate model.
        """
        return {
            'mu_X': np.full(self.n_covariates, 80.0),
            'sigma_X': np.full(self.n_covariates, 0.1),
            'mu_sigma_X': np.full(self.n_covariates, 10.0),
            'sigma_sigma_X': np.full(self.n_covariates, 0.1),
            'mu_A': 2.0,
            'sigma_A': 1.0,
            'mu_sigma_A': 1.0,
            'sigma_sigma_A': 0.1,
            'mu_Y': 12.0,
            'sigma_Y': 0.01,
            'mu_sigma_Y': 0.1,
            'sigma_sigma_Y': 0.01,
            'N_E_mu': 1.0,
            'N_E_sigma': 1.0,
            'N_b_mu': np.full(self.n_covariates, -0.1 / self.n_covariates),
            'N_b_sigma': np.full(self.n_covariates, 0.1),
            'N_d_mu': np.full(self.n_covariates, 0.125 / self.n_covariates),
            'N_d_sigma': np.full(self.n_covariates, 0.1),
        }

    def _initialize_prior_parameters(self, prior_parameters):
        """
        Validate or create the prior-parameter dictionary.
        """
        default_prior_parameters = self._default_prior_parameters()
        if prior_parameters is None:
            prior_parameters = default_prior_parameters
        else:
            extra_items = [item for item in prior_parameters.keys()
                           if item not in default_prior_parameters.keys()]
            err_msg = "{} passed with prior_parameters, " \
                "but not part of model specification.".format(extra_items)
            assert len(extra_items) < 1, err_msg
            tmp = default_prior_parameters.copy()
            tmp.update(prior_parameters)
            prior_parameters = tmp

        prior_parameters = prior_parameters.copy()
        for parameter_name in ('mu_X', 'sigma_X', 'mu_sigma_X',
                               'sigma_sigma_X', 'N_b_mu', 'N_b_sigma',
                               'N_d_mu', 'N_d_sigma'):
            prior_parameters[parameter_name] = self._normalize_vector_parameter(
                prior_parameters[parameter_name], parameter_name)
        return prior_parameters

    def _default_predictive_var_names(self):
        """
        Return variables sampled for predictive calculations.
        """
        return ['alpha_mu_X', 'alpha_sigma_X', 'alpha_mu_A', 'alpha_sigma_A',
                'alpha_mu_Y', 'alpha_sigma_Y', 'N_E', 'N_b', 'N_d']

    def _initialize_predictive_var_names(self, predictive_var_names):
        """
        Validate or create the predictive variable-name list.
        """
        default_predictive_var_names = self._default_predictive_var_names()
        if predictive_var_names is None:
            return default_predictive_var_names
        extra_items = [item for item in predictive_var_names
                       if item not in default_predictive_var_names]
        err_msg = "{} passed with predictive_var_names, " \
            "but not part of model specification.".format(extra_items)
        assert len(extra_items) < 1, err_msg
        return predictive_var_names

    def _build_population_parameters(self):
        """
        Build PyMC variables outside the historical plate.
        """
        alpha_mu_X = pm.Normal('alpha_mu_X', self.prior_parameters['mu_X'],
                               self.prior_parameters['sigma_X'],
                               shape=self.n_covariates)
        alpha_sigma_X = pm.TruncatedNormal(
            'alpha_sigma_X',
            mu=self.prior_parameters['mu_sigma_X'],
            sigma=self.prior_parameters['sigma_sigma_X'],
            lower=0,
            shape=self.n_covariates)
        alpha_mu_A = pm.Normal('alpha_mu_A', self.prior_parameters['mu_A'],
                               self.prior_parameters['sigma_A'])
        alpha_sigma_A = pm.TruncatedNormal(
            'alpha_sigma_A',
            mu=self.prior_parameters['mu_sigma_A'],
            sigma=self.prior_parameters['sigma_sigma_A'],
            lower=0)
        alpha_mu_Y = pm.Normal('alpha_mu_Y', self.prior_parameters['mu_Y'],
                               self.prior_parameters['sigma_Y'])
        alpha_sigma_Y = pm.TruncatedNormal(
            'alpha_sigma_Y',
            mu=self.prior_parameters['mu_sigma_Y'],
            sigma=self.prior_parameters['sigma_sigma_Y'],
            lower=0)
        N_E = pm.Normal('N_E', self.prior_parameters['N_E_mu'],
                        self.prior_parameters['N_E_sigma'])
        N_b = pm.Normal('N_b', self.prior_parameters['N_b_mu'],
                        self.prior_parameters['N_b_sigma'],
                        shape=self.n_covariates)
        N_d = pm.Normal('N_d', self.prior_parameters['N_d_mu'],
                        self.prior_parameters['N_d_sigma'],
                        shape=self.n_covariates)
        return {
            'alpha_mu_X': alpha_mu_X,
            'alpha_sigma_X': alpha_sigma_X,
            'alpha_mu_A': alpha_mu_A,
            'alpha_sigma_A': alpha_sigma_A,
            'alpha_mu_Y': alpha_mu_Y,
            'alpha_sigma_Y': alpha_sigma_Y,
            'N_E': N_E,
            'N_b': N_b,
            'N_d': N_d,
        }

    def _build_model(self):
        """
        Build the PyMC model.
        """
        model = pm.Model()
        with model:
            X_o_ = pm.Data('X_o_', np.zeros(self.n_covariates))
            pred_o_ = pm.Data('pred_o_', np.asarray(0.0))

            parameters = self._build_population_parameters()
            alpha_mu_X = parameters['alpha_mu_X']
            alpha_sigma_X = parameters['alpha_sigma_X']
            alpha_mu_A = parameters['alpha_mu_A']
            alpha_sigma_A = parameters['alpha_sigma_A']
            alpha_mu_Y = parameters['alpha_mu_Y']
            alpha_sigma_Y = parameters['alpha_sigma_Y']
            N_E = parameters['N_E']
            N_b = parameters['N_b']
            N_d = parameters['N_d']

            s = pm.Normal('s', mu=0, sigma=at.sqrt(self.plate_size),
                          shape=self.n_covariates)
            W_chol = pm.WishartBartlett(
                'W_chol',
                S=np.eye(self.n_covariates),
                nu=self.plate_size - 1,
                return_cholesky=True)
            g = pm.Normal('g', mu=0, sigma=1, shape=self.n_covariates)

            beta = pm.Deterministic('beta', N_b + N_E * N_d)
            c = pm.Deterministic('c', alpha_mu_Y + N_E * alpha_mu_A)
            tau = pm.Deterministic(
                'tau',
                at.sqrt(N_E**2 * alpha_sigma_A**2 + alpha_sigma_Y**2))

            if self.fit_intercept:
                h = pm.Normal('h', mu=0, sigma=1)
                W_minus_half_g = at.slinalg.solve_triangular(
                    W_chol.T, g, lower=False)
                phi = pm.Deterministic(
                    'phi',
                    beta + tau * W_minus_half_g / alpha_sigma_X)
                alpha = pm.Deterministic(
                    'alpha',
                    c + tau * h / at.sqrt(self.plate_size) -
                    tau * at.dot(alpha_mu_X / alpha_sigma_X +
                                 s / self.plate_size,
                                 W_minus_half_g))
                pred_reconstructed = alpha + at.dot(X_o_, phi)
            else:
                W = at.dot(W_chol, W_chol.T)
                S_x = pm.Deterministic(
                    'S_x',
                    self.plate_size * alpha_mu_X + alpha_sigma_X * s)
                XTX = pm.Deterministic(
                    'XTX',
                    at.outer(S_x, S_x) / self.plate_size +
                    alpha_sigma_X[:, None] * W * alpha_sigma_X[None, :])
                XTX_chol = at.linalg.cholesky(XTX)
                mean_term = c * at.linalg.solve(XTX, S_x)
                noise_term = tau * at.slinalg.solve_triangular(
                    XTX_chol.T, g, lower=False)
                phi = pm.Deterministic('phi', beta + mean_term + noise_term)
                pred_reconstructed = at.dot(X_o_, phi)

            pm.Normal('X_o', alpha_mu_X, alpha_sigma_X, observed=X_o_,
                      shape=self.n_covariates)
            pm.Normal('prediction_obs', pred_reconstructed, 1e-3,
                      observed=pred_o_)
        return model

    def sample(self, X_o, pred_o, draws=1000, tune=1000, target_accept=.85,
               chains=4, max_treedepth=15, init="jitter+adapt_diag", rng=None):
        """
        Sample from the posterior given an observed patient and prediction.
        """
        X_o = self._normalize_vector_parameter(X_o, 'X_o')
        self.posterior_draws = draws
        pm.set_data({'X_o_': X_o, 'pred_o_': np.asarray(pred_o)},
                    model=self.model)
        with self.model:
            trace = pm.sample(draws=self.posterior_draws,
                              tune=tune,
                              target_accept=target_accept,
                              chains=chains,
                              random_seed=rng,
                              max_treedepth=max_treedepth,
                              init=init,
                              callback=printing_callback)
        self.trace = trace
        self.X_o = X_o
        return

    def _parameter_draws(self, sample_group):
        """
        Convert sampled PyMC variables into one dictionary per draw.
        """
        first_var = self.predictive_var_names[0]
        first_values = sample_group[first_var].values
        n_draws = int(np.prod(first_values.shape[:2]))
        parameter_values = {}
        for var_name in self.predictive_var_names:
            values = sample_group[var_name].values
            parameter_values[var_name] = values.reshape(
                (n_draws,) + values.shape[2:])

        draws = []
        for draw_index in range(n_draws):
            parameters = {}
            for var_name, values in parameter_values.items():
                value = values[draw_index]
                if np.asarray(value).shape == ():
                    value = float(value)
                parameters[var_name] = value
            draws.append(parameters)
        return draws

    def sample_prior_predictive(self, draws=1000, rng=None):
        """
        Sample from the distribution based on the agent's prior beliefs.
        """
        with self.model:
            tmp = pm.sample_prior_predictive(
                draws=draws,
                var_names=self.predictive_var_names,
                random_seed=rng)
        self.prior_samples = tmp
        parameter_draws = self._parameter_draws(tmp.prior)
        self.prior_draws = len(parameter_draws)
        samples = []
        for parameters in parameter_draws:
            self.predictive_scm.update_parameters(parameters)
            X, A, Y, *_ = self.predictive_scm.sample(rng=rng)
            samples.append([X, A, Y])
        self.prior_predictive_samples = samples
        return samples

    def _ensure_prior_samples(self, rng=None):
        """
        Draw prior samples if needed by a predictive method.
        """
        if self.prior_samples is None:
            self.sample_prior_predictive(rng=rng)

    def sample_prior_predictive_do_A(self, X, A_treatment, rng=None):
        """
        Estimate outcomes under do(A) from the agent's prior beliefs.
        """
        self._ensure_prior_samples(rng=rng)
        X = self._normalize_vector_parameter(X, 'X')
        do_A_samples = np.zeros(shape=(self.prior_draws,))
        for i, parameters in enumerate(self._parameter_draws(self.prior_samples.prior)):
            self.predictive_scm.update_parameters(parameters)
            tmp = self.predictive_scm.sample_do_A(A_treatment, X, rng=rng)
            do_A_samples[i] = tmp[0][0]
        do_A_mean = np.mean(do_A_samples)
        do_A_std = np.std(do_A_samples)
        return do_A_samples, do_A_mean, do_A_std

    def sample_prior_predictive_CATE(self, X, A1_treatment, A0_treatment, rng=None):
        """
        Sample the prior CATE distribution.
        """
        self._ensure_prior_samples(rng=rng)
        X = self._normalize_vector_parameter(X, 'X')
        CATE_samples = np.zeros(shape=(self.prior_draws,))
        for i, parameters in enumerate(self._parameter_draws(self.prior_samples.prior)):
            self.predictive_scm.update_parameters(parameters)
            tmp = self.predictive_scm.sample_CATE(
                X, A1_treatment, A0_treatment, rng=rng)
            CATE_samples[i] = tmp[0][0]
        CATE_mean = np.mean(CATE_samples)
        CATE_std = np.std(CATE_samples)
        return CATE_samples, CATE_mean, CATE_std

    def get_prior_CATE_mu(self, A1_treatment, A0_treatment):
        """
        Return the prior mean CATE implied by the agent's N_E prior.
        """
        return (A1_treatment - A0_treatment) * self.prior_parameters['N_E_mu']

    def sample_prior_predictive_summaries(self, rng=None):
        """
        Estimate moments of X, A, and Y in prior predictive samples.
        """
        self._ensure_prior_samples(rng=rng)
        tmp_X = [item[0] for item in self.prior_predictive_samples]
        tmp_A = [item[1] for item in self.prior_predictive_samples]
        tmp_Y = [item[2] for item in self.prior_predictive_samples]
        results = {}
        results['pri_X_mean'] = np.mean(tmp_X)
        results['pri_A_mean'] = np.mean(tmp_A)
        results['pri_Y_mean'] = np.mean(tmp_Y)
        results['pri_X_std'] = np.std(tmp_X)
        results['pri_A_std'] = np.std(tmp_A)
        results['pri_Y_std'] = np.std(tmp_Y)
        results['pri_X_skew'] = skew(np.asarray(tmp_X).reshape(-1))
        results['pri_A_skew'] = skew(tmp_A)[0]
        results['pri_Y_skew'] = skew(tmp_Y)[0]
        results['pri_X_kurtosis'] = kurtosis(np.asarray(tmp_X).reshape(-1))
        results['pri_A_kurtosis'] = kurtosis(tmp_A)[0]
        results['pri_Y_kurtosis'] = kurtosis(tmp_Y)[0]
        return results

    def sample_posterior_predictive(self, rng=None):
        """
        Sample from the agent's posterior beliefs.
        """
        parameter_draws = self._parameter_draws(self.trace.posterior)
        self.posterior_draws = len(parameter_draws)
        samples = []
        for parameters in parameter_draws:
            self.predictive_scm.update_parameters(parameters)
            samples.append(self.predictive_scm.sample(rng=rng))
        self.posterior_predictive_samples = samples
        return samples

    def sample_posterior_predictive_CATE(self, X, A1_treatment, A0_treatment,
                                         rng=None):
        """
        Sample the posterior CATE distribution.
        """
        X = self._normalize_vector_parameter(X, 'X')
        parameter_draws = self._parameter_draws(self.trace.posterior)
        CATE_samples = np.zeros(shape=(len(parameter_draws),))
        for i, parameters in enumerate(parameter_draws):
            self.predictive_scm.update_parameters(parameters)
            tmp = self.predictive_scm.sample_CATE(
                X, A1_treatment, A0_treatment, rng=rng)
            CATE_samples[i] = tmp[0][0]
        CATE_mean = np.mean(CATE_samples)
        CATE_std = np.std(CATE_samples)
        return CATE_samples, CATE_mean, CATE_std

    def sample_posterior_predictive_do_A(self, X, A_treatment, rng=None):
        """
        Estimate outcomes under do(A) from the agent's posterior beliefs.
        """
        X = self._normalize_vector_parameter(X, 'X')
        parameter_draws = self._parameter_draws(self.trace.posterior)
        potential_outcomes = np.zeros(shape=(len(parameter_draws),))
        for i, parameters in enumerate(parameter_draws):
            self.predictive_scm.update_parameters(parameters)
            tmp = self.predictive_scm.sample_do_A(A_treatment, X, rng=rng)
            potential_outcomes[i] = tmp[0][0]
        do_A_mean = np.mean(potential_outcomes)
        do_A_std = np.std(potential_outcomes)
        return potential_outcomes, do_A_mean, do_A_std

    def sample_posterior_predictive_summaries(self, rng=None):
        """
        Estimate moments of X, A, and Y in posterior predictive samples.
        """
        if self.posterior_predictive_samples is None:
            self.sample_posterior_predictive(rng=rng)
        tmp_X = [item[0] for item in self.posterior_predictive_samples]
        tmp_A = [item[1] for item in self.posterior_predictive_samples]
        tmp_Y = [item[2] for item in self.posterior_predictive_samples]
        results = {}
        results['post_X_mean'] = np.mean(tmp_X)
        results['post_A_mean'] = np.mean(tmp_A)
        results['post_Y_mean'] = np.mean(tmp_Y)
        results['post_X_std'] = np.std(tmp_X)
        results['post_A_std'] = np.std(tmp_A)
        results['post_Y_std'] = np.std(tmp_Y)
        results['post_X_skew'] = skew(np.asarray(tmp_X).reshape(-1))
        results['post_A_skew'] = skew(tmp_A)[0]
        results['post_Y_skew'] = skew(tmp_Y)[0]
        results['post_X_kurtosis'] = kurtosis(np.asarray(tmp_X).reshape(-1))
        results['post_A_kurtosis'] = kurtosis(tmp_A)[0]
        results['post_Y_kurtosis'] = kurtosis(tmp_Y)[0]
        return results

    def trace_diagnostics(self):
        """
        Return basic sampler diagnostics for the current posterior trace.
        """
        rhat_values = az.rhat(self.trace)
        ess_bulk_values = az.ess(self.trace, method='bulk')
        ess_tail_values = az.ess(self.trace, method='tail')
        return {
            'max_rhat': float(rhat_values.to_array().max().values),
            'min_ess_bulk': float(ess_bulk_values.to_array().min().values),
            'min_ess_tail': float(ess_tail_values.to_array().min().values),
            'n_divergences': int(self.trace.sample_stats['diverging'].sum()),
        }

    def save_agent_model(self, file_name='multivariate_agent_model',
                         folder='./saved_models/'):
        """
        Save the agent model configuration and available samples.
        """
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        file_path = folder / file_name

        if self.trace is not None:
            self.trace.to_netcdf(f'{file_path}_trace.nc')
            print(f"Trace saved to {file_path}_trace.nc")
        if self.prior_samples is not None:
            self.prior_samples.to_netcdf(f'{file_path}_prior_samples.nc')
            print(f"Prior samples saved to {file_path}_prior_samples.nc")

        trace_backup = self.trace
        prior_samples_backup = self.prior_samples
        model_backup = self.model
        self.trace = None
        self.prior_samples = None
        self.model = None

        joblib.dump(self, f'{file_path}.joblib')

        self.trace = trace_backup
        self.prior_samples = prior_samples_backup
        self.model = model_backup

        print(f"Complete agent model saved to {file_path}.joblib")

    @classmethod
    def load_agent_model(cls, file_name='multivariate_agent_model',
                         folder='./saved_models/'):
        """
        Load a saved multivariate agent model.
        """
        file_path = Path(folder) / file_name
        agent = joblib.load(f'{file_path}.joblib')
        print("Agent model loaded from {}".format(file_path))

        posterior_draws = agent.posterior_draws
        prior_draws = agent.prior_draws
        X_o = agent.X_o

        agent.__init__(
            n_covariates=agent.n_covariates,
            plate_size=agent.plate_size,
            prior_parameters=agent.prior_parameters,
            predictive_var_names=agent.predictive_var_names,
            fit_intercept=agent.fit_intercept,
        )

        agent.posterior_draws = posterior_draws
        agent.prior_draws = prior_draws
        agent.X_o = X_o

        try:
            agent.trace = az.from_netcdf(f'{file_path}_trace.nc')
        except FileNotFoundError:
            agent.trace = None
            print("Trace not loaded (file {}_trace.nc does not exist).".format(
                file_path))

        try:
            agent.prior_samples = az.from_netcdf(
                f'{file_path}_prior_samples.nc')
        except FileNotFoundError:
            agent.prior_samples = None
            print("Prior samples not loaded (file "
                  "{}_prior_samples.nc does not exist).".format(file_path))
        return agent
