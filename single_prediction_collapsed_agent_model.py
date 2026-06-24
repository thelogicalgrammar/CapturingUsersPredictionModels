"""
Single-prediction collapsed Bayesian agent model.

This is a further reduction of :class:`agent_model.MultivariateAgentModel`
for the case where the agent observes a single scalar prediction ``pred_o`` for
a single observed feature vector ``X_o``. Under independent Gaussian priors on
the coordinates of the coefficient vectors N_b and N_d, those vectors are
marginalized analytically, so the only non-auxiliary latent variables left are
N_E and the population-level parameters.

The reduced-plate auxiliary variables (s, W, g, h) of the multivariate model are
replaced by two scalar auxiliaries (R, T), constant in both the plate size n and
the number of covariates k. Conditional on (R, T, N_E, the population-level
parameters, and X_o), the observed prediction is Gaussian: \n
:math:`Pred^o \\sim \\mathcal{N}(\\mu_{red}, V_{red})` \n
with \n
:math:`\\mu_{red} = c + (m_b + N_E m_d) S_o` \n
:math:`V_{red} = Q_o (v_b + 2 N_E v_{bd} + N_E^2 v_d) + \\tau^2 (1/n + R/T)`, \n
where the cross term vanishes (:math:`v_{bd} = 0`) under the independent
coefficient priors assumed by this class. Because PyMC does not provide a
native non-central chi-squared distribution, R
is built through the equivalent central-chi-squared (Cochran) construction \n
:math:`G_R \\sim \\mathcal{N}(0, 1)`,
:math:`Z_R \\sim \\chi^2_{k-1}`,
:math:`R := (\\Delta_X + G_R / \\sqrt{n})^2 + Z_R / n`, \n
with :math:`\\Delta_X^2 := (X^o - \\vec{\\alpha}_{X\\mu})^\\top D_X^{-2}
(X^o - \\vec{\\alpha}_{X\\mu})` and the convention :math:`\\chi^2_0 = 0` when
k = 1. The orthogonal rotation behind Cochran's theorem concentrates the
non-centrality in a single coordinate, so the model samples three scalar
primitive variables (G_R, Z_R, T) regardless of k and n.

Since the CATE depends only on N_E, marginalizing N_b and N_d loses no
information relevant to the decision rule. For the agent-side predictive
quantities (predictive summaries of X, A, and Y, and expected do(A) outcomes)
the marginalized coefficient vectors are recovered when needed: from their
priors for prior predictive sampling, and from their conditional Gaussian
given (N_E, R, T, the population-level parameters, and pred_o) for posterior
predictive sampling. The derivation is in the "Single-prediction collapsed
reduction" document of the paper derivations.
"""

from pathlib import Path

import arviz as az
import joblib
import numpy as np
import pymc as pm
import pytensor.tensor as at
from scipy.stats import kurtosis, skew

import scm


DEBUGGING = False


def printing_callback(trace, draw):
    """
    Function for debugging pymc-sampling.
    """
    if DEBUGGING:
        print(f"Completed draw {draw.draw_idx} from chain {draw.chain}.")


class SinglePredictionCollapsedAgentModel:
    """
    Bayesian agent model for the single-prediction collapsed likelihood.

    The class assumes independent Gaussian priors for the coordinates of N_b
    and N_d. These vectors are marginalized analytically and are not sampled by
    PyMC, so the latent variables of the model are the population-level
    parameters, N_E, and the two scalar auxiliaries (R, T) of the collapsed
    likelihood.

    The model always assumes an OLS predictive model with intercept, and is
    specific to a single observed prediction: it does not extend, in this form,
    to more than one prediction. Because the CATE depends only on N_E, the
    prior and posterior CATE are obtained directly from N_E. The agent-side
    predictive summaries and expected-outcome helpers of the multivariate model
    are still provided: the marginalized coefficient vectors are recovered from
    their priors for prior predictive sampling, and from their conditional
    Gaussian given (N_E, R, T, the population-level parameters, and pred_o)
    for posterior predictive sampling.

    Args:
        n_covariates (int): Number of covariates k (dimension of X).
        plate_size (int): Size of the plate (number of i.i.d. samples used to
            train the predictive model). Must be larger than n_covariates.
        prior_parameters (dict): Dictionary with parameters for the prior
            distributions. If None, defaults are used. Vector-valued entries
            accept a scalar (broadcast to length k) or an array of length k.

    Returns:
        None
    """
    def __init__(self, n_covariates, plate_size, prior_parameters=None):
        self.n_covariates = int(n_covariates)
        if self.n_covariates < 1:
            raise ValueError("n_covariates must be positive.")
        self.plate_size = int(plate_size)
        if self.plate_size <= self.n_covariates:
            raise ValueError("plate_size must be larger than n_covariates.")

        self.prior_parameters = self._initialize_prior_parameters(prior_parameters)
        self.predictive_var_names = self._default_predictive_var_names()

        self.predictive_scm = scm.MultivariateSCM(
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
        self.prior_coefficient_draws = None
        self.posterior_coefficient_draws = None
        self.posterior_draws = None
        self.prior_draws = None
        self.X_o = None
        self.pred_o = None

        self.model = self._build_model()

    def _normalize_vector_parameter(self, parameter, parameter_name):
        """
        Convert scalar or one-dimensional input to a vector of length k.

        A scalar is broadcast to a constant vector of length k. A
        one-dimensional array must already have length k.

        Args:
            parameter (float or array-like): Value to normalize.
            parameter_name (str): Name used in error messages.

        Returns:
            np.ndarray: One-dimensional array of length k.
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
        Return default prior parameters for the collapsed model.

        Vector-valued entries are length-k arrays. The coefficient means
        ``N_b_mu`` and ``N_d_mu`` are scaled by ``1 / k`` so the aggregate
        signal stays close to the scalar case. The coefficient standard
        deviations enter the collapsed likelihood variance directly.

        Returns:
            dict: Default prior-parameter dictionary.
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

        If ``prior_parameters`` is None the defaults are used. Otherwise the
        passed keys are validated against the model specification, merged on
        top of the defaults, and the vector-valued entries are normalized to
        length-k arrays.

        Args:
            prior_parameters (dict or None): User-supplied prior parameters.

        Returns:
            dict: Validated and normalized prior-parameter dictionary.
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
        Return the names of the variables sampled for predictive calculations.

        These are the population-level SCM parameters whose sampled values
        replace the parameters of the internal predictive SCM during prior and
        posterior predictive sampling, plus the collapsed auxiliaries R and T,
        which are needed to reconstruct the marginalized coefficient vectors
        in posterior predictive sampling.

        Returns:
            list: Default predictive variable names.
        """
        return ['alpha_mu_X', 'alpha_sigma_X', 'alpha_mu_A', 'alpha_sigma_A',
                'alpha_mu_Y', 'alpha_sigma_Y', 'N_E', 'R', 'T']

    def _build_population_parameters(self):
        """
        Build the PyMC variables that live outside the historical plate.

        These are the population-level SCM parameters the agent has priors on:
        the covariate parameters (vector-valued alpha_mu_X, alpha_sigma_X), the
        treatment and outcome parameters (alpha_mu_A, alpha_sigma_A, alpha_mu_Y,
        alpha_sigma_Y), and the effect size N_E. The coefficient vectors N_b and
        N_d are not built here because they are marginalized analytically.

        Returns:
            dict: Mapping from parameter name to its PyMC random variable.
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
        return {
            'alpha_mu_X': alpha_mu_X,
            'alpha_sigma_X': alpha_sigma_X,
            'alpha_mu_A': alpha_mu_A,
            'alpha_sigma_A': alpha_sigma_A,
            'alpha_mu_Y': alpha_mu_Y,
            'alpha_sigma_Y': alpha_sigma_Y,
            'N_E': N_E,
        }

    def _build_model(self):
        """
        Build the PyMC model.

        The observed feature vector ``X_o`` and the observed prediction
        ``pred_o`` enter as :class:`pm.Data` placeholders, updated at sampling
        time. The plate is reduced to the two scalar auxiliaries ``R`` and
        ``T``: since PyMC has no native non-central chi-squared, ``R`` is built
        through the central-chi-squared (Cochran) construction \n
        :math:`R = (\\Delta_X + G_R / \\sqrt{n})^2 + Z_R / n` \n
        with ``G_R ~ N(0, 1)``, ``Z_R ~ ChiSquared(k - 1)`` (the convention
        ``ChiSquared(0) = 0`` applies when k = 1), and ``T`` is a centered
        chi-squared with ``plate_size - n_covariates`` degrees of freedom. The
        model therefore samples three scalar primitive variables (G_R, Z_R, T)
        regardless of k and n. The collapsed mean ``prediction_mu`` and
        variance ``prediction_variance`` are formed from the marginalized
        coefficient priors and the auxiliaries, and the observed prediction is
        given the likelihood ``N(prediction_mu, prediction_sigma)``. ``X_o`` is
        also given the covariate likelihood ``N(alpha_mu_X, alpha_sigma_X)``.

        Returns:
            pm.Model: The constructed PyMC model.
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

            Delta_X = pm.Deterministic(
                'Delta_X',
                at.sqrt(at.sum(((X_o_ - alpha_mu_X) / alpha_sigma_X)**2)))
            G_R = pm.Normal('G_R', mu=0, sigma=1)
            if self.n_covariates > 1:
                Z_R = pm.ChiSquared('Z_R', nu=self.n_covariates - 1)
            else:
                # Convention ChiSquared(0) = 0 for k = 1: all non-centrality
                # lives in the single rotated coordinate.
                Z_R = 0.0
            R = pm.Deterministic(
                'R',
                (Delta_X + G_R / np.sqrt(self.plate_size))**2 +
                Z_R / self.plate_size)
            T = pm.ChiSquared('T', nu=self.plate_size - self.n_covariates)

            beta_mu = self.prior_parameters['N_b_mu'] + (
                N_E * self.prior_parameters['N_d_mu'])
            coefficient_variance = at.sum((X_o_ * self.prior_parameters['N_b_sigma'])**2)
            coefficient_variance += N_E**2 * at.sum(
                (X_o_ * self.prior_parameters['N_d_sigma'])**2)

            c = pm.Deterministic('c', alpha_mu_Y + N_E * alpha_mu_A)
            tau = pm.Deterministic(
                'tau',
                at.sqrt(N_E**2 * alpha_sigma_A**2 + alpha_sigma_Y**2))
            prediction_mu = pm.Deterministic(
                'prediction_mu',
                c + at.dot(X_o_, beta_mu))
            prediction_variance = pm.Deterministic(
                'prediction_variance',
                coefficient_variance + tau**2 * (1.0 / self.plate_size + R / T))
            prediction_sigma = pm.Deterministic(
                'prediction_sigma',
                at.sqrt(prediction_variance))

            pm.Normal('X_o', alpha_mu_X, alpha_sigma_X, observed=X_o_,
                      shape=self.n_covariates)
            pm.Normal('prediction_obs', prediction_mu, prediction_sigma,
                      observed=pred_o_)
        return model

    def sample(self, X_o, pred_o, draws=1000, tune=1000, target_accept=.85,
               chains=4, max_treedepth=15, init="jitter+adapt_diag", rng=None):
        """
        Sample from the posterior given observed data.
        Stores the latest trace in self.

        Args:
            X_o (np.ndarray): Observed feature vector of length k.
            pred_o (float): Observed scalar prediction for X_o.
            draws (int): Size of sample to draw from the posterior.
            tune (int): Number of tuning (warmup) iterations.
            target_accept (float): Target acceptance probability for NUTS.
            chains (int): Number of chains to sample.
            max_treedepth (int): Maximum tree depth for NUTS.
            init (str): Initialization method passed to pm.sample.
            rng: Seed or initialized random number generator for
                reproducibility.

        Returns:
            None
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
        self.pred_o = float(np.asarray(pred_o))
        # Posterior coefficient reconstructions belong to the previous trace.
        self.posterior_predictive_samples = None
        self.posterior_coefficient_draws = None
        return

    def _parameter_draws(self, sample_group):
        """
        Convert sampled PyMC variables into one dictionary per draw.

        Chains and draws are flattened into a single draw axis. Each returned
        dictionary maps a predictive variable name to its value for that draw,
        with scalars cast to float and vector-valued parameters (e.g.
        alpha_mu_X) kept as arrays. These dictionaries are then used to update
        the internal predictive SCM and to reconstruct the marginalized
        coefficient vectors.

        Args:
            sample_group: An ArviZ group (e.g. ``trace.posterior`` or
                ``prior_samples.prior``) holding the sampled variables.

        Returns:
            list: One parameter dictionary per draw.
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

    def _update_predictive_scm(self, parameters, N_b, N_d):
        """
        Update the internal predictive SCM with one parameter draw.

        Only the population-level SCM parameters are forwarded; the collapsed
        auxiliaries R and T are not SCM parameters. The coefficient vectors
        N_b and N_d are passed explicitly because they are not part of the
        sampled draw (they are marginalized in the collapsed model).

        Args:
            parameters (dict): One draw as returned by
                :meth:`_parameter_draws`.
            N_b (np.ndarray): Coefficient vector mapping X to Y, length k.
            N_d (np.ndarray): Coefficient vector mapping X to A, length k.
        """
        scm_parameter_names = ('alpha_mu_X', 'alpha_sigma_X', 'alpha_mu_A',
                               'alpha_sigma_A', 'alpha_mu_Y', 'alpha_sigma_Y',
                               'N_E')
        scm_parameters = {name: parameters[name]
                          for name in scm_parameter_names}
        scm_parameters['N_b'] = N_b
        scm_parameters['N_d'] = N_d
        self.predictive_scm.update_parameters(scm_parameters)

    def _sample_prior_coefficients(self, rng):
        """
        Sample the marginalized coefficient vectors from their priors.

        The collapsed model integrates N_b and N_d out of the PyMC graph, but
        their priors are known, so prior predictive sampling can draw them
        directly.

        Args:
            rng (np.random.Generator): Initialized random number generator.

        Returns:
            tuple: ``(N_b, N_d)``, both length-k arrays.
        """
        N_b = rng.normal(self.prior_parameters['N_b_mu'],
                         self.prior_parameters['N_b_sigma'])
        N_d = rng.normal(self.prior_parameters['N_d_mu'],
                         self.prior_parameters['N_d_sigma'])
        return N_b, N_d

    def _sample_conditional_coefficients(self, parameters, rng):
        """
        Sample the marginalized coefficient vectors from their conditional
        Gaussian given one posterior draw and the observed prediction.

        Conditionally on (N_E, R, T, the population-level parameters, and
        X_o), the observed prediction is linear in the stacked coefficient
        vector ``theta = (N_b, N_d)`` with Gaussian noise: \n
        :math:`pred^o = c + a^\\top \\theta + \\epsilon`,
        :math:`\\epsilon \\sim \\mathcal{N}(0, \\tau^2 (1/n + R/T))`, \n
        with ``a = (X_o, N_E X_o)``. The conditional posterior of theta is the
        standard linear-Gaussian update of its independent Gaussian prior.
        Sampling uses Matheron's rule: a prior draw of theta and of the noise
        is corrected by the (scalar-denominator) Kalman gain, which produces
        an exact conditional draw without forming the conditional covariance.

        Args:
            parameters (dict): One posterior draw as returned by
                :meth:`_parameter_draws`, holding N_E, R, T, and the
                population-level parameters.
            rng (np.random.Generator): Initialized random number generator.

        Returns:
            tuple: ``(N_b, N_d)``, both length-k arrays.
        """
        N_E = parameters['N_E']
        prior_mean = np.concatenate([self.prior_parameters['N_b_mu'],
                                     self.prior_parameters['N_d_mu']])
        prior_variance = np.concatenate(
            [self.prior_parameters['N_b_sigma']**2,
             self.prior_parameters['N_d_sigma']**2])
        a = np.concatenate([self.X_o, N_E * self.X_o])

        c = parameters['alpha_mu_Y'] + N_E * parameters['alpha_mu_A']
        tau_squared = (N_E**2 * parameters['alpha_sigma_A']**2 +
                       parameters['alpha_sigma_Y']**2)
        noise_variance = tau_squared * (
            1.0 / self.plate_size + parameters['R'] / parameters['T'])

        theta_prior = rng.normal(prior_mean, np.sqrt(prior_variance))
        prediction_prior = (c + a @ theta_prior +
                            rng.normal(0.0, np.sqrt(noise_variance)))
        gain = prior_variance * a / (a @ (prior_variance * a) + noise_variance)
        theta = theta_prior + gain * (self.pred_o - prediction_prior)
        return theta[:self.n_covariates], theta[self.n_covariates:]

    def sample_prior_predictive(self, draws=1000, rng=None):
        """
        Method for sampling from the distribution based on the agent's prior
        beliefs. For every draw of the predictive parameters from the prior,
        the marginalized coefficient vectors are drawn from their priors, the
        internal predictive SCM is updated, and one (X, A, Y) sample is
        generated from it.

        Args:
            draws (int): Size of sample to draw.
            rng: Seed or initialized random number generator for
                reproducibility.

        Returns:
            list: One ``[X, A, Y]`` triple per draw.
        """
        rng = np.random.default_rng(rng)
        with self.model:
            tmp = pm.sample_prior_predictive(
                draws=draws,
                var_names=self.predictive_var_names,
                random_seed=rng)
        self.prior_samples = tmp
        parameter_draws = self._parameter_draws(tmp.prior)
        self.prior_draws = len(parameter_draws)
        self.prior_coefficient_draws = [
            self._sample_prior_coefficients(rng) for _ in parameter_draws]
        samples = []
        for parameters, (N_b, N_d) in zip(parameter_draws,
                                          self.prior_coefficient_draws):
            self._update_predictive_scm(parameters, N_b, N_d)
            X, A, Y, *_ = self.predictive_scm.sample(rng=rng)
            samples.append([X, A, Y])
        self.prior_predictive_samples = samples
        return samples

    def _ensure_prior_samples(self, rng=None):
        """
        Draw prior samples if they have not been drawn yet.

        Args:
            rng: Seed or initialized random number generator for
                reproducibility, used only if prior samples are drawn here.
        """
        if self.prior_samples is None or self.prior_coefficient_draws is None:
            self.sample_prior_predictive(rng=rng)

    def sample_prior_predictive_do_A(self, X, A_treatment, rng=None):
        """
        Method for estimating the outcome for a patient with features X based
        on the agent's prior beliefs. Draws one potential outcome for every
        prior draw, reusing the coefficient vectors drawn with the prior
        samples.

        Args:
            X (np.array): Feature vector of length k to condition on.
            A_treatment (float): Fixed treatment value.
            rng: Seed or initialized random number generator for
                reproducibility.

        Returns:
            tuple: ``(do_A_samples, do_A_mean, do_A_std)``.
        """
        self._ensure_prior_samples(rng=rng)
        X = self._normalize_vector_parameter(X, 'X')
        do_A_samples = np.zeros(shape=(self.prior_draws,))
        parameter_draws = self._parameter_draws(self.prior_samples.prior)
        for i, (parameters, (N_b, N_d)) in enumerate(
                zip(parameter_draws, self.prior_coefficient_draws)):
            self._update_predictive_scm(parameters, N_b, N_d)
            tmp = self.predictive_scm.sample_do_A(A_treatment, X, rng=rng)
            do_A_samples[i] = tmp[0][0]
        do_A_mean = np.mean(do_A_samples)
        do_A_std = np.std(do_A_samples)
        return do_A_samples, do_A_mean, do_A_std

    def sample_prior_predictive_summaries(self, rng=None):
        """
        Method that estimates the mean, standard deviation, skewness, and
        kurtosis of the variables X, A, and Y in the prior predictive samples.

        Args:
            rng: Seed or initialized random number generator for
                reproducibility. Used only if prior samples were not drawn
                earlier.

        Returns:
            dict: Summary statistics keyed by ``pri_<var>_<moment>``.
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

    def _ensure_posterior_coefficient_draws(self, parameter_draws, rng):
        """
        Reconstruct the marginalized coefficient vectors for every posterior
        draw if they have not been reconstructed yet.

        The reconstructions are cached so that posterior predictive sampling,
        do(A) estimates, and summaries all use the same coefficient draws for
        the same trace.

        Args:
            parameter_draws (list): Posterior draws as returned by
                :meth:`_parameter_draws`.
            rng (np.random.Generator): Initialized random number generator.
        """
        if self.posterior_coefficient_draws is None:
            self.posterior_coefficient_draws = [
                self._sample_conditional_coefficients(parameters, rng)
                for parameters in parameter_draws]

    def sample_posterior_predictive(self, rng=None):
        """
        This method generates a sample from the agent's posterior beliefs.
        Using the trace stored in self, it reconstructs the marginalized
        coefficient vectors from their conditional Gaussian for each posterior
        draw, replaces the parameter values in the internal predictive SCM,
        and draws a posterior predictive sample from it.

        Args:
            rng: Seed or initialized random number generator for
                reproducibility.

        Returns:
            list: One ``(X, A, Y, mu_Y_cond, std_Y_cond)`` sample per draw.
        """
        rng = np.random.default_rng(rng)
        parameter_draws = self._parameter_draws(self.trace.posterior)
        self.posterior_draws = len(parameter_draws)
        self._ensure_posterior_coefficient_draws(parameter_draws, rng)
        samples = []
        for parameters, (N_b, N_d) in zip(parameter_draws,
                                          self.posterior_coefficient_draws):
            self._update_predictive_scm(parameters, N_b, N_d)
            samples.append(self.predictive_scm.sample(rng=rng))
        self.posterior_predictive_samples = samples
        return samples

    def sample_posterior_predictive_do_A(self, X, A_treatment, rng=None):
        """
        Method for estimating the outcome for a patient with features X based
        on the agent's posterior beliefs. Draws one potential outcome for
        every draw in the trace, reusing the reconstructed coefficient vectors
        for the current trace.

        Args:
            X (np.array): Feature vector of length k to condition on.
            A_treatment (float): Fixed treatment value.
            rng: Seed or initialized random number generator for
                reproducibility.

        Returns:
            tuple: ``(potential_outcomes, do_A_mean, do_A_std)``.
        """
        rng = np.random.default_rng(rng)
        X = self._normalize_vector_parameter(X, 'X')
        parameter_draws = self._parameter_draws(self.trace.posterior)
        self._ensure_posterior_coefficient_draws(parameter_draws, rng)
        potential_outcomes = np.zeros(shape=(len(parameter_draws),))
        for i, (parameters, (N_b, N_d)) in enumerate(
                zip(parameter_draws, self.posterior_coefficient_draws)):
            self._update_predictive_scm(parameters, N_b, N_d)
            tmp = self.predictive_scm.sample_do_A(A_treatment, X, rng=rng)
            potential_outcomes[i] = tmp[0][0]
        do_A_mean = np.mean(potential_outcomes)
        do_A_std = np.std(potential_outcomes)
        return potential_outcomes, do_A_mean, do_A_std

    def sample_posterior_predictive_summaries(self, rng=None):
        """
        Method that estimates the mean, standard deviation, skewness, and
        kurtosis of the variables X, A, and Y in the posterior predictive
        samples.

        Args:
            rng: Seed or initialized random number generator for
                reproducibility. Used only if posterior predictive samples
                were not drawn earlier.

        Returns:
            dict: Summary statistics keyed by ``post_<var>_<moment>``.
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

    def sample_prior_predictive_CATE(self, A1_treatment, A0_treatment,
                                     draws=1000, rng=None):
        """
        This method returns a sample from the prior CATE-distribution, and its
        sample mean and standard deviation. Under the collapsed model the CATE
        depends only on N_E, so it is obtained from the prior draws of N_E
        multiplied by ``A1_treatment - A0_treatment``. It does not require
        conditioning on X. The prior draws are shared with the other prior
        predictive methods.

        Args:
            A1_treatment (float): Dosage value of higher dose treatment.
            A0_treatment (float): Dosage value of lower dose treatment.
            draws (int): Size of sample to draw if prior samples are drawn
                here.
            rng: Seed or initialized random number generator for
                reproducibility.

        Returns:
            tuple: ``(CATE_samples, CATE_mean, CATE_std)``.
        """
        if self.prior_samples is None or self.prior_coefficient_draws is None:
            self.sample_prior_predictive(draws=draws, rng=rng)
        N_E_samples = self.prior_samples.prior['N_E'].values.reshape(-1)
        CATE_samples = (A1_treatment - A0_treatment) * N_E_samples
        CATE_mean = np.mean(CATE_samples)
        CATE_std = np.std(CATE_samples)
        return CATE_samples, CATE_mean, CATE_std

    def sample_posterior_predictive_CATE(self, A1_treatment, A0_treatment):
        """
        This method returns a sample from the posterior CATE-distribution, and
        its sample mean and standard deviation. Under the collapsed model the
        CATE depends only on N_E, so it is the posterior draws of N_E multiplied
        by ``A1_treatment - A0_treatment``.

        Args:
            A1_treatment (float): Dosage value of higher dose treatment.
            A0_treatment (float): Dosage value of lower dose treatment.

        Returns:
            tuple: ``(CATE_samples, CATE_mean, CATE_std)``.
        """
        N_E_samples = self.trace.posterior['N_E'].values.reshape(-1)
        self.posterior_draws = N_E_samples.size
        CATE_samples = (A1_treatment - A0_treatment) * N_E_samples
        CATE_mean = np.mean(CATE_samples)
        CATE_std = np.std(CATE_samples)
        return CATE_samples, CATE_mean, CATE_std

    def get_prior_CATE_mu(self, A1_treatment, A0_treatment):
        """
        Return the prior mean of the CATE-distribution. Under the collapsed
        model the treatment effect only depends on N_E and the size of the
        treatments.

        Args:
            A1_treatment (float): Dosage value of higher dose treatment.
            A0_treatment (float): Dosage value of lower dose treatment.
        """
        return (A1_treatment - A0_treatment) * self.prior_parameters['N_E_mu']

    def trace_diagnostics(self):
        """
        Return basic sampler diagnostics for the current posterior trace.

        Returns:
            dict: ``max_rhat``, ``min_ess_bulk``, ``min_ess_tail`` (computed
            with ArviZ), and ``n_divergences``.
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

    def save_agent_model(self, file_name='single_prediction_collapsed_agent_model',
                         folder='./saved_models/'):
        """
        Save the agent model configuration and available samples.

        The trace and prior samples are written separately with ArviZ
        (NetCDF), and the rest of the configuration is pickled with joblib.
        The PyMC model, trace, and prior samples are temporarily detached
        before pickling and restored afterwards.

        Args:
            file_name (str): Base filename without extension. Creates
                ``{file_name}.joblib`` and, when available,
                ``{file_name}_trace.nc`` and ``{file_name}_prior_samples.nc``.
            folder (str): Folder to save the model into.
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
    def load_agent_model(cls, file_name='single_prediction_collapsed_agent_model',
                         folder='./saved_models/'):
        """
        Load a saved single-prediction collapsed agent model.

        The configuration is loaded with joblib and the PyMC model is
        reconstructed by re-running ``__init__``. The trace and prior samples
        are loaded from their NetCDF files if present.

        Args:
            file_name (str): Base filename without extension. Expects to find
                ``{file_name}.joblib`` and, optionally, ``{file_name}_trace.nc``
                and ``{file_name}_prior_samples.nc``.
            folder (str): Folder the model is saved in.

        Returns:
            SinglePredictionCollapsedAgentModel: The loaded agent model.
        """
        file_path = Path(folder) / file_name
        agent = joblib.load(f'{file_path}.joblib')
        print("Agent model loaded from {}".format(file_path))

        posterior_draws = agent.posterior_draws
        prior_draws = agent.prior_draws
        X_o = agent.X_o
        pred_o = agent.pred_o

        agent.__init__(
            n_covariates=agent.n_covariates,
            plate_size=agent.plate_size,
            prior_parameters=agent.prior_parameters,
        )

        agent.posterior_draws = posterior_draws
        agent.prior_draws = prior_draws
        agent.X_o = X_o
        agent.pred_o = pred_o

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
