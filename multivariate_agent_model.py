"""
Multivariate Bayesian agent model.

This is the multivariate counterpart of :class:`agent_model.AgentModel`. The
AgentModel encapsulates both the agent's beliefs and the thought processes that
an ideal Bayesian agent would use in the context of decision support. The agent
has priors, can make a decision based on those, can update his beliefs when
presented with new information in the form of an observation and a prediction
for it, and can make decisions based on his posterior beliefs.

Here the covariate vector ``X`` is k-dimensional, the predictive model is an
OLS regression of ``Y`` on ``X`` (with or without intercept), and the agent
observes a single observed feature vector ``X_o`` together with the scalar
prediction ``pred_o`` produced for it.

Instead of representing the historical plate of training patients explicitly,
the model uses the low-dimensional sufficient-statistic reduction of the plate.
The reduced auxiliary variables are: \n
- ``(s, W, g, h)`` for the OLS model with intercept, and \n
- ``(s, W, g)`` for the OLS model without intercept. \n
These reproduce the OLS coefficients (and intercept) of the external predictive
model without sampling the n historical observations one by one.
"""

import dataclasses
from pathlib import Path

import arviz as az
import joblib
import numpy as np
import pymc as pm
import pytensor.tensor as at
from scipy.stats import kurtosis, skew
from abc import ABC, abstractmethod
import warnings

from multivariate_scm import MultivariateSCM


DEBUGGING = False


def printing_callback(trace, draw):
    """
    Function for debugging pymc-sampling.
    """
    if DEBUGGING:
        print(f"Completed draw {draw.draw_idx} from chain {draw.chain}.")

class AgentModel(ABC):
    """
    This abstract class serves as blueprint for all agent models. The
    hardcoded assumptions are: \n
    1) you have two treatment options A1_treatment and A0_treatment, \n
    2) you are interested in one outcome Y, \n
    3) all features are encoded in X. \n
    The class requires you to define your model and sampling method.
    That is, you have to define your beliefs about the data generating
    process (both prior distributions and SCM), and you have to define
    how you believe the model and prediction were generated from unseen
    data. The last step is necessary to generate posterior samples and
    should preferably be computationally efficient (plate reduction!).
    """
    @abstractmethod
    def __init__(self, plate_size, prior_parameters=None,
                 predictive_var_names=None):
        """
        Override method in subclass, but include at least the attributes
        listed here.
        """
        self.plate_size = int(plate_size)

        # Parameter names are dependent on the sampling method defined (right?).
        #   -where should they be defined?
        # First one includes hardcoded parameter names
        self.prior_parameters = self._initialize_prior_parameters(
            prior_parameters, self._default_prior_parameters())

        # Sampling method is dependent on SCM parameterization.
        self.predictive_scm = None  # Some MultivariateSCM()

        self.model = self._build_model()

        # Setting predictive_var_names saves PyMC computation AND is used to
        # update the Agent's "internal" SCM parameters, which is
        # needed for both prior and posterior predictive sampling.
        # The default is derived from the predictive SCM's own fields, so it
        # requires self.predictive_scm and self.model to already be set.
        self.predictive_var_names = self._initialize_predictive_var_names(
            predictive_var_names)

        self.trace = None
        self.prior_samples = None
        self.prior_predictive_samples = None
        self.posterior_predictive_samples = None
        self.posterior_draws = None
        self.prior_draws = None
        self.X_o = None

    @abstractmethod
    def _default_prior_parameters():
        """
        Return the full default prior-parameter dictionary for the subclass.

        The keys define the model's prior-hyperparameter naming scheme (e.g.
        ``mu_X``, ``mu_sigma_X``), which is distinct from the SCM's own
        attribute names (e.g. ``alpha_mu_X``). ``_initialize_prior_parameters``
        uses this dict both as the fallback when no ``prior_parameters`` are
        passed, and as the reference key-set for validating a partial dict.
        """
        pass

    @abstractmethod
    def _build_model() -> pm.Model:
        # Method not necessaary if AgentModel initializes model some other way?
        pass

    @abstractmethod
    def _build_population_parameters():
        # Exclude this from required methods?
        pass

    @abstractmethod
    def sample(self, X: np.ndarray, pred_o: np.ndarray) -> None:
        """
        Sample from the posterior given observed data.
        Store the latest trace in self.trace. 
        """
        pass

    def _flatten_vector_parameter(self, parameter, parameter_name):
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

    def _initialize_predictive_var_names(self, predictive_var_names):
        """
        Validate or derive the predictive variable-name list.

        The default is the population-level SCM parameter names that are both
        fields of the internal predictive SCM and named random variables in
        the PyMC model. Requires ``self.predictive_scm`` and ``self.model`` to
        already be set.

        Args:
            predictive_var_names (list or None): User-supplied variable names.
                If None, the defaults are used. Otherwise the names are
                validated against the model specification.

        Returns:
            list: Validated predictive variable names.
        """
        scm_field_names = [f.name for f in dataclasses.fields(self.predictive_scm)]
        default_predictive_var_names = [
            name for name in scm_field_names if name in self.model.named_vars]
        if predictive_var_names is None:
            return default_predictive_var_names
        extra_items = [item for item in predictive_var_names
                       if item not in default_predictive_var_names]
        err_msg = "{} passed with predictive_var_names, " \
            "but not part of model specification.".format(extra_items)
        assert len(extra_items) < 1, err_msg
        return predictive_var_names

    def _initialize_prior_parameters(self, prior_parameters, default_prior_parameters):
        """
        Validate or create the prior-parameter dictionary.

        ``default_prior_parameters`` serves two roles: if ``prior_parameters``
        is None, it is used directly as the result. Otherwise it is the
        reference key-set for validation: any key in ``prior_parameters`` not
        found in ``default_prior_parameters`` raises an ``AssertionError``,
        and the validated keys are merged on top of the defaults. In both
        cases the vector-valued entries are then normalized to length-k
        arrays. Note that these keys are prior-hyperparameter names (e.g.
        ``mu_X``), not the SCM's own attribute names (e.g. ``alpha_mu_X``).

        Args:
            prior_parameters (dict or None): User-supplied prior parameters.
            default_prior_parameters (dict): Default prior-parameter
                dictionary, typically from the subclass's
                ``_default_prior_parameters()``. Used both as the fallback
                value and as the valid-key reference set.

        Returns:
            dict: Validated and normalized prior-parameter dictionary.
        """
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
            prior_parameters[parameter_name] = self._flatten_vector_parameter(
                prior_parameters[parameter_name], parameter_name)
        return prior_parameters

    def sample_prior_predictive_do_A(self, X, A_treatment, rng=None):
        """
        Method for estimating the outcome for a patient with features X based
        on the agent's prior beliefs. Draws one potential outcome for every
        prior draw.

        Args:
            X (np.array): Feature vector of length k to condition on.
            A_treatment (float): Fixed treatment value.
            rng: Seed or initialized random number generator for
                reproducibility.

        Returns:
            tuple: ``(do_A_samples, do_A_mean, do_A_std)``.
        """
        self._ensure_prior_samples(rng=rng)
        X = self._flatten_vector_parameter(X, 'X')
        do_A_samples = np.zeros(shape=(self.prior_draws,))
        for i, parameters in enumerate(self._parameter_draws(self.prior_samples.prior)):
            self.predictive_scm.update_parameters(parameters)
            tmp = self.predictive_scm.sample_do_A(A_treatment, X, rng=rng)
            do_A_samples[i] = tmp[0][0]
        do_A_mean = np.mean(do_A_samples)
        do_A_std = np.std(do_A_samples)
        return do_A_samples, do_A_mean, do_A_std

    def sample_prior_predictive(self, draws=1000, rng=None):
        """
        Method for sampling from the distribution based on the agent's prior
        beliefs. For every draw of the predictive parameters from the prior,
        the internal predictive SCM is updated and one (X, A, Y) sample is
        generated from it.

        Args:
            draws (int): Size of sample to draw.
            rng: Seed or initialized random number generator for
                reproducibility.

        Returns:
            list: One ``[X, A, Y]`` triple per draw.
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

    def _parameter_draws(self, sample_group):
        """
        Convert sampled PyMC variables into one dictionary per draw.

        Chains and draws are flattened into a single draw axis. Each returned
        dictionary maps a predictive variable name to its value for that draw,
        with scalars cast to float and vector-valued parameters (e.g. N_b,
        N_d, alpha_mu_X) kept as arrays. These dictionaries are then used to
        update the internal predictive SCM.

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

    def _moment_dict(self, values, key_prefix):
        """
        Compute mean, standard deviation, skewness, and kurtosis for a
        one-dimensional sample.

        Args:
            values (array-like): Sample values. Flattened to one dimension.
            key_prefix (str): Prefix for the returned keys.

        Returns:
            dict: ``{key_prefix}_mean``, ``{key_prefix}_std``,
            ``{key_prefix}_skew``, ``{key_prefix}_kurtosis``.
        """
        values = np.asarray(values).reshape(-1)
        return {
            '{}_mean'.format(key_prefix): np.mean(values),
            '{}_std'.format(key_prefix): np.std(values),
            '{}_skew'.format(key_prefix): skew(values),
            '{}_kurtosis'.format(key_prefix): kurtosis(values),
        }

    def _ensure_prior_samples(self, rng=None):
        """
        Draw prior samples if they have not been drawn yet.

        Args:
            rng: Seed or initialized random number generator for
                reproducibility, used only if prior samples are drawn here.
        """
        if self.prior_samples is None:
            self.sample_prior_predictive(rng=rng)

    def sample_prior_predictive_CATE(self, X, A1_treatment, A0_treatment, rng=None):
        """
        This method returns a sample from the prior CATE-distribution, and its
        sample mean and standard deviation. The current SCM is such that N_E is
        not dependent on X, so the treatment effect only depends on N_E and the
        size of the treatments.

        Args:
            X (np.array): Feature vector of length k to condition on.
            A1_treatment (float): Dosage value of higher dose treatment.
            A0_treatment (float): Dosage value of lower dose treatment.
            rng: Seed or initialized random number generator for
                reproducibility.

        Returns:
            tuple: ``(CATE_samples, CATE_mean, CATE_std)``.
        """
        self._ensure_prior_samples(rng=rng)
        X = self._flatten_vector_parameter(X, 'X')
        CATE_samples = np.zeros(shape=(self.prior_draws,))
        for i, parameters in enumerate(self._parameter_draws(self.prior_samples.prior)):
            self.predictive_scm.update_parameters(parameters)
            tmp = self.predictive_scm.sample_CATE(
                X, A1_treatment, A0_treatment, rng=rng)
            CATE_samples[i] = tmp[0][0]
        CATE_mean = np.mean(CATE_samples)
        CATE_std = np.std(CATE_samples)
        return CATE_samples, CATE_mean, CATE_std

    def sample_posterior_predictive(self, rng=None):  # BUG? This needs to take X as input if we want conditional values.
        """
        This method generates a sample from the agent's posterior beliefs.
        Using the trace stored in self, it replaces the parameter values in
        the internal predictive SCM with each posterior draw and draws a
        posterior predictive sample from it.

        Args:
            rng: Seed or initialized random number generator for
                reproducibility.

        Returns:
            list: One ``(X, A, Y, mu_Y_cond, std_Y_cond)`` sample per draw.
        """
        parameter_draws = self._parameter_draws(self.trace.posterior)  # BUG or inconsistency? Does not check for existence and replaces old samples?
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
        This method returns a sample from the posterior predictive
        CATE-distribution, and its sample mean and standard deviation.

        Args:
            X (np.array): Feature vector of length k to condition on.
            A1_treatment (float): Higher dose treatment value.
            A0_treatment (float): Lower dose treatment value.
            rng: Seed or initialized random number generator for
                reproducibility.

        Returns:
            tuple: ``(CATE_samples, CATE_mean, CATE_std)``.
        """
        X = self._flatten_vector_parameter(X, 'X')
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
        Method for estimating the outcome for a patient with features X based
        on the agent's posterior beliefs. Draws one potential outcome for every
        draw in the trace.

        Args:
            X (np.array): Feature vector of length k to condition on.
            A_treatment (float): Fixed treatment value.
            rng: Seed or initialized random number generator for
                reproducibility.

        Returns:
            tuple: ``(potential_outcomes, do_A_mean, do_A_std)``.
        """
        X = self._flatten_vector_parameter(X, 'X')
        parameter_draws = self._parameter_draws(self.trace.posterior)
        potential_outcomes = np.zeros(shape=(len(parameter_draws),))
        for i, parameters in enumerate(parameter_draws):
            self.predictive_scm.update_parameters(parameters)
            tmp = self.predictive_scm.sample_do_A(A_treatment, X, rng=rng)
            potential_outcomes[i] = tmp[0][0]
        do_A_mean = np.mean(potential_outcomes)
        do_A_std = np.std(potential_outcomes)
        return potential_outcomes, do_A_mean, do_A_std

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

    def save_agent_model(self, file_name='multivariate_agent_model',
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
    def load_agent_model(cls, file_name='multivariate_agent_model',
                         folder='./saved_models/'):
        """
        Load a saved multivariate agent model.

        The configuration is loaded with joblib and the PyMC model is
        reconstructed by re-running ``__init__``. The trace and prior samples
        are loaded from their NetCDF files if present.

        Args:
            file_name (str): Base filename without extension. Expects to find
                ``{file_name}.joblib`` and, optionally, ``{file_name}_trace.nc``
                and ``{file_name}_prior_samples.nc``.
            folder (str): Folder the model is saved in.

        Returns:
            MultivariateAgentModel: The loaded agent model.
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

    def sample_prior_predictive_summaries(self, rng=None):
        """
        Method that estimates the mean, standard deviation, skewness, and
        kurtosis of the variables X, A, and Y in the prior predictive samples.

        X is k-dimensional, so it is summarized per covariate dimension
        (``pri_X_<dim>_<moment>``); A and Y are always scalar.

        Args:
            rng: Seed or initialized random number generator for
                reproducibility. Used only if prior samples were not drawn
                earlier.

        Returns:
            dict: Summary statistics keyed by ``pri_<var>_<moment>`` (A, Y)
            or ``pri_<var>_<dim>_<moment>`` (X).
        """
        self._ensure_prior_samples(rng=rng)
        X = np.asarray([item[0] for item in self.prior_predictive_samples]
                       ).reshape(-1, self.n_covariates)
        A = [item[1] for item in self.prior_predictive_samples]
        Y = [item[2] for item in self.prior_predictive_samples]
        results = {}
        for dim in range(self.n_covariates):
            results.update(self._moment_dict(X[:, dim], 'pri_X_{}'.format(dim)))
        results.update(self._moment_dict(A, 'pri_A'))
        results.update(self._moment_dict(Y, 'pri_Y'))
        return results

    def sample_posterior_predictive_summaries(self, rng=None):
        """
        Method that estimates the mean, standard deviation, skewness, and
        kurtosis of the variables X, A, and Y in the posterior predictive
        samples.

        X is k-dimensional, so it is summarized per covariate dimension
        (``post_X_<dim>_<moment>``); A and Y are always scalar.

        Args:
            rng: Seed or initialized random number generator for
                reproducibility. Used only if posterior predictive samples
                were not drawn earlier.

        Returns:
            dict: Summary statistics keyed by ``post_<var>_<moment>`` (A, Y)
            or ``post_<var>_<dim>_<moment>`` (X).
        """
        if self.posterior_predictive_samples is None:
            self.sample_posterior_predictive(rng=rng)
        X = np.asarray([item[0] for item in self.posterior_predictive_samples]
                       ).reshape(-1, self.n_covariates)
        A = [item[1] for item in self.posterior_predictive_samples]
        Y = [item[2] for item in self.posterior_predictive_samples]
        results = {}
        for dim in range(self.n_covariates):
            results.update(self._moment_dict(X[:, dim], 'post_X_{}'.format(dim)))
        results.update(self._moment_dict(A, 'post_A'))
        results.update(self._moment_dict(Y, 'post_Y'))
        return results


class MultivariateAgentModel(AgentModel):
    """
    Class for a multivariate agent model with structural equations:

    :math:`Y_i = N_b^\\top X_i + N_{E} A_i + N_{Yi}`

    :math:`A_i = N_d^\\top X_i + N_{Ai}`

    :math:`X_i = N_{Xi}`

    where N_{Xi}, N_{Ai}, N_{Yi} are normal noise variables drawn for every
    instantiation of the plate, N_{E} is a normal variable drawn once per
    instantiation of the plate, and the coefficient vectors N_b and N_d are
    likewise drawn once per instantiation. The structural intercept is carried
    by alpha_mu_Y.

    Rather than modeling the plate of plate_size patients explicitly, the model
    reduces it to low-dimensional auxiliary variables. With intercept these are
    (s, W, g, h); without intercept they are (s, W, g). From these the OLS slope
    vector ``phi`` (and, with intercept, the OLS intercept ``alpha``) of the
    external predictive model are reconstructed, and the observed prediction is
    linked to the model through ``alpha + X_o . phi`` (with intercept) or
    ``X_o . phi`` (without intercept).

    Args:
        n_covariates (int): Number of covariates k (dimension of X).
        plate_size (int): Size of the plate (number of i.i.d. samples used to
            train the predictive model). Must be larger than n_covariates.
        prior_parameters (dict): Dictionary with parameters for the prior
            distributions. If None, defaults are used. Vector-valued entries
            accept a scalar (broadcast to length k) or an array of length k.
        predictive_var_names (list): Names of the variables sampled for
            predictive calculations. If None, defaults are used.
        fit_intercept (bool): If True, the predictive model is an OLS with
            intercept and the reduced variables (s, W, g, h) are used. If
            False, the predictive model is slope-only and (s, W, g) are used.

    Returns:
        None
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

        self.prior_parameters = self._initialize_prior_parameters(
            prior_parameters, self._default_prior_parameters())

        # Sampling method is dependent on SCM parameterization.
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

        self.model = self._build_model()

        self.predictive_var_names = self._initialize_predictive_var_names(
            predictive_var_names)

        self.trace = None
        self.prior_samples = None
        self.prior_predictive_samples = None
        self.posterior_predictive_samples = None
        self.posterior_draws = None
        self.prior_draws = None
        self.X_o = None

    def _default_prior_parameters(self):
        """
        Return default prior parameters for the multivariate model.

        Vector-valued entries are length-k arrays. The coefficient means
        ``N_b_mu`` and ``N_d_mu`` are scaled by ``1 / k`` so the aggregate
        signal stays close to the scalar case.

        These keys are prior-hyperparameter names, not the SCM's own
        attribute names. They are mapped onto the ``MultivariateSCM``
        constructor's keyword arguments in ``MultivariateAgentModel.__init__``
        (e.g. ``prior_parameters['mu_X']`` becomes ``alpha_mu_X``).

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

    def _build_population_parameters(self):
        """
        Build the PyMC variables that live outside the historical plate.

        These are the population-level SCM parameters the agent has priors on:
        the covariate parameters (vector-valued alpha_mu_X, alpha_sigma_X), the
        treatment and outcome parameters (alpha_mu_A, alpha_sigma_A, alpha_mu_Y,
        alpha_sigma_Y), the effect size N_E, and the coefficient vectors N_b
        and N_d. Standard deviations use a TruncatedNormal lower-bounded at 0.

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

        The observed feature vector ``X_o`` and the observed prediction
        ``pred_o`` enter as :class:`pm.Data` placeholders, updated at sampling
        time. The plate of plate_size patients is replaced by the reduced
        auxiliary variables ``s`` (sum of covariate noises), ``W_chol``
        (Cholesky factor of the Wishart dispersion), ``g`` (centered A/Y
        noise), and, with intercept, ``h`` (mean A/Y noise).

        With intercept, the OLS slope vector ``phi`` and intercept ``alpha``
        are reconstructed from (s, W, g, h) and the observed prediction is
        modeled as ``alpha + X_o . phi``. Without intercept, ``X^T X`` is
        rebuilt from ``S_x`` and ``W``, the slope-only ``phi`` is reconstructed
        from (s, W, g), and the prediction is modeled as ``X_o . phi``. In both
        cases ``X_o`` is also given the covariate likelihood
        ``N(alpha_mu_X, alpha_sigma_X)``.

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
        X_o = self._flatten_vector_parameter(X_o, 'X_o')
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

    def get_prior_CATE_mu(self, A1_treatment, A0_treatment):  # REMOVE? This has the potential to break something with a different model.
        """
        Return the prior mean of the CATE-distribution. The current SCM is such
        that N_E is not dependent on X and the treatment effect only depends on
        N_E and the size of the treatments.

        Args:
            A1_treatment (float): Dosage value of higher dose treatment.
            A0_treatment (float): Dosage value of lower dose treatment.
        """
        warnings.warn("This method will be deprecated soon.")
        return (A1_treatment - A0_treatment) * self.prior_parameters['N_E_mu']
