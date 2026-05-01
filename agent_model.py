"""
The AgentModel encapsulates both the agent's beliefs and
thought processes that an ideal Bayesian agent would use
in the context of decision support. The agent has priors,
can make a decision based on those, can update his beliefs
when presented with new information in the form of an
observation and a prediction for it, and can make
decisions based on his posterior beliefs.

The Bayesian models specified here correspond to how an ideal
Bayesian agent would update his beliefs given that he sees
one or several observations and receives their predictions
that are based on an OLS regression model without intercept
by default, or with intercept when explicitly requested.
"""
import arviz as az
import joblib
import numpy as np
import pymc as pm
import pytensor.tensor as at
from scipy.stats import skew, kurtosis
import scm

DEBUGGING = False

def printing_callback(trace, draw):
    """
    Function for debugging pymc-sampling.
    """
    if DEBUGGING:
        print(f"Completed draw {draw.draw_idx} from chain {draw.chain}.")
        print("Shape of X: {}".format(trace.get_values('X').shape))
        print("Shape of Y: {}".format(trace.get_values('Y').shape))


class AgentModel():
    """
    Class for agent models with structural equations:

    :math:`Y_i = a + b X_i + N_{E} A_i + N_{Yi}`

    :math:`A_i = d X_i + N_{Ai}`

    :math:`X_i = N_{Xi}`

    where N_{Xi}, N_{Ai}, N_{Yi} are normal noise variables drawn J times
    for every instantiation of the plate and N_{E} is a normal variable
    drawn once per instantiation of the plate.

    Args:
        plate_size (int): size of the plate (number of i.i.d. samples).
        scm_constants (dict): dictionary with structural equation values
            'a', 'b', and 'd'.
        prior_parameters (dict): dictionary with parameters for prior distributions
            If prior_parameters is None, then it defaults to values in the paper (28.8.2025).
        low_dim_aux_plate (bool): If true the plate is estimated using low-dimensional
            auxiliary variables. This speeds up inference. If false, the plate is explicitly 
            modeled.
        fit_intercept (bool): Reserved for backwards-compatible validation.
            Use AgentModelWithIntercept for OLS with intercept.
        test (bool): If true, the model is set up in testing mode enabling multiple observed
            data points to be input. This is mainly for testing that the model works as intended.
    
    Returns:
        None
    """
    def __init__(self, plate_size, prior_parameters=None,
                 scm_constants=None, predictive_var_names=None,
                 low_dim_aux_plate=True, fit_intercept=False, test=False):
        self.plate_size = plate_size

        # Prior parameters
        use_default_scm_constants = scm_constants is None
        default_prior_parameters = self._default_prior_parameters(
            use_default_scm_constants)
        if prior_parameters is None:
            # Use default values.
            self.prior_parameters = default_prior_parameters
        else:
            allowed_parameters = (
                list(default_prior_parameters.keys()) +
                self._optional_prior_parameters()
            )
            extra_items = [item for item in prior_parameters.keys()
                           if item not in allowed_parameters]
            err_msg = "{} passed with prior_parameters, " \
                "but not part of model specification.".format(extra_items)
            assert len(extra_items) < 1, err_msg
            self.prior_parameters = prior_parameters

        self.low_dim_aux_plate = low_dim_aux_plate
        self.fit_intercept = fit_intercept
        self.test = test

        # Variable names of the predictive variables.
        default_predictive_var_names = self._default_predictive_var_names()
        if predictive_var_names is None:
            self.predictive_var_names = default_predictive_var_names
        else:
            extra_items = [item for item in predictive_var_names
                           if item not in default_predictive_var_names]
            err_msg = "{} passed with predictive_var_names, " \
                "but not part of model specification.".format(extra_items)
            assert len(extra_items) < 1, err_msg
            self.predictive_var_names = predictive_var_names

        # SCM parameters. These change during runtime for prior and posterior
        # predictive sampling (samples from the agent's beliefs replace the
        # original values).
        default_scm_constants = self._default_scm_constants()
        if scm_constants is None:
            self.scm_constants = default_scm_constants
        else:
            extra_scm_items = [item for item in scm_constants.keys()
                               if item not in default_scm_constants.keys()]
            err_msg = "{} passed with scm_constants, " \
                "but not part of model specification.".format(extra_scm_items)
            assert len(extra_scm_items) < 1, err_msg
            self.scm_constants = scm_constants

        self._prepare_structural_parameters()
        self._validate_model_configuration()

        # SCM parameters to update in prior predictive and posterior predictive
        # sampling, and an SCM whose variables are updated with samples from the
        # agent's beliefs for predictive sampling.
        self.predictive_scm = scm.SCM(**self.scm_constants)

        # Placeholders
        self.trace = None  # Placeholder for trace
        self.prior_samples = None  # Placeholder for prior samples
        self.prior_predictive_samples = None  # Placeholder for prior predictive samples
        self.posterior_predictive_samples = None  # Placeholder for posterior predictive samples
        self.posterior_draws = None  # Number of draws in trace
        self.prior_draws = None  # Number of draws in prior samples
        self.X_o = None  # Placeholder for observed observation.

        self.model = self._build_model()

    def _default_prior_parameters(self, use_default_scm_constants):
        """
        Return default prior parameters for the base no-intercept model.
        """
        return {'mu_X': 80, 'sigma_X': 0.1, 'mu_sigma_X': 10, 'sigma_sigma_X':0.1,
                'mu_A': 2, 'sigma_A': 1, 'mu_sigma_A': 1, 'sigma_sigma_A':0.1,
                'mu_Y': 0, 'sigma_Y': 0.01, 'mu_sigma_Y': 0.1, 'sigma_sigma_Y' : 0.01,
                'N_E_mu': 1.0, 'N_E_sigma': 1}

    def _optional_prior_parameters(self):
        """
        Return prior parameters accepted beyond the base model defaults.
        """
        return []

    def _default_predictive_var_names(self):
        """
        Return variables sampled for predictive calculations.
        """
        return ['alpha_mu_X', 'alpha_sigma_X', 'alpha_mu_A', 'alpha_sigma_A',
                'alpha_mu_Y', 'alpha_sigma_Y', 'N_E']

    def _default_scm_constants(self):
        """
        Return structural constants for the base no-intercept model.
        """
        return {'a': 12, 'b': -0.1, 'd': 0.125}

    def _initialize_prior_parameters(self, prior_parameters,
                                      use_default_scm_constants):
        """
        Validate or create the prior-parameter dictionary.
        """
        default_prior_parameters = self._default_prior_parameters(
            use_default_scm_constants)
        if prior_parameters is None:
            return default_prior_parameters

        allowed_parameters = (
            list(default_prior_parameters.keys()) + self._optional_prior_parameters()
        )
        extra_items = [item for item in prior_parameters.keys()
                       if item not in allowed_parameters]
        err_msg = "{} passed with prior_parameters, " \
            "but not part of model specification.".format(extra_items)
        assert len(extra_items) < 1, err_msg
        return prior_parameters

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

    def _initialize_scm_constants(self, scm_constants):
        """
        Validate or create the structural constants dictionary.
        """
        default_scm_constants = self._default_scm_constants()
        if scm_constants is None:
            return default_scm_constants

        extra_scm_items = [item for item in scm_constants.keys()
                           if item not in default_scm_constants.keys()]
        err_msg = "{} passed with scm_constants, " \
            "but not part of model specification.".format(extra_scm_items)
        assert len(extra_scm_items) < 1, err_msg
        return scm_constants

    def _prepare_structural_parameters(self):
        """
        Hook for subclasses to transform structural parameters before model build.
        """
        return

    def _validate_model_configuration(self):
        """
        Validate combinations supported by the base no-intercept model.
        """
        if self.fit_intercept:
            raise NotImplementedError(
                "Use AgentModelWithIntercept for OLS with intercept."
            )

    def _build_model(self):
        """
        Build the PyMC graph for the base no-intercept model.
        """
        model = pm.Model()
        with model:
            # Note that the normal distribution in PyMC is parameterized
            # with mu and sigma (not variance, sigma^2).
            # Placeholders for observed data:
            # Named with underscore to distinguish from random variables.
            if not self.test:
                # Default flow.
                X_o_ = pm.Data('X_o_', None)
                pred_o_ = pm.Data('pred_o_', None)
            else:
                # For testing the validity of the code with 100 observed variables.
                # Should make the posterior shift more clearly.
                X_o_ = pm.Data('X_o_', np.zeros(100))
                pred_o_ = pm.Data('pred_o_', np.zeros(100))

            # Outside of the plate:
            alpha_mu_X = pm.Normal('alpha_mu_X', self.prior_parameters['mu_X'],
                                   self.prior_parameters['sigma_X'])
            alpha_sigma_X = pm.TruncatedNormal('alpha_sigma_X',
                                               mu=self.prior_parameters['mu_sigma_X'],
                                               sigma=self.prior_parameters['sigma_sigma_X'],
                                               lower=0)
            alpha_mu_A = pm.Normal('alpha_mu_A', self.prior_parameters['mu_A'],
                                   self.prior_parameters['sigma_A'])
            alpha_sigma_A = pm.TruncatedNormal('alpha_sigma_A',
                                               mu=self.prior_parameters['mu_sigma_A'],
                                               sigma=self.prior_parameters['sigma_sigma_A'],
                                               lower=0)
            alpha_mu_Y = pm.Normal('alpha_mu_Y', self.prior_parameters['mu_Y'],
                                   self.prior_parameters['sigma_Y'])
            alpha_sigma_Y = pm.TruncatedNormal('alpha_sigma_Y',
                                               mu=self.prior_parameters['mu_sigma_Y'],
                                               sigma=self.prior_parameters['sigma_sigma_Y'],
                                               lower=0)
            N_E = pm.Normal('N_E', self.prior_parameters['N_E_mu'],
                            self.prior_parameters['N_E_sigma'])

            if not self.low_dim_aux_plate:
                # Full-plate model
                # Plate in model (for i in 1...plate_size):
                X = pm.Normal('X', alpha_mu_X, alpha_sigma_X, shape=self.plate_size)
                A = pm.Normal('A', self.scm_constants['d'] * X + alpha_mu_A,
                              alpha_sigma_A, shape=self.plate_size)
                Y = pm.Normal('Y', self.scm_constants['a'] +
                              self.scm_constants['b'] * X + N_E * A +
                              alpha_mu_Y, alpha_sigma_Y,
                              shape=self.plate_size)

                # Is this element-wise multiplication? Could use dot product.
                phi = pm.Deterministic('phi', pm.math.sum(X * Y) / pm.math.sum(X**2))
                # Create testing mode where a large number of observations could be input.
                pm.Normal('X_o', alpha_mu_X, alpha_sigma_X, observed=X_o_)
                phi_hat = pred_o_ / X_o_
                pm.Normal('phi_obs', phi, 1e-3, observed=phi_hat)

            else:
                # Low-dimensional auxiliary variable -based model
                # Var(S_X) = n, but pm.Normal takes STANDARD DEVIATION as parameter.
                S_X = pm.Normal('S_X', mu=0, sigma=at.sqrt(self.plate_size))
                S_Y = pm.Normal('S_Y', mu=0, sigma=at.sqrt(self.plate_size))
                S_A = pm.Normal('S_A', mu=0, sigma=at.sqrt(self.plate_size))
                Z_XX = pm.ChiSquared('Z_XX', self.plate_size - 1)  # Independent part of sum of squares (independent of sum of X)
                U_XY = pm.ChiSquared('U_XY', self.plate_size - 1)  # First part of "chi-squared difference distribution"
                V_XY = pm.ChiSquared('V_XY', self.plate_size - 1)  # Second part of "chi-squared difference distribution"
                U_XA = pm.ChiSquared('U_XA', self.plate_size - 1)  # First part of "chi-squared difference distribution"
                V_XA = pm.ChiSquared('V_XA', self.plate_size - 1)  # Second part of "chi-squared difference distribution"
                # N_E = N_E  # Just to be explicit.
                S_9 = pm.Deterministic('S_9', (S_X * S_A / self.plate_size +
                                               (U_XA - V_XA) / 2))
                S_8 = pm.Deterministic('S_8', (S_X * S_Y / self.plate_size +
                                               (U_XY - V_XY) / 2))
                S_7 = pm.Deterministic('S_7', Z_XX + S_X**2 / self.plate_size)
                S_6 = pm.Deterministic('S_6', (self.plate_size * alpha_mu_X * alpha_mu_Y +
                                               alpha_mu_X * alpha_sigma_Y * S_Y +
                                               alpha_mu_Y * alpha_sigma_X * S_X +
                                               alpha_sigma_X * alpha_sigma_Y * S_8))
                S_5 = pm.Deterministic('S_5', N_E * (self.plate_size * alpha_mu_X * alpha_mu_A +
                                               alpha_mu_X * alpha_sigma_A * S_A +
                                               alpha_mu_A * alpha_sigma_X * S_X +
                                               alpha_sigma_X * alpha_sigma_A * S_9))
                S_4 = pm.Deterministic('S_4', self.scm_constants['a'] *
                                       (self.plate_size * alpha_mu_X +
                                        alpha_sigma_X * S_X))
                # Sum of S_1, S_2, and S_3 (denominator):
                S_123 = pm.Deterministic('S_123',
                                         (self.plate_size * alpha_mu_X**2 +
                                          2 * alpha_mu_X * alpha_sigma_X * S_X +
                                          alpha_sigma_X**2 * S_7))
                # Equation for phi.
                phi = pm.Deterministic('phi',
                                       (self.scm_constants['b'] +
                                        self.scm_constants['d'] * N_E +
                                        (S_4 + S_5 + S_6) / S_123))
                pm.Normal('X_o', alpha_mu_X, alpha_sigma_X, observed=X_o_,
                          shape=X_o_.shape)
                phi_hat = pred_o_ / X_o_
                # Note that treating phi as a normally distributed RV instead of a point value
                # is a modeling trick.
                pm.Normal('phi_obs', phi, 1e-3, observed=phi_hat)
        return model

    def _check_predictive_sampling_supported(self):
        """
        Hook for subclasses to guard predictive sampling.
        """
        return

    def _update_predictive_scm(self, parameters):
        """
        Update the SCM used by direct predictive methods.
        """
        self.predictive_scm.update_parameters(parameters)

    def _extract_parameter_array(self, sample_group):
        """
        Reshape prior or posterior samples so that the leading dimension indexes
        draws while preserving any parameter dimensions.
        """
        parameter_array = {}
        for var in self.predictive_var_names:
            values = sample_group[var].values
            parameter_array[var] = values.reshape((-1,) + values.shape[2:])
        return parameter_array

    def _flatten_summary_sample(self, values):
        """
        Flatten predictive samples for scalar summary statistics.
        """
        return np.asarray(values, dtype=float).reshape(-1)

    def sample(self, X_o, pred_o, draws=1000, tune=1000,
               target_accept=.85, chains=4, max_treedepth=15,
               init="jitter+adapt_diag", rng=None, var_names=None,
               progressbar=True, compute_convergence_checks=True):
        """
        Sample from the posterior given observed data.
        Stores latest trace in self.

        Args:
            X_o (np.ndarray): Vector of observed features.
            pred_o (np.ndarray): Vector of observed predictions.
            draws (int): Size of sample to draw from the posterior.
            tune (int): Number of tuning iterations.
            target_accept (float): Target acceptance probability for NUTS.
            chains (int): Number of chains.
            max_treedepth (int): Maximum tree depth for NUTS.
            init (str): Initialization method passed to PyMC.
            var_names (list): Variables to store in the returned trace.
            progressbar (bool): Whether to show the PyMC progress bar.
            compute_convergence_checks (bool): Whether PyMC should run its
                built-in convergence checks after sampling.
        
        Returns:
            None
        """
        self.posterior_draws = draws
        # Replace placeholders with actual data:
        pm.set_data({'X_o_': X_o, 'pred_o_': pred_o}, model=self.model)
        with self.model:
            trace = pm.sample(draws=self.posterior_draws,
                              tune=tune,
                              target_accept=target_accept,
                              chains=chains,
                              random_seed=rng,
                              max_treedepth=max_treedepth,
                              init=init,
                              var_names=var_names,
                              progressbar=progressbar,
                              compute_convergence_checks=compute_convergence_checks,
                              callback=printing_callback,
                              )
        self.trace = trace
        # Store observed variable in self
        self.X_o = X_o
        return

    def sample_prior_predictive(self, draws=1000, rng=None):
        """
        Method for sampling from distribution based on the
        agent's prior beliefs.

        Args:
            draws (int): Size of sample to draw
            rng (int): Random number generator or seed for reproducibility
        """
        self._check_predictive_sampling_supported()
        self.prior_draws = draws
        with self.model:
            tmp = pm.sample_prior_predictive(draws=self.prior_draws,
                                            var_names=self.predictive_var_names,
                                            random_seed=rng)
        self.prior_samples = tmp
        parameter_array = self._extract_parameter_array(tmp.prior)
        samples = []
        for i in range(self.prior_draws):
            # Pick parameters from row i
            parameters = {key: column[i] for key, column in parameter_array.items()}
            # Update scm-parameters
            self._update_predictive_scm(parameters)
            # sample() (generates X, Y, A, mu_A_cond, sigma_A_cond):
            X, A, Y, *_ = self.predictive_scm.sample(rng=rng)
            samples.append([X, A, Y])
        self.prior_predictive_samples = samples
        return samples

    def sample_prior_predictive_do_A(self, X, A_treatment, rng=None):
        """
        Method for estimating the outcome for a patient with features X
        based on the agent's prior beliefs. Draws one potential
        outcome for every draw in trace.

        Args:
            X (np.array): Feature to condition on
            A_treatment (float): Fixed treatment value
            rng (int or rng): Random number generator or seed for reproducibility
        """
        self._check_predictive_sampling_supported()
        if self.prior_samples is None:
            self.sample_prior_predictive(rng=rng)
        parameter_array = self._extract_parameter_array(self.prior_samples.prior)
        do_A_samples = np.zeros(shape=(self.prior_draws,))
        for i in range(self.prior_draws):
            # Pick parameters from row i
            parameters = {key: column[i] for key, column in parameter_array.items()}
            # Update scm-parameters
            self._update_predictive_scm(parameters)
            # Sample prior predictive given do(A)
            # Returns tuple with samples, mean, and standard deviation
            # First (array) will not fall into an array nicely with the second (value) and third (value)
            tmp = self.predictive_scm.sample_do_A(A_treatment, X, rng=rng)
            # Only one sample, hence standard deviation will be zero and mean will equal sample value.
            # Should be sample 1000?
            do_A_samples[i] = tmp[0][0]
        # Estimate mean and standard deviation from sample
        do_A_mean = np.mean(do_A_samples)
        do_A_std = np.std(do_A_samples)
        return do_A_samples, do_A_mean, do_A_std

    def sample_prior_predictive_CATE(self, X, A1_treatment, A0_treatment, rng=None):
        """
        This method returns a sample from the prior CATE-distribution,
        and its sample mean and standard deviation.
        Currently CATE is not dependent on X and is hence ATE. It also does
        not require conditioning on X.

        Args:
            A1_treatment (float): Dosage value of higher dose treatment
            A0_treatment (float): Dosage value of lower dose treatment
            X (np.array): Feature to condition on.
            rng: seed or initialized random number generator for reproducibility
        """
        self._check_predictive_sampling_supported()
        if self.prior_samples is None:
            self.sample_prior_predictive(rng=rng)
        parameter_array = self._extract_parameter_array(self.prior_samples.prior)
        CATE_samples = np.zeros(shape=(self.prior_draws,))
        for i in range(self.prior_draws):
            # Pick parameters from row i
            parameters = {key: column[i] for key, column in parameter_array.items()}
            # Update scm-parameters
            self._update_predictive_scm(parameters)
            # sample_CATE (generates X, Y, A, mu_A_cond, sigma_A_cond)
            CATE_samples[i], *_ = self.predictive_scm.sample_CATE(X, A1_treatment, A0_treatment, rng=rng)
        # Estimate mean and standard deviation from sample
        CATE_mean = np.mean(CATE_samples)
        CATE_std = np.std(CATE_samples)
        return CATE_samples, CATE_mean, CATE_std

    def get_prior_CATE_mu(self, A1_treatment, A0_treatment):
        """
        The method returns the mean of the CATE-distribution times
        A1_treatment - A0_treatment. The current SCM is such that N_E is not
        dependent on X and the treatment effect only depends on
        N_E and size of A_treatment.

        Args:
            A1_treatment (float): Dosage value of higher dose treatment
            A0_treatment (float): Dosage value of lower dose treatment
        """
        return (A1_treatment - A0_treatment) * self.prior_parameters['N_E_mu']
    
    def sample_prior_predictive_summaries(self, rng=None):
        """
        Method that estimates mean, variance, skewness, and kurtosis
        of the variables X, A, and Y in the prior samples.
        
        Args:
            rng (int or rng): Seed or initialized random number generator for
                reproducibility. Used only if prior samples were not drawn earlier.
        """
        self._check_predictive_sampling_supported()
        if self.prior_predictive_samples is None:
            self.sample_prior_predictive(rng=rng)
        tmp_X = self._flatten_summary_sample([item[0] for item in self.prior_predictive_samples])
        tmp_A = self._flatten_summary_sample([item[1] for item in self.prior_predictive_samples])
        tmp_Y = self._flatten_summary_sample([item[2] for item in self.prior_predictive_samples])
        results = {}
        results['pri_X_mean'] = np.mean(tmp_X)
        results['pri_A_mean'] = np.mean(tmp_A)
        results['pri_Y_mean'] = np.mean(tmp_Y)
        results['pri_X_std'] = np.std(tmp_X)
        results['pri_A_std'] = np.std(tmp_A)
        results['pri_Y_std'] = np.std(tmp_Y)
        results['pri_X_skew'] = float(skew(tmp_X))
        results['pri_A_skew'] = float(skew(tmp_A))
        results['pri_Y_skew'] = float(skew(tmp_Y))
        results['pri_X_kurtosis'] = float(kurtosis(tmp_X))
        results['pri_A_kurtosis'] = float(kurtosis(tmp_A))
        results['pri_Y_kurtosis'] = float(kurtosis(tmp_Y))
        return results

    def sample_posterior_predictive(self, rng=None):
        """
        This method generates a sample from the agent's posterior beliefs.
        Using the trace stored in self, this method replaces the parameter
        values in self.predictive_scm and draws a posterior predictive
        sample from this.

        Args:
            rng (int or rng): Random number generator or seed for reproducibility

        Notes:
        Consider VECTORIZING this method.
        """
        self._check_predictive_sampling_supported()
        parameter_array = self._extract_parameter_array(self.trace.posterior)
        samples = []
        for i in range(self.posterior_draws):
            # Pick parameters from row i
            parameters = {key: column[i] for key, column in parameter_array.items()}
            # Update scm-parameters
            self._update_predictive_scm(parameters)
            # sample() (generates X, Y, A, mu_A_cond, sigma_A_cond):
            samples.append(self.predictive_scm.sample(rng=rng))
        self.posterior_predictive_samples = samples
        return samples

    def sample_posterior_predictive_CATE(self, X, A1_treatment, A0_treatment, rng=None):
        """
        This method returns a sample from the posterior predictive
        CATE-distribution, and its sample mean and standard deviation.

        Args:
            X (np.array): Feature to condition on
            A1_treatment (float): Higher dose treatment value
            A0_treatment (float): Lower dose treatment value
            rng (int or rng): Random number generator or seed for reproducibility
        """
        self._check_predictive_sampling_supported()
        parameter_array = self._extract_parameter_array(self.trace.posterior)
        CATE_samples = np.zeros(shape=(self.posterior_draws,))
        for i in range(self.posterior_draws):
            # Pick parameters from row i
            parameters = {key: column[i] for key, column in parameter_array.items()}
            # Update scm-parameters
            self._update_predictive_scm(parameters)
            # Sample (generates CATE for that version of the scm):
            CATE_samples[i], *_ = self.predictive_scm.sample_CATE(X, A1_treatment, A0_treatment, rng=rng)
        # Estimate mean and standard deviation from sample
        CATE_mean = np.mean(CATE_samples)
        CATE_std = np.std(CATE_samples)
        return CATE_samples, CATE_mean, CATE_std

    def sample_posterior_predictive_do_A(self, X, A_treatment, rng=None):
        """
        Method for estimating the outcome for a patient with features X
        based on the agent's posterior beliefs. Draws one potential
        outcome for every draw in trace.

        Args:
            X (np.array): Feature to condition on
            A_treatment (float): Fixed treatment value
            rng (int or rng): Random number generator or seed for reproducibility
        """
        self._check_predictive_sampling_supported()
        parameter_array = self._extract_parameter_array(self.trace.posterior)
        potential_outcomes = np.zeros(shape=(self.posterior_draws,))
        for i in range(self.posterior_draws):
            # Pick parameters from row i
            parameters = {key: column[i] for key, column in parameter_array.items()}
            # Update scm-parameters
            self._update_predictive_scm(parameters)
            # Sample (generates CATE for that version of the scm):
            potential_outcomes[i], *_ = self.predictive_scm.sample_do_A(A_treatment, X, rng=rng)
        # Estimate mean and standard deviation
        do_A_mean = np.mean(potential_outcomes)
        do_A_std = np.std(potential_outcomes)
        return potential_outcomes, do_A_mean, do_A_std
    
    def sample_posterior_predictive_summaries(self, rng=None):
        """
        Docstring for sample_posterior_predictive_summaries
        
        :param self: Description
        """
        self._check_predictive_sampling_supported()
        if self.posterior_predictive_samples is None:
            self.sample_posterior_predictive(rng=rng)
        tmp_X = self._flatten_summary_sample([item[0] for item in self.posterior_predictive_samples])
        tmp_A = self._flatten_summary_sample([item[1] for item in self.posterior_predictive_samples])
        tmp_Y = self._flatten_summary_sample([item[2] for item in self.posterior_predictive_samples])
        results = {}
        results['post_X_mean'] = np.mean(tmp_X)
        results['post_A_mean'] = np.mean(tmp_A)
        results['post_Y_mean'] = np.mean(tmp_Y)
        results['post_X_std'] = np.std(tmp_X)
        results['post_A_std'] = np.std(tmp_A)
        results['post_Y_std'] = np.std(tmp_Y)
        results['post_X_skew'] = float(skew(tmp_X))
        results['post_A_skew'] = float(skew(tmp_A))
        results['post_Y_skew'] = float(skew(tmp_Y))
        results['post_X_kurtosis'] = float(kurtosis(tmp_X))
        results['post_A_kurtosis'] = float(kurtosis(tmp_A))
        results['post_Y_kurtosis'] = float(kurtosis(tmp_Y))
        return results

    def save_agent_model(self, file_name='agent_model', folder='./saved_models/'):
        """
        Save the entire agent model (trace + configuration).

        Args:
            file_name (str): Base filename without extension (e.g., 'agent_model')
                Will create three files:
                - {file_name}.joblib (model configuration)
                - {file_name}_trace.nc (trace data)
                - {file_name}_prior_samples.nc (prior sample data)
            folder (str): Folder to save models into.
        """
        from pathlib import Path
        Path('./saved_models').mkdir(parents=True, exist_ok=True)
        file_name = folder + file_name
        # Save trace separately using ArviZ (if it exists)
        if self.trace is not None:
            self.trace.to_netcdf(f'{file_name}_trace.nc')
            print(f"Trace saved to {file_name}_trace.nc")
        if self.prior_samples is not None:
            self.prior_samples.to_netcdf(f'{file_name}_prior_samples.nc')
            print(f"Prior samples saved to {file_name}_prior_samples.nc")

        # Temporarily remove trace and PyMC model (can't pickle easily)
        trace_backup = self.trace
        prior_samples_backup = self.prior_samples
        model_backup = self.model
        self.trace = None
        self.prior_samples = None
        self.model = None

        # Save the rest (configuration, constants, etc.)
        joblib.dump(self, f'{file_name}.joblib')
        # Restore the attributes
        self.trace = trace_backup
        self.prior_samples = prior_samples_backup
        self.model = model_backup

        print(f"Complete agent model saved to {file_name}.joblib")

    @classmethod
    def load_agent_model(cls, file_name='./agent_model', folder='./saved_models/'):
        """
        Load a saved agent model.

        Args:
            file_name: Base filename without extension (e.g., 'agent_model')
                Expects to find:
                - {file_name}.joblib (model configuration)
                - {file_name}_trace.nc (trace data, optional)
                - {file_name}_prior_samples.nc (prior sample data, optional)
            folder (str): Folder the models are saved in

        Returns:
            agent_model.AgentModel
        """
        file_name = folder + file_name
        # Load the model structure
        agent = joblib.load(f'{file_name}.joblib')
        print("Agent model loaded from {}".format(file_name))

        # Store values that will be overwritten by __init__
        posterior_draws = agent.posterior_draws
        prior_draws = agent.prior_draws
        X_o = agent.X_o

        # Reconstruct PyMC model (it was None when saved)
        # This reinitializes agent.model based on saved configuration
        agent.__init__(
            plate_size=agent.plate_size,
            prior_parameters=agent.prior_parameters,
            scm_constants=agent.scm_constants,
            predictive_var_names=agent.predictive_var_names,
            low_dim_aux_plate=agent.low_dim_aux_plate,
            fit_intercept=getattr(agent, 'fit_intercept', False),
            test=agent.test
        )

        # Restore values that were overwritten by __init__
        agent.posterior_draws = posterior_draws
        agent.prior_draws = prior_draws
        agent.X_o = X_o

        # Load trace
        try:
            agent.trace = az.from_netcdf(f'{file_name}_trace.nc')
        except FileNotFoundError:
            agent.trace = None
            print("Trace not loaded (file {}_trace.nc does not exist.".format(file_name))
        # Load prior samples
        try:
            agent.prior_samples = az.from_netcdf(f'{file_name}_prior_samples.nc')
        except FileNotFoundError:
            agent.prior_samples = None
            print("Prior samples not loaded (file {}_prior_samples.nc does not exist).".format(file_name))
        return agent


class _AgentModelWithCovariates(AgentModel):
    """
    Base class for multivariate extensions.

    This class handles k-dimensional parameter normalization and the
    multivariate shapes of observed data and population parameters, while the
    historical no-intercept AgentModel remains scalar and visually close to
    the upstream main branch.
    """
    def __init__(self, plate_size, prior_parameters=None,
                 scm_constants=None, predictive_var_names=None,
                 low_dim_aux_plate=True, fit_intercept=False, test=False):
        use_default_scm_constants = scm_constants is None
        base_scm_constants = self._initialize_scm_constants(scm_constants)
        self.scm_constants = base_scm_constants
        base_prior_parameters = self._initialize_prior_parameters(
            prior_parameters, use_default_scm_constants)
        self.n_covariates = self._infer_n_covariates(
            base_prior_parameters, base_scm_constants)
        normalized_scm_constants = self._normalize_scm_constants(base_scm_constants)
        normalized_prior_parameters = self._normalize_prior_parameters(
            base_prior_parameters)
        super().__init__(plate_size=plate_size,
                         prior_parameters=normalized_prior_parameters,
                         scm_constants=normalized_scm_constants,
                         predictive_var_names=predictive_var_names,
                         low_dim_aux_plate=low_dim_aux_plate,
                         fit_intercept=fit_intercept,
                         test=test)

    def _covariate_prior_names(self):
        """
        Return names of X-related prior parameters.
        """
        return ('mu_X', 'sigma_X', 'mu_sigma_X', 'sigma_sigma_X')

    def _structural_prior_names(self):
        """
        Return names of structural-coefficient prior parameters.
        """
        return tuple()

    def _infer_n_covariates(self, prior_parameters, scm_constants):
        """
        Infer the number of covariates from X-prior parameters and scm
        constants.
        """
        n_covariates = 1
        for parameter_name in self._covariate_prior_names():
            parameter = np.atleast_1d(np.asarray(prior_parameters[parameter_name], dtype=float))
            if parameter.ndim != 1:
                raise ValueError(f"{parameter_name} must be scalar or one-dimensional.")
            n_covariates = max(n_covariates, parameter.size)
        for parameter_name in self._structural_prior_names():
            if parameter_name in prior_parameters:
                parameter = np.atleast_1d(np.asarray(prior_parameters[parameter_name], dtype=float))
                if parameter.ndim != 1:
                    raise ValueError(f"{parameter_name} must be scalar or one-dimensional.")
                n_covariates = max(n_covariates, parameter.size)
        for parameter_name in ('b', 'd'):
            parameter = np.atleast_1d(np.asarray(scm_constants[parameter_name], dtype=float))
            if parameter.ndim != 1:
                raise ValueError(f"{parameter_name} must be scalar or one-dimensional.")
            n_covariates = max(n_covariates, parameter.size)
        return n_covariates

    def _normalize_covariate_parameter(self, parameter, parameter_name):
        """
        Normalize scalar and one-dimensional covariate parameters.
        """
        parameter = np.atleast_1d(np.asarray(parameter, dtype=float))
        if parameter.ndim != 1:
            raise ValueError(f"{parameter_name} must be scalar or one-dimensional.")
        if parameter.size == 1 and self.n_covariates > 1:
            parameter = np.full(self.n_covariates, parameter.item(), dtype=float)
        elif parameter.size != self.n_covariates:
            text = f"{parameter_name} must have length {self.n_covariates}, not {parameter.size}."
            raise ValueError(text)
        if self.n_covariates == 1:
            return parameter.item()
        return parameter

    def _normalize_prior_parameters(self, prior_parameters):
        """
        Normalize X-related prior parameters to be consistent with k.
        """
        normalized_parameters = prior_parameters.copy()
        for parameter_name in self._covariate_prior_names():
            normalized_parameters[parameter_name] = self._normalize_covariate_parameter(
                prior_parameters[parameter_name], parameter_name)
        for parameter_name in self._structural_prior_names():
            if parameter_name in prior_parameters:
                normalized_parameters[parameter_name] = self._normalize_covariate_parameter(
                    prior_parameters[parameter_name], parameter_name)
        return normalized_parameters

    def _normalize_scm_constants(self, scm_constants):
        """
        Normalize covariate-related scm constants to be consistent with k.
        """
        normalized_constants = scm_constants.copy()
        for parameter_name in ('b', 'd'):
            normalized_constants[parameter_name] = self._normalize_covariate_parameter(
                scm_constants[parameter_name], parameter_name)
        return normalized_constants

    def _build_observed_data(self):
        """
        Create PyMC data placeholders for multivariate observed data.
        """
        if not self.test:
            if self.n_covariates == 1:
                X_o_ = pm.Data('X_o_', np.asarray(0.0))
            else:
                X_o_ = pm.Data('X_o_', np.zeros(self.n_covariates))
            pred_o_ = pm.Data('pred_o_', np.asarray(0.0))
        else:
            if self.n_covariates == 1:
                X_o_ = pm.Data('X_o_', np.zeros(100))
            else:
                X_o_ = pm.Data('X_o_', np.zeros((100, self.n_covariates)))
            pred_o_ = pm.Data('pred_o_', np.zeros(100))
        return X_o_, pred_o_

    def _build_population_parameters(self):
        """
        Build the PyMC variables shared by all multivariate graph variants.
        """
        if self.n_covariates == 1:
            alpha_mu_X = pm.Normal('alpha_mu_X', self.prior_parameters['mu_X'],
                                   self.prior_parameters['sigma_X'])
            alpha_sigma_X = pm.TruncatedNormal('alpha_sigma_X',
                                               mu=self.prior_parameters['mu_sigma_X'],
                                               sigma=self.prior_parameters['sigma_sigma_X'],
                                               lower=0)
        else:
            alpha_mu_X = pm.Normal('alpha_mu_X', self.prior_parameters['mu_X'],
                                   self.prior_parameters['sigma_X'],
                                   shape=self.n_covariates)
            alpha_sigma_X = pm.TruncatedNormal('alpha_sigma_X',
                                               mu=self.prior_parameters['mu_sigma_X'],
                                               sigma=self.prior_parameters['sigma_sigma_X'],
                                               lower=0,
                                               shape=self.n_covariates)
        alpha_mu_A = pm.Normal('alpha_mu_A', self.prior_parameters['mu_A'], self.prior_parameters['sigma_A'])
        alpha_sigma_A = pm.TruncatedNormal('alpha_sigma_A', mu=self.prior_parameters[
            'mu_sigma_A'], sigma=self.prior_parameters['sigma_sigma_A'], lower=0)
        alpha_mu_Y = pm.Normal('alpha_mu_Y', self.prior_parameters['mu_Y'], self.prior_parameters['sigma_Y'])
        alpha_sigma_Y = pm.TruncatedNormal('alpha_sigma_Y', mu=self.prior_parameters[
            'mu_sigma_Y'], sigma=self.prior_parameters['sigma_sigma_Y'], lower=0)
        N_E = pm.Normal('N_E', self.prior_parameters['N_E_mu'], self.prior_parameters['N_E_sigma'])
        return {
            'alpha_mu_X': alpha_mu_X,
            'alpha_sigma_X': alpha_sigma_X,
            'alpha_mu_A': alpha_mu_A,
            'alpha_sigma_A': alpha_sigma_A,
            'alpha_mu_Y': alpha_mu_Y,
            'alpha_sigma_Y': alpha_sigma_Y,
            'N_E': N_E,
        }


class _AgentModelWithRandomStructuralCoefficients(_AgentModelWithCovariates):
    """
    Base class for models with latent structural coefficients.

    Subclasses use N_b and N_d in the PyMC graph while keeping b and d as
    legacy SCM configuration inputs.
    """
    def _default_prior_parameters(self, use_default_scm_constants):
        """
        Return default prior parameters with latent structural coefficients.
        """
        prior_parameters = super()._default_prior_parameters(use_default_scm_constants)
        prior_parameters['mu_Y'] = 12 if use_default_scm_constants else 0
        prior_parameters.update({
            'N_b_mu': self.scm_constants['b'],
            'N_b_sigma': 0.1,
            'N_d_mu': self.scm_constants['d'],
            'N_d_sigma': 0.1,
        })
        return prior_parameters

    def _default_predictive_var_names(self):
        """
        Return variables sampled for predictive calculations.
        """
        return super()._default_predictive_var_names() + ['N_b', 'N_d']

    def _default_scm_constants(self):
        """
        Keep b and d as legacy config inputs, but remove a from the active model.
        """
        return {'a': 0, 'b': -0.1, 'd': 0.125}

    def _structural_prior_names(self):
        """
        Return names of structural-coefficient prior parameters.
        """
        return ('N_b_mu', 'N_b_sigma', 'N_d_mu', 'N_d_sigma')

    def _prepare_structural_parameters(self):
        """
        Absorb the legacy structural intercept and set structural priors.
        """
        self._absorb_scm_intercept()
        self._set_default_structural_priors()

    def _absorb_scm_intercept(self):
        """
        Remove the structural intercept from the active model.
        """
        intercept = float(np.asarray(self.scm_constants.get('a', 0), dtype=float))
        self.prior_parameters = self.prior_parameters.copy()
        mu_Y = np.asarray(self.prior_parameters['mu_Y'], dtype=float) + intercept
        if mu_Y.shape == ():
            mu_Y = mu_Y.item()
        self.prior_parameters['mu_Y'] = mu_Y
        self.scm_constants['a'] = 0

    def _set_default_structural_priors(self):
        """
        Set priors for structural coefficients when they are not provided.
        """
        self.prior_parameters = self.prior_parameters.copy()
        self.prior_parameters.setdefault('N_b_mu', self.scm_constants['b'])
        self.prior_parameters.setdefault('N_b_sigma', 0.1)
        self.prior_parameters.setdefault('N_d_mu', self.scm_constants['d'])
        self.prior_parameters.setdefault('N_d_sigma', 0.1)

    def _validate_model_configuration(self):
        """
        Validate combinations supported by latent-coefficient subclasses.
        """
        return

    def _check_predictive_sampling_supported(self):
        """
        Ensure direct predictive methods sample the latent structural coefficients.
        """
        missing_items = [item for item in ('N_b', 'N_d')
                         if item not in self.predictive_var_names]
        if missing_items:
            raise ValueError(
                "Latent-coefficient predictive methods require {} in "
                "predictive_var_names.".format(missing_items)
            )

    def _update_predictive_scm(self, parameters):
        """
        Map model-level structural coefficients to SCM field names.
        """
        parameters = parameters.copy()
        if 'N_b' in parameters:
            parameters['b'] = parameters.pop('N_b')
        if 'N_d' in parameters:
            parameters['d'] = parameters.pop('N_d')
        super()._update_predictive_scm(parameters)

    def _build_structural_coefficients(self):
        """
        Build latent structural coefficients.
        """
        if self.n_covariates == 1:
            N_b = pm.Normal('N_b', self.prior_parameters['N_b_mu'],
                            self.prior_parameters['N_b_sigma'])
            N_d = pm.Normal('N_d', self.prior_parameters['N_d_mu'],
                            self.prior_parameters['N_d_sigma'])
        else:
            N_b = pm.Normal('N_b', self.prior_parameters['N_b_mu'],
                            self.prior_parameters['N_b_sigma'],
                            shape=self.n_covariates)
            N_d = pm.Normal('N_d', self.prior_parameters['N_d_mu'],
                            self.prior_parameters['N_d_sigma'],
                            shape=self.n_covariates)
        return N_b, N_d


class AgentModelWithFullPlateFixedCoefficients(_AgentModelWithCovariates):
    """
    Agent model for the explicit multivariate OLS-without-intercept plate with
    fixed structural coefficients.

    This subclass keeps the earlier multivariate full-plate extension separate
    from the historical scalar no-intercept AgentModel.
    """
    def __init__(self, plate_size, prior_parameters=None,
                 scm_constants=None, predictive_var_names=None,
                 low_dim_aux_plate=False, fit_intercept=False, test=False):
        if fit_intercept:
            raise ValueError(
                "AgentModelWithFullPlateFixedCoefficients requires "
                "fit_intercept=False."
            )
        if low_dim_aux_plate:
            raise ValueError(
                "AgentModelWithFullPlateFixedCoefficients requires "
                "low_dim_aux_plate=False."
            )
        super().__init__(plate_size=plate_size,
                         prior_parameters=prior_parameters,
                         scm_constants=scm_constants,
                         predictive_var_names=predictive_var_names,
                         low_dim_aux_plate=False,
                         fit_intercept=False,
                         test=test)

    def _validate_model_configuration(self):
        """
        Validate combinations supported by the multivariate fixed-coefficient
        full-plate model.
        """
        if self.fit_intercept:
            raise NotImplementedError(
                "AgentModelWithFullPlateFixedCoefficients requires "
                "fit_intercept=False."
            )
        if self.low_dim_aux_plate:
            raise NotImplementedError(
                "AgentModelWithFullPlateFixedCoefficients requires "
                "low_dim_aux_plate=False."
            )
        if self.n_covariates == 1:
            raise NotImplementedError(
                "AgentModelWithFullPlateFixedCoefficients is only supported "
                "for multivariate inputs. Use AgentModel for k=1."
            )

    def _build_model(self):
        """
        Build the PyMC graph for the explicit multivariate full-plate model
        with fixed structural coefficients.
        """
        model = pm.Model()
        with model:
            X_o_, pred_o_ = self._build_observed_data()
            parameters = self._build_population_parameters()
            alpha_mu_X = parameters['alpha_mu_X']
            alpha_sigma_X = parameters['alpha_sigma_X']
            alpha_mu_A = parameters['alpha_mu_A']
            alpha_sigma_A = parameters['alpha_sigma_A']
            alpha_mu_Y = parameters['alpha_mu_Y']
            alpha_sigma_Y = parameters['alpha_sigma_Y']
            N_E = parameters['N_E']

            X = pm.Normal('X', alpha_mu_X, alpha_sigma_X,
                          shape=(self.plate_size, self.n_covariates))
            A = pm.Normal('A', at.dot(X, self.scm_constants['d']) + alpha_mu_A,
                          alpha_sigma_A, shape=self.plate_size)
            Y = pm.Normal('Y', self.scm_constants['a'] +
                          at.dot(X, self.scm_constants['b']) +
                          N_E * A + alpha_mu_Y,
                          alpha_sigma_Y, shape=self.plate_size)

            XtX = at.dot(X.T, X)
            XtY = at.dot(X.T, Y)
            phi = pm.Deterministic('phi', at.linalg.solve(XtX, XtY))
            if self.test:
                pm.Normal('X_o', alpha_mu_X, alpha_sigma_X,
                          observed=X_o_, shape=X_o_.shape)
            else:
                pm.Normal('X_o', alpha_mu_X, alpha_sigma_X,
                          observed=X_o_, shape=self.n_covariates)
            pred_reconstructed = at.dot(X_o_, phi)
            pm.Normal('phi_obs', pred_reconstructed, 1e-3, observed=pred_o_)
        return model


class AgentModelWithMultivariate(_AgentModelWithRandomStructuralCoefficients):
    """
    Agent model for the multivariate OLS-without-intercept family with latent
    structural coefficients.

    The low_dim_aux_plate flag selects between the explicit full-plate graph
    and the reduced graph using the auxiliary variables s, W, and g.
    """
    def __init__(self, plate_size, prior_parameters=None,
                 scm_constants=None, predictive_var_names=None,
                 low_dim_aux_plate=True, fit_intercept=False, test=False):
        if fit_intercept:
            raise ValueError(
                "AgentModelWithMultivariate requires fit_intercept=False."
            )
        super().__init__(plate_size=plate_size,
                         prior_parameters=prior_parameters,
                         scm_constants=scm_constants,
                         predictive_var_names=predictive_var_names,
                         low_dim_aux_plate=low_dim_aux_plate,
                         fit_intercept=False,
                         test=test)

    def _validate_model_configuration(self):
        """
        Validate combinations supported by the no-intercept latent-coefficient
        family.
        """
        if self.fit_intercept:
            raise NotImplementedError(
                "AgentModelWithMultivariate requires fit_intercept=False."
            )

    def _build_model(self):
        """
        Build the PyMC graph for the no-intercept latent-coefficient model.
        """
        model = pm.Model()
        with model:
            X_o_, pred_o_ = self._build_observed_data()
            parameters = self._build_population_parameters()
            alpha_mu_X = parameters['alpha_mu_X']
            alpha_sigma_X = parameters['alpha_sigma_X']
            alpha_mu_A = parameters['alpha_mu_A']
            alpha_sigma_A = parameters['alpha_sigma_A']
            alpha_mu_Y = parameters['alpha_mu_Y']
            alpha_sigma_Y = parameters['alpha_sigma_Y']
            N_E = parameters['N_E']
            N_b, N_d = self._build_structural_coefficients()

            if not self.low_dim_aux_plate:
                if self.n_covariates == 1:
                    X = pm.Normal('X', alpha_mu_X, alpha_sigma_X,
                                  shape=self.plate_size)
                    A = pm.Normal('A', N_d * X + alpha_mu_A,
                                  alpha_sigma_A, shape=self.plate_size)
                    Y = pm.Normal('Y', N_b * X + N_E * A + alpha_mu_Y,
                                  alpha_sigma_Y, shape=self.plate_size)

                    phi = pm.Deterministic(
                        'phi',
                        pm.math.sum(X * Y) / pm.math.sum(X**2)
                    )
                    pm.Normal('X_o', alpha_mu_X, alpha_sigma_X,
                              observed=X_o_, shape=X_o_.shape)
                    phi_hat = pred_o_ / X_o_
                    pm.Normal('phi_obs', phi, 1e-3, observed=phi_hat)
                else:
                    X = pm.Normal('X', alpha_mu_X, alpha_sigma_X,
                                  shape=(self.plate_size, self.n_covariates))
                    A = pm.Normal('A', at.dot(X, N_d) + alpha_mu_A,
                                  alpha_sigma_A, shape=self.plate_size)
                    Y = pm.Normal('Y', at.dot(X, N_b) + N_E * A + alpha_mu_Y,
                                  alpha_sigma_Y, shape=self.plate_size)

                    XtX = at.dot(X.T, X)
                    XtY = at.dot(X.T, Y)
                    phi = pm.Deterministic('phi', at.linalg.solve(XtX, XtY))
                    if self.test:
                        pm.Normal('X_o', alpha_mu_X, alpha_sigma_X,
                                  observed=X_o_, shape=X_o_.shape)
                    else:
                        pm.Normal('X_o', alpha_mu_X, alpha_sigma_X,
                                  observed=X_o_, shape=self.n_covariates)
                    pred_reconstructed = at.dot(X_o_, phi)
                    pm.Normal('phi_obs', pred_reconstructed, 1e-3,
                              observed=pred_o_)
            else:
                if self.n_covariates == 1:
                    alpha_mu_X_vector = at.stack([alpha_mu_X])
                    alpha_sigma_X_vector = at.stack([alpha_sigma_X])
                    beta = at.stack([N_b + N_E * N_d])
                else:
                    alpha_mu_X_vector = alpha_mu_X
                    alpha_sigma_X_vector = alpha_sigma_X
                    beta = N_b + N_E * N_d

                s = pm.Normal('s', mu=0, sigma=at.sqrt(self.plate_size),
                              shape=self.n_covariates)
                W_chol = pm.WishartBartlett('W_chol', S=np.eye(self.n_covariates),
                                            nu=self.plate_size - 1,
                                            return_cholesky=True)
                g = pm.Normal('g', mu=0, sigma=1, shape=self.n_covariates)

                c = N_E * alpha_mu_A + alpha_mu_Y
                tau = at.sqrt(N_E**2 * alpha_sigma_A**2 + alpha_sigma_Y**2)
                S_x = self.plate_size * alpha_mu_X_vector + alpha_sigma_X_vector * s
                W = at.dot(W_chol, W_chol.T)
                G = (alpha_sigma_X_vector[:, None] * W *
                     alpha_sigma_X_vector[None, :] +
                     at.outer(S_x, S_x) / self.plate_size)
                G_chol = at.linalg.cholesky(G)
                mean_term = c * at.linalg.solve(G, S_x)
                noise_term = tau * at.slinalg.solve_triangular(G_chol.T, g, lower=False)
                phi_vector = beta + mean_term + noise_term

                if self.n_covariates == 1:
                    phi = pm.Deterministic('phi', phi_vector[0])
                    pm.Normal('X_o', alpha_mu_X, alpha_sigma_X,
                              observed=X_o_, shape=X_o_.shape)
                    phi_hat = pred_o_ / X_o_
                    pm.Normal('phi_obs', phi, 1e-3, observed=phi_hat)
                else:
                    phi = pm.Deterministic('phi', phi_vector)
                    if self.test:
                        pm.Normal('X_o', alpha_mu_X, alpha_sigma_X,
                                  observed=X_o_, shape=X_o_.shape)
                    else:
                        pm.Normal('X_o', alpha_mu_X, alpha_sigma_X,
                                  observed=X_o_, shape=self.n_covariates)
                    pred_reconstructed = at.dot(X_o_, phi)
                    pm.Normal('phi_obs', pred_reconstructed, 1e-3, observed=pred_o_)
        return model


class AgentModelWithIntercept(_AgentModelWithRandomStructuralCoefficients):
    """
    Agent model for the OLS-with-intercept family with latent structural
    coefficients.

    The low_dim_aux_plate flag selects between the explicit full-plate graph
    and the reduced graph with auxiliary variables s, W, g, and h.
    """
    def __init__(self, plate_size, prior_parameters=None,
                 scm_constants=None, predictive_var_names=None,
                 low_dim_aux_plate=True, fit_intercept=True, test=False):
        if not fit_intercept:
            raise ValueError("AgentModelWithIntercept requires fit_intercept=True.")
        super().__init__(plate_size=plate_size,
                         prior_parameters=prior_parameters,
                         scm_constants=scm_constants,
                         predictive_var_names=predictive_var_names,
                         low_dim_aux_plate=low_dim_aux_plate,
                         fit_intercept=True,
                         test=test)

    def _build_model(self):
        """
        Build the PyMC graph for the intercept latent-coefficient model.
        """
        model = pm.Model()
        with model:
            X_o_, pred_o_ = self._build_observed_data()
            parameters = self._build_population_parameters()
            alpha_mu_X = parameters['alpha_mu_X']
            alpha_sigma_X = parameters['alpha_sigma_X']
            alpha_mu_A = parameters['alpha_mu_A']
            alpha_sigma_A = parameters['alpha_sigma_A']
            alpha_mu_Y = parameters['alpha_mu_Y']
            alpha_sigma_Y = parameters['alpha_sigma_Y']
            N_E = parameters['N_E']
            N_b, N_d = self._build_structural_coefficients()

            if not self.low_dim_aux_plate:
                if self.n_covariates == 1:
                    X = pm.Normal('X', alpha_mu_X, alpha_sigma_X,
                                  shape=self.plate_size)
                    A = pm.Normal('A', N_d * X + alpha_mu_A,
                                  alpha_sigma_A, shape=self.plate_size)
                    Y = pm.Normal('Y', N_b * X + N_E * A + alpha_mu_Y,
                                  alpha_sigma_Y, shape=self.plate_size)

                    X_mean = at.mean(X)
                    Y_mean = at.mean(Y)
                    X_centered = X - X_mean
                    Y_centered = Y - Y_mean
                    phi = pm.Deterministic(
                        'phi',
                        pm.math.sum(X_centered * Y_centered) /
                        pm.math.sum(X_centered**2)
                    )
                    alpha = pm.Deterministic('alpha', Y_mean - phi * X_mean)
                    pm.Normal('X_o', alpha_mu_X, alpha_sigma_X,
                              observed=X_o_, shape=X_o_.shape)
                    pred_reconstructed = alpha + X_o_ * phi
                else:
                    X = pm.Normal('X', alpha_mu_X, alpha_sigma_X,
                                  shape=(self.plate_size, self.n_covariates))
                    A = pm.Normal('A', at.dot(X, N_d) + alpha_mu_A,
                                  alpha_sigma_A, shape=self.plate_size)
                    Y = pm.Normal('Y', at.dot(X, N_b) + N_E * A + alpha_mu_Y,
                                  alpha_sigma_Y, shape=self.plate_size)

                    X_mean = at.mean(X, axis=0)
                    Y_mean = at.mean(Y)
                    X_centered = X - X_mean
                    Y_centered = Y - Y_mean
                    XtX = at.dot(X_centered.T, X_centered)
                    XtY = at.dot(X_centered.T, Y_centered)
                    phi = pm.Deterministic('phi', at.linalg.solve(XtX, XtY))
                    alpha = pm.Deterministic('alpha', Y_mean - at.dot(X_mean, phi))
                    if self.test:
                        pm.Normal('X_o', alpha_mu_X, alpha_sigma_X,
                                  observed=X_o_, shape=X_o_.shape)
                    else:
                        pm.Normal('X_o', alpha_mu_X, alpha_sigma_X,
                                  observed=X_o_, shape=self.n_covariates)
                    pred_reconstructed = alpha + at.dot(X_o_, phi)
            else:
                if self.n_covariates == 1:
                    alpha_mu_X_vector = at.stack([alpha_mu_X])
                    alpha_sigma_X_vector = at.stack([alpha_sigma_X])
                    beta = at.stack([N_b + N_E * N_d])
                else:
                    alpha_mu_X_vector = alpha_mu_X
                    alpha_sigma_X_vector = alpha_sigma_X
                    beta = N_b + N_E * N_d

                s = pm.Normal('s', mu=0, sigma=at.sqrt(self.plate_size),
                              shape=self.n_covariates)
                W_chol = pm.WishartBartlett('W_chol', S=np.eye(self.n_covariates),
                                            nu=self.plate_size - 1,
                                            return_cholesky=True)
                g = pm.Normal('g', mu=0, sigma=1, shape=self.n_covariates)
                h = pm.Normal('h', mu=0, sigma=1)

                c = N_E * alpha_mu_A + alpha_mu_Y
                tau = at.sqrt(N_E**2 * alpha_sigma_A**2 + alpha_sigma_Y**2)
                W_minus_half_g = at.slinalg.solve_triangular(W_chol.T, g, lower=False)
                phi_vector = beta + tau * W_minus_half_g / alpha_sigma_X_vector
                alpha = pm.Deterministic(
                    'alpha',
                    c + tau * h / at.sqrt(self.plate_size) -
                    tau * at.dot(alpha_mu_X_vector / alpha_sigma_X_vector +
                                 s / self.plate_size,
                                 W_minus_half_g)
                )
                if self.n_covariates == 1:
                    phi = pm.Deterministic('phi', phi_vector[0])
                    pm.Normal('X_o', alpha_mu_X, alpha_sigma_X,
                              observed=X_o_, shape=X_o_.shape)
                    pred_reconstructed = alpha + X_o_ * phi
                else:
                    phi = pm.Deterministic('phi', phi_vector)
                    if self.test:
                        pm.Normal('X_o', alpha_mu_X, alpha_sigma_X,
                                  observed=X_o_, shape=X_o_.shape)
                    else:
                        pm.Normal('X_o', alpha_mu_X, alpha_sigma_X,
                                  observed=X_o_, shape=self.n_covariates)
                    pred_reconstructed = alpha + at.dot(X_o_, phi)
            pm.Normal('phi_obs', pred_reconstructed, 1e-3, observed=pred_o_)
        return model
