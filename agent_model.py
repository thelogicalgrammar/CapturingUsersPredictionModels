"""
Better name: bayesian_agent?
-Could initialize with an SCM that represents the
 agent's internal representation of the world.
-Class could store the trace and perhaps using this
 model the predictions based on the SCM.
-This would make the class contain both the agent's
 beliefs and thought processes (how to update beliefs
 based on new data).

The Bayesian models specified here correspond to how an ideal
Bayesian agent would update his beliefs given that he sees
one or several observations and receives their predictions
that are based on an OLS regression model without intercept.

The agent is considered to have a separate SCM which uses
the agent's (prior or posterior) beliefs to choose the best
treatment for a patient.

Technically only one value of X should be allowed after update.
-Is there any point of allowing for sampling for other X-values?
-The model is NOT VALID for other X values.
-Don't block this (yet), because that would prevent updating with
 e.g. 100 patients' info.

Could be interesting to estimate variance for CATE using sampling
as we should expect correlation between E(Y|X,do(A1)) and E(Y|X,do(A0))
but do not know how large that is.
This applies to both prior and posterior.
"""
import joblib
import pymc as pm
import numpy as np
import pytensor.tensor as at
import arviz as az
import scm

DEBUGGING = False

def printing_callback(trace, draw):
    # For debugging.
    if DEBUGGING:
        print(f"Completed draw {draw.draw_idx} from chain {draw.chain}.")
        print("Shape of X: {}".format(trace.get_values('X').shape))
        print("Shape of Y: {}".format(trace.get_values('Y').shape))


class AgentModel():
    def __init__(self, plate_size, prior_parameters=None, 
                 scm_constants=None, predictive_var_names=None,
                 low_dim_aux_plate=True, test=False):
        """
        Class for agent models.

        The structural equations are:
        X_i = N_{Xi}
        A_i = N_{CC} X_i + N_{Ai}
        Y_i = a + N_{RC} X_i + N_{E} A_i + N_{Yi}
        where N_{Xi}, N_{Ai}, N_{Yi} are normal noise variables drawn J times
        for every instantiation of the plate and N_{E}, N_{CC}, and N_{RC} are
        normal variables drawn once per instantiation of the plate.

        'CC' in N_{CC} implies Historical Current Care
        'RC' in N_{RC} implies Risk Coefficient linking feature to outcome.

        Args:
        plate_size (int): size of the plate (number of i.i.d. samples).
        scm_constants (dict): dictionary with structural equation scm_constant 'a'.
         If scm_constants is None, then it defaults to values in the paper (28.8.2025).
        prior_parameters (dict): dictionary with parameters for prior distributions
         If prior_parameters is None, then it defaults to values in the paper (28.8.2025).
        low_dim_aux_plate (bool): If true the plate is estimated using low-dimensional
         auxiliary variables. This speeds up inference. If false, the plate is explicitly 
         modeled.
        test (bool): If true, the model is set up in testing mode enabling multiple observed
         data points to be input. This is mainly for testing that the model works as intended.
        """
        self.plate_size = plate_size
        if scm_constants is None:
            scm_constants = {'a': 12}  #, 'b': -0.1}  #, 'd': 0.125}
            # N_{E} is essentially replacing "c"
        if prior_parameters is None:
            # Use default values.
            prior_parameters = {'mu_X': 80, 'sigma_X': 1,
                                'lambda_X': 10,
                                'mu_A': 10, 'sigma_A': 1,
                                'lambda_A': 10,
                                'mu_Y': 0, 'sigma_Y': 1,
                                'lambda_Y': 1,
                                'N_E_mu': 20, 'N_E_sigma': 1,
                                'N_CC_mu': 0.125, 'N_CC_sigma': 0.1,
                                'N_RC_mu': -0.1, 'N_RC_sigma': 0.1}
        self.prior_parameters = prior_parameters
        self.low_dim_aux_plate = low_dim_aux_plate
        self.test = test
        # SCM parameters to update in prior predictive and posterior predictive sampling
        # and an SCM which variables are updated with samples from the agent's beliefs
        # for predictive sampling:
        self.scm_constants = scm_constants
        self.predictive_scm = scm.SCM(**scm_constants)
        self.predictive_var_names = predictive_var_names
        self.trace = None  # Placeholder for trace
        self.prior_samples = None  # Placeholder for prior samples
        self.posterior_draws = None  # Number of draws in trace
        self.prior_draws = None  # Number of draws in prior samples
        self.X_o = None  # Placeholder for observed observation.

        self.model = pm.Model()
        with self.model:
            # Note that the normal distribution in PyMC is parameterized
            # with mu and sigma (not variance, sigma^2).
            # Placeholders for observed data:
            # Named with underscore to distinguish from random variables.
            if not test:
                # Default flow.
                X_o_ = pm.Data('X_o_', None)
                pred_o_ = pm.Data('pred_o_', None)
            else:
                # For testing the validity of the code with 100 observed variables.
                # Should make the posterior shift more clearly.
                X_o_ = pm.Data('X_o_', np.zeros(100))
                pred_o_ = pm.Data('pred_o_', np.zeros(100))

            # Outside of the plate:
            alpha_mu_X = pm.Normal('alpha_mu_X', self.prior_parameters['mu_X'], self.prior_parameters['sigma_X'])
            # Is an exponential distribution well-behaved as a prior?
            alpha_sigma_X = pm.Exponential('alpha_sigma_X', self.prior_parameters['lambda_X'])
            alpha_mu_A = pm.Normal('alpha_mu_A', self.prior_parameters['mu_A'], self.prior_parameters['sigma_A'])
            alpha_sigma_A = 1  # pm.Exponential('sigma_A', 10)  # sigma_A is not learned.
            alpha_mu_Y = pm.Normal('alpha_mu_Y', self.prior_parameters['mu_Y'], self.prior_parameters['sigma_Y'])
            alpha_sigma_Y = pm.Exponential('alpha_sigma_Y', self.prior_parameters['lambda_Y'])
            N_E = pm.Normal('N_E', self.prior_parameters['N_E_mu'], self.prior_parameters['N_E_sigma'])
            N_CC = pm.Normal('N_CC', self.prior_parameters['N_CC_mu'], self.prior_parameters['N_CC_sigma'])
            N_RC = pm.Normal('N_RC', self.prior_parameters['N_RC_mu'], self.prior_parameters['N_RC_sigma'])

            if not low_dim_aux_plate:
                # Full-plate model
                # Plate in model (for i in 1...plate_size):
                X = pm.Normal('X', alpha_mu_X, alpha_sigma_X, shape=self.plate_size)
                A = pm.Normal('A', N_CC * X + alpha_mu_A, alpha_sigma_A, shape=self.plate_size)
                Y = pm.Normal('Y', self.scm_constants['a'] + N_RC * X + N_E * A + alpha_mu_Y, alpha_sigma_Y, shape=self.plate_size)

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
                S_7 = pm.Deterministic('S_7', Z_XX - S_X**2 / self.plate_size)
                S_6 = pm.Deterministic('S_6', (self.plate_size * alpha_mu_X * alpha_mu_Y +
                                            alpha_mu_X * alpha_sigma_Y * S_Y +
                                            alpha_mu_Y * alpha_sigma_X * S_X +
                                            alpha_sigma_X * alpha_sigma_Y * S_8))
                S_5 = pm.Deterministic('S_5', N_E * (self.plate_size * alpha_mu_X * alpha_mu_A +
                                            alpha_mu_X * alpha_sigma_A * S_A +
                                            alpha_mu_A * alpha_sigma_X * S_X +
                                            alpha_sigma_X * alpha_sigma_A * S_9))
                S_4 = pm.Deterministic('S_4', self.scm_constants['a'] * (self.plate_size * alpha_mu_X + alpha_sigma_X * S_X))
                # Sum of S_1, S_2, and S_3 (denominator):
                S_123 = pm.Deterministic('S_123', (self.plate_size * alpha_mu_X**2 +  # S_1
                                                 2 * alpha_mu_X * alpha_sigma_X * S_X +  # S_2
                                                 alpha_sigma_X**2 * S_7))  # S_3
                # Equation for phi.
                phi = pm.Deterministic('phi', (N_RC + N_CC * N_E +\
                                        (S_4 + S_5 + S_6) / S_123))
                pm.Normal('X_o', alpha_mu_X, alpha_sigma_X, observed=X_o_, shape=X_o_.shape)  # Shape is already defined at pm.Data(). Probably cannot be changed here.
                phi_hat = pred_o_ / X_o_
                # Note that treating phi as a normally distributed RV instead of a point value
                # is a modeling trick.
                pm.Normal('phi_obs', phi, 1e-3, observed=phi_hat)

    def sample(self, X_o, pred_o, draws=1000, rng=None):
        """
        Sample from the posterior given observed data.
        Stores latest trace in self.

        Args:
        X_o (np.ndarray): Vector of observed features.
        pred_o (np.ndarray): Vector of observed predictions.
        draws (int): Size of sample to draw from the posterior.
        """
        self.posterior_draws = draws
        # Replace placeholders with actual data:
        pm.set_data({'X_o_': X_o, 'pred_o_': pred_o}, model=self.model)
        with self.model:
            trace = pm.sample(draws=self.posterior_draws,
                              tune=1000,
                              target_accept=.95,
                              chains=4,
                              random_seed=rng,
                              max_treedepth=15,
                              init="jitter+adapt_diag",
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
        self.prior_draws = draws
        with self.model:
            tmp = pm.sample_prior_predictive(draws=self.prior_draws,
                                             var_names=self.predictive_var_names,
                                             random_seed=rng)
        self.prior_samples = tmp
        parameter_array = {var: tmp.prior[var].values.flatten() for var in self.predictive_var_names}
        sample = []
        for i in range(self.prior_draws):
            # Pick parameters from row i
            parameters = {key: column[i] for key, column in parameter_array.items()}
            # Update scm-parameters
            self.predictive_scm.update_parameters(parameters)
            # sample() (generates X, Y, A, mu_A_cond, sigma_A_cond):
            X, A, Y, *_ = self.predictive_scm.sample(rng=rng)
            sample.append([X, A, Y])
        return sample
    
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
        if self.prior_samples is None:
            self.sample_prior_predictive(rng=rng)
        parameter_array = {var: self.prior_samples.prior[var].values.flatten() for var in self.predictive_var_names}
        CATE_samples = np.zeros(shape=(self.prior_draws,))
        for i in range(self.prior_draws):
            # Pick parameters from row i
            parameters = {key: column[i] for key, column in parameter_array.items()}
            # Update scm-parameters
            self.predictive_scm.update_parameters(parameters)
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
        """
        return (A1_treatment - A0_treatment) * self.prior_parameters['N_E_mu']

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
        parameter_array = {var: self.trace.posterior[var].values.flatten() for var in self.predictive_var_names}
        samples = []
        for i in range(self.posterior_draws):
            # Pick parameters from row i
            parameters = {key: column[i] for key, column in parameter_array.items()}
            # Update scm-parameters
            self.predictive_scm.update_parameters(parameters)
            # sample() (generates X, Y, A, mu_A_cond, sigma_A_cond):
            samples.append(self.predictive_scm.sample(rng=rng))
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
        parameter_array = {var: self.trace.posterior[var].values.flatten() for var in self.predictive_var_names}
        CATE_samples = np.zeros(shape=(self.posterior_draws,))
        for i in range(self.posterior_draws):
            # Pick parameters from row i
            parameters = {key: column[i] for key, column in parameter_array.items()}
            # Update scm-parameters
            self.predictive_scm.update_parameters(parameters)
            # Sample (generates CATE for that version of the scm):
            CATE_samples[i], *_ = self.predictive_scm.sample_CATE(X, A1_treatment, A0_treatment, rng=rng)
        # Estimate mean and standard deviation from sample
        CATE_mean = np.mean(CATE_samples)
        CATE_std = np.std(CATE_samples)
        return CATE_samples, CATE_mean, CATE_std

    def sample_posterior_do_A(self, X, A_treatment, rng=None):
        """
        Method for estimating the outcome for a patient with features X
        based on the agent's posterior beliefs. Draws one potential
        outcome for every draw in trace.

        Args:
        X (np.array): Feature to condition on
        A_treatment (float): Fixed treatment value
        rng (int or rng): Random number generator or seed for reproducibility
        """
        parameter_array = {var: self.trace.posterior[var].values.flatten() for var in self.predictive_var_names}
        potential_outcomes = np.zeros(shape=(self.posterior_draws,))
        for i in range(self.posterior_draws):
            # Pick parameters from row i
            parameters = {key: column[i] for key, column in parameter_array.items()}
            # Update scm-parameters
            self.predictive_scm.update_parameters(parameters)
            # Sample (generates CATE for that version of the scm):
            potential_outcomes[i], *_ = self.predictive_scm.sample_do_A(X, A_treatment, rng=rng)
        return potential_outcomes

    def save_agent_model(self, file_name='agent_model', folder='./saved_models/'):
        """
        Save the entire agent model (trace + configuration).

        Args:
        file_name (str): Base filename without extension (e.g., 'agent_model')
            Will create two files:
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
        AgentModel instance
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
