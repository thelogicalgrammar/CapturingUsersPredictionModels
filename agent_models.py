"""
Experiment models are defined here.
"""
import pymc as pm
import warnings
import pytensor.tensor as at
from pytensor.tensor import log1p, softplus, sigmoid, logaddexp, dot

DEBUGGING = False


def printing_callback(trace, draw):
    # For debugging.
    if DEBUGGING:
        print(f"Completed draw {draw.draw_idx} from chain {draw.chain}.")
        print("Shape of X: {}".format(trace.get_values('X').shape))
        print("Shape of Y: {}".format(trace.get_values('Y').shape))


class AgentModel():
    def __init__(self, plate_size, constants=None, prior_parameters=None):
        """
        Class for agent models.

        The structural equations are:
        X_i = N_{Xi}
        A_i = d X_i + N_{Ai}
        Y_i = a + b X_i + N_{E} A_i + N_{Yi}
        where N_{Xi}, N_{Ai}, N_{Yi} are normal noise variables drawn J times
        for every instantiation of the plate and N_{E} is a normal variable
        drawn once per instantiation of the plate.

        Args:
        plate_size (int): size of the plate (number of i.i.d. samples).
        constants (dict): dictionary with structural equation constants 'a', 'b', and 'd'.
         If constants is None, then it defaults to values in the paper (28.8.2025).
        prior_parameters (dict): dictionary with parameters for prior distributions
         If prior_parameters is None, then it defaults to values in the paper (28.8.2025).
        """
        self.plate_size = plate_size
        if constants is None:
            constants = {'a': 12, 'b': -0.1, 'd': 0.125}
            # N_{E} is essentially replacing "c"
        if prior_parameters is None:
            prior_parameters = {'mu_X': 80, 'sigma_X': 1,
                                'lambda_X': 10,
                                'mu_A': 10, 'sigma_A': 1,
                                'lambda_A': 10,
                                'mu_Y': 0, 'sigma_Y': 1,
                                'lambda_Y': 1,
                                'N_E_mu': 20, 'N_E_sigma': 1}
        self.constants = constants
        self.prior_parameters = prior_parameters

        self.model = pm.Model()
        with self.model:
            # Note that the normal distribution in PyMC is parameterized
            # with mu and sigma (not variance).
            # Placeholders for observed data:
            # Named with underscore to distinguish from random variables.
            X_o_ = pm.Data('X_o_', None)
            pred_o_ = pm.Data('pred_o_', None)

            # Outside of the plate:
            alpha_mu_X = pm.Normal('alpha_mu_X', self.prior_parameters['mu_X'], self.prior_parameters['sigma_X'])
            # Is an exponential distribution well-behaved as a prior?
            alpha_sigma_X = pm.Exponential('alpha_sigma_X', self.prior_parameters['lambda_X'])
            alpha_mu_A = pm.Normal('alpha_mu_A', self.prior_parameters['mu_A'], self.prior_parameters['sigma_A'])
            alpha_sigma_A = 1  # pm.Exponential('sigma_A', 10)  # sigma_A is not learned.
            alpha_mu_Y = pm.Normal('alpha_mu_Y', self.prior_parameters['mu_Y'], self.prior_parameters['sigma_Y'])
            alpha_sigma_Y = pm.Exponential('alpha_sigma_Y', self.prior_parameters['lambda_Y'])
            N_E = pm.Normal('N_E', self.prior_parameters['N_E_mu'], self.prior_parameters['N_E_sigma'])

            # Plate in model (for i in 1...plate_size):
            X = pm.Normal('X', alpha_mu_X, alpha_sigma_X, shape=self.plate_size)
            # Is it faster to have a Normal distribution with separate parameters (here X_i * d)
            # or to sample from a Standard Normal and then scale them?
            A = pm.Normal('A', self.constants['d'] * X + alpha_mu_A, alpha_sigma_A, shape=self.plate_size)
            Y = pm.Normal('Y', self.constants['a'] + self.constants['b'] * X + N_E * A + alpha_mu_Y, alpha_sigma_Y, shape=self.plate_size)

            # Is this element-wise multiplication? Could use dot product.
            mu_M = pm.Deterministic('mu_M', pm.math.sum(X * Y) / pm.math.sum(X**2))
            sigma_M = pm.Deterministic('sigma_M', pm.math.sqrt(pm.math.sum((mu_M * X - Y)**2) / (self.plate_size - 1)))  # Variance with Bessel's correction

            # What should these be called?
            X_o = pm.Normal('X_o', alpha_mu_X, alpha_sigma_X, observed=X_o_)
            pred_o = pm.Normal('pred_o', mu_M * X_o, sigma_M, observed=pred_o_)

    def sample(self, X_o, pred_o, draws=1000):
        """
        Sample from the posterior given observed data.

        Args:
        X_o (np.ndarray): Vector of observed features.
        pred_o (np.ndarray): Vector of observed predictions.
        draws (int): number of samples to draw from the posterior.
        """
        # Replace placeholders with actual data:
        pm.set_data({'X_o_': X_o, 'pred_o_': pred_o}, model=self.model)
        with self.model:
            trace = pm.sample(draws=draws,
                              tune=1000,
                              target_accept=.99,
                              chains=4,
                              random_seed=42,  # Fixed for reproducibility.
                              max_treedepth=15,
                              init="jitter+adapt_diag",
                              callback=printing_callback,
                              )
        return trace
