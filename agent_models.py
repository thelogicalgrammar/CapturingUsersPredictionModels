"""
Experiment models are defined here.
"""
import pymc as pm
import numpy as np
import pytensor.tensor as at

DEBUGGING = False


def printing_callback(trace, draw):
    # For debugging.
    if DEBUGGING:
        print(f"Completed draw {draw.draw_idx} from chain {draw.chain}.")
        print("Shape of X: {}".format(trace.get_values('X').shape))
        print("Shape of Y: {}".format(trace.get_values('Y').shape))


class AgentModel():
    def __init__(self, plate_size, constants=None, prior_parameters=None, low_dim_aux_plate=True, test=False):
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
        low_dim_aux_plate (bool): If true the plate is estimated using low-dimensional
         auxiliary variables. This speeds up inference. If false, the plate is explicitly 
         modeled.
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
            if not test:
                X_o_ = pm.Data('X_o_', None)
                pred_o_ = pm.Data('pred_o_', None)
            else:
                # For a test with 100 observed variables.
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

            if not low_dim_aux_plate:
                # Plate in model (for i in 1...plate_size):
                X = pm.Normal('X', alpha_mu_X, alpha_sigma_X, shape=self.plate_size)
                # Is it faster to have a Normal distribution with separate parameters (here X_i * d)
                # or to sample from a Standard Normal and then scale them?
                A = pm.Normal('A', self.constants['d'] * X + alpha_mu_A, alpha_sigma_A, shape=self.plate_size)
                Y = pm.Normal('Y', self.constants['a'] + self.constants['b'] * X + N_E * A + alpha_mu_Y, alpha_sigma_Y, shape=self.plate_size)

                # Is this element-wise multiplication? Could use dot product.
                phi = pm.Deterministic('phi', pm.math.sum(X * Y) / pm.math.sum(X**2))
                # Create testing mode where a large number of observations could be input.
                pm.Normal('X_o', alpha_mu_X, alpha_sigma_X, observed=X_o_)
                phi_hat = pred_o_ / X_o_
                pm.Normal('phi_obs', phi, 1e-3, observed=phi_hat)

            else:
                # Derivation has some errors. Missing squareroots for some self.plate_size factors, then the dependent parts of the 
                # normal product distributed RVs are missing. Further the dependent part of the chi-squared RV should be 
                # epsilon_X^2, not epsilon_X. Phi should also be estimated by summing first and dividing last.
                # Maybe move up constant for S_4 to match equations in paper.
                # Maybe replace all variable names with corresponding from paper? Unify notation.
                # Maybe change \epsilon_{S_X} to \sim \mathcal{N}(0, n) to avoid some squareroots.
                # -Are Fausto and Giovanni ok with that?
                epsilon_SX = pm.Normal('epsilon_SX', 0, self.plate_size)
                epsilon_SY = pm.Normal('epsilon_SY', 0, self.plate_size)
                epsilon_SA = pm.Normal('epsilon_SA', 0, self.plate_size)
                gamma_XX = pm.ChiSquared('gamma_XX', self.plate_size - 1)  # Independent part of sum of squares (independent of sum of X). \gamma_{XX}
                gamma_XY_a = pm.ChiSquared('gamma_XY_a', self.plate_size - 1)  # First part of "chi-squared difference distribution" \gamma_{XYa}
                gamma_XY_b = pm.ChiSquared('gamma_XY_b', self.plate_size - 1)  # Second part of "chi-squared difference distribution" \gamma_{XYb}
                gamma_XA_a = pm.ChiSquared('gamma_XA_a', self.plate_size - 1)  # First part of "chi-squared difference distribution" \gamma_{XAa}
                gamma_XA_b = pm.ChiSquared('gamma_XA_b', self.plate_size - 1)  # Second part of "chi-squared difference distribution" \gamma_{XAb}
                # All alpha_mu and alpha_sigma goes into a function.
                # N_E = N_E  # Just to be explicit.
                S_9 = pm.Deterministic('S_9', (epsilon_SX * epsilon_SA / self.plate_size +
                                            (gamma_XA_a - gamma_XA_b) / 2))
                S_8 = pm.Deterministic('S_8', (epsilon_SX * epsilon_SY / self.plate_size +
                                            (gamma_XY_a - gamma_XY_b) / 2))
                S_7 = pm.Deterministic('S_7', gamma_XX - epsilon_SX**2 / self.plate_size)
                S_6 = pm.Deterministic('S_6', (self.plate_size * alpha_mu_X * alpha_mu_Y +
                                            alpha_mu_X * alpha_sigma_Y * epsilon_SY +
                                            alpha_mu_Y * alpha_sigma_X * epsilon_SX +
                                            alpha_sigma_X * alpha_sigma_Y * S_8))
                S_5 = pm.Deterministic('S_5', N_E * (self.plate_size * alpha_mu_X * alpha_mu_A +
                                            alpha_mu_X * alpha_sigma_A * epsilon_SA +
                                            alpha_mu_A * alpha_sigma_X * epsilon_SX +
                                            alpha_sigma_X * alpha_sigma_A * S_9))
                S_4 = pm.Deterministic('S_4', self.constants['a'] * (self.plate_size * alpha_mu_X + alpha_sigma_X * epsilon_SX))
                # Sum of S_1, S_2, and S_3 (denominator):
                S_123 = pm.Deterministic('sum_X_squared', (self.plate_size * alpha_mu_X**2 +  # S_1
                                                 2 * alpha_mu_X * alpha_sigma_X * epsilon_SX +  # S_2
                                                 alpha_sigma_X**2 * S_7))  # S_3
                # Equation for phi.
                phi = pm.Deterministic('phi', (self.constants['b'] + self.constants['d'] * N_E +\
                                        (S_4 + S_5 + S_6) / S_123))
                pm.Normal('X_o', alpha_mu_X, alpha_sigma_X, observed=X_o_, shape=X_o_.shape)
                phi_hat = pred_o_ / X_o_
                pm.Normal('phi_obs', phi, 1.0, observed=phi_hat)
                # From stackexchange https://stats.stackexchange.com/questions/285986/distribution-of-linear-regression-coefficients
                # phi ~ N(\hat{phi}, (X^T X)^{-1} sigma^2) where sigma is the standard deviation of the noise in Y? 
                # For one set of samples, we can treat X as fixed and get 
                # For regression Y = c_1 + c_2 * X + err where err ~ N(0, sigma^2) when c_1 == 0 (as in our cae)
                # Then phi = c_2 + err / sum(X^2)
                # phi ~ N(\hat{phi}, var(err) / sum(X_squared))
                # Note that var(err) == var(Y-phi*X)? Isn't this the distribution of errors around the regression line?
                # As in, not the uncertainty of phi itself?
                # This is _not_ the uncertainty of phi given the parameters of the (prior) noise distributions.
                # This is also _not_ the uncertainty of phi given the parameters sampled from the (prior) noise distribution.
                # This could be the uncertainty of phi given the _sampled_ values of the low-dim. auxiliary variables, but note
                # that this uncertainty will vary for new samples of the low-dim. aux. variables even when
                # the parameters of the noise distributions are kept fixed.
                # Hypothetically we could model the plate many times with the drawn parameters to estimate some sort of
                # uncertainty of phi. Kind of doing Bayesian sampling within Bayesian sampling.
                # -We will have to save that for later.
                # -> Simply use the "phi ~ N(\hat{phi}, 1e-3)" for now.
            #

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
                              target_accept=.95,
                              chains=4,
                              random_seed=42,  # Fixed for reproducibility.
                              #max_treedepth=15,
                              init="jitter+adapt_diag",
                              callback=printing_callback,
                              )
        return trace


with pm.Model() as model:
    pass