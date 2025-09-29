"""
Experiment models are defined here.
"""
import pymc as pm
import warnings
import pytensor.tensor as at

DEBUGGING = False


def printing_callback(trace, draw):
    # For debugging.
    if DEBUGGING:
        print(f"Completed draw {draw.draw_idx} from chain {draw.chain}.")
        print("Shape of X: {}".format(trace.get_values('X').shape))
        print("Shape of Y: {}".format(trace.get_values('Y').shape))


class AgentModel():
    def __init__(self, plate_size, constants=None, prior_parameters=None, low_dim_aux_plate=True):
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

            if not low_dim_aux_plate:
                # Plate in model (for i in 1...plate_size):
                X = pm.Normal('X', alpha_mu_X, alpha_sigma_X, shape=self.plate_size)
                # Is it faster to have a Normal distribution with separate parameters (here X_i * d)
                # or to sample from a Standard Normal and then scale them?
                A = pm.Normal('A', self.constants['d'] * X + alpha_mu_A, alpha_sigma_A, shape=self.plate_size)
                Y = pm.Normal('Y', self.constants['a'] + self.constants['b'] * X + N_E * A + alpha_mu_Y, alpha_sigma_Y, shape=self.plate_size)

                # Is this element-wise multiplication? Could use dot product.
                mu_M = pm.Deterministic('mu_M', pm.math.sum(X * Y) / pm.math.sum(X**2))
                # sigma_M = pm.Deterministic('sigma_M', pm.math.sqrt(pm.math.sum((mu_M * X - Y)**2) / (self.plate_size - 1)))  # Variance with Bessel's correction
                X_o = pm.Normal('X_o', alpha_mu_X, alpha_sigma_X, observed=X_o_)
                warnings.warn("Model currently not complete. It lacks a distribution for phi.")
                # pm.DIST(param1.., obs=phi?...)  # What is the distriburtion of sum(X_i*Y_i)/sum(X_i^2)?
            else:
                epsilon_x = pm.Normal('sum_U_x', 0, 1)  # Not a sum yet, needs to be scaled appropriately.
                epsilon_y = pm.Normal('sum_U_y', 0, 1)  # Not a sum yet, needs to be scaled appropriately.
                epsilon_a = pm.Normal('sum_U_a', 0, 1)  # Not a sum yet, needs to be scaled appropriately.
                Z_xx = pm.ChiSquared('Z_xx', self.plate_size - 1)  # Independent part of sum of squares (independent of sum of X).
                Z_xy_alpha = pm.ChiSquared('Z_xy_alpha', self.plate_size - 1)  # First part of "chi-squared difference distribution"
                Z_xy_beta = pm.ChiSquared('Z_xy_beta', self.plate_size - 1)  # Second part of "chi-squared difference distribution"
                Z_xa_alpha = pm.ChiSquared('Z_xa_alpha', self.plate_size - 1)  # First part of "chi-squared difference distribution"
                Z_xa_beta = pm.ChiSquared('Z_xa_beta', self.plate_size - 1)  # Second part of "chi-squared difference distribution"
                # All alpha_mu and alpha_sigma goes into a function.
                # N_E = N_E  # Just to be explicit.
                sum_X = pm.Deterministic('sum_X', self.plate_size * alpha_mu_X + at.sqrt(self.plate_size) * alpha_sigma_X * epsilon_x)
                sum_X_squared = pm.Deterministic('sum_X_squared', self.plate_size * alpha_mu_X**2 +\
                                                 2 * self.plate_size * alpha_mu_X * alpha_sigma_X * epsilon_x +\
                                                 alpha_sigma_X**2 * (Z_xx - epsilon_x))  # (\sum((epsilon_x_i))/n)^2 = (sqrt(n)*epsilon_x)^2/n = epsilon_x^2
                sum_XA = pm.Deterministic('sum_XA', self.plate_size * alpha_mu_X * alpha_mu_A +\
                                            self.plate_size * alpha_mu_X * alpha_sigma_A * epsilon_a +\
                                            self.plate_size * alpha_mu_A * alpha_sigma_X * epsilon_x +\
                                            alpha_sigma_X * alpha_sigma_A * (Z_xa_alpha - Z_xa_beta) / 2)  # Covariance part.
                sum_XY = pm.Deterministic('sum_XY', self.plate_size * alpha_mu_X * alpha_mu_Y +\
                                            self.plate_size * alpha_mu_X * alpha_sigma_Y * epsilon_y +\
                                            self.plate_size * alpha_mu_Y * alpha_sigma_X * epsilon_x +\
                                            alpha_sigma_X * alpha_sigma_Y * (Z_xy_alpha - Z_xy_beta) / 2)
                phi = pm.Deterministic('phi', self.constants['b'] + self.constants['d'] * N_E +\
                                        self.constants['a'] * sum_X / sum_X_squared +\
                                        N_E * sum_XA / sum_X_squared +\
                                        sum_XY / sum_X_squared)
                pm.Normal('X_o', alpha_mu_X, alpha_sigma_X, observed=X_o_)
                phi_hat = pred_o_ / X_o_
                pm.Normal('phi_obs', phi, 1e-3, observed=phi_hat)
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
                              max_treedepth=15,
                              init="jitter+adapt_diag",
                              callback=printing_callback,
                              )
        return trace


with pm.Model() as model:
    pass