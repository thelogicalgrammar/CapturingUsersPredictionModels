"""
This file contains code for updating a Bayesian model based on
the weights of a regression model. 

WORK-IN-PROGRESS (7.7.2026)
-Keep agent_model import or make stand-alone?
-Some am.AgentModel methods might need to be overwritten. Check.
-Maybe wrap the posterior predictive method in a predict-method?
"""
import numpy as np
import pymc as pm
import pytensor as pt

import agent_model as am
import scm as scm


def printing_callback(trace, draw):
    """
    Function for debugging pymc-sampling.
    """
    if DEBUGGING:
        print(f"Completed draw {draw.draw_idx} from chain {draw.chain}.")


class ModelFromRegressionParameters(am.AgentModel):
    """
    This class implements an agent model for a setting where the agent
    receives one or multiple predictions.
    Normally distributed parameters given ...
    Keep matrix inveersion to get model parameters in or outside of class?

    Class for a scalar agent model with structural equations:

    :math:`Y_i = a + b X_i + N_{E} A_i + N_{Yi}`

    :math:`A_i = d X_i + N_{Ai}`

    :math:`X_i = N_{Xi}`

    where N_{Xi}, N_{Ai}, N_{Yi} are normal noise variables drawn for every
    instantiation of the plate and N_{E} is a normal variable drawn once per
    instantiation of the plate. Unlike :class:`MultivariateAgentModel`, the
    regression coefficients ``a``, ``b``, and ``d`` are fixed constants, not
    learned parameters with their own priors.

    Args:
        plate_size (int): Size of the plate (number of i.i.d. samples used to
            train the predictive model).
        prior_parameters (dict): Dictionary with parameters for the prior
            distributions. If None, defaults are used.
        scm_constants (dict): Dictionary with the fixed structural
            coefficients ``a``, ``b``, and ``d``. If None, defaults are used.
        predictive_var_names (list): Names of the variables sampled for
            predictive calculations. If None, defaults are used.

    Returns:
        None
    """
    def __init__(self, plate_size, prior_parameters=None,
                 scm_constants=None, predictive_var_names=None):
        self.n_covariates = 1
        self.plate_size = int(plate_size)

        default_scm_constants = {'a': 12, 'b': -0.1, 'd': 0.125}
        if scm_constants is None:
            self.scm_constants = default_scm_constants
        else:
            extra_scm_items = [item for item in scm_constants.keys()
                               if item not in default_scm_constants.keys()]
            err_msg = "{} passed with scm_constants, " \
                "but not part of model specification.".format(extra_scm_items)
            assert len(extra_scm_items) < 1, err_msg
            self.scm_constants = scm_constants

        self.prior_parameters = self._initialize_prior_parameters(
            prior_parameters, self._default_prior_parameters())

        # Sampling method is dependent on SCM parameterization.
        self.predictive_scm = scm.SCM_WithConstants(
            alpha_mu_X=self.prior_parameters['mu_X'],
            alpha_sigma_X=self.prior_parameters['sigma_X'],
            alpha_mu_A=self.prior_parameters['mu_A'],
            alpha_sigma_A=self.prior_parameters['sigma_A'],
            alpha_mu_Y=self.prior_parameters['mu_Y'],
            alpha_sigma_Y=self.prior_parameters['sigma_Y'],
            N_E=self.prior_parameters['N_E_mu'],
            a=self.scm_constants['a'],
            b=self.scm_constants['b'],
            d=self.scm_constants['d'],
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
        self.beta_hat = None

        self._constructor_kwargs = dict(
            plate_size=self.plate_size,
            prior_parameters=self.prior_parameters,
            scm_constants=self.scm_constants,
            predictive_var_names=self.predictive_var_names,
        )


    def _build_model(self):
        """
        Build the PyMC model.

        The observed vector of weights \beta_hat included as
        :class:`pm.Data` placeholder, updated at sampling time.

        Returns:
            pm.Model: The constructed PyMC model.
        """
        model = pm.Model()
        with model:
            beta_hat = pm.Data('X_o_', np.zeros(self.n_covariates))

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

            # The SCM defines Y_i as
            # Y_i = N_b^T X_i + N_E N_d^T X_i + N_E N_{A_i} + N_{Y_i}
            # Given a sample from the priors, the regression estimate of the weights
            # are then normally distributed around the true values, i.e.
            # \hat{\beta} ~ N(\beta, (Z^T Z)^{-1} \tau), where
            # Z = [\vec{1}, X],
            # \beta = [[N_b + N_E * N_d], [N_E * \mu_A + \mu_Y]]) and
            # \tau = N_E^2 * \sigma_A^2 + \sigma_Y^2
            # Offset in \beta-vector (using augmented vectors)
            beta_0 = pm.Deterministic('beta_0', N_E * alpha_mu_A + alpha_mu_Y)
            # Feature weights in \beta-vector
            beta_1 = pm.Deterministic('beta_1', N_b + N_E * N_d)
            beta = pm.Deterministic('beta',
                pt.concatenate([pt.reshape(beta_0, (1,)), beta_1], axis=0))
            # \tau-part of variance
            tau = pm.Deterministic('tau', N_E**2 * alpha_sigma_A**2 + alpha_sigma_Y**2)

            # The term (Z^T Z)^{-1} is still needed. Note that Z is the augmented
            # version of X with a leading column of ones. Then
            # Z_i ~ [1, N_p(\mu, \Sigma)], where p is the dimensionality of X.
            # Using centered scatter \bar{x} ~ N_p(\mu, \Sigma/n),
            # we express the Schur complement as
            # S_c = (X - \vec{1} \bar{x}^T)^T (X - \vec{1} \bar{x}^T)
            # Using the Schur-complement S_c ~ Wishart_p(n-1, \Sigma) and
            # centered scatter \bar{x} 
            # we calculate (Z^T Z)^{-1} using the blockformula for inverses.
            # Because \Sigma is diagonal, we can calculate the inverse
            # S_c^{-1} = D^{-1} W^{-1} D^{-1} where the diagonal elements
            # of D contain the corresponding standard deviations. Thus, we sample
            # W ~ Wishart(n-1, I_p),
            V = pt.diag(alpha_sigma_X**2)  # Variance, not standard deviation.
            W = pm.Wishart('W', nu=self.plate_size - 1, V=V)
            # Invert it numerically
            W_inv = pt.linalg.solve(W, pt.eye(self.n_covariates))
            # Insert in equation to get S_c^{-1}.
            D_inv = pt.diag(1/alpha_sigma_X)
            # Inverse of Schur complement:
            Sc_inv = D_inv @ W_inv @ D_inv
            # We next sample \bar{x} ~ N_p(\mu, \Sigma/n)
            D = pt.diag(alpha_sigma_X**2) / self.plate_size
            x_bar = pm.MvNormal('x_bar', mu=alpha_mu_X, cov=D)
            # and lastly we insert these values together with n into the block
            # formula for the inverse.
            top_left = 1/self.plate_size + x_bar.T @ Sc_inv @ x_bar
            top_left = pt.reshape(top_left, (1, 1))
            top_right = -x_bar.T @ Sc_inv
            top_right = pt.reshape(top_right, (1, self.n_covariates))
            bottom_left = -Sc_inv @ x_bar
            bottom_left = pt.reshape(bottom_left, (self.n_covariates, 1))
            bottom_right = Sc_inv
            XtX_inv = pt.concatenate([
                pt.concatenate([top_left, top_right], axis=1),
                pt.concatenate([bottom_left, bottom_right], axis=1)
            ], axis=0)
            # Multiplying this by \tau gives the variance of the vector of
            # parameters \beta.
            beta_var = tau * XtX_inv 

            # Center of prediction if our beliefs were correct:
            mu_beta = pm.Deterministic('mu_pred', beta)
            # Variance of the same
            cov_beta = pm.Deterministic('cov_pred', beta_var)
            pm.MvNormal('pred_o', mu=mu_beta, cov=cov_beta, observed=beta_hat)
        return model

    def sample(self, beta_hat, draws=1000, tune=1000, target_accept=.85,
               chains=4, max_treedepth=15, init="jitter+adapt_diag", rng=None):
        """
        Sample from the posterior given observed data.
        Stores the latest trace in self.

        Args:
            beta_hat (np.array): Weights of the regression model with
            the bias term first.
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
        beta_hat = self._flatten_vector_parameter(beta_hat, 'beta_hat')
        self.posterior_draws = draws
        pm.set_data({'beta_hat': beta_hat}, model=self.model)
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
        self.beta_hat = beta_hat
        return
