"""
Experiment models are defined here.
"""
import pymc as pm
import warnings
import pytensor.tensor as at
from pytensor.tensor import log1p, softplus, sigmoid, logaddexp, dot
import numpy as np


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
            # Placeholders for observed data:
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

            # Plate in model:
            X_i = pm.Normal('X_i', alpha_mu_X, alpha_sigma_X, shape=self.plate_size)
            # Is it faster to have a Normal distribution with separate parameters (here X_i * d)
            # or to sample from a Standard Normal and then scale them?
            A_i = pm.Normal('A_i', self.constants['d'] * X_i + alpha_mu_A, alpha_sigma_A, shape=self.plate_size)
            Y_i = pm.Normal('Y_i', self.constants['a'] + self.constants['b'] * X_i + N_E * A_i + alpha_mu_Y, alpha_sigma_Y, shape=self.plate_size)

            # Is this element-wise multiplication? Could use dot product.
            mu_M = pm.Deterministic('mu_M', pm.math.sum(X_i * Y_i) / pm.math.sum(X_i**2))
            sigma_M = pm.Deterministic('sigma_M', pm.math.sqrt(pm.math.sum((mu_M * X_i - Y_i)**2) / (plate_size - 1)))  # Variance with Bessel's correction

            # What should these be called?
            # And are both somehow included in the likelihood of the observed data?
            X_o = pm.Deterministic('X_o', pm.Normal('N_X_o', alpha_mu_X, alpha_sigma_X, observed=X_o_))
            pred_o = pm.Deterministic('pred_o', pm.Normal('M_j', mu_M * X_o, sigma_M, observed=pred_o_))

    def sample(self, X_o, pred_o, draws=1000):
        """
        Sample from the posterior given observed data.

        Args:
        X_o (np.ndarray): 1-dim observed feature data.
        pred_o (np.ndarray): 1-dim observed prediction data.
        n (int): number of samples to draw from the posterior.
        """
        # Replace placeholders with actual data:
        pm.set_data({'X_o_': X_o, 'pred_o_': pred_o}, model=self.model)
        with self.model:
            trace = pm.sample(draws=draws,
                              tune=1000,
                              target_accept=.99,
                              chains=4,
                              random_seed=42,
                              max_treedepth=15,
                              init="jitter+adapt_diag",
                              )
        return trace



def define_model_1(Xo_obs, Predo_obs, n=100, tiny=1e-3) -> pm.Model:
    """
    Defines the internal model of the agent, enabling Bayesian update on the priors.
    Continuous A. 
    """

    with pm.Model() as m:
        # hyper‑priors
        αXμ  = pm.Normal('αXμ', 80, 1)
        αXσ  = pm.Exponential('αXσ', 10)
        αAμ  = pm.Normal('αAμ', 10, 1)
        αYσ  = pm.Exponential('αYσ', .1)
        NE   = pm.Normal('NE', 20.0, 1)

        # deterministic helpers
        a  = pm.Deterministic('a', 1 + NE)
        b  = pm.Deterministic('b', NE * αAμ)
        sy2 = pm.Deterministic('sy2', NE**2 + αYσ**2)

        # Draw the three low‑dimensional auxiliaries of the n-dimensional plate
        text =  "While T2 might follow a Chi-squared distribution, T1 and T2 are not independent and cannote be sampled separately."
        warnings.warn(text)
        T1 = pm.Normal('T1', 0, 1, shape=())  # will be scaled by sqrt(n)
        T2 = pm.ChiSquared('T2', n)  
        T3 = pm.Normal('T3', 0, 1, shape=())

        # scale them (see derivation)
        T1s = at.sqrt(n) * T1
        T3s = at.sqrt(n) * T3

        sx2 = αXσ**2
        deltax = αXμ / αXσ

        Sxx = pm.Deterministic(
            'Sxx',
            sx2 * (n*deltax**2 + 2*deltax*T1s + T2)
        )
        Sxy = pm.Deterministic(
            'Sxy',
            a*Sxx + b*n*αXμ + b*αXσ*T1s + αXσ*at.sqrt(sy2)*T3s
        )
        # this is the coefficient of the linear model
        # Implicitly builds on observations being centered around 0.
        # This is the OLS estimate of the slope.
        phi = pm.Deterministic('phi', Sxy / Sxx)

        pm.Normal(
            'Xo', 
            mu = αXμ, 
            sigma = αXσ, 
            observed = Xo_obs
        )

        # only observed data in the entire model
        pm.Normal(
            'Predo', 
            mu = phi * Xo_obs, 
            sigma = tiny, 
            observed = Predo_obs
        )

    return m


def define_model_2(x_obs, pred_obs, n_train, eps_pred=1e-2, eps_phi=1e-9, rng=None
                   ) -> pm.Model:
    with pm.Model() as scm_nc:
        α_Xμ   = pm.Normal("α_Xμ", 0, 3)
        α_Xσ   = pm.HalfNormal("α_Xσ", 2)

        Z_X    = pm.Normal("Z_X", 0, 1, shape=n_train,
                           initval=rng.normal(size=n_train))
        X      = pm.Deterministic("X", α_Xμ + α_Xσ * Z_X)

        α_Aμ   = pm.Normal("α_Aμ", 0, 2)
        # Z_A    = pm.Normal("Z_A", 0, 1, shape=n_train)
        # p_A    = pm.Deterministic("p_A", pm.math.sigmoid(α_Aμ + 0.7 * X**2 + Z_A))
        # A      = pm.Bernoulli("A", p=p_A, shape=n_train)

        α_Yσ   = pm.HalfNormal("α_Yσ", 1)
        # Z_Y    = pm.Normal("Z_Y", 0, 1, shape=n_train)
        N_E    = pm.Normal("N_E", -10, 10)
        # Y      = pm.Deterministic("Y", 0.5*X + N_E * A + α_Yσ * Z_Y)

        # phi_hat = pm.Deterministic(
        #     "phi_hat",
        #     at.sum(X * Y) / (at.sum(X**2) + eps_phi)   # <-- safe
        # )

        # ------------- mixture weights -------------------
        # p_A = sigmoid(α_Aμ + 0.7 * X**2)
        logit_pA = α_Aμ + 0.7 * X**2
        p_A = pm.Deterministic("p_A", pm.math.sigmoid(logit_pA))
        p_A = pm.math.clip(p_A, eps_phi, 1.0 - eps_phi)   # <- clipping of p_A to avoid it getting to 1 or 0

        # ------------- Y likelihood (mixture) ------------
        comps = pm.Normal.dist(
            mu = at.stack([X, X + N_E], axis=1),   # shape (n_train, 2)
            sigma = α_Yσ,
        )                                          # broadcasts σ
        w     = at.stack([1 - p_A, p_A], axis=1)   # shape (n_train, 2)

        Y = pm.Mixture("Y", w=w, comp_dists=comps, shape=n_train)

        # ------------- OLS coefficient φ̂ -----------------
        Xc  = X - X.mean()
        Yc  = Y - Y.mean()
        phi_hat = pm.Deterministic(
            "phi_hat", at.sum(Xc * Yc) / (at.sum(Xc**2) + eps_phi)
        )
        # ---- add observed variables -------------

        X_o    = pm.Normal("X_o", α_Xμ, α_Xσ, observed=x_obs)
        Pred_o = pm.Normal("Pred_o", mu=phi_hat*X_o, sigma=eps_pred, observed=pred_obs)

        # scm_nc.debug()

    return scm_nc


def define_model_3(Xo_obs, Predo_obs, n, tiny=1e-3) -> pm.Model:
    """
    """
    with pm.Model() as m:
    
        # hyper-priors
        αXμ = pm.Normal('αXμ', 10, 5)
        αXσ = pm.Exponential('αXσ', 1)

        # parameters that decide
        # the probability of treatment A as a function of X 
        # (in expit space)
        # NOTE: Here we can also use priors for a beta distributino
        # and get the probability of treatment A from the pdf of the beta
        αA = pm.Normal('αA', 5, 1)
        βA = pm.Normal('βA', 0, 1)

        # population-level noise in the outcome Y
        αYσ = pm.Exponential('αYσ', 1)
        # treatment effect 
        NE  = pm.Normal('NE', -0.1, 1)

        # outcome noise for specific patients
        N_Y = pm.Normal('Y', 0,   αYσ,  shape=n)
        # latent (X_i, Y_i) — size-n
        X = pm.Normal('X', αXμ, αXσ, shape=n)

        A = pm.Bernoulli(
            'A',
            # probability of treatment A given X
            p = sigmoid(αA + βA * X),
            # shape=n
        )

        Y = pm.Deterministic(
            'Y',
            # outcome Y is a linear function of X and treatment A
            # intercept is 0
            X + A * NE + N_Y
        )

        # analytic φ* based on X and Y 
        # This represents the OLS estimate of the slope that relates X and Y.
        # NOTE: Does not take into account the treatment at all.
        phi = pm.Deterministic(
            'phi',
            pm.math.dot(X, Y) / pm.math.dot(X, X)
        )

        # NOTE: This is new! - discuss
        pm.Normal(
            'Xo',
            mu = αXμ,
            sigma = αXσ,
            observed = Xo_obs
        )

        # observed pair (X^o, Pred^o)
        pm.Normal(
            'Predo',
            # Assumes that the intercept is 0
            mu = phi * Xo_obs,
            sigma = tiny,
            observed = Predo_obs
        )

    m_marginalized = pmx.marginalize(m, ['A'])

    return m_marginalized


def define_model_4(Xo_obs, Predo_obs, n, sigma_P, rng=None
                   ) -> pm.Model:
    with pm.Model() as m:

        # ─── hyper-priors ──────────────────────────────────────────
        αXμ  = pm.Normal('αXμ',  10, 5)
        αXσ  = pm.HalfNormal('αXσ', 2)  # narrower, keeps funnel shallow
        αAμ  = pm.Normal('αAμ',   5, 1)
        αYσ  = pm.HalfNormal('αYσ', 2)
        NE   = pm.Normal('NE',  -0.1, 0.5)  # slightly tighter

        # ─── NON-CENTRED latent plate ─────────────────────────────
        X_raw = pm.Normal('X_raw',  0, 1, shape=n)
        X     = pm.Deterministic('X', αXμ + αXσ * X_raw)

        # Y: centre it at X + E[A|X]·N_E  (optional but helps)
        mu_Y  = X + (sigmoid(X + αAμ) * NE)
        Y_raw = pm.Normal('Y_raw', 0, 1, shape=n)
        Y     = pm.Deterministic('Y', mu_Y + αYσ * Y_raw)

        # ─── binary-A marginal likelihood (log-mix) ───────────────
        mu0   = X
        mu1   = X + NE
        logp0 = pm.logp(pm.Normal.dist(mu=mu0,  sigma=αYσ), Y)
        logp1 = pm.logp(pm.Normal.dist(mu=mu1,  sigma=αYσ), Y)
        pi    = sigmoid(X + αAμ)
        pm.Potential('mixlike',
            logaddexp(log1p(-pi) + logp0,
                    pm.math.log(pi)    + logp1))

        # ─── analytic φ*  and the observed pair ───────────────────
        phi = pm.Deterministic('phi', dot(X, Y) / dot(X, X))
        pm.Normal('Predo',
                mu = phi * Xo_obs,
                sigma = sigma_P,
                observed = Predo_obs)
    return m
