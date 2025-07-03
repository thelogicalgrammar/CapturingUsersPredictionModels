"""
Experiment model is defined here.
"""
import pymc as pm
import pytensor.tensor as at
from pytensor.tensor import log1p, softplus, sigmoid, logaddexp, dot


def define_model_1(Xo_obs, Predo_obs, n=100, tiny=1e-3):
    """
    Defines the internal model of the agent, enabling Bayesian update on the priors.
    """
    with pm.Model() as m:
        # hyper‑priors
        αXμ  = pm.Normal('αXμ',  10, 5)
        αXσ  = pm.Exponential('αXσ', 1)
        αAμ  = pm.Normal('αAμ',  5, 1)
        αYσ  = pm.Exponential('αYσ', 1)
        NE   = pm.Normal('NE',  -0.1, 1)
    
        # deterministic helpers
        a  = pm.Deterministic('a', 1 + NE)
        b  = pm.Deterministic('b', NE * αAμ)
        sy2 = pm.Deterministic('sy2', NE**2 + αYσ**2)
    
        # Draw the three low‑dimensional auxiliaries of the n-dimensional plate
        T1 = pm.Normal('T1',  0, 1,  shape=())  # will be scaled by sqrt(n)
        T2 = pm.ChiSquared('T2', n)
        T3 = pm.Normal('T3',  0, 1,  shape=())

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
        phi = pm.Deterministic('phi', Sxy / Sxx)

        # only observed data in the entire model
        pm.Normal('Predo', mu = phi * Xo_obs, sigma = tiny, observed = Predo_obs)

    return m


def define_model_2(x_obs, pred_obs, n_train, eps_pred=1e-2, eps_phi=1e-9, rng=None):
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
        p_A      = pm.Deterministic("p_A", pm.math.sigmoid(logit_pA))
        p_A   = pm.math.clip(p_A, eps_phi, 1.0 - eps_phi)   # <- clipping of p_A to avoid it getting to 1 or 0

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


def define_model_3(Xo_obs, Predo_obs, n, tiny):
    """
    Defines the internal model of the agent, enabling Bayesian update on the priors.
    This is the model with binary A, but X and Y are continuous.

    Args:
        Xo_obs (float): The observed value of X.
        Predo_obs (float): The observed prediction value.
        n (int): The number of samples to consider.
        tiny (float): A small value to avoid numerical issues in the model.
    """
    with pm.Model() as m:
        
        # hyper-priors
        αXμ = pm.Normal('αXμ', 10, 5)
        αXσ = pm.Exponential('αXσ', 1)
        αAμ = pm.Normal('αAμ', 5, 1)
        αYσ = pm.Exponential('αYσ', 1)
        NE  = pm.Normal('NE', -0.1, 1)

        # latent (X_i, Y_i) — still size-n
        X = pm.Normal('X', αXμ, αXσ, shape=n)
        Y = pm.Normal('Y', 0, 10, shape=n)          # diffuse prior

        # ---- binary-A marginal likelihood ---------------------------------
        mu0   = X
        mu1   = X + NE
        logp0 = pm.logp(pm.Normal.dist(mu=mu0, sigma=αYσ), Y)
        logp1 = pm.logp(pm.Normal.dist(mu=mu1, sigma=αYσ), Y)
        logit_pi = X + αAμ
        
        pm.Potential('mixlike',
            pm.math.logaddexp(log1p(-sigmoid(logit_pi)) + logp0,
                            softplus(logit_pi)        + logp1))

        # analytic φ* based on X and Y  (still valid!)
        phi = pm.Deterministic('phi', pm.math.dot(X, Y) / pm.math.dot(X, X))

        # observed pair (X^o, Pred^o)
        pm.Normal('Predo',
                mu = phi * Xo_obs,
                sigma = tiny,
                observed = Predo_obs)
    return m


def define_model_4(Xo_obs, Predo_obs, n, sigma_P, rng=None):
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
