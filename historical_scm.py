"""
Data generation, intervention, and CATE estimation
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple


# ---------- helpers ----------
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class SCM:
    alpha_x_mu: float      # μ of N_X
    alpha_x_sigma: float   # σ of N_X        (σ > 0)
    alpha_a_mu: float      # μ of N_A
    alpha_y_sigma: float   # σ of N_Y        (σ > 0)
    n_e: float = 0.5       # coefficient N_E   (e.g. −0.5)
    A_type: str = "continuous" # "binary" or "continuous"
    # Model parameters for the deterministic parts should preferably
    # also be modifiable.

    # ----------------------------------------------------------
    #   Observational sampling
    #   Binary A, continuous x, continuous y
    # ----------------------------------------------------------
    def sample(self, n: int = 1, random_state=None
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates n samples from the SCM.

        Args:
        n (int): Number of observations to generate.
        random_state (int, np.random.Generator, None): Seed for reproducibility.
        """
        rng = np.random.default_rng(random_state)

        # exogenous noise
        N_X = rng.normal(self.alpha_x_mu, self.alpha_x_sigma, size=n)
        N_Y = rng.normal(0.0, self.alpha_y_sigma, size=n)
        N_A = rng.normal(self.alpha_a_mu, 1.0, size=n)

        # endogenous variables
        X = N_X
        if self.A_type == "continuous":
            # Continuous treatment A
            A = 0.125 * X + N_A  # 1/8 of X is added to dosage.
        elif self.A_type == "binary":
            # Binary treatment A
            # p(A = 1 | X, N_A) = sigmoid(0.7 * X**2 + N_A)
            # Note: 0.7 is a hyperparameter that can be adjusted.
            # It controls the strength of the relationship between X and A.
            p_A = sigmoid(0.7 * X**2 + N_A)  # P(A = 1 | X, N_A)
            A = rng.binomial(1, p_A)  # binary treatment
        else:
            raise ValueError("A_type must be 'binary' or 'continuous'.")
        Y = 12 - 0.1 * X + self.n_e * A + N_Y  # Agent assumes the treatement effect is independent of X and constant.
        # Y = 0.5 * X + self.n_e * A + N_Y
        return X, A, Y

    def sample_do_A(self,
             a_fixed,
             X: np.ndarray,
             n: int = 1,
             random_state=None
            ) -> np.ndarray:
        """
        Samples from the SCM given do(A = a_fixed).
        Returns an array with shape (n, *X.shape), where each row
        is an i.i.d. sample from  Y | do(A=a_fixed), X.

        Args:
        a_fixed (int- or array-like, values in {0, 1}): value to intervene on A, or array with X.shape
        X: observed values of X
        n (int): number of samples to draw
        random_state: seed for reproducibility
        """
        # Sanity check
        if self.A_type == 'binary':
            if np.any((np.asarray(a_fixed) != 0) & (np.asarray(a_fixed) != 1)):
                raise ValueError("a_fixed must be 0 or 1 (binary).")

        rng = np.random.default_rng(random_state)
        X = np.asarray(X)
        a_fixed = np.asarray(a_fixed)

        if a_fixed.shape == ():  # broadcast scalar
            a_fixed = np.full_like(X, a_fixed)
        if a_fixed.shape != X.shape:
            raise ValueError("`a_fixed` and `X` must have the same shape "
                             "or `a_fixed` must be scalar.")

        N_Y = rng.normal(0.0, self.alpha_y_sigma, size=(n,) + X.shape)
        # Y_do = 0.5 * X[None, ...] + self.n_e * a_fixed[None, ...] + N_Y
        Y_do = 12 - 0.1 * X[None, ...] + self.n_e * a_fixed + N_Y
        return Y_do

    # ----------------------------------------------------------
    #    Closed-form distribution of  Y | do(A=a), X
    # ----------------------------------------------------------
    def analytic_do_distribution(self, a_fixed, X
                                ) -> Tuple[np.ndarray, float]:
        """
        Returns the mean and variance of Y given X and do(a_fixed).

        Args:
        a_fixed (int or np.ndarray): value to intervene on A, or array with X.shape
        X (np.ndarray): observed values of X
        """
        a_fixed = np.asarray(a_fixed)
        mu = 12 - 0.1 * np.asarray(X) + self.n_e * a_fixed
        var = self.alpha_y_sigma ** 2
        return mu, var


class BinaryA_SCM(SCM):
    """
    A specific SCM with binary treatment A, continuous X and Y.
    This class inherits from SCM and sets the A_type to "binary".
    """
    def __init__(self, alpha_x_mu, alpha_x_sigma, alpha_a_mu,
                 alpha_y_sigma, n_e):
        super().__init__(
            alpha_x_mu=alpha_x_mu,
            alpha_x_sigma=alpha_x_sigma,
            alpha_a_mu=alpha_a_mu,
            alpha_y_sigma=alpha_y_sigma,
            n_e=n_e,
            A_type="binary"
        )



# Looping over different parameters of the SCM
def estimate_cate_from_row(row, *, n_mc: int = 0, seed: int | None = None,
                           mode='binary'):
    """
    This function is used to estimate CATE given a series of parameters to initialize
    the SCM.

    Args:
    row … a Series from df_params  (one set of α's and n_e)
    n_mc (int): If n_mc = 0, then the CATE is computed analytically (fast, exact).
                If n_mc > 0, then the CATE is computed using n_mc draws (Monte-Carlo sampling)
    seed (int, None): Seed for reproducibility of the random sampling.
    """
    if n_mc == 0:
        # Analytic version
        # In this SCM  CATE(X) = E[Y|do(A=1),X] − E[Y|do(A=0),X] = n_e
        #return scm.n_e  # No sense in initializing SCM  as we are only returning the value of n_e.
        return row["NE"]
    else:
        # Monte-Carlo version
        print('Defaulting to model with binary treatment.')
        scm = BinaryA_SCM(
            alpha_x_mu   = row["αXμ"],
            alpha_x_sigma= row["αXσ"],
            alpha_a_mu   = row["αAμ"],
            alpha_y_sigma= row["αYσ"],
            n_e          = row["NE"],
        )
        X, *_ = scm.sample(n=n_mc, random_state=seed)
        # Why do we use the analytic version next?
        # X is sampled. 
        # HARD CODED TO BINARY TREATMENT. CHANGE. SUPPORT ALSO CONTINUOUS TREATMENTS.
        mu1, _ = scm.analytic_do_distribution(1, X)
        mu0, _ = scm.analytic_do_distribution(0, X)
        # This ends up being the value where the MC-version should converge to:
        return (mu1 - mu0).mean()


def sample_CATE(row, X, a1_fixed, a0_fixed):
        """
        This function samples the CATE-distribution from using treatment 1 (higher dosage)
        instead of treatment 0.

        Args:
        row (series from from df_params): Posterior parameters
        X (float): Feature to condition on
        a1_fixed [float]: Dosage value of higher dose treatment
        a0_fixed [float]: Dosage value of lower dose treatment

        Again, what are we more specifically interested in?
        The expected treatment effect for a patient with features X at
        the doctor's office?
        The name specifically indicates that this is NOT ATE!
        """
        # New model for data generation instantiated with posterior parameter values:
        scm = SCM(
            alpha_x_mu   = row["αXμ"],
            alpha_x_sigma= row["αXσ"],
            alpha_a_mu   = row["αAμ"],
            alpha_y_sigma= row["αYσ"],
            n_e          = row["NE"],  # Is this supposed to be updated?
            A_type       = 'continuous'
        )
        # Sample an outcome with treatment 1:
        Y_do1 = scm.sample_do_A(a1_fixed, X)
        # Sample an outcome with treatment 0:
        Y_do0 = scm.sample_do_A(a0_fixed, X)
        # The sample drawn from the CATE-distribution:
        CATE_sample = Y_do1 - Y_do0
        return CATE_sample


def estimate_CATE_parameters(row, X, a1_fixed, a0_fixed, n=1000):
    """
    Sample many points from the CATE-distribution and estimate mean
    and variance?
    """
    pass

def marginal_CATE():
    """
    marginalCATE(X) = E[Y|X, do(A=a1)] - E[Y|X, do(A=a0)]
    """
    pass


if __name__ == "__main__":
    """
    Run a simple test of the SCM class
    """
    import tests.test_historical_scm
    # If no crash, then the test passed.
    print("\nSCM test passed.")
