# import pymc as pm
# import pytensor.tensor as at
import numpy as np
# import arviz as az
# import pandas as pd
# import matplotlib.pyplot as plt
# from pytensor.tensor.random.utils import params_broadcast_shapes  # Not in use.
from typing import Tuple
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression



# ---------- helpers ----------
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class BinaryA_SCM:
    alpha_x_mu: float      # μ of N_X
    alpha_x_sigma: float   # σ of N_X        (σ > 0)
    alpha_a_mu: float      # μ of N_A
    alpha_y_sigma: float   # σ of N_Y        (σ > 0)
    n_e: float             # coefficient N_E   (e.g. −0.5)

    # ----------------------------------------------------------
    #    Observational sampling
    # ----------------------------------------------------------
    def sample(self, n: int = 1, random_state=None
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates n samples from the SCM.

        Args:
        n (int): Number of observations to generate.
        random_state (int, None): Seed for reproducibility.
        """
        rng = np.random.default_rng(random_state)

        # exogenous noise
        N_X = rng.normal(self.alpha_x_mu, self.alpha_x_sigma, size=n)
        N_Y = rng.normal(0.0,             self.alpha_y_sigma, size=n)
        N_A = rng.normal(self.alpha_a_mu, 1.0, size=n)

        # endogenous variables
        X = N_X
        p_A = sigmoid(0.7 * X**2 + N_A)  # P(A = 1 | X, N_A)
        A = rng.binomial(1, p_A)  # binary treatment
        Y = 0.5 * X + self.n_e * A + N_Y
        return X, A, Y

    # ----------------------------------------------------------
    #    Intervene on A :  do(A = a_fixed)
    # ----------------------------------------------------------
    def do_A(self,
             a_fixed,
             X: np.ndarray,
             n: int = 1,
             random_state=None
            ) -> np.ndarray:
        """
        Returns an array with shape  (n, *X.shape), where each row
        is an i.i.d. sample from  Y | do(A=a_fixed), X.

        Args:
        a_fixed (int- or array-like, values in {0, 1}): value to intervene on A, or array with X.shape
        X: observed values of X
        n (int): number of samples to draw
        random_state: seed for reproducibility
        """
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
        Y_do = 0.5 * X[None, ...] + self.n_e * a_fixed[None, ...] + N_Y
        return Y_do

    # ----------------------------------------------------------
    #    Closed-form distribution of  Y | do(A=a), X
    # ----------------------------------------------------------
    def analytic_do_distribution(self, a_fixed, X
                                ) -> Tuple[np.ndarray, float]:
        """
        Returns mean μ(X,a_fixed) and variance σ² of  Y | do(A=a_fixed), X.
            μ = 0.5·X + n_e·a_fixed
            σ² = (alpha_y_sigma)²       (independent of X and a_fixed)
        
        Args:
        a_fixed (int or np.ndarray): value to intervene on A, or array with X.shape
        X (np.ndarray): observed values of X
        """
        a_fixed = np.asarray(a_fixed)
        mu = 0.5 * np.asarray(X) + self.n_e * a_fixed
        var = self.alpha_y_sigma ** 2
        return mu, var


# Looping over different parameters of the SCM
def estimate_cate_from_row(row, *, n_mc: int = 0, seed: int | None = None):
    """
    This function is used to estimate CATE given a series of parameters to initialize
    the SCM.

    Args:
    row … a Series from df_params  (one set of α's and n_e)
    n_mc (int): If n_mc = 0, then the CATE is computed analytically (fast, exact).
                If n_mc > 0, then the CATE is computed using n_mc draws (Monte-Carlo sampling)
    seed (int, None): Seed for reproducibility of the random sampling.
    """
    scm = BinaryA_SCM(
        alpha_x_mu   = row["αXμ"],
        alpha_x_sigma= row["αXσ"],
        alpha_a_mu   = row["αAμ"],
        alpha_y_sigma= row["αYσ"],
        n_e          = row["NE"],
    )

    # Analytic version
    if n_mc == 0:
        # In this SCM  CATE(X) = E[Y|do(A=1),X] − E[Y|do(A=0),X] = n_e
        return scm.n_e
    else:
        # Monte-Carlo version
        rng = np.random.default_rng(seed)
        X, *_ = scm.sample(n=n_mc, random_state=rng)
        mu1, _ = scm.analytic_do_distribution(1, X)
        mu0, _ = scm.analytic_do_distribution(0, X)
        return (mu1 - mu0).mean()


if __name__ == "__main__":
    """
    Run a simple test of the BinaryA_SCM class
    """
    # Create an instance of the BinaryA_SCM class
    binary_scm = BinaryA_SCM(
        alpha_x_mu=0.0,
        alpha_x_sigma=1.0,
        alpha_a_mu=0.0,
        alpha_y_sigma=1.0,
        n_e=-0.5
        )

    # Sample from the SCM
    X, A, Y = binary_scm.sample(n=10, random_state=42)
    print("X:", X)
    print("A:", A)
    print("Y:", Y)

    # Intervene on A
    a_fixed = 1  # Intervening on A to be 1
    Y_do = binary_scm.do_A(a_fixed, X, n=10, random_state=42)
    print("Y | do(A=1):", Y_do)

    tmp = BinaryA_SCM(
        alpha_x_mu=0.0,
        alpha_x_sigma=1.0,
        alpha_a_mu=0.0,
        alpha_y_sigma=1.0,
        n_e=-0.5
    )

    tmp2 = BinaryA_SCM.sample(tmp, n=10, random_state=42)
    print(tmp2)
