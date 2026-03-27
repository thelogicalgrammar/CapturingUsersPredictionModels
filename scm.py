"""
Structural causal model (SCM) for 
-data generation, and
-for agent's internal representation.
Agent uses this model to prescribe treatment
based on his prior beliefs before seeing new
information, and based on his posterior beliefs
after seeing new information.

Notes:
Might be more efficient to reimplement the SCM
class using pymc, especially as the Bayesian agent
(agent model) is implemented using pymc.
"""

import numpy as np
from dataclasses import dataclass


# ---------- helpers ----------
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class SCM:
    """
    Class for SCM with continuous A, continuous X, and continuous Y.
    This SCM encodes information so that \n
    N_Xi ~ N(alpha_mu_X, alpha_sigma_X) \n
    N_Yi ~ N(alpha_mu_Y, alpha_sigma_Y) \n
    N_Ai ~ N(alpha_mu_A, alpha_sigma_A) \n
    Y_i = a + N_B * X_i + N_E * A_i + N_Yi \n
    A_i = N_D * X_i + N_Ai \n
    X_i = N_Xi

    Args:
        alpha_mu_X (float): Mean of X
        alpha_sigma_X (float): Standard deviation of X
        alpha_mu_A (float): Mean of A
        alpha_sigma_A (float): Standard deviation of A
        alpha_mu_Y (float): Mean of Y
        alpha_sigma_Y (float): Standard deviation of Y
        N_E (float): Effect size. The effect size is a fixed number. All
            variability is explained by variance in other RVs.
        a (float): Constant for SCM
        N_B (float): Constant for SCM
        N_D (float): Constant for SCM
    """
    alpha_mu_X: float = 80
    alpha_sigma_X: float = 10
    alpha_mu_A: float = 2
    alpha_sigma_A: float = 1.0
    alpha_mu_Y: float = 0.0
    alpha_sigma_Y: float = 0.1
    N_E: float = 1.0
    a: float = 12.0
    N_B: float = -0.1
    N_D: float = 0.125

    def update_parameters(self, parameter_dict):
        """
        Method to update chosen parameters in SCM

        Args:
            parameter_dict (dict): Dictionary of parameters
                where the keys match parameter names in the scm.
        """
        for key, value in parameter_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError("Unknown parameter: {}".format(key))

    def sample(self, X=None, A=None, draws=1, rng=None):
        """
        This method encodes the structure of the SCM and generates 
        samples of X, Y, and A: \n
        :math:`X, A, Y \sim p(X, Y, A)` \n
        If X is passed, then the output A and Y are conditional on
        the value of X: \n
        :math:`A, Y \sim p(A, Y| X)` \n
        If A is passed, the output Y is conditional on the
        intervention do(A): \n
        :math:`X, Y \sim p(X, Y| do(A))` \n
        If both X and Y are passed, the output is conditional on X
        and on the intervention do(A): \n
        :math:`Y \sim p(Y| X, do(A))` \n
        All of X, A, and Y are always returned. In addition, the distribution
        parameters for Y (mu_Y_cond, and std_Y_cond) are returned. If X
        and/or A are provided, the distribution parameters are conditional
        on the given values. Otherwise they are "conditional" only on the
        sampled values (in contrast how the word "conditional" is normally
        used).

        Args:
            X: Optional input. If provided, output will be conditional on X.
            A: Optional input. If provided, output will be conditional on do(A).
            draws (int): Size of sample to generate.
            rng: seed or initialized random number generator for reproducibility
        """
        # Sample noise when needed:
        rng = np.random.default_rng(rng)
        if X is None:
            # Sample N_X and calculate X
            N_X = rng.normal(self.alpha_mu_X, self.alpha_sigma_X, size=draws)
            X = N_X
        if A is None:
            # Sample N_A and calculate A
            N_A = rng.normal(self.alpha_mu_A, self.alpha_sigma_A, size=draws)
            # Calculate A
            A = self.N_D * X + N_A
        # Sample N_Y, center around zero and add self.alpha_mu_Y later
        # to also be able to extract parameter mu_Y_cond.
        N_Y_0_centered = rng.normal(0, self.alpha_sigma_Y, size=draws)
        # Estimate conditional parameters for Y|X,do(A) ~ N(mu_Y, sigma_Y)
        # If neither X nor A was passed, the generated Y is "conditional" on
        # the sampled values (this is not what "conditional" usually refers to).
        # If X was passed, the output Y is conditional on the passed value. 
        # If A was passed, the output Y is conditional on do(A).
        mu_Y_cond = self.a + self.N_B * X + self.N_E * A + self.alpha_mu_Y
        std_Y_cond = np.full_like(mu_Y_cond, self.alpha_sigma_Y)
        # Model defines the treatement effect as independent of X.
        Y = mu_Y_cond + N_Y_0_centered
        # {'X': X, 'Y': Y, 'A': A, 'mu_Y_cond': mu_Y_cond, 'std_Y_cond': std_Y_cond}
        return X, A, Y, mu_Y_cond, std_Y_cond

    def sample_do_A(self, A_treatment, X: np.ndarray, draws: int = 1,
                    rng=None) -> np.ndarray:
        """
        Generates sample from the SCM given the data and set treatments, i.e. \n
        Y_i ~ p(Y|do(A_treatment), X) \n
        Returns an array with shape (n, *X.shape).
        This is merely a wrapper for the interventional path in self.sample().

        Args:
            A_treatment (int- or array-like): value to intervene on A, or array with X.shape
            X: observed values of X
            draws (int): Size of sample to draw for every observation in A.
            rng: seed or initialized random number generator for reproducibility
        """
        # Either include checks everywhere or nowhere.
        # X = np.asarray(X)
        # A_treatment = np.asarray(A_treatment)
        # if A_treatment.shape == ():  # broadcast scalar
        #     A_treatment = np.full_like(X, A_treatment)
        # if A_treatment.shape != X.shape:
        #     raise ValueError("`A_treatment` and `X` must have the same shape "
        #                      "or `A_treatment` must be scalar.")
        _X, _A, Y_do, *_ = self.sample(X, A=A_treatment, draws=draws, rng=rng)
        Y_do_mean = np.mean(Y_do)
        Y_do_std = np.std(Y_do)
        return Y_do, Y_do_mean, Y_do_std

    def sample_CATE(self, X, A1_treatment, A0_treatment, draws=1, rng=None):
        """
        This method samples the CATE-distribution using treatment 1 (higher dosage)
        instead of treatment 0.
        Given X and two treatment options A1_treatment and A0_treatment

        Args:
            X (float): Feature to condition on
            A1_treatment (float): Dosage value of higher dose treatment
            A0_treatment (float): Dosage value of lower dose treatment
            draws (int): Size of sample to generate
            rng: seed or initialized random number generator for reproducibility
        """
        # Sample an outcome with treatment 1:
        Y_do1, *_ = self.sample_do_A(A1_treatment, X, draws, rng=rng)
        # Sample an outcome with treatment 0:
        Y_do0, *_ = self.sample_do_A(A0_treatment, X, draws, rng=rng)
        # The sample drawn from the CATE-distribution:
        CATE_samples = Y_do1 - Y_do0
        if draws > 2:
            # Estimate mean and standard deviation from sample
            CATE_mean = np.mean(CATE_samples)
            CATE_std = np.std(CATE_samples)
        else:
            CATE_mean = None
            CATE_std = None
        return CATE_samples, CATE_mean, CATE_std

    def get_CATE_mu(self, A1_treatment, A0_treatment):
        """
        This method returns the mean of the CATE-distribution given
        X, A1_treatment, and A0_treatment. Due to how the SCM is defined,
        this value is equivalent to (A1_treatment - A0_treatment) * self.N_E.

        The SCM in this class is defined so that Y given A and X is
        normally distributed and its parameters are deterministically
        determined by the values of A and X. This results in the CATE-
        distibution also being normal and allows for deterministic
        calculation of the mean as: \n
        CATE_mu(X) = E(Y| X, do(A=A1)) - E(Y| X, do(A=A0))

        Args:
            A1_treatment (float): Dosage value of higher dose treatment
            A0_treatment (float): Dosage value of lower dose treatment
        """
        return (A1_treatment - A0_treatment) * self.N_E


if __name__ == "__main__":
    """
    Run a simple test of the SCM class
    """
    import tests.test_scm
    # If no crash, then the test passed.
    print("\nSCM test passed.")
