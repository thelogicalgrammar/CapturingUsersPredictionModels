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

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


class SCM(ABC):
    """
    Abstract Base Class for SCMs.
    You need to overwrite __init__() and sample() in your subclass.
    """

    @abstractmethod
    def __init__():
        pass

    def __post_init__(self):
        self._flatten_covariate_parameters()

    @property
    def n_covariates(self):
        return int(self.alpha_mu_X.size)

    def _flatten_covariate_parameters(self):
        """
        Flatten covariate-related parameters to one-dimensional numpy arrays.
        Scalars are broadcast if another covariate parameter determines k > 1.
        """
        covariate_parameter_names = ('alpha_mu_X', 'alpha_sigma_X', 'b', 'd')
        raw_parameters = {}
        n_covariates = 1
        for parameter_name in covariate_parameter_names:
            parameter = np.atleast_1d(np.asarray(getattr(self, parameter_name), dtype=float))
            if parameter.ndim != 1:
                raise ValueError(f"{parameter_name} must be scalar or one-dimensional.")
            raw_parameters[parameter_name] = parameter
            n_covariates = max(n_covariates, parameter.size)

        for parameter_name, parameter in raw_parameters.items():
            if parameter.size == 1 and n_covariates > 1:
                parameter = np.full(n_covariates, parameter.item(), dtype=float)
            elif parameter.size != n_covariates:
                text = f"{parameter_name} must have length {n_covariates}, not {parameter.size}."
                raise ValueError(text)
            setattr(self, parameter_name, parameter)

    def _coerce_X(self, X):
        """
        Convert X to a matrix representation with shape (n, k) for internal use.
        Returns the original numpy representation, the matrix representation, and
        whether X encodes a single observation.
        """
        X_array = np.asarray(X, dtype=float)
        if self.n_covariates == 1:
            X_matrix = X_array.reshape(-1, 1)
            single_observation = X_array.shape == ()
            return X_array, X_matrix, single_observation

        if X_array.shape == (self.n_covariates,):
            return X_array, X_array.reshape(1, -1), True
        if X_array.ndim >= 2 and X_array.shape[-1] == self.n_covariates:
            return X_array, X_array.reshape(-1, self.n_covariates), False
        text = f"X must have shape ({self.n_covariates},) or (..., {self.n_covariates})."
        raise ValueError(text)

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
        self._flatten_covariate_parameters()

    @abstractmethod
    def sample():
        pass

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


@dataclass
class SCM_WithConstants(SCM):
    """
    Class for SCM with continuous A, continuous X, and continuous Y.
    This SCM encodes information so that \n
    N_Xi ~ N(alpha_mu_X, alpha_sigma_X) \n
    N_Yi ~ N(alpha_mu_Y, alpha_sigma_Y) \n
    N_Ai ~ N(alpha_mu_A, alpha_sigma_A) \n
    Y_i = a + b * X_i + N_E * A_i + N_Yi \n
    A_i = d * X_i + N_Ai \n
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
        b (float): Constant for SCM
        d (float): Constant for SCM
    """
    alpha_mu_X: float = 80
    alpha_sigma_X: float = 10
    alpha_mu_A: float = 2
    alpha_sigma_A: float = 1.0
    alpha_mu_Y: float = 0.0
    alpha_sigma_Y: float = 0.1
    N_E: float = 1.0
    a: float = 12.0
    b: float = -0.1
    d: float = 0.125


    def sample(self, X=None, A=None, draws=1, rng=None):
        r"""
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
            X_matrix = rng.normal(self.alpha_mu_X, self.alpha_sigma_X,
                                  size=(draws, self.n_covariates))
            X_output = X_matrix[:, 0] if self.n_covariates == 1 else X_matrix
            single_observation = False
        else:
            X_output, X_matrix, single_observation = self._coerce_X(X)
        if A is None:
            # Sample N_A and calculate A
            if X is None:
                A_mu = X_matrix @ self.d
            elif single_observation:
                A_mu = np.full(draws, X_matrix[0] @ self.d, dtype=float)
            else:
                A_mu = X_matrix @ self.d
            N_A = rng.normal(self.alpha_mu_A, self.alpha_sigma_A, size=A_mu.shape[0])
            A_array = A_mu + N_A
            A_output = A_array
        else:
            A_output = np.asarray(A, dtype=float)
            if X is None or single_observation:
                if A_output.shape == ():
                    A_array = np.full(draws, A_output.item(), dtype=float)
                else:
                    A_array = A_output.reshape(-1)
                    if A_array.size == 1:
                        A_array = np.full(draws, A_array.item(), dtype=float)
                    elif A_array.size != draws:
                        raise ValueError("A must be scalar or have length equal to draws.")
            else:
                A_array = A_output.reshape(-1)
                if A_array.size == 1:
                    A_array = np.full(X_matrix.shape[0], A_array.item(), dtype=float)
                elif A_array.size != X_matrix.shape[0]:
                    raise ValueError("A must be scalar or match the number of X observations.")
        # Sample N_Y, center around zero and add self.alpha_mu_Y later
        # to also be able to extract parameter mu_Y_cond.
        N_Y_0_centered = rng.normal(0, self.alpha_sigma_Y, size=A_array.shape[0])
        # Estimate conditional parameters for Y|X,do(A) ~ N(mu_Y, sigma_Y)
        # If neither X nor A was passed, the generated Y is "conditional" on
        # the sampled values (this is not what "conditional" usually refers to).
        # If X was passed, the output Y is conditional on the passed value. 
        # If A was passed, the output Y is conditional on do(A).
        if X is None:
            X_effect = X_matrix @ self.b
        elif single_observation:
            X_effect = np.full(A_array.shape[0], X_matrix[0] @ self.b, dtype=float)
        else:
            X_effect = X_matrix @ self.b
        mu_Y_cond = self.a + X_effect + self.N_E * A_array + self.alpha_mu_Y
        std_Y_cond = np.full_like(mu_Y_cond, self.alpha_sigma_Y, dtype=float)
        # Model defines the treatement effect as independent of X.
        Y = mu_Y_cond + N_Y_0_centered
        return X_output, A_output, Y, mu_Y_cond, std_Y_cond
