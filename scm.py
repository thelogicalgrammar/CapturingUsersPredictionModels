"""
Structural causal model (SCM) for
-data generation, and
-for agent's internal representation.
Agent uses this model to prescribe treatment
based on his prior beliefs before seeing new
information, and based on his posterior beliefs
after seeing new information.

Defines the abstract :class:`SCM` base class, the scalar :class:`SCM_WithConstants`
(fixed scalar or k-dimensional-with-fixed-coefficients structural model), and
:class:`MultivariateSCM` (k-dimensional X with its own coefficient vectors
``N_b``/``N_d`` and intercept folded into ``alpha_mu_Y``).

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

    def _covariate_parameter_names(self):
        """
        Names of the covariate-related parameters that need to be
        flattened to length-k arrays by :meth:`_flatten_covariate_parameters`.
        Overridable so subclasses can use different coefficient names.
        """
        return ('alpha_mu_X', 'alpha_sigma_X', 'b', 'd')

    def _flatten_covariate_parameters(self):
        """
        Flatten covariate-related parameters to one-dimensional numpy arrays
        and set ``self.n_covariates`` accordingly.

        Scalars are broadcast if another covariate parameter, or an
        already-set ``self.n_covariates`` (e.g. requested explicitly by a
        subclass's constructor before this runs), determines k > 1.
        """
        covariate_parameter_names = self._covariate_parameter_names()
        raw_parameters = {}
        n_covariates = getattr(self, 'n_covariates', 1)
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
        self.n_covariates = n_covariates

    def _coerce_X(self, X):
        """
        Convert X to a matrix representation with shape (n, k) for internal use.
        Returns the original numpy representation, the matrix representation, and
        whether X encodes a single observation.
        """
        X_array = np.asarray(X, dtype=float)
        if self.n_covariates == 1:
            X_matrix = X_array.reshape(-1, 1)
            single_observation = X_array.shape in ((), (1,))
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
        Sample the conditional mean contrast between two treatments.

        Within one SCM, the outcome noise is averaged out in each
        interventional expectation. The contrast is therefore
        ``(A1_treatment - A0_treatment) * N_E`` for every draw. Uncertainty
        over CATE is introduced by sampling different SCM parameters in the
        agent model, not by independently sampling realized outcomes here.

        Args:
            X (float): Feature to condition on
            A1_treatment (float): Dosage value of higher dose treatment
            A0_treatment (float): Dosage value of lower dose treatment
            draws (int): Size of sample to generate
            rng: Accepted for API compatibility. The conditional mean
                contrast is deterministic within one SCM, so the random
                stream is not consumed.
        """
        _X, X_matrix, single_observation = self._coerce_X(X)
        sample_size = draws if single_observation else X_matrix.shape[0]
        CATE_samples = np.full(
            sample_size,
            self.get_CATE_mu(A1_treatment, A0_treatment),
            dtype=float)
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
        A1_treatment and A0_treatment. Due to how the SCM is defined, this
        value is equivalent to (A1_treatment - A0_treatment) * self.N_E.

        The SCM in this class is defined so that the conditional expectation
        of Y is linear in A. Outcome noise is averaged out in the two
        interventional expectations, which allows deterministic calculation
        of their contrast as: \n
        CATE_mu(X) = E(Y| X, do(A=A1)) - E(Y| X, do(A=A0))

        Args:
            A1_treatment (float): Dosage value of higher dose treatment.
            A0_treatment (float): Dosage value of lower dose treatment.
        """
        return (A1_treatment - A0_treatment) * self.N_E


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


@dataclass
class MultivariateSCM(SCM):
    """
    Class for an SCM with continuous k-dimensional X, continuous A, and
    continuous Y. This SCM encodes information so that \n
    N_Xi ~ N_k(alpha_mu_X, diag(alpha_sigma_X^2)) \n
    N_Yi ~ N(0, alpha_sigma_Y) \n
    N_Ai ~ N(alpha_mu_A, alpha_sigma_A) \n
    X_i = N_Xi \n
    A_i = N_d.T X_i + N_Ai \n
    Y_i = N_b.T X_i + N_E * A_i + N_Yi

    Relative to :class:`SCM_WithConstants`, the fixed scalar coefficients
    ``b`` and ``d`` become the fixed coefficient vectors ``N_b`` and ``N_d``
    of length k, and the structural intercept ``a`` is absorbed into
    ``alpha_mu_Y`` (which is 0 in the scalar paper model and defaults to 12
    here). For k = 1 the model reduces to the scalar SCM, except that X is
    not squeezed to a 1-D array here (see :meth:`SCM_WithConstants.sample`
    for the squeezing behavior).

    Args:
        n_covariates (int): Number of covariates k (dimension of X). Can be
            passed explicitly to force broadcasting of scalar parameters to
            length k, even when no other parameter implies k > 1.
        alpha_mu_X (float or array-like): Mean of X. Scalar (broadcast to
            length k) or one-dimensional array of length k.
        alpha_sigma_X (float or array-like): Standard deviation of X. Scalar
            (broadcast to length k) or one-dimensional array of length k.
        alpha_mu_A (float): Mean of A.
        alpha_sigma_A (float): Standard deviation of A.
        alpha_mu_Y (float): Mean of Y. Also carries the structural intercept.
        alpha_sigma_Y (float): Standard deviation of Y.
        N_E (float): Effect size. The effect size is a fixed number. All
            variability is explained by variance in other RVs.
        N_b (float or array-like): Coefficient vector mapping X to Y. Scalar
            (broadcast to length k) or one-dimensional array of length k.
        N_d (float or array-like): Coefficient vector mapping X to A. Scalar
            (broadcast to length k) or one-dimensional array of length k.
    """
    n_covariates: int = 2
    alpha_mu_X: object = 80.0
    alpha_sigma_X: object = 10.0
    alpha_mu_A: float = 2.0
    alpha_sigma_A: float = 1.0
    alpha_mu_Y: float = 12.0
    alpha_sigma_Y: float = 0.1
    N_E: float = 1.0
    N_b: object = -0.05
    N_d: object = 0.0625

    def _covariate_parameter_names(self):
        return ('alpha_mu_X', 'alpha_sigma_X', 'N_b', 'N_d')

    def sample(self, X=None, A=None, draws=1, rng=None):
        """
        This method encodes the structure of the SCM and generates samples
        of X, A, and Y: \n
        :math:`X, A, Y \\sim p(X, Y, A)` \n
        If X is passed, then the output A and Y are conditional on the value
        of X: \n
        :math:`A, Y \\sim p(A, Y| X)` \n
        If A is passed, the output Y is conditional on the intervention
        do(A): \n
        :math:`X, Y \\sim p(X, Y| do(A))` \n
        If both X and A are passed, the output is conditional on X and on the
        intervention do(A): \n
        :math:`Y \\sim p(Y| X, do(A))` \n
        All of X, A, and Y are always returned. In addition, the conditional
        distribution parameters for Y (mu_Y_cond, and std_Y_cond) are
        returned. If X and/or A are provided, these parameters are conditional
        on the given values. Otherwise they are "conditional" only on the
        sampled values (in contrast to how the word "conditional" is normally
        used).

        Shapes follow the dimension of the input. When X is a single ``(k,)``
        observation, A and Y have shape ``(draws,)``. When X is a batch with
        shape ``(n, k)``, A and Y have shape ``(n,)``. When X is not provided,
        X has shape ``(draws, k)`` and A and Y have shape ``(draws,)``.

        Args:
            X: Optional input, shape ``(k,)`` or ``(n, k)``. If provided,
                output will be conditional on X.
            A: Optional input. If provided, output will be conditional on
                do(A). Scalar (broadcast) or array matching the number of
                observations.
            draws (int): Size of sample to generate when X is not provided.
            rng: Seed or initialized random number generator for
                reproducibility.

        Returns:
            tuple: ``(X, A, Y, mu_Y_cond, std_Y_cond)``.
        """
        rng = np.random.default_rng(rng)
        if X is None:
            X_matrix = rng.normal(self.alpha_mu_X, self.alpha_sigma_X,
                                  size=(draws, self.n_covariates))
            X_output = X_matrix
            single_observation = False
        else:
            X_output, X_matrix, single_observation = self._coerce_X(X)

        if A is None:
            if single_observation:
                A_mu = np.full(draws, X_matrix[0] @ self.N_d, dtype=float)
            else:
                A_mu = X_matrix @ self.N_d
            N_A = rng.normal(self.alpha_mu_A, self.alpha_sigma_A,
                             size=A_mu.shape[0])
            A_output = A_mu + N_A
        else:
            A_array = np.asarray(A, dtype=float)
            if A_array.shape == ():
                size = draws if single_observation else X_matrix.shape[0]
                A_output = np.full(size, A_array.item(), dtype=float)
            else:
                A_output = A_array.reshape(-1)
                expected_size = draws if single_observation else X_matrix.shape[0]
                if A_output.size != expected_size:
                    text = "A must be scalar or have length {}.".format(expected_size)
                    raise ValueError(text)

        if single_observation:
            X_effect = np.full(A_output.shape[0], X_matrix[0] @ self.N_b,
                               dtype=float)
        else:
            X_effect = X_matrix @ self.N_b
        N_Y_centered = rng.normal(0, self.alpha_sigma_Y, size=A_output.shape[0])
        mu_Y_cond = self.alpha_mu_Y + X_effect + self.N_E * A_output
        std_Y_cond = np.full_like(mu_Y_cond, self.alpha_sigma_Y, dtype=float)
        Y = mu_Y_cond + N_Y_centered
        return X_output, A_output, Y, mu_Y_cond, std_Y_cond
