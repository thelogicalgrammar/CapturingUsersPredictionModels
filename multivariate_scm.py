"""
Structural causal model (SCM) with k-dimensional covariates.

This is the multivariate counterpart of :class:`scm.SCM`. It is used both
-for data generation, and
-for the agent's internal representation in the multivariate model.
The agent uses this model to prescribe treatment based on his prior beliefs
before seeing new information, and based on his posterior beliefs after seeing
new information.

The covariate vector ``X`` is k-dimensional, while the treatment ``A`` and the
outcome ``Y`` stay scalar. The structural equations are: \n
N_Xi ~ N_k(alpha_mu_X, diag(alpha_sigma_X^2)) \n
N_Yi ~ N(0, alpha_sigma_Y) \n
N_Ai ~ N(alpha_mu_A, alpha_sigma_A) \n
X_i = N_Xi \n
A_i = N_d.T X_i + N_Ai \n
Y_i = N_b.T X_i + N_E * A_i + N_Yi

Relative to the scalar :class:`scm.SCM`, the scalar coefficients ``b`` and ``d``
become the coefficient vectors ``N_b`` and ``N_d`` of length k, the products
``b * X`` and ``d * X`` become the dot products ``N_b.T X`` and ``N_d.T X``,
and the structural intercept ``a`` is absorbed into ``alpha_mu_Y`` (which is 0
in the scalar paper model and defaults to 12 here). For k = 1 the model reduces
to the scalar SCM.

Notes:
Vector-valued parameters (``alpha_mu_X``, ``alpha_sigma_X``, ``N_b``, ``N_d``)
accept either a scalar (broadcast to length k) or a one-dimensional array of
length k.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class MultivariateSCM:
    """
    Class for an SCM with continuous k-dimensional X, continuous A, and
    continuous Y. This SCM encodes information so that \n
    N_Xi ~ N_k(alpha_mu_X, diag(alpha_sigma_X^2)) \n
    N_Yi ~ N(0, alpha_sigma_Y) \n
    N_Ai ~ N(alpha_mu_A, alpha_sigma_A) \n
    X_i = N_Xi \n
    A_i = N_d.T X_i + N_Ai \n
    Y_i = N_b.T X_i + N_E * A_i + N_Yi

    The structural intercept of the scalar SCM is absorbed into alpha_mu_Y.

    Args:
        n_covariates (int): Number of covariates k (dimension of X).
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

    def __post_init__(self):
        """
        Validate n_covariates and normalize the vector-valued parameters
        to one-dimensional arrays of length k after dataclass construction.
        """
        self.n_covariates = int(self.n_covariates)
        if self.n_covariates < 1:
            raise ValueError("n_covariates must be positive.")
        self._normalize_vector_parameters()

    def _normalize_vector_parameter(self, parameter, parameter_name):
        """
        Convert scalar or one-dimensional input to a vector of length k.

        A scalar is broadcast to a constant vector of length k. A
        one-dimensional array must already have length k.

        Args:
            parameter (float or array-like): Value to normalize.
            parameter_name (str): Name used in error messages.

        Returns:
            np.ndarray: One-dimensional array of length k.
        """
        parameter = np.atleast_1d(np.asarray(parameter, dtype=float))
        if parameter.ndim != 1:
            raise ValueError("{} must be scalar or one-dimensional.".format(parameter_name))
        if parameter.size == 1:
            parameter = np.full(self.n_covariates, parameter.item(), dtype=float)
        elif parameter.size != self.n_covariates:
            text = "{} must have length {}, not {}.".format(
                parameter_name, self.n_covariates, parameter.size)
            raise ValueError(text)
        return parameter

    def _normalize_vector_parameters(self):
        """
        Normalize all vector-valued SCM parameters in place.

        Applies :meth:`_normalize_vector_parameter` to ``alpha_mu_X``,
        ``alpha_sigma_X``, ``N_b``, and ``N_d`` so the rest of the class can
        assume one-dimensional arrays of length k.
        """
        for parameter_name in ('alpha_mu_X', 'alpha_sigma_X', 'N_b', 'N_d'):
            parameter = self._normalize_vector_parameter(
                getattr(self, parameter_name), parameter_name)
            setattr(self, parameter_name, parameter)

    def _coerce_X(self, X):
        """
        Return X both as an array and as a two-dimensional matrix.

        Accepts a single observation with shape ``(k,)`` or a batch with
        shape ``(n, k)`` and always exposes a two-dimensional matrix view,
        so downstream code does not have to special-case the single-row
        matrix that numpy would otherwise collapse to a vector.

        Args:
            X (array-like): Observation(s) of X, shape ``(k,)`` or ``(n, k)``.

        Returns:
            tuple: ``(X_array, X_matrix, single_observation)`` where
            ``X_array`` is the input as an array, ``X_matrix`` is its
            two-dimensional view, and ``single_observation`` is True when a
            single ``(k,)`` observation was passed.
        """
        X_array = np.asarray(X, dtype=float)
        if X_array.shape == (self.n_covariates,):
            return X_array, X_array.reshape(1, -1), True
        if X_array.ndim == 2 and X_array.shape[1] == self.n_covariates:
            return X_array, X_array, False
        text = "X must have shape ({},) or (n, {}).".format(
            self.n_covariates, self.n_covariates)
        raise ValueError(text)

    def update_parameters(self, parameter_dict):
        """
        Method to update chosen parameters in SCM.

        After updating, the vector-valued parameters are renormalized to
        length-k arrays so a scalar or vector update is handled consistently.

        Args:
            parameter_dict (dict): Dictionary of parameters where the keys
                match parameter names in the SCM.
        """
        for key, value in parameter_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError("Unknown parameter: {}".format(key))
        self._normalize_vector_parameters()

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

    def sample_do_A(self, A_treatment, X, draws=1, rng=None):
        """
        Generates sample from the SCM given the data and set treatments, i.e. \n
        :math:`Y_i \\sim p(Y|do(A\\_treatment), X)` \n
        This is merely a wrapper for the interventional path in
        :meth:`sample`.

        Args:
            A_treatment (int- or array-like): Value to intervene on A, or
                array matching the number of observations in X.
            X: Observed value(s) of X, shape ``(k,)`` or ``(n, k)``.
            draws (int): Size of sample to draw for every observation in A.
            rng: Seed or initialized random number generator for
                reproducibility.

        Returns:
            tuple: ``(Y_do, Y_do_mean, Y_do_std)``.
        """
        _X, _A, Y_do, *_ = self.sample(X=X, A=A_treatment, draws=draws, rng=rng)
        Y_do_mean = np.mean(Y_do)
        Y_do_std = np.std(Y_do)
        return Y_do, Y_do_mean, Y_do_std

    def sample_CATE(self, X, A1_treatment, A0_treatment, draws=1, rng=None):
        """
        This method samples the CATE-distribution using treatment 1 (higher
        dosage) instead of treatment 0, given X and two treatment options
        A1_treatment and A0_treatment.

        Args:
            X: Feature(s) to condition on, shape ``(k,)`` or ``(n, k)``.
            A1_treatment (float): Dosage value of higher dose treatment.
            A0_treatment (float): Dosage value of lower dose treatment.
            draws (int): Size of sample to generate.
            rng: Seed or initialized random number generator for
                reproducibility.

        Returns:
            tuple: ``(CATE_samples, CATE_mean, CATE_std)``. The mean and std
            are None when ``draws <= 2``.
        """
        Y_do1, *_ = self.sample_do_A(A1_treatment, X, draws=draws, rng=rng)
        Y_do0, *_ = self.sample_do_A(A0_treatment, X, draws=draws, rng=rng)
        CATE_samples = Y_do1 - Y_do0
        if draws > 2:
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

        The SCM in this class is defined so that Y given A and X is normally
        distributed and its parameters are deterministically determined by the
        values of A and X. This results in the CATE-distribution also being
        normal and allows for deterministic calculation of the mean as: \n
        CATE_mu(X) = E(Y| X, do(A=A1)) - E(Y| X, do(A=A0))

        Args:
            A1_treatment (float): Dosage value of higher dose treatment.
            A0_treatment (float): Dosage value of lower dose treatment.
        """
        return (A1_treatment - A0_treatment) * self.N_E
