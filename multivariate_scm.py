"""
Structural causal model with k-dimensional covariates.

The model is:

    A_i = N_d.T X_i + N_Ai
    Y_i = N_b.T X_i + N_E A_i + N_Yi
    X_i = N_Xi

Any structural intercept is included in alpha_mu_Y.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class MultivariateSCM:
    """
    SCM with continuous X, A, and Y.
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
        self.n_covariates = int(self.n_covariates)
        if self.n_covariates < 1:
            raise ValueError("n_covariates must be positive.")
        self._normalize_vector_parameters()

    def _normalize_vector_parameter(self, parameter, parameter_name):
        """
        Convert scalar or one-dimensional input to a vector of length k.
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
        Normalize all vector-valued SCM parameters.
        """
        for parameter_name in ('alpha_mu_X', 'alpha_sigma_X', 'N_b', 'N_d'):
            parameter = self._normalize_vector_parameter(
                getattr(self, parameter_name), parameter_name)
            setattr(self, parameter_name, parameter)

    def _coerce_X(self, X):
        """
        Return X as an array and as a two-dimensional matrix.
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
        """
        for key, value in parameter_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError("Unknown parameter: {}".format(key))
        self._normalize_vector_parameters()

    def sample(self, X=None, A=None, draws=1, rng=None):
        """
        Generate samples from the SCM.
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
        Generate Y samples given X and the intervention do(A_treatment).
        """
        _X, _A, Y_do, *_ = self.sample(X=X, A=A_treatment, draws=draws, rng=rng)
        Y_do_mean = np.mean(Y_do)
        Y_do_std = np.std(Y_do)
        return Y_do, Y_do_mean, Y_do_std

    def sample_CATE(self, X, A1_treatment, A0_treatment, draws=1, rng=None):
        """
        Sample the CATE distribution for two treatment choices.
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
        Return the mean CATE implied by the SCM.
        """
        return (A1_treatment - A0_treatment) * self.N_E
