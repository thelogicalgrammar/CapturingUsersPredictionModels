"""
3D illustration of the joint likelihood of the collapsed model over two
features.

For a grid of hypothetical patients :math:`X^o = (x_1, x_2)`, the external
prediction model produces the prediction :math:`pred(X^o)`, and the agent
evaluates the joint likelihood of observing that patient together with that
prediction under his prior beliefs: \n
:math:`L(x_1, x_2) = p(X^o \\mid \\alpha_{\\mu_X}, \\alpha_{\\sigma_X})
\\cdot p(pred(X^o) \\mid X^o)` \n
The first factor is the covariate likelihood of the agent model. The second
factor is the collapsed likelihood
:math:`\\mathcal{N}(pred(X^o); \\mu_{red}, V_{red})` with the auxiliaries
(R, T) integrated out by Monte Carlo, using the same Cochran construction
of R as ``single_prediction_collapsed_agent_model.py``. All population-level
parameters are fixed at the agent's prior means, so the surface shows where
in feature space the agent expects to see a patient with the decision
support's own prediction - i.e. what the belief update treats as plausible
before any data is observed.

The figure plots feature one and feature two on the horizontal axes and the
joint likelihood as the height of the surface.

Run from the repository root:

.. code-block:: bash

    python plot_joint_likelihood_surface.py

Optionally one agent prior can be modified, with the same convention as
experiment_collapsed.py. For example,

.. code-block:: bash

    python plot_joint_likelihood_surface.py N_E_mu -20

shows the surface for an agent whose prior belief about the effect size is
badly misaligned. The figure is saved to
``./fig/joint_likelihood_surface.pdf`` (with the modified prior appended to
the file name when one is given).
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
# Project specific:
import scm


# Two features by construction: the illustration is a surface over a plane.
n_covariates = 2
plate_size = 1000
# Number of Monte Carlo draws used to integrate out the auxiliaries (R, T).
marginal_draws = 512
# Grid resolution per feature axis, spanning +-3 standard deviations.
grid_points = 61
grid_standard_deviations = 3.0
scm_noise_parameters = {'n_covariates': n_covariates,
                        'alpha_mu_X': 80.0, 'alpha_sigma_X': 10.0,
                        'alpha_mu_A': 2.0, 'alpha_sigma_A': 1.0,
                        'alpha_mu_Y': 12.0, 'alpha_sigma_Y': 0.1,
                        'N_E': 1.0,
                        'N_b': -0.1 / n_covariates,
                        'N_d': 0.125 / n_covariates}
# Agent prior beliefs, fixed at their means for the illustration.
# Scalar entries are broadcast to length k.
agent_priors =\
    {'mu_X': 80.0, 'mu_sigma_X': 10.0,
     'mu_A': 2.0, 'mu_sigma_A': 1.0,
     'mu_Y': 12.0, 'mu_sigma_Y': 0.1,
     'N_E_mu': 1.0,
     'N_b_mu': -0.1 / n_covariates, 'N_b_sigma': 0.1,
     'N_d_mu': 0.125 / n_covariates, 'N_d_sigma': 0.1}
rng = 4  # Selected by fair die


def train_prediction_model(scm_noise_parameters, plate_size, rng):
    """
    Train the external prediction model on one historical dataset.

    Mirrors steps 1-3 of experiment_collapsed.py: sample the historical
    plate from the data-generating SCM and fit an OLS regression with
    intercept on (X, Y).

    Args:
        scm_noise_parameters (dict): Parameters of the data-generating SCM.
        plate_size (int): Number of historical patients.
        rng: Seed or initialized random number generator for
            reproducibility.

    Returns:
        LinearRegression: The fitted prediction model.
    """
    data_scm = scm.MultivariateSCM(**scm_noise_parameters)
    X, A, Y_historical, *_ = data_scm.sample(draws=plate_size, rng=rng)
    return LinearRegression(fit_intercept=True).fit(X, Y_historical)


def joint_likelihood_surface(feature_grid, pred_model, agent_priors,
                             plate_size, marginal_draws, rng):
    """
    Evaluate the joint likelihood on a grid of two-feature patients.

    For every grid point x the joint likelihood factorizes as \n
    :math:`L(x) = \\prod_j \\mathcal{N}(x_j; \\mu_{X,j}, \\sigma_{X,j})
    \\cdot \\mathbb{E}_{R,T}[\\mathcal{N}(pred(x); \\mu_{red}(x),
    \\sqrt{V_{red}(x, R, T)})]` \n
    with the expectation over (R, T) estimated from marginal_draws Monte
    Carlo draws shared across grid points. R is drawn with the Cochran
    construction, so its non-centrality follows the grid point through
    :math:`\\Delta_X^2(x)`.

    Args:
        feature_grid (np.ndarray): Grid values shared by both feature axes.
        pred_model (LinearRegression): The fitted prediction model.
        agent_priors (dict): Agent prior beliefs (means used as plug-ins).
        plate_size (int): Number of historical patients n.
        marginal_draws (int): Monte Carlo draws for the (R, T) expectation.
        rng: Seed or initialized random number generator for
            reproducibility.

    Returns:
        tuple: ``(grid_1, grid_2, likelihood)`` as 2D arrays for plotting.
    """
    rng = np.random.default_rng(rng)
    grid_1, grid_2 = np.meshgrid(feature_grid, feature_grid)
    points = np.column_stack([grid_1.ravel(), grid_2.ravel()])
    predictions = pred_model.predict(points)

    mu_X = np.full(n_covariates, agent_priors['mu_X'])
    sigma_X = np.full(n_covariates, agent_priors['mu_sigma_X'])
    N_E = agent_priors['N_E_mu']
    N_b_mu = np.full(n_covariates, agent_priors['N_b_mu'])
    N_b_sigma = np.full(n_covariates, agent_priors['N_b_sigma'])
    N_d_mu = np.full(n_covariates, agent_priors['N_d_mu'])
    N_d_sigma = np.full(n_covariates, agent_priors['N_d_sigma'])

    # Covariate likelihood p(X_o | alpha_mu_X, alpha_sigma_X).
    covariate_likelihood = np.prod(
        norm.pdf(points, loc=mu_X, scale=sigma_X), axis=1)

    # Collapsed likelihood terms (population parameters at prior means).
    c = agent_priors['mu_Y'] + N_E * agent_priors['mu_A']
    tau_squared = (N_E**2 * agent_priors['mu_sigma_A']**2 +
                   agent_priors['mu_sigma_Y']**2)
    prediction_mu = c + points @ (N_b_mu + N_E * N_d_mu)
    coefficient_variance = np.sum((points * N_b_sigma)**2, axis=1)
    coefficient_variance += N_E**2 * np.sum((points * N_d_sigma)**2, axis=1)

    # Integrate out (R, T) by Monte Carlo, draws shared across the grid.
    # R follows the grid point through its non-centrality Delta_X^2(x).
    delta = np.sqrt(np.sum(((points - mu_X) / sigma_X)**2, axis=1))
    G_R = rng.normal(size=marginal_draws)
    C_R = rng.chisquare(df=n_covariates - 1, size=marginal_draws)
    T = rng.chisquare(df=plate_size - n_covariates, size=marginal_draws)
    R = ((delta[:, None] + G_R[None, :] / np.sqrt(plate_size))**2 +
         C_R[None, :] / plate_size)
    variance = (coefficient_variance[:, None] +
                tau_squared * (1.0 / plate_size + R / T[None, :]))
    prediction_likelihood = np.mean(
        norm.pdf(predictions[:, None],
                 loc=prediction_mu[:, None],
                 scale=np.sqrt(variance)),
        axis=1)

    likelihood = covariate_likelihood * prediction_likelihood
    return grid_1, grid_2, likelihood.reshape(grid_1.shape)


def plot_surface(grid_1, grid_2, likelihood,
                 file_name='./fig/joint_likelihood_surface.pdf'):
    """
    Plot the joint likelihood as a 3D surface over the two features.

    Args:
        grid_1 (np.ndarray): 2D grid of feature one.
        grid_2 (np.ndarray): 2D grid of feature two.
        likelihood (np.ndarray): 2D grid of joint likelihood values.
        file_name (str): Path of the saved figure.
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection='3d')
    surface = ax.plot_surface(grid_1, grid_2, likelihood,
                              cmap='viridis', linewidth=0)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Joint likelihood')
    fig.colorbar(surface, shrink=0.6, pad=0.12)
    Path(file_name).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(file_name)
    print('Figure saved to {}'.format(file_name))
    plt.show()


def main(file_name='./fig/joint_likelihood_surface.pdf'):
    """
    Train the prediction model, evaluate the surface, and plot it.

    Args:
        file_name (str): Path of the saved figure.
    """
    generator = np.random.default_rng(rng)
    pred_model = train_prediction_model(scm_noise_parameters, plate_size,
                                        generator)
    half_width = grid_standard_deviations * agent_priors['mu_sigma_X']
    feature_grid = np.linspace(agent_priors['mu_X'] - half_width,
                               agent_priors['mu_X'] + half_width,
                               grid_points)
    grid_1, grid_2, likelihood = joint_likelihood_surface(
        feature_grid, pred_model, agent_priors, plate_size,
        marginal_draws, generator)
    plot_surface(grid_1, grid_2, likelihood, file_name=file_name)


if __name__ == '__main__':
    file_name = './fig/joint_likelihood_surface.pdf'
    if len(sys.argv) >= 3:
        # Modify one agent prior, same convention as experiment_collapsed.py.
        param_name = sys.argv[1]
        param_value = float(sys.argv[2])
        if param_name not in agent_priors.keys():
            print('Parameter has to exist in agent_priors. See')
            print(agent_priors)
            text = '{} not a valid parameter!'.format(param_name)
            raise ValueError(text)
        agent_priors[param_name] = param_value
        file_name = './fig/joint_likelihood_surface_{}_{}.pdf'.format(
            param_name, param_value)
        print('Plotting with agent prior {} set to {}...'.format(
            param_name, param_value))
    main(file_name=file_name)
