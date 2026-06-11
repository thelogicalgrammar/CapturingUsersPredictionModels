"""
This module provides functionality to run experiments that test agent decision-making
with and without decision support systems, using the single-prediction collapsed
agent model for k-dimensional covariates.

This is the multivariate counterpart of experiment.py. The data-generating process
and the prediction model are multivariate, the prediction model is fitted with an
intercept, and the agent's belief update uses the collapsed likelihood, which
marginalizes the structural coefficient vectors N_b and N_d analytically.

The script takes three command line parameters where the first one
is the key in agent_priors to modify, the second one what value
agent_priors[key] should take, and the third one what value every
coordinate of X_o should take.

Example:

.. code-block:: bash

    python experiment_collapsed.py N_E_mu -20 72

This would set the value of the prior distribution of N_E to -20 and
use 72 as the value of every feature of the observed patient.


Notes:
The collapsed agent model marginalizes N_b and N_d analytically. For the
agent-side predictive summaries and expected do(A) outcomes, the marginalized
coefficient vectors are reconstructed when needed: from their priors for prior
predictive sampling, and from their conditional Gaussian given (N_E, R, T, the
population-level parameters, and pred_o) for posterior predictive sampling.
The output schema is therefore identical to experiment.py, with no padded
fields.
"""
import sys
import time
# import matplotlib.pyplot as plt  # Not used in array job.
import numpy as np
from sklearn.linear_model import LinearRegression
# Project specific:
import multivariate_scm as mscm
import utils
import single_prediction_collapsed_agent_model as spcam



# Set number of covariates and number of observations used to train the
# prediction model. The plate size is also the plate size used in the
# agent's belief model.
n_covariates = 2
plate_size = 1000
# Coefficients are scaled by 1 / k so that the aggregate signal of the SCM
# stays close to the univariate case.
scm_noise_parameters = {'n_covariates': n_covariates,
                        'alpha_mu_X': 80.0, 'alpha_sigma_X': 10.0,
                        'alpha_mu_A': 2.0, 'alpha_sigma_A': 1.0,
                        'alpha_mu_Y': 12.0, 'alpha_sigma_Y': 0.1,
                        'N_E': 1.0,
                        'N_b': -0.1 / n_covariates,
                        'N_d': 0.125 / n_covariates}
draws = 1000  # Number of draws for trace. Warmup is set to 1000.
# Default values of agent priors. One value is overwritten in main().
# Scalar entries are broadcast to length k by the agent model.
agent_priors =\
    {'mu_X': 80.0, 'sigma_X': 0.1, 'mu_sigma_X': 10.0, 'sigma_sigma_X': 0.1,
     'mu_A': 2.0, 'sigma_A': 1.0, 'mu_sigma_A': 1.0, 'sigma_sigma_A': 0.1,
     'mu_Y': 12.0, 'sigma_Y': 0.01, 'mu_sigma_Y': 0.1, 'sigma_sigma_Y': 0.01,
     'N_E_mu': 1.0, 'N_E_sigma': 1.0,
     'N_b_mu': -0.1 / n_covariates, 'N_b_sigma': 0.1,
     'N_d_mu': 0.125 / n_covariates, 'N_d_sigma': 0.1}
# Treatment info
A1_treatment = 20
A0_treatment = 10
# Only if treatment A1 is expected to produce an outcome that is threshold=5 above
# the expected outcome with A0 is A1 used. This corresponds to e.g. A1 being more
# expensive or to A1 having larger side effects.
threshold = 5
# The collapsed model has no scm_constants, but the key is kept so the output
# matches the shared results schema of utils.write_results_to_file().
parameter_descriptions = {'agent_priors': 'Default',
                          'scm_constants': 'Default',
                          'scm_noise_parameters': 'Default'}
rng = 4  # Selected by fair die


def main(X_o, agent_priors, modified_prior,
         A1_treatment=A1_treatment, A0_treatment=A0_treatment,
         n_covariates=n_covariates, plate_size=plate_size, draws=draws,
         scm_noise_parameters=scm_noise_parameters,
         parameter_descriptions=parameter_descriptions,
         rng=rng):
    """
    Args:
        X_o (np.array): Feature vector of the observed patient (length k).
        agent_priors (dict): Dictionary with all values that encodes the
            agent's prior beliefs.
        modified_prior (dict): Information about which prior was changed
            and how much.
        A1_treatment (float): Value of treatment 1.
        A0_treatment (float): Value of treatment 0.
        n_covariates (int): Number of covariates k.
        plate_size (int): Size of plate / number or observations to generate
            for regression model. Measure of "evidence" for predictions from
            decision support model.
        draws (int): Size of sample to generate in sampling events.
        scm_noise_parameters (dict): Dict with keys needed in MultivariateSCM-class.
        rng (int or rng): Seed or rng for reproducibility.
    """
    start_time = time.time()
    # 1. SCM-instance to generate population data under the
    #    historical treatment policy.
    #    X is sampled from the population distribution, A
    #    is the treatment given under the historical treatment
    #    policy given X, and Y is the outcome given X and A.
    data_scm = mscm.MultivariateSCM(**scm_noise_parameters)

    # 2. Generate data. This is identical to the true
    #    underlying generating process and the outcomes
    #    Y are outcomes under the historical treatment
    #    policy.
    X, A, Y_historical, *_ = data_scm.sample(draws=plate_size, rng=rng)

    # 3. Train prediction model on historical data.
    #    Calculated as the Moore-Penrose pseudo-inverse. No randomness.
    #    The collapsed reduction assumes an OLS model with intercept.
    pred_model = LinearRegression(fit_intercept=True).fit(X, Y_historical)

    # 4. The prediction model is next used for decision support. The physician
    #    inputs the data of the patient and the prediction model produces a prediction.
    pred_o = pred_model.predict(X_o.reshape(1, -1))  # The observed prediction
    # print("The predicted outcome for this patient is {}.".format(pred_o[0]))

    # 5. Define the agent model.
    #    Using same prior structure for agent's internal beliefs.
    agent_model = spcam.SinglePredictionCollapsedAgentModel(
        n_covariates=n_covariates,
        plate_size=plate_size,
        prior_parameters=agent_priors)

    # 6. Predictive estimates and treatment decision under prior beliefs.
    #    Under the collapsed model the CATE depends only on N_E.
    prior_CATE_mu = agent_model.get_prior_CATE_mu(A1_treatment, A0_treatment)
    prior_CATE_samples_, prior_CATE_mean, prior_CATE_std = \
        agent_model.sample_prior_predictive_CATE(A1_treatment, A0_treatment,
                                                 draws=draws, rng=rng)
    prior_treatment = A1_treatment if prior_CATE_mean > threshold else A0_treatment
    prior_predictive_summaries = agent_model.sample_prior_predictive_summaries(rng=rng)

    # 7. Predictive estimates and treatment decision under posterior beliefs
    # The agent's beliefs are updated / the posterior given the observed data is sampled.
    agent_model.sample(X_o=np.asarray(X_o), pred_o=np.asarray(pred_o[0]),
                       draws=draws, rng=rng)
    # Model diagnostics
    trace_diagnostics = agent_model.trace_diagnostics()

    # Using updated agent beliefs to come to a decision on what is the best treatment
    posterior_CATE_samples_, posterior_CATE_mean, posterior_CATE_std =\
        agent_model.sample_posterior_predictive_CATE(A1_treatment, A0_treatment)
    posterior_treatment = A1_treatment if posterior_CATE_mean > threshold else A0_treatment
    posterior_predictive_summaries = agent_model.sample_posterior_predictive_summaries(rng=rng)

    # 8. Ground truth of CATE and potential outcomes
    #    based on the data generating SCM
    true_CATE_mu = data_scm.get_CATE_mu(A1_treatment, A0_treatment)
    true_CATE_samples_, true_CATE_mean, true_CATE_std =\
        data_scm.sample_CATE(X_o, A1_treatment, A0_treatment, draws=draws, rng=rng)
    # Potential outcomes under treatment A1 and A0
    # This currently generates one single draw for each and the
    # SCM includes randomness.
    potential_outcome_A1_samples_, potential_outcome_A1_mean, potential_outcome_A1_std =\
        data_scm.sample_do_A(A1_treatment, X_o, draws=draws, rng=rng)
    potential_outcome_A0_samples_, potential_outcome_A0_mean, potential_outcome_A0_std =\
        data_scm.sample_do_A(A0_treatment, X_o, draws=draws, rng=rng)

    # 9. Expected outcomes given agent's prior and posterior beliefs
    #    under treatment A0 and A1. The marginalized coefficient vectors are
    #    reconstructed inside the agent model.
    _, expected_prior_outcome_A1_mean, expected_prior_outcome_A1_std =\
        agent_model.sample_prior_predictive_do_A(X_o, A1_treatment, rng=rng)
    _, expected_prior_outcome_A0_mean, expected_prior_outcome_A0_std =\
        agent_model.sample_prior_predictive_do_A(X_o, A0_treatment, rng=rng)
    _, expected_posterior_outcome_A1_mean, expected_posterior_outcome_A1_std =\
        agent_model.sample_posterior_predictive_do_A(X_o, A1_treatment, rng=rng)
    _, expected_posterior_outcome_A0_mean, expected_posterior_outcome_A0_std =\
        agent_model.sample_posterior_predictive_do_A(X_o, A0_treatment, rng=rng)
    expected_outcomes = {
        'expected_prior_outcome_A1_mean': expected_prior_outcome_A1_mean,
        'expected_prior_outcome_A1_std': expected_prior_outcome_A1_std,
        'expected_prior_outcome_A0_mean': expected_prior_outcome_A0_mean,
        'expected_prior_outcome_A0_std': expected_prior_outcome_A0_std,
        'expected_posterior_outcome_A1_mean': expected_posterior_outcome_A1_mean,
        'expected_posterior_outcome_A1_std': expected_posterior_outcome_A1_std,
        'expected_posterior_outcome_A0_mean': expected_posterior_outcome_A0_mean,
        'expected_posterior_outcome_A0_std': expected_posterior_outcome_A0_std
    }

    # 10. Compare results

    # Is the physician's decision correct given the true_CATE_mean values?
    prior_decision_correct = True if np.sign(true_CATE_mean - threshold) == np.sign(prior_CATE_mean - threshold) else False
    posterior_decision_correct = True if np.sign(true_CATE_mean - threshold) == np.sign(posterior_CATE_mean - threshold) else False
    # Which decision is best? Assuming that threshold accounts for negative side effects.
    true_best_treatment = A1_treatment if np.sign(true_CATE_mean - threshold) > 0 else A0_treatment
    # Did the decision change (+1 for better, 0 no change, -1 for worse)?
    if np.sign(prior_CATE_mean  - threshold) != np.sign(posterior_CATE_mean - threshold):
        # Set to 1 if changed for the better and -1 if changed for the worse
        decision_changed = 1 if np.sign(posterior_CATE_mean - threshold) == np.sign(true_CATE_mean - threshold) else -1
    else:
        # Set to 0 if no change.
        decision_changed = 0

    # For what fraction of patients is the treatment decision based on prior_CATE_mean,
    # posterior_CATE_mean, and true_CATE_mean correct?
    frac_correct_prior_decision = np.mean(np.sign(true_CATE_samples_ - threshold) == np.sign(prior_CATE_mean - threshold))
    frac_correct_posterior_decision = np.mean(np.sign(true_CATE_samples_ - threshold) == np.sign(posterior_CATE_mean - threshold))
    # Fraction of population that would have benefited from a
    # decision based on the average true CATE (baseline):
    frac_correct_true_best_treatment = np.mean(np.sign(true_CATE_samples_ - threshold) == np.sign(true_CATE_mean - threshold))

    # How much do the patients on average benefit?
    # Average outcome Y under prior decision
    Y_mean_prior_treatment = potential_outcome_A1_mean if np.sign(prior_CATE_mean - threshold) > 0 \
        else potential_outcome_A0_mean
    # Average outcome Y under posterior decision:
    Y_mean_posterior_treatment = potential_outcome_A1_mean if np.sign(posterior_CATE_mean - threshold) > 0 \
        else potential_outcome_A0_mean
    # Average outcome under best decision based on true_CATE_mean
    Y_mean_true_best_treatment = potential_outcome_A1_mean if np.sign(true_CATE_mean - threshold) > 0 \
        else potential_outcome_A0_mean
    # Also save prediction pred_o
    outputs = {'pred_o': pred_o[0],
               'threshold': threshold,
               'prior_treatment': prior_treatment,
               'posterior_treatment': posterior_treatment,
               'true_best_treatment': true_best_treatment,
               'prior_decision_correct': prior_decision_correct,
               'posterior_decision_correct': posterior_decision_correct,
               'decision_changed': decision_changed,
               'frac_correct_prior_decision': frac_correct_prior_decision,
               'frac_correct_posterior_decision': frac_correct_posterior_decision,
               'frac_correct_true_best_treatment': frac_correct_true_best_treatment,
               'Y_mean_prior_treatment': Y_mean_prior_treatment,
               'Y_mean_posterior_treatment': Y_mean_posterior_treatment,
               'Y_mean_true_best_treatment': Y_mean_true_best_treatment,
               'potential_outcome_A1_mean': potential_outcome_A1_mean,
               'potential_outcome_A1_std': potential_outcome_A1_std,
               'potential_outcome_A0_mean': potential_outcome_A0_mean,
               'potential_outcome_A0_std': potential_outcome_A0_std,
               }

    # Collect results
    # Create unique id from timestamp and random number
    current_time = time.strftime("%Y%m%d%H%M%S")
    # Uses its own rng without seed to not have all experiments
    # get identical IDs:
    experiment_id = current_time + str(np.random.randint(1000, 9999))
    compiled_results = {'experiment_id': experiment_id,
                        'true_CATE_mu': true_CATE_mu,
                        'true_CATE_mean': true_CATE_mean,
                        'true_CATE_std': true_CATE_std,
                        'prior_CATE_mu': prior_CATE_mu,
                        'prior_CATE_mean': prior_CATE_mean,
                        'prior_CATE_std': prior_CATE_std,
                        'posterior_CATE_mean': posterior_CATE_mean,
                        'posterior_CATE_std': posterior_CATE_std,
                        'plate_size': plate_size,
                        'X_o': X_o,
                        'A1_treatment': A1_treatment,
                        'A0_treatment': A0_treatment}
    compiled_results.update(outputs)
    compiled_results.update(parameter_descriptions)  # Include parameter descriptions
    end_time = time.time()
    running_time = end_time - start_time
    current_time = time.strftime("%H:%M:%S %d.%m.%Y")
    compiled_results['time_stamp'] = current_time
    compiled_results['running_time'] = running_time
    compiled_results.update(modified_prior)
    compiled_results.update(trace_diagnostics)
    compiled_results.update(prior_predictive_summaries)
    compiled_results.update(posterior_predictive_summaries)
    compiled_results.update(expected_outcomes)

    utils.write_results_to_file(compiled_results, file_name='collapsed_results.csv')

    # Skip saving model in array run.
    # Save model to disk:
    # agent_model.save_agent_model(experiment_id)


if __name__ == '__main__':
    try:
        # Read in values:
        param_name = sys.argv[1]
        param_value = float(sys.argv[2])
    except:
        # Absolutely has to throw an error if input is not specified or not
        # well specified to prevent erroneous interpretation of results.
        raise ValueError("Use script as 'python experiment_collapsed.py N_E_mu -20 72'")
    try:
        # Value for every coordinate of X_o.
        X_o = np.full(n_covariates, float(sys.argv[3]))
    except:
        # Sample X_o if not provided. Don't set seed here.
        X_o = np.random.normal(loc=scm_noise_parameters['alpha_mu_X'],
                               scale=scm_noise_parameters['alpha_sigma_X'],
                               size=n_covariates)
        X_o = np.asarray(X_o)
    if param_name not in agent_priors.keys():
        print("Parameter has to exist in agent_priors. See")
        print(agent_priors)
        text = "{} not a valid parameter!".format(param_name)
        raise ValueError(text)
    if agent_priors[param_name] == param_value:
        # Nothing is changed.
        parameter_descriptions['agent_priors'] = 'Default'
    else:
        # Modify agent parameters
        agent_priors[param_name] = param_value
        # Helper variable for easy view of experiment parameters:
        parameter_descriptions['agent_priors'] = 'Modified'
    # Store info in all cases as some parameter scans will include default values
    modified_prior = {'modified_prior': param_name,
                    'modified_prior_value': param_value}
    # Run main() with set parameters and X_o.
    print("Running collapsed experiment with X_o set to {} and agent prior {} set to {}...".format(
        X_o, param_name, param_value))
    # Initialize rng:
    rng = np.random.default_rng(rng)
    main(X_o, agent_priors, modified_prior, parameter_descriptions=parameter_descriptions, rng=rng)
