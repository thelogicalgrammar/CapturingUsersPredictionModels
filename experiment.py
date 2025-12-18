"""
This module provides functionality to run experiments that test agent decision-making
with and without decision support systems.

The script takes three command line parameters where the first one
is the key in agent_priors to modify, the second one what value 
agent_priors[key] should take, and the third one what value X_o
should take.

Example:

.. code-block:: bash

    python experiment.py N_E_mu -20 72

This would set the value of the prior distribution of N_E to -20 and
use 72kg as the feature of the observed patient.


Notes:
Could perhaps implement support for changing plate size and SCM parameters
"""
import sys
import time
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
# Project specific:
import scm
import utils
import agent_model as am



# Set number of observations used to train prediction model.
# This is also the plate size used in the agent's belief model.
plate_size = 1000
scm_noise_parameters = {'alpha_mu_X': 80, 'alpha_sigma_X': 10,
                        'alpha_mu_A': 2, 'alpha_sigma_A': 1.0,
                        'alpha_mu_Y': 0.0, 'alpha_sigma_Y': 0.1,
                        'N_E': 1.0}

# Set scm agent priors to the same
predictive_var_names = [k for k in scm_noise_parameters.keys() if k != 'alpha_sigma_A']
scm_constants = {'a': 12.0, 'b': -0.1, 'd': 0.125}
draws = 1000  # Number of draws for trace. Warmup is set to 1000.
# Default values of agent priors. One value is overwritten in main().
agent_priors =\
    {'mu_X': 80, 'sigma_X': 0.1, 'mu_sigma_X': 10, 'sigma_sigma_X':0.1,
     'mu_A': 2, 'sigma_A': 1, 'mu_sigma_A': 1, 'sigma_sigma_A':0.1,
     'mu_Y': 0, 'sigma_Y': 0.01, 'mu_sigma_Y': 0.1, 'sigma_sigma_Y' : 0.01,
     'N_E_mu': 1.0, 'N_E_sigma': 1}
# changed agent prior sigma_Y from 1 to 0.1, this is a choice
# changed prior of lambda_X from 10 to 0.1, this was a mistake
# Treatment info
A1_treatment = 20
A0_treatment = 10
# Only if treatment A1 is expected to produce an outcome that is threshold=2 above
# the expected outcome with A0 is A1 used. This corresponds to e.g. A1 being more
# expensive or to A1 having larger side effects.
threshold = 5
parameter_descriptions = {'agent_priors': 'Default',
                          'scm_constants': 'Default',
                          'scm_noise_parameters': 'Default'}
rng = 4  # Selected by fair die

def main(X_o, agent_priors, modified_prior,
         A1_treatment=A1_treatment, A0_treatment=A0_treatment,
         plate_size=plate_size, draws=1000,
         scm_noise_parameters=scm_noise_parameters,
         scm_constants=scm_constants,
         parameter_descriptions=parameter_descriptions,
         rng=rng):
    """
    Args:
    X_o (np.array): Feature value of the observed patient (e.g. in kg).
    agent_priors (dict): Dictionary with all values that encodes the
    agent's prior beliefs.
    modified_prior (dict): Information about which prior was changed
    and how much.
    A1_treatment (float): Value of treatment 1.
    A0_treatment (float): Value of treatment 0.
    plate_size (int): Size of plate / number or observations to generate
    for regression model. Measure of "evidence" for predictions from
    decision support model.
    draws (int): Size of sample to generate in sampling events.
    scm_noise_parameters (dict): Dict with keys needed in SCM-class.
    scm_constants (dict): Constants defining the relationships in the SCM.
    rng (int or rng): Seed or rng for reproducibility.
    """
    start_time = time.time()
    # 1. SCM-instance to generate population data under the
    #    historical treatment policy.
    #    X is sampled from the population distribution, A
    #    is the treatment given under the historical treatment
    #    policy given X, and Y is the outcome given X and A.
    data_scm = scm.SCM(**scm_noise_parameters)

    # 2. Generate data. This is identical to the true
    #    underlying generating process and the outcomes
    #    Y are outcomes under the historical treatment
    #    policy.
    X, A, Y_historical, *_ = data_scm.sample(draws=plate_size, rng=rng)

    # 3. Train prediction model on historical data. 
    #    Calculated as the Moore-Penrose pseudo-inverse. No randomness.
    pred_model = LinearRegression(fit_intercept=False).fit(X.reshape(-1, 1), Y_historical)

    # 4. The prediction model is next used for decision support. The physician
    #    inputs the data of the patient and the prediction model produces a prediction.

    pred_o = pred_model.predict(X_o.reshape(-1, 1))  # The observed prediction

    # print("The predicted outcome for this patient weighing {}kg is {}.".format(X_o, pred_o[0]))

    # 5. Define the agent model.
    #    Using same _constants_ for agent's internal SCM as for data generating SCM.
    agent_model = am.AgentModel(prior_parameters=agent_priors,
                                scm_constants=scm_constants,
                                predictive_var_names=predictive_var_names,
                                plate_size=plate_size)

    # 6. Predictive estimates and treatment decision under prior beliefs
    prior_CATE_mu = agent_model.get_prior_CATE_mu(A1_treatment, A0_treatment)
    prior_CATE_samples_, prior_CATE_mean, prior_CATE_std = \
        agent_model.sample_prior_predictive_CATE(X_o, A1_treatment, A0_treatment)
    prior_treatment = A1_treatment if prior_CATE_mean > threshold else A0_treatment
    prior_predictive_summaries = agent_model.sample_prior_predictive_summaries(rng=rng)

    # 7. Predictive estimates and treatment decision under posterior beliefs
    # The agent's beliefs are updated / the posterior given the observed data is sampled.
    agent_model.sample(X_o=np.asarray(X_o), pred_o=np.asarray(pred_o[0]), draws=draws, rng=rng)
    # Model diagnostics
    rhat_values = az.rhat(agent_model.trace)
    ess_bulk_values = az.ess(agent_model.trace, method='bulk')
    ess_tail_values = az.ess(agent_model.trace, method='tail')
    trace_diagnostics = {
        'max_rhat': float(rhat_values.to_array().max().values),
        'min_ess_bulk': float(ess_bulk_values.to_array().min().values),
        'min_ess_tail': float(ess_tail_values.to_array().min().values),
        'n_divergences': int(agent_model.trace.sample_stats['diverging'].sum())
        }

    # Using updated agent beliefs to come to a decision on what is the best treatment
    posterior_CATE_samples_, posterior_CATE_mean, posterior_CATE_std =\
        agent_model.sample_posterior_predictive_CATE(X_o, A1_treatment, A0_treatment, rng=rng)
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
    # Estimate mean?
    potential_outcome_A1_samples_, potential_outcome_A1_mean, potential_outcome_A1_std =\
        data_scm.sample_do_A(A1_treatment, X_o, draws=draws, rng=rng)
    potential_outcome_A0_samples_, potential_outcome_A0_mean, potential_outcome_A0_std =\
        data_scm.sample_do_A(A0_treatment, X_o, draws=draws, rng=rng)

    # 9. Compare results

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

    # How much do the patients on average benefit? # TODO: change this to account for utility threshold
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
    # Wrap into function.
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

    # Add prior predictive summaries
    compiled_results.update(prior_predictive_summaries)
    # Add posterior predictive summaries
    compiled_results.update(posterior_predictive_summaries)

    # Add something general about model convergence and quality metrics.

    utils.write_results_to_file(compiled_results)

    # Save model to disk:
    agent_model.save_agent_model(experiment_id)
    
    # Testing that loading works.
    # tmp = am.AgentModel.load_agent_model()
    # What is the posterior distribution of the agent's belief about N_E?
    # az.plot_posterior(agent_model.trace, var_names=['N_E'])
    # plt.title("Posterior distribution of agent's belief about N_E")
    # plt.savefig('./fig/basic_example_posterior_NE.png')
    # added plot to save the posterior distribution of pairwise correlations
    az.plot_pair(agent_model.trace, #var_names=['N_E', 'alpha_mu_A', 'alpha_mu_Y'],
                 kind='scatter')
    plt.title("Posterior distribution of correlation between mu_A and N_E")
    plt.savefig('./fig/basic_example_posterior_corr_all_vars.png')
    plt.clf()


if __name__ == '__main__':
    try:
        # Read in values:
        param_name = sys.argv[1]
        param_value = float(sys.argv[2])
    except:
        # Absolutely has to throw an error if input is not specified or not
        # well specified to prevent erroneous interpretation of results.
        raise ValueError("Use script as 'python experiment.py N_A_mu 23 72'")
    try:
        # Value for X_o
        X_o = np.array(float(sys.argv[3]))
    except:
        # Sample X_o if not provided.
        # Don't set seed here.
        X_o = np.random.normal(loc=scm_noise_parameters['alpha_mu_X'],
                               scale=scm_noise_parameters['alpha_sigma_X'])
        X_o = np.array(X_o)
    if param_name not in agent_priors.keys():
        print("Parameter has to exist in agent_priors. See")
        print(agent_priors)
        text = "{} not a valid parameter!".format(param_name)
        raise ValueError(text)
    agent_priors[param_name] = param_value
    modified_prior = {'modified_prior': param_name,
                      'modified_prior_value': param_value}
    # Helper variable for easy view of experiment parameters:
    parameter_descriptions['agent_priors'] = 'Modified'
    # Modify agent parameters
    # Run main() with set parameters and X_o.
    print("Running experiment with X_o set to {} and agent prior {} set to {}...".format(
        X_o, param_name, param_value))
    # Initialize rng:
    rng = np.random.default_rng(rng)
    main(X_o, agent_priors, modified_prior, parameter_descriptions=parameter_descriptions, rng=rng)
