"""
Notes:
-Include measure of uncertainty (e.g. variance) for CATE!
"""

import arviz as az
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import agent_model as am
import scm

# Set number of observations used to train prediction model.
# This is also the plate size used later.
plate_size = 1000
scm_noise_parameters = {'alpha_mu_X': 80, 'alpha_sigma_X': 10, 
                        'alpha_mu_A': 10, 'alpha_sigma_A': 1.0,
                        'alpha_mu_Y': 0.0, 'alpha_sigma_Y': 0.1, 'N_E': 20}
# Set scm agent priors to the same
predictive_var_names = [k for k in scm_noise_parameters.keys() if k != 'alpha_sigma_A']
scm_constants = {'a': 12.0, 'b': -0.1, 'd': 0.125}
X_o = 80
draws = 1000  # Number of draws for trace. Warmup is set to 1000.
# Add different options for agent priors we want to test
agent_priors_aligned = {'mu_X': 80, 'sigma_X': 1, 'lambda_X': 10, 'mu_A': 10, 'sigma_A': 1, 'lambda_A': 10,
                        'mu_Y': 0, 'sigma_Y': 1, 'lambda_Y': 1, 'N_E_mu': 20, 'N_E_sigma': 1}
# AGENT_PRIORS_ALIGNED =...
agent_priors = agent_priors_aligned
A1_fixed = 100
A0_fixed = 50


def main(A1_fixed=A1_fixed, A0_fixed=A0_fixed, agent_priors=agent_priors_aligned,
         scm_noise_parameters=scm_noise_parameters, scm_constants=scm_constants,
         plate_size=plate_size, X_o=X_o, draws=1000):
    """
    Args:
    scm_parameters (dict): Dict with keys corresponding to SCM-class.
    plate_size (int): Size of plate / number or observations to generate
     for regression model. Measure of "evidence" for predictions from
     decision support model.
    X_o (float): Value of observed observations. This could correspond
     to e.g. weight of patient.
    """
    # 1. Instantiate SCM to generate the "historical" data. Currently
    #    this is thought of as the SCM representing how data is generated under the historical policy.
    data_scm = scm.SCM(**scm_noise_parameters)

    # 2. Generate data. This is identical to the true
    #    underlying generating process and the outcomes
    #    Y are outcomes under the historical treatment
    #    policy.
    X, A, Y_historical, *_ = data_scm.sample(draws=plate_size)

    # 3. Train prediction model on historical data. 
    pred_model = LinearRegression(fit_intercept=False).fit(X.reshape(-1, 1), Y_historical)

    # 4. The prediction model is now used for decision support. The physician
    #    inputs the data of the patient and the prediction model produces a prediction.
    X_o = np.asarray(X_o)  # The weight of the observed patient (can be other feature).
    pred_o = pred_model.predict(X_o.reshape(-1, 1))  # The observed prediction
    print("The predicted outcome for this patient weighing {}kg is {}.".format(X_o, pred_o[0]))

    # 5. Define agent model. The agent's priors are set at initialization of class
    #    by passing constants and prior_parameters dictionaries.
    # Using same constants for agent's internal SCM as for data generating SCM.
    agent_model = am.AgentModel(prior_parameters=agent_priors,
                                scm_constants=scm_constants,
                                predictive_var_names=predictive_var_names,
                                plate_size=plate_size)

    # 6. The agent's belief is updated / the posterior given the observed data is estimated.
    agent_model.sample(X_o=np.asarray(X_o), pred_o=np.asarray(pred_o[0]), draws=draws)

    # 7. Using updated agent beliefs to come to a decision on what is the best treatment
    posterior_CATE_samples_, posterior_CATE_mean, posterior_CATE_std = \
        agent_model.sample_posterior_predictive_CATE(X_o, A1_fixed, A0_fixed)
    if posterior_CATE_mean > 0:
        selected_treatment = A1_fixed
    else:
        selected_treatment = A0_fixed
    #selected_treatment = agent_model.decide_treatment(X_o, A1_fixed, A0_fixed)

    # 8. Generate patient outcome in the real world given the treatment decision
    #    based on decision support (DS)
    Y_with_DS = data_scm.sample_do_A(selected_treatment, X_o)
    # this concludes the calculation of what the agent does WITH decision support


    # 10. Calculate outcomes for the agent without decision support.
    # Fetch N_E_mu from agent priors? Sampling is more general, although
    # unnecessary here.
    prior_CATE_mu = agent_model.get_prior_CATE_mu(A1_fixed, A0_fixed)
    prior_CATE_samples_, prior_CATE_mean, prior_CATE_std = \
        agent_model.sample_prior_predictive_CATE(X_o, A1_fixed, A0_fixed)

    if prior_CATE_mu > 0:
        # Apply treatment A1:
        # Estimate outcome from SCM:
        # We might even go as far as to rewrite this function to return only the expectation
        # of Y given A_fixed and X_o to not have to deal with variability.
        Y_without_DS = data_scm.sample_do_A(A1_fixed, X_o)
    else:
        Y_without_DS = data_scm.sample_do_A(A0_fixed, X_o)


    # Ground truth (from data generating SCM)
    true_CATE_mu = data_scm.get_CATE_mu(A1_fixed, A0_fixed)
    true_CATE_samples_, true_CATE_mean, true_CATE_std =\
        data_scm.sample_CATE(X_o, A1_fixed, A0_fixed, draws=1000)

    # Save outputs to file:
    # Include mu, mean, and std for prior, posterior, and data generating SCM.
    # Also include timestamp and some identifier which uniquely maps the
    # results to a run.
    #

    # Wrap into function.
    new_row = ['experiment_parameters'] 
    new_row += [true_CATE_mu, true_CATE_mean, true_CATE_std]
    new_row += [prior_CATE_mu, prior_CATE_mean, prior_CATE_std]
    new_row += [None, posterior_CATE_mean, posterior_CATE_std]
    # new_row += ['do_A1_outcome', 'do_A0_outcome']  # Based on data_scm
    text = '"'
    for item in new_row:
        text += '"' + str(item) + '",'
    text += '\n'
    with open('results.csv', 'a') as f:
        f.write(text)

    # Save model to disk:
    agent_model.save_agent_model()
    
    # Testing that loading works.
    # tmp = am.AgentModel.load_agent_model()
    # What is the posterior distribution of the agent's belief about N_E?
    az.plot_posterior(agent_model.trace, var_names=['N_E'])
    plt.title("Posterior distribution of agent's belief about N_E")
    plt.savefig('./fig/basic_example_posterior_NE.png')
    plt.clf()

    # Started with N_E ~ Normal(20,1) prior.
    # -Could compare prior with posterior.
    # -Could also estimate how the prediction for patient with X_o changes with
    # some intervention level A. Model is linear, so higher is better, but
    # we can track the change for some specific level. 
    # -Could track change in uncertainty of outcome for patient.


    if False:
        # 2.1 Plot features against outcomes. Treatment dosage color coded:
        plt.scatter(X, Y, c=A)
        plt.xlabel("Weight (X)")
        plt.ylabel("Outcome (Y)")
        plt.title("Effect of dosage (A) on outcome for different patients (X)")
        plt.colorbar(label="A")
        plt.savefig('./fig/basic_example_historical_data.png')
        plt.clf()
