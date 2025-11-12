"""
Tests to ensure that the agent model is working as intended.

Comparisons between full plate and low-dimensional auxiliary variable models
-The model with the full plate can be compared to the model with
low-dimensional auxiliary variables.
-As the sampling is done differently for the two approaches, setting the
random seed does not help.

Simulation-based calibration
-The expectation would be for parameters to fall within some set intervals
of their true values a certain percentage of the time.
-Can that be done here? Our data generating model is not the same as
the model we are fitting.
-If we implement a testing model for the model, we could potentially
feed thousands of identical X_o and pred_o observations to get the
distributions to converge to one value.
-A model with tiny plate should be more highly affected by the data.

Posterior predictive checks
-Generate data from the posterior and see if it looks like the observed data.
-Can it be done for our model? Does the difference between SCM and the linear
model used by the agent cause problems?

Sensitivity analysis of priors
-If we change the priors, how much does it affect the posterior and is the
change in the right direction?

Consistency checks
-If we have a tiny plate and one extreme observation, the posterior should shift only a little.
-If we have a large plate and one extreme observation, the posterior should shift more.

Sampler diagnostics

Ad hoc tests
-test whether the posterior means of the full model and the low-dim one are "sufficiently"
 close to represent the same posterior.
-Different plate size. What are we expecting?
-Break dependencies in SCM e.g. by setting a=0 to simplify. Then X_o and pred_o should
 induce a change in the distribution of Y.

 
KL-divergene

Likelihood
Compare likelihood of low-dim and full plate models given the same data

Sampling metrics
"""
import pymc as pm
import agent_models as am
import numpy as np
import tests.statistical_tests_for_similarity as sts
import tests.visual_tests as vt
import warnings

# The similarity checks are based on the variables
# defined here.
PRIOR_PREDICTIVE_VAR_NAMES = ['phi']
POSTERIOR_PREDICTIVE_VAR_NAMES = ['phi_obs', 'X_o']
POSTERIOR_VAR_NAMES = ['alpha_mu_X', 'N_E', 'alpha_sigma_X']
SAMPLE_SIZE = 10000


def check_similarity(observations_A, observations_B, var_name):
    """
    Wrapper for all statistical tests in sts.
    None of the tests check for dependences between variables.
    """
    # Test for rank similarity
    tmp1 = sts.bayesian_test_for_rank_similarity(observations_A, observations_B)
    # Test for similarity of means:
    tmp2 = sts.bayesian_test_for_similarity_of_means(observations_A, observations_B)
    # Test for similarity of mean, variance, and skewness (bootstrap)
    tmp3 = sts.bootstrap_test_for_similarity(observations_B, observations_A)
    # Normalized Wasserstein distance
    tmp4 = sts.normalized_wasserstein_distance(observations_A, observations_B)

    if tmp1['success'] and tmp2['success'] and tmp3['success'] and tmp4['success']:
        print("=== Numerical tests for similarity of {} passed. ===".format(var_name))
        success = True
    else:
        text = "Similarity was not confirmed."
        warnings.warn(text)
        success = False
    return {'success': success,
            'rank': tmp1,
            'mean': tmp2,
            'bootstrap': tmp3,
            'wasserstein': tmp4}


def prior_predictive_check_for_similarity(model_A, model_B, 
                                          var_names=PRIOR_PREDICTIVE_VAR_NAMES,
                                          sample_size=SAMPLE_SIZE):
    """
    Prior predictive check for similarity between models.
    The function generates a number of observations of the variable
    defined from both models and compares the observations.

    Args:
    model_A: Can be e.g. a low_dim_model. 
    full_plate_model:
    var_names ([str, ]): List with names of variables to run similarity
     check on.
    sample_size (int): Size of sample (number of observations) to
     generate from both models for comparison.
    """
    # Generate observations from both models
    with model_A.model:
        model_A_prior_predictive = pm.sample_prior_predictive(samples=sample_size)
    with model_B.model:
        model_B_prior_predictive = pm.sample_prior_predictive(samples=sample_size)
    # Extract numpy arrays of phi-observations:
    success = []
    visualization = vt.VisualTests(var_names)
    for var_name in var_names:
        observations_model_A = model_A_prior_predictive.prior[var_name].values.flatten()
        observations_model_B = model_B_prior_predictive.prior[var_name].values.flatten()
        success.append(check_similarity(observations_model_A, observations_model_B, var_name))
        visualization.plot(observations_model_A, observations_model_B, var_name)
    visualization.show()

    return success


def posterior_check_for_similarity(trace_A, trace_B, 
                                   var_names=POSTERIOR_VAR_NAMES):
    """
    Args:
    trace_1: trace of model 1
    trace_2: trace of model 2
    var_names ([str, ]): List of names of variables to perform check
     over. Defaults to list specified at beginning of file.
    """
    print("=" * 40)
    print("POSTERIOR CHECKS FOR SIMILARITY")
    success = []
    visualization = vt.VisualTests(var_names)
    for var_name in var_names:
        print("-" * 40)
        print("Checking similarity of {}...".format(var_name))
        observations_A = trace_A.posterior[var_name].values.flatten()
        observations_B = trace_B.posterior[var_name].values.flatten()
        success.append(check_similarity(observations_A, observations_B, var_name))
        visualization.plot(observations_A, observations_B, var_name)
    visualization.show()

    return success


def posterior_predictive_check_for_similarity(model_A, model_B,
                                              trace_A, trace_B,
                                              var_names=POSTERIOR_PREDICTIVE_VAR_NAMES):
    """
    Args:
    model_A (AgentModel): Model that generated trace_A.
    model_B (AgentModel): Model that generated trace_B.
    trace_A: trace of model A
    trace_B: trace of model B
    var_names ([str, ]): List of variable names for variables
     to perform similarity check on.
    """
    with model_A.model:
        # Model is needed to map the posterior parameters to a
        # DAG, and trace contains the sampled posterior parameters.
        posterior_samples_A = pm.sample_posterior_predictive(trace_A)
    with model_B.model:
        posterior_samples_B = pm.sample_posterior_predictive(trace_B)

    success = []
    visualization = vt.VisualTests(var_names)
    # Extract numpy arrays of phi-observations:
    print("POSTERIOR PREDICTEIVE CHECKS FOR SIMILARITY")
    for var_name in var_names:
        # Extract one variable at a time and perform similarity check
        print("-" * 40)
        print("Checking similarity of {}...".format(var_name))
        observations_A = posterior_samples_A.posterior_predictive[var_name].values.flatten()
        observations_B = posterior_samples_B.posterior_predictive[var_name].values.flatten()
        success.append(check_similarity(observations_A, observations_B, var_name))
        visualization.plot(observations_A, observations_B, var_name)
    visualization.show()

    return success



if __name__ == '__main__':
    """
    Also check:
    -different plate sizes.
    -different prior parameters.
    -different observations (not perfectly centered!)
    -updating with just one observation (instead of 100)

    Can we figure out automatically which variables exist in both models
    and which ones are predictive?
    -Could run tests automatically on those.
    -Does it complicate "modifications" too much? Maybe.
    """
    # Test similarity between full plate and low-dim models
    PLATE_SIZE = 5
    sample_size = 10000  # What should this number be?
    test=False
    prior_parameters = {'mu_X': 80, 'sigma_X': 1,
                    'lambda_X': 10,
                    'mu_A': 10, 'sigma_A': 1,
                    'lambda_A': 10,
                    'mu_Y': 0, 'sigma_Y': 1,
                    'lambda_Y': 1,
                    'N_E_mu': 20, 'N_E_sigma': 1}

    agent_model_low_dim = am.AgentModel(plate_size=PLATE_SIZE, test=True, prior_parameters=prior_parameters)
    agent_model_full_plate = am.AgentModel(plate_size=PLATE_SIZE, low_dim_aux_plate=False, test=True, prior_parameters=prior_parameters)

    # Prior predictive check
    import tests.test_agent_models as tam
    tmp1 = tam.prior_predictive_check_for_similarity(agent_model_low_dim,
                                                agent_model_full_plate)
    # Posterior checks
    # Use 100 observed variables centered around alpha_X_mu with prediction that corresponds 
    # to model parameters generated from SCM plus some noise
    X_o = np.asarray(100 * [80.0])  # A person that would weight 80 kg.
    X_o += np.random.normal(0, 10, size=X_o.shape)
    phi = 5.0059035
    pred_o = np.asarray(100 * [80.0 * phi])  # Observed prediction
    pred_o += np.random.normal(0, 10, size=pred_o.shape)

    print("Sampling low-dimensional auxiliary variable model... This can take a while.")
    trace_low_dim = agent_model_low_dim.sample(X_o=np.asarray(X_o), pred_o=np.asarray(pred_o), draws=1000)
    print("Sampling full plate model... This might take 30 minutes.")
    trace_full_plate = agent_model_full_plate.sample(X_o=np.asarray(X_o), pred_o=np.asarray(pred_o), draws=1000)

    tmp2 = posterior_check_for_similarity(trace_low_dim, trace_full_plate)

    # Posterior predictive check
    tmp3 = posterior_predictive_check_for_similarity(agent_model_low_dim,
                                                         agent_model_full_plate,
                                                         trace_low_dim,
                                                         trace_full_plate)
