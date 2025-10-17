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

"""
#import pymc as pm
# from pymc.sampling.jax import sample_numpyro_nuts
import agent_models as am
import scm
import arviz as az
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
import warnings

def test_1(noise=False, test=True):
    """
    Compare full plate model to low-dimensional auxiliary variable model.
    
    If noise=False, the models will be tested with 100 identical X_o and
    and pred_o. This will make the posterior behave mostly like point estimates.
    -This should be useful to see that both the full plate and low-dimensional
    models are implemented correctly and produce the (almost) identical posterior.
    
    If noise=True, some noise is added to these observations. This should produce
    distributions with some spread.

    # Work-in-progress:
    If test=False, the models will be tested with one X_o and pred_o observation.
    -This is more like the real use case.
    If noise=True, the 100 observations will be input.

    """
    # Larger plate size should result in narrower distribution of phi_obs and in
    # a larger effect of the observed data on the posterior.
    # "If we see 'this extreme value' of phi_obs, then our prior beliefs must have 
    # been way off."
    PLATE_SIZE = 5
    test=False
    prior_parameters = {'mu_X': 80, 'sigma_X': 1,
                    'lambda_X': 10,
                    'mu_A': 10, 'sigma_A': 1,
                    'lambda_A': 10,
                    'mu_Y': 0, 'sigma_Y': 1,
                    'lambda_Y': 1,
                    'N_E_mu': 20, 'N_E_sigma': 1}

    agent_model_low_dim = am.AgentModel(plate_size=PLATE_SIZE, test=test, prior_parameters=prior_parameters)
    agent_model_full_plate = am.AgentModel(plate_size=PLATE_SIZE, low_dim_aux_plate=False, test=test, prior_parameters=prior_parameters)

    if test:
        # Input maximum likelihood observation (peak of X-distribution etc.)
        X_o = np.asarray(100 * [80.0])  # A person that would weight 80 kg. This falls in the middle of the
        # prior for X.
        # phi = 5.0059035
        pred_o = np.asarray(100 * [400.47228])  # Prediction you actually saw. This is the value produced
        if noise:
            # Maybe add noise to this for some variation?
            X_o += np.random.normal(0, 10, size=X_o.shape)
            # Maybe scramble the predictions a bit as well?
            pred_o += np.random.normal(0, 10, size=pred_o.shape)
    else:
        # Use only one observation:
        phi = 5.0059035
        X_o = np.asarray(200.0)  #80.0)  # A person that would weight 80 kg. This falls in the middle of the
        pred_o = np.asarray(X_o * phi)
    # by the linear model when 10000 obsevations was used to fit it.
    print("Sampling low-dimensional auxiliary variable model... This can take a while.")
    trace_low_dim = agent_model_low_dim.sample(X_o=np.asarray(X_o), pred_o=np.asarray(pred_o), draws=1000)  # not element-wise yet?
    print("Sampling full plate model... This might take 30 minutes.")
    trace_full_plate = agent_model_full_plate.sample(X_o=np.asarray(X_o), pred_o=np.asarray(pred_o), draws=1000)

    summary_low_dim_plate_100 = az.summary(trace_low_dim, hdi_prob=0.95)
    summary_full_plate_plate_100 = az.summary(trace_full_plate, hdi_prob=0.95)
    print(summary_low_dim_plate_100)
    print(summary_full_plate_plate_100)

    # az.plot_trace(trace_low_dim, var_names="alpha_mu_Y")
    # az.plot_trace(trace_full_plate, var_names="alpha_mu_Y")
    # az.plot_trace(trace_low_dim, var_names="alpha_mu_X")
    # az.plot_trace(trace_full_plate, var_names="alpha_mu_X")
    # az.plot_trace(trace_low_dim, var_names="alpha_mu_A")
    # az.plot_trace(trace_full_plate, var_names="alpha_mu_A")
    # az.plot_trace(trace_low_dim, var_names="N_E")
    # az.plot_trace(trace_full_plate, var_names="N_E")

    az.plot_trace(trace_low_dim)
    plt.show()
    az.plot_energy(trace_low_dim)
    plt.show()
    az.plot_autocorr(trace_low_dim)
    plt.show()
    az.plot_rank(trace_low_dim)
    plt.show()


def test_2(a, b):
    """
    Test for parametric similarity of full plate and low-dim model.
    """
    for var in ['alpha_mu_X', 'alpha_mu_Y', 'alpha_mu_A']:
        compare(a, b, var)

def compare(a, b, var_name='alpha_mu_X'):
    """
    This function estimates from two sets of points whether
    one on average takes on higher values than the other.
    It is basically a Bayesian test for similarity.

    This could alternatively be formulated as p of 'a' being within
    epsilon-distance from 'b', i.e. p(|a-b| < epsilon)

    Args:
    a (trace, arviz.data.inference_data.InferenceData): Object that can be 
        accessed with a.posterior.[variable] for the posterior samples
    b (trace): Same as above.
    var_name (str): Name of the variable to compare on.
    """
    # Equality would be a *perfect* match.
    # Counting pairwise differences. Comparing twice 
    # If 90% of all pairs match, then both N_a_larger and N_a_smaller will be way
    # below 0.5. Add half of the equal N to both for centering?

    # Chains merged
    N_equal = np.sum(np.equal(a.posterior[var_name].values, b.posterior[var_name].values))
    N_b_larger = np.sum(np.greater(b.posterior[var_name].values, a.posterior[var_name].values))
    N_b_smaller = np.sum(np.less(b.posterior[var_name].values, a.posterior[var_name].values))
    N_obs = a.posterior[var_name].size
    # Sanity check
    text = "Error: Number of samples does not add up to total number of samples for {}.".format(var_name)
    assert N_b_larger + N_b_smaller + N_equal == N_obs, text

    # Bayesian pairwise rank test:
    p_b_larger = (N_b_larger + .5 * N_equal) / N_obs
    p_b_smaller = (N_b_smaller + .5 * N_equal) / N_obs
    print("Bayesian pairwise rank test for {} in traces.".format(var_name))
    print("p(b > a) = {:.2%}".format(p_b_larger))
    print("p(b < a) = {:.2%}".format(p_b_smaller))
    # Does the observed value fall within expected bounds?
    if N_b_larger + .5 * N_equal > 0.5 * N_obs:
        # Equally scoring observations split equally between upper and lower bound
        upper_bound = N_b_larger + N_equal / 2
        lower_bound = N_b_smaller + N_equal / 2
    else:
        upper_bound = N_b_smaller + N_equal / 2
        lower_bound = N_b_larger + N_equal / 2
    # Comparing results to binomial distribution
    cdf_upper = binom.cdf(upper_bound, N_obs, 0.5)
    cdf_lower = binom.cdf(lower_bound, N_obs, 0.5)
    two_sided_mass = cdf_upper - cdf_lower
    print("Pairwise rank difference within {:.2%} of expected if both traces come from the same distribution.".format(two_sided_mass))
    if two_sided_mass > 0.95:
        text = "The posterior distributions of alpha_X_mu differn more than 5%. \n" +\
            "This might indicate a bug in model specification. Rerun test, perhaps fix."
        raise Warning(text)
    
    # Z-test for means (assuming the trace of a is from the correct distribution)
    a_mean = np.mean(a.posterior[var_name])
    a_std = np.std(a.posterior[var_name])
    b_mean = np.mean(b.posterior[var_name])
    b_std = np.std(b.posterior[var_name])
    print("Mean value and standard deviation for trace-a: {:.2}, {:.2}".format(a_mean, a_std))
    print("Mean value and standard deviation for trace-c: {:.2}, {:.2}".format(b_mean, b_std))
    print("Z-test comparing b-trace to a-trace.")
    Z_score = np.abs(b_mean - a_mean) / a_std
    print("Z-score for b-trace: {:.2}".format(Z_score))
    if Z_score > .05:
        # What is the appropriate Z-score for raising an error?
        text = "Z-score too high to assume that the distributions of {} in the two traces are from the same distribution."
        raise Warning(text)  # Maybe raise error instead of warning.
    
    # Maybe add KL-divergence?
    return


if __name__ == '__main__':
    # Test similarity between full plate and low-dim models
    PLATE_SIZE = 5
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

    # Use 100 observed variables centered around alpha_X_mu with prediction that corresponds 
    # to model parameters generated from SCM plus some noise
    X_o = np.asarray(100 * [80.0])  # A person that would weight 80 kg.
    X_o += np.random.normal(0, 10, size=X_o.shape)
    phi = 5.0059035
    pred_o = np.asarray(100 * [80.0 * phi])  # Observed prediction
    pred_o += np.random.normal(0, 10, size=pred_o.shape)

    print("Sampling low-dimensional auxiliary variable model... This can take a while.")
    trace_low_dim = agent_model_low_dim.sample(X_o=np.asarray(X_o), pred_o=np.asarray(pred_o), draws=1000)  # not element-wise yet?
    print("Sampling full plate model... This might take 30 minutes.")
    trace_full_plate = agent_model_full_plate.sample(X_o=np.asarray(X_o), pred_o=np.asarray(pred_o), draws=1000)

    test_2(trace_full_plate, trace_low_dim)

    az.plot_trace(trace_low_dim, var_names="alpha_mu_Y")
    plt.savefig('./tests/fig/trace_low_dim_model.pdf')
    plt.clf()
    az.plot_energy(trace_low_dim)
    plt.savefig('./tests/fig/energy_low_dim_model.pdf')
    plt.clf()
    az.plot_autocorr(trace_low_dim)
    plt.savefig('./tests/fig/autocorrelation_low_dim_model.pdf')
    plt.clf()
    az.plot_rank(trace_low_dim)
    plt.savefig('./tests/fig/rank_low_dim.pdf')
    plt.clf()

    az.plot_trace(trace_full_plate, var_names="alpha_mu_Y")
    plt.savefig('./tests/fig/model_full_plate_model.pdf')
    plt.clf()
    az.plot_energy(trace_full_plate)
    plt.savefig('./tests/fig/energy_full_plate_model.pdf')
    plt.clf()
    az.plot_autocorr(trace_full_plate)
    plt.savefig('./tests/fig/autocorrelation_full_plate_model.pdf')
    plt.clf()
    az.plot_rank(trace_full_plate)
    plt.savefig('./tests/fig/rank_full_plate.pdf')
    plt.clf()


    # Prior predictive checks
    # Posterior checks
    # Posterior predictive checks
