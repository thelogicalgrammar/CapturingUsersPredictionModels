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

"""
import pymc as pm
# from pymc.sampling.jax import sample_numpyro_nuts
import agent_models as am
import historical_scm
import arviz as az
import numpy as np
import matplotlib.pyplot as plt



def test_1():
    """
    Compare full plate model to low-dimensional auxiliary variable model.
    How? Visually? By comparing posterior distribution statistics?
    """
    # Larger plate size should result in narrower distribution of phi_obs and in
    # a larger effect of the observed data on the posterior.
    # "If we see 'this extreme value' of phi_obs, then our prior beliefs must have 
    # been way off."
    PLATE_SIZE = 100
    agent_model_low_dim = am.AgentModel(plate_size=PLATE_SIZE, test=True)
    agent_model_full_plate = am.AgentModel(plate_size=PLATE_SIZE, low_dim_aux_plate=False, test=True)

    # Input maximum likelihood observation (peak of X-distribution etc.)
    X_o = np.asarray(100 * [80.0])  # A person that would weight 80 kg. This falls in the middle of the
    # prior for X.
    # phi = 5.0059035
    pred_o = np.asarray(100 * [400.47228])  # Prediction you actually saw. This is the value produced
    # by the linear model when 10000 obsevations was used to fit it.
    print("Sampling low-dimensional auxiliary variable model... This can take a while.")
    trace_low_dim = agent_model_low_dim.sample(X_o=np.asarray(X_o), pred_o=np.asarray(pred_o), draws=1000)  # not element-wise yet?
    print("Sampling full plate model... This might take 30 minutes.")
    trace_full_plate = agent_model_full_plate.sample(X_o=np.asarray(X_o), pred_o=np.asarray(pred_o), draws=1000)

    summary_low_dim_plate_100 = az.summary(trace_low_dim, hdi_prob=0.95)
    summary_full_plate_plate_100 = az.summary(trace_full_plate, hdi_prob=0.95)
    print(summary_low_dim_plate_100)
    print(summary_full_plate_plate_100)


    az.plot_trace(trace_low_dim, var_names="alpha_mu_Y")
    az.plot_trace(trace_full_plate, var_names="alpha_mu_Y")
    plt.show()
    """
    Output with plate_size = 7.
    Agent should be more affected by data than with a larger plate.

    # Output of running the code
    # alpha_mu_X in both cases are identical as are alpha_sigma_X.
    # alpha_mu_A, alpha_mu_Y, and alpha sigma_Y are further apart.
    # N_E is very close in both cases.
    # Is this difference "large"?
    >>> print(summary_full_plate)
                    mean      sd  hdi_2.5%  hdi_97.5%  mcse_mean  mcse_sd  ess_bulk  ess_tail  r_hat
    alpha_mu_X      80.000   0.000    80.000     80.000      0.000    0.000      23.0      95.0   1.30
    alpha_mu_A       9.431   0.580     8.814     10.067      0.289    0.025       4.0       4.0   4.42
    alpha_mu_Y       1.025   1.116    -0.357      2.726      0.556    0.270       4.0      12.0   4.60
    N_E             20.170   0.578    19.257     20.830      0.288    0.144       4.0      12.0   5.22
    X[0]            80.000   0.000    80.000     80.000      0.000    0.000      11.0      88.0   1.72
    X[1]            80.000   0.000    80.000     80.000      0.000    0.000      32.0      71.0   1.18
    X[2]            80.000   0.000    80.000     80.000      0.000    0.000      82.0      67.0   1.72
    X[3]            80.000   0.000    80.000     80.000      0.000    0.000      25.0      82.0   1.64
    X[4]            80.000   0.000    80.000     80.000      0.000    0.000      17.0      81.0   1.60
    X[5]            80.000   0.000    80.000     80.000      0.000    0.000      17.0      66.0   1.17
    X[6]            80.000   0.000    80.000     80.000      0.000    0.000      11.0      81.0   2.22
    A[0]            19.545   1.128    18.140     21.273      0.562    0.278       4.0       4.0   3.81
    A[1]            19.018   0.317    18.826     19.567      0.158    0.091       4.0      13.0   5.54
    A[2]            19.683   0.560    18.770     20.225      0.279    0.141       4.0       4.0   7.39
    A[3]            19.247   0.733    18.081     20.085      0.365    0.186       4.0       4.0  10.10
    A[4]            20.177   1.169    18.677     21.936      0.582    0.285       4.0      13.0   3.84
    A[5]            20.618   1.723    19.273     23.547      0.858    0.476       4.0       4.0   3.87
    A[6]            19.110   0.837    18.098     20.376      0.417    0.193       4.0       4.0   5.35
    Y[0]           397.490  15.349   371.938    411.755      7.644    4.075       4.0       4.0   9.06
    Y[1]           388.926   7.597   379.835    399.315      3.784    1.273       4.0      26.0   7.57
    Y[2]           401.014  12.205   382.448    412.883      6.078    2.498       4.0      11.0   4.89
    Y[3]           393.178  15.354   375.286    415.515      7.646    3.025       4.0      25.0   6.53
    Y[4]           412.481  17.876   381.896    427.157      8.902    5.030       4.0      11.0   4.60
    Y[5]           420.992  21.188   402.159    456.961     10.552    5.890       4.0       4.0   3.86
    Y[6]           389.213  16.834   369.824    416.107      8.383    4.340       4.0      11.0   3.72
    alpha_sigma_X    0.000   0.000     0.000      0.000      0.000    0.000       4.0       4.0   5.40
    alpha_sigma_Y    1.120   0.795     0.054      1.980      0.396    0.109       4.0       4.0   4.09
    phi              5.006   0.000     5.006      5.006      0.000    0.000       4.0       4.0   3.82
    """


"""
Output with plate size 100:

>>> print(summary_full_plate_plate_100)
                  mean      sd  hdi_2.5%  hdi_97.5%  mcse_mean  mcse_sd  ess_bulk  ess_tail        r_hat
alpha_mu_X      80.000   0.000    80.000     80.000      0.000    0.000       4.0       4.0  31633592.87
alpha_mu_A      10.546   0.646     9.965     11.532      0.321    0.133       4.0       4.0  31633592.87
alpha_mu_Y      -0.480   0.344    -0.840      0.068      0.172    0.083       5.0      25.0         3.83
N_E             19.359   0.590    18.466     19.936      0.294    0.119       4.0       4.0  31633592.87
X[0]            80.000   0.000    80.000     80.000      0.000    0.000       4.0       4.0  31633592.87
...                ...     ...       ...        ...        ...      ...       ...       ...          ...
Y[98]          396.885   8.324   386.712    407.999      4.145    1.344       4.0       4.0  31633592.87
Y[99]          385.661  12.210   369.459    399.120      6.080    1.632       4.0       4.0  31633592.87
alpha_sigma_X    0.000   0.000     0.000      0.000      0.000    0.000       4.0       4.0  31633592.87
alpha_sigma_Y    0.865   0.416     0.394      1.471      0.207    0.080       4.0       6.0         3.75
phi              5.006   0.000     5.006      5.006      0.000    0.000       4.0       4.0  31633592.87
"""