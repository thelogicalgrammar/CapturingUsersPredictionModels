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
#import pymc as pm
# from pymc.sampling.jax import sample_numpyro_nuts
import agent_models as am
#import historical_scm
import arviz as az
import numpy as np
import matplotlib.pyplot as plt



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
    PLATE_SIZE = 100
    agent_model_low_dim = am.AgentModel(plate_size=PLATE_SIZE, test=test)
    agent_model_full_plate = am.AgentModel(plate_size=PLATE_SIZE, low_dim_aux_plate=False, test=test)

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
        X_o = np.asarray(80.0)  # A person that would weight 80 kg. This falls in the middle of the
        pred_o = np.asarray(400.47228)
    # by the linear model when 10000 obsevations was used to fit it.
    print("Sampling low-dimensional auxiliary variable model... This can take a while.")
    trace_low_dim = agent_model_low_dim.sample(X_o=np.asarray(X_o), pred_o=np.asarray(pred_o), draws=1000)  # not element-wise yet?
    print("Sampling full plate model... This might take 30 minutes.")
    trace_full_plate = agent_model_full_plate.sample(X_o=np.asarray(X_o), pred_o=np.asarray(pred_o), draws=1000)

    summary_low_dim_plate_100 = az.summary(trace_low_dim, hdi_prob=0.95)
    summary_full_plate_plate_100 = az.summary(trace_full_plate, hdi_prob=0.95)
    print(summary_low_dim_plate_100)
    print(summary_full_plate_plate_100)

    # We currently have a distribution only for mu_X. All others are one point value for every chain (I think).
    az.plot_trace(trace_low_dim, var_names="alpha_mu_Y")
    az.plot_trace(trace_full_plate, var_names="alpha_mu_Y")
    az.plot_trace(trace_low_dim, var_names="alpha_mu_X")
    az.plot_trace(trace_full_plate, var_names="alpha_mu_X")
    az.plot_trace(trace_low_dim, var_names="alpha_mu_A")
    az.plot_trace(trace_full_plate, var_names="alpha_mu_A")
    az.plot_trace(trace_low_dim, var_names="N_E")
    az.plot_trace(trace_full_plate, var_names="N_E")
    plt.show()

    """
    Output of test_1(noise=False, test=True) with seed=42:

>>> print(summary_low_dim_plate_100)
                      mean         sd     hdi_2.5%    hdi_97.5%  mcse_mean   mcse_sd  ess_bulk  ess_tail  r_hat
alpha_mu_X          80.000      0.000       80.000       80.000      0.000     0.000    4117.0    2704.0   1.00
alpha_mu_A          10.090      0.493        9.296       10.539      0.246     0.119       4.0      12.0   3.38
alpha_mu_Y          -0.431      0.386       -1.085       -0.099      0.192     0.106       4.0      24.0   3.32
N_E                 19.770      0.500       19.310       20.567      0.249     0.117       4.0      11.0   3.42
epsilon_SX          -0.305      0.800       -0.927        1.052      0.398     0.219       4.0      14.0   3.47
epsilon_SY          -0.403      0.398       -0.895        0.130      0.198     0.067       4.0      14.0   3.47
epsilon_SA          -0.100      0.204       -0.347        0.157      0.102     0.027       4.0      11.0   4.17
alpha_sigma_X        0.000      0.000        0.000        0.000      0.000     0.000       5.0      14.0   3.10
alpha_sigma_Y        0.868      0.133        0.700        1.044      0.066     0.023       4.0      11.0   4.17
gamma_XX            92.847      4.994       86.583      100.546      2.487     1.263       4.0      14.0   3.34
gamma_XY_a         104.634     10.496       91.172      120.633      5.227     2.648       4.0      11.0   3.73
gamma_XY_b          90.976     12.359       78.473      110.077      6.155     2.689       4.0      12.0   3.22
gamma_XA_a          93.947      7.670       83.098      102.174      3.820     1.274       5.0      11.0   3.08
gamma_XA_b          84.800     12.427       67.046      102.180      6.189     3.093       4.0      16.0   4.19
S_9                  4.574      7.047       -1.105       16.514      3.509     1.932       4.0      11.0   3.24
S_8                  6.828     11.056       -9.446       21.081      5.506     2.568       4.0      11.0   3.66
S_7                 92.840      4.996       86.576      100.544      2.488     1.264       4.0      14.0   3.34
S_6              -3477.981   3088.135    -8722.058     -867.843   1537.879   854.230       4.0      21.0   3.33
S_5            1593689.000  39393.174  1528991.893  1627772.597  19617.642  9983.155       4.0      14.0   3.46
S_4              96000.000      0.000    96000.000    96000.000      0.000     0.000    4078.0    2873.0   1.00
S_123           640000.000      0.000   640000.000   640000.000      0.000     0.000    4078.0    2873.0   1.00
phi                  5.006      0.000        5.006        5.006      0.000     0.000       4.0      14.0   4.11
>>> print(summary_full_plate_plate_100)
                  mean      sd  hdi_2.5%  hdi_97.5%  mcse_mean  mcse_sd  ess_bulk  ess_tail  r_hat
alpha_mu_X      80.000   0.000    80.000     80.000      0.000    0.000    3972.0      93.0   1.08
alpha_mu_A      10.265   0.732     9.593     11.503      0.365    0.202       4.0      13.0   3.59
alpha_mu_Y      -0.329   0.253    -0.650      0.059      0.126    0.064       4.0      12.0   3.61
N_E             19.604   0.682    18.445     20.199      0.340    0.189       4.0      13.0   3.59
X[0]            80.000   0.000    80.000     80.000      0.000    0.000    4238.0      63.0   1.11
...                ...     ...       ...        ...        ...      ...       ...       ...    ...
Y[98]          395.637   9.334   386.077    408.243      4.648    1.261       5.0      22.0   3.23
Y[99]          387.807  11.074   368.701    395.277      5.515    3.158       5.0      23.0   3.29
alpha_sigma_X    0.000   0.000     0.000      0.000      0.000    0.000       4.0      11.0   3.54
alpha_sigma_Y    0.962   0.355     0.583      1.517      0.177    0.082       4.0      11.0   3.35
phi              5.006   0.000     5.006      5.006      0.000    0.000       4.0      11.0   3.51

[307 rows x 9 columns]
"""