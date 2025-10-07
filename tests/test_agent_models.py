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
    # Model defined with pm.Normal(..., phi, 1.0, observed=phi_hat) for low-dim model. Full model has sigma=1e-3.
>>> print(summary_low_dim_plate_100)
                      mean        sd     hdi_2.5%    hdi_97.5%  mcse_mean   mcse_sd  ess_bulk  ess_tail  r_hat
alpha_mu_X          80.000     0.000       80.000       80.000      0.000     0.000    4121.0     944.0   1.03
alpha_mu_A          10.142     0.224        9.811       10.361      0.111     0.042       4.0      15.0   3.43
alpha_mu_Y          -0.297     0.483       -1.061        0.277      0.241     0.124       4.0      18.0   3.32
N_E                 19.826     0.498       19.323       20.466      0.248     0.053       4.0      18.0   3.19
epsilon_SX          -0.414     0.524       -0.742        0.493      0.261     0.150       4.0      12.0   3.31
epsilon_SY          -0.479     0.647       -1.518        0.217      0.322     0.162       4.0      16.0   3.34
epsilon_SA          -0.159     0.234       -0.544        0.076      0.116     0.062       4.0      14.0   4.21
alpha_sigma_X        0.000     0.000        0.000        0.000      0.000     0.000       4.0      14.0   3.53
alpha_sigma_Y        0.832     0.151        0.715        1.084      0.075     0.040       4.0      13.0   3.56
gamma_XX           105.647    13.448       92.226      124.702      6.697     2.170       4.0      11.0   4.02
gamma_XY_a          99.175    17.570       86.239      129.207      8.750     4.909       4.0      11.0   3.25
gamma_XY_b          92.358     9.223       84.608      107.392      4.593     2.287       4.0      13.0   3.60
gamma_XA_a         108.776    14.711       87.217      123.061      7.326     2.660       4.0      13.0   3.23
gamma_XA_b         101.713    15.545       84.430      126.015      7.741     3.672       4.0      13.0   3.50
S_9                  3.532    13.379      -11.408       18.602      6.663     0.861       4.0      14.0   3.45
S_8                  3.407    11.131       -6.846       22.101      5.543     3.022       4.0      12.0   3.25
S_7                105.643    13.449       92.221      124.697      6.697     2.169       4.0      11.0   4.02
S_6              -2406.276  3839.002    -8500.184     2115.348   1911.808   989.046       4.0      19.0   3.31
S_5            1607489.636  9000.474  1600348.984  1622732.585   4482.200  2462.519       4.0      11.0   4.09
S_4              96000.000     0.000    96000.000    96000.000      0.000     0.000    4105.0    1524.0   1.03
S_123           640000.000     0.000   640000.000   640000.000      0.000     0.000    4105.0    1576.0   1.03
phi                  5.036     0.067        4.968        5.114      0.033     0.004       4.0      11.0   4.09
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



Run 2 with pm.Normal(..., phi, 0.1, observed=phi_hat)

>>> print(summary_low_dim_plate_100)
                      mean         sd     hdi_2.5%    hdi_97.5%  mcse_mean   mcse_sd  ess_bulk  ess_tail  r_hat
alpha_mu_X          80.000      0.000       80.000       80.000      0.000     0.000    3762.0    2626.0   1.01
alpha_mu_A          10.145      0.497        9.564       10.744      0.248     0.051       4.0      11.0   3.73
alpha_mu_Y          -0.225      0.351       -0.547        0.297      0.175     0.066       4.0      13.0   3.42
N_E                 19.705      0.495       19.122       20.253      0.246     0.039       4.0      11.0   3.54
epsilon_SX          -0.284      0.937       -1.251        1.264      0.466     0.251       4.0      11.0   3.83
epsilon_SY          -0.312      0.310       -0.647        0.042      0.154     0.019       4.0      11.0   3.49
epsilon_SA          -0.046      0.136       -0.229        0.150      0.068     0.033       4.0      15.0   3.67
alpha_sigma_X        0.000      0.000        0.000        0.000      0.000     0.000       4.0      12.0   3.21
alpha_sigma_Y        1.004      0.249        0.610        1.254      0.124     0.058       4.0      13.0   3.78
gamma_XX           103.622     11.025       93.906      122.261      5.491     3.028       4.0      11.0   3.50
gamma_XY_a          89.920      8.803       77.279      102.165      4.384     2.191       4.0      11.0   3.43
gamma_XY_b          91.931     11.986       77.574      105.088      5.969     1.001       4.0      11.0   3.87
gamma_XA_a          97.454     10.446       83.658      111.219      5.202     1.927       4.0      11.0   3.91
gamma_XA_b          92.192      4.412       85.114       96.816      2.197     1.090       4.0      16.0   3.38
S_9                  2.632      4.630       -2.330        9.549      2.306     0.927       4.0      11.0   3.35
S_8                 -1.007      4.134       -7.595        3.830      2.059     1.071       4.0      11.0   3.48
S_7                103.612     11.021       93.904      122.245      5.488     3.027       4.0      11.0   3.50
S_6              -1828.735   2796.516    -4416.688     2311.453   1392.654   520.795       5.0      18.0   3.06
S_5            1597119.424  38493.852  1549195.861  1643574.804  19169.784  4839.767       4.0      12.0   3.33
S_4              96000.000      0.000    96000.000    96000.000      0.000     0.000    3544.0    2454.0   1.01
S_123           640000.000      0.000   640000.000   640000.000      0.000     0.000    3544.0    2446.0   1.01
phi                  5.006      0.005        5.001        5.014      0.003     0.001       5.0      12.0   3.21
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