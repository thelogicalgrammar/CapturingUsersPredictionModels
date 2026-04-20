"""
Code to run multiple experiments consequtively in series.

Basic setup: Everything else is kept at default values except for one
variable. X_o is sampled from the true distribution. Select which
parameters to sweep in main() by uncommenting or adding items to the
dict. Run series by:

.. code-block:: bash

    python experiment_series.py


Use utils.read_and_plot() for plotting.
"""
import numpy as np
import experiment

# Number of runs for every parameter value
N_ITERATIONS = 50
PLATE_SIZE = 1000
DRAWS = 1000
TUNE = 1000
TARGET_ACCEPT = .85
CHAINS = 4
MAX_TREEDEPTH = 15
INIT = "jitter+adapt_diag"
LOW_DIM_AUX_PLATE = True

# Default data generating parameters
scm_noise_parameters = {'alpha_mu_X': 80, 'alpha_sigma_X': 10,
                        'alpha_mu_A': 2, 'alpha_sigma_A': 1.0,
                        'alpha_mu_Y': 0.0, 'alpha_sigma_Y': 0.1,
                        'N_E': 1.0}

# DEFAULT Agent priors
agent_priors =\
    {'mu_X': 80, 'sigma_X': 0.1, 'mu_sigma_X': 10, 'sigma_sigma_X':0.1,
     'mu_A': 2, 'sigma_A': 1, 'mu_sigma_A': 1, 'sigma_sigma_A':0.1,
     'mu_Y': 0, 'sigma_Y': 0.01, 'mu_sigma_Y': 0.1, 'sigma_sigma_Y' : 0.01,
     'N_E_mu': 1.0, 'N_E_sigma': 1}


parameter_descriptions = {'agent_priors': 'Default',
                          'scm_constants': 'Default',
                          'scm_noise_parameters': 'Default'}


def main():
    # Generate parameters for individual experiments:
    deviations = [-2, -1, 0, 1, 2]
    # Data SCM does not have a standard deviation for N_E
    # mu_A is _just_the mean of the noice. Acts like a constant shift from d*X
    # Might make sense to reparameterize d to N_d (or something) to be able to capture
    # the belief that a historical treatment policy was the opposite of the actual.
    params = {'N_E_mu': [-20, 0, 20, 40, 60],
              'mu_X': [scm_noise_parameters['alpha_mu_X'] + deviation * scm_noise_parameters['alpha_sigma_X'] for deviation in deviations],
              'mu_A': [scm_noise_parameters['alpha_mu_A'] + deviation * scm_noise_parameters['alpha_sigma_A'] for deviation in deviations],
              'mu_Y': [scm_noise_parameters['alpha_mu_Y'] + deviation * scm_noise_parameters['alpha_sigma_Y'] for deviation in deviations],
              }
    # Flatten:
    parameter_list = [(key, value) for key, values in params.items() for value in values]

    rng = np.random.default_rng(seed = 4)

    for iteration in range(N_ITERATIONS):
        # One run for every parameter combination first, then second run for the same etc.
        for param_name, value in parameter_list:
            # Sample X_o
            # Data generating parameters are fixed
            X_o = rng.normal(scm_noise_parameters['alpha_mu_X'], scm_noise_parameters['alpha_sigma_X'])
            X_o = np.array(X_o)
            # Make a copy of agent_priors to not affect next run
            tmp_agent_priors = agent_priors.copy()
            # Keep everything else fixed, except for one value.
            tmp_agent_priors[param_name] = value
            modified_prior = {'modified_prior': param_name,
                            'modified_prior_value': value}
            # Helper variable for easy view of experiment parameters:
            parameter_descriptions['agent_priors'] = 'Modified'
            print("=" * 40)
            print("Iteration {}".format(iteration))
            print("Running experiment with {} set to {}. X_o is {}".format(param_name, value, X_o))
            experiment.main(X_o, tmp_agent_priors, modified_prior,
                            plate_size=PLATE_SIZE,
                            draws=DRAWS,
                            tune=TUNE,
                            target_accept=TARGET_ACCEPT,
                            chains=CHAINS,
                            max_treedepth=MAX_TREEDEPTH,
                            init=INIT,
                            low_dim_aux_plate=LOW_DIM_AUX_PLATE,
                            parameter_descriptions=parameter_descriptions,
                            rng=rng)


if __name__ == '__main__':
    # Run the serialized experiment.
    main()
