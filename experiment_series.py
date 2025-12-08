"""
Code to run a series of experiments consequtively.

Basic setup: Keep everything else at default values, change one
Testing sensitivity of outcome under posterior decision to changes in priors.
-N_E_mu: mu + [-2, -1, 0, 1, 2]  # N_E does not have a defined std in SCM.
-mu_X: -2std, -1 std, 0, +1 std, +2 std
-mu_Y: -2std, -1 std, 0, +1 std, +2 std
-mu_A: -2std, -1 std, 0, +1 std, +2 std

-5(?) initial runs for each parameter combination
-Run more if time allows

"""
import numpy as np
import experiment
import utils

# Number of runs for every parameter value
N_ITERATIONS = 50


# Default data generating parameters
scm_noise_parameters = {'alpha_mu_X': 80, 'alpha_sigma_X': 10,
                        'alpha_mu_A': 2, 'alpha_sigma_A': 1.0,
                        'alpha_mu_Y': 0.0, 'alpha_sigma_Y': 0.1,
                        'N_E': 1.0, 'N_CC': 0.125, 'N_RC': -0.1}

# DEFAULT Agent priors
agent_priors =\
    {'mu_X': 80, 'sigma_X': 1, 'lambda_X': 10, 'mu_A': 2, 'sigma_A': 1, # 'lambda_A': 10,  # lambda_A is not used in model
     'mu_Y': 0, 'sigma_Y': 1, 'lambda_Y': 1, 'N_E_mu': 1.0, 'N_E_sigma': 1, 
     'N_CC_mu': 0.125, 'N_CC_sigma': 0.1, 'N_RC_mu': -0.1, 'N_RC_sigma': 0.1}

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
    params = {'N_E_mu': [scm_noise_parameters['N_E'] + 1.0 * item for item in deviations],
              'mu_X': [scm_noise_parameters['alpha_mu_X'] + deviation * scm_noise_parameters['alpha_sigma_X'] for deviation in deviations],
              'mu_A': [scm_noise_parameters['alpha_mu_A'] + deviation * scm_noise_parameters['alpha_sigma_A'] for deviation in deviations],
              'mu_Y': [scm_noise_parameters['alpha_mu_Y'] + deviation * scm_noise_parameters['alpha_sigma_Y'] for deviation in deviations],
              'N_CC_mu': [agent_priors['N_CC_mu'] + deviation * 0.1 for deviation in deviations],
              'N_RC_mu': [agent_priors['N_RC_mu'] + deviation * 0.1 for deviation in deviations]
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
            experiment.main(X_o, tmp_agent_priors, modified_prior, rng=rng,
                            scm_noise_parameters=scm_noise_parameters,
                            parameter_descriptions=parameter_descriptions)


if __name__ == '__main__':
    # Run the serialized experiment.
    main()
    # Read and plot *everything* in results.csv:
    utils.read_and_plot()
