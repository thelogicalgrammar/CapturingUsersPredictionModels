"""
This file generates a parameter file for slurm-parameters
"""
import os
import shutil
import numpy as np


# Noise parameters here only for setting scanning parameter values.
# These noise parameters do not affect any other aspect of the program.
scm_noise_parameters = {'alpha_mu_X': 80, 'alpha_sigma_X': 10,
                        'alpha_mu_A': 2, 'alpha_sigma_A': 1.0,
                        'alpha_mu_Y': 0.0, 'alpha_sigma_Y': 0.1,
                        'N_E': 1.0}


def generate_slurm_parameter_file(parameters_file='./slurm/tmp/parameters.txt'):
    """
    Function for generating parameters for slurm script

    Args:
        parameters_file (str): File name of file to write parameters to.
    """
    # Check if tmp exists:
    if not os.path.exists('./slurm/tmp/'):
        # Create path:
        os.mkdir('./slurm/tmp/')
    # Define parameter values to scan
    n_iterations = 50
    # Parameters for all "narrow" and "wide" plots
    deviations = [i/10 for i in range(-30, 31)]  # 31 because not inclusive of upper bound.
    params = {
            # The following takes care of mu_A wide
            'mu_A': [scm_noise_parameters['alpha_mu_A'] + deviation * scm_noise_parameters['alpha_sigma_A'] for deviation in deviations] +
                [scm_noise_parameters['alpha_mu_A'] + deviation / 10 * scm_noise_parameters['alpha_sigma_A'] for deviation in deviations],
            # This takes care of mu_X wide
            'mu_X': [scm_noise_parameters['alpha_mu_X'] + deviation * scm_noise_parameters['alpha_sigma_X'] for deviation in deviations] +
                [scm_noise_parameters['alpha_mu_X'] + deviation / 10 * scm_noise_parameters['alpha_sigma_X'] for deviation in deviations],
            # mu_Y narrow: This still needs multiple decimals stored in parameters.txt to work.
            'mu_Y': [scm_noise_parameters['alpha_mu_Y'] + deviation * scm_noise_parameters['alpha_sigma_Y'] for deviation in deviations] +
                [scm_noise_parameters['alpha_mu_Y'] + deviation / 10 * scm_noise_parameters['alpha_sigma_Y'] for deviation in deviations],
            # N_E_mu "narrow": This covers [-0.5, 2.5].
            'N_E_mu': [scm_noise_parameters['N_E'] + deviation * 0.05 for deviation in deviations] + 
                [scm_noise_parameters['N_E'] + deviation / 10 * 0.05 for deviation in deviations],
           }

    # Parameters for scan over controlled values of X_o (+-4 SD):
    # Default parameters for scan with no priors changed
    # params = {'mu_X': [scm_noise_parameters['alpha_mu_X']]}
    # X_o_list = [i / 10 * scm_noise_parameters['alpha_sigma_X'] +
    #             scm_noise_parameters['alpha_mu_X'] for i in range(-40, 41)]

    with open(parameters_file, 'w') as file:
        # Write parameters for scan over parameter values
        for _ in range(n_iterations):
            for key in params.keys():
                for value in params[key]:
                    if 'X_o_list' in locals():
                        for X_o in X_o_list:
                            file.write(f'{key} {value:.1f} {X_o}\n')
                    else:
                        X_o = np.random.normal(loc=scm_noise_parameters['alpha_mu_X'],
                                     scale=scm_noise_parameters['alpha_sigma_X'])
                        # The next line cuts away anything beyond the first decimal.
                        file.write(f'{key} {value:.6f} {X_o}\n')
    # Create a copy of same file for use by completion_check.py
    shutil.copy(parameters_file, './slurm/tmp/original_parameters.txt')

    print("Parameters stored with 6 decimals.")
    print("Slurm job parameters for noise parameters {} created successfully".format(params.keys()))


if __name__ == '__main__':
    # Generate parameter file.
    generate_slurm_parameter_file()
