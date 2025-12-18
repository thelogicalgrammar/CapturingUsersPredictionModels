"""
This file generates a parameter file for slurm-parameters
"""
import os


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
    deviations = [i for i in range(-5, 5)]
    params = {#'N_E_mu': [-20, 0, 20, 40, 60],
            #'mu_X': [scm_noise_parameters['alpha_mu_X'] + deviation * scm_noise_parameters['alpha_sigma_X'] for deviation in deviations],
            'mu_A': [scm_noise_parameters['alpha_mu_A'] + deviation * scm_noise_parameters['alpha_sigma_A'] for deviation in deviations],
            #'mu_Y': [scm_noise_parameters['alpha_mu_Y'] + deviation * scm_noise_parameters['alpha_sigma_Y'] for deviation in deviations],
            }

    parameters_file = open(parameters_file, 'w')
    for iteration in range(n_iterations):
        for key in params.keys():
            for value in params[key]:
                # Generate parameter line
                txt = key + ' ' + str(value) + '\n'
                # srun python crf_experiment.py criteo2_1to1.csv ./datasets/criteo2/ 1.0 {0} {1} {2}
                # python -m experiments.crf_experiment criteo2_1to1.csv ./datasets/criteo2/ split_undersampling uplift_rf analytic
                parameters_file.write(txt)
    print("Slurm job parameters for noise parameters {} created successfully".format(params.keys()))


if __name__ == '__main__':
    # Generate parameter file.
    generate_slurm_parameter_file()
