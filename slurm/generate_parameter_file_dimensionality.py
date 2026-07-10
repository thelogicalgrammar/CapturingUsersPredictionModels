"""
Generates a parameter file for the dimensionality SLURM array job.
Each line is: <param_name> <param_value> <n_covariates>

X_o is always sampled internally by experiment_dimensionality.py.

After running this script, update line 3 in array_job_dimensionality.sh:
    #SBATCH --array=0-<N-1>
where N = (number of lines in parameters_dimensionality.txt) / 4.

Run from the repo root::

    python slurm/generate_parameter_file_dimensionality.py
"""
import os
import shutil
import numpy as np


# Reference noise parameters (only used to derive scanning grids).
scm_noise_parameters = {
    'alpha_mu_X':    80.0,
    'alpha_sigma_X': 10.0,
    'alpha_mu_A':     2.0,
    'alpha_sigma_A':  1.0,
    'alpha_mu_Y':     0.0,
    'alpha_sigma_Y':  0.1,
    'N_E':            1.0,
}

# Values of n_covariates to scan.
N_COVARIATES_LIST = [1, 2, 4, 8, 16, 32]


def generate_slurm_parameter_file(
        parameters_file='./slurm/tmp/parameters_dimensionality.txt'):
    """
    Write parameter combinations to a text file for the SLURM array job.

    Args:
        parameters_file (str): Output path for the parameter file.
    """
    os.makedirs(os.path.dirname(parameters_file), exist_ok=True)

    n_iterations = 50
    deviations   = [-30, -10, -3, -1, 0, 1, 3, 10, 30]
    #list(range(-30, 31))   # -30 … +30

    # Same parameter grid as the scalar experiment.
    params = {
        'mu_A':   [scm_noise_parameters['alpha_mu_A']  + d * scm_noise_parameters['alpha_sigma_A']  for d in deviations],
        'mu_X':   [scm_noise_parameters['alpha_mu_X']  + d * scm_noise_parameters['alpha_sigma_X']  for d in deviations],
        'mu_Y':   [scm_noise_parameters['alpha_mu_Y']  + d * scm_noise_parameters['alpha_sigma_Y']  for d in deviations],
        'N_E_mu': [scm_noise_parameters['N_E']         + d * 0.05                                   for d in deviations],
    }

    total = 0
    with open(parameters_file, 'w') as f:
        for _ in range(n_iterations):
            for key, values in params.items():
                for value in values:
                    for k in N_COVARIATES_LIST:
                        f.write(f'{key} {value:.6f} {k}\n')
                        total += 1

    shutil.copy(parameters_file, './slurm/tmp/original_parameters_dimensionality.txt')

    n_tasks = total // 4
    print(f"Wrote {total} lines to {parameters_file}.")
    print(f"Set #SBATCH --array=0-{n_tasks - 1} in array_job_dimensionality.sh")
    print(f"Parameter grid: {list(params.keys())} × k={N_COVARIATES_LIST} × {n_iterations} iterations")


if __name__ == '__main__':
    generate_slurm_parameter_file()
