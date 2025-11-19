"""

"""
from pathlib import Path


def write_results_to_file(result_dict, file_name='results.csv', path='./results/'):
    """
    Function for writing results to file.

    Args:
    result_dict [dict]: Dict of results to store.
    """
    # Check for existence of folder. Create if it does not exist.
    Path(path).mkdir(parents=True, exist_ok=True)
    # Check for existence of result file.
    full_path = path + file_name
    if not Path(path + file_name).exists():
        # Create file and write header to file
        headers = ('"Experiment ID","Time Stamp",'
                   '"X_o","Modified prior","New value of modified prior",'
                   '"True CATE (mu)","Prior CATE (mu)","Posterior CATE (sample mean)",'
                   '"True CATE (sample mean)","Prior CATE (sample mean)",'
                   '"True CATE Standard Deviation (sample estimate)",'
                   '"Prior CATE Standard Deviation (sample estimate)",'
                   '"Posterior CATE Standard Deviation (sample estimate)",'
                   '"Plate Size","A1_treatment","A0_treatment",'
                   '"Agent Priors","SCM Noise Parameters",'
                   '"SCM Constants",'
                   '"Running Time"\n')
        with open(full_path, 'w') as f:
            f.write(headers)
    # Write contents to file:
    key_ordering = ['experiment_id', 'time_stamp']
    key_ordering += ['X_o', 'modified_prior', 'modified_prior_value']
    key_ordering += ['true_CATE_mu', 'prior_CATE_mu', 'posterior_CATE_mean',
                     'true_CATE_mean', 'prior_CATE_mean']
    key_ordering += ['true_CATE_std', 'prior_CATE_std', 'posterior_CATE_std']
    key_ordering += ['plate_size', 'A1_treatment', 'A0_treatment']
    key_ordering += ['agent_priors', 'scm_noise_parameters']
    key_ordering += ['scm_constants']
    key_ordering += ['running_time']
    text = ''
    for key in key_ordering:
        text += '"' + str(result_dict[key]) + '",'
    text = text[:-1]  # Remove trailing comma
    text += '\n'  # Add newline
    with open(full_path, 'a') as f:
        f.write(text)
    print("Results saved to {}".format(full_path))
