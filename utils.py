"""

"""
import csv
import numpy as np
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
                   '"X_o","pred_o","Modified prior","New value of modified prior",'
                   '"True CATE (sample mean)","Prior CATE (sample mean)","Posterior CATE (sample mean)",'
                   '"True CATE Standard Deviation (sample estimate)",'
                   '"Prior CATE Standard Deviation (sample estimate)",'
                   '"Posterior CATE Standard Deviation (sample estimate)",'
                   '"True CATE (mu)","Prior CATE (mu)",'
                   '"prior_treatment","posterior_treatment","true_best_treatment",'
                   '"prior_decision_correct","posterior_decision_correct","decision_changed",'
                   '"frac_correct_prior_decision","frac_correct_posterior_decision","frac_correct_true_best_treatment",'
                   '"Y_mean_prior_treatment","Y_mean_posterior_treatment","Y_mean_true_best_treatment",'
                   '"potential_outcome_A1_mean","potential_outcome_A1_std","potential_outcome_A0_mean","potential_outcome_A0_std",'
                   '"Plate Size","A1_treatment","A0_treatment",'
                   '"Agent Priors","SCM Noise Parameters",'
                   '"SCM Constants",'
                   '"Running Time",'
                   '"max(R-hat)","min(ESS), bulk","min(ESS), tail","Divergences"\n')
        with open(full_path, 'w') as f:
            f.write(headers)
    else:
        print("Result-saving changed 26.11.2025. Delete result files created before that for correct headers.")

    # Write contents to file:
    key_ordering = ['experiment_id', 'time_stamp']
    key_ordering += ['X_o', 'pred_o', 'modified_prior', 'modified_prior_value']
    key_ordering += ['true_CATE_mean', 'prior_CATE_mean', 'posterior_CATE_mean']
    key_ordering += ['true_CATE_std', 'prior_CATE_std', 'posterior_CATE_std']
    key_ordering += ['true_CATE_mu', 'prior_CATE_mu', ]
    key_ordering += ["prior_treatment", "posterior_treatment", "true_best_treatment"]
    key_ordering += ["prior_decision_correct", "posterior_decision_correct", "decision_changed"]
    key_ordering += ["frac_correct_prior_decision", "frac_correct_posterior_decision", "frac_correct_true_best_treatment"]
    key_ordering += ["Y_mean_prior_treatment", "Y_mean_posterior_treatment", "Y_mean_true_best_treatment"]
    key_ordering += ["potential_outcome_A1_mean", "potential_outcome_A1_std", "potential_outcome_A0_mean", "potential_outcome_A0_std"]

    key_ordering += ['plate_size', 'A1_treatment', 'A0_treatment']
    key_ordering += ['agent_priors', 'scm_noise_parameters']
    key_ordering += ['scm_constants']
    key_ordering += ['running_time']
    key_ordering += ['max_rhat', 'min_ess_bulk', 'min_ess_tail', 'n_divergences']
    text = ''
    for key in key_ordering:
        text += '"' + str(result_dict[key]) + '",'
    text = text[:-1]  # Remove trailing comma
    text += '\n'  # Add newline
    with open(full_path, 'a') as f:
        f.write(text)
    print("Results saved to {}".format(full_path))


def read_and_summarise_results(file_name='./results/results.csv'):
    """
    """
    results = []
    with open(file_name, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        for row in reader:
            results.append(row)

    # Separate results by modified value, estimate means and standard deviations
    # keys = ['N_E_mu', 'mu_X', 'mu_A', 'mu_Y']
    results_summary = {'N_E_mu': None, 'mu_X': None, 'mu_A': None, 'mu_Y': None}

    # Find column 
    var_name_idx = headers.index('Modified prior')
    var_value_idx = headers.index('New value of modified prior')
    prior_Y_var_idx = headers.index('Y_mean_prior_treatment')
    posterior_Y_var_idx = headers.index('Y_mean_posterior_treatment')
    true_Y_var_idx = headers.index('Y_mean_true_best_treatment')
    true_Y_list = []
    for key in results_summary.keys():
        # Subset results on matching rows:
        tmp_rows = []
        for row in results:
            if key == row[var_name_idx]:
                tmp_rows.append(row)
                # Practical to store true_Y here:
                true_Y_list.append(float(row[true_Y_var_idx]))
        # Bin results based on matching values for the changed parameter
        tmp_prior_values = []
        tmp_posterior_values = []
        for value in np.unique(tmp_rows[var_value_idx]):
            results_for_var_name = {}  # Reset
            for row in tmp_rows:
                if row[var_value_idx] == value:
                    tmp_prior_values.append(float(row[prior_Y_var_idx]))
                    tmp_posterior_values.append(float(row[posterior_Y_var_idx]))
            # Calculate summaries:
            prior_mean_Y = np.mean(tmp_prior_values)
            prior_std_Y = np.std(tmp_prior_values)
            posterior_mean_Y = np.mean(tmp_posterior_values)
            posterior_std_Y = np.std(tmp_posterior_values)
            # Reset tmp and add results to dict
            results_for_value = {}
            results_for_value['value'] = value
            results_for_value['prior_mean_Y'] = prior_mean_Y
            results_for_value['posterior_mean_Y'] = posterior_mean_Y
            results_for_value['prior_std_Y'] = prior_std_Y
            results_for_value['posterior_std_Y'] = posterior_std_Y

            results_for_var_name[str(value)] = results_for_value
        results_summary[key] = results_for_var_name
    results_summary['true_mean_Y'] = np.mean(true_Y_list)
    results_summary['true_std_Y'] = np.std(true_Y_list)
    return results_summary


def visualize():
    """
    Unchecked code produced by Claude. Absolutely needs to be looked through.
    """
    print("This cannot be correct. Priors and posteriors align perfectly. FIX!")
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Read data
    df = pd.read_csv('./results/results.csv')

    # Get all unique parameters
    params = df['Modified prior'].unique()

    # Create subplot for each parameter
    fig, axes = plt.subplots(len(params), 1, figsize=(12, 6*len(params)))
    if len(params) == 1:
        axes = [axes]

    for ax, param_name in zip(axes, params):
        # Filter data
        df_filtered = df[df['Modified prior'] == param_name]
        
        # Reshape
        df_long = pd.melt(
            df_filtered,
            id_vars=['New value of modified prior'],
            value_vars=['Y_mean_prior_treatment', 'Y_mean_posterior_treatment'],
            var_name='decision_type',
            value_name='Y_mean'
        )
        df_long['decision_type'] = df_long['decision_type'].map({
            'Y_mean_prior_treatment': 'Prior',
            'Y_mean_posterior_treatment': 'Posterior'
        })
        
        # Plot
        sns.boxplot(data=df_long, x='New value of modified prior', y='Y_mean',
                    hue='decision_type', palette=['lightblue', 'lightcoral'], ax=ax)
        ax.set_xlabel(f'{param_name} value')
        ax.set_ylabel('Mean Outcome (Y)')
        ax.set_title(f'{param_name}')

    plt.tight_layout()
    plt.savefig('./fig/all_boxplots.png', dpi=300)
    #plt.show()
