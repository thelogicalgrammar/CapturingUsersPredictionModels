"""
Functions for writing results to file, for reading
results from said file, and for plotting results.
"""
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


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
                   '"Threshold",'
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
                   '"max(R-hat)","min(ESS), bulk","min(ESS), tail","Divergences",'
                   '"pri_X_mean","pri_A_mean","pri_Y_mean","pri_X_std",'
                   '"pri_A_std","pri_Y_std","pri_X_skew","pri_A_skew",'
                   '"pri_Y_skew","pri_X_kurtosis","pri_A_kurtosis","pri_Y_kurtosis",'
                   '"post_X_mean","post_A_mean","post_Y_mean","post_X_std",'
                   '"post_A_std","post_Y_std","post_X_skew","post_A_skew",'
                   '"post_Y_skew","post_X_kurtosis","post_A_kurtosis","post_Y_kurtosis"\n')
        with open(full_path, 'w') as f:
            f.write(headers)
    else:
        print("Result-saving changed 11.12.2025. Delete result files created before that for extended headers.")

    # Write contents to file:
    key_ordering = ['experiment_id', 'time_stamp']
    key_ordering += ['X_o', 'pred_o', 'modified_prior', 'modified_prior_value']
    key_ordering += ['threshold']
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
    key_ordering += ['pri_X_mean', 'pri_A_mean', 'pri_Y_mean', 'pri_X_std']
    key_ordering += ['pri_A_std', 'pri_Y_std', 'pri_X_skew', 'pri_A_skew']
    key_ordering += ['pri_Y_skew', 'pri_X_kurtosis', 'pri_A_kurtosis', 'pri_Y_kurtosis']
    key_ordering += ['post_X_mean', 'post_A_mean', 'post_Y_mean', 'post_X_std']
    key_ordering += ['post_A_std', 'post_Y_std', 'post_X_skew', 'post_A_skew']
    key_ordering += ['post_Y_skew', 'post_X_kurtosis', 'post_A_kurtosis', 'post_Y_kurtosis']
    text = ''
    for key in key_ordering:
        text += '"' + str(result_dict[key]) + '",'
    text = text[:-1]  # Remove trailing comma
    text += '\n'  # Add newline
    with open(full_path, 'a') as f:
        f.write(text)
    print("Results saved to {}".format(full_path))


def read_outcomes_from_file(file_name='./results/results.csv'):
    """
    This function reads in results from the results.csv file. The
    function extracts the outcome under the true SCM given the
    agent's decision based on their posterior beliefs, and separates
    this by name of modified prior and groups the results by unique
    values of the modified prior.

    E.g. results_summary['N_E_mu']['parameter_values'] contains a list of
    unique values that N_E_mu have taken in the results.csv-file, and
    results_summary['N_E_mu']['value_lists'] contains lists of outcomes
    where every list corresponds to the parameter value at corresponding
    index and every item in the list is the outcome for one randomly
    sample X_o.

    Args:
        file_name (str): Path and file name of results file.
    """
    results = []
    with open(file_name, 'r') as f:
        reader = csv.reader(f)
        # Separate headers from contents
        headers = next(reader)
        for row in reader:
            results.append(row)

    # Separate results by modified parameter name and new value:
    # Find column 
    var_name_idx = headers.index('Modified prior')
    # Find unique row names:
    modified_var_names = set([row[var_name_idx] for row in results])
    # Create dict to store the results in:
    results_summary = {item: None for item in modified_var_names}
    # Find index of column with value of modified prior:
    var_value_idx = headers.index('New value of modified prior')
    posterior_Y_var_idx = headers.index('Y_mean_posterior_treatment')
    prior_Y_var_idx = headers.index('Y_mean_prior_treatment')
    true_Y_var_idx = headers.index('Y_mean_true_best_treatment')
    prior_CATE_idx = headers.index('Prior CATE (sample mean)')
    posterior_CATE_idx = headers.index('Posterior CATE (sample mean)')
    true_CATE_idx = headers.index('True CATE (sample mean)')
    # For e.g. the different values that 'N_E_mu' takes we want the
    # mean outcome Y over all sampled X_o.
    for key in results_summary.keys():
        # Pick out all rows that matches key:
        tmp_samples = [row for row in results if row[var_name_idx] == key]
        # Finding unique values of modified row:
        unique_values = list(set([float(item[var_value_idx]) for item in tmp_samples]))
        unique_values.sort()
        # Group the rows by the value of the modified prior and extract the outcome:
        results_summary[key] = {'parameter_values': unique_values}
        results_summary[key]['prior_value_lists'] = [
            [float(row[prior_Y_var_idx]) for row in tmp_samples
             if float(row[var_value_idx]) == param_value] for param_value in unique_values]
        results_summary[key]['posterior_value_lists'] = [
            [float(row[posterior_Y_var_idx]) for row in tmp_samples
             if float(row[var_value_idx]) == param_value] for param_value in unique_values]
        results_summary[key]['prior_CATE_lists'] = [
            [float(row[prior_CATE_idx]) for row in tmp_samples
             if float(row[var_value_idx]) == param_value] for param_value in unique_values]
        results_summary[key]['posterior_CATE_lists'] = [
            [float(row[posterior_CATE_idx]) for row in tmp_samples
             if float(row[var_value_idx]) == param_value] for param_value in unique_values]
        results_summary[key]['true_CATE_lists'] = [
            [float(row[true_CATE_idx]) for row in tmp_samples
             if float(row[var_value_idx]) == param_value] for param_value in unique_values]
            # Could add calculation of mean, standard deviation etc. here.
    return results_summary


def plot_graph(categories, outcomes, variable_name):
    """
    Args:
        categorires ([]): Labels for categories
        outcomes ([[]]): List of lists with outcomes as values
        variable_name (str): Name of modified prior parameter that the
            outcomes are associated with.
    """
    file_name = './fig/{}_boxplot.pdf'.format(variable_name)
    plt.figure(figsize=(6, 4))
    plt.boxplot(outcomes, labels=categories)
    plt.xlabel("Belief in {}".format(variable_name))
    plt.ylabel("Outcome Y")
    plt.title("Influence of belief on treatment effect on real-world outcomes")
    plt.tight_layout()
    plt.savefig(file_name)
    plt.clf()


def plot_prior_and_posterior_CATE(summary, variable_name):
    prior_values = summary['prior_CATE_lists']
    posterior_values = summary['posterior_CATE_lists']
    values = summary['prior_CATE_lists'] + summary['posterior_CATE_lists']

    categories = summary['parameter_values']
    prior_categories = [[category] * len(prior_values[i]) for i, category in enumerate(categories)]
    posterior_categories = [[category] * len(posterior_values[i]) for i, category in enumerate(categories)]
    categories = prior_categories + posterior_categories
    categories = [item for row in categories for item in row]
    prior_series = [['Prior predictive'] * len(val) for val in prior_values]
    prior_series = [item for row in prior_series for item in row]
    posterior_series = [['Posterior predictive'] * len(val) for val in posterior_values]
    posterior_series = [item for row in posterior_series for item in row]
    series = prior_series + posterior_series
    values = [item for row in values for item in row]

    data = {'Prior belief': categories,
            'CATE estimates': series,
            'CATE': values}
    sns.boxplot(data=data, x='Prior belief', y='CATE', hue='CATE estimates')
    plt.tight_layout()
    file_name = './fig/{}_boxplot.pdf'.format(variable_name)
    plt.savefig(file_name)
    plt.clf()



def read_and_plot():
    """
    Function that reads in all results from ./results/results.csv and
    creates boxplots for all variables found in column "Modified variables".
    """
    print("Reading in results from ./results/results.csv...")
    results = read_outcomes_from_file()
    print("Creating boxplots and writing to file...")
    print("Existing boxplot-files are overwritten.")
    for key in results.keys():
        plot_prior_and_posterior_CATE(results[key], key)
        # plot_graph(categories=results[key]['parameter_values'],
        #            outcomes=results[key]['prior_value_lists'],
        #            variable_name=key)
