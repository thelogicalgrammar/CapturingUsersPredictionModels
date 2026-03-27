"""
Functions for writing results to file, for reading
results from said file, and for plotting results.
"""
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def write_results_to_file(result_dict, file_name='results.csv', path='./results/'):
    """
    Function for writing results to file.

    Args:
        result_dict [dict]: Dict of results to store.
        file_name (str): Name of file to write results to.
        path (str): Path where to store result file.
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
                   '"post_Y_skew","post_X_kurtosis","post_A_kurtosis","post_Y_kurtosis",'
                   '"expected_prior_outcome_A1_mean","expected_prior_outcome_A1_std",'
                   '"expected_prior_outcome_A0_mean","expected_prior_outcome_A0_std",'
                   '"expected_posterior_outcome_A1_mean","expected_posterior_outcome_A1_std",'
                   '"expected_posterior_outcome_A0_mean","expected_posterior_outcome_A0_std"\n')
        with open(full_path, 'w') as f:
            f.write(headers)

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
    key_ordering += ['expected_prior_outcome_A1_mean', 'expected_prior_outcome_A1_std',
                     'expected_prior_outcome_A0_mean', 'expected_prior_outcome_A0_std',
                     'expected_posterior_outcome_A1_mean', 'expected_posterior_outcome_A1_std',
                     'expected_posterior_outcome_A0_mean', 'expected_posterior_outcome_A0_std']
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
    # Not in use. Only valid for very specific models.
    # true_Y_var_idx = headers.index('Y_mean_true_best_treatment')
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
            [float(row[prior_Y_var_idx]) for row in tmp_samples  # prior predictive Y
             if float(row[var_value_idx]) == param_value] for param_value in unique_values]
        results_summary[key]['posterior_value_lists'] = [  # posterior predictive Y
            [float(row[posterior_Y_var_idx]) for row in tmp_samples
             if float(row[var_value_idx]) == param_value] for param_value in unique_values]
        results_summary[key]['prior_CATE_lists'] = [   # prior predictive CATE
            [float(row[prior_CATE_idx]) for row in tmp_samples
             if float(row[var_value_idx]) == param_value] for param_value in unique_values]
        results_summary[key]['posterior_CATE_lists'] = [  # posterior predictive CATE
            [float(row[posterior_CATE_idx]) for row in tmp_samples
             if float(row[var_value_idx]) == param_value] for param_value in unique_values]
        results_summary[key]['true_CATE_lists'] = [  # CATE based on scm (data generating model)
            [float(row[true_CATE_idx]) for row in tmp_samples
             if float(row[var_value_idx]) == param_value] for param_value in unique_values]
            # Could add calculation of mean, standard deviation etc. here.
    return results_summary


def parse_results(mod_var_name='mu_X', file_name='./results/results.csv',
                  field_names=['pri_Y_mean', 'post_Y_mean',
                               'potential_outcome_A1_mean',
                               'potential_outcome_A0_mean']):
    """
    Reads result-file and extracts the requested fields to
    a numpy-array for e.g. easy plotting. By default, these
    are the expected Y-values given the agent's prior and
    posterior beliefs, the true Y values with and without
    treatment, and the standard deviations for all of these.

    Args:
        mod_var_name (str): Name of variable of interest (variable that was modified from
            defaults). Selects only corresponding rows from from the result-file.
        file_name (str): Path and file name of result-file.
        field_names ([str]): List of strings containing the column names of interest to extract
            from the result-file.
    """
    # Read file
    results = []
    with open(file_name, 'r') as f:
        reader = csv.reader(f)
        # Separate headers from contents
        headers = next(reader)
        for row in reader:
            results.append(row)
    # Find column index for column with modified variable name:
    mod_var_name_idx = headers.index('Modified prior')
    mod_var_value_idx = headers.index('New value of modified prior')
    # Find column indices matching field names:
    field_idx = [headers.index(name) for name in field_names]
    # Make list of lists with all relevant values.
    # Inner list-comprehension extracts everything relevant from a row.
    mod_var_values = [float(row[mod_var_value_idx]) for row in results if row[mod_var_name_idx] == mod_var_name]
    mod_var_values = np.array(mod_var_values)
    outputs = [[float(row[idx]) for idx in field_idx] for row \
               in results if row[mod_var_name_idx] == mod_var_name]
    outputs = np.array(outputs)  # Outputs of every run of requested field_names
    # Sort array by value of modified variable
    sort_idx = np.argsort(mod_var_values)
    mod_var_values = mod_var_values[sort_idx]
    outputs = outputs[sort_idx, :]
    # Aggregate
    unique_mod_var_values = np.unique(mod_var_values)
    summary_stats = np.zeros(shape=(unique_mod_var_values.shape[0], len(field_idx) * 2 + 1))
    for i, val in enumerate(unique_mod_var_values):
        mask = mod_var_values == val
        subset = outputs[mask, :]
        # First column contains value of selected mod_var
        summary_stats[i, 0] = val
        # Of the remaining columns, the first half contain the means of the values in columns field_names,
        # the remaining contains the standard deviations.
        summary_stats[i, 1:] = np.concat((subset.mean(axis=0), subset.std(axis=0)))
    # Add mod_var_values to leading column in outputs:
    outputs = np.concat((mod_var_values.reshape(-1, 1), outputs), axis=1)
    # Return
    return (summary_stats, outputs, field_names)


def plot_CATE_for_var(var, mean_of_x, std_of_x, x_label, file_name, x_lim, y_lim, scan):
    """
    Args:
        var (str): Name of variable of interest
        mean_of_x (float): Mean of horizontal scale (x) to center on.
        std_of_x (float): Standard deviation of the horizontal scale (x).
            Used for normalizing plot.
        x_label (str): Label that will be visible under the x-axis
        file_name (str): File name and path of the result file to parse.
        x_lim ((float, float)): +-Limits of horizontal axis
        y_lim ((float, float)): +-Limits of vertical axis

    Notes:
    Uncertainty of true model (with default values) is tiny.
    """
    fields = ['Prior CATE (sample mean)',
              'Posterior CATE (sample mean)',
              'True CATE (sample mean)']
    label_names = ['Without ML-DS', 'With ML-DS', 'True value']
    # Threshold used in experiments. If CATE is above this, then treatment A1
    # is used. Otherwise A0 is used.
    threshold = 5

    n_observations = 4000
    fig, ax = plt.subplots()
    for field, label_name in zip(fields, label_names):
        tmp = parse_results(mod_var_name=var, field_names=[field],
                            file_name=file_name)
        values = tmp[0]
        # Measure X in units of standard deviation
        if std_of_x is not None:
            x = (values[:, 0] - mean_of_x) / std_of_x
        else:
            x = values[:, 0]  # No normalization
        y = values[:, 1]
        ax.plot(x, y)
        # Standard deviation of means multiplied by sqrt(n) to get std of observations.
        # Interpretation: Uncertainty over all X_o (not entirely koscher).
        yerr = np.sqrt(n_observations) * values[:, 2]
        ax.fill_between(x, y+yerr, y-yerr, alpha=0.2, label=label_name)
        if field == 'Posterior CATE (sample mean)':
            # x_post same as for all plots
            y_post = tmp[0][:, 1]
            yerr_post = np.sqrt(n_observations) * tmp[0][:, 2]
        elif field == 'True CATE (sample mean)':
            y_true = tmp[0][:, 1]
    # Check when posterior is within +-1 std from true mean:
    # That would indicate that the certainty has correctly increased.
    tmp1 = (y_true > y_post - yerr_post)
    tmp2 = (y_true < y_post + yerr_post)
    mask = tmp1 & tmp2
    ax.fill_between(x, 0, 1, where=mask, transform=ax.get_xaxis_transform(),
                    color='lightgray', alpha=.4)
    # Add dotted line for threshold selection
    ax.axhline(y=threshold, linestyle='--', color='black', label='Treatment threshold')
    if x_lim is not None:
        plt.xlim(x_lim)
        plt.ylim(y_lim)
    plt.legend()
    # plt.xlabel('Prior beliefs on $\\{}$'.format(label))
    plt.xlabel(x_label)
    plt.ylabel('CATE')
    #plt.title(title)
    plt.savefig('./fig/{}_CATE_{}.pdf'.format(scan, var))


def plot_CATEs(file_name='./results/results_18_2.csv'):
    """
    Function for plotting CATEs.
    
    Args:
        file_name (str): Path and name of result file.
    """
    scans = ['wide', 'focused']
    var_names = ['mu_X', 'N_E_mu', 'mu_Y', 'mu_A']
    mean_of_vars = [80.0, None, 0.0, 2.0]
    std_of_vars = [10, None, 0.1, 1]  # N_E does not really have a variance.
    # Limits for wide scan results:
    x_lim_arrays = [[(-30, 30), (-.5, 2.5), (-30, 30), (-30, 30)],
                    [(-3, 3), (.7, 1.3), (-3, 3), (-3, 3)]]  # This second for focused plots
    y_lims = [(None, None), (-30, 50), (-15, 35), (None, None)]
    x_labels = ['Deviation of prior belief $\\alpha_{\\mu_X}$ from $\\mu_X$ in units of SD ($\\alpha_{\\sigma_X}$)',
                'Prior belief on $N_E$',
                'Deviation of prior belief $\\alpha_{\\mu_Y}$ from $\\mu_Y$ in units of SD ($\\alpha_{\\sigma_Y}$)',
                'Deviation of prior belief $\\alpha_{\\mu_Y}$ from $\\mu_A$ in units of SD ($\\alpha_{\\sigma_A}$)']
    for scan, x_lims in zip(scans, x_lim_arrays):
        for var_name, x_label, mean_of_var, std_of_var, x_lim, y_lim in zip(var_names, x_labels, mean_of_vars, std_of_vars, x_lims, y_lims):
            plot_CATE_for_var(var=var_name, std_of_x=std_of_var,
                              mean_of_x=mean_of_var, x_label=x_label,
                              file_name=file_name, x_lim=x_lim,
                              y_lim=y_lim, scan=scan)


def plot_all_Y(file_name='./results/results_18_2.csv'):
    """
    Method for plotting all boxplots for outcomes Y
    under treatment selected based on beliefs.

    Args:
        file_name (str): Path and name of result file.
    """
    scans = ['wide', 'focused']
    var_names = ['mu_X', 'N_E_mu', 'mu_Y', 'mu_A']
    mean_of_vars = [80.0, None, 0.0, 2.0]
    std_of_vars = [10.0, None, 0.1, 1.0]  # N_E does not really have a variance.
    keep_rates = [2, 1, 1, 1]
    # labels = ['Prior beliefs on $\\mu_X$', 'Prior beliefs on $N_E$',
    #             'Prior beliefs on $\\mu_Y$', 'Prior beliefs on $\\mu_A$']
    # Limits for wide and focused scan plots:
    x_lim_array = [[(-30, 30), (-.5, 2.5), (-30, 30), (-30, 30)],
                   [(-3, 3), (.7, 1.3), (-3, 3), (-3, 3)]]
    # Forcing sns to use matplotlib styles:
    plt.clf()
    sns.set_theme(style='white')
    sns.set_context(rc=plt.rcParams)
    sns.color_palette('tab10', desat=1.0)

    results = read_outcomes_from_file(file_name=file_name)
    for scan, x_lims in zip(scans, x_lim_array):
        for key, mean_x, std_x, keep_rate, xlim in zip(var_names, mean_of_vars, std_of_vars, keep_rates, x_lims):
            # Select only results that falls precisely on a predefined grid
            # to keep graph legible:
            tmp_res = match_to_grid(results[key], key, scan_type=scan)
            # Plot
            plot_prior_and_posterior_Y(summary=tmp_res, variable_name=key,
                                             mean_x=mean_x, std_x=std_x,
                                             xlim=xlim, scan=scan, keep_rate=keep_rate)


def match_to_grid(summary, mod_var_name, scan_type):
    """
    Function to subset result data to contain only items that
    match some grid. This is needed for the boxplot that does
    not treat the x-axis as continuous.
    """
    # Parameters used for standardization of units in plots
    scm_noise_parameters = {'alpha_mu_X': 80, 'alpha_sigma_X': 10,
                            'alpha_mu_A': 2, 'alpha_sigma_A': 1.0,
                            'alpha_mu_Y': 0.0, 'alpha_sigma_Y': 0.1,
                            'N_E': 1.0}

    deviations = [i for i in range(-30, 31)]  # 31 because not inclusive of upper bound.
    if scan_type == 'wide':
        params = {
            'mu_A': [scm_noise_parameters['alpha_mu_A'] + deviation * scm_noise_parameters['alpha_sigma_A'] for deviation in deviations],
            'mu_X': [scm_noise_parameters['alpha_mu_X'] + deviation * scm_noise_parameters['alpha_sigma_X'] for deviation in deviations],
            'mu_Y': [scm_noise_parameters['alpha_mu_Y'] + deviation * scm_noise_parameters['alpha_sigma_Y'] for deviation in deviations],
            'N_E_mu': [scm_noise_parameters['N_E'] + deviation * 0.05 for deviation in deviations]
            }
    elif scan_type == 'focused':
        params = {
            'mu_A': [scm_noise_parameters['alpha_mu_A'] + deviation / 10 * scm_noise_parameters['alpha_sigma_A'] for deviation in deviations],
            'mu_X': [scm_noise_parameters['alpha_mu_X'] + deviation / 10 * scm_noise_parameters['alpha_sigma_X'] for deviation in deviations],
            'mu_Y': [scm_noise_parameters['alpha_mu_Y'] + deviation / 10 * scm_noise_parameters['alpha_sigma_Y'] for deviation in deviations],
            'N_E_mu': [scm_noise_parameters['N_E'] + deviation / 10 * 0.05 for deviation in deviations]
            }
    # Summary stats and outputs both have a leading column with the x-value (prior setting)
    grid = params[mod_var_name]
    tmp = {}
    for key in summary.keys():
        tmp[key] = []
        for i, param_value in enumerate(summary['parameter_values']):
            if param_value in grid:
                tmp[key].append(summary[key][i])
    return tmp


def plot_prior_and_posterior_Y(summary, variable_name, mean_x, std_x=None,
                               xlim=(None, None), scan=None, keep_rate=None):
    """
    Function for creating boxplots
    
    :param summary: Description
    :param variable_name: Description
    mean_x (float):  Mean of x (for normalization)
    std_x (float): Standard deviation of x (for normalization)
    scan (str): 'wide' or 'focused'
    """
    # Would need to drop boxes so that we at most have 30 boxes of
    # both groups.
    # We also need to reduce the amount of ticks
    # The horizontal axis should be changed to units of SD
    # Maybe pick a range so that we can keep a fixed fraction
    # like 1/2, 1/4, 1/8 etc to keep the scale constant.

    # Clip range of plot (from original)?
    # Same range as for belief-plots? Would enable comparison side by side.

    categories = summary['parameter_values']
    # Center and normalize (if possible):
    if std_x is not None:
        categories = [(item - mean_x) / std_x for item in categories]
    # Select which ones to include based on range in units of SD
    if xlim[0] is not None:
        keep_idx = [True if xlim[0] <= val <= xlim[1] else False for val in categories]
        # Further drop observations when necessary
        keep_idx = [True if keep & (i%keep_rate==0) else False for i, keep in enumerate(keep_idx)]
    else:
        keep_idx = [True if (i%keep_rate==0) else False for i, _ in enumerate(categories)]
    # Pick selected categories
    categories = [item for item, keep in zip(categories, keep_idx) if keep]
    #categories = [item for i, item in enumerate(categories) if i%keep_rate==0]

    prior_values = summary['posterior_value_lists']
    #prior_values = [item for i, item in enumerate(prior_values) if i%keep_rate==0]
    prior_values = [item for item, keep in zip(prior_values, keep_idx) if keep]
    posterior_values = summary['prior_value_lists']
    #posterior_values = [item for i, item in enumerate(posterior_values) if i%keep_rate==0]
    posterior_values = [item for item, keep in zip(posterior_values, keep_idx) if keep]
    #prior_value_list = [item for i, item in enumerate(summary['prior_value_lists']) if i%keep_rate==0]
    prior_value_list = [item for item, keep in zip(summary['prior_value_lists'], keep_idx) if keep]
    #posterior_value_list = [item for i, item in enumerate(summary['posterior_value_lists']) if i%keep_rate==0]
    posterior_value_list = [item for item, keep in zip(summary['posterior_value_lists'], keep_idx) if keep]
    values = prior_value_list + posterior_value_list
    #values = summary['prior_value_lists'] + summary['posterior_value_lists']
    #values = [item for i, item in enumerate(values) if i%inverse_drop_rate==0]


    prior_categories = [[category] * len(prior_values[i]) for i, category in enumerate(categories)]
    posterior_categories = [[category] * len(posterior_values[i]) for i, category in enumerate(categories)]
    categories = prior_categories + posterior_categories
    categories = [item for row in categories for item in row]
    prior_series = [['Without ML-DS'] * len(val) for val in prior_values]
    prior_series = [item for row in prior_series for item in row]
    posterior_series = [['With ML-DS'] * len(val) for val in posterior_values]
    posterior_series = [item for row in posterior_series for item in row]
    series = prior_series + posterior_series
    values = [item for row in values for item in row]

    data = {'Prior belief': categories,  # x-value
            'Outcomes': series,  # "Prior predictive" etc
            'Y': values}  # 
    # Format ticks:
    x_ticks = np.unique(categories)
    if variable_name=='N_E_mu':
        # Two decimals in ticks for both N_E plots:
        x_ticks = ['{:.2f}'.format(val) if i%5==0 else '' for i, val in enumerate(x_ticks)]
        # x_ticks = [str(val) if i%5==0 else '' for i, val in enumerate(x_ticks)]
    elif scan == 'wide':
        # No decimals in ticks:
        x_ticks = ['{:.0f}'.format(val) if i%5==0 else '' for i, val in enumerate(x_ticks)]
    elif variable_name=='mu_A' and scan == 'focused':
        x_ticks = ['{:.1f}'.format(val) if i%5==0 else '' for i, val in enumerate(x_ticks)]
    elif variable_name == 'mu_Y' and scan == 'focused':
        x_ticks = ['{:.1f}'.format(val) if i%5==0 else '' for i, val in enumerate(x_ticks)]
        
    else:
        x_ticks = [str(val) if i%5==0 else '' for i, val in enumerate(x_ticks)]

    sns.boxplot(data=data, x='Prior belief', y='Y', hue='Outcomes', saturation=1.0)
    ax = plt.gca()
    ax.set_xticklabels(x_ticks)
    #plt.tight_layout()
    file_name = './fig/{}_{}_Y_boxplot.pdf'.format(scan, variable_name)
    plt.savefig(file_name)
    plt.clf()


def plot_all(file_name='./results/results_18_2.csv'):
    """
    Function that reads in all results from ./results/results.csv and
    creates boxplots for all variables found in column "Modified variables".
    Note that the plotting methods are custom made for specific plots
    and might need tweaking with new data (e.g. plot ranges).
    """
    print("Creating line- and boxplots.")
    print("Existing files are overwritten.")
    # Create all CATE-plots:
    plot_CATEs(file_name)
    # Create all boxplots:
    plot_all_Y(file_name)
