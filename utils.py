"""
Functions for writing results to file, for reading
results from said file, and for plotting results.
"""
import ast
import csv
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns


# Single source of truth for the results schema: (dict key, display header)
# pairs, in column order. Both the CSV header row and the row-writing key
# order are derived from this list, so they can't drift apart.
RESULT_COLUMNS = [
    ('experiment_id', 'Experiment ID'),
    ('time_stamp', 'Time Stamp'),
    ('X_o', 'X_o'),
    ('pred_o', 'pred_o'),
    ('modified_prior', 'Modified prior'),
    ('modified_prior_value', 'New value of modified prior'),
    ('threshold', 'Threshold'),
    ('true_CATE_mean', 'True CATE (sample mean)'),
    ('prior_CATE_mean', 'Prior CATE (sample mean)'),
    ('posterior_CATE_mean', 'Posterior CATE (sample mean)'),
    ('true_CATE_std', 'True CATE Standard Deviation (sample estimate)'),
    ('prior_CATE_std', 'Prior CATE Standard Deviation (sample estimate)'),
    ('posterior_CATE_std', 'Posterior CATE Standard Deviation (sample estimate)'),
    ('true_CATE_mu', 'True CATE (mu)'),
    ('prior_CATE_mu', 'Prior CATE (mu)'),
    ('prior_treatment', 'prior_treatment'),
    ('posterior_treatment', 'posterior_treatment'),
    ('true_best_treatment', 'true_best_treatment'),
    ('prior_decision_correct', 'prior_decision_correct'),
    ('posterior_decision_correct', 'posterior_decision_correct'),
    ('decision_changed', 'decision_changed'),
    ('frac_correct_prior_decision', 'frac_correct_prior_decision'),
    ('frac_correct_posterior_decision', 'frac_correct_posterior_decision'),
    ('frac_correct_true_best_treatment', 'frac_correct_true_best_treatment'),
    ('Y_mean_prior_treatment', 'Y_mean_prior_treatment'),
    ('Y_mean_posterior_treatment', 'Y_mean_posterior_treatment'),
    ('Y_mean_true_best_treatment', 'Y_mean_true_best_treatment'),
    ('potential_outcome_A1_mean', 'potential_outcome_A1_mean'),
    ('potential_outcome_A1_std', 'potential_outcome_A1_std'),
    ('potential_outcome_A0_mean', 'potential_outcome_A0_mean'),
    ('potential_outcome_A0_std', 'potential_outcome_A0_std'),
    ('plate_size', 'Plate Size'),
    ('A1_treatment', 'A1_treatment'),
    ('A0_treatment', 'A0_treatment'),
    ('agent_priors', 'Agent Priors'),
    ('scm_noise_parameters', 'SCM Noise Parameters'),
    ('scm_constants', 'SCM Constants'),
    ('running_time', 'Running Time'),
    # Design-specific provenance is part of the shared, versioned schema.
    # Drivers that do not provide these fields retain a missing value.
    ('result_schema_version', 'result_schema_version'),
    ('experiment_design', 'experiment_design'),
    ('experimental_unit', 'experimental_unit'),
    ('replicate_id', 'replicate_id'),
    ('patient_id', 'patient_id'),
    ('data_seed', 'data_seed'),
    ('sampler_seed', 'sampler_seed'),
    ('sampler_stream_key', 'sampler_stream_key'),
    ('observation_seed_scheme', 'observation_seed_scheme'),
    ('fit_intercept', 'fit_intercept'),
    ('source_row', 'source_row'),
    ('historical_data_sha256', 'historical_data_sha256'),
    ('prediction_model_intercept', 'prediction_model_intercept'),
    ('max_rhat', 'max(R-hat)'),
    ('min_ess_bulk', 'min(ESS), bulk'),
    ('min_ess_tail', 'min(ESS), tail'),
    ('n_divergences', 'Divergences'),
    ('pri_X_mean', 'pri_X_mean'),
    ('pri_A_mean', 'pri_A_mean'),
    ('pri_Y_mean', 'pri_Y_mean'),
    ('pri_X_std', 'pri_X_std'),
    ('pri_A_std', 'pri_A_std'),
    ('pri_Y_std', 'pri_Y_std'),
    ('pri_X_skew', 'pri_X_skew'),
    ('pri_A_skew', 'pri_A_skew'),
    ('pri_Y_skew', 'pri_Y_skew'),
    ('pri_X_kurtosis', 'pri_X_kurtosis'),
    ('pri_A_kurtosis', 'pri_A_kurtosis'),
    ('pri_Y_kurtosis', 'pri_Y_kurtosis'),
    ('post_X_mean', 'post_X_mean'),
    ('post_A_mean', 'post_A_mean'),
    ('post_Y_mean', 'post_Y_mean'),
    ('post_X_std', 'post_X_std'),
    ('post_A_std', 'post_A_std'),
    ('post_Y_std', 'post_Y_std'),
    ('post_X_skew', 'post_X_skew'),
    ('post_A_skew', 'post_A_skew'),
    ('post_Y_skew', 'post_Y_skew'),
    ('post_X_kurtosis', 'post_X_kurtosis'),
    ('post_A_kurtosis', 'post_A_kurtosis'),
    ('post_Y_kurtosis', 'post_Y_kurtosis'),
    ('expected_prior_outcome_A1_mean', 'expected_prior_outcome_A1_mean'),
    ('expected_prior_outcome_A1_std', 'expected_prior_outcome_A1_std'),
    ('expected_prior_outcome_A0_mean', 'expected_prior_outcome_A0_mean'),
    ('expected_prior_outcome_A0_std', 'expected_prior_outcome_A0_std'),
    ('expected_posterior_outcome_A1_mean', 'expected_posterior_outcome_A1_mean'),
    ('expected_posterior_outcome_A1_std', 'expected_posterior_outcome_A1_std'),
    ('expected_posterior_outcome_A0_mean', 'expected_posterior_outcome_A0_mean'),
    ('expected_posterior_outcome_A0_std', 'expected_posterior_outcome_A0_std'),
    # Posterior parameter summaries (mean/std/skew/kurtosis of trace.posterior)
    # Scalar parameters
    ('param_N_E_mean',          'param_N_E_mean'),
    ('param_N_E_std',           'param_N_E_std'),
    ('param_N_E_skew',          'param_N_E_skew'),
    ('param_N_E_kurtosis',      'param_N_E_kurtosis'),
    ('posterior_negative_sign_probability', 'Posterior P(N_E < 0)'),
    ('param_alpha_mu_A_mean',   'param_alpha_mu_A_mean'),
    ('param_alpha_mu_A_std',    'param_alpha_mu_A_std'),
    ('param_alpha_mu_A_skew',   'param_alpha_mu_A_skew'),
    ('param_alpha_mu_A_kurtosis', 'param_alpha_mu_A_kurtosis'),
    ('param_alpha_sigma_A_mean',  'param_alpha_sigma_A_mean'),
    ('param_alpha_sigma_A_std',   'param_alpha_sigma_A_std'),
    ('param_alpha_sigma_A_skew',  'param_alpha_sigma_A_skew'),
    ('param_alpha_sigma_A_kurtosis', 'param_alpha_sigma_A_kurtosis'),
    ('param_alpha_mu_Y_mean',   'param_alpha_mu_Y_mean'),
    ('param_alpha_mu_Y_std',    'param_alpha_mu_Y_std'),
    ('param_alpha_mu_Y_skew',   'param_alpha_mu_Y_skew'),
    ('param_alpha_mu_Y_kurtosis', 'param_alpha_mu_Y_kurtosis'),
    ('param_alpha_sigma_Y_mean',  'param_alpha_sigma_Y_mean'),
    ('param_alpha_sigma_Y_std',   'param_alpha_sigma_Y_std'),
    ('param_alpha_sigma_Y_skew',  'param_alpha_sigma_Y_skew'),
    ('param_alpha_sigma_Y_kurtosis', 'param_alpha_sigma_Y_kurtosis'),
    # Vector parameters, k=1 (no dimension index)
    ('param_alpha_mu_X_mean',   'param_alpha_mu_X_mean'),
    ('param_alpha_mu_X_std',    'param_alpha_mu_X_std'),
    ('param_alpha_mu_X_skew',   'param_alpha_mu_X_skew'),
    ('param_alpha_mu_X_kurtosis', 'param_alpha_mu_X_kurtosis'),
    ('param_alpha_sigma_X_mean',  'param_alpha_sigma_X_mean'),
    ('param_alpha_sigma_X_std',   'param_alpha_sigma_X_std'),
    ('param_alpha_sigma_X_skew',  'param_alpha_sigma_X_skew'),
    ('param_alpha_sigma_X_kurtosis', 'param_alpha_sigma_X_kurtosis'),
    ('param_N_b_mean',   'param_N_b_mean'),
    ('param_N_b_std',    'param_N_b_std'),
    ('param_N_b_skew',   'param_N_b_skew'),
    ('param_N_b_kurtosis', 'param_N_b_kurtosis'),
    ('param_N_d_mean',   'param_N_d_mean'),
    ('param_N_d_std',    'param_N_d_std'),
    ('param_N_d_skew',   'param_N_d_skew'),
    ('param_N_d_kurtosis', 'param_N_d_kurtosis'),
    # Sampler settings make each result row self-describing.
    ('n_draws',       'n_draws'),
    ('n_tune',        'n_tune'),
    ('n_chains',      'n_chains'),
    ('target_accept', 'target_accept'),
]


def write_results_to_file(result_dict, file_name='results.csv', path='./results/'):
    """
    Function for writing results to file.

    Existing files must have exactly the same header as the row being
    appended. A schema mismatch raises ``ValueError`` instead of producing a
    malformed CSV.

    Args:
        result_dict [dict]: Dict of results to store.
        file_name (str): Name of file to write results to.
        path (str): Path where to store result file.
    """
    # Check for existence of folder. Create if it does not exist.
    Path(path).mkdir(parents=True, exist_ok=True)
    full_path = path + file_name

    # Callers with multivariate (k > 1) predictive summaries (e.g.
    # pri_X_0_mean, pri_X_1_mean, ...) pass extra keys not covered by
    # RESULT_COLUMNS. Append them as dynamic columns, using the key itself
    # as the header text since there's no pretty name for them. Assumes a
    # given results file is only ever appended to with a consistent key
    # set (e.g. constant n_covariates).
    fixed_keys = {key for key, _ in RESULT_COLUMNS}
    extra_keys = sorted(key for key in result_dict if key not in fixed_keys)
    columns = RESULT_COLUMNS + [(key, key) for key in extra_keys]
    fieldnames = [key for key, _ in columns]

    row = {key: result_dict.get(key, float('nan')) for key in fieldnames}

    file_is_new = not Path(full_path).exists()
    if not file_is_new:
        with open(full_path, newline='') as existing_file:
            existing_header = next(csv.reader(existing_file), None)
        expected_header = [display_name for _, display_name in columns]
        if existing_header != expected_header:
            raise ValueError(
                "existing results file has a different schema: {}".format(
                    full_path))
    with open(full_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL,
                                lineterminator='\n')
        if file_is_new:
            writer.writerow(dict(columns))  # Header row, using display names.
        writer.writerow(row)
    print("Results saved to {}".format(full_path))


def load_results(file_name='./results/results.csv'):
    """
    Load the results CSV and group rows by modified prior name.

    Each row in the CSV represents one experiment run. The returned dict
    maps each unique value of the 'Modified prior' column to a DataFrame
    containing all runs for that prior, sorted by 'New value of modified
    prior'.

    Args:
        file_name (str): Path and file name of the results CSV.

    Returns:
        dict[str, pd.DataFrame]: One DataFrame per modified prior name.
    """
    df = pd.read_csv(file_name)
    return {
        name: group.sort_values('New value of modified prior').reset_index(drop=True)
        for name, group in df.groupby('Modified prior')
    }


def plot_CATE_for_var(var, mean_of_x, std_of_x, x_label, file_name, x_lim, y_lim, scan):
    """
    Args:
        var (str): Name of modified prior variable.
        mean_of_x (float): Mean of horizontal scale (x) to center on.
        std_of_x (float): Standard deviation of the horizontal scale (x).
            Used for normalizing plot.
        x_label (str): Label that will be visible under the x-axis.
        file_name (str): File name and path of the result file.
        x_lim ((float, float)): +-Limits of horizontal axis.
        y_lim ((float, float)): +-Limits of vertical axis.
    """
    fields = ['Prior CATE (sample mean)',
              'Posterior CATE (sample mean)',
              'True CATE (sample mean)']
    label_names = ['Without ML-DS', 'With ML-DS', 'True value']
    threshold = 5
    n_observations = 4000

    df = load_results(file_name)[var]
    summary = df.groupby('New value of modified prior')[fields].agg(['mean', 'std'])

    if std_of_x is not None:
        x = (summary.index.values - mean_of_x) / std_of_x
    else:
        x = summary.index.values

    fig, ax = plt.subplots()
    for field, label_name in zip(fields, label_names):
        y = summary[(field, 'mean')].values
        # Std of means * sqrt(n) converts to std of observations.
        yerr = np.sqrt(n_observations) * summary[(field, 'std')].values
        ax.plot(x, y)
        ax.fill_between(x, y + yerr, y - yerr, alpha=0.2, label=label_name)
        if field == 'Posterior CATE (sample mean)':
            y_post, yerr_post = y, yerr
        elif field == 'True CATE (sample mean)':
            y_true = y

    mask = (y_true > y_post - yerr_post) & (y_true < y_post + yerr_post)
    ax.fill_between(x, 0, 1, where=mask, transform=ax.get_xaxis_transform(),
                    color='lightgray', alpha=.4)
    ax.axhline(y=threshold, linestyle='--', color='black', label='Treatment threshold')
    if x_lim is not None:
        plt.xlim(x_lim)
        plt.ylim(y_lim)
    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel('CATE')
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

    results = load_results(file_name=file_name)
    for scan, x_lims in zip(scans, x_lim_array):
        for key, mean_x, std_x, keep_rate, xlim in zip(var_names, mean_of_vars, std_of_vars, keep_rates, x_lims):
            df_grid = match_to_grid(results[key], key, scan_type=scan)
            plot_prior_and_posterior_Y(df=df_grid, variable_name=key,
                                       mean_x=mean_x, std_x=std_x,
                                       xlim=xlim, scan=scan, keep_rate=keep_rate)


def _compute_grid(mod_var_name, scan_type):
    """
    Return the predefined list of prior values for the given variable
    and scan type. Used to subset results to a legible set of x-positions
    in the boxplot.
    """
    scm_noise_parameters = {'alpha_mu_X': 80, 'alpha_sigma_X': 10,
                            'alpha_mu_A': 2, 'alpha_sigma_A': 1.0,
                            'alpha_mu_Y': 0.0, 'alpha_sigma_Y': 0.1,
                            'N_E': 1.0}
    deviations = range(-30, 31)
    if scan_type == 'wide':
        params = {
            'mu_A':   [scm_noise_parameters['alpha_mu_A'] + d * scm_noise_parameters['alpha_sigma_A'] for d in deviations],
            'mu_X':   [scm_noise_parameters['alpha_mu_X'] + d * scm_noise_parameters['alpha_sigma_X'] for d in deviations],
            'mu_Y':   [scm_noise_parameters['alpha_mu_Y'] + d * scm_noise_parameters['alpha_sigma_Y'] for d in deviations],
            'N_E_mu': [scm_noise_parameters['N_E'] + d * 0.05 for d in deviations],
        }
    elif scan_type == 'focused':
        params = {
            'mu_A':   [scm_noise_parameters['alpha_mu_A'] + d / 10 * scm_noise_parameters['alpha_sigma_A'] for d in deviations],
            'mu_X':   [scm_noise_parameters['alpha_mu_X'] + d / 10 * scm_noise_parameters['alpha_sigma_X'] for d in deviations],
            'mu_Y':   [scm_noise_parameters['alpha_mu_Y'] + d / 10 * scm_noise_parameters['alpha_sigma_Y'] for d in deviations],
            'N_E_mu': [scm_noise_parameters['N_E'] + d / 10 * 0.05 for d in deviations],
        }
    return params[mod_var_name]


def match_to_grid(df, mod_var_name, scan_type):
    """
    Subset a results DataFrame to rows whose modified-prior value falls
    on the predefined grid for the given scan type. Needed for the
    boxplot, which treats the x-axis as categorical.

    Args:
        df (pd.DataFrame): Results for one modified prior (from load_results).
        mod_var_name (str): Name of the modified prior (e.g. 'mu_X').
        scan_type (str): 'wide' or 'focused'.

    Returns:
        pd.DataFrame: Filtered copy of df.
    """
    grid = _compute_grid(mod_var_name, scan_type)
    return df[df['New value of modified prior'].isin(grid)].copy()


def _select_display_values(df, param_col, max_boxes):
    """Pick up to max_boxes evenly-spaced unique x-values from df."""
    unique_vals = sorted(df[param_col].unique())
    step = max(1, len(unique_vals) // max_boxes)
    return unique_vals[::step]


def _box_style(color):
    """Consistent matplotlib boxplot style for one series."""
    return dict(
        patch_artist=True,
        medianprops=dict(color='black', linewidth=1.5),
        boxprops=dict(facecolor=color, alpha=0.7),
        whiskerprops=dict(color=color),
        capprops=dict(color=color),
        flierprops=dict(marker='o', markerfacecolor=color, markersize=3, alpha=0.5),
    )


def _draw_cate_panel(ax, df, param_col, colors, threshold, n_observations,
                     y_lim_cate=None):
    """Draw the CATE line plot onto ax. Returns x_all (sorted unique x values)."""
    cate_fields = ['Prior CATE (sample mean)', 'Posterior CATE (sample mean)']
    cate_labels = ['Without ML-DS', 'With ML-DS']
    summary = df.groupby(param_col)[cate_fields].agg(['mean', 'std'])
    x_all = summary.index.values
    y_post = yerr_post = None

    for i, (field, label) in enumerate(zip(cate_fields, cate_labels)):
        y = summary[(field, 'mean')].values
        yerr = np.sqrt(n_observations) * summary[(field, 'std')].values
        ax.plot(x_all, y, color=colors[i], label=label)
        ax.fill_between(x_all, y - yerr, y + yerr, alpha=0.2, color=colors[i])
        if field == 'Posterior CATE (sample mean)':
            y_post, yerr_post = y, yerr

    true_cate_mean = df['True CATE (sample mean)'].mean()
    ax.axhline(y=true_cate_mean, color='black', linestyle=(0, (8, 12)), label='True value')

    if y_post is not None:
        mask = (true_cate_mean > y_post - yerr_post) & (true_cate_mean < y_post + yerr_post)
        ax.fill_between(x_all, 0, 1, where=mask,
                        transform=ax.get_xaxis_transform(),
                        color='lightgray', alpha=0.4)
    ax.axhline(y=threshold, linestyle='--', color='black', label='Treatment threshold')
    if y_lim_cate is not None:
        ax.set_ylim(y_lim_cate)
    return x_all


def _draw_outcome_panel(ax, df, param_col, x_all, colors):
    """Draw the E[Y|do(A)] line plot onto ax."""
    outcome_series = [
        ('expected_prior_outcome_A0_mean',     'expected_prior_outcome_A0_std',     colors[0], '-',  'Without ML-DS (A=0)'),
        ('expected_posterior_outcome_A0_mean', 'expected_posterior_outcome_A0_std', colors[1], '-',  'With ML-DS (A=0)'),
        ('expected_prior_outcome_A1_mean',     'expected_prior_outcome_A1_std',     colors[0], '--', 'Without ML-DS (A=1)'),
        ('expected_posterior_outcome_A1_mean', 'expected_posterior_outcome_A1_std', colors[1], '--', 'With ML-DS (A=1)'),
    ]
    for mean_col, std_col, color, ls, label in outcome_series:
        y = df.groupby(param_col)[mean_col].mean().values
        yerr = df.groupby(param_col)[std_col].mean().values
        ax.plot(x_all, y, color=color, linestyle=ls, label=label)
        ax.fill_between(x_all, y - yerr, y + yerr, alpha=0.2, color=color)
    true_a0_mean = df['potential_outcome_A0_mean'].mean()
    true_a1_mean = df['potential_outcome_A1_mean'].mean()
    ax.axhline(y=true_a0_mean, color='black', linestyle=(0, (8, 12)), label='True value (A=0)')
    ax.axhline(y=true_a1_mean, color='black', linestyle=(0, (4, 6)),  label='True value (A=1)')


def _draw_boxplot_panel(ax, df, param_col, max_boxes, prior_color, posterior_color,
                        x_lim=None):
    """Draw the Y outcome boxplot onto ax."""
    display_vals = _select_display_values(df, param_col, max_boxes)
    df_box = df[df[param_col].isin(display_vals)]
    step = display_vals[1] - display_vals[0] if len(display_vals) > 1 else 1.0
    offset = step * 0.2
    width = step * 0.3
    prior_data = [df_box[df_box[param_col] == v]['Y_mean_prior_treatment'].values
                  for v in display_vals]
    posterior_data = [df_box[df_box[param_col] == v]['Y_mean_posterior_treatment'].values
                      for v in display_vals]
    ax.boxplot(prior_data,
               positions=[v - offset for v in display_vals],
               widths=width, **_box_style(prior_color))
    ax.boxplot(posterior_data,
               positions=[v + offset for v in display_vals],
               widths=width, **_box_style(posterior_color))
    if x_lim is not None:
        ax.set_xlim(x_lim)
    label_every = max(1, len(display_vals) // 10)
    ax.set_xticks(display_vals)
    ax.set_xticklabels(
        ['{:.3g}'.format(v) if i % label_every == 0 else ''
         for i, v in enumerate(display_vals)],
        rotation=0, ha='center')


def plot_combined_for_var(df, variable_name, mean_x, std_x, x_label,
                          x_lim=None, y_lim_cate=None, max_boxes=30,
                          threshold=5, scan='', n_observations=4000):
    """
    Combined CATE line plot (top), E[Y|do(A)] line plot (middle), and Y outcome
    boxplot (bottom) with a shared continuous x-axis.

    Args:
        df (pd.DataFrame): Results for one modified prior (from load_results).
        variable_name (str): Name of the modified prior, used in the output
            file name.
        mean_x (float): Center of the x-axis (for normalization). Pass None
            to skip normalization.
        std_x (float): Scale of the x-axis (for normalization). Pass None
            to skip normalization.
        x_label (str): Label for the shared x-axis.
        x_lim ((float, float)): Horizontal axis limits after normalization.
        y_lim_cate ((float, float)): Vertical axis limits for the CATE panel.
        max_boxes (int): Maximum number of x-positions in the boxplot.
        threshold (float): CATE threshold line drawn in the CATE panel.
        scan (str): Scan label used in the output file name.
        n_observations (int): Multiplier converting std-of-means to
            std-of-observations for the CATE uncertainty band.
    """
    param_col = 'New value of modified prior'
    df = df.copy()
    if std_x is not None:
        df[param_col] = (df[param_col] - mean_x) / std_x

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    prior_color, posterior_color = colors[0], colors[1]

    if x_lim is not None:
        effective_xlim = x_lim
    else:
        x_min = df[param_col].min()
        x_max = df[param_col].max()
        margin = (x_max - x_min) * 0.03
        effective_xlim = (x_min - margin, x_max + margin)

    fig, (ax_top, ax_mid, ax_bot) = plt.subplots(
        3, 1, sharex=True, figsize=(12, 11),
        gridspec_kw={'height_ratios': [1, 1, 1]})
    fig.subplots_adjust(hspace=0.05)

    x_all = _draw_cate_panel(ax_mid, df, param_col, colors, threshold,
                              n_observations, y_lim_cate)
    ax_mid.set_ylabel('CATE')
    ax_mid.legend(loc='best')

    _draw_outcome_panel(ax_top, df, param_col, x_all, colors)
    ax_top.set_ylabel('E[Y | do(A)]')
    ax_top.legend(loc='best')

    _draw_boxplot_panel(ax_bot, df, param_col, max_boxes, prior_color,
                        posterior_color, effective_xlim)
    ax_bot.legend(handles=[
        mpatches.Patch(facecolor=prior_color, alpha=0.7, label='Without ML-DS'),
        mpatches.Patch(facecolor=posterior_color, alpha=0.7, label='With ML-DS'),
    ])
    ax_bot.set_ylabel('Y outcome')
    ax_bot.set_xlabel(x_label)

    # Set xlim once; sharex=True propagates it to the other two panels.
    ax_top.set_xlim(effective_xlim)

    plt.savefig('./fig/{}_{}_combined.pdf'.format(scan, variable_name),
                bbox_inches='tight')
    plt.clf()


def plot_combined_grid(file_name='./results/results_18_2.csv', scan='wide',
                       threshold=5, n_observations=4000, max_boxes=30):
    """
    Create a 3×3 grid figure for mu_X, mu_A, and N_E.

    Columns correspond to the three variables; rows correspond to the CATE
    panel, the E[Y|do(A)] panel, and the Y outcome boxplot. Each row shares
    a common y-axis scale across all three variable columns.

    Args:
        file_name (str): Path to the results CSV file.
        scan (str): 'wide' or 'focused'; selects x-axis limits.
        threshold (float): CATE threshold line value.
        n_observations (int): Multiplier for CATE uncertainty band.
        max_boxes (int): Maximum number of x-positions in the boxplot.
    """
    var_names  = ['mu_X', 'mu_A', 'N_E_mu']
    mean_of_vars = [80.0,  2.0,   None]
    std_of_vars  = [10.0,  1.0,   None]
    x_lims_by_scan = {
        'wide':    [(-30, 30), (-30, 30), (-0.5, 2.5)],
        'focused': [(-3, 3),   (-3, 3),   None],
    }
    col_titles = [
        r'$\alpha_{\mu_X}$',
        r'$\alpha_{\mu_A}$',
        r'$N_E$',
    ]
    x_labels = [
        r'Deviation of $\alpha_{\mu_X}$ from $\mu_X$ in units of SD',
        r'Deviation of $\alpha_{\mu_A}$ from $\mu_A$ in units of SD',
        r'Prior belief on $N_E$',
    ]

    x_lims = x_lims_by_scan.get(scan, x_lims_by_scan['wide'])
    param_col = 'New value of modified prior'
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    prior_color, posterior_color = colors[0], colors[1]

    results = load_results(file_name)

    fig, axes = plt.subplots(3, 3, sharex='col', sharey='row', figsize=(16, 11))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    for col, (var, mean_x, std_x, x_lim, col_title, x_label) in enumerate(
            zip(var_names, mean_of_vars, std_of_vars,
                x_lims, col_titles, x_labels)):
        df = results[var].copy()
        if std_x is not None:
            df[param_col] = (df[param_col] - mean_x) / std_x

        ax_out  = axes[0, col]
        ax_cate = axes[1, col]
        ax_bot  = axes[2, col]
        show_legend = (col == 2)

        if x_lim is not None:
            effective_xlim = x_lim
        else:
            x_min = df[param_col].min()
            x_max = df[param_col].max()
            margin = (x_max - x_min) * 0.03
            effective_xlim = (x_min - margin, x_max + margin)

        x_all = _draw_cate_panel(ax_cate, df, param_col, colors, threshold,
                                  n_observations)
        if show_legend:
            ax_cate.legend(loc='best', fontsize='small')

        _draw_outcome_panel(ax_out, df, param_col, x_all, colors)
        ax_out.set_title(col_title)
        if show_legend:
            ax_out.legend(loc='best', fontsize='small')

        _draw_boxplot_panel(ax_bot, df, param_col, max_boxes,
                            prior_color, posterior_color, effective_xlim)
        if show_legend:
            ax_bot.legend(handles=[
                mpatches.Patch(facecolor=prior_color,     alpha=0.7, label='Without ML-DS'),
                mpatches.Patch(facecolor=posterior_color, alpha=0.7, label='With ML-DS'),
            ], fontsize='small')
        ax_bot.set_xlabel(x_label)

        # Set xlim once; sharex='col' propagates it to the other two panels.
        ax_out.set_xlim(effective_xlim)

    axes[0, 0].set_ylabel('E[Y | do(A)]')
    axes[1, 0].set_ylabel('CATE')
    axes[2, 0].set_ylabel('Y outcome')

    plt.savefig('./fig/{}_grid_combined.pdf'.format(scan), bbox_inches='tight')
    plt.clf()


def plot_all_combined(file_name='./results/results_18_2.csv', scan='wide'):
    """
    Create combined figures for all modified priors for one scan type.

    Args:
        file_name (str): Path and name of result file.
        scan (str): 'wide' (default) or 'focused'.
    """
    var_names = ['mu_X', 'N_E_mu', 'mu_Y', 'mu_A']
    mean_of_vars = [80.0, None, 0.0, 2.0]
    std_of_vars = [10.0, None, 0.1, 1.0]
    x_lims_by_scan = {
        'wide':    [(-30, 30), (-0.5, 2.5), (-30, 30), (-30, 30)],
        'focused': [(-3, 3),   None,  (-3, 3),   (-3, 3)],
    }
    y_lim_cates = [(None, None), (-30, 50), (-15, 35), (None, None)]
    x_labels = [
        r'Deviation of prior belief $\alpha_{\mu_X}$ from $\mu_X$ in units of SD ($\alpha_{\sigma_X}$)',
        r'Prior belief on $N_E$',
        r'Deviation of prior belief $\alpha_{\mu_Y}$ from $\mu_Y$ in units of SD ($\alpha_{\sigma_Y}$)',
        r'Deviation of prior belief $\alpha_{\mu_A}$ from $\mu_A$ in units of SD ($\alpha_{\sigma_A}$)',
    ]

    x_lims = x_lims_by_scan.get(scan, x_lims_by_scan['wide'])
    results = load_results(file_name)
    for var, mean_x, std_x, x_lim, y_lim_cate, x_label in zip(
            var_names, mean_of_vars, std_of_vars,
            x_lims, y_lim_cates, x_labels):
        plot_combined_for_var(
            df=results[var],
            variable_name=var,
            mean_x=mean_x,
            std_x=std_x,
            x_label=x_label,
            x_lim=x_lim,
            y_lim_cate=y_lim_cate if y_lim_cate != (None, None) else None,
            scan=scan,
        )


def plot_cate_change_for_var(df, variable_name, mean_x, std_x, x_label,
                             scan='', misalignment_sds=(-5, 0, 5)):
    """
    Plot ΔCATE (posterior − prior) vs X_o for three levels of prior misalignment.

    Selects the rows whose "New value of modified prior" is closest to
    mean_x + sd * std_x for each sd in misalignment_sds (~50 rows per level,
    covering the range of X_o values).

    Args:
        df (pd.DataFrame): Results for one modified prior (from load_results).
        variable_name (str): Name of the modified prior.
        mean_x (float): True (aligned) value of the prior parameter.
        std_x (float): 1 SD unit for this parameter.
        x_label (str): X-axis label.
        scan (str): Label used in the output file name.
        misalignment_sds (tuple): SD offsets to plot (default −1, 0, +1).
    """
    param_col = 'New value of modified prior'
    available_vals = sorted(df[param_col].unique())

    sd_labels = {-5: r'$-5$ SD', 0: 'Aligned', 5: r'$+5$ SD'}
    selected = {
        sd: min(available_vals, key=lambda v: abs(v - (mean_x + sd * std_x)))
        for sd in misalignment_sds
    }

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, sharex=True, figsize=(10, 8),
        gridspec_kw={'height_ratios': [1, 1]})
    fig.subplots_adjust(hspace=0.05)

    for i, sd in enumerate(misalignment_sds):
        val = selected[sd]
        subset = df[df[param_col] == val].sort_values('X_o')
        x = subset['X_o'].values
        prior_cate = subset['Prior CATE (sample mean)'].values
        post_cate = subset['Posterior CATE (sample mean)'].values
        color = colors[i]
        label = sd_labels.get(sd, '{:+d} SD'.format(sd))

        ax_top.plot(x, post_cate - prior_cate, color=color, label=label)
        ax_bot.plot(x, post_cate,  color=color, linestyle='-',
                    label='Posterior ({})'.format(label))
        ax_bot.plot(x, prior_cate, color=color, linestyle='--',
                    label='Prior ({})'.format(label))

    ax_top.axhline(y=0, color='black', linestyle=':', linewidth=0.8)
    ax_top.set_ylabel('Posterior CATE − Prior CATE')
    ax_top.legend(loc='best')

    ax_bot.set_ylabel('CATE')
    ax_bot.set_xlabel(x_label)
    ax_bot.legend(loc='best')

    plt.savefig('./fig/{}_{}_cate_change.pdf'.format(scan, variable_name),
                bbox_inches='tight')
    plt.clf()


def plot_all_cate_change(file_name='./results/results.csv', scan=''):
    """
    Create ΔCATE vs X_o plots for all modified prior variables.

    Args:
        file_name (str): Path to the results CSV file.
        scan (str): Label used in output file names.
    """
    var_names    = ['mu_X', 'mu_Y', 'mu_A', 'N_E_mu']
    mean_of_vars = [80.0,    0.0,    2.0,   1.0]
    std_of_vars  = [10.0,    0.1,    1.0,   1.0]
    x_labels = [
        r'Observation $X^o$',
        r'Observation $X^o$',
        r'Observation $X^o$',
        r'Observation $X^o$'
    ]

    results = load_results(file_name)
    for var, mean_x, std_x, x_label in zip(
            var_names, mean_of_vars, std_of_vars, x_labels):
        plot_cate_change_for_var(
            df=results[var],
            variable_name=var,
            mean_x=mean_x,
            std_x=std_x,
            x_label=x_label,
            scan=scan,
        )


def plot_prior_and_posterior_Y(df, variable_name, mean_x, std_x=None,
                               xlim=(None, None), scan=None, keep_rate=None):
    """
    Create side-by-side boxplots of prior and posterior Y outcomes.

    Args:
        df (pd.DataFrame): Results for one modified prior (from match_to_grid).
        variable_name (str): Name of the modified prior (e.g. 'mu_X').
        mean_x (float): Mean of x for normalization.
        std_x (float): Standard deviation of x for normalization.
        xlim ((float, float)): +-Limits of horizontal axis after normalization.
        scan (str): 'wide' or 'focused'.
        keep_rate (int): Keep every nth x-value to reduce clutter.
    """
    param_col = 'New value of modified prior'
    df = df.copy()

    if std_x is not None:
        df[param_col] = (df[param_col] - mean_x) / std_x

    if xlim[0] is not None:
        df = df[(df[param_col] >= xlim[0]) & (df[param_col] <= xlim[1])]

    if keep_rate is not None and keep_rate > 1:
        unique_vals = sorted(df[param_col].unique())
        keep_vals = {v for i, v in enumerate(unique_vals) if i % keep_rate == 0}
        df = df[df[param_col].isin(keep_vals)]

    melted = df.melt(
        id_vars=param_col,
        value_vars=['Y_mean_prior_treatment', 'Y_mean_posterior_treatment'],
        var_name='Outcomes',
        value_name='Y',
    )
    melted['Outcomes'] = melted['Outcomes'].map({
        'Y_mean_prior_treatment': 'Without ML-DS',
        'Y_mean_posterior_treatment': 'With ML-DS',
    })
    melted = melted.rename(columns={param_col: 'Prior belief'})

    x_ticks = np.unique(melted['Prior belief'].values)
    if variable_name == 'N_E_mu':
        x_tick_labels = ['{:.2f}'.format(v) if i % 5 == 0 else '' for i, v in enumerate(x_ticks)]
    elif scan == 'wide':
        x_tick_labels = ['{:.0f}'.format(v) if i % 5 == 0 else '' for i, v in enumerate(x_ticks)]
    elif variable_name in ('mu_A', 'mu_Y') and scan == 'focused':
        x_tick_labels = ['{:.1f}'.format(v) if i % 5 == 0 else '' for i, v in enumerate(x_ticks)]
    else:
        x_tick_labels = [str(v) if i % 5 == 0 else '' for i, v in enumerate(x_ticks)]

    sns.boxplot(data=melted, x='Prior belief', y='Y', hue='Outcomes', saturation=1.0)
    plt.gca().set_xticklabels(x_tick_labels)
    plt.savefig('./fig/{}_{}_Y_boxplot.pdf'.format(scan, variable_name))
    plt.clf()


def plot_all(file_name='./results/results_18_2.csv'):
    """
    Create all plots from the results file: separate CATE line plots,
    separate Y boxplots, and combined figures.

    Args:
        file_name (str): Path and name of result file.
    """
    print("Creating plots. Existing files are overwritten.")
    # plot_CATEs(file_name)
    # plot_all_Y(file_name)
    # Plot 3x1 graphs with E(Y|do(A)), CATE, and outcomes for every variable in the data:
    plot_all_combined(file_name)
    # Plot 3x3 graph with all of the above for X, A, N_E to include in paper.
    plot_combined_grid(file_name, scan='wide')
    # Plot effect of X_o on size and magnitude of posterior belief in CATE.
    # Value of X_o seems to have no effect at all, probably due to strong priors.
    plot_all_cate_change()


def _parse_dimensionality(x_o_value):
    parsed = ast.literal_eval(str(x_o_value))
    return len(parsed) if isinstance(parsed, list) else 1


_PRIOR_REFERENCE = {
    'mu_A':   (2.0,  1.0),
    'mu_X':   (80.0, 10.0),
    'mu_Y':   (0.0,  0.1),
    'N_E_mu': (1.0,  0.05),
}


def _to_std_label(prior_name, value):
    ref, sigma = _PRIOR_REFERENCE.get(prior_name, (0.0, 1.0))
    n_std = round((value - ref) / sigma)
    return '{:+d}σ'.format(n_std)


_MARKERS_BY_ABS_RANK = ['o', 's', '^', 'D', 'X']


def _line_emphasis(abs_n_std, sorted_unique_abs):
    """Return (linewidth, alpha, zorder) based on rank of abs_n_std (smallest = most emphasis)."""
    rank  = sorted_unique_abs.index(abs_n_std)
    n     = max(len(sorted_unique_abs) - 1, 1)
    t     = rank / n                        # 0 = most central, 1 = most extreme
    lw    = 2.5 - 1.8 * t                  # 2.5 → 0.7
    alpha = 1.0 - 0.7 * t                  # 1.0 → 0.3
    return lw, alpha, n - rank


def plot_dimensionality(file_name='./results/results_dimensionality.csv'):
    prior_col   = 'Modified prior'
    value_col   = 'New value of modified prior'
    outcome_col = 'Posterior CATE (sample mean)'
    dim_col     = 'dimensionality'

    df = pd.read_csv(file_name)
    df[dim_col] = df['X_o'].apply(_parse_dimensionality)

    true_cate = df['True CATE (sample mean)'].mean()

    for prior_name, prior_df in df.groupby(prior_col):
        grouped  = prior_df.groupby([dim_col, value_col])
        agg_mean = grouped[outcome_col].mean().unstack(level=value_col)

        dim_values   = sorted(agg_mean.index.tolist())
        prior_values = sorted(agg_mean.columns.tolist())

        ref, sigma = _PRIOR_REFERENCE.get(prior_name, (0.0, 1.0))
        abs_stds   = sorted({abs(round((v - ref) / sigma)) for v in prior_values})

        fig, ax = plt.subplots()
        for pval in prior_values:
            if abs(round((pval - ref) / sigma)) == 30:
                continue
            n_std     = round((pval - ref) / sigma)
            abs_n_std = abs(n_std)
            abs_rank  = abs_stds.index(abs_n_std)

            color  = 'black' if n_std == 0 else ('#D6604D' if n_std > 0 else '#2166AC')
            marker = _MARKERS_BY_ABS_RANK[abs_rank % len(_MARKERS_BY_ABS_RANK)]
            lw, alpha, zo = _line_emphasis(abs_n_std, abs_stds)

            y    = agg_mean[pval].values
            ax.plot(dim_values, y,
                    color=color, marker=marker, linewidth=lw,
                    alpha=alpha, zorder=zo, label=_to_std_label(prior_name, pval))

        ax.axhline(true_cate, color='gray', linewidth=0.8,
                   linestyle=(0, (6, 6)), label='True CATE')

        ax.set_xticks(dim_values)
        ax.set_xticklabels([str(d) for d in dim_values])
        ax.set_xlabel('X dimensionality (k)')
        ax.set_ylabel('Posterior CATE (sample mean)')
        ax.set_title('Effect of dimensionality on posterior CATE — {}'.format(prior_name))
        ax.legend(title='Misalignment of prior mean', loc='best')

        plt.savefig('./fig/dim_{}.pdf'.format(prior_name), bbox_inches='tight')
        plt.clf()


def plot_dimensionality_improvement(file_name='./results/results_dimensionality.csv'):
    prior_col = 'Modified prior'
    value_col = 'New value of modified prior'
    dim_col   = 'dimensionality'

    df = pd.read_csv(file_name)
    df[dim_col]  = df['X_o'].apply(_parse_dimensionality)
    true_cate    = df['True CATE (sample mean)'].mean()

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for prior_name, prior_df in df.groupby(prior_col):
        ref, sigma = _PRIOR_REFERENCE.get(prior_name, (0.0, 1.0))

        grouped      = prior_df.groupby([dim_col, value_col])
        agg_prior    = grouped['Prior CATE (sample mean)'].mean().unstack(level=value_col)
        agg_posterior= grouped['Posterior CATE (sample mean)'].mean().unstack(level=value_col)

        dim_values   = sorted(agg_prior.index.tolist())
        prior_values = sorted(
            [v for v in agg_prior.columns if abs(round((v - ref) / sigma)) != 30],
            key=lambda v: round((v - ref) / sigma)
        )

        n_plots = len(prior_values)
        ncols   = 2
        nrows   = (n_plots + 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(10, 3 * nrows),
                                 sharex=True, sharey=True)

        for idx, pval in enumerate(prior_values):
            ax    = axes.flat[idx]
            label = _to_std_label(prior_name, pval)

            ax.plot(dim_values, agg_prior[pval].values,
                    color=colors[0], linestyle='--', marker='o',
                    label='Prior' if idx == 0 else '_nolegend_')
            ax.plot(dim_values, agg_posterior[pval].values,
                    color=colors[1], linestyle='-', marker='o',
                    label='Posterior' if idx == 0 else '_nolegend_')
            ax.axhline(true_cate, color='gray', linewidth=0.8,
                       linestyle=(0, (6, 6)),
                       label='True CATE' if idx == 0 else '_nolegend_')

            ax.set_title(label)
            ax.set_xticks(dim_values)
            ax.set_xticklabels([str(d) for d in dim_values])

        for idx in range(n_plots, nrows * ncols):
            axes.flat[idx].set_visible(False)

        for ax in axes[:, 0]:
            ax.set_ylabel('CATE (sample mean)')
        for ax in axes[-1, :]:
            ax.set_xlabel('X dimensionality (k)')

        fig.suptitle('CATE estimation error vs dimensionality — {}'.format(prior_name))
        fig.legend(*axes.flat[0].get_legend_handles_labels(),
                   loc='lower right', bbox_to_anchor=(0.98, 0.02))

        plt.savefig('./fig/dim_improvement_{}.pdf'.format(prior_name), bbox_inches='tight')
        plt.clf()
