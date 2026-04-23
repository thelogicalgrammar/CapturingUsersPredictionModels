"""
Run one multivariate scalability experiment.

This script is intentionally separate from experiment.py. The original
experiment pipeline is kept as the paper-facing baseline, while this
driver focuses on:

- full-plate and reduced plate runs
- OLS with and without intercept
- reproducible baseline parameterization across k
- runtime measurement
- posterior diagnostics logging for later aggregation
"""
import argparse
import csv
import json
import time
from pathlib import Path

import arviz as az
import numpy as np
from sklearn.linear_model import LinearRegression

import agent_model as am
import scm


scm_noise_parameters_default = {
    'alpha_mu_X': 80.0,
    'alpha_sigma_X': 10.0,
    'alpha_mu_A': 2.0,
    'alpha_sigma_A': 1.0,
    'alpha_mu_Y': 0.0,
    'alpha_sigma_Y': 0.1,
    'N_E': 1.0,
}
scm_constants_default = {'a': 12.0, 'b': -0.1, 'd': 0.125}
agent_priors_default = {
    'mu_X': 80.0,
    'sigma_X': 0.1,
    'mu_sigma_X': 10.0,
    'sigma_sigma_X': 0.1,
    'mu_A': 2.0,
    'sigma_A': 1.0,
    'mu_sigma_A': 1.0,
    'sigma_sigma_A': 0.1,
    'mu_Y': 0.0,
    'sigma_Y': 0.01,
    'mu_sigma_Y': 0.1,
    'sigma_sigma_Y': 0.01,
    'N_E_mu': 1.0,
    'N_E_sigma': 1.0,
}
A1_treatment = 20.0
A0_treatment = 10.0
threshold = 5.0
model_variants = (
    'full_no_intercept',
    'reduced_no_intercept',
    'full_intercept',
    'reduced_intercept',
)


def format_features_for_regression(X):
    """
    Convert scalar and multivariate feature arrays to the two-dimensional
    representation expected by scikit-learn.
    """
    X = np.asarray(X)
    if X.ndim == 1:
        return X.reshape(-1, 1)
    if X.ndim == 2:
        return X
    raise ValueError("X must be one- or two-dimensional.")


def format_observation_for_regression(X_o):
    """
    Convert one observed patient feature vector to the two-dimensional shape
    expected by scikit-learn prediction methods.
    """
    X_o = np.asarray(X_o)
    if X_o.ndim == 0:
        return X_o.reshape(-1, 1)
    if X_o.ndim == 1:
        return X_o.reshape(1, -1)
    raise ValueError("X_o must be scalar or one-dimensional.")


def serialize_value(value):
    """
    Convert numpy values to JSON- and CSV-friendly Python values.
    """
    if isinstance(value, dict):
        return {key: serialize_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def serialize_row(row):
    """
    Convert all values in one dictionary row.
    """
    return {key: serialize_value(value) for key, value in row.items()}


def scale_covariate_coefficients(value, n_covariates, coefficient_scaling):
    """
    Scale one covariate coefficient across k variables.
    """
    if coefficient_scaling == 'inverse_k':
        scaled_value = value / n_covariates
    elif coefficient_scaling == 'none':
        scaled_value = value
    else:
        raise ValueError(f"Unknown coefficient scaling '{coefficient_scaling}'.")
    coefficients = np.full(n_covariates, scaled_value, dtype=float)
    if n_covariates == 1:
        return coefficients.item()
    return coefficients


def repeat_covariate_parameter(value, n_covariates):
    """
    Repeat a scalar covariate parameter across k variables.
    """
    values = np.full(n_covariates, value, dtype=float)
    if n_covariates == 1:
        return values.item()
    return values


def build_scm_noise_parameters(n_covariates):
    """
    Build the default multivariate noise specification.
    """
    return {
        'alpha_mu_X': repeat_covariate_parameter(scm_noise_parameters_default['alpha_mu_X'], n_covariates),
        'alpha_sigma_X': repeat_covariate_parameter(scm_noise_parameters_default['alpha_sigma_X'], n_covariates),
        'alpha_mu_A': scm_noise_parameters_default['alpha_mu_A'],
        'alpha_sigma_A': scm_noise_parameters_default['alpha_sigma_A'],
        'alpha_mu_Y': scm_noise_parameters_default['alpha_mu_Y'],
        'alpha_sigma_Y': scm_noise_parameters_default['alpha_sigma_Y'],
        'N_E': scm_noise_parameters_default['N_E'],
    }


def build_scm_constants(n_covariates, coefficient_scaling):
    """
    Build the structural constants for one scalability run.
    """
    return {
        'a': scm_constants_default['a'],
        'b': scale_covariate_coefficients(scm_constants_default['b'], n_covariates, coefficient_scaling),
        'd': scale_covariate_coefficients(scm_constants_default['d'], n_covariates, coefficient_scaling),
    }


def build_agent_priors(n_covariates):
    """
    Build the default agent priors for one scalability run.
    """
    return {
        'mu_X': repeat_covariate_parameter(agent_priors_default['mu_X'], n_covariates),
        'sigma_X': repeat_covariate_parameter(agent_priors_default['sigma_X'], n_covariates),
        'mu_sigma_X': repeat_covariate_parameter(agent_priors_default['mu_sigma_X'], n_covariates),
        'sigma_sigma_X': repeat_covariate_parameter(agent_priors_default['sigma_sigma_X'], n_covariates),
        'mu_A': agent_priors_default['mu_A'],
        'sigma_A': agent_priors_default['sigma_A'],
        'mu_sigma_A': agent_priors_default['mu_sigma_A'],
        'sigma_sigma_A': agent_priors_default['sigma_sigma_A'],
        'mu_Y': agent_priors_default['mu_Y'],
        'sigma_Y': agent_priors_default['sigma_Y'],
        'mu_sigma_Y': agent_priors_default['mu_sigma_Y'],
        'sigma_sigma_Y': agent_priors_default['sigma_sigma_Y'],
        'N_E_mu': agent_priors_default['N_E_mu'],
        'N_E_sigma': agent_priors_default['N_E_sigma'],
    }


def build_observed_patient(n_covariates, scm_noise_parameters, x_o_mode, rng):
    """
    Build the observed patient features either from the SCM mean or by sampling.
    """
    if x_o_mode == 'mean':
        X_o = np.asarray(scm_noise_parameters['alpha_mu_X'], dtype=float)
    elif x_o_mode == 'sample':
        X_o = np.asarray(
            rng.normal(loc=scm_noise_parameters['alpha_mu_X'],
                       scale=scm_noise_parameters['alpha_sigma_X'],
                       size=np.atleast_1d(scm_noise_parameters['alpha_mu_X']).shape),
            dtype=float,
        )
    else:
        raise ValueError(f"Unknown X_o mode '{x_o_mode}'.")
    if n_covariates == 1:
        return np.asarray(X_o).reshape(()).item()
    return X_o.reshape(n_covariates)


def model_uses_intercept(model_variant):
    """
    Return whether the external prediction model includes an intercept.
    """
    if model_variant in ('full_intercept', 'reduced_intercept'):
        return True
    if model_variant in ('full_no_intercept', 'reduced_no_intercept'):
        return False
    raise ValueError(f"Unknown model variant '{model_variant}'.")


def model_uses_reduction(model_variant):
    """
    Return whether the agent model uses a reduced plate.
    """
    if model_variant in ('reduced_no_intercept', 'reduced_intercept'):
        return True
    if model_variant in ('full_no_intercept', 'full_intercept'):
        return False
    raise ValueError(f"Unknown model variant '{model_variant}'.")


def build_agent_model(model_variant, agent_priors, scm_constants, plate_size):
    """
    Build the AgentModel subclass for one scalability run.
    """
    if model_variant == 'full_no_intercept':
        return am.AgentModelWithMultivariate(
            prior_parameters=agent_priors,
            scm_constants=scm_constants,
            plate_size=plate_size,
            low_dim_aux_plate=False,
        )
    if model_variant == 'reduced_no_intercept':
        return am.AgentModelWithMultivariate(
            prior_parameters=agent_priors,
            scm_constants=scm_constants,
            plate_size=plate_size,
            low_dim_aux_plate=True,
        )
    if model_variant == 'full_intercept':
        return am.AgentModelWithIntercept(
            prior_parameters=agent_priors,
            scm_constants=scm_constants,
            plate_size=plate_size,
            low_dim_aux_plate=False,
        )
    if model_variant == 'reduced_intercept':
        return am.AgentModelWithIntercept(
            prior_parameters=agent_priors,
            scm_constants=scm_constants,
            plate_size=plate_size,
            low_dim_aux_plate=True,
        )
    raise ValueError(f"Unknown model variant '{model_variant}'.")


def build_trace_var_names(model_variant):
    """
    Return variables to store in scalability traces.

    Full plates contain many latent observations. They are sampled by NUTS but
    do not need to be stored for runtime and convergence diagnostics.
    """
    var_names = [
        'alpha_mu_X', 'alpha_sigma_X', 'alpha_mu_A', 'alpha_sigma_A',
        'alpha_mu_Y', 'alpha_sigma_Y', 'N_E', 'N_b', 'N_d', 'phi',
    ]
    if model_uses_intercept(model_variant):
        var_names.append('alpha')
    return var_names


def write_single_row_csv(row, output_path):
    """
    Write one summary row to CSV.
    """
    row = serialize_row(row)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def write_rows_csv(rows, output_path):
    """
    Write a list of rows to CSV.
    """
    serialized_rows = [serialize_row(row) for row in rows]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(serialized_rows[0].keys()))
        writer.writeheader()
        writer.writerows(serialized_rows)


def summarize_trace_diagnostics(trace):
    """
    Return run-level diagnostics and parameter-level diagnostics.
    """
    diagnostics = az.summary(trace, kind='diagnostics', round_to=None)
    diagnostics = diagnostics.reset_index().rename(columns={'index': 'parameter'})
    diagnostics_rows = diagnostics.to_dict(orient='records')
    diagnostics_summary = {
        'n_divergences': int(trace.sample_stats['diverging'].sum().values),
        'max_rhat': float(diagnostics['r_hat'].max()),
        'min_ess_bulk': float(diagnostics['ess_bulk'].min()),
        'min_ess_tail': float(diagnostics['ess_tail'].min()),
    }
    return diagnostics_summary, diagnostics_rows


def build_output_paths(output_dir, run_name):
    """
    Build output file paths for one run.
    """
    output_dir = Path(output_dir)
    return {
        'summary_json': output_dir / f'{run_name}_summary.json',
        'summary_csv': output_dir / f'{run_name}_summary.csv',
        'diagnostics_csv': output_dir / f'{run_name}_diagnostics.csv',
    }


def run_scalability_experiment(n_covariates, plate_size, draws, tune,
                               chains, target_accept, max_treedepth, init,
                               model_variant, coefficient_scaling, x_o_mode, seed,
                               output_dir, run_name,
                               A1_treatment=A1_treatment,
                               A0_treatment=A0_treatment,
                               threshold=threshold):
    """
    Run a single multivariate scalability experiment.
    """
    total_start_time = time.time()
    rng = np.random.default_rng(seed)

    scm_noise_parameters = build_scm_noise_parameters(n_covariates)
    scm_constants = build_scm_constants(n_covariates, coefficient_scaling)
    agent_priors = build_agent_priors(n_covariates)
    X_o = build_observed_patient(n_covariates, scm_noise_parameters, x_o_mode, rng)

    historical_data_start_time = time.time()
    data_scm = scm.SCM(**scm_noise_parameters, **scm_constants)
    X, A, Y_historical, *_ = data_scm.sample(draws=plate_size, rng=rng)
    historical_data_runtime = time.time() - historical_data_start_time

    prediction_model_start_time = time.time()
    fit_intercept = model_uses_intercept(model_variant)
    pred_model = LinearRegression(fit_intercept=fit_intercept).fit(
        format_features_for_regression(X), Y_historical)
    pred_o = float(pred_model.predict(format_observation_for_regression(X_o))[0])
    prediction_model_runtime = time.time() - prediction_model_start_time

    agent_model = build_agent_model(
        model_variant=model_variant,
        agent_priors=agent_priors,
        scm_constants=scm_constants,
        plate_size=plate_size,
    )

    prior_runtime_start_time = time.time()
    agent_model.sample_prior_predictive(draws=draws, rng=rng)
    prior_CATE_mu = agent_model.get_prior_CATE_mu(A1_treatment, A0_treatment)
    prior_CATE_samples, prior_CATE_mean, prior_CATE_std = agent_model.sample_prior_predictive_CATE(
        X_o, A1_treatment, A0_treatment, rng=rng)
    prior_treatment = A1_treatment if prior_CATE_mean > threshold else A0_treatment
    prior_runtime = time.time() - prior_runtime_start_time

    posterior_sampling_start_time = time.time()
    agent_model.sample(
        X_o=np.asarray(X_o),
        pred_o=np.asarray(pred_o),
        draws=draws,
        tune=tune,
        target_accept=target_accept,
        chains=chains,
        max_treedepth=max_treedepth,
        init=init,
        rng=rng,
        var_names=build_trace_var_names(model_variant),
        progressbar=False,
        compute_convergence_checks=False,
    )
    posterior_sampling_runtime = time.time() - posterior_sampling_start_time

    diagnostics_summary, diagnostics_rows = summarize_trace_diagnostics(agent_model.trace)

    posterior_runtime_start_time = time.time()
    posterior_CATE_samples, posterior_CATE_mean, posterior_CATE_std = (
        agent_model.sample_posterior_predictive_CATE(
            X_o, A1_treatment, A0_treatment, rng=rng)
    )
    posterior_treatment = A1_treatment if posterior_CATE_mean > threshold else A0_treatment
    posterior_runtime = time.time() - posterior_runtime_start_time

    ground_truth_start_time = time.time()
    true_CATE_mu = data_scm.get_CATE_mu(A1_treatment, A0_treatment)
    true_CATE_samples, true_CATE_mean, true_CATE_std = data_scm.sample_CATE(
        X_o, A1_treatment, A0_treatment, draws=draws, rng=rng)
    potential_outcome_A1_samples, potential_outcome_A1_mean, potential_outcome_A1_std = (
        data_scm.sample_do_A(A1_treatment, X_o, draws=draws, rng=rng)
    )
    potential_outcome_A0_samples, potential_outcome_A0_mean, potential_outcome_A0_std = (
        data_scm.sample_do_A(A0_treatment, X_o, draws=draws, rng=rng)
    )
    ground_truth_runtime = time.time() - ground_truth_start_time

    prior_decision_correct = bool(
        np.sign(true_CATE_mean - threshold) == np.sign(prior_CATE_mean - threshold))
    posterior_decision_correct = bool(
        np.sign(true_CATE_mean - threshold) == np.sign(posterior_CATE_mean - threshold))
    true_best_treatment = A1_treatment if np.sign(true_CATE_mean - threshold) > 0 else A0_treatment
    if np.sign(prior_CATE_mean - threshold) != np.sign(posterior_CATE_mean - threshold):
        decision_changed = 1 if np.sign(posterior_CATE_mean - threshold) == np.sign(true_CATE_mean - threshold) else -1
    else:
        decision_changed = 0

    frac_correct_prior_decision = float(np.mean(
        np.sign(prior_CATE_mean - threshold) == np.sign(true_CATE_samples - threshold)))
    frac_correct_posterior_decision = float(np.mean(
        np.sign(posterior_CATE_mean - threshold) == np.sign(true_CATE_samples - threshold)))
    frac_correct_true_best_treatment = float(np.mean(
        np.sign(true_CATE_mean - threshold) == np.sign(true_CATE_samples - threshold)))

    Y_mean_prior_treatment = potential_outcome_A1_mean if np.sign(prior_CATE_mean - threshold) > 0 else potential_outcome_A0_mean
    Y_mean_posterior_treatment = potential_outcome_A1_mean if np.sign(posterior_CATE_mean - threshold) > 0 else potential_outcome_A0_mean
    Y_mean_true_best_treatment = potential_outcome_A1_mean if np.sign(true_CATE_mean - threshold) > 0 else potential_outcome_A0_mean

    total_runtime = time.time() - total_start_time
    summary = {
        'run_name': run_name,
        'seed': seed,
        'model_variant': model_variant,
        'fit_intercept': fit_intercept,
        'uses_reduction': model_uses_reduction(model_variant),
        'n_covariates': n_covariates,
        'plate_size': plate_size,
        'draws': draws,
        'tune': tune,
        'chains': chains,
        'target_accept': target_accept,
        'max_treedepth': max_treedepth,
        'init': init,
        'coefficient_scaling': coefficient_scaling,
        'x_o_mode': x_o_mode,
        'X_o': X_o,
        'pred_o': pred_o,
        'A1_treatment': A1_treatment,
        'A0_treatment': A0_treatment,
        'threshold': threshold,
        'scm_noise_parameters': scm_noise_parameters,
        'scm_constants': scm_constants,
        'agent_priors': agent_priors,
        'true_CATE_mu': true_CATE_mu,
        'true_CATE_mean': true_CATE_mean,
        'true_CATE_std': true_CATE_std,
        'prior_CATE_mu': prior_CATE_mu,
        'prior_CATE_mean': prior_CATE_mean,
        'prior_CATE_std': prior_CATE_std,
        'posterior_CATE_mean': posterior_CATE_mean,
        'posterior_CATE_std': posterior_CATE_std,
        'prior_treatment': prior_treatment,
        'posterior_treatment': posterior_treatment,
        'true_best_treatment': true_best_treatment,
        'prior_decision_correct': prior_decision_correct,
        'posterior_decision_correct': posterior_decision_correct,
        'decision_changed': decision_changed,
        'frac_correct_prior_decision': frac_correct_prior_decision,
        'frac_correct_posterior_decision': frac_correct_posterior_decision,
        'frac_correct_true_best_treatment': frac_correct_true_best_treatment,
        'Y_mean_prior_treatment': Y_mean_prior_treatment,
        'Y_mean_posterior_treatment': Y_mean_posterior_treatment,
        'Y_mean_true_best_treatment': Y_mean_true_best_treatment,
        'potential_outcome_A1_mean': potential_outcome_A1_mean,
        'potential_outcome_A1_std': potential_outcome_A1_std,
        'potential_outcome_A0_mean': potential_outcome_A0_mean,
        'potential_outcome_A0_std': potential_outcome_A0_std,
        'historical_data_runtime': historical_data_runtime,
        'prediction_model_runtime': prediction_model_runtime,
        'prior_runtime': prior_runtime,
        'posterior_sampling_runtime': posterior_sampling_runtime,
        'posterior_runtime': posterior_runtime,
        'ground_truth_runtime': ground_truth_runtime,
        'total_runtime': total_runtime,
    }
    summary.update(diagnostics_summary)

    for row in diagnostics_rows:
        row['run_name'] = run_name
        row['model_variant'] = model_variant
        row['n_covariates'] = n_covariates
        row['plate_size'] = plate_size
        row['draws'] = draws
        row['tune'] = tune
        row['chains'] = chains
        row['seed'] = seed

    output_paths = build_output_paths(output_dir, run_name)
    output_paths['summary_json'].parent.mkdir(parents=True, exist_ok=True)
    with output_paths['summary_json'].open('w') as file:
        json.dump(serialize_row(summary), file, indent=2)
    write_single_row_csv(summary, output_paths['summary_csv'])
    write_rows_csv(diagnostics_rows, output_paths['diagnostics_csv'])

    return summary, diagnostics_rows, output_paths


def parse_args():
    """
    Parse command line arguments for one scalability run.
    """
    parser = argparse.ArgumentParser(
        description="Run one multivariate scalability experiment."
    )
    parser.add_argument('--n-covariates', type=int, required=True)
    parser.add_argument('--plate-size', type=int, required=True)
    parser.add_argument('--draws', type=int, default=1000)
    parser.add_argument('--tune', type=int, default=1000)
    parser.add_argument('--chains', type=int, default=4)
    parser.add_argument('--target-accept', type=float, default=.85)
    parser.add_argument('--max-treedepth', type=int, default=15)
    parser.add_argument('--init', type=str, default='jitter+adapt_diag')
    parser.add_argument('--model-variant', choices=model_variants,
                        default='full_no_intercept')
    parser.add_argument('--coefficient-scaling', choices=('inverse_k', 'none'),
                        default='inverse_k')
    parser.add_argument('--x-o-mode', choices=('mean', 'sample'), default='mean')
    parser.add_argument('--seed', type=int, default=4)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--run-name', type=str, default=None)
    return parser.parse_args()


def main():
    """
    Parse CLI arguments and run one experiment.
    """
    args = parse_args()
    run_name = args.run_name
    if run_name is None:
        time_stamp = time.strftime('%Y%m%d_%H%M%S')
        run_name = (
            f"{args.model_variant}_k{args.n_covariates}_plate{args.plate_size}_"
            f"seed{args.seed}_{time_stamp}"
        )
    summary, diagnostics_rows, output_paths = run_scalability_experiment(
        n_covariates=args.n_covariates,
        plate_size=args.plate_size,
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        target_accept=args.target_accept,
        max_treedepth=args.max_treedepth,
        init=args.init,
        model_variant=args.model_variant,
        coefficient_scaling=args.coefficient_scaling,
        x_o_mode=args.x_o_mode,
        seed=args.seed,
        output_dir=args.output_dir,
        run_name=run_name,
    )
    print(f"Run completed: {summary['run_name']}")
    print(f"Summary JSON: {output_paths['summary_json']}")
    print(f"Summary CSV: {output_paths['summary_csv']}")
    print(f"Diagnostics CSV: {output_paths['diagnostics_csv']}")
    print(
        "Model: "
        f"variant={summary['model_variant']}, "
        f"fit_intercept={summary['fit_intercept']}, "
        f"uses_reduction={summary['uses_reduction']}"
    )
    print(
        "Diagnostics: "
        f"divergences={summary['n_divergences']}, "
        f"max_rhat={summary['max_rhat']:.4f}, "
        f"min_ess_bulk={summary['min_ess_bulk']:.2f}, "
        f"min_ess_tail={summary['min_ess_tail']:.2f}"
    )
    print(f"Stored {len(diagnostics_rows)} parameter diagnostics.")


if __name__ == '__main__':
    main()
