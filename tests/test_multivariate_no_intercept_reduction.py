"""
Validation of the multivariate reduced plate model without intercept.

The reduction is compared against a brute-force plate simulation where
the full plate is generated from the SCM and OLS is fit without intercept.
For k = 1, the condensed formula is also compared against the historical
eight-variable scalar no-intercept reduction.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import solve_triangular
from scipy.stats import ks_2samp, truncnorm, wishart
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scm
import tests.statistical_tests_for_similarity as sts


def make_prior_parameters(k):
    """
    Create priors with the same general scale as the codebase defaults.
    """
    if k == 1:
        return {
            'mu_X': 80.0, 'sigma_X': 0.1,
            'mu_sigma_X': 10.0, 'sigma_sigma_X': 0.1,
            'mu_A': 2.0, 'sigma_A': 1.0,
            'mu_sigma_A': 1.0, 'sigma_sigma_A': 0.1,
            'mu_Y': 12.0, 'sigma_Y': 0.01,
            'mu_sigma_Y': 0.1, 'sigma_sigma_Y': 0.01,
            'N_E_mu': 1.0, 'N_E_sigma': 1.0,
            'N_b_mu': -0.1, 'N_b_sigma': 0.1,
            'N_d_mu': 0.125, 'N_d_sigma': 0.1,
        }
    N_b_mu = np.linspace(-0.1, 0.1, k)
    N_d_mu = np.linspace(0.125, -0.05, k)
    return {
        'mu_X': np.linspace(80.0, 80.0 + 5.0 * (k - 1), k),
        'sigma_X': np.full(k, 0.1),
        'mu_sigma_X': np.linspace(10.0, 10.0 + (k - 1), k),
        'sigma_sigma_X': np.full(k, 0.1),
        'mu_A': 2.0, 'sigma_A': 1.0,
        'mu_sigma_A': 1.0, 'sigma_sigma_A': 0.1,
        'mu_Y': 12.0, 'sigma_Y': 0.01,
        'mu_sigma_Y': 0.1, 'sigma_sigma_Y': 0.01,
        'N_E_mu': 1.0, 'N_E_sigma': 1.0,
        'N_b_mu': N_b_mu, 'N_b_sigma': np.full(k, 0.1),
        'N_d_mu': N_d_mu, 'N_d_sigma': np.full(k, 0.1),
    }


def make_scm_constants(k):
    """
    Create legacy SCM constants with non-trivial multivariate effects.
    """
    if k == 1:
        return {'a': 0.0, 'b': -0.1, 'd': 0.125}
    return {
        'a': 0.0,
        'b': np.linspace(-0.1, 0.1, k),
        'd': np.linspace(0.125, -0.05, k),
    }


def summarize_samples(samples):
    """
    Summaries printed and stored for each variable comparison.
    """
    quantiles = np.quantile(samples, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        'mean': float(np.mean(samples)),
        'std': float(np.std(samples)),
        'q05': float(quantiles[0]),
        'q25': float(quantiles[1]),
        'q50': float(quantiles[2]),
        'q75': float(quantiles[3]),
        'q95': float(quantiles[4]),
    }


def compare_samples(reference_samples, reduced_samples, ks_pvalue_threshold=0.001):
    """
    Compare two one-dimensional samples.
    """
    ks_result = ks_2samp(reference_samples, reduced_samples)
    wasserstein_result = sts.normalized_wasserstein_distance(
        reference_samples, reduced_samples, max_diff=0.1
    )
    success = bool(
        ks_result.pvalue > ks_pvalue_threshold
        and wasserstein_result['success']
    )
    reference_summary = summarize_samples(reference_samples)
    reduced_summary = summarize_samples(reduced_samples)
    return {
        'success': success,
        'reference_summary': reference_summary,
        'reduced_summary': reduced_summary,
        'mean_difference': float(reference_summary['mean'] - reduced_summary['mean']),
        'std_difference': float(reference_summary['std'] - reduced_summary['std']),
        'ks_statistic': float(ks_result.statistic),
        'ks_pvalue': float(ks_result.pvalue),
        'ks_pvalue_threshold': float(ks_pvalue_threshold),
        'wasserstein_test': wasserstein_result,
    }


def draw_positive_samples(mu, sigma, rng, size):
    """
    Sample from the truncated normal priors used for scale parameters.
    """
    lower = (0 - mu) / sigma
    upper = np.full_like(lower, np.inf, dtype=float)
    return truncnorm.rvs(lower, upper, loc=mu, scale=sigma, size=size, random_state=rng)


def draw_parameter_samples(k, repetitions, rng):
    """
    Draw the population-level parameters from the agent priors.
    """
    rng = np.random.default_rng(rng)
    prior_parameters = make_prior_parameters(k)
    parameter_samples = {}
    if k == 1:
        parameter_samples['alpha_mu_X'] = rng.normal(
            prior_parameters['mu_X'], prior_parameters['sigma_X'], size=repetitions
        )
        parameter_samples['alpha_sigma_X'] = draw_positive_samples(
            prior_parameters['mu_sigma_X'], prior_parameters['sigma_sigma_X'], rng, repetitions
        )
    else:
        parameter_samples['alpha_mu_X'] = rng.normal(
            prior_parameters['mu_X'], prior_parameters['sigma_X'], size=(repetitions, k)
        )
        parameter_samples['alpha_sigma_X'] = np.column_stack([
            draw_positive_samples(
                prior_parameters['mu_sigma_X'][j],
                prior_parameters['sigma_sigma_X'][j],
                rng,
                repetitions,
            )
            for j in range(k)
        ])
    parameter_samples['alpha_mu_A'] = rng.normal(
        prior_parameters['mu_A'], prior_parameters['sigma_A'], size=repetitions
    )
    parameter_samples['alpha_sigma_A'] = draw_positive_samples(
        prior_parameters['mu_sigma_A'], prior_parameters['sigma_sigma_A'], rng, repetitions
    )
    parameter_samples['alpha_mu_Y'] = rng.normal(
        prior_parameters['mu_Y'], prior_parameters['sigma_Y'], size=repetitions
    )
    parameter_samples['alpha_sigma_Y'] = draw_positive_samples(
        prior_parameters['mu_sigma_Y'], prior_parameters['sigma_sigma_Y'], rng, repetitions
    )
    parameter_samples['N_E'] = rng.normal(
        prior_parameters['N_E_mu'], prior_parameters['N_E_sigma'], size=repetitions
    )
    if k == 1:
        parameter_samples['N_b'] = rng.normal(
            prior_parameters['N_b_mu'], prior_parameters['N_b_sigma'], size=repetitions
        )
        parameter_samples['N_d'] = rng.normal(
            prior_parameters['N_d_mu'], prior_parameters['N_d_sigma'], size=repetitions
        )
    else:
        parameter_samples['N_b'] = rng.normal(
            prior_parameters['N_b_mu'], prior_parameters['N_b_sigma'], size=(repetitions, k)
        )
        parameter_samples['N_d'] = rng.normal(
            prior_parameters['N_d_mu'], prior_parameters['N_d_sigma'], size=(repetitions, k)
        )
    return parameter_samples


def draw_reduced_samples(k, plate_size, repetitions, rng):
    """
    Draw from the reduced no-intercept formula directly.
    """
    rng = np.random.default_rng(rng)
    parameter_samples = draw_parameter_samples(k, repetitions, rng)
    if k == 1:
        s = rng.normal(0, np.sqrt(plate_size), size=repetitions)
        W = rng.chisquare(plate_size - 1, size=repetitions)
        g = rng.normal(0, 1, size=repetitions)
        S_x = plate_size * parameter_samples['alpha_mu_X'] + parameter_samples['alpha_sigma_X'] * s
        G = parameter_samples['alpha_sigma_X']**2 * W + S_x**2 / plate_size
        beta = parameter_samples['N_b'] + parameter_samples['N_E'] * parameter_samples['N_d']
        c = parameter_samples['N_E'] * parameter_samples['alpha_mu_A'] + parameter_samples['alpha_mu_Y']
        tau = np.sqrt(parameter_samples['N_E']**2 * parameter_samples['alpha_sigma_A']**2 +
                      parameter_samples['alpha_sigma_Y']**2)
        phi_samples = beta + c * S_x / G + tau * g / np.sqrt(G)
        return phi_samples, parameter_samples

    s = rng.normal(0, np.sqrt(plate_size), size=(repetitions, k))
    W = wishart.rvs(df=plate_size - 1, scale=np.eye(k), size=repetitions, random_state=rng)
    g = rng.normal(0, 1, size=(repetitions, k))
    beta = parameter_samples['N_b'] + parameter_samples['N_E'][:, None] * parameter_samples['N_d']
    c = parameter_samples['N_E'] * parameter_samples['alpha_mu_A'] + parameter_samples['alpha_mu_Y']
    tau = np.sqrt(parameter_samples['N_E']**2 * parameter_samples['alpha_sigma_A']**2 +
                  parameter_samples['alpha_sigma_Y']**2)
    phi_samples = np.zeros_like(g)
    for i in range(repetitions):
        D_X = np.diag(parameter_samples['alpha_sigma_X'][i])
        S_x = plate_size * parameter_samples['alpha_mu_X'][i] + D_X @ s[i]
        G = D_X @ W[i] @ D_X + np.outer(S_x, S_x) / plate_size
        G_chol = np.linalg.cholesky(G)
        mean_term = c[i] * np.linalg.solve(G, S_x)
        noise_term = tau[i] * solve_triangular(G_chol.T, g[i], lower=False)
        phi_samples[i] = beta[i] + mean_term + noise_term
    return phi_samples, parameter_samples


def smoke_test_agent_model(k, plate_size, seed):
    """
    Check that the actual AgentModel path builds and samples.
    """
    import pymc as pm
    import agent_model as am

    model = am.AgentModelWithMultivariate(
        plate_size=plate_size,
        prior_parameters=make_prior_parameters(k),
        scm_constants=make_scm_constants(k),
        low_dim_aux_plate=True,
    )
    with model.model:
        samples = pm.sample_prior_predictive(
            draws=3,
            var_names=['phi'],
            random_seed=seed,
        )
    phi_samples = samples.prior['phi'].values.reshape((-1,) + samples.prior['phi'].values.shape[2:])
    if k == 1:
        assert phi_samples.shape == (3,)
    else:
        assert phi_samples.shape == (3, k)

    X_o = 80.0 if k == 1 else make_prior_parameters(k)['mu_X']
    prior_samples = model.sample_prior_predictive(draws=3, rng=seed)
    prior_do, prior_do_mean, prior_do_std = model.sample_prior_predictive_do_A(
        X_o, 20, rng=seed)
    prior_summary = model.sample_prior_predictive_summaries(rng=seed)
    assert len(prior_samples) == 3
    assert prior_do.shape == (3,)
    assert np.isfinite(prior_do_mean)
    assert np.isfinite(prior_do_std)
    assert len(prior_summary) == 12


def draw_full_plate_samples(k, plate_size, parameter_samples, rng):
    """
    Draw brute-force full plates and fit OLS without intercept.
    """
    rng = np.random.default_rng(rng)
    scm_constants = make_scm_constants(k)
    phi_samples = np.zeros((parameter_samples['N_E'].shape[0], k))

    for i in range(parameter_samples['N_E'].shape[0]):
        current_scm = scm.SCM(
            alpha_mu_X=parameter_samples['alpha_mu_X'][i],
            alpha_sigma_X=parameter_samples['alpha_sigma_X'][i],
            alpha_mu_A=parameter_samples['alpha_mu_A'][i],
            alpha_sigma_A=parameter_samples['alpha_sigma_A'][i],
            alpha_mu_Y=parameter_samples['alpha_mu_Y'][i],
            alpha_sigma_Y=parameter_samples['alpha_sigma_Y'][i],
            N_E=parameter_samples['N_E'][i],
            a=scm_constants['a'],
            b=parameter_samples['N_b'][i],
            d=parameter_samples['N_d'][i],
        )
        X, A, Y, *_ = current_scm.sample(draws=plate_size, rng=rng)
        if k == 1:
            X_for_fit = X.reshape(-1, 1)
        else:
            X_for_fit = X
        reg = LinearRegression(fit_intercept=False).fit(X_for_fit, Y)
        phi_samples[i] = np.asarray(reg.coef_, dtype=float).reshape(-1)
    if k == 1:
        return phi_samples[:, 0]
    return phi_samples


def draw_historical_scalar_no_intercept_reduction_samples(
        plate_size, repetitions, rng):
    """
    Draw phi using the historical eight-variable scalar no-intercept reduction.
    """
    rng = np.random.default_rng(rng)
    prior_parameters = make_prior_parameters(1)
    alpha_mu_X = rng.normal(prior_parameters['mu_X'], prior_parameters['sigma_X'], repetitions)
    alpha_sigma_X = draw_positive_samples(
        prior_parameters['mu_sigma_X'], prior_parameters['sigma_sigma_X'], rng, repetitions
    )
    alpha_mu_A = rng.normal(prior_parameters['mu_A'], prior_parameters['sigma_A'], repetitions)
    alpha_sigma_A = draw_positive_samples(
        prior_parameters['mu_sigma_A'], prior_parameters['sigma_sigma_A'], rng, repetitions
    )
    alpha_mu_Y = rng.normal(prior_parameters['mu_Y'], prior_parameters['sigma_Y'], repetitions)
    alpha_sigma_Y = draw_positive_samples(
        prior_parameters['mu_sigma_Y'], prior_parameters['sigma_sigma_Y'], rng, repetitions
    )
    N_E = rng.normal(prior_parameters['N_E_mu'], prior_parameters['N_E_sigma'], repetitions)

    S_X = rng.normal(0, np.sqrt(plate_size), repetitions)
    S_Y = rng.normal(0, np.sqrt(plate_size), repetitions)
    S_A = rng.normal(0, np.sqrt(plate_size), repetitions)
    Z_XX = rng.chisquare(plate_size - 1, repetitions)
    U_XY = rng.chisquare(plate_size - 1, repetitions)
    V_XY = rng.chisquare(plate_size - 1, repetitions)
    U_XA = rng.chisquare(plate_size - 1, repetitions)
    V_XA = rng.chisquare(plate_size - 1, repetitions)

    S_9 = S_X * S_A / plate_size + (U_XA - V_XA) / 2
    S_8 = S_X * S_Y / plate_size + (U_XY - V_XY) / 2
    S_7 = Z_XX + S_X**2 / plate_size
    S_6 = (plate_size * alpha_mu_X * alpha_mu_Y +
           alpha_mu_X * alpha_sigma_Y * S_Y +
           alpha_mu_Y * alpha_sigma_X * S_X +
           alpha_sigma_X * alpha_sigma_Y * S_8)
    S_5 = N_E * (plate_size * alpha_mu_X * alpha_mu_A +
                 alpha_mu_X * alpha_sigma_A * S_A +
                 alpha_mu_A * alpha_sigma_X * S_X +
                 alpha_sigma_X * alpha_sigma_A * S_9)
    S_4 = 0.0
    S_123 = (plate_size * alpha_mu_X**2 +
             2 * alpha_mu_X * alpha_sigma_X * S_X +
             alpha_sigma_X**2 * S_7)
    return -0.1 + 0.125 * N_E + (S_4 + S_5 + S_6) / S_123


def draw_condensed_scalar_constant_samples(plate_size, repetitions, rng):
    """
    Draw phi using the k=1 condensed no-intercept formula with fixed b and d.
    """
    rng = np.random.default_rng(rng)
    prior_parameters = make_prior_parameters(1)
    alpha_mu_X = rng.normal(prior_parameters['mu_X'], prior_parameters['sigma_X'], repetitions)
    alpha_sigma_X = draw_positive_samples(
        prior_parameters['mu_sigma_X'], prior_parameters['sigma_sigma_X'], rng, repetitions
    )
    alpha_mu_A = rng.normal(prior_parameters['mu_A'], prior_parameters['sigma_A'], repetitions)
    alpha_sigma_A = draw_positive_samples(
        prior_parameters['mu_sigma_A'], prior_parameters['sigma_sigma_A'], rng, repetitions
    )
    alpha_mu_Y = rng.normal(prior_parameters['mu_Y'], prior_parameters['sigma_Y'], repetitions)
    alpha_sigma_Y = draw_positive_samples(
        prior_parameters['mu_sigma_Y'], prior_parameters['sigma_sigma_Y'], rng, repetitions
    )
    N_E = rng.normal(prior_parameters['N_E_mu'], prior_parameters['N_E_sigma'], repetitions)

    s = rng.normal(0, np.sqrt(plate_size), repetitions)
    W = rng.chisquare(plate_size - 1, repetitions)
    g = rng.normal(0, 1, repetitions)
    S_x = plate_size * alpha_mu_X + alpha_sigma_X * s
    G = alpha_sigma_X**2 * W + S_x**2 / plate_size
    beta = -0.1 + 0.125 * N_E
    c = N_E * alpha_mu_A + alpha_mu_Y
    tau = np.sqrt(N_E**2 * alpha_sigma_A**2 + alpha_sigma_Y**2)
    return beta + c * S_x / G + tau * g / np.sqrt(G)


def validate_scalar_against_historical_no_intercept_reduction(
        plate_size, repetitions, seed):
    """
    Validate that the k=1 formula matches the historical scalar no-intercept
    reduction.
    """
    historical_samples = draw_historical_scalar_no_intercept_reduction_samples(
        plate_size=plate_size, repetitions=repetitions, rng=seed
    )
    condensed_samples = draw_condensed_scalar_constant_samples(
        plate_size=plate_size, repetitions=repetitions, rng=seed + 1
    )
    result = compare_samples(historical_samples, condensed_samples)
    print("-" * 60)
    print("Comparing k = 1 condensed formula against the "
          "historical scalar no-intercept reduction")
    print("KS p-value: {:.6f}".format(result['ks_pvalue']))
    print("Normalized Wasserstein: {:.6f}".format(
        result['wasserstein_test']['normalized_wasserstein_distance']))
    print("Result: {}".format("PASS" if result['success'] else "FAIL"))
    return result


def validate_one_k(k, plate_size, repetitions, seed, run_agent_smoke=True):
    """
    Validate the reduced model against the brute-force reference for one k.
    """
    if run_agent_smoke:
        smoke_test_agent_model(k, plate_size, seed)
    print("=" * 60)
    print("Validating k = {} with {} repetitions and plate size {}".format(
        k, repetitions, plate_size))
    phi_reduced, parameter_samples = draw_reduced_samples(
        k=k, plate_size=plate_size, repetitions=repetitions, rng=seed
    )
    phi_reference = draw_full_plate_samples(
        k=k, plate_size=plate_size, parameter_samples=parameter_samples, rng=seed + 1
    )

    results = {'k': k, 'plate_size': plate_size, 'repetitions': repetitions, 'seed': seed}
    phi_results = []
    if k == 1:
        phi_reference_matrix = phi_reference.reshape(-1, 1)
        phi_reduced_matrix = phi_reduced.reshape(-1, 1)
    else:
        phi_reference_matrix = phi_reference
        phi_reduced_matrix = phi_reduced
    for j in range(k):
        print("-" * 60)
        print("Comparing phi_{}".format(j))
        phi_results.append(
            compare_samples(phi_reference_matrix[:, j], phi_reduced_matrix[:, j])
        )
    results['phi'] = phi_results
    results['success'] = bool(all(item['success'] for item in results['phi']))
    if results['success']:
        print("PASS for k = {}".format(k))
    else:
        print("FAIL for k = {}".format(k))
    return results


def write_results(results, output_dir):
    """
    Save json and csv summaries if an output directory is provided.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'summary.json', 'w') as file:
        json.dump(results, file, indent=2)

    rows = [
        'k,variable,success,reference_mean,reduced_mean,reference_std,'
        'reduced_std,ks_pvalue,normalized_wasserstein_distance',
    ]
    scalar_result = results.get('scalar_historical_no_intercept_comparison')
    if scalar_result is not None:
        rows.append(
            '{},{},{},{:.12f},{:.12f},{:.12f},{:.12f},{:.12f},{:.12f}'.format(
                1,
                'historical_scalar_no_intercept_phi',
                int(scalar_result['success']),
                scalar_result['reference_summary']['mean'],
                scalar_result['reduced_summary']['mean'],
                scalar_result['reference_summary']['std'],
                scalar_result['reduced_summary']['std'],
                scalar_result['ks_pvalue'],
                scalar_result['wasserstein_test']['normalized_wasserstein_distance'],
            )
        )
    for item in results['results']:
        for j, phi_result in enumerate(item['phi']):
            rows.append(
                '{},{},{},{:.12f},{:.12f},{:.12f},{:.12f},{:.12f},{:.12f}'.format(
                    item['k'],
                    'phi_{}'.format(j),
                    int(phi_result['success']),
                    phi_result['reference_summary']['mean'],
                    phi_result['reduced_summary']['mean'],
                    phi_result['reference_summary']['std'],
                    phi_result['reduced_summary']['std'],
                    phi_result['ks_pvalue'],
                    phi_result['wasserstein_test']['normalized_wasserstein_distance'],
                )
            )
    with open(output_dir / 'summary.csv', 'w') as file:
        file.write('\n'.join(rows) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plate-size', type=int, default=1000)
    parser.add_argument('--repetitions', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=4)
    parser.add_argument('--k-values', type=int, nargs='+', default=[1, 2, 5])
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--skip-agent-smoke', action='store_true')
    parser.add_argument('--skip-scalar-historical-no-intercept-check',
                        action='store_true')
    args = parser.parse_args()

    start_time = time.time()
    scalar_result = None
    if not args.skip_scalar_historical_no_intercept_check:
        scalar_result = validate_scalar_against_historical_no_intercept_reduction(
            plate_size=args.plate_size,
            repetitions=args.repetitions,
            seed=args.seed,
        )
    results = []
    for k in args.k_values:
        results.append(
            validate_one_k(
                k=k,
                plate_size=args.plate_size,
                repetitions=args.repetitions,
                seed=args.seed,
                run_agent_smoke=not args.skip_agent_smoke,
            )
        )
    overall_success = all(item['success'] for item in results)
    if scalar_result is not None:
        overall_success = overall_success and scalar_result['success']
    elapsed_time = time.time() - start_time
    summary = {
        'plate_size': args.plate_size,
        'repetitions': args.repetitions,
        'seed': args.seed,
        'k_values': args.k_values,
        'elapsed_time': elapsed_time,
        'success': overall_success,
        'scalar_historical_no_intercept_comparison': scalar_result,
        'results': results,
    }
    if args.output_dir is not None:
        write_results(summary, args.output_dir)

    print("=" * 60)
    print("Overall result: {}".format("PASS" if overall_success else "FAIL"))
    print("Elapsed time: {:.2f} seconds".format(elapsed_time))
    if not overall_success:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
