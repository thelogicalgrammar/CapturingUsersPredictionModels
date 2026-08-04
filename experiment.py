"""
This module provides functionality to run experiments that test agent decision-making
with and without decision support systems.

The script takes three command line parameters where the first one
is the key in agent_priors to modify, the second one what value 
agent_priors[key] should take, and the third one what value X_o
should take.

Example:

.. code-block:: bash

    python experiment.py N_E_mu -20 72

This would set the value of the prior distribution of N_E to -20 and
use 72kg as the feature of the observed patient.


Notes:
Could perhaps implement support for changing plate size and SCM parameters
"""
import argparse
import hashlib
import os
import time
import uuid
from pathlib import Path

_cache_root = os.environ.get(
    'XDG_CACHE_HOME', os.path.expanduser('~/.cache'))
os.makedirs(os.path.join(_cache_root, 'arviz'), exist_ok=True)

import arviz as az
import matplotlib.pyplot as plt  # Not used in array job.
import numpy as np
from sklearn.linear_model import LinearRegression
# Project specific:
import scm
import utils
import agent_model as am
from experiment_randomness import (
    NO_INTERCEPT_SCAN_CENTERS,
    RESULT_SCHEMA_VERSION,
    SCHEME as INDEPENDENT_DATA_SEED_SCHEME,
    format_stream_key,
    run_rng,
    run_stream_key,
)



# Set number of observations used to train prediction model.
# This is also the plate size used in the agent's belief model.
plate_size = 1000
scm_noise_parameters = {'alpha_mu_X': 80, 'alpha_sigma_X': 10,
                        'alpha_mu_A': 2, 'alpha_sigma_A': 1.0,
                        'alpha_mu_Y': 0.0, 'alpha_sigma_Y': 0.1,
                        'N_E': 1.0}

# Set scm agent priors to the same
predictive_var_names = list(scm_noise_parameters)
scm_constants = {'a': 12.0, 'b': -0.1, 'd': 0.125}
draws = 1000  # Number of draws for trace. Warmup is set to 1000.
# Default values of agent priors. One value is overwritten in main().
agent_priors =\
    {'mu_X': 80, 'sigma_X': 0.1, 'mu_sigma_X': 10, 'sigma_sigma_X':0.1,
     'mu_A': 2, 'sigma_A': 1, 'mu_sigma_A': 1, 'sigma_sigma_A':0.1,
     'mu_Y': 0, 'sigma_Y': 0.01, 'mu_sigma_Y': 0.1, 'sigma_sigma_Y' : 0.01,
     'N_E_mu': 1.0, 'N_E_sigma': 1}
# changed agent prior sigma_Y from 1 to 0.1, this is a choice
# changed prior of lambda_X from 10 to 0.1, this was a mistake
# Treatment info
A1_treatment = 20
A0_treatment = 10
# Only if treatment A1 is expected to produce an outcome that is threshold=2 above
# the expected outcome with A0 is A1 used. This corresponds to e.g. A1 being more
# expensive or to A1 having larger side effects.
threshold = 5
parameter_descriptions = {'agent_priors': 'Default',
                          'scm_constants': 'Default',
                          'scm_noise_parameters': 'Default'}
rng = 4  # Selected by fair die


def _array_fingerprint(*arrays):
    """Return a stable SHA-256 fingerprint for typed NumPy arrays."""
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(array.dtype.str.encode('ascii'))
        digest.update(str(array.shape).encode('ascii'))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _publish_without_overwrite(tmp_path, final_path):
    """Atomically publish a staged file without replacing an existing one."""
    tmp_path = Path(tmp_path)
    final_path = Path(final_path)
    try:
        os.link(tmp_path, final_path)
    except FileExistsError:
        raise FileExistsError(
            'refusing to overwrite existing result: {}'.format(final_path))
    finally:
        tmp_path.unlink(missing_ok=True)


def main(X_o, agent_priors, modified_prior,
         A1_treatment=A1_treatment, A0_treatment=A0_treatment,
         plate_size=plate_size, draws=draws,
         scm_noise_parameters=scm_noise_parameters,
         scm_constants=scm_constants,
         parameter_descriptions=parameter_descriptions,
         rng=rng, data_rng=None, sampler_settings=None,
         output_dir='./results', output_file='results.csv',
         result_metadata=None, plot_pair=True, refuse_overwrite=False):
    """
    Args:
        X_o (np.array): Feature value of the observed patient (e.g. in kg).
        agent_priors (dict): Dictionary with all values that encodes the
            agent's prior beliefs.
        modified_prior (dict): Information about which prior was changed
            and how much.
        A1_treatment (float): Value of treatment 1.
        A0_treatment (float): Value of treatment 0.
        plate_size (int): Size of plate / number or observations to generate
            for regression model. Measure of "evidence" for predictions from
            decision support model.
        draws (int): Size of sample to generate in sampling events.
        scm_noise_parameters (dict): Dict with keys needed in SCM-class.
        scm_constants (dict): Constants defining the relationships in the SCM.
        rng (int or rng): Seed or rng for reproducibility.
        data_rng: Independent seed or generator for the historical dataset.
        sampler_settings (dict or None): Explicit NUTS settings.
        output_dir (str or pathlib.Path): Result directory.
        output_file (str): Result CSV name.
        result_metadata (dict or None): Additional result provenance.
        plot_pair (bool): Whether to write the posterior pair plot.
        refuse_overwrite (bool): Atomically reject an existing result file.
    """
    start_time = time.time()
    rng = np.random.default_rng(rng)
    historical_data_rng = (
        rng if data_rng is None else np.random.default_rng(data_rng))
    if sampler_settings is None:
        sampler_settings = {
            'draws': draws,
            'tune': 1000,
            'chains': 4,
            'target_accept': 0.85,
        }
    else:
        sampler_settings = sampler_settings.copy()
    draws = sampler_settings['draws']
    output_dir = Path(output_dir)
    if refuse_overwrite and (output_dir / output_file).exists():
        raise FileExistsError(
            'refusing to overwrite existing result: {}'.format(
                output_dir / output_file))
    # 1. SCM-instance to generate population data under the
    #    historical treatment policy.
    #    X is sampled from the population distribution, A
    #    is the treatment given under the historical treatment
    #    policy given X, and Y is the outcome given X and A.
    data_scm = scm.SCM_WithConstants(**scm_noise_parameters, **scm_constants)

    # 2. Generate data. This is identical to the true
    #    underlying generating process and the outcomes
    #    Y are outcomes under the historical treatment
    #    policy.
    X, A, Y_historical, *_ = data_scm.sample(
        draws=plate_size, rng=historical_data_rng)

    # 3. Train prediction model on historical data.
    #    Calculated as the Moore-Penrose pseudo-inverse. No randomness.
    pred_model = LinearRegression(fit_intercept=False).fit(X.reshape(-1, 1), Y_historical)
    historical_data_sha256 = _array_fingerprint(X, A, Y_historical)

    # 4. The prediction model is next used for decision support. The physician
    #    inputs the data of the patient and the prediction model produces a prediction.
    pred_o = pred_model.predict(X_o.reshape(-1, 1))  # The observed prediction
    # print("The predicted outcome for this patient weighing {}kg is {}.".format(X_o, pred_o[0]))

    # 5. Define the agent model.
    #    Using same _constants_ for agent's internal SCM as for data generating SCM.
    agent_model = am.ScalarAgentModel(prior_parameters=agent_priors,
                                       scm_constants=scm_constants,
                                       predictive_var_names=predictive_var_names,
                                       plate_size=plate_size)

    # 6. Predictive estimates and treatment decision under prior beliefs
    prior_CATE_mu = agent_model.get_prior_CATE_mu(A1_treatment, A0_treatment)
    prior_CATE_samples_, prior_CATE_mean, prior_CATE_std = \
        agent_model.sample_prior_predictive_CATE(
            X_o, A1_treatment, A0_treatment, rng=rng)
    prior_high_treatment = prior_CATE_mu > threshold
    prior_treatment = (
        A1_treatment if prior_high_treatment else A0_treatment)
    prior_predictive_summaries = agent_model.sample_prior_predictive_summaries(rng=rng)

    # 7. Predictive estimates and treatment decision under posterior beliefs
    # The agent's beliefs are updated / the posterior given the observed data is sampled.
    agent_model.sample(
        X_o=np.asarray(X_o), pred_o=np.asarray(pred_o[0]), rng=rng,
        **sampler_settings)
    # Model diagnostics
    trace_diagnostics = agent_model.trace_diagnostics()

    # Using updated agent beliefs to come to a decision on what is the best treatment
    posterior_CATE_samples_, posterior_CATE_mean, posterior_CATE_std =\
        agent_model.sample_posterior_predictive_CATE(X_o, A1_treatment, A0_treatment, rng=rng)
    posterior_high_treatment = posterior_CATE_mean > threshold
    posterior_treatment = (
        A1_treatment if posterior_high_treatment else A0_treatment)
    posterior_predictive_summaries = agent_model.sample_posterior_predictive_summaries(rng=rng)
    posterior_parameter_summaries = agent_model.posterior_parameter_summaries()

    # 8. Ground truth of CATE and potential outcomes
    #    based on the data generating SCM
    true_CATE_mu = data_scm.get_CATE_mu(A1_treatment, A0_treatment)
    true_CATE_samples_, true_CATE_mean, true_CATE_std =\
        data_scm.sample_CATE(X_o, A1_treatment, A0_treatment, draws=draws, rng=rng)
    # Potential outcomes under treatment A1 and A0
    # This currently generates one single draw for each and the
    # SCM includes randomness.
    # Estimate mean?
    potential_outcome_A1_samples_, potential_outcome_A1_mean, potential_outcome_A1_std =\
        data_scm.sample_do_A(A1_treatment, X_o, draws=draws, rng=rng)
    potential_outcome_A0_samples_, potential_outcome_A0_mean, potential_outcome_A0_std =\
        data_scm.sample_do_A(A0_treatment, X_o, draws=draws, rng=rng)

    # 9. Expected outcomes given agent's prior and posterior beliefs
    # given under treatment A0 and A1:
    _, expected_prior_outcome_A1_mean, expected_prior_outcome_A1_std =\
        agent_model.sample_prior_predictive_do_A(X_o, A1_treatment, rng=rng)
    _, expected_prior_outcome_A0_mean, expected_prior_outcome_A0_std =\
        agent_model.sample_prior_predictive_do_A(X_o, A0_treatment, rng=rng)
    _, expected_posterior_outcome_A1_mean, expected_posterior_outcome_A1_std =\
        agent_model.sample_posterior_predictive_do_A(X_o, A1_treatment, rng=rng)
    _, expected_posterior_outcome_A0_mean, expected_posterior_outcome_A0_std =\
        agent_model.sample_posterior_predictive_do_A(X_o, A0_treatment, rng=rng)
    expected_outcomes = {
        'expected_prior_outcome_A1_mean': expected_prior_outcome_A1_mean, 
        'expected_prior_outcome_A1_std': expected_prior_outcome_A1_std,
        'expected_prior_outcome_A0_mean': expected_prior_outcome_A0_mean, 
        'expected_prior_outcome_A0_std': expected_prior_outcome_A0_std,
        'expected_posterior_outcome_A1_mean': expected_posterior_outcome_A1_mean, 
        'expected_posterior_outcome_A1_std': expected_posterior_outcome_A1_std,
        'expected_posterior_outcome_A0_mean': expected_posterior_outcome_A0_mean, 
        'expected_posterior_outcome_A0_std': expected_posterior_outcome_A0_std
    }
    # 9. Compare results

    true_high_treatment = true_CATE_mu > threshold
    true_sample_high_treatment = true_CATE_samples_ > threshold
    prior_decision_correct = prior_high_treatment == true_high_treatment
    posterior_decision_correct = \
        posterior_high_treatment == true_high_treatment
    # Which decision is best? Assuming that threshold accounts for negative side effects.
    true_best_treatment = (
        A1_treatment if true_high_treatment else A0_treatment)
    # Did the decision change (+1 for better, 0 no change, -1 for worse)?
    if prior_high_treatment != posterior_high_treatment:
        # Set to 1 if changed for the better and -1 if changed for the worse
        decision_changed = (
            1 if posterior_high_treatment == true_high_treatment else -1)
    else:
        # Set to 0 if no change.
        decision_changed = 0

    # For what fraction of patients is each treatment decision correct?
    frac_correct_prior_decision = np.mean(
        true_sample_high_treatment == prior_high_treatment)
    frac_correct_posterior_decision = np.mean(
        true_sample_high_treatment == posterior_high_treatment)
    # Fraction of population that would have benefited from a
    # decision based on the average true CATE (baseline):
    frac_correct_true_best_treatment = np.mean(
        true_sample_high_treatment == true_high_treatment)

    # How much do the patients on average benefit?
    # Average outcome Y under prior decision
    Y_mean_prior_treatment = (
        potential_outcome_A1_mean
        if prior_high_treatment else potential_outcome_A0_mean)
    # Average outcome Y under posterior decision:
    Y_mean_posterior_treatment = (
        potential_outcome_A1_mean
        if posterior_high_treatment else potential_outcome_A0_mean)
    # Average outcome under best decision based on true_CATE_mu
    Y_mean_true_best_treatment = (
        potential_outcome_A1_mean
        if true_high_treatment else potential_outcome_A0_mean)
    # Also save prediction pred_o
    outputs = {'pred_o': pred_o[0],
            'threshold': threshold,
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
            }

    # Collect results.
    experiment_id = uuid.uuid4().hex
    compiled_results = {'experiment_id': experiment_id,
                        'true_CATE_mu': true_CATE_mu,
                        'true_CATE_mean': true_CATE_mean,
                        'true_CATE_std': true_CATE_std,
                        'prior_CATE_mu': prior_CATE_mu,
                        'prior_CATE_mean': prior_CATE_mean,
                        'prior_CATE_std': prior_CATE_std,
                        'posterior_CATE_mean': posterior_CATE_mean,
                        'posterior_CATE_std': posterior_CATE_std,
                        'plate_size': plate_size,
                        'X_o': float(np.asarray(X_o)),
                        'X_o_0': float(np.asarray(X_o)),
                        'A1_treatment': A1_treatment,
                        'A0_treatment': A0_treatment,
                        'historical_data_sha256':
                            historical_data_sha256,
                        'prediction_model_intercept':
                            float(pred_model.intercept_),
                        'prediction_model_coef_0':
                            float(np.ravel(pred_model.coef_)[0])}
    compiled_results.update(outputs)
    compiled_results.update(parameter_descriptions)  # Include parameter descriptions
    end_time = time.time()
    running_time = end_time - start_time
    current_time = time.strftime("%H:%M:%S %d.%m.%Y")
    compiled_results['time_stamp'] = current_time
    compiled_results['running_time'] = running_time
    compiled_results.update(modified_prior)
    compiled_results.update(trace_diagnostics)

    # Add prior predictive summaries
    compiled_results.update(prior_predictive_summaries)
    # Add posterior predictive summaries
    compiled_results.update(posterior_predictive_summaries)
    # Add posterior parameter summaries
    compiled_results.update(posterior_parameter_summaries)
    # Add expected predictive outputs:
    compiled_results.update(expected_outcomes)
    if result_metadata:
        duplicate_metadata = set(result_metadata).intersection(
            compiled_results)
        if duplicate_metadata:
            raise ValueError(
                'result_metadata duplicates result fields: {}'.format(
                    sorted(duplicate_metadata)))
        compiled_results.update(result_metadata)
    compiled_results['agent_model'] = 'scalar'
    compiled_results['fit_intercept'] = False
    compiled_results['n_covariates'] = 1
    compiled_results['n_draws'] = sampler_settings['draws']
    compiled_results['n_tune'] = sampler_settings['tune']
    compiled_results['n_chains'] = sampler_settings['chains']
    compiled_results['target_accept'] = \
        sampler_settings['target_accept']

    if refuse_overwrite:
        final_path = output_dir / output_file
        tmp_file = '{}.tmp.{}.{}'.format(
            output_file, os.getpid(), uuid.uuid4().hex)
        output_dir.mkdir(parents=True, exist_ok=True)
        utils.write_results_to_file(
            compiled_results, file_name=tmp_file,
            path=str(output_dir) + os.sep)
        _publish_without_overwrite(output_dir / tmp_file, final_path)
    else:
        utils.write_results_to_file(
            compiled_results, file_name=output_file,
            path=str(output_dir) + os.sep)

    # Skip saving model in array run.
    # Save model to disk:
    # agent_model.save_agent_model(experiment_id)

    # Testing that loading works.
    # tmp = am.ScalarAgentModel.load_agent_model()
    # What is the posterior distribution of the agent's belief about N_E?
    # az.plot_posterior(agent_model.trace, var_names=['N_E', 'alpha_mu_A'])
    # plt.title(r"Posterior distribution of correlation between $mu_A$ and $N_E$")
    # plt.savefig('./fig/basic_example_posterior_NE.pdf')
    # added plot to save the posterior distribution of pairwise correlations

    if plot_pair:
        az.plot_pair(
            agent_model.trace, var_names=['N_E', 'alpha_mu_A'],
            kind='scatter')
        plt.title(
            r"Joint posterior samples of $\alpha_{A_\mu}$ and $N_E$")
        plt.xlabel(r'$N_E$')
        plt.ylabel(r'$\alpha_{A_\mu}$')
        plt.tight_layout()
        plt.savefig('./fig/basic_example_posterior_corr.pdf')
    # plt.clf()


def _positive_int(value):
    """Parse a positive integer for argparse."""
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError('expected a positive integer')
    return value


def _nonnegative_int(value):
    """Parse a non-negative integer for argparse."""
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError(
            'expected a non-negative integer')
    return value


def _target_accept(value):
    """Parse a valid NUTS target acceptance probability."""
    value = float(value)
    if not 0.0 < value < 1.0:
        raise argparse.ArgumentTypeError(
            'expected a value strictly between 0 and 1')
    return value


def _parse_args():
    """Parse the univariate experiment command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('param_name')
    parser.add_argument('param_value', type=float)
    parser.add_argument('x_o', nargs='?', type=float)
    parser.add_argument(
        '--experiment-design', choices=('univariate_no_intercept',))
    parser.add_argument('--base-seed', type=_nonnegative_int)
    parser.add_argument('--scan-deviation', type=int)
    parser.add_argument('--replicate', type=_nonnegative_int)
    parser.add_argument('--draws', type=_positive_int, default=draws)
    parser.add_argument('--tune', type=_positive_int, default=1000)
    parser.add_argument('--chains', type=_positive_int, default=4)
    parser.add_argument(
        '--target-accept', type=_target_accept, default=0.85)
    parser.add_argument('--output-dir', default='./results')
    args = parser.parse_args()
    if args.experiment_design is not None:
        if (args.base_seed is None or args.scan_deviation is None or
                args.replicate is None):
            parser.error(
                '--experiment-design requires --base-seed, '
                '--scan-deviation and --replicate')
        if args.x_o is not None:
            parser.error(
                'x_o is generated by --experiment-design and cannot be '
                'supplied directly')
    elif (args.base_seed is not None or args.scan_deviation is not None or
          args.replicate is not None):
        parser.error(
            '--base-seed, --scan-deviation and --replicate require '
            '--experiment-design')
    return args


if __name__ == '__main__':
    args = _parse_args()
    if args.param_name not in agent_priors:
        raise ValueError(
            '{} is not a valid prior parameter'.format(args.param_name))

    run_priors = agent_priors.copy()
    run_descriptions = parameter_descriptions.copy()
    run_descriptions['agent_priors'] = (
        'Default'
        if run_priors[args.param_name] == args.param_value
        else 'Modified')
    run_priors[args.param_name] = args.param_value
    modified_prior = {
        'modified_prior': args.param_name,
        'modified_prior_value': args.param_value,
    }
    data_rng = None
    run_rng_ = np.random.default_rng(rng)
    result_metadata = None
    output_file = 'results.csv'
    plot_pair = True
    refuse_overwrite = False

    if args.experiment_design is not None:
        center, step = NO_INTERCEPT_SCAN_CENTERS[args.param_name]
        expected_value = center + args.scan_deviation * step
        if not np.isclose(
                args.param_value, expected_value, rtol=0.0, atol=1e-7):
            raise ValueError(
                '{} value {} does not match scan deviation {:+d}'.format(
                    args.param_name, args.param_value,
                    args.scan_deviation))
        key_args = (
            args.experiment_design,
            args.param_name,
            args.scan_deviation,
            1,
            args.replicate,
            '1k',
        )
        data_key = run_stream_key('historical_data', *key_args)
        observation_key = run_stream_key('observation', *key_args)
        sampler_key = run_stream_key('sampler', *key_args)
        data_rng = run_rng(
            args.base_seed, 'historical_data', *key_args)
        observation_rng = run_rng(
            args.base_seed, 'observation', *key_args)
        run_rng_ = run_rng(args.base_seed, 'sampler', *key_args)
        X_o = np.asarray(observation_rng.normal(
            loc=scm_noise_parameters['alpha_mu_X'],
            scale=scm_noise_parameters['alpha_sigma_X']))
        modified_prior['seed'] = args.replicate
        modified_prior['coef_sigma_scaling'] = '1/k'
        result_metadata = {
            'result_schema_version': RESULT_SCHEMA_VERSION,
            'experiment_design': args.experiment_design,
            'experimental_unit': 'historical_dataset_and_patient',
            'replicate_id': args.replicate,
            'patient_id': format_stream_key(observation_key),
            'data_seed': args.base_seed,
            'historical_data_stream_key':
                format_stream_key(data_key),
            'sampler_seed': args.base_seed,
            'sampler_stream_key': format_stream_key(sampler_key),
            'observation_seed_scheme': INDEPENDENT_DATA_SEED_SCHEME,
        }
        output_file = (
            'results_{}_{:.6f}_k1_seed{}.csv'.format(
                args.param_name, args.param_value, args.replicate))
        plot_pair = False
        refuse_overwrite = True
    else:
        X_o = np.asarray(
            args.x_o if args.x_o is not None else
            np.random.normal(
                loc=scm_noise_parameters['alpha_mu_X'],
                scale=scm_noise_parameters['alpha_sigma_X']))

    print(
        'Running experiment with X_o set to {} and agent prior {} '
        'set to {}...'.format(
            X_o, args.param_name, args.param_value))
    sampler_settings = {
        'draws': args.draws,
        'tune': args.tune,
        'chains': args.chains,
        'target_accept': args.target_accept,
    }
    main(
        X_o, run_priors, modified_prior,
        parameter_descriptions=run_descriptions,
        rng=run_rng_, data_rng=data_rng,
        sampler_settings=sampler_settings,
        output_dir=args.output_dir,
        output_file=output_file,
        result_metadata=result_metadata,
        plot_pair=plot_pair,
        refuse_overwrite=refuse_overwrite)
