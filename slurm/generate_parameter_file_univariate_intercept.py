"""
Generate the dense univariate-with-intercept experiment manifest.

The observed patients are reused row by row from ``results/wide_results.csv``
so the generalized with-intercept formulation can be compared with the
historical no-intercept scan on exactly the same observations. The historical
``mu_Y`` values are translated from a zero-centered structural-noise mean to
the intercept-aware center at 12.
"""
import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dimensionality_config import (
    AGENT_MODEL,
    A0_TREATMENT,
    A1_TREATMENT,
    DECISION_THRESHOLD,
    FIT_INTERCEPT,
    FIXED_RNG_SEED,
    N_CHAINS,
    N_DRAWS,
    N_TUNE,
    PLATE_SIZE,
    RESULT_SCHEMA_VERSION,
    SCAN_CENTERS,
    TARGET_ACCEPT,
    WIDE_RESULTS_SHA256,
)
from slurm.generate_parameter_file_dimensionality import (
    parse_integer_selection,
    write_atomic,
)


SOURCE_CENTERS = {
    'mu_A': (2.0, 1.0),
    'mu_X': (80.0, 10.0),
    'mu_Y': (0.0, 0.1),
    'N_E_mu': (1.0, 0.05),
}
DEFAULT_DEVIATIONS = tuple(range(-30, 31))
DEFAULT_SOURCE = Path('results/wide_results.csv')
DEFAULT_OUTPUT = Path('slurm/tmp/parameters_univariate_intercept_wide.txt')


def _load_observations(source_file):
    """Group archived observations by prior family and scan offset."""
    source_file = Path(source_file)
    grouped = defaultdict(list)
    with source_file.open(newline='') as stream:
        for source_row, row in enumerate(csv.DictReader(stream)):
            param = row['Modified prior']
            if param not in SOURCE_CENTERS:
                raise ValueError(
                    'unknown prior family in source row {}: {!r}'.format(
                        source_row, param))
            value = float(row['New value of modified prior'])
            observed_x = float(row['X_o'])
            if not math.isfinite(value) or not math.isfinite(observed_x):
                raise ValueError(
                    'source row {} contains a non-finite value'.format(
                        source_row))
            center, step = SOURCE_CENTERS[param]
            raw_offset = (value - center) / step
            offset = round(raw_offset)
            if abs(raw_offset - offset) > 1e-7:
                raise ValueError(
                    'source row {} is off the registered scan grid'.format(
                        source_row))
            grouped[(param, offset)].append((source_row, observed_x))
    expected = {
        (param, offset)
        for param in SOURCE_CENTERS
        for offset in DEFAULT_DEVIATIONS
    }
    if set(grouped) != expected:
        missing = sorted(expected - set(grouped))
        extra = sorted(set(grouped) - expected)
        raise ValueError(
            'source grid mismatch: missing={}, extra={}'.format(
                missing[:3], extra[:3]))
    incorrect_counts = {
        key: len(values)
        for key, values in grouped.items() if len(values) != 50
    }
    if incorrect_counts:
        raise ValueError(
            'source cells must contain exactly 50 observations: {}'.format(
                sorted(incorrect_counts.items())[:3]))
    return grouped


def generate_univariate_intercept_manifest(
        source_file=DEFAULT_SOURCE, parameters_file=DEFAULT_OUTPUT,
        replicates=None, deviations=None, fixed_rng_seed=FIXED_RNG_SEED,
        design_file=None):
    """
    Write an extensible manifest for the dense univariate intercept scan.

    Args:
        source_file (str or pathlib.Path): Historical wide-results CSV.
        parameters_file (str or pathlib.Path): Output manifest.
        replicates (sequence of int or None): Per-cell archived patient IDs.
        deviations (sequence of int or None): Scan-step offsets.
        fixed_rng_seed (int): Base random seed.
        design_file (str or pathlib.Path or None): JSON sidecar path.

    Returns:
        pathlib.Path: Generated manifest path.
    """
    source_file = Path(source_file)
    parameters_file = Path(parameters_file)
    parameters_file.parent.mkdir(parents=True, exist_ok=True)
    source_sha256 = hashlib.sha256(source_file.read_bytes()).hexdigest()
    if source_sha256 != WIDE_RESULTS_SHA256:
        raise ValueError(
            'source CSV does not match the registered wide-results snapshot')
    replicates = tuple(range(50) if replicates is None else replicates)
    deviations = tuple(
        DEFAULT_DEVIATIONS if deviations is None else deviations)
    if (not replicates or len(set(replicates)) != len(replicates) or
            any(not isinstance(value, int) or value < 0
                for value in replicates)):
        raise ValueError('replicates must be unique non-negative integers')
    if not set(replicates) <= set(range(50)):
        raise ValueError('replicates must be selected from 0 through 49')
    if (not deviations or len(set(deviations)) != len(deviations) or
            any(not isinstance(value, int) for value in deviations)):
        raise ValueError('deviations must be unique integers')
    if not set(deviations) <= set(DEFAULT_DEVIATIONS):
        raise ValueError('deviations must be selected from -30 through 30')
    if not isinstance(fixed_rng_seed, int) or fixed_rng_seed < 0:
        raise ValueError('fixed_rng_seed must be a non-negative integer')

    grouped = _load_observations(source_file)
    lines = []
    for param, (target_center, step) in SCAN_CENTERS.items():
        for deviation in deviations:
            observations = grouped.get((param, deviation), ())
            if len(observations) != 50:
                raise ValueError(
                    '{} offset {:+d}: expected 50 archived observations, '
                    'found {}'.format(param, deviation, len(observations)))
            target_value = target_center + deviation * step
            for replicate in replicates:
                try:
                    source_row, observed_x = observations[replicate]
                except IndexError:
                    raise ValueError(
                        'replicate {} is unavailable for {} offset {:+d}'
                        .format(replicate, param, deviation))
                lines.append(
                    '{} {:.6f} {} {:.17g} {}\n'.format(
                        param, target_value, replicate,
                        observed_x, source_row))

    manifest_content = ''.join(lines)
    write_atomic(parameters_file, manifest_content)
    if design_file is None:
        design_file = parameters_file.with_name(
            parameters_file.stem + '.design.json')
    design_file = Path(design_file)
    design = {
        'design_version': 1,
        'result_schema_version': RESULT_SCHEMA_VERSION,
        'experiment_design': 'univariate_intercept_wide',
        'agent_model': AGENT_MODEL,
        'manifest': parameters_file.name,
        'manifest_sha256': hashlib.sha256(
            manifest_content.encode('utf-8')).hexdigest(),
        'line_count': len(lines),
        'source_file': 'results/wide_results.csv',
        'source_sha256': source_sha256,
        'parameters': {
            key: {'center': center, 'step': step}
            for key, (center, step) in SCAN_CENTERS.items()
        },
        'deviations': list(deviations),
        'replicates': list(replicates),
        'n_covariates': 1,
        'scaling': '1k',
        'fixed_rng_seed': fixed_rng_seed,
        'observation_seed_scheme': 'archived_observation',
        'pairing': 'row_matched_to_historical_no_intercept_scan',
        'sampler': {
            'draws': N_DRAWS,
            'tune': N_TUNE,
            'chains': N_CHAINS,
            'target_accept': TARGET_ACCEPT,
        },
        'experiment': {
            'plate_size': PLATE_SIZE,
            'a1_treatment': A1_TREATMENT,
            'a0_treatment': A0_TREATMENT,
            'decision_threshold': DECISION_THRESHOLD,
            'fit_intercept': FIT_INTERCEPT,
        },
    }
    write_atomic(
        design_file, json.dumps(design, indent=2, sort_keys=True) + '\n')

    print('Wrote {} lines to {}.'.format(len(lines), parameters_file))
    print('Design metadata: {}'.format(design_file))
    return parameters_file


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default=str(DEFAULT_SOURCE))
    parser.add_argument('--output', default=str(DEFAULT_OUTPUT))
    parser.add_argument('--design-output')
    parser.add_argument('--replicates', default='0-49')
    parser.add_argument(
        '--deviations', nargs='+', type=int, default=DEFAULT_DEVIATIONS)
    parser.add_argument(
        '--fixed-rng-seed', type=int, default=FIXED_RNG_SEED)
    args = parser.parse_args()
    try:
        replicate_ids = parse_integer_selection(args.replicates)
    except ValueError as exc:
        parser.error(str(exc))
    generate_univariate_intercept_manifest(
        source_file=args.source,
        parameters_file=args.output,
        replicates=replicate_ids,
        deviations=args.deviations,
        fixed_rng_seed=args.fixed_rng_seed,
        design_file=args.design_output)
