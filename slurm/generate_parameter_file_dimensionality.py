"""
Generate a manifest for the dimensionality SLURM array job.

Each line contains ``param_name param_value n_covariates replicate``. The
``sqrtk`` scaling adds a fifth field with that value.

Run from the repo root::

    python slurm/generate_parameter_file_dimensionality.py
    python slurm/generate_parameter_file_dimensionality.py \
        --sigma-scaling sqrtk \
        --output slurm/tmp/parameters_dimensionality_sqrtk.txt
"""
import argparse
import hashlib
import json
import os
import sys
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dimensionality_config import (  # noqa: E402
    AGENT_MODEL,
    A0_TREATMENT,
    A1_TREATMENT,
    DECISION_THRESHOLD,
    DEFAULT_SCALING,
    DIMENSIONALITY_EXPERIMENT_DESIGN,
    DIMENSIONALITY_PAIRING,
    EXPLORATORY_DEVIATIONS,
    FIT_INTERCEPT,
    FIXED_RNG_SEED,
    N_CHAINS,
    N_COVARIATES,
    N_DRAWS,
    N_REPLICATES,
    N_TUNE,
    OBSERVATION_SEED_SCHEME,
    PLATE_SIZE,
    RESULT_SCHEMA_VERSION,
    SCALING_CHOICES,
    SCAN_CENTERS,
    SCAN_DEVIATIONS,
    TARGET_ACCEPT,
)


DEFAULT_PARAMETERS_FILE = Path('slurm/tmp/parameters_dimensionality.txt')


def write_atomic(path, content):
    """Write text without exposing a partial manifest or design file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(
        '{}.tmp.{}.{}'.format(path.name, os.getpid(), uuid.uuid4().hex))
    tmp.write_text(content)
    os.replace(tmp, path)


def parse_integer_selection(spec):
    """Parse ``0-49`` or a comma-separated integer list."""
    if '-' in spec and ',' not in spec:
        fields = spec.split('-')
        if len(fields) != 2:
            raise ValueError("expected an inclusive range such as '0-49'")
        lower, upper = map(int, fields)
        if upper < lower:
            raise ValueError('range end precedes range start')
        return tuple(range(lower, upper + 1))
    values = tuple(int(field) for field in spec.split(',') if field)
    if not values:
        raise ValueError('selection is empty')
    return values


def generate_slurm_parameter_file(
        parameters_file=None, sigma_scaling=DEFAULT_SCALING,
        replicates=None, deviations=None, n_covariates=None,
        fixed_rng_seed=FIXED_RNG_SEED, design_file=None):
    """
    Write parameter combinations to a text file for the SLURM array job.

    Args:
        parameters_file (str or pathlib.Path or None): Output path. When
            omitted, ``1k`` uses ``parameters_dimensionality.txt`` and
            ``sqrtk`` uses ``parameters_dimensionality_sqrtk.txt``.
        sigma_scaling (str): Key in
            :data:`dimensionality_config.SCALING_CHOICES`.
        replicates (sequence of int or None): Within-cell replication IDs.
        deviations (sequence of int or None): Scan-step offsets.
        n_covariates (sequence of int or None): Dimensionalities.
        fixed_rng_seed (int): Base random seed.
        design_file (str or pathlib.Path or None): JSON design sidecar path.

    Returns:
        pathlib.Path: Path of the generated manifest.
    """
    if sigma_scaling not in SCALING_CHOICES:
        raise ValueError(
            "sigma_scaling must be one of {}, not {!r}".format(
                tuple(SCALING_CHOICES), sigma_scaling))
    if parameters_file is None:
        parameters_file = DEFAULT_PARAMETERS_FILE
        if sigma_scaling == 'sqrtk':
            parameters_file = parameters_file.with_name(
                parameters_file.stem + '_sqrtk' + parameters_file.suffix)
    parameters_file = Path(parameters_file)
    parameters_file.parent.mkdir(parents=True, exist_ok=True)
    replicates = tuple(
        range(N_REPLICATES) if replicates is None else replicates)
    deviations = tuple(
        SCAN_DEVIATIONS if deviations is None else deviations)
    n_covariates = tuple(
        N_COVARIATES if n_covariates is None else n_covariates)
    if (not replicates or len(set(replicates)) != len(replicates) or
            any(not isinstance(value, int) or value < 0
                for value in replicates)):
        raise ValueError('replicates must be unique non-negative integers')
    if (not deviations or len(set(deviations)) != len(deviations) or
            any(not isinstance(value, int) for value in deviations)):
        raise ValueError('deviations must be unique integers')
    allowed_deviations = set(SCAN_DEVIATIONS) | set(EXPLORATORY_DEVIATIONS)
    if not set(deviations) <= allowed_deviations:
        raise ValueError(
            'deviations must be registered in dimensionality_config')
    if (not n_covariates or
            len(set(n_covariates)) != len(n_covariates) or
            any(not isinstance(value, int) or value <= 0
                for value in n_covariates)):
        raise ValueError('n_covariates must be unique positive integers')
    if not set(n_covariates) <= set(N_COVARIATES):
        raise ValueError(
            'n_covariates must be registered in dimensionality_config')
    if not isinstance(fixed_rng_seed, int) or fixed_rng_seed < 0:
        raise ValueError('fixed_rng_seed must be a non-negative integer')

    params = {
        param: [center + deviation * step
                for deviation in deviations]
        for param, (center, step) in SCAN_CENTERS.items()
    }

    scaling_field = '' if sigma_scaling == '1k' else ' sqrtk'
    lines = []
    for replicate in replicates:
        for key, values in params.items():
            for value in values:
                for k in n_covariates:
                    lines.append(
                        f'{key} {value:.6f} {k} '
                        f'{replicate}{scaling_field}\n')
    manifest_content = ''.join(lines)
    write_atomic(parameters_file, manifest_content)
    total = len(lines)

    if design_file is None:
        design_file = parameters_file.with_name(
            parameters_file.stem + '.design.json')
    design_file = Path(design_file)
    design = {
        'design_version': 2,
        'result_schema_version': RESULT_SCHEMA_VERSION,
        'experiment_design': DIMENSIONALITY_EXPERIMENT_DESIGN,
        'agent_model': AGENT_MODEL,
        'manifest': parameters_file.name,
        'manifest_sha256': hashlib.sha256(
            manifest_content.encode('utf-8')).hexdigest(),
        'line_count': total,
        'parameters': {
            key: {'center': center, 'step': step}
            for key, (center, step) in SCAN_CENTERS.items()
        },
        'deviations': list(deviations),
        'n_covariates': list(n_covariates),
        'replicates': list(replicates),
        'scaling': sigma_scaling,
        'fixed_rng_seed': fixed_rng_seed,
        'observation_seed_scheme': OBSERVATION_SEED_SCHEME,
        'pairing': DIMENSIONALITY_PAIRING,
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

    print(f"Wrote {total} lines to {parameters_file}.")
    print(
        f"Parameter grid: {list(params)} x k={list(n_covariates)} "
        f"x {len(replicates)} replicates")
    print(f"Design metadata: {design_file}")
    return parameters_file


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--sigma-scaling',
        choices=tuple(SCALING_CHOICES),
        default=DEFAULT_SCALING)
    parser.add_argument('--output')
    parser.add_argument(
        '--replicates', default='0-{}'.format(N_REPLICATES - 1),
        help="inclusive range (for example '0-49') or comma-separated IDs")
    parser.add_argument(
        '--deviations', nargs='+', type=int, default=SCAN_DEVIATIONS)
    parser.add_argument(
        '--n-covariates', nargs='+', type=int, default=N_COVARIATES)
    parser.add_argument(
        '--fixed-rng-seed', type=int, default=FIXED_RNG_SEED)
    parser.add_argument('--design-output')
    args = parser.parse_args()
    try:
        replicates = parse_integer_selection(args.replicates)
    except ValueError as exc:
        parser.error(str(exc))
    generate_slurm_parameter_file(
        parameters_file=args.output,
        sigma_scaling=args.sigma_scaling,
        replicates=replicates,
        deviations=args.deviations,
        n_covariates=args.n_covariates,
        fixed_rng_seed=args.fixed_rng_seed,
        design_file=args.design_output)
