"""Generate a manifest with an independent dataset and patient per run."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dimensionality_config import (  # noqa: E402
    AGENT_MODEL,
    A0_TREATMENT,
    A1_TREATMENT,
    DECISION_THRESHOLD,
    N_CHAINS,
    N_COVARIATES,
    N_DRAWS,
    N_TUNE,
    PLATE_SIZE,
    SCAN_CENTERS,
    TARGET_ACCEPT,
)
from experiment_randomness import (  # noqa: E402
    BASE_SEED,
    NO_INTERCEPT_SCAN_CENTERS,
    RESULT_SCHEMA_VERSION,
    SCHEME,
)
from slurm.generate_parameter_file_dimensionality import (  # noqa: E402
    parse_integer_selection,
    write_atomic,
)


DESIGNS = {
    'univariate_no_intercept': {
        'agent_model': 'scalar_sign_marginalized',
        'centers': NO_INTERCEPT_SCAN_CENTERS,
        'deviations': tuple(range(-30, 31)),
        'n_covariates': (1,),
        'scaling': '1k',
        'fit_intercept': False,
    },
    'univariate_intercept': {
        'agent_model': AGENT_MODEL,
        'centers': SCAN_CENTERS,
        'deviations': tuple(range(-30, 31)),
        'n_covariates': (1,),
        'scaling': '1k',
        'fit_intercept': True,
    },
    'multivariate_intercept': {
        'agent_model': AGENT_MODEL,
        'centers': SCAN_CENTERS,
        'deviations': (-20, -10, -3, -1, 0, 1, 3, 10, 20),
        'n_covariates': N_COVARIATES,
        'scaling': 'sqrtk',
        'fit_intercept': True,
    },
}


def generate_manifest(
        experiment_design, output, replicates=None, deviations=None,
        n_covariates=None, base_seed=BASE_SEED, design_output=None):
    """Write a manifest and its design sidecar."""
    design = DESIGNS[experiment_design]
    replicates = tuple(
        range(50) if replicates is None else replicates)
    deviations = tuple(
        design['deviations'] if deviations is None else deviations)
    n_covariates = tuple(
        design['n_covariates']
        if n_covariates is None else n_covariates)
    if (not replicates or len(set(replicates)) != len(replicates) or
            any(not isinstance(value, int) or value < 0
                for value in replicates)):
        raise ValueError(
            'replicates must be unique non-negative integers')
    if (not deviations or len(set(deviations)) != len(deviations) or
            any(not isinstance(value, int) or not -30 <= value <= 30
                for value in deviations)):
        raise ValueError(
            'deviations must be unique integers from -30 through 30')
    if (not n_covariates or
            len(set(n_covariates)) != len(n_covariates) or
            not set(n_covariates) <= set(design['n_covariates'])):
        raise ValueError(
            'n_covariates are incompatible with the experiment design')
    if not isinstance(base_seed, int) or base_seed < 0:
        raise ValueError('base_seed must be a non-negative integer')

    lines = []
    for param, (center, step) in design['centers'].items():
        for deviation in deviations:
            value = center + deviation * step
            for k in n_covariates:
                for replicate in replicates:
                    lines.append(
                        '{} {:.6f} {} {} {} {}\n'.format(
                            param, value, k, replicate,
                            design['scaling'], deviation))
    content = ''.join(lines)
    output = Path(output)
    write_atomic(output, content)
    if design_output is None:
        design_output = output.with_name(
            output.stem + '.design.json')
    design_output = Path(design_output)
    metadata = {
        'design_version': 1,
        'result_schema_version': RESULT_SCHEMA_VERSION,
        'experiment_design': experiment_design,
        'agent_model': design['agent_model'],
        'coefficient_mode': 'fixed',
        'manifest': output.name,
        'manifest_sha256': hashlib.sha256(
            content.encode('utf-8')).hexdigest(),
        'line_count': len(lines),
        'parameters': {
            key: {'center': center, 'step': step}
            for key, (center, step) in design['centers'].items()
        },
        'deviations': list(deviations),
        'n_covariates': list(n_covariates),
        'replicates': list(replicates),
        'scaling': design['scaling'],
        'base_seed': base_seed,
        'seed_scheme': SCHEME,
        'randomization': {
            'historical_data': 'independent_per_manifest_row',
            'observation': 'independent_per_manifest_row',
            'sampler': 'independent_per_manifest_row',
        },
        'sampler': {
            'draws': N_DRAWS,
            'tune': N_TUNE,
            'chains': N_CHAINS,
            'target_accept': TARGET_ACCEPT,
            'max_treedepth': 15,
            'init': 'jitter+adapt_diag',
        },
        'experiment': {
            'plate_size': PLATE_SIZE,
            'a1_treatment': A1_TREATMENT,
            'a0_treatment': A0_TREATMENT,
            'decision_threshold': DECISION_THRESHOLD,
            'fit_intercept': design['fit_intercept'],
        },
    }
    write_atomic(
        design_output,
        json.dumps(metadata, indent=2, sort_keys=True) + '\n')
    print('Wrote {} lines to {}.'.format(len(lines), output))
    print('Design metadata: {}'.format(design_output))
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--experiment-design', choices=tuple(DESIGNS), required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--design-output')
    parser.add_argument('--replicates', default='0-49')
    parser.add_argument('--deviations', nargs='+', type=int)
    parser.add_argument('--n-covariates', nargs='+', type=int)
    parser.add_argument('--base-seed', type=int, default=BASE_SEED)
    args = parser.parse_args()
    try:
        replicate_ids = parse_integer_selection(args.replicates)
        generate_manifest(
            args.experiment_design,
            args.output,
            replicates=replicate_ids,
            deviations=args.deviations,
            n_covariates=args.n_covariates,
            base_seed=args.base_seed,
            design_output=args.design_output)
    except ValueError as exc:
        parser.error(str(exc))
