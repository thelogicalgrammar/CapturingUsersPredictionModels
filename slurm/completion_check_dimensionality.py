"""
Checks which jobs from the dimensionality experiment finished and writes
a new parameter file for re-running the ones that didn't.

Unlike the scalar completion_check.py (which matches on X_o), this script
matches on exact (param_name, param_value, n_covariates, seed, scaling)
identity: X_o is sampled internally and not stored in the parameter file,
so the run seed stands in for identity instead. That seed is not a plain
replicate index - it is a reproducible value derived from (param_name,
param_value, n_covariates, replicate slot, scaling) via
dimensionality_config.replicate_seed(), so distinct misaligned-prior
values don't share a random stream. Every completed row already carries
its seed and scaling in the 'seed' and 'coef_sigma_scaling' columns
(merged in from experiment metadata), so re-run lines can name the exact
missing job rather than a bare count.

Usage::

    python slurm/completion_check_dimensionality.py

After running, update #SBATCH --array in array_job_dimensionality.sh
to (lines in parameters_dimensionality.txt / 4) - 1.
"""
import csv
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dimensionality_config import DEFAULT_SCALING, SCALING_CHOICES  # noqa: E402


RESULTS_FILE    = './results/results_dimensionality.csv'
ORIGINAL_PARAMS = './slurm/tmp/original_parameters_dimensionality.txt'
NEW_PARAMS      = './slurm/tmp/parameters_dimensionality.txt'

# Results store the long form ('1/k', '1/sqrt(k)'); manifest lines store the
# short key ('1k', 'sqrtk'). Both directions are needed to match identity.
_SCALING_LONG_TO_SHORT = {long: short for short, long in SCALING_CHOICES.items()}


def _read_planned(original_params=ORIGINAL_PARAMS):
    """Return the ordered list of exact planned (name, value, k, seed, scaling) jobs."""
    planned = []
    with open(original_params, 'r') as f:
        for line_number, line in enumerate(f, start=1):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) not in (4, 5):
                raise ValueError(
                    'line {} has {} fields, expected 4 or 5: {!r}'.format(
                        line_number, len(parts), line))
            name, value, k, seed = parts[0], float(parts[1]), int(parts[2]), int(parts[3])
            scaling = parts[4] if len(parts) == 5 else DEFAULT_SCALING
            if scaling not in SCALING_CHOICES:
                raise ValueError(
                    'line {} has unknown scaling {!r}'.format(line_number, scaling))
            planned.append((name, value, k, seed, scaling))
    return planned


def _read_completed(results_file=RESULTS_FILE):
    """Return the set of exact completed (name, value, k, seed, scaling) jobs."""
    completed = set()
    with open(results_file, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        name_idx    = headers.index('Modified prior')
        value_idx   = headers.index('New value of modified prior')
        k_idx       = headers.index('n_covariates')
        seed_idx    = headers.index('seed')
        scaling_idx = headers.index('coef_sigma_scaling')
        min_cols = max(name_idx, value_idx, k_idx, seed_idx, scaling_idx) + 1
        for row in reader:
            if len(row) < min_cols:
                continue
            scaling_long = row[scaling_idx]
            if scaling_long not in _SCALING_LONG_TO_SHORT:
                continue
            key = (
                row[name_idx],
                float(row[value_idx]),
                int(float(row[k_idx])),
                # seed is a large (up to 2**63) derived value, not a
                # small replicate index - round-tripping it through
                # float() loses precision (doubles carry ~15-17
                # significant digits), silently corrupting the match.
                int(row[seed_idx]),
                _SCALING_LONG_TO_SHORT[scaling_long],
            )
            completed.add(key)
    return completed


def run_completion_check(
        results_file=RESULTS_FILE, original_params=ORIGINAL_PARAMS,
        new_params=NEW_PARAMS):
    completed = _read_completed(results_file)
    planned = _read_planned(original_params)

    # Preserve original manifest order; drop duplicate planned lines (a
    # replicate can only ever be planned once) while keeping first-seen order.
    seen = set()
    uncompleted_jobs = []
    for job in planned:
        if job in seen:
            continue
        seen.add(job)
        if job not in completed:
            uncompleted_jobs.append(job)

    with open(new_params, 'w') as f:
        for name, value, k, seed, scaling in uncompleted_jobs:
            scaling_field = '' if scaling == DEFAULT_SCALING else ' {}'.format(scaling)
            f.write('{} {:.6f} {} {}{}\n'.format(name, value, k, seed, scaling_field))

    if not uncompleted_jobs:
        print("All jobs are completed.")
    else:
        # Ceiling division: a remainder must still get its own task, or
        # the last 1-3 lines are silently never assigned to any task.
        n_tasks = (len(uncompleted_jobs) + 3) // 4
        print('{} uncompleted job runs.'.format(len(uncompleted_jobs)))
        print('Parameters written to {}'.format(new_params))
        print('Set #SBATCH --array=0-{} in array_job_dimensionality.sh'.format(
            n_tasks - 1))
        print('Consider adding more time to the cluster allocation.')


def summarize_results(result_file=RESULTS_FILE):
    """
    Print counts of completed runs per (param_name, n_covariates, value).
    """
    with open(result_file, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        name_idx  = headers.index('Modified prior')
        value_idx = headers.index('New value of modified prior')
        k_idx     = headers.index('n_covariates')
        rows = list(reader)

    counts = Counter(
        (row[name_idx], int(float(row[k_idx])), float(row[value_idx]))
        for row in rows
    )

    by_name_k = {}
    for (name, k, value), count in counts.items():
        by_name_k.setdefault((name, k), {})[value] = count

    for (name, k) in sorted(by_name_k):
        print('=' * 40)
        print('{} (k={})'.format(name, k))
        print('=' * 40)
        for value in sorted(by_name_k[(name, k)]):
            print('  {}: {}'.format(value, by_name_k[(name, k)][value]))


if __name__ == '__main__':
    run_completion_check()
