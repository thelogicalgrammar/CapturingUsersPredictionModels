"""
Checks which jobs from the dimensionality experiment finished and writes
a new parameter file for re-running the ones that didn't.

Unlike the scalar completion_check.py (which matches on X_o), this script
uses a count-based approach: X_o is sampled internally and not stored in
the parameter file, so we count how many results exist per
(param_name, param_value, n_covariates) and re-queue the deficit.

Usage::

    python slurm/completion_check_dimensionality.py

After running, update #SBATCH --array in array_job_dimensionality.sh
to (lines in parameters_dimensionality.txt / 4) - 1.
"""
import csv
from collections import Counter


RESULTS_FILE    = './results/results_dimensionality.csv'
ORIGINAL_PARAMS = './slurm/tmp/original_parameters_dimensionality.txt'
NEW_PARAMS      = './slurm/tmp/parameters_dimensionality.txt'


def run_completion_check():
    # 1. Count completed runs per (param_name, param_value, n_covariates).
    completed = Counter()
    with open(RESULTS_FILE, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        name_idx  = headers.index('Modified prior')
        value_idx = headers.index('New value of modified prior')
        k_idx     = headers.index('n_covariates')
        for row in reader:
            key = (row[name_idx], float(row[value_idx]), int(float(row[k_idx])))
            completed[key] += 1

    # 2. Count planned runs per (param_name, param_value, n_covariates).
    planned = Counter()
    all_rows = []
    with open(ORIGINAL_PARAMS, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            name, value, k = parts[0], float(parts[1]), int(parts[2])
            key = (name, value, k)
            planned[key] += 1
            all_rows.append((name, value, k))

    # 3. Compute deficit for each key and write re-run lines.
    uncompleted_jobs = []
    for key, count in planned.items():
        deficit = count - completed.get(key, 0)
        for _ in range(deficit):
            uncompleted_jobs.append(key)

    # Preserve the original ordering (first-seen key order) for reproducibility.
    seen_order = []
    seen_set = set()
    for name, value, k in all_rows:
        key = (name, value, k)
        if key not in seen_set:
            seen_set.add(key)
            seen_order.append(key)
    uncompleted_jobs.sort(
        key=lambda x: seen_order.index(x) if x in seen_set else len(seen_order)
    )

    with open(NEW_PARAMS, 'w') as f:
        for name, value, k in uncompleted_jobs:
            f.write('{} {:.6f} {}\n'.format(name, value, k))

    if not uncompleted_jobs:
        print("All jobs are completed.")
    else:
        n_tasks = max(1, len(uncompleted_jobs) // 4)
        print('{} uncompleted job runs.'.format(len(uncompleted_jobs)))
        print('Parameters written to {}'.format(NEW_PARAMS))
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
