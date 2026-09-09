"""
Compile the per-run dimensionality result files in ``results/dim_runs/``
into a single consolidated CSV that ``utils.plot_dimensionality`` and
``utils.plot_dimensionality_improvement`` can read directly.

experiment_dimensionality.py writes one atomic CSV per run instead of
appending to a shared file (see ``_publish_without_overwrite`` in that
module), because rows for different covariate dimensionalities carry
different extra columns (``X_o_0``, ``X_o_1``, ..., ``prediction_model_coef_0``,
...) and ``utils.write_results_to_file``'s schema check rejects any header
mismatch when appending to a shared file. This script concatenates the
per-run files with pandas instead, which aligns rows by column name and
fills a row's missing extra columns with NaN rather than requiring
identical headers. The plotting functions only read the fixed columns
(prior/value/CATE/X_o), so the differing extra columns are harmless.

Run from the repository root::

    python slurm/compile_dimensionality_results.py

By default this refuses to overwrite an existing output file; pass
--force to replace it.
"""
import argparse
from pathlib import Path

import pandas as pd


DEFAULT_RUNS_DIR = Path('./results/dim_runs')
DEFAULT_OUTPUT = Path('./results/results_dimensionality.csv')


def compile_dimensionality_results(
        runs_dir=DEFAULT_RUNS_DIR, output=DEFAULT_OUTPUT, force=False):
    """
    Concatenate every per-run dimensionality CSV into one results file.

    Args:
        runs_dir (str or Path): Directory of per-run CSVs written by
            experiment_dimensionality.py (default './results/dim_runs').
        output (str or Path): Path for the compiled CSV (default
            './results/results_dimensionality.csv', the path
            ``utils.plot_dimensionality`` and
            ``utils.plot_dimensionality_improvement`` read by default).
        force (bool): Overwrite an existing output file. Off by default: a
            stale compiled file should be replaced deliberately, not
            silently, the same way a per-run file is never silently
            overwritten.

    Returns:
        pathlib.Path: Path of the written file.
    """
    runs_dir = Path(runs_dir)
    output = Path(output)
    run_files = sorted(runs_dir.glob('results_*.csv'))
    if not run_files:
        raise FileNotFoundError(
            'no per-run result files found in {}'.format(runs_dir))
    if output.exists() and not force:
        raise FileExistsError(
            'refusing to overwrite existing compiled results: {} '
            '(pass force=True / --force to replace it)'.format(output))

    frames = [pd.read_csv(run_file) for run_file in run_files]
    combined = pd.concat(frames, ignore_index=True, sort=False)

    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output, index=False)
    print('Compiled {} runs from {} into {}'.format(
        len(combined), runs_dir, output))
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-dir', default=str(DEFAULT_RUNS_DIR))
    parser.add_argument('--output', default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        '--force', action='store_true',
        help='overwrite an existing output file')
    args = parser.parse_args()
    compile_dimensionality_results(
        runs_dir=args.runs_dir, output=args.output, force=args.force)
