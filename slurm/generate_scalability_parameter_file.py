"""
Generate a parameter file for Demetra scalability runs.
"""
import argparse
import itertools
from pathlib import Path


def parse_args():
    """
    Parse CLI arguments.
    """
    parser = argparse.ArgumentParser(description="Generate scalability parameters for Slurm array jobs.")
    parser.add_argument('--output-file', type=Path,
                        default=Path('./slurm/tmp/scalability_parameters.txt'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--k-values', type=int, nargs='+', default=[1, 2, 3, 5, 10])
    parser.add_argument('--plate-sizes', type=int, nargs='+', default=[1000])
    parser.add_argument('--seeds', type=int, nargs='+', default=[4])
    parser.add_argument('--draws', type=int, default=1000)
    parser.add_argument('--tune', type=int, default=1000)
    parser.add_argument('--chains', type=int, default=4)
    parser.add_argument('--target-accept', type=float, default=.85)
    parser.add_argument('--max-treedepth', type=int, default=15)
    parser.add_argument('--init', type=str, default='jitter+adapt_diag')
    parser.add_argument('--coefficient-scaling', choices=('inverse_k', 'none'),
                        default='inverse_k')
    parser.add_argument('--x-o-mode', choices=('mean', 'sample'), default='mean')
    return parser.parse_args()


def main():
    """
    Generate the parameter file.
    """
    args = parse_args()
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for n_covariates, plate_size, seed in itertools.product(
            args.k_values, args.plate_sizes, args.seeds):
        run_name = f"k{n_covariates}_plate{plate_size}_seed{seed}"
        line = (
            f"--n-covariates {n_covariates} "
            f"--plate-size {plate_size} "
            f"--draws {args.draws} "
            f"--tune {args.tune} "
            f"--chains {args.chains} "
            f"--target-accept {args.target_accept} "
            f"--max-treedepth {args.max_treedepth} "
            f"--init {args.init} "
            f"--coefficient-scaling {args.coefficient_scaling} "
            f"--x-o-mode {args.x_o_mode} "
            f"--seed {seed} "
            f"--output-dir {args.output_dir} "
            f"--run-name {run_name}"
        )
        lines.append(line)

    with args.output_file.open('w') as file:
        for line in lines:
            file.write(f"{line}\n")

    print(f"Wrote {len(lines)} parameter lines to {args.output_file}")


if __name__ == '__main__':
    main()
