#!/bin/bash
#SBATCH -p rome
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --job-name=dim_bookkeeping
#SBATCH --output=logs/dim_bookkeeping_%j.out
#SBATCH --error=logs/dim_bookkeeping_%j.err

# Compiles the per-run dimensionality results and runs the completion
# check on a compute node instead of the shared login node. Both steps
# read many small files from the shared filesystem and can take several
# minutes; that I/O load doesn't belong on a login node shared with other
# users. Neither step does numeric computation, so a single core and
# modest memory are enough - --time may need to grow as results/dim_runs
# accumulates more files (compiling took ~1 minute at ~4000 files, and
# several minutes once past ~6000-7000 in earlier testing).
#
# Submit from the repo root:
#
#     sbatch slurm/compile_and_check_dimensionality.sh

module purge
module load 2025
module load Python/3.13.1-GCCcore-14.2.0

source ~/env/bin/activate

cd ~/CapturingUsersPredictionModels

python slurm/compile_dimensionality_results.py --force
python slurm/completion_check_dimensionality.py
