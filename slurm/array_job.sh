#!/bin/bash
#SBATCH -p rome
#SBATCH --array=0-2  # Number of jobs. Should be 1/4 of #lines in parameters.txt.
#SBATCH --cpus-per-task=16  # Minimum allocation. Might make sense to up this to 128 (on thin node) for file transfers.
#SBATCH --time=05:00
#SBATCH --requeue  # Resubmit job if crashed due to system
#SBATCH --output=logs/job_%A_%a.out
#SBATCH --error=logs/job_%A_%a.err

# Load suitable modules
module purge
module load 2025
module load OpenBLAS/0.3.29-GCC-14.2.0
module load Python/3.13.1-GCCcore-14.2.0

# Load virtualenvironment with appropriate libraries
source ~/env/bin/activate

# Thread control
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

# Batch of 4 instances to run in *this task
# Start-line for batch (notice 1-indexing)
START_LINE=$((SLURM_ARRAY_TASK_ID * 4 + 1))
# Parameter values for *this batch
PARAMS=()
for i in {0..3}; do
    PARAMS[$i]=$(sed -n "$((START_LINE + i))p" ~/CapturingUsersPredictionModels/slurm/tmp/parameters.txt)
done

# Run experiment batch (4 processes in parallel!)
for i in {0..3}; do
    if [ -n "${PARAMS[$i]}" ]; then
        python experiment.py ${PARAMS[$i]} &
    fi
done

wait

echo "Job $SLURM_ARRAY_TASK_ID completed"
