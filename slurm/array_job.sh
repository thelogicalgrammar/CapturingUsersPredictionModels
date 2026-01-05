#!/bin/bash
#SBATCH -p thin
#SBATCH --array=0-99  # 100 jobs. Should be 1/4 of #lines in parameters
#SBATCH --cpus-per-task=16  # Minimum allocation. Might make sense to up this to 128 (on thin node) for file transfers.
#SBATCH --mem-per-task=2G
#SBATCH --mem=28G  # Minimum allocation
#SBATCH --time=30:00
#SBATCH --requeue  # Resubmit job if crashed due to system
#SBATCH --output=logs/job_%A_%a.out
#SBATCH --error=logs/job_%A_%a.err

# Launch new tasks with 4 second intervals
sleep $((SLURM_ARRAY_TASK_ID * 4))

# Load suitable modules
module load 2022
module load Python/3.10.4-GCCcore-11.3.0
# Load virtualenvironment with appropriate libraries
source ~/venv/bin/activate

# Thread control
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

# Batch of 4 instances to run in *this task
# Start-line for batch
START_LINE=$((SLURM_ARRAY_TASK_ID * 4 + 1))
# Parameter values for this batch
PARAMS=()
for i in {0..3}; do
    PARAMS[$i]=$(sed -n "$((START_LINE + i))p" ./slurm/tmp/parameters.txt)
done

# Run experiment batch
for i in {0..3}; do
    if [ -n "${PARAMS[$i]}" ]; then
        python experiment.py ${PARAMS[$i]} &
    fi
done

wait

echo "Job $SLURM_ARRAY_TASK_ID completed"
