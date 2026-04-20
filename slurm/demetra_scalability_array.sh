#!/bin/bash
#SBATCH --job-name=multivariate_scaling
#SBATCH --partition=turing-wide
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-0
#SBATCH --output=/share/malelab/dtugnoli/2-step_agent/logs/%x_%A_%a.out
#SBATCH --error=/share/malelab/dtugnoli/2-step_agent/logs/%x_%A_%a.err

set -euo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/share/malelab/dtugnoli/2-step_agent}"
PROJECT_ROOT="${PROJECT_ROOT:-${WORKSPACE_ROOT}/CapturingUsersPredictionModels}"
CONDA_ENV="${CONDA_ENV:-/share/malelab/dtugnoli/conda_envs/capturing_users_prediction_models_cpu}"
PARAMETERS_FILE="${PARAMETERS_FILE:-${WORKSPACE_ROOT}/slurm/tmp/scalability_parameters.txt}"
LOG_DIR="${WORKSPACE_ROOT}/logs"

module purge
module load miniconda3/22.11.1-gcc-13.2.0-tn534fv
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export PYTENSOR_FLAGS="base_compiledir=${TMPDIR:-/tmp}/pytensor_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

mkdir -p "${LOG_DIR}"
cd "${PROJECT_ROOT}"

echo "Workspace root: ${WORKSPACE_ROOT}"
echo "Project root: ${PROJECT_ROOT}"
echo "Conda env: ${CONDA_ENV}"
echo "Parameters file: ${PARAMETERS_FILE}"
echo "Slurm task id: ${SLURM_ARRAY_TASK_ID}"

PARAMS=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${PARAMETERS_FILE}")
if [ -z "${PARAMS}" ]; then
    echo "No parameters found for task ${SLURM_ARRAY_TASK_ID} in ${PARAMETERS_FILE}"
    exit 1
fi

echo "Resolved parameters: ${PARAMS}"
python scalability_experiment.py ${PARAMS}
