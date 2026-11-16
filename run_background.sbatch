#!/bin/bash
# Local-background step for one finished image. Submit from the Nuclear-Quant directory:
#   sbatch run_background.sbatch --czi /path/to/stack.czi --result-dir results/<sample>/d60 --marker ERG
#SBATCH --job-name=nuclear_quant_bg
#SBATCH --partition=all
#SBATCH --cpus-per-task=4
#SBATCH --mem=45G
#SBATCH --time=0
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1
HERE=/home/users/pablogaxiola/Imaging/Nuclear-Quant
export PYTHONPATH=$HERE/deps:$HERE   # czifile reader + nq_segment helpers
/home/users/pablogaxiola/conda/envs/image_analysis/bin/python "$HERE/nq_background.py" "$@"
