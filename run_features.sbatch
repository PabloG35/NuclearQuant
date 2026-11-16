#!/bin/bash
# Features (local background, percentile, vessel distance) for one finished image. From the Nuclear-Quant directory:
#   sbatch run_features.sbatch --czi "00_raw/<file>.czi" --result-dir results/<sample>/d60 --marker ERG --cd31 488
#SBATCH --job-name=nuclear_quant_feat
#SBATCH --partition=all
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=0
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1
HERE=/home/users/pablogaxiola/Imaging/Nuclear-Quant
export PYTHONPATH=$HERE/deps:$HERE   # czifile reader + nq_segment / nq_background helpers
/home/users/pablogaxiola/conda/envs/image_analysis/bin/python "$HERE/nq_features.py" "$@"
