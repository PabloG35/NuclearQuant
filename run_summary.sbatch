#!/bin/bash
# Per-image summary (positives, their intensity, vessel length). From the Nuclear-Quant directory:
#   sbatch run_summary.sbatch --marker ERG --column ERG_p75_ratio --threshold 2 --gate-column CD31_dist_um --gate-max 3 \
#       --result-dirs results/*/d60 --out summary_ERG
#SBATCH --job-name=nuclear_quant_summary
#SBATCH --partition=all
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=0
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1
HERE=/home/users/pablogaxiola/Imaging/Nuclear-Quant
cd "$SLURM_SUBMIT_DIR"
/home/users/pablogaxiola/conda/envs/image_analysis/bin/python "$HERE/nq_summary.py" "$@"
