#!/bin/bash
# Submit from the Nuclear-Quant directory, after `mkdir -p logs`. All arguments go to nq_segment.py:
#   sbatch run.sbatch --czi /path/to/stack.czi --nuclear "Hoechst 33342" --marker "cFos=Alexa Fluor 594" --out-root results
#SBATCH --job-name=nuclear_quant
#SBATCH --partition=all
#SBATCH --cpus-per-task=6
#SBATCH --mem=45G
#SBATCH --time=0
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1
HERE=/home/users/pablogaxiola/Imaging/Nuclear-Quant
export PYTHONPATH=$HERE/deps   # isolated czifile reader (copied from the NeuN pilot)
/home/users/pablogaxiola/conda/envs/image_analysis/bin/python "$HERE/nq_segment.py" "$@"
