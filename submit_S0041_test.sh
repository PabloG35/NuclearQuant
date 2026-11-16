#!/bin/bash
# S0041 test set (CD31 488 / ERG 647 / H33342). Run from the Nuclear-Quant directory: bash submit_S0041_test.sh
# Starts jobs after a delay (default 2 hours) so the upload can finish; --expect-bytes aborts on a partial file.
# Main run: all 7 stacks, 8-bit ones scaled to 16-bit (results/). Comparison run: 8-bit stacks kept in 8 bits
# (results_native8/), only for the files that are really 8-bit.
set -euo pipefail
DELAY=${DELAY:-now}
DEP=${DEP:-}   # optional job id: start only after it succeeds (afterok), cancel if it fails
depargs=(); [ -n "$DEP" ] && depargs=(--dependency="afterok:$DEP" --kill-on-invalid-dep=yes)
RAW=/home/users/pablogaxiola/Imaging/Nuclear-Quant/00_raw
mkdir -p logs
while IFS='|' read -r short bytes name; do
  common=(--czi "$RAW/$name" --nuclear Hoechst --marker "ERG=Alexa Fluor 647" --expect-bytes "$bytes")
  id=$(sbatch --parsable ${depargs[@]+"${depargs[@]}"} --begin="$DELAY" --job-name="nq_$short" run.sbatch "${common[@]}" --out-root results)
  echo "main   $short job $id"
done < s0041_files.txt
while IFS='|' read -r short bytes name; do
  case "$short" in *3k*) ;; *) continue;; esac
  common=(--czi "$RAW/$name" --nuclear Hoechst --marker "ERG=Alexa Fluor 647" --expect-bytes "$bytes")
  id=$(sbatch --parsable ${depargs[@]+"${depargs[@]}"} --mem=32G --begin="$DELAY" --job-name="nq8_$short" run.sbatch "${common[@]}" --native-depth --only-if-8bit --out-root results_native8)
  echo "native $short job $id"
done < s0041_files.txt
