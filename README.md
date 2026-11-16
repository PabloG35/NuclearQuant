# Nuclear-Quant

Reusable counting of nuclear markers (c-Fos, ERG, Lef1, Sox17, NeuN, ...) in 3D CZI stacks, using a
nuclear counterstain (Hoechst 33342) as the mask. Logic taken from the NeuN pilot
(`../NeuN_cortex/.../segmentation_size_sweep_v1`, `../NeuN_Quantification/NeuN-Quant-0000`), made marker-agnostic.

## Method
1. `nq_segment.py` — Cellpose-SAM 4.1.1 (`cpsam`, CPU) 3D segmentation of the nuclear channel only:
   diameter **60 px** (default; the NeuN pilot showed no notable difference vs 45 px, 30 px was cancelled),
   norm3D percentiles 1–99, cellprob 0, min 15 voxels, Z/XY anisotropy from the CZI, original voxel grid.
   Marker channels are only measured (mean raw intensity per nucleus). Outputs per sample in
   `results/<sample>/d60/`: `parameters.json` (incl. input SHA-256), `nuclei_labels.tif`, `objects.tsv`,
   `overlay_z*.png`, `complete.json`. Refuses to overwrite a completed result.
2. `nq_classify.py` — per marker, positive/negative call and counts per image (`counts.tsv`, `method.json`,
   `QC_distribution.png`). Methods: `otsu` (pooled log1p, as in NeuN; exploratory), `control` (percentile of a
   negative-control image), `fixed`.
3. `run.sbatch` — Slurm wrapper for step 1 (6 CPUs, 96 GB, no time limit, as the NeuN runs).

## Bit depth and comparability (fail-safe)
- Every stack is put on the 16-bit scale: 8-bit data are multiplied by 257 (exact round trip). `parameters.json`
  records `source_dtype`, `converted_8bit_to_16bit`, per-channel `distinct_levels` and `fraction_saturated`, and the
  marker channel's acquisition settings (gain/voltage/pinhole... as found in the CZI). Warnings go to stderr.
- The x257 scaling does not restore the 256-level quantisation of an 8-bit image, nor does it make images taken with
  different laser/gain comparable. `nq_classify.py` therefore REFUSES pooled thresholds (`otsu`, `control`, `fixed`)
  when images mix 8/16-bit or differ in the marker channel's settings; use `--method otsu-per-image`, split the
  images, or pass `--allow-mixed` (warning kept in `method.json`). `counts.tsv` lists each image's `source_dtype`.
- `python tests/selftest.py` checks this logic on synthetic data (no CZI needed).

## Use
```
python nq_segment.py --czi stack.czi --list-channels        # login node is fine, reads metadata only
mkdir -p logs
sbatch run.sbatch --czi stack.czi --nuclear "Hoechst 33342" --marker "cFos=Alexa Fluor 594" --out-root results
python nq_classify.py --marker cFos --objects results/*/d60/objects.tsv --method otsu --out classify_cFos
```
Channels are chosen by dye name ("Hoechst", "Alexa Fluor 647") or an explicit index `#2`; a bare number is a wavelength, not an index. Several `--marker LABEL=CHANNEL` can be given for a multi-marker stack. Pilot: add `--crop-xy 256`.
Python env: `~/conda/envs/image_analysis`; the CZI reader lives in `deps/`.

## Caveats
- Counts are per image, nuclear objects, not validated stereological totals. Border nuclei are kept
  (`*_interior` columns exclude them).
- The NeuN cutoff was an exploratory Otsu with no negative control analysed. For rare-positive markers (c-Fos)
  Otsu is unreliable: prefer a negative-control image (`--method control`).
- 60 px (12.4 µm at 0.207 µm/px) was tuned on cortical neuron nuclei; endothelial nuclei (ERG, Sox17) are smaller
  and elongated: check overlays before trusting counts. Compare images only at the same resolution.
- Nothing here has been run yet on a new marker.
