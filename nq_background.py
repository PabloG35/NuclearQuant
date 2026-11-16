"""Nuclear-Quant step 1b: local background of the marker around every nucleus.

Reads the finished segmentation of ONE image (nuclei_labels.tif + objects.tsv + parameters.json written by
nq_segment.py) and the same CZI, and for each nucleus measures the marker in a thin shell of background voxels
around it (3D, true um distances; voxels that belong to other nuclei are never part of a shell). It writes
objects_bg.tsv = objects.tsv plus, for marker M:
  M_shell_median   median marker intensity in the shell (robust to bright puncta)
  M_shell_n_vox    voxels in the shell (net/ratio are empty below --min-shell-vox)
  M_net            M_mean_raw - M_shell_median      (can be negative)
  M_ratio          M_mean_raw / M_shell_median      (fold over local background)
The segmentation is NOT rerun. 8-bit stacks are read on the same scale as the original run (x257, or native 8-bit
for results_native8), so the columns stay consistent with M_mean_raw.

Then classify with the new columns, e.g.
  python nq_classify.py --marker ERG --column ERG_ratio --method fixed --threshold 2 \
      --objects results/*/d60/objects_bg.tsv --out classify_ERG_ratio

  sbatch run_background.sbatch --czi stack.czi --result-dir results/<sample>/d60 --marker ERG
"""
import argparse, csv, json, pathlib, sys, time


def shell_background(labels, signal, spacing, shell_um, gap_um=0.):
    """Per-label median and mean of `signal` over the background shell around each label.

    labels: ZYX integer array (0 = background); spacing: (z, y, x) in um; the shell is every background voxel
    within shell_um of a nucleus, minus anything within gap_um (skips the blurred mask edge). Each shell voxel is
    given to its nearest nucleus. Returns {label: (median, mean, n_voxels)}.
    """
    import numpy as np
    from skimage.segmentation import expand_labels
    if labels.shape != signal.shape:
        raise ValueError('labels %s and signal %s differ in shape' % (labels.shape, signal.shape))
    grown = expand_labels(labels, distance=shell_um, spacing=spacing)
    shell = (grown > 0) & (labels == 0)
    if gap_um > 0:
        shell &= expand_labels(labels, distance=gap_um, spacing=spacing) == 0
    lab, val = grown[shell], signal[shell]
    del grown, shell
    if not lab.size:
        return {}
    order = np.argsort(lab, kind='stable')
    lab, val = lab[order], val[order]
    cuts = np.flatnonzero(np.diff(lab)) + 1
    firsts = lab[np.concatenate([[0], cuts])]
    out = {}
    for l, v in zip(firsts.tolist(), np.split(val, cuts)):
        out[int(l)] = (float(np.median(v)), float(v.mean()), int(v.size))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--czi', required=True, type=pathlib.Path)
    ap.add_argument('--result-dir', required=True, type=pathlib.Path, help='results/<sample>/d60 of the finished run')
    ap.add_argument('--marker', required=True, help='marker label used in nq_segment.py (e.g. ERG)')
    ap.add_argument('--shell-um', type=float, default=3.0, help='shell width around each nucleus (default 3 um)')
    ap.add_argument('--gap-um', type=float, default=0.5, help='skip this distance next to the mask (default 0.5 um)')
    ap.add_argument('--min-shell-vox', type=int, default=50, help='net/ratio left empty below this many shell voxels')
    ap.add_argument('--force', action='store_true', help='overwrite an existing objects_bg.tsv')
    a = ap.parse_args()

    rd = a.result_dir
    par = json.loads((rd / 'parameters.json').read_text())
    if not (rd / 'complete.json').exists():
        raise SystemExit('%s has no complete.json: segmentation not finished' % rd)
    if a.marker not in par['marker_channels']:
        raise SystemExit('marker %r not in %s' % (a.marker, list(par['marker_channels'])))
    out_tsv = rd / 'objects_bg.tsv'
    if out_tsv.exists() and not a.force:
        raise SystemExit('%s exists; use --force to overwrite' % out_tsv)

    from nq_segment import read_czi, sha256
    import numpy as np, tifffile
    if sha256(a.czi) != par['input_sha256']:
        raise SystemExit('%s is not the file that produced %s (SHA-256 differs)' % (a.czi, rd))
    native = par['source_dtype'] == 'uint8' and not par['converted_8bit_to_16bit']
    raw, dyes, scale, _ = read_czi(a.czi, native)
    if dyes != par['channels_in_file']:
        raise SystemExit('channels differ from the original run: %s vs %s' % (dyes, par['channels_in_file']))
    sig = raw[dyes.index(par['marker_channels'][a.marker])]
    del raw
    z0, y0, x0 = par['crop_origin_zyx']; nz, ny, nx = par['shape_zyx']
    sig = sig[z0:z0 + nz, y0:y0 + ny, x0:x0 + nx]
    labels = tifffile.imread(rd / 'nuclei_labels.tif')
    if labels.shape != sig.shape:
        raise SystemExit('labels %s vs marker %s: wrong CZI or crop' % (labels.shape, sig.shape))

    t0 = time.time()
    spacing = (scale['Z'], scale['Y'], scale['X'])
    bg = shell_background(labels, sig, spacing, a.shell_um, a.gap_um)

    m = a.marker
    new = ['%s_shell_median' % m, '%s_shell_n_vox' % m, '%s_net' % m, '%s_ratio' % m]
    with (rd / 'objects.tsv').open() as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    n_short = 0
    for r in rows:
        med, _, n = bg.get(int(r['label']), (float('nan'), float('nan'), 0))
        mean = float(r['%s_mean_raw' % m])
        ok = n >= a.min_shell_vox
        n_short += not ok
        r[new[0]] = '%.6g' % med if n else ''
        r[new[1]] = n
        r[new[2]] = '%.6g' % (mean - med) if ok else ''
        r[new[3]] = '%.6g' % (mean / med) if ok and med > 0 else ''
    with out_tsv.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter='\t'); w.writeheader(); w.writerows(rows)
    summary = dict(marker=m, shell_um=a.shell_um, gap_um=a.gap_um, min_shell_vox=a.min_shell_vox, n_objects=len(rows),
                   n_without_valid_shell=n_short, native_depth=native, elapsed_seconds=time.time() - t0,
                   output=str(out_tsv))
    (rd / 'background.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
