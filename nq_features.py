"""Nuclear-Quant step 1c: per-nucleus features to call a marker (e.g. ERG) against local background AND vessels.

Reads the finished segmentation of ONE image (nuclei_labels.tif, objects.tsv, parameters.json from nq_segment.py) and
the same CZI. Nothing is re-segmented. Writes objects_feat.tsv = objects.tsv plus, for marker M:
  M_shell_median  median of M in a thin shell of background voxels around the nucleus (other nuclei never included)
  M_net, M_ratio  M_mean_raw - shell / M_mean_raw / shell
  M_p75_raw       75th percentile of M over the nucleus voxels (--percentile); less diluted than the mean when the
                  mask is bigger than the labelled nucleus (thin, elongated endothelial nuclei)
  M_p75_ratio     M_p75_raw / shell
and, with --cd31 (e.g. --cd31 488, a dye-name substring as in nq_segment.py):
  CD31_dist_um    distance in um from the nucleus to the nearest vessel voxel (0 = overlaps a vessel)
It also writes cd31_mask.tif (uint8, 1 = vessel) and features.json (CD31 threshold, foreground fraction, ...).

Vessel mask: Gaussian smoothing (--cd31-smooth-um), automatic threshold on the whole stack (--cd31-method li|otsu|
triangle, or fix it with --cd31-threshold), then connected pieces smaller than --cd31-min-vox are dropped. CHECK IT BY
EYE with nq_call_overlay.py --show-cd31 before trusting any number that uses CD31_dist_um.

  sbatch run_features.sbatch --czi stack.czi --result-dir results/<sample>/d60 --marker ERG --cd31 488
  python nq_classify.py --marker ERG --column ERG_p75_ratio --method fixed --threshold 2 \
      --gate-column CD31_dist_um --gate-max 3 --objects results/*/d60/objects_feat.tsv --out classify_ERG
"""
import argparse, csv, json, pathlib, time


def nucleus_percentile(labels, signal, q):
    """{label: q-th percentile of `signal` over that nucleus' voxels}."""
    import numpy as np
    m = labels > 0
    lab, val = labels[m], signal[m]
    if not lab.size:
        return {}
    order = np.argsort(lab, kind='stable')
    lab, val = lab[order], val[order]
    cuts = np.flatnonzero(np.diff(lab)) + 1
    firsts = lab[np.concatenate([[0], cuts])]
    return {int(l): float(np.percentile(v, q)) for l, v in zip(firsts.tolist(), np.split(val, cuts))}


def segment_vessels(cd31, spacing, smooth_um=0.5, method='li', threshold=None, min_vox=1000):
    """Boolean ZYX vessel mask from the CD31 channel, plus a dict describing what was done."""
    import numpy as np
    from scipy import ndimage
    from skimage import filters
    sm = ndimage.gaussian_filter(cd31.astype(np.float32), [smooth_um / s for s in spacing])
    fixed = threshold is not None
    if not fixed:
        fn = dict(li=filters.threshold_li, otsu=filters.threshold_otsu, triangle=filters.threshold_triangle)[method]
        threshold = float(fn(sm[:, ::4, ::4]))
    mask = sm > threshold
    del sm
    comp, n = ndimage.label(mask)
    kept = 0
    if n:
        keep = np.bincount(comp.ravel()) >= min_vox
        keep[0] = False
        kept = int(keep.sum())
        mask = keep[comp]
    del comp
    return mask, dict(method='fixed' if fixed else method, threshold=float(threshold),
                      smooth_um=smooth_um, min_vox=min_vox, n_pieces_kept=kept, n_pieces_found=int(n),
                      foreground_fraction=float(mask.mean()))


def vessel_distance(labels, vessel, spacing):
    """{label: minimum distance (um) from that nucleus to a vessel voxel}; empty if there is no vessel at all."""
    import numpy as np
    from scipy import ndimage
    if not vessel.any():
        return {}
    d = ndimage.distance_transform_edt(~vessel, sampling=spacing)
    idx = np.unique(labels[labels > 0])
    return dict(zip(idx.tolist(), np.atleast_1d(ndimage.minimum(d, labels, idx)).tolist()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--czi', required=True, type=pathlib.Path)
    ap.add_argument('--result-dir', required=True, type=pathlib.Path)
    ap.add_argument('--marker', required=True, help='marker label used in nq_segment.py (e.g. ERG)')
    ap.add_argument('--percentile', type=float, default=75., help='nucleus percentile of the marker (default 75)')
    ap.add_argument('--shell-um', type=float, default=3.0)
    ap.add_argument('--gap-um', type=float, default=0.5)
    ap.add_argument('--min-shell-vox', type=int, default=50)
    ap.add_argument('--cd31', help='vessel channel: dye-name substring or #index (e.g. 488); omit to skip vessels')
    ap.add_argument('--cd31-smooth-um', type=float, default=0.5)
    ap.add_argument('--cd31-method', choices=['li', 'otsu', 'triangle'], default='li')
    ap.add_argument('--cd31-threshold', type=float, help='fixed threshold on the smoothed channel (16-bit scale)')
    ap.add_argument('--cd31-min-vox', type=int, default=1000)
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()

    rd = a.result_dir
    par = json.loads((rd / 'parameters.json').read_text())
    if not (rd / 'complete.json').exists():
        raise SystemExit('%s has no complete.json: segmentation not finished' % rd)
    if a.marker not in par['marker_channels']:
        raise SystemExit('marker %r not in %s' % (a.marker, list(par['marker_channels'])))
    out_tsv = rd / 'objects_feat.tsv'
    if out_tsv.exists() and not a.force:
        raise SystemExit('%s exists; use --force to overwrite' % out_tsv)

    from nq_segment import read_czi, sha256, pick_channel
    from nq_background import shell_background
    import numpy as np, tifffile
    if sha256(a.czi) != par['input_sha256']:
        raise SystemExit('%s is not the file that produced %s (SHA-256 differs)' % (a.czi, rd))
    native = par['source_dtype'] == 'uint8' and not par['converted_8bit_to_16bit']
    raw, dyes, scale, _ = read_czi(a.czi, native)
    if dyes != par['channels_in_file']:
        raise SystemExit('channels differ from the original run: %s vs %s' % (dyes, par['channels_in_file']))
    z0, y0, x0 = par['crop_origin_zyx']; nz, ny, nx = par['shape_zyx']
    cut = lambda v: v[z0:z0 + nz, y0:y0 + ny, x0:x0 + nx]
    sig = cut(raw[dyes.index(par['marker_channels'][a.marker])])
    cd31 = None
    if a.cd31:
        ci = pick_channel(a.cd31, dyes)
        if dyes[ci] == par['marker_channels'][a.marker] or dyes[ci] == par['nuclear_channel']:
            raise SystemExit('--cd31 %r selects the marker or nuclear channel (%s)' % (a.cd31, dyes[ci]))
        cd31 = cut(raw[ci])
    del raw
    labels = tifffile.imread(rd / 'nuclei_labels.tif')
    if labels.shape != sig.shape:
        raise SystemExit('labels %s vs marker %s: wrong CZI or crop' % (labels.shape, sig.shape))

    t0 = time.time()
    spacing = (scale['Z'], scale['Y'], scale['X'])
    bg = shell_background(labels, sig, spacing, a.shell_um, a.gap_um)
    pct = nucleus_percentile(labels, sig, a.percentile)
    vdist, vinfo = {}, None
    if cd31 is not None:
        vessel, vinfo = segment_vessels(cd31, spacing, a.cd31_smooth_um, a.cd31_method, a.cd31_threshold,
                                        a.cd31_min_vox)
        tifffile.imwrite(rd / 'cd31_mask.tif', vessel.astype(np.uint8), compression='zlib')
        vinfo['channel'] = dyes[ci]
        vdist = vessel_distance(labels, vessel, spacing)
        del vessel

    m, q = a.marker, '%g' % a.percentile
    new = ['%s_shell_median' % m, '%s_shell_n_vox' % m, '%s_net' % m, '%s_ratio' % m,
           '%s_p%s_raw' % (m, q), '%s_p%s_ratio' % (m, q)] + (['CD31_dist_um'] if cd31 is not None else [])
    with (rd / 'objects.tsv').open() as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    n_short = 0
    for r in rows:
        l = int(r['label'])
        med, _, n = bg.get(l, (float('nan'), float('nan'), 0))
        mean = float(r['%s_mean_raw' % m]); p = pct.get(l, float('nan'))
        ok = n >= a.min_shell_vox and med > 0
        n_short += not ok
        r[new[0]] = '%.6g' % med if n else ''
        r[new[1]] = n
        r[new[2]] = '%.6g' % (mean - med) if n >= a.min_shell_vox else ''
        r[new[3]] = '%.6g' % (mean / med) if ok else ''
        r[new[4]] = '%.6g' % p
        r[new[5]] = '%.6g' % (p / med) if ok else ''
        if cd31 is not None:
            r['CD31_dist_um'] = '%.4g' % vdist[l] if l in vdist else ''
    with out_tsv.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter='\t'); w.writeheader(); w.writerows(rows)
    summary = dict(marker=m, percentile=a.percentile, shell_um=a.shell_um, gap_um=a.gap_um, n_objects=len(rows),
                   n_without_valid_shell=n_short, native_depth=native, vessels=vinfo,
                   elapsed_seconds=time.time() - t0, output=str(out_tsv))
    (rd / 'features.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
