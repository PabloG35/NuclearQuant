"""Nuclear-Quant: one summary row per image — how many positives, how strong they are, and per vessel length.

For every finished image (needs objects_feat.tsv and cd31_mask.tif from nq_features.py --cd31) it applies the same
call as nq_classify.py (value > threshold AND gate <= max) and reports:
  n_nuclei, n_pos, frac_pos        counts of nuclei and of positive nuclei
  pos_<col>_median / q25 / q75     the value column among the positives only (a change of LEVEL, not of number)
  bg_median                        median of <marker>_shell_median over all nuclei: the image's own background level
  vessel_len_mm                    vessel length: 3D skeleton of cd31_mask.tif, summed as a minimum spanning tree
                                   over the skeleton voxels with true um distances (cached in vessel_length.json)
  vessel_vol_frac                  fraction of the stack that is vessel mask
  pos_per_mm_vessel                n_pos / vessel_len_mm      (positives per mm of vessel)
  pos_per_mm3                      n_pos / tissue volume (mm^3 of the imaged stack)
The vessel length is only as good as the vessel mask: look at it with nq_call_overlay.py --show-cd31 first.

  python nq_summary.py --marker ERG --column ERG_p75_ratio --threshold 2 --gate-column CD31_dist_um --gate-max 3 \
      --result-dirs results/*/d60 --out summary_ERG
"""
import argparse, csv, json, pathlib


def vessel_length_um(vessel, spacing):
    """Length (um) of the centre line of a boolean ZYX vessel mask. Returns (length_um, n_skeleton_voxels)."""
    import numpy as np
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import minimum_spanning_tree
    from skimage.morphology import skeletonize
    if not vessel.any():
        return 0., 0
    from scipy import ndimage
    zs, ys, xs = ndimage.find_objects(vessel.astype(np.uint8))[0]
    pad = lambda sl, n: slice(max(sl.start - 1, 0), min(sl.stop + 1, n))
    box = (pad(zs, vessel.shape[0]), pad(ys, vessel.shape[1]), pad(xs, vessel.shape[2]))
    sk = skeletonize(vessel[box])
    pts = np.argwhere(sk)
    n = len(pts)
    if n < 2:
        return 0., n
    ids = np.full(sk.shape, -1, np.int64)
    ids[tuple(pts.T)] = np.arange(n)
    half = [(dz, dy, dx) for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
            if (dz, dy, dx) > (0, 0, 0)]                           # 13 of the 26 neighbours: each pair once
    rows, cols, w = [], [], []
    for off in half:
        q = pts + off
        ok = ((q >= 0) & (q < sk.shape)).all(1)
        j = np.full(n, -1, np.int64)
        j[ok] = ids[tuple(q[ok].T)]
        hit = j >= 0
        rows.append(np.flatnonzero(hit)); cols.append(j[hit])
        w.append(np.full(int(hit.sum()), float(np.sqrt(sum((o * s) ** 2 for o, s in zip(off, spacing))))))
    g = csr_matrix((np.concatenate(w), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    return float(minimum_spanning_tree(g).sum()), n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--marker', required=True)
    ap.add_argument('--column', help='value column (default <marker>_p75_ratio)')
    ap.add_argument('--threshold', required=True, type=float)
    ap.add_argument('--gate-column', default='CD31_dist_um')
    ap.add_argument('--gate-max', type=float, default=3.0)
    ap.add_argument('--result-dirs', nargs='+', required=True, type=pathlib.Path)
    ap.add_argument('--recompute-length', action='store_true')
    ap.add_argument('--out', required=True, type=pathlib.Path, help='prefix: writes <out>.tsv')
    a = ap.parse_args()
    import numpy as np, tifffile
    col = a.column or '%s_p75_ratio' % a.marker
    out_rows = []
    for rd in a.result_dirs:
        need = [rd / n for n in ('objects_feat.tsv', 'cd31_mask.tif', 'parameters.json')]
        if not all(p.exists() for p in need):
            print('skip %s: missing %s' % (rd.parent.name, [p.name for p in need if not p.exists()])); continue
        par = json.loads((rd / 'parameters.json').read_text())
        sp = (par['voxel_um']['Z'], par['voxel_um']['Y'], par['voxel_um']['X'])
        with (rd / 'objects_feat.tsv').open() as f:
            rows = list(csv.DictReader(f, delimiter='\t'))
        val = np.array([float(r[col]) if r[col] != '' else np.nan for r in rows])
        gate = np.array([float(r[a.gate_column]) if r[a.gate_column] != '' else np.nan for r in rows])
        bg = np.array([float(r['%s_shell_median' % a.marker]) if r['%s_shell_median' % a.marker] != '' else np.nan
                       for r in rows])
        ok = ~np.isnan(val)
        pos = ok & (val > a.threshold) & (gate <= a.gate_max)
        vessel = tifffile.imread(rd / 'cd31_mask.tif') > 0
        cache = rd / 'vessel_length.json'
        if cache.exists() and not a.recompute_length:
            c = json.loads(cache.read_text()); length_um = c['length_um']
        else:
            length_um, nsk = vessel_length_um(vessel, sp)
            cache.write_text(json.dumps(dict(length_um=length_um, n_skeleton_voxels=nsk, spacing_zyx_um=sp)))
        vol_mm3 = float(np.prod(par['shape_zyx'])) * sp[0] * sp[1] * sp[2] * 1e-9
        length_mm = length_um / 1000.
        q = lambda p: float(np.percentile(val[pos], p)) if pos.any() else float('nan')
        out_rows.append(dict(
            sample=rd.parent.name.replace('S0041_', '').replace('_CD31_488_ERG_647_H33342', '')
                   .replace('_Hippo_Cortex', '').replace('_SSp', '').replace('_Raw', ''),
            n_nuclei=int(ok.sum()), n_pos=int(pos.sum()), frac_pos=round(float(pos.sum() / max(ok.sum(), 1)), 4),
            pos_median=round(q(50), 3), pos_q25=round(q(25), 3), pos_q75=round(q(75), 3),
            bg_median=round(float(np.nanmedian(bg)), 1), vessel_len_mm=round(length_mm, 2),
            vessel_vol_frac=round(float(vessel.mean()), 4),
            pos_per_mm_vessel=round(float(pos.sum() / length_mm), 1) if length_mm > 0 else float('nan'),
            pos_per_mm3=round(float(pos.sum() / vol_mm3)) if vol_mm3 > 0 else float('nan')))
        del vessel
    if not out_rows:
        raise SystemExit('nothing to summarise')
    out = a.out.with_suffix('.tsv')
    with out.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]), delimiter='\t'); w.writeheader(); w.writerows(out_rows)
    keys = list(out_rows[0])
    width = [max(len(k), *(len(str(r[k])) for r in out_rows)) for k in keys]
    print('  '.join(k.ljust(wd) for k, wd in zip(keys, width)))
    for r in out_rows:
        print('  '.join(str(r[k]).ljust(wd) for k, wd in zip(keys, width)))
    print('-> %s   (call: %s > %g, %s <= %g)' % (out, col, a.threshold, a.gate_column, a.gate_max))


if __name__ == '__main__':
    main()
