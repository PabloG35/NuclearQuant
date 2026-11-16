"""Nuclear-Quant QC: draw the positive/negative call on z-slices, to check the call by eye.

For one finished image, nuclei whose value is ABOVE --threshold are outlined in green, all others in grey. Two PNGs
are written per slice: *_comp.png (marker = red, nuclear = blue) and *_marker.png (marker alone, grey scale), so a
call can be judged without the blue channel in the way. Display brightness is set per image from percentiles of the
marker channel, so compare calls, not brightness, between images.

  python nq_call_overlay.py --czi stack.czi --result-dir results/<sample>/d60 --marker ERG \
      --column ERG_p75_ratio --threshold 2 --gate-column CD31_dist_um --gate-max 3 --show-cd31 488 --z 8 15 23
  Reads objects_feat.tsv (nq_features.py), else objects_bg.tsv, else objects.tsv. With --gate-column/--gate-max the
  call is the same one nq_classify.py makes (value > threshold AND gate <= max). --show-cd31 SPEC adds a third PNG:
  the vessel channel with the vessel mask outline in cyan (needs cd31_mask.tif from nq_features.py).
Run with PYTHONPATH=deps:. so the CZI reader and nq_segment are found. z is 1-based, as in overlay_z*.png.
"""
import argparse, csv, json, math, pathlib


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--czi', required=True, type=pathlib.Path)
    ap.add_argument('--result-dir', required=True, type=pathlib.Path)
    ap.add_argument('--marker', required=True)
    ap.add_argument('--column', help='value column (default <marker>_mean_raw)')
    ap.add_argument('--threshold', required=True, type=float, help='positive when value > threshold')
    ap.add_argument('--gate-column', help='e.g. CD31_dist_um')
    ap.add_argument('--gate-max', type=float)
    ap.add_argument('--show-cd31', metavar='SPEC', help='vessel channel (dye substring or #index): also draw the vessel mask')
    ap.add_argument('--z', type=int, nargs='+', help='1-based z planes (default: the middle one)')
    ap.add_argument('--max-px', type=int, default=1024, help='longest side of the PNG (default 1024)')
    ap.add_argument('--out-dir', type=pathlib.Path, help='default: <result-dir>/calls')
    a = ap.parse_args()

    rd = a.result_dir
    par = json.loads((rd / 'parameters.json').read_text())
    col = a.column or '%s_mean_raw' % a.marker
    if bool(a.gate_column) != (a.gate_max is not None):
        raise SystemExit('--gate-column and --gate-max go together')
    src = next((rd / n for n in ('objects_feat.tsv', 'objects_bg.tsv', 'objects.tsv') if (rd / n).exists()), None)
    if src is None:
        raise SystemExit('no objects table in %s' % rd)
    with src.open() as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    if col not in rows[0]:
        raise SystemExit('column %s not in %s (run nq_features.py first?)' % (col, src))
    if a.gate_column and a.gate_column not in rows[0]:
        raise SystemExit('gate column %s not in %s' % (a.gate_column, src))

    from nq_segment import read_czi, sha256, pick_channel
    import numpy as np, tifffile
    from scipy import ndimage
    from skimage import segmentation
    from PIL import Image
    if sha256(a.czi) != par['input_sha256']:
        raise SystemExit('%s is not the file that produced %s (SHA-256 differs)' % (a.czi, rd))
    native = par['source_dtype'] == 'uint8' and not par['converted_8bit_to_16bit']
    raw, dyes, _, _ = read_czi(a.czi, native)
    if dyes != par['channels_in_file']:
        raise SystemExit('channels differ from the original run')
    z0, y0, x0 = par['crop_origin_zyx']; nz, ny, nx = par['shape_zyx']
    cut = lambda v: v[z0:z0 + nz, y0:y0 + ny, x0:x0 + nx]
    nuc = cut(raw[dyes.index(par['nuclear_channel'])]); sig = cut(raw[dyes.index(par['marker_channels'][a.marker])])
    cd31, vessel = None, None
    if a.show_cd31:
        if not (rd / 'cd31_mask.tif').exists():
            raise SystemExit('%s has no cd31_mask.tif: run nq_features.py with --cd31 first' % rd)
        cd31 = cut(raw[pick_channel(a.show_cd31, dyes)])
        vessel = tifffile.imread(rd / 'cd31_mask.tif') > 0
    del raw
    labels = tifffile.imread(rd / 'nuclei_labels.tif')
    if labels.shape != sig.shape:
        raise SystemExit('labels %s vs marker %s: wrong CZI or crop' % (labels.shape, sig.shape))

    pos = np.zeros(int(labels.max()) + 1, bool)
    for r in rows:
        if r[col] == '' or not float(r[col]) > a.threshold:
            continue
        if a.gate_column and (r[a.gate_column] == '' or float(r[a.gate_column]) > a.gate_max):
            continue
        pos[int(r['label'])] = True
    print('%d of %d nuclei called positive (%s > %g%s)' % (pos.sum(), len(rows), col, a.threshold,
          ', %s <= %g' % (a.gate_column, a.gate_max) if a.gate_column else ''))

    lim = lambda v: np.percentile(v[:, ::4, ::4], [1, 99.7])
    ln, ls = lim(nuc), lim(sig)
    lc = lim(cd31) if cd31 is not None else None
    to8 = lambda v, l: (255 * np.clip((v.astype(np.float32) - l[0]) / max(float(l[1] - l[0]), 1), 0, 1)).astype(np.uint8)
    out = a.out_dir or rd / 'calls'
    out.mkdir(parents=True, exist_ok=True)
    f = max(1, math.ceil(max(sig.shape[1:]) / a.max_px))
    for z in (a.z or [nz // 2 + 1]):
        if not 1 <= z <= nz:
            raise SystemExit('z=%d outside 1..%d' % (z, nz))
        lab = labels[z - 1]; is_pos = pos[lab]
        b_pos = segmentation.find_boundaries(np.where(is_pos, lab, 0), mode='inner')
        b_neg = segmentation.find_boundaries(np.where(~is_pos & (lab > 0), lab, 0), mode='inner')
        grow = lambda b, n: ndimage.binary_dilation(b, iterations=n) if n > 0 else b
        b_neg, b_pos = grow(b_neg, f - 1), grow(b_pos, f)
        tag = 'z%02d' % z
        m8 = to8(sig[z - 1], ls)
        panels = [('comp', np.stack([m8, np.zeros_like(m8), to8(nuc[z - 1], ln)], -1)),
                  ('marker', np.stack([m8] * 3, -1))]
        if cd31 is not None:
            c8 = to8(cd31[z - 1], lc)
            vb = grow(segmentation.find_boundaries(vessel[z - 1], mode='inner'), f - 1)
            cbase = np.stack([c8] * 3, -1); cbase[vb] = [0, 200, 255]
            panels.append(('cd31', cbase))
        for name, base in panels:
            base[b_neg] = [150, 150, 150]
            base[b_pos] = [0, 255, 0]
            im = Image.fromarray(base)
            if f > 1:
                im = im.resize((im.width // f, im.height // f), Image.LANCZOS)
            im.save(out / ('call_%s_%s.png' % (tag, name)))
        n_pos = len(np.unique(lab[is_pos])); n_all = len(np.unique(lab[lab > 0]))
        print('%s: %d of %d nuclei in this plane called positive -> %s' % (tag, n_pos, n_all, out))


if __name__ == '__main__':
    main()
