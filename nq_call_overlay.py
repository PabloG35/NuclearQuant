"""Nuclear-Quant QC: draw the positive/negative call on z-slices, to check the call by eye.

For one finished image, nuclei whose value is ABOVE --threshold are outlined in green, all others in grey. Two PNGs
are written per slice: *_comp.png (marker = red, nuclear = blue) and *_marker.png (marker alone, grey scale), so a
call can be judged without the blue channel in the way. Display brightness is set per image from percentiles of the
marker channel, so compare calls, not brightness, between images.

  python nq_call_overlay.py --czi stack.czi --result-dir results/<sample>/d60 --marker ERG \
      --column ERG_ratio --threshold 2 --z 8 15 23
  (needs objects_bg.tsv for ERG_ratio / ERG_net; falls back to objects.tsv for <marker>_mean_raw)
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
    ap.add_argument('--z', type=int, nargs='+', help='1-based z planes (default: the middle one)')
    ap.add_argument('--max-px', type=int, default=1024, help='longest side of the PNG (default 1024)')
    ap.add_argument('--out-dir', type=pathlib.Path, help='default: <result-dir>/calls')
    a = ap.parse_args()

    rd = a.result_dir
    par = json.loads((rd / 'parameters.json').read_text())
    col = a.column or '%s_mean_raw' % a.marker
    src = rd / 'objects_bg.tsv' if (rd / 'objects_bg.tsv').exists() else rd / 'objects.tsv'
    with src.open() as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    if col not in rows[0]:
        raise SystemExit('column %s not in %s (run nq_background.py first?)' % (col, src))

    from nq_segment import read_czi, sha256
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
    del raw
    labels = tifffile.imread(rd / 'nuclei_labels.tif')
    if labels.shape != sig.shape:
        raise SystemExit('labels %s vs marker %s: wrong CZI or crop' % (labels.shape, sig.shape))

    pos = np.zeros(int(labels.max()) + 1, bool)
    for r in rows:
        if r[col] != '' and float(r[col]) > a.threshold:
            pos[int(r['label'])] = True
    print('%d of %d nuclei above %g (%s)' % (pos.sum(), len(rows), a.threshold, col))

    lim = lambda v: np.percentile(v[:, ::4, ::4], [1, 99.7])
    ln, ls = lim(nuc), lim(sig)
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
        for name, base in (('comp', np.stack([m8, np.zeros_like(m8), to8(nuc[z - 1], ln)], -1)),
                           ('marker', np.stack([m8] * 3, -1))):
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
