"""Nuclear-Quant step 2: call nuclei positive/negative for ONE marker and count per sample.

Reads the objects.tsv (+ parameters.json) written by nq_segment.py (no segmentation rerun).
Threshold methods (applied to log1p of the mean nuclear marker intensity):
  otsu            one Otsu cutoff pooled over all given samples (the NeuN pilot method). Exploratory; it assumes
                  a roughly bimodal population and misleads for rare-positive markers such as c-Fos.
  otsu-per-image  one Otsu cutoff per image: robust to gain/depth differences, but forces a split in every image
                  (an image with no positives still gets a cutoff), so do not use it for rare markers.
  control         cutoff = PERCENTILE of the nuclei in --control objects.tsv files (negative-control images).
  fixed           cutoff given with --threshold (raw 16-bit-scale units).
  fold-median     per-image normalisation: value / median of that image's nuclei (most nuclei are negative, so the
                  median tracks the image's own background level); positive when value > --fold x median.
                  Removes the image-to-image shift in background that breaks pooled thresholds. Use --sweep to see
                  how the counts change with the fold before choosing one.

Options that apply to every method: --min-volume-um3 drops small fragments before anything is computed;
--column picks another column (e.g. ERG_net or ERG_ratio from nq_background.py) instead of <marker>_mean_raw.

Fail-safe: pooled methods (otsu, control, fixed) are refused when the images come from mixed source bit depths
(8- vs 16-bit) or when the marker channel's recorded acquisition settings differ (gain, voltage, pinhole, ...),
because the raw numbers are then not comparable. Override with --allow-mixed (the warning is kept in method.json).

  python nq_classify.py --marker cFos --objects results/*/d60/objects.tsv --method otsu --out classify_cFos
"""
import argparse, csv, json, pathlib, os
os.environ.setdefault('MPLBACKEND', 'Agg')


def load(paths, col, marker):
    import numpy as np
    out = {}
    for p in map(pathlib.Path, paths):
        with p.open() as f: rows = list(csv.DictReader(f, delimiter='\t'))
        if col not in rows[0]: raise SystemExit('%s has no column %s' % (p, col))
        par = json.loads((p.parent / 'parameters.json').read_text()) if (p.parent / 'parameters.json').exists() else {}
        dye = par.get('marker_channels', {}).get(marker)
        out[p.parent.parent.name] = dict(
            v=np.array([float(r[col]) if r[col] != '' else np.nan for r in rows]),
            xy=np.array([int(r['touches_xy_border']) for r in rows]),
            z=np.array([int(r['touches_z_border']) for r in rows]),
            vol=np.array([float(r['volume_um3']) if 'volume_um3' in r else np.nan for r in rows]),
            dtype=par.get('source_dtype', 'unknown'),
            acq=par.get('acquisition_by_dye', {}).get(dye) if dye else None)
    return out


def comparability_problems(data):
    """List reasons why raw marker intensities of these images should not be pooled."""
    problems = []
    dtypes = {d['dtype'] for d in data.values()}
    if len(dtypes) > 1:
        problems.append('mixed source bit depths: ' + ', '.join('%s=%s' % (s, d['dtype']) for s, d in data.items()))
    if 'unknown' in dtypes:
        problems.append('some images have no parameters.json, so their bit depth is unknown')
    acqs = {json.dumps(d['acq'], sort_keys=True) for d in data.values() if d['acq'] is not None}
    if len(acqs) > 1:
        problems.append('marker-channel acquisition settings differ between images')
    return problems


def main():
    import numpy as np
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--marker', required=True, help='marker label used in nq_segment.py')
    ap.add_argument('--objects', nargs='+', required=True)
    ap.add_argument('--method', choices=['otsu', 'otsu-per-image', 'control', 'fixed', 'fold-median'], default='otsu')
    ap.add_argument('--fold', type=float, help='fold-median: positive when value > FOLD x the image median')
    ap.add_argument('--sweep', type=float, nargs='*', metavar='FOLD',
                    help='also write sweep.tsv: n_positive per image for these folds (default 1.5 2 2.5 3 4 5)')
    ap.add_argument('--min-volume-um3', type=float, default=0.,
                    help='drop objects smaller than this (um^3) before thresholds and counts (default: keep all)')
    ap.add_argument('--show-volumes', action='store_true',
                    help='print volume_um3 percentiles per image (to choose --min-volume-um3) and exit')
    ap.add_argument('--column', help='value column (default <marker>_mean_raw)')
    ap.add_argument('--control', nargs='*', default=[], help='objects.tsv of negative-control images')
    ap.add_argument('--percentile', type=float, default=99.5)
    ap.add_argument('--threshold', type=float)
    ap.add_argument('--allow-mixed', action='store_true', help='pool despite mixed bit depth / acquisition')
    ap.add_argument('--out', type=pathlib.Path, required=True)
    a = ap.parse_args()
    col = a.column or '%s_mean_raw' % a.marker
    data = load(a.objects, col, a.marker)
    no_value = {}
    for s_, d in data.items():
        ok = ~np.isnan(d['v']); no_value[s_] = int((~ok).sum())
        if not ok.all():
            for k in ('v', 'xy', 'z', 'vol'): d[k] = d[k][ok]
    if a.show_volumes:
        qs = [1, 2, 5, 10, 25, 50, 75, 95]
        print('volume_um3 percentiles  ' + '  '.join('p%d' % q for q in qs) + '   n_objects')
        for s_, d in data.items():
            print('%-45s' % s_[:45] + '  '.join('%7.1f' % x for x in np.nanpercentile(d['vol'], qs)) + '   %d' % len(d['vol']))
        return
    dropped = {}
    if a.min_volume_um3 > 0:
        for s_, d in data.items():
            if np.isnan(d['vol']).any(): raise SystemExit('%s: no volume_um3 column, cannot filter by volume' % s_)
            keep = d['vol'] >= a.min_volume_um3
            dropped[s_] = int((~keep).sum())
            for k in ('v', 'xy', 'z', 'vol'): d[k] = d[k][keep]
            if not len(d['v']): raise SystemExit('%s: every object is below --min-volume-um3' % s_)
    if a.method in ('otsu', 'otsu-per-image', 'control') and any((d['v'] < 0).any() for d in data.values()):
        raise SystemExit('Some values are negative (background-subtracted column?): log-based methods cannot be used; '
                         'use --method fold-median or fixed.')
    problems = comparability_problems(data)
    scale_free = a.method == 'fold-median' or col.endswith('_ratio')   # fold changes do not depend on bit depth or gain
    if a.method != 'otsu-per-image' and not scale_free and problems and not a.allow_mixed:
        raise SystemExit('Refusing a pooled threshold:\n  - ' + '\n  - '.join(problems) +
                         '\nUse --method otsu-per-image, split the images by setting, or pass --allow-mixed.')
    from skimage.filters import threshold_otsu
    otsu = lambda v: float(np.expm1(threshold_otsu(np.log1p(v), nbins=256)))
    pooled = np.concatenate([d['v'] for d in data.values()])
    if a.method == 'otsu':
        thr = {s: otsu(pooled) for s in data}
    elif a.method == 'otsu-per-image':
        thr = {s: otsu(d['v']) for s, d in data.items()}
    elif a.method == 'control':
        if not a.control: raise SystemExit('--method control needs --control')
        cdata = load(a.control, col, a.marker)
        cprob = comparability_problems({**data, **{'ctrl_' + k: v for k, v in cdata.items()}})
        if cprob and not a.allow_mixed:
            raise SystemExit('Control not comparable with the images:\n  - ' + '\n  - '.join(cprob))
        c = float(np.percentile(np.concatenate([d['v'] for d in cdata.values()]), a.percentile))
        thr = {s: c for s in data}
    elif a.method == 'fold-median':
        if a.fold is None: raise SystemExit('--method fold-median needs --fold')
        thr = {s: a.fold * float(np.median(d['v'])) for s, d in data.items()}
    else:
        if a.threshold is None: raise SystemExit('--method fixed needs --threshold')
        thr = {s: a.threshold for s in data}
    a.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for s, d in data.items():
        pos = d['v'] > thr[s]; inner = (d['xy'] == 0) & (d['z'] == 0)
        rows.append(dict(sample=s, marker=a.marker, source_dtype=d['dtype'], threshold=thr[s],
                         image_median=float(np.median(d['v'])), n_dropped_small=dropped.get(s, 0), n_dropped_no_value=no_value.get(s, 0),
                         n_nuclei=len(pos),
                         n_positive=int(pos.sum()), fraction_positive=float(pos.mean()),
                         n_nuclei_interior=int(inner.sum()), n_positive_interior=int((pos & inner).sum())))
    with (a.out / 'counts.tsv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter='\t'); w.writeheader(); w.writerows(rows)
    if a.sweep is not None:
        folds = a.sweep or [1.5, 2, 2.5, 3, 4, 5]
        with (a.out / 'sweep.tsv').open('w', newline='') as f:
            w = csv.writer(f, delimiter='\t'); w.writerow(['sample', 'n_nuclei'] + ['fold_%g' % x for x in folds])
            for s, d in data.items():
                m = float(np.median(d['v']))
                w.writerow([s, len(d['v'])] + [int((d['v'] > x * m).sum()) for x in folds])
    (a.out / 'method.json').write_text(json.dumps(dict(
        marker=a.marker, method=a.method, threshold_by_sample=thr, fold=a.fold, column=col,
        min_volume_um3=a.min_volume_um3, n_dropped_small=dropped,
        percentile=a.percentile if a.method == 'control' else None,
        comparability_warnings=problems, allow_mixed=a.allow_mixed, n_pooled_nuclei=int(len(pooled)),
        note='Counts per image. Border nuclei are kept in n_positive; *_interior columns exclude them. '
             'The cutoff is not a validated specificity; check QC histograms and a negative control. '
             '8-bit sources were scaled x257 to the 16-bit scale (still only 256 distinct levels).'), indent=2))
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    for s, d in data.items():
        h = ax.hist(np.log1p(np.clip(d['v'], 0, None)), bins=100, histtype='step', label='%s (%s)' % (s, d['dtype']))
        ax.axvline(np.log1p(max(thr[s], 0)), color=h[2][0].get_edgecolor(), ls='--', lw=0.8)
    ax.set_xlabel('log1p mean nuclear %s (16-bit scale)' % a.marker)
    ax.set_ylabel('nuclei'); ax.legend(fontsize=6); fig.tight_layout(); fig.savefig(a.out / 'QC_distribution.png', dpi=150)
    for r in rows: print(r)


if __name__ == '__main__':
    main()
