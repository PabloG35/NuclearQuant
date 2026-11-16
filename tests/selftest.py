"""Synthetic checks of the bit-depth fail-safe. Run: python tests/selftest.py"""
import csv, json, pathlib, subprocess, sys, tempfile
import numpy as np
here = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(here))
from nq_segment import harmonize_depth, pick_channel

# 1. 8-bit -> 16-bit exact scaling and stats on ORIGINAL values
a = np.zeros((1, 2, 4, 4), np.uint8); a[0, 0, 0, 0] = 255; a[0, 1, 1, 1] = 100
r, src, st = harmonize_depth(a)
assert r.dtype == np.uint16 and src == 'uint8' and r.max() == 65535 and r[0, 1, 1, 1] == 25700
assert st[0]['distinct_levels'] == 3 and abs(st[0]['fraction_saturated'] - 1 / 32) < 1e-9
assert (np.round(r / 257).astype(np.uint8) == a).all()          # exact round trip
b = np.full((1, 2, 4, 4), 1000, np.uint16)
r2, src2, _ = harmonize_depth(b)
assert src2 == 'uint16' and (r2 == b).all()
try:
    harmonize_depth(np.zeros((1, 1, 2, 2), np.float32)); raise AssertionError('float accepted')
except SystemExit:
    pass

# 1b. channel selection: names, wavelengths and explicit indices
d = ['Alexa Fluor 647', 'Alexa Fluor 488', 'Hoechst 33342']
assert pick_channel('Hoechst', d) == 2 and pick_channel('647', d) == 0 and pick_channel('#1', d) == 1
for bad in ('Fluor', '999', '#5'):
    try:
        pick_channel(bad, d); raise AssertionError(bad)
    except SystemExit:
        pass

# 2. classifier refuses mixed pooling, allows per-image / override
tmp = pathlib.Path(tempfile.mkdtemp()); rng = np.random.default_rng(0)
def make(name, dtype, gain):
    d = tmp / 'results' / name / 'd60'; d.mkdir(parents=True)
    v = np.concatenate([rng.normal(500, 50, 300), rng.normal(5000, 500, 100)])
    with (d / 'objects.tsv').open('w', newline='') as f:
        w = csv.writer(f, delimiter='\t'); w.writerow(['label', 'touches_xy_border', 'touches_z_border', 'm_mean_raw'])
        for i, x in enumerate(v): w.writerow([i + 1, 0, 0, x])
    (d / 'parameters.json').write_text(json.dumps(dict(source_dtype=dtype, marker_channels={'m': 'X'},
                                                       acquisition_by_dye={'X': {'Gain': gain}})))
    return str(d / 'objects.tsv')
f1, f2, f3 = make('a', 'uint16', '600'), make('b', 'uint8', '600'), make('c', 'uint16', '700')
run = lambda *args: subprocess.run([sys.executable, str(here / 'nq_classify.py'), '--marker', 'm', '--out',
                                    str(tmp / 'o'), *args], capture_output=True, text=True)
assert run('--objects', f1, f2).returncode != 0                   # mixed dtype refused
assert run('--objects', f1, f3).returncode != 0                   # different gain refused
assert run('--objects', f1, f2, '--allow-mixed').returncode == 0
assert run('--objects', f1, f2, f3, '--method', 'otsu-per-image').returncode == 0
assert run('--objects', f1, '--method', 'otsu').returncode == 0   # single homogeneous image ok

# 3. volume filter, fold-median (scale-free), sweep, show-volumes, NaN values
def make2(name, dtype, base, n_frag=40):
    d = tmp / 'results2' / name / 'd60'; d.mkdir(parents=True)
    v = np.concatenate([rng.normal(base, base * 0.1, 300), rng.normal(base * 6, base * 0.5, 100),
                        rng.normal(base * 8, base * 0.5, n_frag)])      # fragments look bright but are tiny
    vol = np.concatenate([rng.normal(500, 30, 400), rng.normal(4, 1, n_frag)])
    with (d / 'objects.tsv').open('w', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(['label', 'volume_um3', 'touches_xy_border', 'touches_z_border', 'm_mean_raw', 'm_ratio'])
        for i, (x, vv) in enumerate(zip(v, vol)):
            w.writerow([i + 1, vv, 0, 0, x, '' if i == 0 else x / base])
    (d / 'parameters.json').write_text(json.dumps(dict(source_dtype=dtype, marker_channels={'m': 'X'},
                                                       acquisition_by_dye={'X': {'Gain': '600'}})))
    return str(d / 'objects.tsv')
g1, g2 = make2('lo', 'uint16', 500), make2('hi', 'uint8', 5000)          # background differs 10x, different dtype
out = tmp / 'o2'
run2 = lambda *args: subprocess.run([sys.executable, str(here / 'nq_classify.py'), '--marker', 'm', '--out', str(out),
                                     *args], capture_output=True, text=True)
assert run2('--objects', g1, g2, '--method', 'fixed', '--threshold', '2000').returncode != 0   # pooled still refused
r = run2('--objects', g1, g2, '--method', 'fold-median', '--fold', '3', '--min-volume-um3', '50', '--sweep', '2', '3')
assert r.returncode == 0, r.stderr
rows = list(csv.DictReader((out / 'counts.tsv').open(), delimiter='\t'))
assert [int(x['n_nuclei']) for x in rows] == [400, 400]                      # 40 fragments dropped from each
assert [int(x['n_dropped_small']) for x in rows] == [40, 40]
assert [int(x['n_positive']) for x in rows] == [100, 100]                    # same call at 10x different background
sw = list(csv.DictReader((out / 'sweep.tsv').open(), delimiter='\t'))
assert all(int(x['fold_3']) == 100 for x in sw)
r = run2('--objects', g1, g2, '--method', 'fixed', '--threshold', '3', '--column', 'm_ratio', '--min-volume-um3', '50')
assert r.returncode == 0, r.stderr                                           # *_ratio columns pool across dtypes
rows = list(csv.DictReader((out / 'counts.tsv').open(), delimiter='\t'))
assert [int(x['n_dropped_no_value']) for x in rows] == [1, 1]               # empty ratio -> dropped, not a crash
r = run2('--objects', g1, '--show-volumes')
assert r.returncode == 0 and 'p50' in r.stdout

# 4. local background shell (nq_background.shell_background)
from nq_background import shell_background
lab = np.zeros((8, 40, 40), np.uint32); lab[2:6, 5:15, 5:15] = 1; lab[2:6, 25:35, 25:35] = 2
sig = np.full(lab.shape, 100., np.float32); sig[lab > 0] = 1000.; sig[:, :, 15:20] = 400.   # nucleus 1 sits next to a bright zone
sig[lab == 2] = 1000.
res = shell_background(lab, sig, (1.0, 0.5, 0.5), shell_um=2.0, gap_um=0.5)
assert set(res) == {1, 2} and res[2][0] == 100.0                             # nucleus masks never enter a shell
assert res[1][1] > res[2][1] and res[1][0] == 100.0 and res[1][2] > 0        # bright strip lifts the mean, not the median
assert shell_background(np.zeros_like(lab), sig, (1, 1, 1), 2.0) == {}

# 5. nq_features end to end (monkeypatched CZI reader), classifier gate, overlay with vessels
import tifffile, nq_segment, nq_features, nq_call_overlay
fx = pathlib.Path(tempfile.mkdtemp()); czi = fx / 'fake.czi'; czi.write_bytes(b'x' * 1000)
rd = fx / 'results' / 'S' / 'd60'; rd.mkdir(parents=True)
lab = np.zeros((6, 120, 120), np.uint32); lab[1:5, 56:66, 10:30] = 1; lab[1:5, 100:115, 10:30] = 2; lab[1:5, 56:66, 70:90] = 3
erg = np.full(lab.shape, 50, np.uint16); erg[lab == 1] = 500; erg[lab == 2] = 500; erg[lab == 3] = 60
erg[1:5, 56:66, 10:20] = 200                                      # nucleus 1: half dim, so p75 > mean
vessel_ch = np.full(lab.shape, 30, np.uint16); vessel_ch[:, 48:53, :] = 3000
nuc = np.where(lab > 0, 800, 20).astype(np.uint16)
tifffile.imwrite(rd / 'nuclei_labels.tif', lab)
chans = ['H33342', 'Alexa Fluor 647', 'Alexa Fluor 488']; raw = np.stack([nuc, erg, vessel_ch])
sc = dict(X=.5, Y=.5, Z=1.)
par = dict(input_sha256=nq_segment.sha256(czi), source_dtype='uint16', converted_8bit_to_16bit=False,
           channels_in_file=chans, nuclear_channel='H33342', marker_channels={'ERG': 'Alexa Fluor 647'},
           crop_origin_zyx=[0, 0, 0], shape_zyx=list(lab.shape))
(rd / 'parameters.json').write_text(json.dumps(par)); (rd / 'complete.json').write_text('{}')
means = {1: erg[lab == 1].mean(), 2: 500., 3: 60.}
with (rd / 'objects.tsv').open('w', newline='') as f:
    w = csv.writer(f, delimiter='\t')
    w.writerow(['label', 'volume_um3', 'touches_xy_border', 'touches_z_border', 'ERG_mean_raw'])
    for l in (1, 2, 3): w.writerow([l, 100, 0, 0, means[l]])
nq_segment.read_czi = lambda path, native=False: (raw, chans, sc, {})
sys.argv = ['x', '--czi', str(czi), '--result-dir', str(rd), '--marker', 'ERG', '--cd31', '488', '--shell-um', '3']
nq_features.main()
ft = {int(r['label']): r for r in csv.DictReader((rd / 'objects_feat.tsv').open(), delimiter='\t')}
assert float(ft[1]['CD31_dist_um']) < 3 and float(ft[2]['CD31_dist_um']) > 15      # near vs far from the vessel
assert float(ft[1]['ERG_p75_raw']) == 500.0 and float(ft[1]['ERG_p75_raw']) > float(ft[1]['ERG_mean_raw'])
assert 0 < json.loads((rd / 'features.json').read_text())['vessels']['foreground_fraction'] < 0.2
assert (rd / 'cd31_mask.tif').exists()
try:
    sys.argv = ['x', '--czi', str(czi), '--result-dir', str(rd), '--marker', 'ERG']; nq_features.main()
    raise AssertionError('overwrite accepted')
except SystemExit:
    pass
# classifier: nucleus 1 near vessel and bright, nucleus 2 bright but far, nucleus 3 dim
fo = tmp / 'o3'
rc = lambda *args: subprocess.run([sys.executable, str(here / 'nq_classify.py'), '--marker', 'ERG', '--out', str(fo),
                                   '--objects', str(rd / 'objects_feat.tsv'), *args], capture_output=True, text=True)
r = rc('--column', 'ERG_p75_ratio', '--method', 'fixed', '--threshold', '2'); assert r.returncode == 0, r.stderr
row = list(csv.DictReader((fo / 'counts.tsv').open(), delimiter='\t'))[0]
assert int(row['n_positive']) == 2                                                   # 1 and 2 without the gate
r = rc('--column', 'ERG_p75_ratio', '--method', 'fixed', '--threshold', '2', '--gate-column', 'CD31_dist_um',
       '--gate-max', '5'); assert r.returncode == 0, r.stderr
row = list(csv.DictReader((fo / 'counts.tsv').open(), delimiter='\t'))[0]
assert int(row['n_positive']) == 1 and int(row['n_positive_ungated']) == 2           # the far nucleus is rejected
assert rc('--gate-column', 'CD31_dist_um').returncode != 0                           # gate needs --gate-max
sys.argv = ['x', '--czi', str(czi), '--result-dir', str(rd), '--marker', 'ERG', '--column', 'ERG_p75_ratio',
            '--threshold', '2', '--gate-column', 'CD31_dist_um', '--gate-max', '5', '--show-cd31', '488',
            '--z', '3', '--max-px', '60']
nq_call_overlay.main()
assert sorted(p.name for p in (rd / 'calls').glob('*.png')) == ['call_z03_cd31.png', 'call_z03_comp.png',
                                                                'call_z03_marker.png']

# 6. vessel length (skeleton + spanning tree, um) and the per-image summary
import nq_summary
tube = np.zeros((7, 40, 140), bool); tube[1:6, 17:23, 10:110] = True               # 100 voxels long along x
L, nsk = nq_summary.vessel_length_um(tube, (1.0, 0.5, 0.5))
assert 40 < L < 55, L                                                              # ~ 99 steps x 0.5 um
Lz, _ = nq_summary.vessel_length_um(np.transpose(tube, (2, 1, 0)), (1.0, 0.5, 0.5))  # same tube along z: 1.0 um steps
assert 90 < Lz < 105, Lz
assert nq_summary.vessel_length_um(np.zeros((3, 5, 5), bool), (1, 1, 1))[0] == 0.
par['voxel_um'] = sc; (rd / 'parameters.json').write_text(json.dumps(par))
r = subprocess.run([sys.executable, str(here / 'nq_summary.py'), '--marker', 'ERG', '--threshold', '2', '--gate-max', '5',
                    '--result-dirs', str(rd), '--out', str(fx / 'summ')], capture_output=True, text=True)
assert r.returncode == 0, r.stderr
sm = list(csv.DictReader((fx / 'summ.tsv').open(), delimiter='\t'))[0]
assert sm['n_pos'] == '1' and sm['n_nuclei'] == '3' and float(sm['vessel_len_mm']) > 0 and float(sm['pos_per_mm_vessel']) > 0
assert (rd / 'vessel_length.json').exists()
print('selftest OK')
