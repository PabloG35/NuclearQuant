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
print('selftest OK')
