"""Nuclear-Quant step 1: 3D nuclear segmentation of one CZI + per-nucleus marker intensity.

One nuclear counterstain channel (e.g. Hoechst 33342) is segmented in 3D with Cellpose-SAM.
Any number of marker channels (c-Fos, ERG, Lef1, Sox17, NeuN, ...) are only MEASURED inside
each nuclear mask (mean raw intensity). No positive/negative call is made here; see nq_classify.py.

Examples
  python nq_segment.py --czi "S1 cFos 594 H33342.czi" --list-channels
  python nq_segment.py --czi "S1 cFos 594 H33342.czi" --nuclear "Hoechst 33342" \
      --marker "cFos=Alexa Fluor 594" --out-root results
"""
import argparse, csv, hashlib, json, os, pathlib, re, sys, time
import xml.etree.ElementTree as ET

DEFAULT_DIAMETER_PX = 60  # chosen from the NeuN pilot; 45 px gave no notable difference (30 px was cancelled)
MODEL = '/home/users/pablogaxiola/.cellpose/models/cpsam'


ACQ_TAGS = {'Fluor', 'ExcitationWavelength', 'EmissionWavelength', 'Voltage', 'Gain', 'DigitalGain',
            'PinholeSize', 'PixelType', 'BitCountRange', 'Attenuation', 'Intensity'}


def harmonize_depth(raw, native=False):
    """Put every stack on the 16-bit scale and describe what the source really was.

    8-bit data are multiplied by 257 (0..255 -> 0..65535, exact round trip), so marker intensities of 8- and
    16-bit images share one numeric scale. This does NOT restore the 256-level quantisation, so the per-channel
    count of distinct levels and the fraction of saturated voxels are recorded (on the ORIGINAL values).
    """
    import numpy as np
    if raw.dtype not in (np.uint8, np.uint16):
        raise SystemExit('Unsupported pixel type %s (only uint8 and uint16)' % raw.dtype)
    top = int(np.iinfo(raw.dtype).max)
    stats = []
    for ch in raw:
        hist = np.bincount(ch.ravel(), minlength=top + 1)
        stats.append(dict(distinct_levels=int(np.count_nonzero(hist)),
                          fraction_saturated=float(hist[top] / ch.size)))
    source = str(raw.dtype)
    if raw.dtype == np.uint8 and not native:
        raw = raw.astype(np.uint16) * np.uint16(257)
    return raw, source, stats


def read_czi(path, native=False):
    """Return (CZYX uint16 array, dye names, voxel um {'X','Y','Z'}, info).

    info = dict(source_dtype, converted_8bit_to_16bit, channel_stats, acquisition_by_dye).
    """
    import numpy as np, czifile
    with czifile.CziFile(str(path)) as c:
        xml = ET.fromstring(c.metadata())
        chans = xml.findall('.//Information/Image/Dimensions/Channels/Channel')
        dyes = [ch.findtext('Fluor') or ch.get('Name') for ch in chans]
        scale = {d.get('Id'): float(d.findtext('Value')) * 1e6 for d in xml.findall('.//Scaling/Items/Distance')}
        acq = {d: {t.tag: (t.text or '').strip() for t in ch.iter() if t.tag in ACQ_TAGS and (t.text or '').strip()}
               for d, ch in zip(dyes, chans)}
        raw = np.squeeze(c.asarray(max_workers=1))
    if raw.ndim != 4 or raw.shape[0] != len(dyes):
        raise SystemExit('Expected a single-scene, single-timepoint CZYX stack; got shape %s with channels %s'
                         % (raw.shape, dyes))
    if abs(scale['X'] - scale['Y']) > 1e-8:
        raise SystemExit('X and Y calibration differ: %s' % scale)
    raw, source, stats = harmonize_depth(raw, native)
    info = dict(source_dtype=source, converted_8bit_to_16bit=source == 'uint8' and not native, native_depth=native,
                channel_stats=dict(zip(dyes, stats)), acquisition_by_dye=acq)
    return raw, dyes, scale, info


def pick_channel(spec, dyes):
    """spec is a dye name (case-insensitive substring, e.g. 'Fluor 647') or an explicit 0-based index '#2'.

    A bare number such as '647' is a wavelength inside a dye name, never an index.
    """
    if spec.startswith('#'):
        i = int(spec[1:])
        if i >= len(dyes): raise SystemExit('Channel index %s out of range for %s' % (spec, dyes))
        return i
    hits = [i for i, d in enumerate(dyes) if spec.lower() in (d or '').lower()]
    if len(hits) != 1:
        raise SystemExit('Channel %r matches %d channels in %s' % (spec, len(hits), dyes))
    return hits[0]


def safe(label):
    return re.sub(r'[^A-Za-z0-9_-]+', '_', label)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''): h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--czi', required=True, type=pathlib.Path)
    ap.add_argument('--list-channels', action='store_true', help='print channels and exit')
    ap.add_argument('--nuclear', help='nuclear counterstain: dye name (substring) or 0-based index')
    ap.add_argument('--marker', action='append', default=[], metavar='LABEL=CHANNEL',
                    help='marker to measure, e.g. "cFos=Alexa Fluor 594" (repeatable)')
    ap.add_argument('--diameter', type=int, default=DEFAULT_DIAMETER_PX, help='Cellpose diameter in XY px (default 60)')
    ap.add_argument('--sample', help='sample name (default: CZI file name without spaces)')
    ap.add_argument('--out-root', type=pathlib.Path, default=pathlib.Path('results'))
    ap.add_argument('--native-depth', action='store_true',
                    help='keep 8-bit stacks in 8 bits (no x257); marker means then are in 8-bit units. For '
                         'comparison runs only: record them in a separate --out-root')
    ap.add_argument('--only-if-8bit', action='store_true', help='exit quietly unless the stack is 8-bit')
    ap.add_argument('--expect-bytes', type=int, help='abort if the CZI size differs (guards a partial upload)')
    ap.add_argument('--crop-xy', type=int, help='pilot: segment only a centred SIZE x SIZE crop')
    a = ap.parse_args()

    if a.expect_bytes is not None and a.czi.stat().st_size != a.expect_bytes:
        raise SystemExit('%s is %d bytes, expected %d: upload incomplete?' % (a.czi, a.czi.stat().st_size, a.expect_bytes))
    raw, dyes, scale, info = read_czi(a.czi, a.native_depth)
    if a.only_if_8bit and info['source_dtype'] != 'uint8':
        print('%s is %s, not 8-bit: nothing to compare, exiting without output.' % (a.czi.name, info['source_dtype']))
        return
    if a.list_channels:
        print(json.dumps(dict(file=str(a.czi), shape_czyx=list(raw.shape), channels=dyes, voxel_um=scale, **info),
                         indent=2))
        return
    if info['converted_8bit_to_16bit']:
        print('WARNING: %s is 8-bit; scaled x257 to the 16-bit scale (only 256 distinct levels).' % a.czi.name,
              file=sys.stderr)
    for d, s in info['channel_stats'].items():
        if s['fraction_saturated'] > 1e-3:
            print('WARNING: %.2f%% of voxels saturated in channel %s' % (100 * s['fraction_saturated'], d),
                  file=sys.stderr)
    if not a.nuclear or not a.marker:
        raise SystemExit('--nuclear and at least one --marker are required')
    nuc_i = pick_channel(a.nuclear, dyes)
    markers = {}
    for m in a.marker:
        label, _, spec = m.partition('=')
        if not spec: raise SystemExit('--marker must look like LABEL=CHANNEL, got %r' % m)
        i = pick_channel(spec, dyes)
        if i == nuc_i: raise SystemExit('Marker %r is the nuclear channel' % label)
        markers[safe(label)] = i

    sample = safe(a.sample or a.czi.stem.replace(' ', '_'))
    out = a.out_root / sample / ('d%02d' % a.diameter)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'complete.json').exists():
        raise SystemExit('%s already complete; refusing to overwrite' % out)

    import numpy as np, torch, tifffile, importlib.metadata
    from cellpose import models
    from skimage import measure, segmentation
    from PIL import Image
    torch.set_num_threads(int(os.getenv('SLURM_CPUS_PER_TASK', '8')))
    t0 = time.time()

    nuc = raw[nuc_i]
    chans = {k: raw[i] for k, i in markers.items()}
    origin = [0, 0, 0]
    if a.crop_xy:
        s = a.crop_xy; y0 = (nuc.shape[1] - s) // 2; x0 = (nuc.shape[2] - s) // 2
        nuc = nuc[:, y0:y0 + s, x0:x0 + s].copy()
        chans = {k: v[:, y0:y0 + s, x0:x0 + s].copy() for k, v in chans.items()}
        origin = [0, y0, x0]
    del raw
    aniso = scale['Z'] / scale['X']
    params = dict(sample=sample, input=str(a.czi.resolve()), input_sha256=sha256(a.czi), channels_in_file=dyes,
                  nuclear_channel=dyes[nuc_i], marker_channels={k: dyes[i] for k, i in markers.items()},
                  source_dtype=info['source_dtype'], converted_8bit_to_16bit=info['converted_8bit_to_16bit'],
                  channel_stats=info['channel_stats'], acquisition_by_dye=info['acquisition_by_dye'],
                  shape_zyx=list(nuc.shape), crop_origin_zyx=origin, voxel_um=scale, anisotropy=aniso,
                  cellpose_version=importlib.metadata.version('cellpose'), model='cpsam',
                  diameter_px=a.diameter, diameter_um=a.diameter * scale['X'], do_3D=True,
                  cellprob_threshold=0., min_size_voxels=15, flow3D_smooth=0, batch_size=4,
                  niter=int(200 * a.diameter / 30),
                  normalize=dict(normalize=True, norm3D=True, percentile=[1., 99.]),
                  slurm_job_id=os.getenv('SLURM_JOB_ID'))
    (out / 'parameters.json').write_text(json.dumps(params, indent=2))

    model = models.CellposeModel(gpu=False, pretrained_model=MODEL, use_bfloat16=False)
    x = np.zeros((*nuc.shape, 3), np.float32); x[..., 0] = nuc
    masks, _, _ = model.eval(x, channel_axis=-1, z_axis=0, diameter=a.diameter, do_3D=True, anisotropy=aniso,
                             resample=True, normalize=params['normalize'], cellprob_threshold=0., min_size=15,
                             flow3D_smooth=0, batch_size=4, niter=params['niter'])
    del x, model
    masks = masks.astype(np.uint32, copy=False)
    if masks.max() == 0: raise SystemExit('No nuclei detected; inspect the image')
    tifffile.imwrite(out / 'nuclei_labels.tif', masks, compression='zlib', photometric='minisblack',
                     metadata={'axes': 'ZYX', 'unit': 'um', 'spacing': scale['Z']})

    flat = masks.ravel(); counts = np.bincount(flat)
    mean = lambda v: np.bincount(flat, weights=v.ravel(), minlength=len(counts)) / np.maximum(counts, 1)
    nuc_mean = mean(nuc); marker_mean = {k: mean(v) for k, v in chans.items()}
    xy_b = set(np.unique(np.concatenate([masks[:, 0].ravel(), masks[:, -1].ravel(),
                                         masks[:, :, 0].ravel(), masks[:, :, -1].ravel()])).tolist())
    z_b = set(np.unique(np.concatenate([masks[0].ravel(), masks[-1].ravel()])).tolist())
    vox = scale['X'] * scale['Y'] * scale['Z']
    fields = ['label', 'volume_voxels', 'volume_um3', 'equivalent_sphere_diameter_um', 'z_px', 'y_px', 'x_px',
              'touches_xy_border', 'touches_z_border', 'nuclear_mean_raw'] + ['%s_mean_raw' % k for k in markers]
    vols = []
    with (out / 'objects.tsv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter='\t'); w.writeheader()
        for p in measure.regionprops(masks):
            l = p.label; v = float(counts[l] * vox); vols.append(v); z, y, xx = p.centroid
            row = dict(label=l, volume_voxels=int(counts[l]), volume_um3=v,
                       equivalent_sphere_diameter_um=(6 * v / np.pi) ** (1 / 3), z_px=z, y_px=y, x_px=xx,
                       touches_xy_border=int(l in xy_b), touches_z_border=int(l in z_b), nuclear_mean_raw=nuc_mean[l])
            row.update({'%s_mean_raw' % k: marker_mean[k][l] for k in markers})
            w.writerow(row)

    # QC overlays: nuclear = blue, first marker = red, nuclear outlines = yellow.
    to8 = lambda v, lim: (255 * np.clip((v.astype(np.float32) - lim[0]) / max(float(lim[1] - lim[0]), 1), 0, 1)).astype(np.uint8)
    first = next(iter(chans.values()))
    ln, lm = np.percentile(nuc, [1, 99.7]), np.percentile(first, [1, 99.7])
    for z in np.linspace(0, nuc.shape[0] - 1, 5).round().astype(int):
        rgb = np.zeros((*masks.shape[1:], 3), np.uint8)
        rgb[..., 0] = to8(first[z], lm); rgb[..., 2] = to8(nuc[z], ln)
        rgb[segmentation.find_boundaries(masks[z], mode='inner')] = [255, 255, 0]
        im = Image.fromarray(rgb); im.resize((min(1024, im.width), min(1024, im.height))).save(out / ('overlay_z%02d.png' % (z + 1)))

    summary = dict(sample=sample, diameter_px=a.diameter, n_objects=len(vols), median_volume_um3=float(np.median(vols)),
                   n_touching_xy=len(xy_b - {0}), n_touching_z=len(z_b - {0}),
                   elapsed_seconds=time.time() - t0, output=str(out), status='complete')
    (out / 'complete.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
