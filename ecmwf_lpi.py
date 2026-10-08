"""ECMWF LPI ingredients and HRDPS teacher transfer, on native 0.25 degree cells.

The transfer is experimental: agreement with HRDPS is not lightning verification.
Regional ingredients persist separately from the rolling raw GRIB retention.
"""
from __future__ import annotations

import hashlib
import json
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
from eccodes import codes_get, codes_get_array, codes_grib_new_from_file, codes_release
from scipy.ndimage import gaussian_filter1d
from scipy.special import expit, logit

import ecmwf_convective_data as conv
import lpi_model as teacher

VERSION = 'ecmwf_lpi_transfer_v1'
MODEL_PATH = Path(__file__).parent / 'models/lpi/ecmwf_lpi_transfer_v1.json'
LEVELS = (1000, 925, 850, 700, 600, 500, 400, 300, 250)
FINE_LEVELS = (1000, 985, 970, 950, 925, 900, 875, 850, 800, 750, 700, 650, 600, 550, 500, 450, 400, 350, 300, 250)
EXTRA_LEVELS = (1000, 600, 400, 300, 250)


def archive_path(data_root, stamp):
    return Path(data_root) / 'derived/ecmwf_lpi/schema_v1' / stamp / 'ingredients.npz'


def atomic_npz(path, **values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f'.{os.getpid()}.tmp')
    try:
        with tmp.open('wb') as f:
            np.savez_compressed(f, **values)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def read_regional(paths, wanted):
    """Decode each local GRIB once; keys are (shortName, pressure level, step)."""
    result = {}
    lat = lon = slices = None
    for path in paths:
        with open(path, 'rb') as f:
            while (gid := codes_grib_new_from_file(f)) is not None:
                try:
                    name = str(codes_get(gid, 'shortName'))
                    level = int(codes_get(gid, 'level')) if str(codes_get(gid, 'typeOfLevel')) == 'isobaricInhPa' else 0
                    key = (name, level, int(codes_get(gid, 'step')))
                    if key not in wanted:
                        continue
                    ny, nx = int(codes_get(gid, 'Nj')), int(codes_get(gid, 'Ni'))
                    if slices is None:
                        la = codes_get_array(gid, 'latitudes').reshape(ny, nx)
                        lo = codes_get_array(gid, 'longitudes').reshape(ny, nx)
                        lo = np.where(lo > 180, lo - 360, lo)
                        slices = conv._regional_slices(la, lo)
                        lat, lon = la[slices].astype('f4'), lo[slices].astype('f4')
                    values = codes_get_array(gid, 'values').reshape(ny, nx)[slices].astype('f4')
                    if int(codes_get(gid, 'bitmapPresent')):
                        bitmap = codes_get_array(gid, 'bitmap').reshape(ny, nx)[slices]
                        values[bitmap == 0] = np.nan
                    result[key] = values
                finally:
                    codes_release(gid)
    return result, lat, lon


def download_profiles(stamp, step, target):
    """Use the source's JSON inventory and validated single-message byte ranges."""
    target = Path(target)
    if target.exists():
        return target
    date, cycle = stamp[:8], stamp[9:11]
    base = f'https://storage.googleapis.com/ecmwf-open-data/{date}/{cycle}z/ifs/0p25/oper/{date}{cycle}0000-{step}h-oper-fc'
    session = requests.Session()
    response = session.get(base + '.index', timeout=60)
    response.raise_for_status()
    rows = [json.loads(s) for s in response.text.splitlines()]
    wanted = {(name, level) for name in ('t', 'r') for level in EXTRA_LEVELS}
    selected = [r for r in rows if (r.get('param'), int(r.get('levelist', 0))) in wanted and r.get('levtype') == 'pl']
    if len(selected) != len(wanted):
        raise RuntimeError(f'Incomplete source profile inventory for {stamp} F{step:03d}')
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(f'.{os.getpid()}.tmp')
    try:
        with tmp.open('wb') as out:
            for row in selected:
                start, size = int(row['_offset']), int(row['_length'])
                r = session.get(base + '.grib2', headers={'Range': f'bytes={start}-{start + size - 1}'}, timeout=120)
                r.raise_for_status()
                if r.status_code != 206 or len(r.content) != size or not r.content.startswith(b'GRIB'):
                    raise RuntimeError('ECMWF server did not return the requested GRIB byte range')
                out.write(r.content)
        tmp.replace(target)
    finally:
        tmp.unlink(missing_ok=True)
        session.close()
    target.with_suffix('.source.json').write_text(json.dumps({'inventory_url': base + '.index', 'inventory_sha256': hashlib.sha256(response.content).hexdigest(), 'messages': selected}, indent=2))
    return target


def interpolate_profile(values):
    """Log-pressure interpolation, with no extrapolation beyond source endpoints."""
    from scipy.interpolate import interp1d
    return interp1d(np.log(LEVELS[::-1]), values[::-1], axis=0, bounds_error=True)(np.log(FINE_LEVELS)).astype('f4')


def ingredients(fields, cape, ps, step, previous):
    from make_ensemble_control_fourpanel import surface_based_lifted_index_values
    t = interpolate_profile(np.stack([fields['t', p, step] for p in LEVELS]))
    rh = np.clip(interpolate_profile(np.stack([fields['r', p, step] for p in LEVELS])), 0, 100)
    tc = t - 273.15
    edges = np.r_[1007.5, (np.asarray(FINE_LEVELS[:-1]) + np.asarray(FINE_LEVELS[1:])) / 2, 225]
    thickness = np.clip(np.minimum(ps[None], edges[:-1, None, None]) - edges[1:, None, None], 0, np.diff(-edges)[:, None, None])
    valid = np.isfinite(t) & np.isfinite(rh) & (thickness > 0)
    weight = np.where(valid & (tc >= -20) & (tc <= 0), np.exp(-.5 * ((tc + 15) / 5.5)**2) * thickness, 0)
    depth = weight.sum(0)
    charge = np.divide((np.where(valid, rh, 0) * weight).sum(0), depth, out=np.full(ps.shape, np.nan), where=depth > 0)
    mw = np.where(valid & (tc >= -30) & (tc <= -5), thickness, 0)
    mid = np.divide((np.where(valid, rh, 0) * mw).sum(0), mw.sum(0), out=np.full(ps.shape, np.nan), where=mw.sum(0) > 0)
    st, sd = fields['2t', 0, step], fields['2d', 0, step]
    es = lambda c: 6.112 * np.exp(17.67 * c / (c + 243.5))
    srh = np.clip(100 * es(sd - 273.15) / es(st - 273.15), 0, 100)
    parcels = [surface_based_lifted_index_values(st, sd, ps, fields['t', 500, step])]
    for p in (1000, 925, 850, 700):
        pt, pr = fields['t', p, step], np.clip(fields['r', p, step], .01, 100)
        gamma = np.log(pr / 100) + 17.67 * (pt - 273.15) / (pt - 29.65)
        td = 243.5 * gamma / (17.67 - gamma) + 273.15
        li = surface_based_lifted_index_values(pt, td, np.full(ps.shape, p), fields['t', 500, step])
        parcels.append(np.where((ps >= p) & (ps - p <= 300), li, np.nan))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        li = np.nanmin(parcels, axis=0)
        sub = np.nanmean([srh] + [np.where(ps >= p, rh[FINE_LEVELS.index(p)], np.nan) for p in (850, 800, 750, 700)], axis=0)
    ascent = np.maximum.reduce([np.zeros(ps.shape)] + [np.where(ps >= p, -fields['w', p, step] * 287.05 * fields['t', p, step] / (9.80665 * p * 100), 0) for p in (500, 700)])
    interval = step - previous if step else 3
    rain = np.maximum(0, fields['tp', 0, step] - fields['tp', 0, previous]) * 1000 * 3 / interval if step else np.zeros(ps.shape)
    rate = np.maximum(0, fields['tprate', 0, step]) * 3600
    return np.stack([li, cape, charge, depth, mid, ascent, rain, rate, srh, sub]).astype('f4')


def ensure_archive(data_root, stamp, steps):
    steps = tuple(sorted(set(map(int, steps)) | {0}))
    target = archive_path(data_root, stamp)
    cached = {}
    cached_lat = cached_lon = None
    if target.exists():
        with np.load(target) as z:
            if str(z['version']) != VERSION:
                raise ValueError('Incompatible ECMWF ingredient archive')
            if set(steps) <= set(z['steps'].tolist()):
                return target
            cached = {int(s): a for s, a in zip(z['steps'], z['inputs'])}
            cached_lat, cached_lon = z['lat'], z['lon']
    max_step = max(steps)
    if max_step > 360 or any(s < 0 or (s % (3 if s <= 144 else 6)) for s in steps):
        raise ValueError('Unsupported ECMWF forecast step')
    # Retain the complete native sequence. No synthetic hourly snapshots are made.
    steps = tuple(range(0, min(144, max_step) + 1, 3)) + tuple(range(150, max_step + 1, 6))
    fresh = tuple(s for s in steps if s not in cached)
    raw = Path(data_root) / 'raw/ecmwf/realtime' / stamp[:8] / stamp[9:11]
    supplemental = target.parent / 'source_profiles'
    with ThreadPoolExecutor(max_workers=6) as pool:
        extras = list(pool.map(lambda s: download_profiles(stamp, s, supplemental / f'f{s:03d}.grib2'), fresh))
    wanted = {(n, p, s) for s in fresh for n in ('t', 'r') for p in LEVELS}
    wanted |= {('w', p, s) for s in fresh for p in (500, 700)}
    wanted |= {(n, 0, s) for s in fresh for n in ('2t', '2d', 'tp', 'tprate')}
    wanted |= {('tp', 0, max(0, s - (3 if s <= 144 else 6))) for s in fresh}
    fields, lat, lon = read_regional([raw / 'pl_cf.grib2', raw / 'sfc_cf.grib2', *extras], wanted)
    missing = wanted - set(fields)
    if missing:
        raise RuntimeError(f'Missing ECMWF ingredients: {sorted(missing)[:12]}')
    if cached_lat is not None and (not np.array_equal(cached_lat, lat) or not np.array_equal(cached_lon, lon)):
        raise ValueError('ECMWF ingredient grid changed during extension')
    cpath = conv.ensure_archive(Path(data_root), stamp[:8], stamp[9:11], steps)
    ca = conv.load_archive(cpath)
    if not np.array_equal(ca.lat, lat) or not np.array_equal(ca.lon, lon):
        raise ValueError('ECMWF ingredient grids differ')
    arrays = []
    for s in steps:
        if s in cached:
            arrays.append(cached[s])
            continue
        i = int(np.flatnonzero(ca.steps == s)[0])
        prev = max(0, s - (3 if s <= 144 else 6))
        arrays.append(ingredients(fields, ca.mucape[i], ca.surface_pressure_pa[i] / 100, s, prev))
    atomic_npz(target, version=np.asarray(VERSION), steps=np.asarray(steps), lat=lat, lon=lon, inputs=np.stack(arrays), source_levels=np.asarray(LEVELS), input_names=np.asarray(teacher.KEYS), source_cadence_hours=np.where(np.asarray(steps) <= 144, 3, 6))
    # The compact regional archive and source inventories survive raw cleanup.
    for path in extras:
        path.unlink(missing_ok=True)
    return target


def features(a, lat, interval_hours=None):
    """Same twenty feature definitions, native snapshots, physical 20 km smoothing."""
    valid = np.isfinite(a).all(1)
    count = valid.sum(0)
    hours = np.ones(len(a), dtype='f4') if interval_hours is None else np.asarray(interval_hours, dtype='f4')
    if hours.shape != (len(a),) or np.any(hours <= 0):
        raise ValueError('Each native snapshot must have a positive represented interval')
    represented = (valid * hours[:, None, None]).sum(0)
    ramp = teacher.ramp
    z = [ramp(a[:,0],1,-5), ramp(a[:,1],75,800), ramp(a[:,2],45,80), ramp(a[:,3],35,150), ramp(a[:,4],35,75), ramp(a[:,5],.005,.05), np.maximum(ramp(a[:,6],.05,1.5), ramp(a[:,7],.02,.8)), np.clip(a[:,8]/100,0,1), np.clip(a[:,9]/100,0,1)]
    li, cape, rh, dep, mid, u, rain, surface, sub = [np.where(valid, x, np.nan) for x in z]
    charge = np.sqrt(rh * dep)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        peak = lambda x: np.nanmax(x, axis=0)
        mean = lambda x: np.divide(np.nansum(x * hours[:,None,None],axis=0), represented, out=np.full(count.shape,np.nan), where=represented>0)
        fraction = lambda x: np.divide((x * hours[:,None,None]).sum(0), represented, out=np.full(count.shape, np.nan), where=represented > 0)
        f = np.stack([peak(li),peak(cape),peak(rh),peak(dep),peak(mid),peak(u),peak(rain),mean(li),mean(charge),mean(u),fraction(li>.25),fraction(u>.2),peak(li*charge),peak(li*charge*u),peak(li*charge*rain),peak(cape*charge*u),peak(li*charge*u*(1-sub)),peak(surface),peak(sub),peak(li*u)]).astype('f4')
    dy = 111.2 * abs(float(lat[1,0] - lat[0,0]))
    dx = 111.2 * .25 * np.cos(np.deg2rad(lat[:,0]))
    def smooth(x):
        # East-west distance varies substantially over BC; use latitude-specific kernels.
        rows = np.stack([gaussian_filter1d(row, 20 / spacing, mode='constant', cval=0) for row, spacing in zip(x, dx)])
        return gaussian_filter1d(rows, 20 / dy, axis=0, mode='constant', cval=0)
    out = []
    for x in f:
        good = np.isfinite(x)
        weight = smooth(good.astype('f4'))
        num = smooth(np.where(good, x, 0))
        y = np.divide(num, weight, out=np.full_like(x, np.nan), where=weight > 0)
        out.append(np.where(good & (weight >= .95), y, np.nan))
    return np.stack(out), count


def design(f, duration, kind):
    x = np.asarray(f).reshape(20, -1).T
    p = teacher.probability(f, duration).ravel() / 100
    base = logit(np.clip(p, 1e-6, 1 - 1e-6))
    if kind == 'logit':
        return np.c_[np.ones(len(x)), base]
    if kind == 'feature_residual':
        return np.c_[np.ones(len(x)), base, x]
    raise ValueError(kind)


def predict(f, duration, calibrated=False, model=None):
    if not calibrated:
        return teacher.probability(f, duration)
    model = json.loads(MODEL_PATH.read_text()) if model is None else model
    fit = model['models'][duration]
    if model['teacher_sha256'] != hashlib.sha256(teacher.MODEL_PATH.read_bytes()).hexdigest():
        raise ValueError('ECMWF transfer model requires its frozen HRDPS teacher')
    return (100 * expit(design(f, duration, fit['kind']) @ np.asarray(fit['beta']))).reshape(f.shape[1:]).astype('f4')


def forecast(data_root, stamp, fhour, duration='3h', calibrated=False):
    """Native snapshots only; daily products support six-hour source cadence."""
    end = int(fhour)
    if end < 0 or end > 360 or end % (3 if end <= 144 else 6):
        raise ValueError('Forecast end must be a native ECMWF step between F000 and F360')
    if duration == '3h' and end > 144:
        raise ValueError('Three-hour products require the three-hour source cadence, ending at F144')
    cadence = 3 if end <= 144 else 6
    start = end - (24 if duration == '24h' else cadence)
    steps = [0] if end == 0 else [s for s in range(3, min(end,144)+1,3) if s > start] + [s for s in range(150,end+1,6) if s > start]
    if duration not in ('3h', '24h'):
        raise ValueError('Duration must be 3h or 24h')
    if duration == '24h' and start < 0:
        raise ValueError('Daily forecasts require a complete 24-hour window')
    path = ensure_archive(data_root, stamp, steps)
    with np.load(path) as z:
        ids = [int(np.flatnonzero(z['steps'] == s)[0]) for s in steps]
        a, lat, lon = z['inputs'][ids], z['lat'], z['lon']
    f, count = features(a, lat, [3 if s <= 144 else 6 for s in steps])
    p = predict(f, duration, calibrated=calibrated)
    return p, lat, lon, count
