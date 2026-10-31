"""Read Open Supernova Catalog (OSC) JSON files into plain Python/numpy structures.

The OSC (Guillochon et al. 2017) is offline, but its data live on in the
astrocatalogs/sne-* GitHub repositories, one JSON file per supernova.
"""
import gzip
import json
import os

import numpy as np


def _first_float(entries, default=np.nan):
    """OSC quantities are lists of {"value": ...}; return the first numeric value."""
    for e in entries or []:
        try:
            return float(e['value'])
        except (KeyError, TypeError, ValueError):
            continue
    return default


def _hms_to_deg(s, hours):
    parts = [float(p) for p in str(s).replace(' ', ':').split(':')]
    sign = -1.0 if str(s).strip().startswith('-') else 1.0
    parts[0] = abs(parts[0])
    val = sum(p / 60.0 ** i for i, p in enumerate(parts))
    return sign * val * (15.0 if hours else 1.0)


def osc_path(raw_dir, name):
    for fn in (name + '.json', name + '.json.gz'):
        p = os.path.join(raw_dir, fn)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f'{name} not found in {raw_dir}')


def load_osc_merged(raw_dir, name):
    """load_osc for <name>.json plus any case-variant duplicates saved as
    <name>__dup*.json (the OSC split e.g. ASASSN-14lp / asassn-14lp); their spectra
    and photometry are appended (exact duplicates are removed later)."""
    meta, spectra, phot = load_osc(osc_path(raw_dir, name))
    for fn in sorted(os.listdir(raw_dir)):
        if fn.startswith(name + '__dup'):
            _, s2, p2 = load_osc(os.path.join(raw_dir, fn))
            spectra += s2
            phot += p2
    return meta, spectra, phot


def load_osc(path):
    """Return (meta, spectra, photometry) for one OSC JSON file.

    meta        dict of scalar metadata
    spectra     list of dicts with numpy arrays 'wave', 'flux', 'err' (err may be None)
    photometry  list of raw OSC photometry dicts (strings, filtered later)
    """
    op = gzip.open if path.endswith('.gz') else open
    with op(path, 'rt') as f:
        d = json.load(f)
    name, s = next(iter(d.items()))

    ra = dec = np.nan
    if s.get('ra'):
        ra = _hms_to_deg(s['ra'][0]['value'], hours=True)
    if s.get('dec'):
        dec = _hms_to_deg(s['dec'][0]['value'], hours=False)
    meta = dict(
        name=name,
        redshift=_first_float(s.get('redshift')),
        ebv_mw=_first_float(s.get('ebv')),
        ra=ra, dec=dec,
        claimedtype=(s.get('claimedtype') or [{}])[0].get('value', ''),
        host=(s.get('host') or [{}])[0].get('value', ''),
        osc_maxdate=(s.get('maxdate') or [{}])[0].get('value', ''),
        osc_maxband=(s.get('maxband') or [{}])[0].get('value', ''),
        sources={src['alias']: src.get('bibcode', src.get('name', '')) for src in s.get('sources', [])},
    )

    spectra = []
    for i, x in enumerate(s.get('spectra', [])):
        if 'time' not in x or not x.get('data'):
            continue
        rows = [r for r in x['data'] if len(r) >= 2]
        try:
            arr = np.array([[float(v) for v in r[:3]] + [np.nan] * (3 - len(r[:3])) for r in rows])
        except ValueError:
            continue
        has_err = arr.shape[1] == 3 and np.isfinite(arr[:, 2]).any()
        spectra.append(dict(
            osc_index=i,
            mjd=float(x['time']),
            wave=arr[:, 0], flux=arr[:, 1], err=arr[:, 2] if has_err else None,
            u_fluxes=x.get('u_fluxes', ''),
            deredshifted=str(x.get('deredshifted', '')).lower() == 'true',
            spec_redshift=_first_float([{'value': x['redshift']}]) if 'redshift' in x else np.nan,
            exclude=x.get('exclude'),
            instrument=x.get('instrument', ''), telescope=x.get('telescope', ''),
            filename=x.get('filename', ''), source=x.get('source', ''),
        ))
    return meta, spectra, s.get('photometry', [])
