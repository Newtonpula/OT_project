"""Read the CSP SN II spectra of Gutierrez et al. (2017, ApJ 850, 89).

Source: https://csp.obs.carnegiescience.edu/data/Gutierrez_et_al_2017.tar.gz
(extracted to data/raw/csp/Gutierrez_et_al_2017/<SNyyX>/*.fits). 1-D FITS with a
linear wavelength solution, flux in erg/s/cm2/A, observed frame. Many headers
carry illegal keywords or non-ASCII bytes, so the needed cards are parsed by hand.
"""
import datetime
import glob
import os
import re
import warnings

import numpy as np
from astropy.io import fits

SOURCE = 'Gutierrez+2017 (CSP), csp.obs.carnegiescience.edu'
MONTHS = {m: i + 1 for i, m in enumerate(['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                                          'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])}


def csp_dirname(name):
    """SN1986L -> SN86L, SN2003bn -> SN03bn."""
    m = re.match(r'SN(\d{4})(\w+)$', name)
    return f'SN{m.group(1)[2:]}{m.group(2)}' if m else None


def _raw_header(path):
    """Primary-header cards as {KEY: raw value string}, robust to broken cards."""
    cards = {}
    with open(path, 'rb') as f:
        while True:
            block = f.read(2880)
            if len(block) < 2880:
                break
            for i in range(0, 2880, 80):
                card = block[i:i + 80].decode('ascii', errors='replace')
                key = card[:8].strip()
                if key == 'END':
                    return cards
                if card[8:10] != '= ' or key in cards:
                    continue
                val = card[10:]
                if val.lstrip().startswith("'"):
                    v = val.lstrip()[1:]
                    val = v[:v.find("'")] if "'" in v else v
                else:
                    val = val.split('/')[0]
                cards[key] = val.strip()
    return cards


def _num(cards, *keys):
    for k in keys:
        try:
            return float(cards[k])
        except (KeyError, ValueError):
            continue
    return np.nan


def _mjd(dt):
    return (dt - datetime.datetime(1858, 11, 17)).total_seconds() / 86400.0


def _parse_date(cards, fname):
    """MJD from MJD-OBS / JD, else DATE-OBS (+UT), else the date in the file name."""
    mjd = _num(cards, 'MJD-OBS', 'MJD')
    if np.isfinite(mjd) and mjd > 40000:
        return mjd, 'MJD-OBS'
    jd = _num(cards, 'JD')
    if np.isfinite(jd) and jd > 2440000:
        return jd - 2400000.5, 'JD'
    year = lambda yy: 1900 + yy if yy > 50 else 2000 + yy
    file_day = None
    m = re.search(r'(\d{2})([a-z]{3})(\d{2})', fname.lower())       # e.g. 05nov86
    if m and m.group(2) in MONTHS:
        file_day = datetime.datetime(year(int(m.group(3))), MONTHS[m.group(2)], int(m.group(1)))
    date = cards.get('DATE-OBS', '').replace(' ', '')
    day = None
    m = re.match(r'(\d{4})-(\d{1,2})-(\d{1,2})(?:T(\d+):(\d+):([\d.]+))?', date)
    if m:
        day = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        if m.group(4):
            day += datetime.timedelta(hours=int(m.group(4)), minutes=int(m.group(5)), seconds=float(m.group(6)))
            return _mjd(day), 'DATE-OBS'
    m = re.match(r'(\d{1,2})/(\d{1,2})/(\d{2})$', date)
    if m:
        # old IRAF headers mix dd/mm/yy and mm/dd/yy: keep the reading closest to the file-name date
        a, b, yy = int(m.group(1)), int(m.group(2)), year(int(m.group(3)))
        cands = [datetime.datetime(yy, mo, dd) for dd, mo in ((a, b), (b, a)) if 1 <= mo <= 12 and 1 <= dd <= 31]
        if cands:
            day = min(cands, key=lambda c: abs((c - file_day).days)) if file_day else cands[0]
    if day is not None and (file_day is None or abs((day - file_day).days) <= 2):
        ut = re.match(r'(\d+):(\d+):([\d.]+)', cards.get('UT', '') or cards.get('UTSTART', '') or cards.get('UT-START', ''))
        if ut:
            return _mjd(day + datetime.timedelta(hours=int(ut.group(1)), minutes=int(ut.group(2)),
                                                 seconds=float(ut.group(3)))), 'DATE-OBS+UT'
        return _mjd(day), 'DATE-OBS (day only)'
    if file_day is not None:
        return _mjd(file_day), 'file name (day only)'
    return np.nan, 'none'


def _read_one(path):
    cards = _raw_header(path)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with fits.open(path, ignore_missing_end=True, output_verify='ignore') as h:
            data = np.asarray(h[0].data, float)
    flux = data.reshape(-1, data.shape[-1])[0] if data.ndim > 1 else data
    crval = _num(cards, 'CRVAL1')
    cdelt = _num(cards, 'CD1_1', 'CDELT1')
    crpix = _num(cards, 'CRPIX1')
    crpix = 1.0 if not np.isfinite(crpix) else crpix
    if not (np.isfinite(crval) and np.isfinite(cdelt)) or cards.get('DC-FLAG', '0') not in ('0', ''):
        return None
    wave = crval + (np.arange(len(flux)) + 1 - crpix) * cdelt
    mjd, how = _parse_date(cards, os.path.basename(path))
    return dict(wave=wave, flux=flux, mjd=mjd, date_from=how,
                telescope=cards.get('TELESCOP', ''), instrument=cards.get('INSTRUME', ''))


def _merge_arms(blue, red, max_gap=20.0):
    """Join separately reduced blue/red arms of one night (average in the overlap)."""
    if red['wave'][0] - blue['wave'][-1] > max_gap:
        return None
    ov_b = blue['wave'] >= red['wave'][0]
    ov_r = red['wave'] <= blue['wave'][-1]
    wave = np.concatenate([blue['wave'][~ov_b], blue['wave'][ov_b], red['wave'][~ov_r]])
    fb = blue['flux'][ov_b]
    fr = np.interp(blue['wave'][ov_b], red['wave'], red['flux'])
    flux = np.concatenate([blue['flux'][~ov_b], 0.5 * (fb + fr), red['flux'][~ov_r]])
    return dict(blue, wave=wave, flux=flux)


def load_csp_spectra(csp_dir, name):
    """Spectra of one SN in the same dict format as snprep.osc.load_osc."""
    sub = csp_dirname(name)
    files = sorted(glob.glob(os.path.join(csp_dir, sub, '*.fits'))) if sub else []
    recs = {}
    for f in files:
        r = _read_one(f)
        if r is None or not np.isfinite(r['mjd']) or len(r['wave']) < 50:
            continue
        r['filename'] = os.path.basename(f)
        # CSP-I names: SN07il_b01_3P6_EF_01dec07 / _r01_ (blue/red grism, same night)
        m = re.match(r'(.+)_([br])(\d\d)_(.+)\.fits$', r['filename'])
        key = (m.group(1), m.group(3), m.group(4)) if m else (r['filename'],)
        recs.setdefault(key, {})['r' if m and m.group(2) == 'r' else 'b'] = r
    out = []
    for arms in recs.values():
        if 'b' in arms and 'r' in arms:
            merged = _merge_arms(arms['b'], arms['r'])
            group = [merged] if merged is not None else [arms['b'], arms['r']]
            if merged is not None:
                merged['filename'] = arms['b']['filename'] + '+' + arms['r']['filename']
        else:
            group = list(arms.values())
        for r in group:
            out.append(dict(osc_index=-1, mjd=r['mjd'], wave=r['wave'], flux=r['flux'], err=None,
                            u_fluxes='erg/s/cm^2/Angstrom', deredshifted=False, spec_redshift=np.nan,
                            exclude=None, instrument=r['instrument'], telescope=r['telescope'],
                            filename=r['filename'], source=SOURCE, date_from=r['date_from']))
    return out
