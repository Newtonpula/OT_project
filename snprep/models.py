"""CMFGEN model spectra used for the method tests of the paper (Sect. 4, Figs. 3-5).

SN Ia: Dessart, Blondin, Hillier & Khokhlov 2014, MNRAS 441, 532 (DDC and PDDEL delayed detonations),
       f_lambda in erg/s/cm2/A at 10 pc (archive README), on 1500-10000 A every 5 A.
SN II: Dessart, Hillier, Waldman & Livne 2013, MNRAS 433, 1745 (15 Msun RSG explosions), f_lambda in
       erg/s/cm2/A at 1 kpc (CMFGEN convention, no README; the nebular luminosity matches 56Co decay
       for the 56Ni masses of their Table 1), native resolution from 17 A to 76 microns.
Both are returned at 10 pc, so synthetic magnitudes are absolute magnitudes.
"""
import glob
import os
import re

import numpy as np
from scipy.optimize import curve_fit

from .photometry import _olivares

D13_TO_10PC = (1000.0 / 10.0) ** 2
# Dessart et al. 2013, Table 1: Z, R* [Rsun], E_kin [1e51 erg], M(56Ni) [Msun] of the released models
D13_TABLE1 = {
    'm15mlt1': (0.020, 1107, 1.24, 0.121), 'm15mlt3': (0.020, 501, 1.34, 0.086),
    'm15z2m2': (0.020, 768, 1.27, 0.050), 'm15z2m3': (0.002, 524, 1.35, 0.081),
    'm15z4m2': (0.040, 804, 1.24, 0.095), 'm15z8m3': (0.008, 611, 1.27, 0.036),
}


def load_dessart14(raw_dir):
    """{model: dict(t_exp, wave, flux (N, L), filename, raw_prefix)}; DDC*_0p5d -> DDC*, PDDEL*n -> PDDEL*."""
    out = {}
    for f in sorted(glob.glob(os.path.join(raw_dir, '*', '*_spec_1500_10000_interp5_*d.dat'))):
        prefix, t = re.match(r'(.+)_spec_1500_10000_interp5_([\d.]+)d\.dat$', os.path.basename(f)).groups()
        name = re.sub(r'(_0p5d|n)$', '', prefix)
        w, fl = np.loadtxt(f, unpack=True)
        m = out.setdefault(name, dict(t_exp=[], flux=[], filename=[], raw_prefix=prefix, wave=w))
        assert np.array_equal(w, m['wave'])
        m['t_exp'].append(float(t))
        m['flux'].append(fl)
        m['filename'].append(os.path.basename(f))
    return {k: _sorted(m) for k, m in out.items()}


def load_dessart13(raw_dir):
    """{model: dict(t_exp, wave (list per epoch), flux (list), filename, skipped)} at 10 pc.

    Times come from the *_list file (index -> days since explosion); .fl files without a list entry
    have no time and are skipped.
    """
    out = {}
    for d in sorted(glob.glob(os.path.join(raw_dir, '*', ''))):
        name = os.path.basename(os.path.normpath(d))
        lst = glob.glob(os.path.join(d, '*_list'))[0]
        times = dict(line.split() for line in open(lst) if line.strip())
        m = dict(t_exp=[], wave=[], flux=[], filename=[], raw_prefix=os.path.basename(lst)[:-5])
        for key, t in times.items():
            w, fl = np.loadtxt(os.path.join(d, key + '.fl'), unpack=True)
            m['t_exp'].append(float(t))
            m['wave'].append(w)
            m['flux'].append(fl * D13_TO_10PC)
            m['filename'].append(key + '.fl')
        m['skipped'] = sorted(set(os.listdir(d)) - set(m['filename']) - {os.path.basename(lst)})
        out[name] = _sorted(m)
    return out


def _sorted(m):
    i = np.argsort(m['t_exp'])
    for k in ('t_exp', 'flux', 'filename') + (('wave',) if isinstance(m['wave'], list) else ()):
        m[k] = [m[k][j] for j in i] if isinstance(m[k], list) else m[k][i]
    m['t_exp'] = np.array(m['t_exp'])
    return m


def vmax_quadratic(t, V, half=5.0):
    """V-band maximum (paper Sect. 4): second-order polynomial fitted within +-half days of the brightest epoch."""
    tpk = t[np.argmin(V)]
    sel = np.abs(t - tpk) <= half
    c = np.polyfit(t[sel] - tpk, V[sel], 2)
    assert c[0] > 0 and sel.sum() >= 4
    return tpk - c[1] / (2 * c[0]), 'quadratic_Vmax'


def tpt_olivares(t, V):
    """Midpoint of the plateau-to-tail transition t_PT: Olivares et al. (2010) fit to the V light curve,
    started at the steepest fading between consecutive epochs."""
    ok = np.isfinite(V)
    t, V = t[ok], V[ok]
    t0 = t[np.argmax(np.diff(V) / np.diff(t))]
    sel = (t > t0 - 60) & (t < t0 + 100)
    p, _ = curve_fit(_olivares, t[sel], V[sel], p0=[2.0, t0, 3.0, 0.01, np.interp(t0 + 20, t, V)], maxfev=10000,
                     bounds=([0.5, t0 - 25, 0.5, -0.02, -25], [6, t0 + 25, 12, 0.05, 0]))
    return p[1], 'olivares_V', p
