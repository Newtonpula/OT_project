"""Synthetic photometry in Bessell BVRI (Vega) and SDSS gri (AB).

Filter curves are the sncosmo `bessell*` (Bessell 1990), `sdss*` and `uvot::b/v` curves,
stored as photon-counting transmissions in data/meta/filters. Zero points are
computed self-consistently from the CALSPEC Vega spectrum (Vega = 0.03 mag in
BVRI) and from the AB definition (3631 Jy).
"""
import os

import numpy as np

C_AA = 2.99792458e18          # speed of light [A/s]
FILES = {'B': 'bessellb', 'V': 'bessellv', 'R': 'bessellr', 'I': 'besselli',
         'g': 'sdssg', 'r': 'sdssr', 'i': 'sdssi', 'b_uvot': 'uvot_b', 'v_uvot': 'uvot_v'}
# Vega magnitude: 0.03 in Johnson-Cousins; the UVOT system sets Vega = 0 (Poole et al. 2008).
VEGA_MAG = {'B': 0.03, 'V': 0.03, 'R': 0.03, 'I': 0.03, 'b_uvot': 0.0, 'v_uvot': 0.0}


class Filters:
    def __init__(self, filter_dir):
        self.curves = {}
        for b, fn in FILES.items():
            w, t = np.loadtxt(os.path.join(filter_dir, fn + '.dat'), unpack=True)
            t = np.clip(t, 0, None)
            self.curves[b] = (w, t)
        vw, vf = np.loadtxt(os.path.join(filter_dir, 'vega_calspec.dat'), unpack=True)
        self.F0 = {}            # flux density [erg/s/cm2/A] of a 0-mag source (photon-weighted mean)
        self.ab_minus_vega = {}
        for b, (w, t) in self.curves.items():
            f_vega = np.interp(w, vw, vf) * 10 ** (0.4 * VEGA_MAG.get(b, 0.03))
            f_ab = 3631e-23 * C_AA / w ** 2
            F_vega = np.trapezoid(f_vega * t * w, w) / np.trapezoid(t * w, w)
            F_ab = np.trapezoid(f_ab * t * w, w) / np.trapezoid(t * w, w)
            self.F0[b] = F_vega if b in VEGA_MAG else F_ab
            self.ab_minus_vega[b] = 2.5 * np.log10(F_ab / F_vega)
        self.wave_eff = {b: np.trapezoid(t * w * w, w) / np.trapezoid(t * w, w)
                         for b, (w, t) in self.curves.items()}

    def synflux(self, wave, flux, band, min_cover=0.95):
        """Photon-weighted mean f_lambda of a spectrum in `band`.

        Returns (F, cover) where cover is the fraction of the band's photon
        response covered by the spectrum; F is NaN if cover < min_cover.
        """
        w, t = self.curves[band]
        lo, hi = wave[0], wave[-1]
        wt = t * w
        tot = np.trapezoid(wt, w)
        inside = (w >= lo) & (w <= hi)
        # Fine grid inside the overlap to integrate the spectrum properly.
        a, b = max(lo, w[0]), min(hi, w[-1])
        if b <= a:
            return np.nan, 0.0
        grid = np.concatenate([[a], wave[(wave > a) & (wave < b)], w[inside & (w > a) & (w < b)], [b]])
        grid = np.unique(grid)
        tg = np.interp(grid, w, t, left=0, right=0) * grid
        cover = np.trapezoid(tg, grid) / tot
        if cover < min_cover:
            return np.nan, cover
        fg = np.interp(grid, wave, flux)
        return np.trapezoid(fg * tg, grid) / np.trapezoid(tg, grid), cover

    def mag_to_flux(self, band, mag):
        return self.F0[band] * 10 ** (-0.4 * np.asarray(mag))
