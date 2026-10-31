"""Convenience reader for data/processed/sn_spectra_ot.h5.

    from snprep.load import SNDataset
    ds = SNDataset()                                  # default path
    ds.names(sn_type='Ia', min_ot_ready=3)
    m = ds.ot_measures('SN2011fe')                    # unbalanced-OT input
    m['wave'], m['mass'], m['t'], m['total']          # (L,), (N, L), (N,), (N,)
    lc = ds.total_flux_curve('SN2011fe')              # daily total flux from photometry
"""
import os

import h5py
import numpy as np

DEFAULT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'data', 'processed', 'sn_spectra_ot.h5')


class SNDataset:
    def __init__(self, path=DEFAULT):
        self.path = path
        with h5py.File(path, 'r') as h:
            self.wave = h['wave_rest'][:]
            self.edges = h['wave_edges'][:]
            self.bands = h.attrs['bands'].split(',')     # column order of per-band arrays
            self._info = {n: dict(g.attrs) | dict(n_ot_ready=int(g['spectra/ot_ready'][:].sum()) if g['spectra'].attrs['n'] else 0)
                          for n, g in h['sne'].items()}

    def names(self, sn_type=None, min_ot_ready=0):
        return [n for n, a in self._info.items()
                if (sn_type is None or a['type'] == sn_type) and a['n_ot_ready'] >= min_ot_ready]

    def info(self, name):
        return self._info[name]

    def spectra(self, name, subset='ot_ready'):
        """All per-spectrum arrays of one SN. subset: 'ot_ready' | 'golden' | 'all' (calibrated)."""
        with h5py.File(self.path, 'r') as h:
            s = h['sne'][name]['spectra']
            if not s.attrs['n']:
                return None
            sel = np.ones(s.attrs['n'], bool) if subset == 'all' else s[subset][:]
            out = {k: s[k][:][sel] for k in s if isinstance(s[k], h5py.Dataset)}
            out['a_lambda_mw'] = h['sne'][name]['a_lambda_mw'][:]
        for k in ('instrument', 'filename', 'source', 'u_fluxes', 'origin'):
            out[k] = np.array([x.decode() if isinstance(x, bytes) else x for x in out[k]])
        out['wave'] = self.wave
        return out

    def ot_measures(self, name, window='sn', subset='ot_ready', clip_negative=True, deredden_mw=False):
        """Spectra as non-negative discrete measures on the common rest-frame grid.

        window      'sn' (per-SN common window, all spectra fully defined) or (lo, hi) in A
        returns dict(wave, mass[N, L] in erg/s/cm2 per 5 A bin, t (rest-frame days from
        t_ref, or from the first spectrum if t_ref is unknown), mjd, total = mass.sum(1))
        Spectra not fully covering the window are dropped.
        """
        sp = self.spectra(name, subset)
        a = self._info[name]
        lo, hi = a['window_rest'] if window == 'sn' else window
        cols = (self.edges[:-1] >= lo) & (self.edges[1:] <= hi)
        flux = sp['flux'][:, cols]
        keep = np.all(np.isfinite(flux), axis=1)
        flux = flux[keep]
        if deredden_mw:
            flux = flux * 10 ** (0.4 * sp['a_lambda_mw'][cols])
        if clip_negative:
            flux = np.clip(flux, 0, None)
        mass = flux * np.diff(self.edges)[cols]
        z = a['redshift']
        t0 = a['t_ref'] if np.isfinite(a['t_ref']) else sp['mjd'][keep].min()
        return dict(wave=self.wave[cols], mass=mass, total=mass.sum(1), mjd=sp['mjd'][keep],
                    t=(sp['mjd'][keep] - t0) / (1 + z), window=(lo, hi),
                    calib_rel_err=sp['K_mad_rel'][keep], filename=sp['filename'][keep])

    def total_flux_curve(self, name):
        """Daily total flux in the SN window from photometry (see README)."""
        with h5py.File(self.path, 'r') as h:
            l = h['sne'][name]['lightcurve']
            return {k: l[k][:] for k in l}

    def photometry(self, name):
        with h5py.File(self.path, 'r') as h:
            p = h['sne'][name]['photometry']
            out = {k: p[k][:] for k in p}
        for k in ('band', 'system', 'source'):
            out[k] = np.array([x.decode() if isinstance(x, bytes) else x for x in out[k]])
        return out
