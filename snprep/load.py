"""Convenience reader for data/processed/sn_spectra_ot.h5.

    from snprep.load import SNDataset
    ds = SNDataset()                                  # default path
    ds.names(sn_type='Ia', min_ot_ready=3)
    m = ds.ot_measures('SN2011fe')                    # unbalanced-OT input
    m['wave'], m['mass'], m['t'], m['total']          # (L,), (N, L), (N,), (N,)
    lc = ds.total_flux_curve('SN2011fe')              # daily total flux from photometry

    bb = BBDataset()                                  # black-body test series (paper Fig. 2)
    m = bb.ot_measures('paper')                       # same keys; t = temperature [K]

    md = ModelDataset()                               # CMFGEN model spectra (paper Sect. 4, Figs. 3-5)
    m = md.ot_measures('DDC15', phase_range=(-5, 30)) # same keys; t = days from V max (Ia) or t_PT (II)
"""
import os

import h5py
import numpy as np

PROCESSED = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'processed')
DEFAULT = os.path.join(PROCESSED, 'sn_spectra_ot.h5')
BB_DEFAULT = os.path.join(PROCESSED, 'blackbody_spectra_ot.h5')
MODEL_DEFAULT = os.path.join(PROCESSED, 'model_spectra_ot.h5')


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


class BBDataset:
    """Reader for data/processed/blackbody_spectra_ot.h5 (scripts/build_blackbody.py).

    Series 'paper' (the 10 curves of the paper's Fig. 2) and 'dense' (50 K steps); temperature plays
    the role of time, alpha = (T - 5000) / 5000 is the barycenter weight between the 5000 K and 10000 K ends.
    """

    def __init__(self, path=BB_DEFAULT):
        self.path = path
        with h5py.File(path, 'r') as h:
            self.wave = h['wave_rest'][:]
            self.edges = h['wave_edges'][:]
            self._info = {n: dict(g.attrs) for n, g in h['series'].items()}

    def names(self):
        return list(self._info)

    def info(self, name):
        return self._info[name]

    def spectra(self, name):
        with h5py.File(self.path, 'r') as h:
            s = h['series'][name]['spectra']
            out = {k: s[k][:] for k in s}
        out['wave'] = self.wave
        return out

    def ot_measures(self, name, window=None):
        """Black bodies as discrete measures, same keys as SNDataset.ot_measures.

        window      None (full 1000-15000 A grid) or (lo, hi) in A; (3000, 10000) gives the SN grid
        returns dict(wave, mass[N, L] in erg/s/cm2/sr per 5 A bin, total = mass.sum(1),
        t = T = temperature [K], alpha)
        """
        sp = self.spectra(name)
        lo, hi = self._info[name]['window_rest'] if window is None else window
        cols = (self.edges[:-1] >= lo) & (self.edges[1:] <= hi)
        mass = sp['flux'][:, cols] * np.diff(self.edges)[cols]
        return dict(wave=self.wave[cols], mass=mass, total=mass.sum(1), t=sp['temperature'],
                    T=sp['temperature'], alpha=sp['alpha'], window=(lo, hi))


class ModelDataset:
    """Reader for data/processed/model_spectra_ot.h5 (scripts/build_models.py).

    CMFGEN models of the paper's Sect. 4 tests: SNe Ia (Dessart+2014) and SNe II (Dessart+2013), every spectrum
    on one grid at 10 pc; mag and F_band are the synthetic photometry that the paper rescales against.
    """

    def __init__(self, path=MODEL_DEFAULT):
        self.path = path
        with h5py.File(path, 'r') as h:
            self.wave = h['wave_rest'][:]
            self.edges = h['wave_edges'][:]
            self.bands = h.attrs['bands'].split(',')     # column order of mag and F_band
            self._info = {n: dict(g.attrs) for n, g in h['models'].items()}

    def names(self, sn_type=None):
        return [n for n, a in self._info.items() if sn_type is None or a['type'] == sn_type]

    def info(self, name):
        return self._info[name]

    def spectra(self, name):
        with h5py.File(self.path, 'r') as h:
            s = h['models'][name]['spectra']
            out = {k: s[k][:] for k in s}
        out['filename'] = np.array([x.decode() if isinstance(x, bytes) else x for x in out['filename']])
        out['wave'] = self.wave
        return out

    def ot_measures(self, name, window=None, phase_range=None):
        """Model spectra as discrete measures, same keys as SNDataset.ot_measures.

        window       None (full 1500-10000 A grid) or (lo, hi) in A; (3000, 10000) gives the SN grid
        phase_range  None or (lo, hi) in days from t_ref; the paper's Fig. 3 covers (-5, 30) for SNe Ia
                     and (-120, 0) for SNe II
        returns dict(wave, mass[N, L] in erg/s/cm2 at 10 pc per 5 A bin, total = mass.sum(1),
        t = phase (days from V max for Ia, from t_PT for II), t_exp (days since explosion),
        F_band[N, n_bands], mag[N, n_bands])
        """
        sp = self.spectra(name)
        lo, hi = self._info[name]['window_rest'] if window is None else window
        cols = (self.edges[:-1] >= lo) & (self.edges[1:] <= hi)
        rows = np.ones(len(sp['phase']), bool) if phase_range is None else \
            (sp['phase'] >= phase_range[0]) & (sp['phase'] <= phase_range[1])
        mass = sp['flux'][rows][:, cols] * np.diff(self.edges)[cols]
        return dict(wave=self.wave[cols], mass=mass, total=mass.sum(1), t=sp['phase'][rows], t_exp=sp['t_exp'][rows],
                    F_band=sp['F_band'][rows], mag=sp['mag'][rows], window=(lo, hi))
