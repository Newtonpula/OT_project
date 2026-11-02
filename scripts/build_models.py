"""Build the model spectral time series used for the method tests of the paper (Sect. 4, Figs. 3-5).

    python scripts/download_models.py     # first: CMFGEN spectra from Zenodo into data/raw/models/
    python scripts/build_models.py

Outputs (data/processed/):
    model_spectra_ot.h5                 16 SN Ia (Dessart+2014) + 6 SN II (Dessart+2013) models on 5 A bins, 1500-10000 A
    summary_models.csv                  one row per model
    summary_model_spectra.csv           one row per spectrum
    validation/models_overview.png      V light curves with t_ref, and two spectral sequences
"""
import csv
import datetime
import os
import sys

import h5py
import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from snprep import models as MD                     # noqa: E402
from snprep.spectra import rebin_flux_conserving     # noqa: E402
from snprep.synphot import Filters                   # noqa: E402

RAW = os.path.join(ROOT, 'data', 'raw', 'models')
FILT = os.path.join(ROOT, 'data', 'meta', 'filters')
OUT = os.path.join(ROOT, 'data', 'processed')
STR = h5py.string_dtype()
GRID_EDGES = np.arange(1500.0, 10000.0 + 0.1, 5.0)   # edges line up with the SN grid (3000-10000 A)
GRID = 0.5 * (GRID_EDGES[1:] + GRID_EDGES[:-1])
BANDS = ['B', 'V', 'R', 'I', 'g', 'r', 'i']           # paper: Johnson-Cousins and Sloan bands
PC10 = 4 * np.pi * (10 * 3.0856775814913673e18) ** 2  # cm^2: L = PC10 * flux integral at 10 pc
C1, C2, INK, MUTED = '#2a78d6', '#eb6834', '#0b0b0b', '#8a8984'
SOURCES = {
    'Ia': ('Dessart, Blondin, Hillier & Khokhlov 2014, MNRAS 441, 532', '10.5281/zenodo.5524943',
           'dessart14_snia/SNIa_DDC_DDC10M4_PDDEL_D14b'),
    'II': ('Dessart, Hillier, Waldman & Livne 2013, MNRAS 433, 1745', '10.5281/zenodo.5524916', 'dessart13_snii'),
}


def grid_model(m, filters):
    waves = m['wave'] if isinstance(m['wave'], list) else [m['wave']] * len(m['t_exp'])
    flux = np.array([rebin_flux_conserving(w, f, GRID_EDGES) for w, f in zip(waves, m['flux'])])
    F = np.array([[filters.synflux(GRID, f, b)[0] for b in BANDS] for f in flux])
    mag = -2.5 * np.log10(F / np.array([filters.F0[b] for b in BANDS]))
    full = np.array([np.trapezoid(f, w) for w, f in zip(waves, m['flux'])])
    inside = np.array([np.trapezoid(f[(w >= GRID_EDGES[0]) & (w <= GRID_EDGES[-1])],
                                    w[(w >= GRID_EDGES[0]) & (w <= GRID_EDGES[-1])]) for w, f in zip(waves, m['flux'])])
    mass_window = flux @ np.diff(GRID_EDGES)
    return dict(t_exp=m['t_exp'], flux=flux, mass_window=mass_window, mass_full=full, F_band=F, mag=mag,
                filename=m['filename'], rebin_int_ratio=mass_window / inside)


def main():
    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                         'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'grid.color': '#e6e5e0', 'grid.linewidth': 0.6, 'legend.frameon': False})
    filters = Filters(FILT)
    raw = {'Ia': MD.load_dessart14(os.path.join(RAW, 'dessart14_snia')),
           'II': MD.load_dessart13(os.path.join(RAW, 'dessart13_snii'))}
    iV = BANDS.index('V')
    models, log = {}, []
    for typ, ms in raw.items():
        for name, m in ms.items():
            s = grid_model(m, filters)
            if typ == 'Ia':
                t_ref, method = MD.vmax_quadratic(s['t_exp'], s['mag'][:, iV])
            else:
                t_ref, method, p = MD.tpt_olivares(s['t_exp'], s['mag'][:, iV])
                log.append(f'{name}: Olivares fit a0={p[0]:.2f} mag, t_PT={p[1]:.1f} d, w0={p[2]:.1f} d')
            if m.get('skipped'):
                log.append(f'{name}: no time in the _list file, skipped: {", ".join(m["skipped"])}')
            s['phase'] = s['t_exp'] - t_ref
            models[name] = dict(type=typ, t_ref=t_ref, t_ref_method=method, raw_prefix=m['raw_prefix'], s=s)
    check(models, log)

    h5path = os.path.join(OUT, 'model_spectra_ot.h5')
    with h5py.File(h5path, 'w') as h:
        h.attrs['description'] = ('CMFGEN model spectral time series used for the method tests of Ramirez et al. 2024 '
                                  '(A&A, arXiv:2409.10701), Sect. 4: SNe Ia from Dessart et al. 2014, SNe II from '
                                  'Dessart et al. 2013 (Zenodo, CC-BY-4.0).')
        h.attrs['created'] = datetime.datetime.now().isoformat(timespec='seconds')
        h.attrs['flux_units'] = ('erg s^-1 cm^-2 A^-1 at 10 pc, averaged over each 5 A bin (SN II spectra, given at 1 kpc, '
                                 f'are multiplied by {MD.D13_TO_10PC:g})')
        h.attrs['mass_units'] = 'erg s^-1 cm^-2 at 10 pc (x 4 pi (10 pc)^2 = 1.1965e40 cm^2 gives erg/s)'
        h.attrs['bands'] = ','.join(BANDS)     # column order of mag and F_band
        h.attrs['mag_system'] = 'Vega (BVRI, Vega = 0.03 mag) and AB (gri), absolute magnitudes; filters in data/meta/filters'
        h.create_dataset('wave_rest', data=GRID)
        h.create_dataset('wave_edges', data=GRID_EDGES)
        mg = h.create_group('models')
        for name, mm in models.items():
            g = mg.create_group(name)
            ref, doi, raw_dir = SOURCES[mm['type']]
            g.attrs.update(type=mm['type'], reference=ref, doi=doi, raw_dir=f'data/raw/models/{raw_dir}',
                           raw_prefix=mm['raw_prefix'], t_ref=mm['t_ref'], t_ref_method=mm['t_ref_method'],
                           window_rest=np.array([GRID_EDGES[0], GRID_EDGES[-1]]))
            if name in MD.D13_TABLE1:
                g.attrs.update(dict(zip(('Z', 'R_star_Rsun', 'E_kin_1e51erg', 'M_Ni56_Msun'), MD.D13_TABLE1[name])))
            sp = g.create_group('spectra')
            sp.attrs['n'] = len(mm['s']['t_exp'])
            for k, v in mm['s'].items():
                if k == 'filename':
                    sp.create_dataset(k, data=np.array(v, dtype=object), dtype=STR)
                else:
                    sp.create_dataset(k, data=v)
            sp['mag'].attrs['columns'] = sp['F_band'].attrs['columns'] = ','.join(BANDS)

    write_csvs(models)
    fig_overview(models)
    with open(os.path.join(OUT, 'build_models_log.txt'), 'w') as f:
        f.write('\n'.join(log) + '\n')
    n = {t: sum(mm['type'] == t for mm in models.values()) for t in ('Ia', 'II')}
    ns = {t: sum(len(mm['s']['t_exp']) for mm in models.values() if mm['type'] == t) for t in ('Ia', 'II')}
    print(f'wrote {h5path}: {n["Ia"]} SN Ia models ({ns["Ia"]} spectra), {n["II"]} SN II models ({ns["II"]} spectra)')


def check(models, log):
    iV = BANDS.index('V')
    for name, mm in models.items():
        s = mm['s']
        assert np.isfinite(s['flux']).all() and (s['flux'] >= 0).all(), name
        assert np.isfinite(s['mag']).all(), name
        assert np.abs(s['rebin_int_ratio'] - 1).max() < 2e-3, (name, s['rebin_int_ratio'])
        if mm['type'] == 'Ia':
            assert 14 < mm['t_ref'] < 25, (name, mm['t_ref'])
            msg = f'peak M_V {s["mag"][:, iV].min():.2f}, t(V max) {mm["t_ref"]:.2f} d after explosion'
        else:
            assert 90 < mm['t_ref'] < 200, (name, mm['t_ref'])
            # nebular luminosity vs 56Co decay power (full trapping) for the 56Ni mass of Dessart+2013 Table 1
            t = s['t_exp'][-1]
            L = s['mass_full'][-1] * PC10
            L_co = MD.D13_TABLE1[name][3] * 1.45e43 * (np.exp(-t / 111.3) - np.exp(-t / 8.8))
            msg = (f'M_V(50 d) {np.interp(50, s["t_exp"], s["mag"][:, iV]):.2f}, t_PT {mm["t_ref"]:.1f} d, '
                   f'L({t:.0f} d) / 56Co decay power {L / L_co:.2f}')
        line = (f'{name:9s} {mm["type"]:2s} {len(s["t_exp"]):3d} spectra, t_exp {s["t_exp"][0]:6.2f}-{s["t_exp"][-1]:6.1f} d, '
                f'phase {s["phase"][0]:7.1f} to {s["phase"][-1]:6.1f} d | {msg} | rebin integral ratio '
                f'{s["rebin_int_ratio"].min():.5f}-{s["rebin_int_ratio"].max():.5f}')
        print(line)
        log.append(line)


def write_csvs(models):
    fmt = lambda v: f'{float(v):.10g}' if isinstance(v, (float, np.floating)) else v
    iV = BANDS.index('V')
    rows, spec_rows = [], []
    for name, mm in models.items():
        s, t1 = mm['s'], MD.D13_TABLE1.get(name, (np.nan,) * 4)
        rows.append(dict(name=name, type=mm['type'], n_spec=len(s['t_exp']), t_exp_min=s['t_exp'][0],
                         t_exp_max=s['t_exp'][-1], t_ref=mm['t_ref'], t_ref_method=mm['t_ref_method'],
                         phase_min=s['phase'][0], phase_max=s['phase'][-1], M_V_peak=s['mag'][:, iV].min(),
                         M_V_50d=np.interp(50, s['t_exp'], s['mag'][:, iV]), M_Ni56=t1[3], raw_prefix=mm['raw_prefix']))
        for i in range(len(s['t_exp'])):
            spec_rows.append(dict(model=name, type=mm['type'], t_exp=s['t_exp'][i], phase=s['phase'][i],
                                  filename=s['filename'][i], mass_window=s['mass_window'][i], mass_full=s['mass_full'][i],
                                  **{f'mag_{b}': s['mag'][i, j] for j, b in enumerate(BANDS)}))
    for fn, rws in (('summary_models.csv', rows), ('summary_model_spectra.csv', spec_rows)):
        with open(os.path.join(OUT, fn), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rws[0]))
            w.writeheader()
            for rw in rws:
                w.writerow({k: fmt(v) for k, v in rw.items()})


def fig_overview(models):
    iV = BANDS.index('V')
    fig, axs = plt.subplots(2, 2, figsize=(11, 7.5))
    for ax, typ, xl in ((axs[0, 0], 'Ia', 'days from V maximum'), (axs[0, 1], 'II', 'days from t_PT')):
        for name, mm in models.items():
            if mm['type'] == typ:
                s = mm['s']
                ax.plot(s['phase'], s['mag'][:, iV], lw=0.9, marker='.', ms=2.5, label=name)
        ax.axvline(0, color=MUTED, lw=0.8, ls='--')
        ax.invert_yaxis()
        ax.set_xlabel(xl)
        ax.set_ylabel('M$_V$ (synthetic)')
        ax.set_title(f'SN {typ} models: V light curves')
        ax.legend(fontsize=6, ncol=2 if typ == 'Ia' else 1)
    axs[0, 0].set_xlim(-25, 80)
    for ax, name, rng in ((axs[1, 0], 'DDC15', (-10, 40)), (axs[1, 1], 'm15mlt3', (-120, 0))):
        s = models[name]['s']
        sel = np.where((s['phase'] >= rng[0]) & (s['phase'] <= rng[1]))[0]
        cmap = plt.get_cmap('Blues')
        for k, i in enumerate(sel):
            ax.plot(GRID, s['flux'][i], color=cmap(0.3 + 0.7 * k / max(len(sel) - 1, 1)), lw=0.6)
        ax.set_yscale('log')
        ax.set_ylim(s['flux'][sel].max() * 1e-3, s['flux'][sel].max() * 2)
        ax.set_xlabel('rest wavelength [Å]')
        ax.set_ylabel('f$_\\lambda$ at 10 pc [erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]')
        ax.set_title(f'{name}: {len(sel)} spectra, phase {rng[0]} to {rng[1]} d (light to dark)')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'validation', 'models_overview.png'), dpi=110)
    plt.close(fig)


if __name__ == '__main__':
    main()
