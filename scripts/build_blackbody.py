"""Build the black-body test series of the paper's Fig. 2 (Ramirez et al. 2024, Sect. 3).

    python scripts/build_blackbody.py

Outputs (data/processed/):
    blackbody_spectra_ot.h5          B_lambda(T), 5000-10000 K, on 5 A bins over 1000-15000 A
    summary_blackbody.csv            one row per spectrum
    validation/blackbody_series.png  the black curves of the paper's Fig. 2
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
from snprep import blackbody as BB          # noqa: E402

OUT = os.path.join(ROOT, 'data', 'processed')
C1, INK, MUTED = '#2a78d6', '#0b0b0b', '#8a8984'
SERIES = {
    'paper': (BB.PAPER_T, 'The 10 black-body curves of the paper Fig. 2: np.linspace(5000, 10000, 10) K.'),
    'dense': (np.arange(BB.T_MIN, BB.T_MAX + 1, 50.0), '5000-10000 K in 50 K steps: ground truth at intermediate alpha.'),
}


def build_series(T):
    flux = BB.planck_bin_mean(BB.GRID_EDGES, T)
    mass_window = flux @ np.diff(BB.GRID_EDGES)
    mass_full = BB.planck_total(T)
    lam_pk = BB.peak_wave(T)
    return dict(temperature=T, alpha=(T - BB.T_MIN) / (BB.T_MAX - BB.T_MIN), flux=flux,
                mass_window=mass_window, mass_full=mass_full, peak_wave=lam_pk,
                peak_flux=BB.planck(lam_pk, T).diagonal())


def check(name, s):
    T, edges = s['temperature'], BB.GRID_EDGES
    band = BB.planck_band(edges[0], edges[-1], T)
    err_total = np.abs(s['mass_window'] / band - 1).max()
    i_pk = s['flux'].argmax(1)
    in_peak_bin = np.all((edges[i_pk] <= s['peak_wave']) & (s['peak_wave'] <= edges[i_pk + 1]))
    msg = (f'{name}: {len(T)} spectra, {T[0]:.1f}-{T[-1]:.1f} K | window integral vs analytic: max rel err {err_total:.1e} | '
           f'argmax bin contains Wien peak: {in_peak_bin} | in-window fraction {(s["mass_window"] / s["mass_full"]).min():.3f}-'
           f'{(s["mass_window"] / s["mass_full"]).max():.3f}')
    assert err_total < 1e-9 and in_peak_bin, msg
    if name == 'paper':                     # per-bin quadrature vs analytic band integrals (memory-heavy: paper series only)
        exact = BB.planck_band(edges[:-1], edges[1:], T[:, None]) / np.diff(edges)
        err_bin = np.abs(s['flux'] / exact - 1).max()
        msg += f' | per-bin vs analytic: max rel err {err_bin:.1e} | B_lambda peak at 10000 K: {s["peak_flux"][-1]:.4g}'
        assert err_bin < 1e-9, msg
    print(msg)


def fig_series(series):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    d, p = series['dense'], series['paper']
    for f in d['flux']:
        ax.plot(BB.GRID, f, color=C1, lw=0.4, alpha=0.45)
    for k, f in enumerate(p['flux']):
        ax.plot(BB.GRID, f, color=INK, lw=2 if k in (0, len(p['flux']) - 1) else 0.9)
    ax.plot([], [], color=INK, lw=1.2, label='paper series (10 curves of Fig. 2)')
    ax.plot([], [], color=C1, lw=1.2, label='dense series (50 K steps)')
    ax.annotate('10000 K', (4200, p['flux'][-1].max() * 0.93), color=INK)
    ax.annotate('5000 K', (6200, -0.35e7), color=INK)
    ax.set_xlim(0, 15000)
    ax.set_ylim(-0.6e7, None)
    ax.set_xlabel('Wavelength [Å]')
    ax.set_ylabel('B$_\\lambda$ [erg / (cm$^2$ s Å sr)]')
    ax.set_title('Black-body series, bin-averaged on the 5 Å grid')
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'validation', 'blackbody_series.png'), dpi=110)
    plt.close(fig)


def main():
    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                         'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'grid.color': '#e6e5e0', 'grid.linewidth': 0.6, 'legend.frameon': False})
    series = {name: build_series(T) for name, (T, _) in SERIES.items()}
    for name, s in series.items():
        check(name, s)

    os.makedirs(os.path.join(OUT, 'validation'), exist_ok=True)
    h5path = os.path.join(OUT, 'blackbody_spectra_ot.h5')
    window = (BB.GRID_EDGES[0], BB.GRID_EDGES[-1])
    with h5py.File(h5path, 'w') as h:
        h.attrs['description'] = ('Planck black-body spectra, 5000-10000 K, for the OT interpolation test of '
                                  'Ramirez et al. 2024 (A&A, arXiv:2409.10701), Sect. 3 and Fig. 2.')
        h.attrs['created'] = datetime.datetime.now().isoformat(timespec='seconds')
        h.attrs['flux_units'] = 'erg s^-1 cm^-2 A^-1 sr^-1 (spectral radiance B_lambda, averaged over each 5 A bin)'
        h.attrs['mass_units'] = 'erg s^-1 cm^-2 sr^-1'
        h.attrs['constants'] = f'h={BB.H} erg s, c={BB.C} cm/s, k={BB.K_B} erg/K (exact SI 2019 values, cgs)'
        h.create_dataset('wave_rest', data=BB.GRID)
        h.create_dataset('wave_edges', data=BB.GRID_EDGES)
        sg = h.create_group('series')
        for name, (_, desc) in SERIES.items():
            g = sg.create_group(name)
            g.attrs.update(description=desc, window_rest=window)
            sp = g.create_group('spectra')
            sp.attrs['n'] = len(series[name]['temperature'])
            for k, v in series[name].items():
                sp.create_dataset(k, data=v)

    rows = [dict(series=name, temperature=s['temperature'][i], alpha=s['alpha'][i], peak_wave=s['peak_wave'][i],
                 peak_flux=s['peak_flux'][i], mass_window=s['mass_window'][i], mass_full=s['mass_full'][i],
                 frac_in_window=s['mass_window'][i] / s['mass_full'][i])
            for name, s in series.items() for i in range(len(s['temperature']))]
    with open(os.path.join(OUT, 'summary_blackbody.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for rw in rows:
            w.writerow({k: (f'{float(v):.10g}' if isinstance(v, (float, np.floating)) else v) for k, v in rw.items()})
    fig_series(series)
    print(f'wrote {h5path}: ' + ', '.join(f'{n} ({len(s["temperature"])} spectra)' for n, s in series.items()))


if __name__ == '__main__':
    main()
