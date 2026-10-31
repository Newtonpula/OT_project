"""Build the OT-ready spectral time-series dataset of Ramirez et al. (2024).

    python scripts/build_dataset.py            # all 141 SNe of Table C.1
    python scripts/build_dataset.py SN2011fe   # subset

Outputs (data/processed/):
    sn_spectra_ot.h5       calibrated, gridded spectra + photometry + total-flux curves
                           (spectra: OSC + CSP SN II release of Gutierrez+2017 when present)
    summary_sne.csv        one row per SN
    summary_spectra.csv    one row per OSC spectrum (also rejected ones)
    build_log.txt
"""
import csv
import datetime
import os
import sys
from multiprocessing import Pool

import h5py
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from snprep import pipeline as PL           # noqa: E402
from snprep.photometry import BANDS         # noqa: E402
from snprep.synphot import Filters, FILES, VEGA_MAG   # noqa: E402

RAW = os.path.join(ROOT, 'data', 'raw', 'osc')
CSP = os.path.join(ROOT, 'data', 'raw', 'csp', 'Gutierrez_et_al_2017')   # optional
FILT = os.path.join(ROOT, 'data', 'meta', 'filters')
OUT = os.path.join(ROOT, 'data', 'processed')
STR = h5py.string_dtype()


def _run(args):
    name, typ = args
    try:
        return name, PL.process_sn(name, typ, RAW, FILT, CSP), None
    except Exception as e:                     # keep going; report at the end
        import traceback
        return name, None, traceback.format_exc()


def write_sn(g, row, sn, specs, phot, fits, lc):
    for k in ('type', 'claimedtype', 'host', 't_ref_method', 'osc_maxdate'):
        g.attrs[k] = str(sn[k])
    for k in ('redshift', 'ebv_mw', 'ra', 'dec', 't_ref'):
        g.attrs[k] = float(sn[k])
    g.attrs['window_rest'] = np.array(sn['window'], float)
    g.attrs['n_input_spectra'] = sn['n_raw_spec']     # OSC + CSP, before cleaning/dedup
    g.attrs['n_csp_spectra'] = sn['n_csp_spec']
    g.attrs['paper_n_spec'] = int(row['n_spec_paper'])
    g.attrs['paper_phase_range'] = np.array([float(row['phase_min']), float(row['phase_max'])])
    g.attrs['paper_phot_bands'] = row['phot_bands_paper']
    g.attrs['osc_sources'] = '; '.join(f'{k}={v}' for k, v in sn['sources'].items())
    g.create_dataset('a_lambda_mw', data=sn['a_lambda_mw'].astype('f4'))

    cal = [r for r in specs if r['calibrated']]
    s = g.create_group('spectra')
    s.attrs['n'] = len(cal)
    if cal:
        num = lambda k, dt='f8': np.array([r[k] for r in cal], dt)
        for k in ('mjd', 'phase', 'K_tilde', 'K_mad_rel', 'snr', 'mass_window', 'mass_full', 'mass_sed',
                  'rebin_int_ratio', 'cov_lo', 'cov_hi'):
            s.create_dataset(k, data=num(k))
        for k in ('golden', 'ot_ready'):
            s.create_dataset(k, data=num(k, bool))
        s.create_dataset('n_bands', data=num('n_bands', 'i2'))
        s.create_dataset('osc_index', data=num('osc_index', 'i4'))
        for k in ('instrument', 'filename', 'source', 'u_fluxes', 'origin'):
            s.create_dataset(k, data=np.array([str(r[k]) for r in cal], dtype=object), dtype=STR)
        Kb, Fs, Fo, Fe = (np.full((len(cal), len(BANDS)), np.nan) for _ in range(4))
        for i, r in enumerate(cal):
            for j, b in enumerate(BANDS):
                if b in r['K']:
                    K, Fsyn, Fobs, Fobs_err, _ = r['K'][b]
                    Kb[i, j], Fs[i, j], Fo[i, j], Fe[i, j] = K / r['K_tilde'], Fsyn / r['K_tilde'], Fobs, Fobs_err
        s.create_dataset('K_band_over_K_tilde', data=Kb).attrs['columns'] = ','.join(BANDS)
        s.create_dataset('F_syn_band', data=Fs)       # synthetic flux of the calibrated spectrum (observed frame)
        s.create_dataset('F_obs_band', data=Fo)       # LOESS-interpolated observed flux at the spectrum epoch
        s.create_dataset('F_obs_err_band', data=Fe)
        s.create_dataset('flux', data=np.array([r['flux_grid'] for r in cal]), compression='gzip', shuffle=True)
        s.create_dataset('flux_err', data=np.array([r['err_grid'] for r in cal], 'f4'), compression='gzip', shuffle=True)
        s.create_dataset('flags', data=np.array([r['flag_grid'] for r in cal], 'u1'), compression='gzip')
        nat = s.create_group('native')
        for i, r in enumerate(cal):
            if not r['golden']:
                continue
            d = nat.create_group(str(i))
            d.create_dataset('wave_obs', data=r['w'].astype('f8'), compression='gzip')
            d.create_dataset('flux_obs', data=(r['f'] / r['K_tilde']).astype('f4'), compression='gzip')
            if r['e'] is not None:
                d.create_dataset('err_obs', data=(r['e'] / r['K_tilde']).astype('f4'), compression='gzip')
            d.create_dataset('flags', data=r['flags'], compression='gzip')

    p = g.create_group('photometry')
    used = np.zeros(len(phot), bool)
    for b, f in fits.items():
        idx = np.where(phot['band'] == b)[0]
        used[idx[f['keep']]] = True
    p.create_dataset('mjd', data=phot['mjd'])
    p.create_dataset('mag', data=phot['mag'])
    p.create_dataset('mag_err', data=phot['err'])
    p.create_dataset('err_assumed', data=phot['err_assumed'])
    p.create_dataset('used_in_fit', data=used)
    for k in ('band', 'system', 'source'):
        p.create_dataset(k, data=phot[k].astype(object), dtype=STR)

    l = g.create_group('lightcurve')
    for k in ('mjd', 'phase', 'mag', 'mag_err', 'n_bands', 'mass_template', 'mass_template_err',
              'mass_sed', 'dt_nearest_spec'):
        l.create_dataset(k, data=np.asarray(lc[k]))
    l.attrs['bands'] = ','.join(BANDS)


def main(names=None):
    rows = list(csv.DictReader(open(os.path.join(ROOT, 'data', 'meta', 'paper_table_C1.csv'))))
    if names:
        rows = [r for r in rows if r['name'] in names]
    jobs = [(r['name'], r['type']) for r in rows]
    with Pool(min(8, os.cpu_count() or 1)) as pool:
        results = dict((n, (res, err)) for n, res, err in pool.imap_unordered(_run, jobs))

    os.makedirs(OUT, exist_ok=True)
    h5path = os.path.join(OUT, 'sn_spectra_ot.h5' if not names else 'sn_spectra_ot_subset.h5')
    filters = Filters(FILT)
    log, sne_rows, spec_rows = [], [], []
    with h5py.File(h5path, 'w') as h:
        h.attrs['description'] = ('Flux-calibrated SN Ia/II spectral time series (sample of Ramirez et al. 2024, '
                                  'A&A, arXiv:2409.10701) rebuilt from the Open Supernova Catalog.')
        h.attrs['created'] = datetime.datetime.now().isoformat(timespec='seconds')
        h.attrs['flux_units'] = 'erg s^-1 cm^-2 A^-1 (rest-frame wavelength; f_rest = f_obs*(1+z), so integrals equal observed fluxes)'
        h.attrs['mass_units'] = 'erg s^-1 cm^-2'
        h.attrs['golden_criterion'] = f'MAD(K)/median(K) < {PL.GOLDEN_MAD_REL} with >= {PL.MIN_BANDS} bands'
        h.attrs['flag_bits'] = '1=spike removed, 2=telluric bridged, 4=host line removed'
        h.attrs['bands'] = ','.join(BANDS)       # column order of all per-band arrays
        h.create_dataset('wave_rest', data=PL.GRID)
        h.create_dataset('wave_edges', data=PL.GRID_EDGES)
        fg = h.create_group('filters')
        for b in BANDS:
            w, t = filters.curves[b]
            d = fg.create_dataset(b, data=np.c_[w, t])
            d.attrs.update(name=FILES[b], F0=filters.F0[b], ab_minus_vega=filters.ab_minus_vega[b],
                           wave_eff=filters.wave_eff[b], system='Vega' if b in VEGA_MAG else 'AB')
        sg = h.create_group('sne')
        for r in rows:
            res, err = results[r['name']]
            if res is None:
                log.append(f"FAILED {r['name']}\n{err}")
                continue
            sn, specs, phot, fits, lc = res
            log.extend(sn['log'])
            write_sn(sg.create_group(r['name']), r, sn, specs, phot, fits, lc)
            gold = [x for x in specs if x['golden']]
            otr = [x for x in specs if x.get('ot_ready')]
            ph = [x['phase'] for x in gold if np.isfinite(x['phase'])]
            sne_rows.append(dict(
                name=r['name'], type=r['type'], z=sn['redshift'], ebv_mw=sn['ebv_mw'],
                t_ref=sn['t_ref'], t_ref_method=sn['t_ref_method'], phot_bands=''.join(sorted(fits, key=BANDS.index)),
                n_phot=len(phot), n_input_spec=sn['n_raw_spec'], n_csp_spec=sn['n_csp_spec'], n_calibrated=sum(x['calibrated'] for x in specs),
                n_golden=len(gold), n_ot_ready=len(otr), paper_n_spec=r['n_spec_paper'],
                phase_min=np.nanmin(ph) if ph else np.nan, phase_max=np.nanmax(ph) if ph else np.nan,
                paper_phase_min=r['phase_min'], paper_phase_max=r['phase_max'],
                window_lo=sn['window'][0], window_hi=sn['window'][1]))
            for x in specs:
                spec_rows.append(dict(
                    sn=r['name'], type=r['type'], mjd=x['mjd'], phase=x['phase'], origin=x['origin'], instrument=x['instrument'],
                    filename=x['filename'], u_fluxes=x['u_fluxes'], duplicate=x['duplicate'], n_bands=x['n_bands'],
                    K_tilde=x['K_tilde'], K_mad_rel=x['K_mad_rel'], golden=x['golden'], ot_ready=x.get('ot_ready', False),
                    obs_lo=x['w'][0], obs_hi=x['w'][-1], snr=x['snr'], mass_window=x.get('mass_window', np.nan),
                    mass_sed=x.get('mass_sed', np.nan), n_spike_px=int((x['flags'] & 1 > 0).sum()),
                    n_tell_px=int((x['flags'] & 2 > 0).sum()), n_host_px=int((x['flags'] & 4 > 0).sum()),
                    **{f'K_{b}': (x['K'][b][0] / x['K_tilde'] if b in x['K'] else np.nan) for b in BANDS}))

    suffix = '' if not names else '_subset'
    for fn, rws in ((f'summary_sne{suffix}.csv', sne_rows), (f'summary_spectra{suffix}.csv', spec_rows)):
        if rws:
            with open(os.path.join(OUT, fn), 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(rws[0]))
                w.writeheader()
                for rw in rws:
                    w.writerow({k: (f'{v:.10g}' if isinstance(v, float) else v) for k, v in rw.items()})
    with open(os.path.join(OUT, f'build_log{suffix}.txt'), 'w') as f:
        f.write('\n'.join(log) + '\n')
    n_fail = sum(1 for v in results.values() if v[0] is None)
    print(f'wrote {h5path}: {len(sne_rows)} SNe, {sum(x["golden"] for x in spec_rows)} golden spectra, '
          f'{sum(x["ot_ready"] for x in spec_rows)} OT-ready, {n_fail} failures')


if __name__ == '__main__':
    main(sys.argv[1:] or None)
