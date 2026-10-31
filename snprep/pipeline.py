"""Per-supernova processing: clean -> calibrate against photometry -> grid -> masses."""
import os

import numpy as np
import extinction

from . import photometry as P
from . import spectra as S
from .csp import load_csp_spectra
from .osc import load_osc_merged
from .synphot import Filters

# Common rest-frame grid (A). 5 A bins cover 3000-10000 A in 1400 bins.
GRID_EDGES = np.arange(3000.0, 10000.0 + 0.1, 5.0)
GRID = 0.5 * (GRID_EDGES[1:] + GRID_EDGES[:-1])
GOLDEN_MAD_REL = 0.10       # paper: MAD(K)/K~ < 10 %
MIN_BANDS = 2               # need >= 2 bands for a MAD
MIN_WINDOW = 2000.0         # A, minimum width of the per-SN common window


def _lc_fits(phot):
    fits = {}
    for b in P.BANDS:
        sel = phot['band'] == b
        if sel.sum() >= 4:
            fits[b] = P.loess_fit(phot['mjd'][sel], phot['mag'][sel], phot['err'][sel])
    return {b: f for b, f in fits.items() if f is not None}


def _phot_at(fits, filters, t0, max_side_gap):
    """Observed flux density (and error) per band from the LOESS fits at MJD t0."""
    out = {}
    for b, fit in fits.items():
        if not P.coverage_ok(fit['t'], t0, max_side_gap=max_side_gap):
            continue
        m, s = fit['predict'](t0)
        F = filters.mag_to_flux(b, m[0])
        out[b] = (F, F * 0.4 * np.log(10) * s[0], m[0], s[0])
    return out


def _dedupe(specs):
    """Flag OSC duplicates: the same spectrum imported from several sources,
    often with the epoch rounded to the day. Criterion: |dt| < 0.75 d and
    near-identical flux shape (corr > 0.9995 on the overlap).
    Keep the copy with the more precise epoch (fractional MJD), then wider coverage."""
    def rank(r):
        return (r['mjd'] % 1 != 0, r['w'][-1] - r['w'][0])
    dup = np.zeros(len(specs), bool)
    for i in range(len(specs)):
        for j in range(i):
            if dup[i] or dup[j] or abs(specs[i]['mjd'] - specs[j]['mjd']) >= 0.75:
                continue
            wi, fi = specs[i]['w'], specs[i]['f']
            wj, fj = specs[j]['w'], specs[j]['f']
            lo, hi = max(wi[0], wj[0]), min(wi[-1], wj[-1])
            if hi - lo < 1000:
                continue
            g = np.linspace(lo, hi, 800)
            if np.corrcoef(np.interp(g, wi, fi), np.interp(g, wj, fj))[0, 1] < 0.9995:
                continue
            dup[j if rank(specs[i]) >= rank(specs[j]) else i] = True
    return dup


def _choose_window(cov_lo, cov_hi):
    """Rest-frame window [lo, hi] maximising (#spectra fully covering it) x width."""
    best = (0, np.nan, np.nan)
    for lo in np.unique(cov_lo):
        for hi in np.unique(cov_hi):
            if hi - lo < MIN_WINDOW:
                continue
            n = np.sum((cov_lo <= lo) & (cov_hi >= hi))
            if n * (hi - lo) > best[0]:
                best = (n * (hi - lo), lo, hi)
    return best[1], best[2]


def _sed_mass(phot_t, filters, lo_obs, hi_obs):
    """Photometry-only pseudo-bolometric flux: trapezoid through band fluxes
    at their effective wavelengths, flat extrapolation to the window edges."""
    if len(phot_t) < 2:
        return np.nan
    lam = np.array([filters.wave_eff[b] for b in phot_t])
    F = np.array([phot_t[b][0] for b in phot_t])
    o = np.argsort(lam)
    lam, F = lam[o], F[o]
    x = np.concatenate([[lo_obs], lam[(lam > lo_obs) & (lam < hi_obs)], [hi_obs]])
    y = np.interp(x, lam, F)
    return np.trapezoid(y, x)


def process_sn(name, sn_type, raw_dir, filter_dir, csp_dir=None):
    filters = Filters(filter_dir)
    meta, raw_specs, raw_phot = load_osc_merged(raw_dir, name)
    for sp in raw_specs:
        sp['origin'] = 'OSC'
    z = meta['redshift']
    log = []
    if csp_dir and os.path.isdir(csp_dir):
        # Gutierrez+2017 CSP SN II spectra; copies already in the OSC are removed by _dedupe
        csp = load_csp_spectra(csp_dir, name)
        for sp in csp:
            sp['origin'] = 'CSP'
        raw_specs = raw_specs + csp
        if csp:
            log.append(f'{name}: +{len(csp)} CSP spectra (Gutierrez+2017)')

    phot = P.select_photometry(raw_phot, filters.ab_minus_vega)
    phot, rejected = P.reject_discrepant_sources(phot)
    for b, src, off, n in rejected:
        log.append(f'{name}: dropped {n} {b} points from source {src} '
                   f'({"; ".join(meta["sources"].get(a, a) for a in src.split(","))}), offset {off:+.2f} mag')
    fits = _lc_fits(phot)
    max_gap = 10.0 if sn_type == 'Ia' else 20.0   # SNe II evolve slowly on the plateau

    if sn_type == 'Ia':
        t_ref, ref_method = P.vmax_quadratic(phot, fits)
    else:
        t_ref, ref_method, _ = P.tpt_olivares(fits)
    log.append(f'{name}: z={z:.5f}, {len(phot)} phot pts, bands={sorted(fits)}, t_ref={t_ref:.2f} ({ref_method})')

    # ---- clean + calibrate every spectrum
    specs = []
    for sp in raw_specs:
        w, f, e, flags = S.clean_spectrum(sp, z)
        if len(w) < 50 or np.nanmedian(f) <= 0:
            continue
        rec = dict(sp, w=w, f=f, e=e, flags=flags)
        rec['K'] = {}
        phot_t = _phot_at(fits, filters, sp['mjd'], max_gap)
        for b in P.BANDS:
            Fsyn, cover = filters.synflux(w, f, b)
            if np.isfinite(Fsyn) and b in phot_t and Fsyn > 0:
                rec['K'][b] = (Fsyn / phot_t[b][0], Fsyn, phot_t[b][0], phot_t[b][1], cover)
        Ks = np.array([v[0] for v in rec['K'].values()])
        rec['n_bands'] = len(Ks)
        if len(Ks):
            rec['K_tilde'] = np.median(Ks)
            rec['K_mad_rel'] = np.median(np.abs(Ks - rec['K_tilde'])) / rec['K_tilde']
        else:
            rec['K_tilde'] = rec['K_mad_rel'] = np.nan
        rec['snr'] = S.snr_estimate(f)
        specs.append(rec)
    specs.sort(key=lambda r: r['mjd'])
    dup = _dedupe(specs)
    for r, d in zip(specs, dup):
        r['duplicate'] = bool(d)
        r['golden'] = (not d) and r['n_bands'] >= MIN_BANDS and r['K_mad_rel'] < GOLDEN_MAD_REL
        r['calibrated'] = (not d) and r['n_bands'] >= MIN_BANDS

    cal = [r for r in specs if r['calibrated']]
    # ---- calibrated, rest-frame, gridded spectra
    for r in cal:
        fcal = r['f'] / r['K_tilde']
        wr = r['w'] / (1 + z)
        r['flux_grid'] = S.rebin_flux_conserving(wr, fcal * (1 + z), GRID_EDGES)
        r['err_grid'] = (S.rebin_error(wr, r['e'] / r['K_tilde'] * (1 + z), GRID_EDGES)
                         if r['e'] is not None else np.full(len(GRID), np.nan))
        r['flag_grid'] = S.rebin_flags(wr, r['flags'], GRID_EDGES)
        ok = np.isfinite(r['flux_grid'])
        r['cov_lo'] = GRID_EDGES[:-1][ok].min() if ok.any() else np.nan
        r['cov_hi'] = GRID_EDGES[1:][ok].max() if ok.any() else np.nan
        # integral check: native vs gridded (over the gridded range)
        inside = (wr >= r['cov_lo']) & (wr <= r['cov_hi'])
        native_int = np.trapezoid(fcal[inside] * (1 + z), wr[inside])
        grid_int = np.nansum(r['flux_grid'] * np.diff(GRID_EDGES))
        r['rebin_int_ratio'] = grid_int / native_int if native_int else np.nan

    gold = [r for r in cal if r['golden']]
    if gold:
        lo, hi = _choose_window(np.array([r['cov_lo'] for r in gold]), np.array([r['cov_hi'] for r in gold]))
    else:
        lo = hi = np.nan
    in_win = (GRID_EDGES[:-1] >= lo) & (GRID_EDGES[1:] <= hi)
    for r in cal:
        r['ot_ready'] = bool(r['golden'] and r['cov_lo'] <= lo and r['cov_hi'] >= hi)
        seg = r['flux_grid'][in_win]
        r['mass_window'] = np.sum(seg * 5.0) if np.all(np.isfinite(seg)) and in_win.any() else np.nan
        r['mass_full'] = np.nansum(r['flux_grid'] * 5.0)
        # SED-only photometric mass at the spectrum epoch, same window
        phot_t = _phot_at(fits, filters, r['mjd'], max_gap)
        r['mass_sed'] = _sed_mass(phot_t, filters, lo * (1 + z), hi * (1 + z)) if np.isfinite(lo) else np.nan

    # ---- daily photometric light curve and total-flux curves
    lc = _daily_lightcurve(fits, filters, cal, z, lo, hi, max_gap)

    ebv = meta['ebv_mw'] if np.isfinite(meta['ebv_mw']) else 0.0
    a_mw = extinction.fitzpatrick99(GRID * (1 + z), 3.1 * ebv, 3.1)

    phase = lambda t: (t - t_ref) / (1 + z)
    for r in specs:
        r['phase'] = phase(r['mjd'])
        for k in ('wave', 'flux', 'err'):       # raw OSC arrays no longer needed
            r.pop(k, None)
    lc['phase'] = phase(lc['mjd'])

    sn = dict(meta, type=sn_type, t_ref=t_ref, t_ref_method=ref_method, window=(lo, hi),
              a_lambda_mw=a_mw, n_raw_spec=len(raw_specs),
              n_csp_spec=sum(sp['origin'] == 'CSP' for sp in raw_specs), log=log)
    fits = {b: {k: v for k, v in f.items() if k != 'predict'} for b, f in fits.items()}   # picklable
    return sn, specs, phot, fits, lc


def _daily_lightcurve(fits, filters, cal, z, lo, hi, max_gap):
    """Daily LOESS magnitudes and two total-flux ('mass') estimates in the SN window.

    mass_template: nearest golden (OT-ready) spectra on each side, rescaled to the
                   photometry at t by the median band ratio (the paper's K~ logic),
                   inverse-distance averaged.
    mass_sed:      photometry-only trapezoid SED integral.
    """
    if not fits:
        return dict(mjd=np.array([]), mag=np.zeros((0, len(P.BANDS))), mag_err=np.zeros((0, len(P.BANDS))),
                    mass_template=np.array([]), mass_template_err=np.array([]),
                    mass_sed=np.array([]), n_bands=np.array([]), dt_nearest_spec=np.array([]))
    t0 = min(f['t'].min() for f in fits.values())
    t1 = max(f['t'].max() for f in fits.values())
    grid = np.arange(np.floor(t0), np.ceil(t1) + 1, 1.0)
    mag = np.full((len(grid), len(P.BANDS)), np.nan)
    err = np.full((len(grid), len(P.BANDS)), np.nan)
    for k, b in enumerate(P.BANDS):
        if b not in fits:
            continue
        m, s = fits[b]['predict'](grid)
        ok = np.array([P.coverage_ok(fits[b]['t'], t, max_side_gap=max_gap) for t in grid])
        mag[ok, k], err[ok, k] = m[ok], s[ok]
    nb = np.isfinite(mag).sum(1)

    ref = [r for r in cal if r.get('ot_ready')]
    t_ref = np.array([r['mjd'] for r in ref])
    m_tpl = np.full(len(grid), np.nan)
    m_tpl_err = np.full(len(grid), np.nan)
    m_sed = np.full(len(grid), np.nan)
    dt_near = np.full(len(grid), np.nan)
    for i, t in enumerate(grid):
        if nb[i] < 2:
            continue
        Fobs = {b: filters.mag_to_flux(b, mag[i, k]) for k, b in enumerate(P.BANDS) if np.isfinite(mag[i, k])}
        if np.isfinite(lo):
            m_sed[i] = _sed_mass({b: (F,) for b, F in Fobs.items()}, filters, lo * (1 + z), hi * (1 + z))
        if not len(ref):
            continue
        est, wts = [], []
        left = np.where(t_ref <= t)[0]
        right = np.where(t_ref > t)[0]
        for j in ([left[-1]] if len(left) else []) + ([right[0]] if len(right) else []):
            r = ref[j]
            ratios = [r['K'][b][1] / r['K_tilde'] / Fobs[b] for b in r['K'] if b in Fobs]
            if len(ratios) < 2:
                continue
            est.append(r['mass_window'] / np.median(ratios))
            wts.append(1.0 / (abs(t - r['mjd']) + 1.0))
        if est:
            est, wts = np.array(est), np.array(wts)
            m_tpl[i] = np.sum(est * wts) / np.sum(wts)
            m_tpl_err[i] = 0.5 * np.ptp(est) if len(est) > 1 else np.nan
            dt_near[i] = np.min(np.abs(t_ref - t))
    # Total-flux curves are meant for interpolation between spectra: outside the span of
    # the OT-ready spectra (+-5 d) they would be extrapolations on sparse photometry.
    span = (t_ref.min() - 5, t_ref.max() + 5) if len(ref) else (np.inf, -np.inf)
    outside = (grid < span[0]) | (grid > span[1])
    m_tpl[outside] = m_tpl_err[outside] = m_sed[outside] = np.nan
    return dict(mjd=grid, mag=mag, mag_err=err, n_bands=nb, mass_template=m_tpl,
                mass_template_err=m_tpl_err, mass_sed=m_sed, dt_nearest_spec=dt_near)
