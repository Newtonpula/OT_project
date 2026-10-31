"""Validation of data/processed/sn_spectra_ot.h5.

    python scripts/validate_dataset.py

Writes figures and data/processed/validation/validation_report.md:
  1. zero points vs paper Table B.1
  2. calibration: per-band K_x/K~ and leave-one-band-out residuals (paper Fig. 8 analogue)
  3. sample: golden counts and phase ranges vs paper Table C.1
  4. total flux: rebinning conservation, spectral vs photometric (SED) totals,
     leave-one-out test of the daily photometric total-flux curve
  5. usability: unbalanced-OT barycenter smoke test (numpy, no POT needed)
  6. figures: cleaning QA, per-SN galleries
"""
import csv
import os
import sys

import h5py
import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from snprep import spectra as S                 # noqa: E402
from snprep.load import SNDataset               # noqa: E402
from snprep.osc import load_osc_merged          # noqa: E402

H5 = os.path.join(ROOT, 'data', 'processed', 'sn_spectra_ot.h5')
OUT = os.path.join(ROOT, 'data', 'processed', 'validation')
C1, C2, C3, INK, MUTED = '#2a78d6', '#eb6834', '#1baf7a', '#0b0b0b', '#8a8984'
PAPER_F0 = dict(B=6.460e-9, V=3.675e-9, R=2.232e-9, I=1.177e-9, g=4.791e-9, r=2.818e-9, i=1.907e-9)
PAPER_LOO = {  # paper Fig. 8: fraction |phi| < 0.1, leave-one-out (Ia, II)
    'B': (67.6, 68.2), 'V': (85.1, 83.0), 'R': (84.0, 75.4), 'I': (60.6, 80.6),
    'g': (63.5, 54.5), 'r': (76.6, 84.5), 'i': (56.2, 84.7)}

plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#e6e5e0', 'grid.linewidth': 0.6, 'legend.frameon': False})


def rpt(lines, s=''):
    print(s)
    lines.append(s)


def robust_sigma(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))


# ---------------------------------------------------------------- 1. zero points
def check_zero_points(h, L):
    rpt(L, '## 1. Zero points (flux of a 0-mag source, photon-weighted)\n')
    rpt(L, '| band | system | F0 this work | F0 paper (Table B.1) | ratio |')
    rpt(L, '|---|---|---|---|---|')
    for b in h.attrs['bands'].split(','):
        a = h['filters'][b].attrs
        p = PAPER_F0.get(b)
        rpt(L, f"| {b} | {a['system']} | {a['F0']:.4e} | {p:.3e} | {a['F0'] / p:.3f} |" if p else
            f"| {b} | {a['system']} | {a['F0']:.4e} | - | - |")
    rpt(L)


# ---------------------------------------------------------------- 2. calibration
def check_calibration(h, L):
    rpt(L, '## 2. Calibration against photometry\n')
    bands = h.attrs['bands'].split(',')        # column order of K_band_over_K_tilde
    per_band = {b: [] for b in bands}
    loo = {t: {b: [] for b in bands} for t in ('Ia', 'II')}
    mad_all = {t: [] for t in ('Ia', 'II')}
    for name, g in h['sne'].items():
        t = g.attrs['type']
        s = g['spectra']
        if not s.attrs['n']:
            continue
        K = s['K_band_over_K_tilde'][:]
        gold = s['golden'][:]
        mad_all[t].extend(s['K_mad_rel'][:].tolist())
        for row in K[gold]:
            ok = np.isfinite(row)
            for j in np.where(ok)[0]:
                per_band[bands[j]].append(row[j])
            if ok.sum() >= 4:                      # leave-one-band-out, as in the paper
                for j in np.where(ok)[0]:
                    others = row[ok & (np.arange(len(row)) != j)]
                    loo[t][bands[j]].append(row[j] / np.median(others) - 1)
    for t in ('Ia', 'II'):
        m = np.array(mad_all[t])
        rpt(L, f'- {t}: {len(m)} calibrated spectra, MAD(K)/K~ median {np.median(m):.3f}; '
               f'{np.mean(m < 0.1) * 100:.1f}% pass the 10% golden cut')
    rpt(L, '\nPer-band K_x / K~ for golden spectra (a band-dependent offset = zero-point or system mismatch):\n')
    rpt(L, '| band | N | median | robust sigma | frac within 10% |')
    rpt(L, '|---|---|---|---|---|')
    for b in bands:
        x = np.array(per_band[b])
        if len(x):
            rpt(L, f'| {b} | {len(x)} | {np.median(x):.3f} | {robust_sigma(x):.3f} | {np.mean(np.abs(x - 1) < 0.1) * 100:.1f}% |')
    rpt(L, '\nLeave-one-band-out relative residual phi = F_syn/F_obs - 1 (golden spectra with >= 4 bands).')
    rpt(L, 'Paper Fig. 8 values are for OT-*interpolated* spectra, so they are a looser reference:\n')
    rpt(L, '| band | N (Ia) | |phi|<0.1 Ia | paper Ia | N (II) | |phi|<0.1 II | paper II |')
    rpt(L, '|---|---|---|---|---|---|---|')
    for b in bands:
        a, c = np.array(loo['Ia'][b]), np.array(loo['II'][b])
        pa, pc = PAPER_LOO.get(b, ('-', '-'))
        fa = f'{np.mean(np.abs(a) < 0.1) * 100:.1f}%' if len(a) else '-'
        fc = f'{np.mean(np.abs(c) < 0.1) * 100:.1f}%' if len(c) else '-'
        rpt(L, f'| {b} | {len(a)} | {fa} | {pa} | {len(c)} | {fc} | {pc} |')
    rpt(L)

    fig, axs = plt.subplots(1, 7, figsize=(15, 2.6), sharey=False)
    for ax, b in zip(axs, ['B', 'V', 'R', 'I', 'g', 'r', 'i']):
        for t, c in (('Ia', C1), ('II', C2)):
            x = np.array(loo[t][b])
            if len(x):
                ax.hist(np.clip(x, -0.3, 0.3), bins=np.linspace(-0.3, 0.3, 31), histtype='step', lw=1.5, color=c,
                        density=True, label=f'{t} ({np.mean(np.abs(x) < 0.1) * 100:.0f}%)')
        ax.axvline(-0.1, color=MUTED, ls=':', lw=0.8)
        ax.axvline(0.1, color=MUTED, ls=':', lw=0.8)
        ax.set_title(b)
        ax.set_yticks([])
        ax.legend(fontsize=7, loc='upper left')
    axs[3].set_xlabel('leave-one-band-out residual  F_syn/F_obs − 1  (golden spectra, ≥4 bands)')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'calibration_loo_residuals.png'), dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------- 3. sample
def check_sample(L):
    rpt(L, '## 3. Sample vs paper Table C.1\n')
    rows = list(csv.DictReader(open(os.path.join(ROOT, 'data', 'processed', 'summary_sne.csv'))))
    f = lambda r, k: float(r[k]) if r[k] not in ('', 'nan') else np.nan
    for t in ('Ia', 'II'):
        R = [r for r in rows if r['type'] == t]
        rpt(L, f"- {t}: {len(R)} SNe; golden spectra {sum(int(r['n_golden']) for r in R)}, OT-ready "
               f"{sum(int(r['n_ot_ready']) for r in R)} (paper golden: {sum(int(r['paper_n_spec']) for r in R)}); "
               f"SNe with >= 3 OT-ready spectra: {sum(int(r['n_ot_ready']) >= 3 for r in R)}")
    low = [r for r in rows if int(r['n_golden']) < 0.5 * int(r['paper_n_spec'])]
    rpt(L, '\nSNe with fewer than half of the paper\'s spectra (spectra absent from both OSC and the CSP release, '
           'or photometry too sparse to calibrate):\n')
    rpt(L, '| SN | type | input spectra (OSC+CSP) | calibrated | golden | paper | phot bands |')
    rpt(L, '|---|---|---|---|---|---|---|')
    for r in low:
        rpt(L, f"| {r['name']} | {r['type']} | {r['n_input_spec']} | {r['n_calibrated']} | {r['n_golden']} | "
               f"{r['paper_n_spec']} | {r['phot_bands']} |")
    d0 = np.array([f(r, 'phase_min') - f(r, 'paper_phase_min') for r in rows])
    ok = np.isfinite(d0)
    rpt(L, f"\nPhase of the earliest golden spectrum minus the paper's: median {np.median(d0[ok]):+.1f} d, "
           f"|diff| <= 2 d for {np.sum(np.abs(d0[ok]) <= 2)}/{ok.sum()} SNe (differences come from different "
           "spectrum subsets and from t_ref; SNe Ia: V max, SNe II: t_PT).")
    noref = [r['name'] for r in rows if not np.isfinite(f(r, 't_ref'))]
    rpt(L, f"- t_ref could not be measured for {len(noref)} SNe: {', '.join(noref)} (phase = NaN, use MJD).\n")

    fig, ax = plt.subplots(figsize=(4.2, 4))
    for t, c in (('Ia', C1), ('II', C2)):
        R = [r for r in rows if r['type'] == t]
        ax.scatter([int(r['paper_n_spec']) for r in R], [int(r['n_golden']) for r in R], s=14, color=c,
                   edgecolor='white', lw=0.5, label=t)
    ax.plot([0, 80], [0, 80], color=MUTED, lw=0.8, ls='--')
    ax.set_xlabel('golden spectra in paper (Table C.1)')
    ax.set_ylabel('golden spectra here')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'sample_vs_paper.png'), dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------- 4. total flux
def check_total_flux(h, ds, L):
    rpt(L, '## 4. Total flux ("mass") per epoch\n')
    rr, sed_ratio, ph_ratio, types = [], [], [], []
    loo_err, loo_dt = [], []
    for name, g in h['sne'].items():
        s = g['spectra']
        if not s.attrs['n']:
            continue
        ot = s['ot_ready'][:]
        rr.extend(s['rebin_int_ratio'][:][ot].tolist())
        mw, ms = s['mass_window'][:][ot], s['mass_sed'][:][ot]
        ok = np.isfinite(mw) & np.isfinite(ms)
        sed_ratio.extend((ms[ok] / mw[ok]).tolist())
        ph_ratio.extend(s['phase'][:][ot][ok].tolist())
        types.extend([g.attrs['type']] * ok.sum())
        if ot.sum() >= 3:
            e, dt = _loo_template(s)
            loo_err.extend(e)
            loo_dt.extend(dt)
    rr, sed_ratio, ph_ratio, types = map(np.array, (rr, sed_ratio, ph_ratio, types))
    rpt(L, f'- Rebinning conserves flux: grid integral / native integral = {np.median(rr):.5f} '
           f'(1st-99th pct {np.percentile(rr, 1):.5f}-{np.percentile(rr, 99):.5f}).')
    for t in ('Ia', 'II'):
        x = sed_ratio[types == t]
        rpt(L, f'- {t}: photometry-only SED total / spectral total = {np.median(x):.3f} '
               f'(robust sigma {robust_sigma(x):.3f}, N={len(x)}).')
    e, dt = np.array(loo_err), np.array(loo_dt)
    rpt(L, f'- Leave-one-out test of the daily photometric total-flux curve (mass_template) at {len(e)} '
           f'spectral epochs, predicting each OT-ready spectrum\'s total from its neighbours + photometry: '
           f'median |rel. error| {np.median(np.abs(e)) * 100:.1f}%, 90th pct {np.percentile(np.abs(e), 90) * 100:.1f}%; '
           f'for neighbour gaps <= 10 d: {np.median(np.abs(e[dt <= 10])) * 100:.1f}%.\n')

    fig, axs = plt.subplots(1, 2, figsize=(9, 3.2))
    for t, c in (('Ia', C1), ('II', C2)):
        sel = types == t
        axs[0].scatter(ph_ratio[sel], sed_ratio[sel], s=5, color=c, alpha=0.6, lw=0, label=t)
    axs[0].axhline(1, color=MUTED, lw=0.8)
    axs[0].set_ylim(0.7, 1.3)
    axs[0].set_xlim(-130, 150)
    axs[0].set_xlabel('phase [d]')
    axs[0].set_ylabel('SED-from-photometry / spectral total')
    axs[0].legend()
    axs[1].hist(np.clip(e, -0.3, 0.3), bins=np.linspace(-0.3, 0.3, 41), color=C1, histtype='stepfilled', alpha=0.85)
    axs[1].set_xlabel('LOO error of photometric total-flux curve')
    axs[1].set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'total_flux_checks.png'), dpi=120)
    plt.close(fig)


def _loo_template(s):
    """Leave-one-out test of the photometric total-flux estimator (lightcurve/mass_template):
    predict the total flux of OT-ready spectrum j from its neighbours j-1, j+1 rescaled to
    the photometry at t_j (median band ratio), without using spectrum j itself."""
    ot = s['ot_ready'][:]
    mjd = s['mjd'][:][ot]
    mass = s['mass_window'][:][ot]
    Fsyn, Fobs = s['F_syn_band'][:][ot], s['F_obs_band'][:][ot]
    errs, dts = [], []
    for j in range(1, len(mjd) - 1):
        est, w = [], []
        for k in (j - 1, j + 1):
            ratio = Fsyn[k] / Fobs[j]
            ratio = ratio[np.isfinite(ratio)]
            if len(ratio) < 2:
                continue
            est.append(mass[k] / np.median(ratio))
            w.append(1 / (abs(mjd[j] - mjd[k]) + 1))
        if est:
            pred = np.sum(np.array(est) * w) / np.sum(w)
            errs.append(pred / mass[j] - 1)
            dts.append(max(mjd[j] - mjd[j - 1], mjd[j + 1] - mjd[j]))
    return errs, dts


# ---------------------------------------------------------------- 5. UOT smoke test
def uot_barycenter(A, M, reg, reg_m, weights, n_iter=3000, tol=1e-7):
    """Unbalanced Sinkhorn barycenter (Chizat et al. 2018; the scheme of POT's
    ot.unbalanced.barycenter_unbalanced_sinkhorn). Scalings are kept as logs and
    the barycenter update (exponent 1/(1-fi), huge for small reg) is done with
    logsumexp, so a small `reg` neither underflows nor overflows."""
    from scipy.special import logsumexp
    K = np.exp(-M / reg)
    fi = reg_m / (reg_m + reg)

    def log_matmul(Kmat, logx):         # log(Kmat @ exp(logx)), column-wise max shift
        mx = logx.max(axis=0, keepdims=True)
        return np.log(np.maximum(Kmat @ np.exp(logx - mx), 1e-300)) + mx

    logA = np.log(A)
    logv = np.zeros_like(A)
    logq = np.zeros(A.shape[0])
    for _ in range(n_iter):
        logu = fi * (logA - log_matmul(K, logv))
        logKtu = log_matmul(K.T, logu)
        logq_new = logsumexp(np.log(weights)[None, :] + (1 - fi) * logKtu, axis=1) / (1 - fi)
        logv = fi * (logq_new[:, None] - logKtu)
        done = np.max(np.abs(np.exp(logq_new) - np.exp(logq))) < tol * np.exp(logq_new).max()
        logq = logq_new
        if done:
            break
    return np.exp(logq)


def check_uot(ds, L, max_gap=40.0, rebin=4, reg=3e-4, reg_m=1.0):
    rpt(L, '## 5. Usability: unbalanced-OT interpolation smoke test\n')
    res = []
    for name in ds.names(min_ot_ready=3):
        m = ds.ot_measures(name)
        t, mass = m['t'], m['mass']
        nb = mass.shape[1] // rebin * rebin
        X = mass[:, :nb].reshape(len(t), -1, rebin).sum(2)        # 20 A bins for speed
        x = m['wave'][:nb].reshape(-1, rebin).mean(1)
        # one interior triplet per SN: the widest bracket <= max_gap with the target near alpha = 0.5
        cands = [j for j in range(1, len(t) - 1) if 0 < t[j + 1] - t[j - 1] <= max_gap]
        if not cands:
            continue
        alpha = lambda j: (t[j] - t[j - 1]) / (t[j + 1] - t[j - 1])
        j = max(cands, key=lambda j: (t[j + 1] - t[j - 1]) * (1 - 2 * abs(alpha(j) - 0.5)))
        a = alpha(j)
        scale = 0.5 * (X[j - 1].sum() + X[j + 1].sum())       # work in O(1) units, restore after
        A = np.c_[X[j - 1], X[j + 1]] / scale + 1e-12
        M = (x[:, None] - x[None, :]) ** 2
        M /= M.max()
        q = uot_barycenter(A, M, reg=reg, reg_m=reg_m, weights=np.array([1 - a, a]), n_iter=5000) * scale
        lin = (1 - a) * X[j - 1] + a * X[j + 1]
        truth = X[j]
        good = truth > 0.05 * truth.max()
        eps = lambda y: np.mean(np.abs(y / y.sum() - truth / truth.sum())[good] / (truth / truth.sum())[good])
        res.append((name, ds.info(name)['type'], t[j + 1] - t[j - 1], eps(q), eps(lin),
                    q.sum() / truth.sum() - 1, lin.sum() / truth.sum() - 1))
    if not res:
        return
    arr = np.array([r[2:] for r in res], float)
    rpt(L, f'For each of {len(res)} SNe, the OT-ready spectrum with the widest bracketing gap '
           f'(<= {max_gap:g} d, target near the middle) is predicted from its two neighbours. 20 A bins; '
           f'cost = squared wavelength distance normalised to 1; reg={reg:g} (entropic blur ~ sqrt(reg) x window '
           f'width, ~{np.sqrt(reg) * 4000:.0f} A for a 4000 A window), reg_m={reg_m:g}. '
           'eps = mean |relative residual| of mass-normalised spectra (paper Eq. 4); '
           'mass error = predicted/true total - 1.\n')
    rpt(L, '| bracket gap | N | eps UOT | eps linear | |mass err| UOT | |mass err| linear |')
    rpt(L, '|---|---|---|---|---|---|')
    for lo, hi in ((0, 10), (10, 20), (20, max_gap + 1)):
        s = (arr[:, 0] > lo) & (arr[:, 0] <= hi)
        if s.any():
            rpt(L, f'| {lo}-{min(hi, max_gap):g} d | {s.sum()} | {np.median(arr[s, 1]):.3f} | {np.median(arr[s, 2]):.3f} | '
                   f'{np.median(np.abs(arr[s, 3])) * 100:.1f}% | {np.median(np.abs(arr[s, 4])) * 100:.1f}% |')
    rpt(L, '\nThis only shows that the arrays work as UOT input (non-negative, finite, common support, physical '
           'masses that can be rescaled to O(1)); it is an untuned baseline, not a result about the method.\n')
    return res


# ---------------------------------------------------------------- 6. figures
def fig_cleaning(L):
    rows = list(csv.DictReader(open(os.path.join(ROOT, 'data', 'processed', 'summary_spectra.csv'))))
    rows = [r for r in rows if r['golden'] == 'True']
    rows.sort(key=lambda r: -int(r['n_spike_px']))
    pick = rows[:2] + [r for r in rows if int(r['n_host_px']) > 0][:1] + [rows[len(rows) // 2]]
    fig, axs = plt.subplots(len(pick), 1, figsize=(10, 2.3 * len(pick)))
    for ax, r in zip(axs, pick):
        meta, specs, _ = load_osc_merged(os.path.join(ROOT, 'data', 'raw', 'osc'), r['sn'])
        sp = next(x for x in specs if x['filename'] == r['filename'] and abs(x['mjd'] - float(r['mjd'])) < 1e-3)
        w0, f0, _ = S.basic_clean(sp['wave'], sp['flux'], sp['err'], sp['exclude'])
        if sp['deredshifted']:
            w0 = w0 * (1 + (sp['spec_redshift'] if np.isfinite(sp['spec_redshift']) else meta['redshift']))
        w, f, e, fl = S.clean_spectrum(sp, meta['redshift'])
        ax.plot(w0, f0, color=MUTED, lw=0.6, label='raw (OSC)')
        ax.plot(w, f, color=C1, lw=0.7, label='cleaned')
        for bit, c, lab in ((1, C2, 'spike'), (2, C3, 'telluric'), (4, '#6250d6', 'host line')):
            sel = (fl & bit) > 0
            if sel.any():
                ax.scatter(w[sel], f[sel], s=6, color=c, zorder=3, label=lab)
        ax.set_title(f"{r['sn']}  MJD {float(r['mjd']):.2f}  {r['instrument']}  ({r['n_spike_px']} spike px)", fontsize=8)
        ax.set_ylim(np.percentile(f, 0.5) - 0.1 * np.ptp(f), np.percentile(f, 99.8) * 1.15)
        ax.legend(fontsize=7, ncol=5, loc='upper right')
    axs[-1].set_xlabel('observed wavelength [Å]')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'cleaning_examples.png'), dpi=120)
    plt.close(fig)


def fig_gallery(ds, names):
    for name in names:
        m = ds.ot_measures(name)
        lc = ds.total_flux_curve(name)
        fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw=dict(width_ratios=[1.3, 1]))
        ax = axs[0]
        norm = m['mass'].max()
        n = len(m['t'])
        cmap = plt.get_cmap('Blues')
        for k in range(n):
            ax.plot(m['wave'], m['mass'][k] / norm * 3 + 0 * k, color=cmap(0.35 + 0.65 * k / max(n - 1, 1)), lw=0.7)
        ax.set_xlabel('rest wavelength [Å]')
        ax.set_ylabel('mass per 5 Å bin (relative, absolute scale kept)')
        ax.set_title(f'{name}: {n} OT-ready spectra, early (light) to late (dark)')
        ax = axs[1]
        t0 = ds.info(name)['t_ref'] if np.isfinite(ds.info(name)['t_ref']) else m['mjd'].min()
        z = ds.info(name)['redshift']
        tl = (lc['mjd'] - t0) / (1 + z)
        # NaNs left in place so that lines break across gaps without photometry
        ax.plot(tl, lc['mass_template'], color=C1, lw=2, label='daily total (photometry + nearest spectra)')
        ax.plot(tl, lc['mass_sed'], color=C3, lw=2, ls='--', label='daily total (photometry SED only)')
        ax.scatter(m['t'], m['total'], s=28, color=C2, edgecolor='white', lw=1, zorder=3, label='spectral total')
        ax.set_yscale('log')
        ax.set_xlabel('rest-frame days from t_ref' if np.isfinite(ds.info(name)['t_ref']) else 'rest-frame days')
        ax.set_ylabel('total flux in window [erg s$^{-1}$ cm$^{-2}$]')
        lo, hi = m['window']
        ax.set_title(f'window {lo:.0f}–{hi:.0f} Å (rest)')
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f'gallery_{name}.png'), dpi=110)
        plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    L = ['# Validation report: sn_spectra_ot.h5\n']
    ds = SNDataset(H5)
    with h5py.File(H5, 'r') as h:
        check_zero_points(h, L)
        check_calibration(h, L)
        check_sample(L)
        check_total_flux(h, ds, L)
    check_uot(ds, L)
    fig_cleaning(L)
    fig_gallery(ds, ['SN2011fe', 'SN2006cp', 'SN1999em', 'SN2006be'])
    rpt(L, '## 6. Figures\n')
    for fn in sorted(os.listdir(OUT)):
        if fn.endswith('.png'):
            rpt(L, f'- `{fn}`')
    with open(os.path.join(OUT, 'validation_report.md'), 'w') as f:
        f.write('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
