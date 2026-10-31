"""Spectrum cleaning (paper Appendix A) and flux-conserving resampling.

Pixel-level bookkeeping uses bit flags so every modification is traceable:
"""
import numpy as np

FLAG_SPIKE = 1        # cosmic ray / spike replaced (Whitaker & Hayes 2018 modified z-score)
FLAG_TELLURIC = 2     # inside O2 A/B telluric band, bridged linearly
FLAG_HOSTLINE = 4     # narrow host-galaxy emission line removed
FLAG_EXCLUDED = 8     # range flagged 'exclude' by OSC / trimmed edge (not used in output)

# Observed-frame telluric bands bridged linearly (O2 B and A bands).
TELLURIC_BANDS = [(6860.0, 6950.0), (7590.0, 7700.0)]
# Narrow host emission lines (rest, A): [OII], Hb, [OIII], [NII], Ha, [SII].
HOST_LINES = [3727.4, 4861.3, 4958.9, 5006.8, 6548.0, 6562.8, 6583.4, 6716.4, 6730.8]


def basic_clean(wave, flux, err, exclude=None):
    """Sort, drop non-finite/duplicate pixels, apply OSC 'exclude' ranges."""
    good = np.isfinite(wave) & np.isfinite(flux) & (wave > 0)
    if err is not None:
        err = np.where(np.isfinite(err) & (err > 0), err, np.nan)
    o = np.argsort(wave[good], kind='stable')
    w, f = wave[good][o], flux[good][o]
    e = err[good][o] if err is not None else None
    w, u = np.unique(w, return_index=True)
    f = f[u]
    e = e[u] if e is not None else None
    # Absurd values (some merged OSC spectra carry 1e11-1e12 junk next to 1e-15 data).
    keep = np.abs(f) < 1e3 * np.median(np.abs(f)) if len(f) else np.ones(0, bool)
    for ex in exclude or []:
        if 'below' in ex:
            keep &= w >= float(ex['below'])
        if 'above' in ex:
            keep &= w <= float(ex['above'])
    return w[keep], f[keep], (e[keep] if e is not None else None)


def trim_edges(w, f, e, nwin=15, nrun=25, max_level=20.0):
    """Trim noisy ends: drop leading/trailing pixels until, for `nrun`
    consecutive pixels, the running median of the flux is positive, the pixel
    scatter is below the local signal, and the level is < `max_level` x the
    median of the central half of the spectrum (merged UV segments of some OSC
    spectra sit 100-1000x above the optical flux)."""
    n = len(f)
    if n < 3 * nwin:
        return w, f, e
    pad = nwin // 2
    fp = np.pad(f, pad, mode='edge')
    win = np.lib.stride_tricks.sliding_window_view(fp, nwin)[:n]
    med = np.median(win, axis=1)
    mad = np.median(np.abs(win - med[:, None]), axis=1) * 1.4826
    central = np.median(f[n // 4: 3 * n // 4])
    ok = (med > 0) & (mad < med) & (med < max_level * abs(central))
    run = np.convolve(ok.astype(int), np.ones(min(nrun, n), int), mode='valid') == min(nrun, n)
    if not run.any():
        return w[:0], f[:0], (e[:0] if e is not None else None)
    i0 = np.argmax(run)
    i1 = len(run) - np.argmax(run[::-1]) - 1 + min(nrun, n)
    sl = slice(i0, i1)
    return w[sl], f[sl], (e[sl] if e is not None else None)


def _running(f, nwin, func):
    pad = nwin // 2
    return func(np.lib.stride_tricks.sliding_window_view(np.pad(f, pad, mode='reflect'), nwin)[:len(f)], axis=1)


def remove_spikes(f, threshold=3.5, m=3, n_iter=3, nsig=5.0, max_width=3, min_frac=0.02):
    """Cosmic Ray Deleter algorithm (Ramirez et al. 2024, Appendix A):
    modified z-score of the first difference, |z| > threshold marks candidate
    spikes, which are replaced by the mean of the non-spike pixels within +-m.

    On smooth high-S/N spectra the z-score alone also fires on every steep real
    feature, so a candidate is confirmed only if it (i) deviates from a
    (2m+1)-pixel running median by > nsig x the local robust scatter of
    (f - running median) and (ii) belongs to a run of <= max_width such pixels.
    """
    f = f.copy()
    flags = np.zeros(len(f), bool)
    for _ in range(n_iter):
        d = np.diff(f, prepend=f[0])
        M = np.median(d)
        mad = np.median(np.abs(d - M))
        if mad == 0:
            break
        z = 0.6745 * (d - M) / mad
        cand = np.abs(z) > threshold
        cand[:-1] |= cand[1:]          # a spike at i gives |z| large at i and i+1
        resid = f - _running(f, 2 * m + 1, np.median)
        sigma = 1.4826 * _running(np.abs(resid), 51, np.median)
        level = np.abs(_running(f, 51, np.median))
        # (iii) and > min_frac of the local flux: on oversampled, very smooth spectra
        # sigma -> 0 and every extremum would otherwise qualify
        spike = cand & (np.abs(resid) > np.maximum(nsig * sigma, min_frac * level))
        # wide runs are real (or noise-dominated) structure, not cosmic rays
        edges = np.flatnonzero(np.diff(np.concatenate([[0], spike.astype(int), [0]])))
        for a, b in zip(edges[::2], edges[1::2]):
            if b - a > max_width:
                spike[a:b] = False
        if not spike.any():
            break
        new = f.copy()
        for i in np.flatnonzero(spike):
            nb = [j for j in range(max(i - m, 0), min(i + m + 1, len(f))) if not spike[j]]
            if nb:
                new[i] = np.mean(f[nb])
                flags[i] = True
        f = new
    return f, flags


def bridge(w, f, lo, hi, nedge=5):
    """Replace f inside [lo, hi] by a straight line between the edge medians."""
    inside = (w >= lo) & (w <= hi)
    left = np.where(w < lo)[0][-nedge:]
    right = np.where(w > hi)[0][:nedge]
    if not inside.any() or len(left) == 0 or len(right) == 0:
        return f, np.zeros(len(f), bool)
    x0, y0 = np.median(w[left]), np.median(f[left])
    x1, y1 = np.median(w[right]), np.median(f[right])
    f = f.copy()
    f[inside] = y0 + (y1 - y0) * (w[inside] - x0) / (x1 - x0)
    return f, inside


def remove_host_lines(w, f, z, half_window=40.0, max_fwhm=15.0, nsig=5.0):
    """Remove narrow host emission lines at the SN redshift.

    Only an excess that is narrower than `max_fwhm` A (so intrinsic broad SN
    lines such as SN II H-alpha are untouched) and > nsig above a running
    median continuum is bridged.
    """
    flags = np.zeros(len(f), bool)
    f = f.copy()
    for lam0 in HOST_LINES:
        lc = lam0 * (1 + z)
        reg = (w > lc - half_window) & (w < lc + half_window)
        core = np.abs(w - lc) < 12.0
        if reg.sum() < 15 or core.sum() < 2:
            continue
        cont = np.median(f[reg & ~core])
        noise = 1.4826 * np.median(np.abs(f[reg & ~core] - cont))
        exc = f[core] - cont
        if noise <= 0 or exc.max() < nsig * noise:
            continue
        above = core & (f - cont > 0.5 * exc.max())
        width = w[above].max() - w[above].min() if above.sum() > 1 else 0.0
        if width > max_fwhm:
            continue
        lo, hi = w[above].min() - 3.0, w[above].max() + 3.0
        f, ins = bridge(w, f, lo, hi, nedge=3)
        flags |= ins
    return f, flags


def clean_spectrum(spec, z):
    """Full cleaning of one OSC spectrum. Returns observed-frame arrays + flags."""
    w, f, e = basic_clean(spec['wave'], spec['flux'], spec['err'], spec['exclude'])
    if spec['deredshifted']:
        zs = spec['spec_redshift'] if np.isfinite(spec['spec_redshift']) else z
        w = w * (1 + zs)            # back to the observed frame (photometry is observed-frame)
    w, f, e = trim_edges(w, f, e)
    flags = np.zeros(len(w), np.uint8)
    if len(w) < 50:
        return w, f, e, flags
    f, sp = remove_spikes(f)
    flags[sp] |= FLAG_SPIKE
    f, hl = remove_host_lines(w, f, z)
    flags[hl] |= FLAG_HOSTLINE
    for lo, hi in TELLURIC_BANDS:
        f, tl = bridge(w, f, lo, hi)
        flags[tl] |= FLAG_TELLURIC
    return w, f, e, flags


def snr_estimate(f):
    """Median S/N per pixel from the robust scatter of second differences."""
    d2 = np.diff(f, 2)
    noise = 1.4826 * np.median(np.abs(d2 - np.median(d2))) / np.sqrt(6)
    return np.median(f) / noise if noise > 0 else np.inf


def rebin_flux_conserving(w, f, edges):
    """Average f_lambda in each bin [edges[k], edges[k+1]] using the exact
    integral of the linear interpolant; bins not fully covered -> NaN.
    sum(out * diff(edges)) equals the integral of the spectrum over the bins.
    """
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(w))])

    def F(x):
        x = np.clip(x, w[0], w[-1])
        i = np.clip(np.searchsorted(w, x) - 1, 0, len(w) - 2)
        dx = x - w[i]
        slope = (f[i + 1] - f[i]) / (w[i + 1] - w[i])
        return cum[i] + f[i] * dx + 0.5 * slope * dx ** 2

    out = (F(edges[1:]) - F(edges[:-1])) / np.diff(edges)
    covered = (edges[:-1] >= w[0]) & (edges[1:] <= w[-1])
    out[~covered] = np.nan
    return out


def rebin_flags(w, flags, edges):
    """OR of the pixel flags falling in each bin."""
    idx = np.searchsorted(edges, w) - 1
    out = np.zeros(len(edges) - 1, np.uint8)
    ok = (idx >= 0) & (idx < len(out))
    np.bitwise_or.at(out, idx[ok], flags[ok])
    return out


def rebin_error(w, e, edges):
    """Approximate per-bin error: per-pixel errors averaged in quadrature over the bin."""
    if e is None:
        return np.full(len(edges) - 1, np.nan)
    good = np.isfinite(e)
    if good.sum() < 2:
        return np.full(len(edges) - 1, np.nan)
    idx = np.searchsorted(edges, w[good]) - 1
    nb = len(edges) - 1
    ok = (idx >= 0) & (idx < nb)
    s2 = np.bincount(idx[ok], weights=e[good][ok] ** 2, minlength=nb)
    cnt = np.bincount(idx[ok], minlength=nb)
    with np.errstate(invalid='ignore', divide='ignore'):
        out = np.sqrt(s2) / cnt
    # bins narrower than the native pixel: fall back to interpolated pixel error
    interp = np.interp(0.5 * (edges[1:] + edges[:-1]), w[good], e[good], left=np.nan, right=np.nan)
    return np.where(cnt > 0, out, interp)
