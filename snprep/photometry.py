"""Photometry selection, LOESS light-curve interpolation and phase references.

Ramirez et al. (2024) interpolate light curves with the Automated Loess
Regression (ALR, Rodriguez et al. 2019). `loess_fit` below follows the same
idea: local quadratic regression with tricube weights, error weighting,
smoothing span chosen by AICc, and iterative outlier rejection.
"""
import numpy as np
from scipy.optimize import curve_fit

# OSC band name -> canonical band. Paper uses Johnson-Cousins BVRI + SDSS gri.
# Swift/UVOT b and v (several SNe in the paper only have these) are kept as
# separate bands with their own filter curves.
BAND_MAP = {'B': 'B', 'V': 'V', 'R': 'R', 'Rc': 'R', 'I': 'I', 'Ic': 'I',
            'g': 'g', "g'": 'g', 'r': 'r', "r'": 'r', 'i': 'i', "i'": 'i'}
UVOT_MAP = {'B': 'b_uvot', 'V': 'v_uvot'}
BANDS = ['B', 'V', 'R', 'I', 'g', 'r', 'i', 'b_uvot', 'v_uvot']
VEGA_BANDS = {'B', 'V', 'R', 'I', 'b_uvot', 'v_uvot'}


def select_photometry(raw_phot, ab_minus_vega):
    """Filter OSC photometry to usable BVRIgri magnitudes.

    Returns a structured numpy array (mjd, band, mag, err, err_assumed, system, source)
    with BVRI on the Vega system and gri on the AB system.
    `ab_minus_vega[band]` converts magnitudes reported in the other system.
    """
    out = []
    for p in raw_phot:
        band = BAND_MAP.get(p.get('band'))
        if band is None or 'magnitude' not in p or 'time' not in p:
            continue
        if str(p.get('upperlimit', '')).lower() == 'true' or str(p.get('includeshost', '')).lower() == 'true':
            continue
        if 'model' in p or 'realization' in p:
            continue
        tel = f"{p.get('telescope', '')} {p.get('instrument', '')} {p.get('system', '')}".lower()
        if 'swift' in tel or 'uvot' in tel:      # UVOT b/v are not Johnson B/V
            band = UVOT_MAP.get(p.get('band'))
            if band is None:
                continue
        try:
            t = float(p['time'][0] if isinstance(p['time'], list) else p['time'])
            m = float(p['magnitude'])
        except (TypeError, ValueError):
            continue
        e = np.nan
        for k in ('e_magnitude',):
            if k in p:
                e = float(p[k])
        if not np.isfinite(e) and 'e_lower_magnitude' in p and 'e_upper_magnitude' in p:
            e = 0.5 * (float(p['e_lower_magnitude']) + float(p['e_upper_magnitude']))
        system = p.get('system') or ''
        # BVRI points labelled 'AB' in the OSC all come from Smartt+2015 (PESSTO) and
        # Graham+2017 (LCO); synthetic photometry of their spectra shows they are Vega
        # magnitudes (converting them leaves R/I off by the AB-Vega offsets, 0.16/0.41 mag),
        # so that label is ignored. gri labelled 'Vega' would be converted to AB.
        if band not in VEGA_BANDS and system == 'Vega':
            m += ab_minus_vega[band]
        if not (np.isfinite(t) and np.isfinite(m) and 5 < m < 30):
            continue
        out.append((t, band, m, e, system, p.get('source', '')))

    dt = [('mjd', 'f8'), ('band', 'U6'), ('mag', 'f8'), ('err', 'f8'),
          ('err_assumed', '?'), ('system', 'U16'), ('source', 'U32')]
    arr = np.array([(t, b, m, e, False, s, src) for t, b, m, e, s, src in out], dtype=dt)
    if len(arr) == 0:
        return arr
    # Missing / zero errors: assume the band median (>= 0.05 mag) and flag it.
    for b in BANDS:
        sel = arr['band'] == b
        good = sel & np.isfinite(arr['err']) & (arr['err'] > 0)
        med = np.median(arr['err'][good]) if good.any() else 0.05
        bad = sel & ~good
        arr['err'][bad] = max(med, 0.05)
        arr['err_assumed'][bad] = True
    arr['err'] = np.maximum(arr['err'], 0.01)   # error floor
    # Same epoch/band/mag listed twice (multiple OSC sources) -> keep one.
    key = np.array([f'{t:.2f}{b}{m:.3f}' for t, b, m in zip(arr['mjd'], arr['band'], arr['mag'])])
    _, keep = np.unique(key, return_index=True)
    arr = arr[np.sort(keep)]
    arr = arr[np.argsort(arr['mjd'], kind='stable')]
    return arr[~isolated_spikes(arr)]


def isolated_spikes(phot, jump=1.0, agree=0.5, max_dt=30.0):
    """Single points that jump > `jump` mag away from both time-neighbours
    (same band, within `max_dt` d) while those neighbours agree within `agree`
    mag. A real fast drop (e.g. SN II plateau end) moves monotonically and is kept."""
    bad = np.zeros(len(phot), bool)
    for b in BANDS:
        idx = np.where(phot['band'] == b)[0]
        t, m = phot['mjd'][idx], phot['mag'][idx]
        for k in range(1, len(idx) - 1):
            if t[k + 1] - t[k - 1] > 2 * max_dt:
                continue
            if (abs(m[k] - m[k - 1]) > jump and abs(m[k] - m[k + 1]) > jump
                    and abs(m[k - 1] - m[k + 1]) < agree):
                bad[idx[k]] = True
    return bad


# --------------------------------------------------------------------------
# LOESS

def _loess_matrix(x, w, x0, span_n, deg=2):
    """Rows of the linear smoother L (len(x0) x len(x)): yhat(x0) = L @ y.

    The window always holds >= deg+3 distinct epochs (0.1 d resolution): the same
    observation listed by two sources a few minutes apart must not count twice,
    otherwise the local quadratic is singular and the prediction explodes."""
    n = len(x)
    L = np.zeros((len(x0), n))
    k = int(min(max(span_n, deg + 2), n))
    tu = np.unique(np.round(x, 1))
    ku = min(deg + 3, len(tu))
    for j, xx in enumerate(x0):
        d = np.abs(x - xx)
        h = max(np.partition(d, k - 1)[k - 1], np.partition(np.abs(tu - xx), ku - 1)[ku - 1])
        h = max(h * 1.0001, 1e-6)
        u = d / h
        kw = np.where(u < 1, (1 - u ** 3) ** 3, 0.0) * w
        sel = kw > 0
        X = np.vander(x[sel] - xx, deg + 1, increasing=True)
        W = kw[sel]
        A = X.T @ (W[:, None] * X)
        try:
            coef_rows = np.linalg.solve(A + 1e-12 * np.eye(deg + 1) * np.trace(A), (X * W[:, None]).T)
        except np.linalg.LinAlgError:
            coef_rows = np.linalg.pinv(A) @ (X * W[:, None]).T
        L[j, sel] = coef_rows[0]
    return L


def loess_fit(t, y, err, spans=(0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0), clip=5.0, max_iter=5,
              clip_floor=0.03, protect_gap=5.0, cv_floor=0.02):
    """Fit an error-weighted LOESS; pick the span by leave-one-out CV; reject outliers.

    The span minimises the closed-form LOOCV score
    sum_i w_i (r_i / (1 - L_ii))^2 with w_i = 1/(err_i^2 + cv_floor^2), i.e. the
    expected interpolation error, which is what the calibration needs.

    Outliers: |residual| > clip * sqrt(err^2 + s_int^2 + clip_floor^2). Points next
    to a gap > `protect_gap` days are never rejected: in sparse stretches (e.g.
    the SN II plateau drop) a large residual means the smoother cannot follow
    the data, not that the point is bad.

    Returns dict with the retained data, chosen span (in points) and an
    `predict(x0)` helper returning (yhat, sigma).
    """
    t, y, err = map(np.asarray, (t, y, err))
    gaps = np.diff(t)
    protected = np.zeros(len(t), bool)
    protected[:-1] |= gaps > protect_gap
    protected[1:] |= gaps > protect_gap
    keep = np.ones(len(t), bool)
    best = None
    for _ in range(max_iter):
        tt, yy, ee = t[keep], y[keep], err[keep]
        n = len(tt)
        if n < 4:
            return None
        w = 1.0 / ee ** 2
        best = None
        for s in spans:
            span_n = max(int(np.ceil(s * n)), 5)
            if span_n > n and best is not None:
                continue
            L = _loess_matrix(tt, w, tt, span_n)
            res = yy - L @ yy
            trL = np.trace(L)
            lii = np.clip(np.diag(L), None, 0.99)
            cv = np.sum((res / (1 - lii)) ** 2 / (ee ** 2 + cv_floor ** 2))
            if best is None or cv < best[0]:
                best = (cv, span_n, res, trL)
        if best is None:
            return None
        _, span_n, res, trL = best
        # Extra scatter beyond the quoted errors (inter-source offsets etc.).
        chi2 = np.sum(res ** 2 / ee ** 2) / max(n - trL, 1)
        s_int = np.sqrt(max(np.median(res ** 2 - ee ** 2), 0.0))
        z = np.abs(res) / np.sqrt(ee ** 2 + s_int ** 2 + clip_floor ** 2)
        idx = np.where(keep)[0]
        bad = (z > clip) & ~protected[idx]
        if not bad.any():
            break
        # drop only the worst few per iteration so one outlier cannot drag neighbours out
        worst = np.argsort(-z * bad)[:max(1, bad.sum() // 3)]
        keep[idx[worst[bad[worst]]]] = False

    tt, yy, ee = t[keep], y[keep], err[keep]
    w = 1.0 / ee ** 2
    span_n = best[1]

    def predict(x0):
        x0 = np.atleast_1d(np.asarray(x0, float))
        L = _loess_matrix(tt, w, x0, span_n)
        yhat = L @ yy
        sig = np.sqrt((L ** 2) @ (ee ** 2 + s_int ** 2))
        return yhat, sig

    return dict(t=tt, y=yy, err=ee, keep=keep, span_n=span_n, s_int=s_int,
                chi2_red=chi2, predict=predict)


def reject_discrepant_sources(phot, max_offset=0.25, min_pts=3, err_floor=0.05):
    """Drop (band, source) groups that disagree with the other sources.

    For every source, the median offset of its points from a LOESS fit to all
    *other* sources of that band is measured (errors floored at `err_floor` so
    that one source with tiny quoted errors cannot dominate the reference).
    The single most discrepant source is dropped if |offset| > `max_offset`
    mag, and the procedure repeats. Returns (kept photometry, rejections).
    """
    keep = np.ones(len(phot), bool)
    rejected = []
    for b in BANDS:
        inb = np.where(phot['band'] == b)[0]
        while True:
            live = inb[keep[inb]]
            srcs = np.unique(phot['source'][live])
            if len(srcs) < 2:
                break
            offs = {}
            for s in srcs:
                mine = live[phot['source'][live] == s]
                other = live[phot['source'][live] != s]
                if len(mine) < min_pts or len(other) < 4:
                    continue
                fit = loess_fit(phot['mjd'][other], phot['mag'][other], np.maximum(phot['err'][other], err_floor))
                if fit is None:
                    continue
                tm = phot['mjd'][mine]
                inside = np.array([coverage_ok(fit['t'], t, max_side_gap=10.0) for t in tm])
                if inside.sum() < min_pts:
                    continue
                yhat, _ = fit['predict'](tm[inside])
                offs[s] = (np.median(phot['mag'][mine][inside] - yhat), mine)
            if not offs:
                break
            worst = max(offs, key=lambda s: abs(offs[s][0]))
            off, mine = offs[worst]
            if abs(off) <= max_offset:
                break
            keep[mine] = False
            rejected.append((b, worst, round(float(off), 3), int(len(mine))))
    return phot[keep], rejected


def coverage_ok(t_data, t0, max_side_gap=10.0, max_extrap=2.0):
    """Is t0 well sampled by the light curve? Points on both sides within
    `max_side_gap` days, or (at the ends) the nearest point within `max_extrap`."""
    t_data = np.asarray(t_data)
    before = t_data[t_data <= t0]
    after = t_data[t_data >= t0]
    if len(before) and len(after):
        return (t0 - before.max() <= max_side_gap) and (after.min() - t0 <= max_side_gap)
    near = np.min(np.abs(t_data - t0)) if len(t_data) else np.inf
    return near <= max_extrap


# --------------------------------------------------------------------------
# Phase references

def vmax_quadratic(phot, lc_fits):
    """Epoch of V-band maximum: quadratic fit to V data around the LOESS peak (SNe Ia)."""
    for band in ('V', 'v_uvot', 'B', 'b_uvot', 'g', 'r'):
        fit = lc_fits.get(band)
        if fit is None:
            continue
        t, y, e = fit['t'], fit['y'], fit['err']
        grid = np.arange(t.min(), t.max(), 0.1)
        yhat, _ = fit['predict'](grid)
        tpk = grid[np.argmin(yhat)]
        if tpk - t.min() < 1 or t.max() - tpk < 1:     # peak at the edge = not observed
            continue
        for half in (5.0, 7.0, 10.0):
            sel = np.abs(t - tpk) <= half
            if sel.sum() >= 4 and (t[sel] < tpk).sum() >= 1 and (t[sel] > tpk).sum() >= 1:
                c = np.polyfit(t[sel] - tpk, y[sel], 2, w=1 / e[sel])
                if c[0] > 0:
                    tq = tpk - c[1] / (2 * c[0])
                    if abs(tq - tpk) < half:
                        # Offsets of B, g, r maxima relative to V (typical SN Ia values).
                        off = {'V': 0.0, 'v_uvot': 0.0, 'B': 1.5, 'b_uvot': 1.5, 'g': 1.0, 'r': -0.5}[band]
                        return tq + off, f'quadratic_{band}max' + ('' if band == 'V' else f'+{off:g}d')
        return tpk, f'loess_{band}max'
    return np.nan, 'none'


def _olivares(t, a0, tpt, w0, p0, m0):
    """Olivares et al. (2010) transition + radioactive tail (V band, mag)."""
    return -a0 / (1 + np.exp((t - tpt) / w0)) + p0 * (t - tpt) + m0


def tpt_olivares(lc_fits, min_drop=0.8):
    """Midpoint of the plateau-to-tail transition, t_PT (SNe II), from V (fallback: R, r, I, i).

    The starting guess is the epoch of the steepest fading of the LOESS curve
    (largest brightness drop over +-15 d, which must exceed `min_drop` mag);
    the Olivares et al. (2010) model is then fitted to data within
    [t0 - 60, t0 + 100] d with the drop amplitude a0 >= 0.8 mag.
    """
    for band in ('V', 'R', 'r', 'I', 'i'):
        fit = lc_fits.get(band)
        if fit is None or len(fit['t']) < 8:
            continue
        t, y, e = fit['t'], fit['y'], np.hypot(fit['err'], max(fit['s_int'], 0.02))
        grid = np.arange(t.min(), t.max(), 0.5)
        yh, _ = fit['predict'](grid)
        # LOESS is meaningless inside long gaps: only search where sampled on both sides.
        sampled = np.array([coverage_ok(t, g, max_side_gap=20.0, max_extrap=0.0) for g in grid])
        yh[~sampled] = np.nan
        drop = np.interp(grid + 15, grid, yh, right=np.nan) - np.interp(grid - 15, grid, yh, left=np.nan)
        drop[~(sampled & np.interp(grid + 15, grid, sampled, right=0).astype(bool)
               & np.interp(grid - 15, grid, sampled, left=0).astype(bool))] = np.nan
        if not np.isfinite(drop).any() or np.nanmax(drop) < min_drop:
            continue
        t0 = grid[np.nanargmax(drop)]
        best = None
        for tpt0 in t0 + np.arange(-10, 11, 2.5):
            sel = (t > tpt0 - 60) & (t < tpt0 + 100)
            if sel.sum() < 8:
                continue
            p0 = [2.0, tpt0, 3.0, 0.01, np.interp(tpt0 + 20, t, y)]
            try:
                p, _ = curve_fit(_olivares, t[sel], y[sel], p0=p0, sigma=e[sel], maxfev=4000,
                                 bounds=([0.8, t0 - 25, 0.5, -0.02, 5], [6, t0 + 25, 12, 0.05, 30]))
            except (RuntimeError, ValueError):
                continue
            chi2 = np.sum(((y[sel] - _olivares(t[sel], *p)) / e[sel]) ** 2) / (sel.sum() - 5)
            # Real transition must be sampled on both sides (plateau before, drop/tail after),
            # with data close to t_PT so that it is not just placed somewhere inside a gap.
            near = 2 * p[2] + 10
            ok = ((t[sel] > p[1]).sum() >= 2) and ((t[sel] < p[1] - p[2]).sum() >= 3) \
                and ((t > p[1] - near) & (t < p[1])).any() and ((t > p[1]) & (t < p[1] + near)).any()
            if ok and (best is None or chi2 < best[0]):
                best = (chi2, p)
        if best is not None:
            return best[1][1], f'olivares_{band}', best[1]
    return np.nan, 'none', None
