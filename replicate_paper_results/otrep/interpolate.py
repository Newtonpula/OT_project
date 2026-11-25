"""Interpolate between two spectra in time, with balanced OT (as in the paper) or linearly.

Spectra are unit-sum shapes rho, one per row, as returned by `otrep.data.load`. Both methods return unit-sum
shapes; `rescale()` turns them into masses with a prescribed total, which replaces the paper's K~ rescaling
(methods.md, Sect. 6).
"""
import numpy as np

from .barycenter import barycenters


def alpha_between(t, t_a, t_b):
    """Weight of the later spectrum, alpha = (t - t_a) / (t_b - t_a) (paper, Sect. 3)."""
    t, t_a, t_b = (np.asarray(x, float) for x in (t, t_a, t_b))
    return (t - t_a) / (t_b - t_a)


def interpolate(rho_a, rho_b, alpha, method='ot', M=None, reg=1e-3, solver='sinkhorn', edges=None, **kwargs):
    """Interpolate unit-sum shapes at weight alpha (alpha = 0 gives rho_a, alpha = 1 gives rho_b).

    rho_a, rho_b  (L,) or (m, L) shapes; a single pair is broadcast against an array of alphas
    alpha         scalar or (m,)
    method        'ot'         Wasserstein barycenter with weights [1 - alpha, alpha]; the solver picks how
                  'linear'     (1 - alpha) rho_a + alpha rho_b, the paper's L2 barycenter
                  'geometric'  rho_a**(1 - alpha) * rho_b**alpha, normalised: the pixel-wise weighted geometric mean
    solver        for 'ot': 'sinkhorn', 'sinkhorn_log', 'paper' or 'exact' (see barycenter.py)
    M, reg        cost matrix and regularisation of the entropic solvers
    edges         (L + 1,) bin edges, for solver='exact'
    kwargs        passed on to `barycenters()` (backend=..., stop_thr=..., num_iter_max=...)

    Returns (shapes, info): shapes is (m, L); info is the solver info for 'ot', and None otherwise.
    """
    alpha = np.atleast_1d(np.asarray(alpha, float))
    rho_a, rho_b = np.atleast_2d(rho_a), np.atleast_2d(rho_b)
    m, L = max(len(alpha), len(rho_a), len(rho_b)), rho_a.shape[1]
    alpha = np.broadcast_to(alpha, (m,))
    rho_a, rho_b = np.broadcast_to(rho_a, (m, L)), np.broadcast_to(rho_b, (m, L))
    if method == 'linear':
        return (1 - alpha)[:, None] * rho_a + alpha[:, None] * rho_b, None
    if method == 'geometric':
        g = rho_a ** (1 - alpha)[:, None] * rho_b ** alpha[:, None]
        return g / g.sum(axis=1, keepdims=True), None
    if method == 'ot':
        if M is None and solver != 'exact':
            raise ValueError('method="ot" needs the cost matrix M')
        A = np.stack([rho_a.T, rho_b.T], axis=-1)            # (L, m, 2): one problem per alpha
        W = np.stack([1 - alpha, alpha], axis=-1)            # (m, 2)
        bary, info = barycenters(A, M, reg, W, method=solver, edges=edges, **kwargs)
        return bary.T, info
    raise ValueError(f'unknown method {method!r}')


def run_methods(left, right, alpha, total, M, edges, reg, solvers, backend='auto', baselines=('linear', 'geometric')):
    """The non-OT baselines and every OT solver on stacked pairs, each result rescaled to `total`.

    left, right  (m, L) unit-sum shapes of the pairs; alpha (m,); total (m,) target totals
    Returns ({key: (m, L) masses}, {solver: info}).
    """
    out = {b: rescale(interpolate(left, right, alpha, method=b)[0], total) for b in baselines}
    infos = {}
    for s in solvers:
        shapes, infos[s] = interpolate(left, right, alpha, method='ot', M=M, reg=reg, solver=s, edges=edges,
                                       backend=backend)
        out[s] = rescale(shapes, total)
    return out, infos


def rescale(shapes, total):
    """Scale shapes (m, L) so that row i sums to total[i]."""
    shapes = np.atleast_2d(shapes)
    return shapes / shapes.sum(axis=1, keepdims=True) * np.asarray(total, float).reshape(-1, 1)


def combine(spectra, sig, taper=None):
    """Weighted mean of interpolations of one target (Eqs. 5-6, without the paper's MAD(K) term).

    spectra  (m, L) interpolated spectra of the same epoch
    sig      (m,) Sigma(eps) of each; the weights are 1 / sig^2
    taper    optional (m, L) edge weights C_i(lambda) of Eq. 6
    """
    spectra = np.atleast_2d(spectra)
    w = (1.0 / np.asarray(sig, float) ** 2)[:, None] * (1.0 if taper is None else taper)
    return (w * spectra).sum(axis=0) / np.broadcast_to(w, spectra.shape).sum(axis=0)
