"""Entropic Wasserstein barycenters, as computed in the paper with POT's `ot.bregman.barycenter`.

If POT is installed, it is used directly. Otherwise a NumPy port of the same algorithm runs instead: iterative
Bregman projections (Benamou et al. 2015), following POT's code line by line. The port solves many independent
barycenter problems at once, which is what the interpolation tests need (one problem per alpha or per pair of
spectra).

Methods:

    'sinkhorn'      POT's default solver, run to convergence. Its starting scalings use the *unweighted* geometric
                    mean of the distributions, and the iterations keep a trace of that start. For alpha != 0.5
                    the barycenter is pulled toward the alpha = 0.5 one; at alpha = 0 it does not reduce to the
                    first spectrum.
    'sinkhorn_log'  starts from v = 1 and converges to the entropic barycenter of the weighted problem.
    'paper'         the authors' settings, from their code (github.com/mramirezz/STS_OT, Codes/Funciones_colab.ipynb,
                    best_reg): POT's default solver with numItermax = 10, at the first reg of
                    5 * 10**np.arange(-7, -1, 0.1) whose result is finite. `reg` is ignored. With such a narrow kernel
                    and so few iterations almost no mass moves, and the result is close to the pixel-wise weighted
                    geometric mean of the inputs.
    'exact'         the exact (unregularised) W2 barycenter of 1D histograms, i.e. the weighted mean of their
                    quantile functions; needs the bin edges. `M` and `reg` are ignored.
"""
import warnings

import numpy as np

try:
    import ot as _pot
except ImportError:  # POT not installed: the NumPy port is used
    _pot = None

HAVE_POT = _pot is not None

# Defaults of POT's ot.bregman.barycenter
NUM_ITER_MAX = 10000
STOP_THR = 1e-4

# The authors' settings (STS_OT, Funciones_colab.ipynb, best_reg)
PAPER_REGS = 5 * 10 ** np.arange(-7, -1, 0.1)
PAPER_NUM_ITER = 10


def cost_matrix(wave):
    """Squared distance between bin centres, divided by its maximum.

    This is POT's `ot.dist(x, x)` followed by `M /= M.max()`; on a uniform grid it equals
    `ot.utils.dist0(n)` divided by its maximum.
    """
    x = np.asarray(wave, float)
    M = (x[:, None] - x[None, :]) ** 2
    return M / M.max()


def barycenters(A, M, reg, weights, method='sinkhorn', backend='auto',
                num_iter_max=NUM_ITER_MAX, stop_thr=STOP_THR, edges=None):
    """Solve G independent barycenter problems.

    A        (n, G, k) histograms: k distributions on n bins for each of the G problems; each sums to 1
    M        (n, n) cost matrix (not used by 'exact')
    reg      entropic regularisation (not used by 'paper' and 'exact')
    weights  (G, k) barycenter weights, each row summing to 1
    method   'sinkhorn', 'sinkhorn_log', 'paper' or 'exact' (see the module docstring)
    backend  'auto' (POT if installed), 'pot' or 'numpy'
    edges    (n + 1,) bin edges, for 'exact'

    Returns (bary, info): bary is (n, G); info holds n_iter (G,), err (G,), the backend used and, for 'paper',
    the reg chosen for each problem.
    """
    A = np.asarray(A, float)
    W = np.asarray(weights, float)
    if method == 'exact':
        if edges is None:
            raise ValueError('method="exact" needs the bin edges')
        return _exact_w2(A, np.asarray(edges, float), W), dict(n_iter=np.zeros(A.shape[1], int),
                                                               err=np.zeros(A.shape[1]), backend='numpy')
    if backend == 'auto':
        backend = 'pot' if HAVE_POT else 'numpy'
    if method == 'paper':
        bary, info = _paper(A, M, W, backend)
        info['backend'] = backend
        return bary, info
    if backend == 'pot':
        if not HAVE_POT:
            raise ImportError('POT is not installed (pip install pot)')
        out, n_iter, err = [], [], []
        for g in range(A.shape[1]):
            b, log = _pot.bregman.barycenter(A[:, g], M, reg, W[g], method=method, numItermax=num_iter_max,
                                             stopThr=stop_thr, log=True)
            out.append(b)
            n_iter.append(log.get('niter', -1))
            err.append(log['err'][-1] if log.get('err') else np.nan)
        bary, info = np.stack(out, axis=1), dict(n_iter=np.array(n_iter), err=np.array(err))
    elif backend == 'numpy':
        solver = {'sinkhorn': _ibp, 'sinkhorn_log': _ibp_log}[method]
        bary, info = solver(A, M, reg, W, num_iter_max, stop_thr)
    else:
        raise ValueError(f'unknown backend {backend!r}')
    info['backend'] = backend
    if not np.all(np.isfinite(bary)):
        raise FloatingPointError(f'non-finite barycenter (reg={reg}); try method="sinkhorn_log" or a larger reg')
    return bary, info


def _paper(A, M, W, backend):
    """The authors' best_reg: for every problem, the first reg of PAPER_REGS whose 10-iteration result is finite.

    If no reg works, the last result is kept, as their code does.
    """
    n, G, _ = A.shape
    bary = np.full((n, G), np.nan)
    reg_used = np.full(G, np.nan)
    todo = np.arange(G)
    for reg in PAPER_REGS:
        with warnings.catch_warnings(), np.errstate(all='ignore'):
            warnings.simplefilter('ignore')
            if backend == 'pot':
                b = np.stack([_pot.bregman.barycenter(A[:, g], M, reg, W[g], method='sinkhorn',
                                                      numItermax=PAPER_NUM_ITER) for g in todo], axis=1)
            else:
                b, _ = _ibp(A[:, todo], M, reg, W[todo], PAPER_NUM_ITER, STOP_THR)
        ok = np.all(np.isfinite(b), axis=0)
        bary[:, todo[ok]], reg_used[todo[ok]] = b[:, ok], reg
        todo = todo[~ok]
        if not len(todo):
            break
    if len(todo):
        bary[:, todo] = b[:, ~ok]
    return bary, dict(n_iter=np.full(G, PAPER_NUM_ITER), err=np.full(G, np.nan), reg=reg_used)


def _exact_w2(A, edges, W):
    """Exact W2 barycenters of 1D histograms (mass spread uniformly within each bin).

    The quantile functions are piecewise linear, with breaks at the cumulative masses; their weighted mean is
    exact on the union of those breaks.
    """
    n, G, k = A.shape
    out = np.empty((n, G))
    for g in range(G):
        cdfs = []
        for j in range(k):
            c = np.r_[0.0, np.cumsum(A[:, g, j])]
            cdfs.append(c / c[-1])
        p = np.unique(np.concatenate(cdfs))
        Q = sum(W[g, j] * np.interp(p, cdfs[j], edges) for j in range(k))
        out[:, g] = np.diff(np.interp(edges, Q, p))
    return out


class _Active:
    """Book-keeping for the problems that are still iterating; each one stops on its own."""

    def __init__(self, n, G, num_iter_max):
        self.idx = np.arange(G)
        self.bary = np.full((n, G), np.nan)
        self.n_iter = np.full(G, num_iter_max, int)
        self.err = np.full(G, np.nan)

    def finish(self, done, bary, ii, err):
        """Store the problems in `done` (a mask over the active ones); return the mask of those to keep."""
        g = self.idx[done]
        self.bary[:, g], self.n_iter[g], self.err[g] = bary[:, done], ii, err[done]
        self.idx = self.idx[~done]
        return ~done

    def result(self, bary_last, err_last, num_iter_max):
        if len(self.idx):
            warnings.warn(f'{len(self.idx)} barycenter(s) did not converge in {num_iter_max} iterations')
            self.bary[:, self.idx], self.err[self.idx] = bary_last, err_last
        return self.bary, dict(n_iter=self.n_iter, err=self.err)


def _ibp(A, M, reg, W, num_iter_max, stop_thr):
    """POT's barycenter_sinkhorn, vectorised over G problems.

    As in POT, a problem is checked at every iteration with ii % 10 == 1 and stops there once its error,
    sum_i std_k(UKv), is below stop_thr.
    """
    n, G, k = A.shape
    K = np.exp(-M / reg)
    act = _Active(n, G, num_iter_max)
    UKv = (K @ (A.reshape(n, -1) / K.sum(axis=0)[:, None])).reshape(n, G, k)
    u = np.exp(np.log(UKv).mean(axis=-1, keepdims=True)) / UKv
    gbar, err = None, np.full(G, np.nan)
    for ii in range(num_iter_max):
        u2 = u.reshape(n, -1)
        UKv = (u2 * (K.T @ (A.reshape(n, -1) / (K @ u2)))).reshape(u.shape)
        gbar = np.exp(np.einsum('ngk,gk->ng', np.log(UKv), W))
        u = u * gbar[..., None] / UKv
        if ii % 10 == 1:
            err = UKv.std(axis=-1).sum(axis=0)
            done = err < stop_thr
            if done.any():
                keep = act.finish(done, gbar, ii, err)
                A, W, u, gbar, err = A[:, keep], W[keep], u[:, keep], gbar[:, keep], err[keep]
                if not len(act.idx):
                    break
    return act.result(gbar, err, num_iter_max)


def _log_matmul(K, logx):
    """log(K @ exp(logx)), column by column, with each column shifted by its maximum before exp."""
    mx = logx.max(axis=0, keepdims=True)
    mx = np.where(np.isfinite(mx), mx, 0.0)
    return np.log(np.maximum(K @ np.exp(logx - mx), 1e-300)) + mx


def _ibp_log(A, M, reg, W, num_iter_max, stop_thr):
    """POT's _barycenter_sinkhorn_log (method='sinkhorn_log'), vectorised over G problems.

    The scalings are stored as logs, so that a small reg does not overflow, and they start from v = 1.
    Where POT takes an exact logsumexp over the cost matrix, this uses a matrix product with a per-column
    shift; the two differ only where exp(-M / reg) underflows.
    """
    n, G, k = A.shape
    K = np.exp(-M / reg)
    act = _Active(n, G, num_iter_max)
    logA = np.log(A + 1e-16)
    logv = np.zeros_like(A)
    logb, err = None, np.full(G, np.nan)
    for ii in range(num_iter_max):
        logu = logA - _log_matmul(K, logv.reshape(n, -1)).reshape(logv.shape)   # u = a / (K v)
        logKtu = _log_matmul(K.T, logu.reshape(n, -1)).reshape(logv.shape)
        logm = logv + logKtu                                                       # second marginals
        logb = np.einsum('ngk,gk->ng', logm, W)
        logv = logb[..., None] - logKtu                                            # v = b / (K^T u)
        if ii % 10 == 1:
            err = np.exp(logm).std(axis=-1).sum(axis=0)
            done = err < stop_thr
            if done.any():
                keep = act.finish(done, np.exp(logb), ii, err)
                logA, W, logv, logb, err = logA[:, keep], W[keep], logv[:, keep], logb[:, keep], err[keep]
                if not len(act.idx):
                    break
    return act.result(np.exp(logb), err, num_iter_max)
