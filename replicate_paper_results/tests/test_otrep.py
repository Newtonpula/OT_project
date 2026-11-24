"""Checks of the shared otrep package.

    python replicate_paper_results/tests/test_otrep.py      (or: pytest replicate_paper_results/tests)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from otrep import sigma  # noqa: E402
from otrep.barycenter import HAVE_POT, barycenters, cost_matrix  # noqa: E402
from otrep.interpolate import alpha_between, combine, interpolate, rescale  # noqa: E402
from otrep.metrics import eps  # noqa: E402
from otrep.pairs import triplets, windows  # noqa: E402


def gauss(x, mu, sigma):
    g = np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    return g / g.sum()


X = np.arange(100.0)
A_PAIR = np.c_[gauss(X, 20, 5), gauss(X, 60, 8)]      # the two distributions of POT's 1D barycenter demo
M_TOY = cost_matrix(X)


def test_cost_matrix():
    n = len(X)
    i = np.arange(n)
    assert np.allclose(M_TOY, ((i[:, None] - i[None, :]) / (n - 1)) ** 2)
    assert np.allclose(cost_matrix(5.0 * X + 3000), M_TOY)          # any uniform grid gives the same matrix


def test_alpha():
    assert np.allclose(alpha_between([3, 4, 5], 2, 6), [0.25, 0.5, 0.75])   # the paper's example


def test_linear_and_rescale():
    shapes, info = interpolate(A_PAIR[:, 0], A_PAIR[:, 1], [0.0, 0.3, 1.0], method='linear')
    assert info is None
    assert np.allclose(shapes[0], A_PAIR[:, 0]) and np.allclose(shapes[2], A_PAIR[:, 1])
    assert np.allclose(shapes.sum(1), 1)
    assert np.allclose(rescale(shapes, [1, 2, 3]).sum(1), [1, 2, 3])


def test_ot_barycenter_mass_and_location():
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    shapes, info = interpolate(A_PAIR[:, 0], A_PAIR[:, 1], alphas, method='ot', M=M_TOY, reg=1e-3,
                               solver='sinkhorn_log', backend='numpy')
    assert np.allclose(shapes.sum(1), 1, atol=1e-6)
    assert np.allclose(shapes @ X, [20, 30, 40, 50, 60], atol=0.05)    # means follow displacement interpolation
    assert np.all(info['n_iter'] < 10000)


def test_pot_default_pulls_toward_middle():
    """POT's default solver starts from an unweighted geometric mean, which biases unequal weights."""
    kw = dict(method='ot', M=M_TOY, reg=1e-3, backend='numpy')
    s_def, _ = interpolate(A_PAIR[:, 0], A_PAIR[:, 1], [0.0, 0.5], solver='sinkhorn', **kw)
    s_log, _ = interpolate(A_PAIR[:, 0], A_PAIR[:, 1], [0.0, 0.5], solver='sinkhorn_log', **kw)
    assert abs((s_def[1] - s_log[1]) @ X) < 0.01                        # equal weights: same barycenter
    assert (s_def[0] @ X) - 20 > 1.0                                    # alpha = 0: pulled toward the middle


def test_batched_equals_single():
    alphas = [0.2, 0.5, 0.9]
    together, _ = interpolate(A_PAIR[:, 0], A_PAIR[:, 1], alphas, method='ot', M=M_TOY, reg=1e-2, backend='numpy')
    for a, row in zip(alphas, together):
        alone, _ = interpolate(A_PAIR[:, 0], A_PAIR[:, 1], a, method='ot', M=M_TOY, reg=1e-2, backend='numpy')
        assert np.allclose(alone[0], row, rtol=1e-10, atol=1e-14)


def test_log_domain_agrees():
    W = np.array([[0.5, 0.5]])
    b1, _ = barycenters(A_PAIR[:, None, :], M_TOY, 1e-2, W, method='sinkhorn', backend='numpy', stop_thr=1e-10)
    b2, _ = barycenters(A_PAIR[:, None, :], M_TOY, 1e-2, W, method='sinkhorn_log', backend='numpy', stop_thr=1e-10)
    assert np.abs(b1 - b2).sum() < 1e-6


def test_exact_w2():
    edges = np.arange(101.0) - 0.5
    shapes, _ = interpolate(A_PAIR[:, 0], A_PAIR[:, 1], [0.0, 0.5, 1.0], method='ot', solver='exact', edges=edges)
    assert np.allclose(shapes.sum(1), 1)
    assert np.allclose(shapes[0], A_PAIR[:, 0], atol=1e-12) and np.allclose(shapes[2], A_PAIR[:, 1], atol=1e-12)
    mean = shapes[1] @ X
    std = np.sqrt(shapes[1] @ X ** 2 - mean ** 2)
    assert abs(mean - 40) < 0.01 and abs(std - 6.5) < 0.05          # W2 geodesic of Gaussians: mean and std linear


def test_paper_settings_give_geometric_mean():
    soft = (A_PAIR + 1e-3) / (A_PAIR + 1e-3).sum(0)                   # spectra have no near-zero tails
    alphas = [0.3, 0.5]
    shapes, info = interpolate(soft[:, 0], soft[:, 1], alphas, method='ot', M=M_TOY, solver='paper', backend='numpy')
    geo, _ = interpolate(soft[:, 0], soft[:, 1], alphas, method='geometric')
    assert np.allclose(info['reg'], 5e-7)                              # first reg of the authors' scan
    assert np.abs(shapes / shapes.sum(1, keepdims=True) - geo).sum(1).max() < 1e-6


def test_pot_agrees():
    if not HAVE_POT:
        print('  (POT not installed: skipped)')
        return
    W = np.array([[0.3, 0.7]])
    b_np, _ = barycenters(A_PAIR[:, None, :], M_TOY, 1e-3, W, backend='numpy')
    b_pot, _ = barycenters(A_PAIR[:, None, :], M_TOY, 1e-3, W, backend='pot')
    assert np.allclose(b_np, b_pot, rtol=1e-8, atol=1e-12)


def test_triplets():
    idx, alpha = triplets([0.0, 1.0, 2.0, 3.0, 4.0], (0.45, 0.55))
    assert idx.tolist() == [[0, 1, 2], [0, 2, 4], [1, 2, 3], [2, 3, 4]]
    assert np.allclose(alpha, 0.5)
    idx, _ = triplets([0.0, 1.0, 2.0, 3.0, 4.0], (0.45, 0.55), max_gap=2.5)
    assert [0, 2, 4] not in idx.tolist()


def test_windows_match_fig4():
    t = np.arange(6.0)
    got = [(c['target'], c['pairs']) for c in windows(t[:5])]
    assert got == [(1, [(0, 2), (0, 3), (0, 4)]),                      # cycle 1
                   (2, [(0, 3), (0, 4), (1, 3), (1, 4)]),              # cycle 2
                   (3, [(0, 4), (1, 4), (2, 4)])]                      # cycle 3
    cycle4 = [(c['target'], c['pairs']) for c in windows(t) if c['W'] == 6 and c['p'] == 1]
    assert cycle4 == [(1, [(0, 3), (0, 4), (0, 5)]), (2, [(0, 3), (0, 4), (0, 5)])]
    assert all(t[c['pairs'][-1][1]] - t[c['pairs'][0][0]] <= 4 for c in windows(t, max_gap=4))


def test_sigma_plane_and_combine():
    rng = np.random.default_rng(1)
    dph, ph = rng.uniform(0, 40, 50), rng.uniform(-5, 30, 50)
    plane = sigma.fit_plane(dph, ph, 0.002 * dph + 0.001 * ph + 0.01)
    assert np.allclose([plane['a'], plane['b'], plane['c'], plane['r2']], [0.002, 0.001, 0.01, 1.0])
    assert sigma.sigma(plane, 0.0, -100.0) == sigma.FLOOR                # clipped where the plane drops below zero
    assert np.allclose(combine([[1.0, 1.0], [3.0, 3.0]], [1.0, 1.0]), 2.0)
    assert np.allclose(combine([[1.0, 1.0], [3.0, 3.0]], [1.0, 2.0]), 1.4)   # weights 1 and 1/4


def test_eps():
    f = np.array([1.0, 2.0, 4.0, 0.0])
    assert eps(f, f) == 0
    assert np.isclose(eps(f, 1.1 * f), 0.1)                            # the zero bin is skipped
    assert np.isclose(eps(f, 1.1 * f, mask=[True, False, False, False]), 0.1)


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print(f'ok  {name}')
