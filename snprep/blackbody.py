"""Planck black-body spectra for the paper's Fig. 2 test (Ramirez et al. 2024, Sect. 3).

B_lambda is the spectral radiance in erg s^-1 cm^-2 A^-1 sr^-1, the unit of the paper's Fig. 2 y axis.
Wavelengths are in A, temperatures in K; functions broadcast as (T..., wave...).
"""
import numpy as np

H = 6.62607015e-27            # erg s     (exact, SI 2019)
C = 2.99792458e10             # cm s^-1   (exact)
K_B = 1.380649e-16            # erg K^-1  (exact)
C2 = H * C / K_B * 1e8        # second radiation constant hc/k [A K]
SIGMA = 2 * np.pi ** 5 * K_B ** 4 / (15 * H ** 3 * C ** 2)   # Stefan-Boltzmann [erg s^-1 cm^-2 K^-4]
X_WIEN = 4.965114231744276    # root of x = 5 (1 - e^-x): lambda_peak = C2 / (X_WIEN T)

# paper Fig. 2: 5000-10000 K on 1000-15000 A; the 10 black curves are np.linspace(5000, 10000, 10)
# (read off their peak heights, 4.1, 3.1, 2.3, 1.6, ... x 1e7; 500 K steps would give 3.2, 2.4, 1.8, ...)
T_MIN, T_MAX = 5000.0, 10000.0
PAPER_T = np.linspace(T_MIN, T_MAX, 10)
GRID_EDGES = np.arange(1000.0, 15000.0 + 1, 5.0)   # 5 A bins; edges line up with the SN grid (3000-10000 A)
GRID = 0.5 * (GRID_EDGES[:-1] + GRID_EDGES[1:])


def planck(wave, T):
    """B_lambda(T) at `wave`; shape T.shape + wave.shape."""
    lam = np.asarray(wave, float) * 1e-8                        # cm
    x = C2 / np.multiply.outer(np.asarray(T, float), np.asarray(wave, float))
    return 2 * H * C ** 2 / lam ** 5 / np.expm1(x) * 1e-8       # per cm -> per A


def planck_bin_mean(edges, T, n_gl=8):
    """Mean of B_lambda over each bin [edges[i], edges[i+1]] (Gauss-Legendre, exact to rounding for 5 A bins)."""
    u, w = np.polynomial.legendre.leggauss(n_gl)
    lo, hi = edges[:-1, None], edges[1:, None]
    nodes = 0.5 * (lo + hi) + 0.5 * (hi - lo) * u              # (L, n_gl)
    return planck(nodes, T) @ w / 2


def _tail(x, n_terms=200):
    """int_x^inf u^3 / (e^u - 1) du, series in e^-nx (converges for x > 0; x >= 0.96 on the grid)."""
    n = np.arange(1, n_terms + 1)
    x = np.asarray(x, float)[..., None]
    return (np.exp(-n * x) * (x ** 3 / n + 3 * x ** 2 / n ** 2 + 6 * x / n ** 3 + 6 / n ** 4)).sum(-1)


def planck_band(lo, hi, T):
    """Analytic int_lo^hi B_lambda dlambda [erg s^-1 cm^-2 sr^-1] (independent of the quadrature above)."""
    T = np.asarray(T, float)
    return planck_total(T) * 15 / np.pi ** 4 * (_tail(C2 / (hi * T)) - _tail(C2 / (lo * T)))


def planck_total(T):
    """int_0^inf B_lambda dlambda = sigma T^4 / pi [erg s^-1 cm^-2 sr^-1]."""
    return SIGMA * np.asarray(T, float) ** 4 / np.pi


def peak_wave(T):
    """Wien peak of B_lambda [A]."""
    return C2 / (X_WIEN * np.asarray(T, float))
