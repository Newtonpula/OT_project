"""Replicate Fig. 2 of Ramirez et al. (2024): OT and linear interpolation between two black bodies.

    python replicate_paper_results/fig2_blackbody.py [--reg 1e-3] [--solvers paper sinkhorn_log exact] [--alpha T]

The 5000 K and 10000 K Planck spectra are interpolated as unit-sum shapes, with every OT solver in --solvers
(otrep/barycenter.py), linearly, and as a geometric mean. Each result is rescaled with the true total of the
black body it is compared with (methods.md, Sect. 6).

The paper does not say how alpha maps to temperature, so two mappings are evaluated:

    T     alpha = (T - T_a) / (T_b - T_a)            temperature plays the role of time (the paper's alpha)
    invT  alpha = (1/T - 1/T_a) / (1/T_b - 1/T_a)    the W2 geodesic of black bodies

Normalised black bodies are rescaled copies of one distribution in lambda * T, so their quantile functions scale as
1/T. Displacement interpolation therefore yields exact black bodies whose 1/T is linear in alpha.

Outputs (replicate_paper_results/outputs/):
    fig2_blackbody_<solver>_alpha-<map>_reg<reg>.png  the two panels of Fig. 2 (top: OT path, bottom: linear path)
    fig2_checks_reg<reg>[_<solvers>].png              Eq. 4 residual against temperature, for both mappings
    fig2_checks_reg<reg>[_<solvers>].csv              the numbers behind fig2_checks (50 K steps)
    (the solver suffix appears only when --solvers differs from the default)

Needs data/processed/blackbody_spectra_ot.h5 (python scripts/build_blackbody.py).
"""
import argparse
import csv
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from otrep import data, plotting  # noqa: E402
from otrep.barycenter import cost_matrix  # noqa: E402
from otrep.config import BASELINES, SOLVERS  # noqa: E402
from otrep.interpolate import alpha_between, interpolate, rescale  # noqa: E402
from otrep.metrics import eps  # noqa: E402

EPS_WINDOW = (3000.0, 10000.0)   # bins for Eq. 4: the SN grid, where every black body here is well above zero
FLUX_UNIT = 1e7                  # y unit of the paper's Fig. 2
MAPPINGS = {'T': 'α linear in T (temperature as time)', 'invT': 'α linear in 1/T (W2 geodesic of black bodies)'}
DEFAULT_SOLVERS = ['paper', 'sinkhorn_log', 'exact']


def alpha_for(T, T_a, T_b, mapping):
    T = np.asarray(T, float)
    return alpha_between(T, T_a, T_b) if mapping == 'T' else alpha_between(1 / T, 1 / T_a, 1 / T_b)


def interpolate_series(m, M, reg, solvers, backend, mapping):
    """Interpolations between the first and last spectra of series m, at the alpha of every spectrum, each rescaled
    with that spectrum's true total.

    Returns alpha, {series key: masses (N, L)} and {solver: (info, seconds)}.
    """
    alpha = alpha_for(m['t'], m['t'][0], m['t'][-1], mapping)
    rho_a, rho_b = m['rho'][0], m['rho'][-1]
    out, runs = {}, {}
    for b in BASELINES:
        out[b] = rescale(interpolate(rho_a, rho_b, alpha, method=b)[0], m['total'])
    for s in solvers:
        t0 = time.time()
        shapes, info = interpolate(rho_a, rho_b, alpha, method='ot', M=M, reg=reg, solver=s, edges=m['edges'],
                                   backend=backend)
        out[s], runs[s] = rescale(shapes, m['total']), (info, time.time() - t0)
    return alpha, out, runs


def plot_fig2(m, interp, solver, reg, mapping):
    """The two panels of the paper's Fig. 2, for one OT solver."""
    plotting.use_style()
    S = plotting.SERIES
    flux = lambda mass: mass / m['dlam'] / FLUX_UNIT  # noqa: E731
    fig, axes = plt.subplots(2, 1, figsize=(6.8, 6.6), sharex=True, sharey=True)
    reg_note = '' if solver in ('paper', 'exact') else f', reg = {reg:g}'
    panels = ((solver, f'Wasserstein interpolation path ({S[solver]["label"]}{reg_note})'),
              ('linear', 'L2 interpolation path (linear)'))
    for ax, (key, title) in zip(axes, panels):
        for k in range(len(m['t'])):
            ax.plot(m['wave'], flux(m['mass'][k]), color=S['truth']['color'], lw=1.0, zorder=2)
            ax.plot(m['wave'], flux(interp[key][k]), color=S[key]['color'], lw=1.6, ls='--', zorder=3)
        hot = flux(m['mass'][-1])
        j = int(np.argmax(hot))
        ax.annotate('10000 K', (m['wave'][j], hot[j]), xytext=(8, 0), textcoords='offset points',
                    color=plotting.SECONDARY, fontsize=9, va='center')
        cool = flux(m['mass'][0])
        j = int(np.argmax(cool))
        ax.annotate('5000 K', (m['wave'][j], cool[j]), xytext=(7800, 1.3), textcoords='data',
                    color=plotting.SECONDARY, fontsize=9,
                    arrowprops=dict(arrowstyle='-', color=plotting.MUTED, lw=0.8, shrinkA=2, shrinkB=1))
        handles = [Line2D([], [], color=S['truth']['color'], lw=1.0,
                          label=f'Black bodies, 5000–10000 K ({len(m["t"])} temperatures)'),
                   Line2D([], [], color=S[key]['color'], lw=1.6, ls='--', label=S[key]['label'])]
        ax.legend(handles=handles, loc='upper right')
        ax.set_title(title, loc='left')
        ax.set_ylabel(r'$B_\lambda$ [$10^7$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$ sr$^{-1}$]')
        ax.set_ylim(bottom=0)
    axes[-1].set_xlabel('Wavelength [Å]')
    axes[-1].set_xlim(m['wave'][0] - 2.5, m['wave'][-1] + 2.5)
    fig.suptitle(f'Black bodies between 5000 K and 10000 K; {MAPPINGS[mapping]}', x=0.01, ha='left',
                 fontsize=10, color=plotting.SECONDARY)
    return plotting.save(fig, f'fig2_blackbody_{solver}_alpha-{mapping}_reg{reg:.0e}.png')


def checks_name(reg, solvers, ext):
    suffix = '' if list(solvers) == DEFAULT_SOLVERS else '_' + '_'.join(solvers)
    return f'fig2_checks_reg{reg:.0e}{suffix}.{ext}'


def plot_checks(T, eps_by, reg, solvers):
    """Eq. 4 residual of every interpolation against temperature, one panel per alpha mapping."""
    plotting.use_style()
    S = plotting.SERIES
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), sharey=True)
    for ax, mapping in zip(axes, MAPPINGS):
        for key, e in eps_by[mapping].items():
            # the geometric mean coincides with the authors' settings: draw it as a wide halo underneath
            halo = dict(lw=4.5, alpha=0.45, ls='-', zorder=1) if key == 'geometric' else dict(ls=S[key]['ls'], zorder=2)
            ax.plot(T, 100 * e, color=S[key]['color'], label=S[key]['label'], **halo)
        ax.set_title(MAPPINGS[mapping], loc='left')
        ax.set_xlabel('Temperature [K]')
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel(f'ε (Eq. 4) over {EPS_WINDOW[0]:.0f}–{EPS_WINDOW[1]:.0f} Å [%]')
    axes[1].legend(loc='upper right')
    return plotting.save(fig, checks_name(reg, solvers, 'png'))


def main():
    p = argparse.ArgumentParser(description='Replicate Fig. 2 of Ramirez et al. (2024).')
    p.add_argument('--reg', type=float, default=1e-3,
                   help='entropic regularisation of the converged solvers (sinkhorn, sinkhorn_log)')
    p.add_argument('--solvers', nargs='+', default=DEFAULT_SOLVERS, choices=SOLVERS)
    p.add_argument('--alpha', default='T', choices=list(MAPPINGS), help='alpha mapping used for the Fig. 2 panels')
    p.add_argument('--backend', default='auto', choices=['auto', 'pot', 'numpy'])
    args = p.parse_args()

    paper, dense = data.load('bb', 'paper'), data.load('bb', 'dense')
    assert np.array_equal(paper['wave'], dense['wave'])
    M = cost_matrix(paper['wave'])

    _, fig_interp, _ = interpolate_series(paper, M, args.reg, args.solvers, args.backend, args.alpha)
    for s in args.solvers:
        print('wrote', plot_fig2(paper, fig_interp, s, args.reg, args.alpha))

    keys = BASELINES + list(args.solvers)
    win = (dense['wave'] >= EPS_WINDOW[0]) & (dense['wave'] <= EPS_WINDOW[1])
    alpha, eps_by, runs = {}, {}, {}
    for mapping in MAPPINGS:
        alpha[mapping], interp, runs[mapping] = interpolate_series(dense, M, args.reg, args.solvers, args.backend,
                                                                   mapping)
        eps_by[mapping] = {k: eps(dense['mass'], interp[k], mask=win) for k in keys}
    print('wrote', plot_checks(dense['t'], eps_by, args.reg, args.solvers))

    path = plotting.OUTPUT_DIR / checks_name(args.reg, args.solvers, 'csv')
    cols = [(mp, k) for mp in MAPPINGS for k in keys]
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['T_K'] + [f'alpha_{mp}' for mp in MAPPINGS] + [f'eps_{k}_alpha-{mp}' for mp, k in cols])
        for i in range(len(dense['t'])):
            w.writerow([f'{dense["t"][i]:.0f}'] + [f'{alpha[mp][i]:.4f}' for mp in MAPPINGS]
                       + [f'{eps_by[mp][k][i]:.5f}' for mp, k in cols])
    print('wrote', path)

    T = dense['t']
    inner = (T > T[0]) & (T < T[-1])
    mid = int(np.argmin(np.abs(T - 7500)))
    print(f'\nreg = {args.reg:g} (converged solvers); eps (Eq. 4) over {EPS_WINDOW[0]:.0f}-{EPS_WINDOW[1]:.0f} A, '
          '50 K steps')
    for mapping in MAPPINGS:
        print(f'  {MAPPINGS[mapping]}')
        print(f'  {"series":14s}{"5000 K":>9s}{"7500 K":>9s}{"10000 K":>9s}{"max inside":>12s}')
        for k in keys:
            e = 100 * eps_by[mapping][k]
            print(f'  {k:14s}{e[0]:8.2f}%{e[mid]:8.2f}%{e[-1]:8.2f}%{e[inner].max():11.2f}%')
    for mapping in MAPPINGS:
        for s, (info, sec) in runs[mapping].items():
            print(f'{s} (alpha-{mapping}): backend {info["backend"]}, {sec:.0f} s, '
                  f'iterations {info["n_iter"].min()}-{info["n_iter"].max()}')


if __name__ == '__main__':
    main()
