"""Test 3 of Ramirez et al. (2024) on the observed SNe (Sect. 4.2): a leave-one-out check of OT and linear
interpolation, and the daily spectral time series of Fig. 6.

    python replicate_paper_results/fig6_observed.py [--solvers paper exact] [--sn SN2006cp SN2006be]

Only `mass` and `total` of the SN dataset are used (methods.md, Sect. 6). The paper's photometric validation (Figs. 7,
8, C.1, C.2) is not reproduced. Instead, two things are done.

Leave-one-out. Every interior OT-ready spectrum of every SN with at least three is withheld in turn and interpolated
from the others, in two ways:

  single    from the tightest pair of other spectra that brackets it (normally its two neighbours), as in Test 1;
  weighted  from every pair of the other spectra that brackets it and spans less than 40 d (Sect. 4.2). The pairs are
            averaged with weights 1 / Sigma(eps)^2 (Eq. 6 without MAD(K)).
            - Sigma(eps) is each method's Test 1 plane for the SN type, with the phase clipped to the model range.
            - The edge taper C(lambda) cancels, because every OT-ready spectrum covers the SN's common window.

Each interpolation is rescaled with the withheld spectrum's total and compared with it through Eq. 4 over the SN's
common window. The withheld spectrum is an observation, so eps includes its noise.

Fig. 6. Daily series (integer days between the first and last OT-ready spectra) of the --sn supernovae. They are built
from all OT-ready spectra in the weighted way, and drawn as normalised flux with offsets over the observed spectra.

Outputs (replicate_paper_results/outputs/):
    test3_loo_<suffix>.csv                              one row per withheld spectrum
    test3_loo_<single|weighted>_<solver>_<suffix>.png  eps against the gap of the tightest bracketing pair
    fig6_<SN>_<suffix>.png                              Fig. 6 for each --sn supernova

Needs data/processed/sn_spectra_ot.h5, and sigma_plane_<suffix>.json written by fig3_model_pairs.py.
"""
import argparse
import csv
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from otrep import data, plotting, sigma  # noqa: E402
from otrep.barycenter import cost_matrix  # noqa: E402
from otrep.config import BASELINES, MODEL_PHASES, OBS_PHASES, SOLVERS, TEST3_MAX_GAP, suffix  # noqa: E402
from otrep.interpolate import combine, run_methods  # noqa: E402
from otrep.metrics import eps  # noqa: E402
from otrep.pairs import bracketing  # noqa: E402


def solve(m, problems, total, args):
    """Interpolate the stacked problems (i, j, t0) of one SN with every method; row n is rescaled to total[n]."""
    t = m['t']
    i, j, t0 = (np.array([p[c] for p in problems]) for c in range(3))
    alpha = (t0 - t[i]) / (t[j] - t[i])
    out, _ = run_methods(m['rho'][i], m['rho'][j], alpha, total, cost_matrix(m['wave']), m['edges'], args.reg,
                         args.solvers, args.backend, baselines=BASELINES)
    return out


def weighted(interp, rows, gaps, phase, sn_type, planes):
    """Eq. 6 without MAD(K): the interpolations `rows` of one epoch, averaged with weights 1 / Sigma(eps)^2."""
    ph = np.clip(phase, *MODEL_PHASES[sn_type])
    return {k: combine(f[rows], sigma.sigma(planes[k][sn_type], gaps, ph)) for k, f in interp.items()}


def collect(m, epochs, max_gap, exclude_self):
    """Problems for interpolating m at each epoch: the bracketing pairs (excluding the epoch's own spectrum)."""
    t = m['t']
    problems, slots = [], []
    for k, t0 in epochs:
        pairs = bracketing(t, t0, max_gap, exclude=(k,) if exclude_self else ())
        if not pairs:
            continue
        rows = np.arange(len(problems), len(problems) + len(pairs))
        problems += [(i, j, t0) for i, j in pairs]
        gaps = np.array([t[j] - t[i] for i, j in pairs])
        # the tightest bracketing pair: normally the two neighbours (not when two spectra share an epoch)
        slots.append(dict(k=k, t0=t0, rows=rows, gaps=gaps, single=rows[int(np.argmin(gaps))]))
    return problems, slots


def leave_one_out(name, sn_type, planes, args):
    """One result row per withheld interior spectrum of SN `name`."""
    m = data.load('sn', name)
    t = m['t']
    problems, slots = collect(m, [(k, t[k]) for k in range(1, len(t) - 1)], args.max_gap, exclude_self=True)
    if not slots:
        return []
    interp = solve(m, problems, np.concatenate([[m['total'][s['k']]] * len(s['rows']) for s in slots]), args)
    out = []
    for s in slots:
        k, truth = s['k'], m['mass'][s['k']]
        comb = weighted(interp, s['rows'], s['gaps'], t[k], sn_type, planes)
        row = dict(sn=name, type=sn_type, phase=t[k], n_pairs=len(s['rows']), gap_single=s['gaps'].min(),
                   gap_max=s['gaps'].max())
        for key in interp:
            row[f'eps_single_{key}'] = eps(truth, interp[key][s['single']])
            row[f'eps_weighted_{key}'] = eps(truth, comb[key])
        out.append(row)
    return out


def daily_series(name, sn_type, planes, args):
    """Unit-sum spectra of SN `name` at every integer day between its first and last OT-ready spectra."""
    m = data.load('sn', name)
    t = m['t']
    days = np.arange(np.ceil(t[0]), np.floor(t[-1]) + 1)
    problems, slots = collect(m, [(-1, d) for d in days if not np.isclose(d, t).any()], args.max_gap, False)
    interp = solve(m, problems, np.ones(len(problems)), args)
    series = {k: np.array([weighted(interp, s['rows'], s['gaps'], s['t0'], sn_type, planes)[k] for s in slots])
              for k in interp}
    return m, np.array([s['t0'] for s in slots]), series


def plot_fig6(name, m, days, series, keys, sfx, step=0.5):
    """Daily series of each method (one column each) over the observed spectra, as normalised flux with offsets."""
    plotting.use_style()
    S = plotting.SERIES
    t = m['t']
    norm = lambda f: f / f.mean()  # noqa: E731
    fig, axes = plt.subplots(1, len(keys), figsize=(3.9 * len(keys), 2.2 + 0.13 * (t[-1] - t[0])), sharey=True)
    for ax, key in zip(axes, keys):
        for d, f in zip(days, series[key]):
            ax.plot(m['wave'], norm(f) - step * (d - t[0]), color=S[key]['color'], lw=0.8, zorder=2)
        for tk, f in zip(t, m['mass']):
            ax.plot(m['wave'], norm(f) - step * (tk - t[0]), color=S['truth']['color'], lw=1.1, zorder=3)
        ax.set_title(S[key]['label'], loc='left', fontsize=9.5)
        ax.set_xlabel('Rest wavelength [Å]')
        ax.grid(False)
    last = None
    for tk, f in zip(t, m['mass']):
        if last is not None and tk - last < 0.6:              # spectra of the same night share one label
            continue
        n_same = int(np.sum(np.abs(t - tk) < 0.6))
        axes[-1].annotate(f'{tk:.1f} d' + (f' ×{n_same}' if n_same > 1 else ''),
                          (m['wave'][-1], norm(f)[-1] - step * (tk - t[0])), xytext=(4, 0),
                          textcoords='offset points', fontsize=7.5, va='center', color=plotting.SECONDARY)
        last = tk
    axes[0].set_ylabel('Normalised flux + offset')
    axes[0].set_yticks([])
    handles = [Line2D([], [], color=S['truth']['color'], lw=1.1, label='observed (OT-ready)'),
               Line2D([], [], color=plotting.MUTED, lw=0.8, label='interpolated, one per day (colour = method)')]
    fig.legend(handles=handles, loc='upper right', ncol=2, fontsize=8.5, bbox_to_anchor=(0.99, 1.0))
    fig.suptitle(f'{name} (SN {data.info("sn", name)["type"]}): daily series from the {len(t)} OT-ready spectra; '
                 f'pairs < {TEST3_MAX_GAP:g} d, weights 1/Σ(ε)²', x=0.01, y=1.0, ha='left', fontsize=10,
                 color=plotting.SECONDARY)
    return plotting.save(fig, f'fig6_{name}_{sfx}.png')


def summarize(rows, keys):
    for sn_type in ('Ia', 'II'):
        sel = [r for r in rows if r['type'] == sn_type]
        if not sel:
            continue
        gap = np.array([r['gap_single'] for r in sel])
        print(f'\nSN {sn_type}: {len(sel)} withheld spectra from {len({r["sn"] for r in sel})} SNe; gap of the '
              f'tightest bracketing pair: median {np.median(gap):.1f} d')
        print(f'  {"method":14s}{"single: median":>16s}{"< 10%":>8s}{"weighted: median":>19s}{"< 10%":>8s}'
              f'{"beats linear (single)":>24s}')
        lin = np.array([r['eps_single_linear'] for r in sel])
        for k in keys:
            es = np.array([r[f'eps_single_{k}'] for r in sel])
            ew = np.array([r[f'eps_weighted_{k}'] for r in sel])
            beats = '' if k == 'linear' else f'{100 * np.mean(es < lin):23.0f}%'
            print(f'  {k:14s}{100 * np.median(es):15.1f}%{100 * np.mean(es < 0.1):7.0f}%'
                  f'{100 * np.median(ew):18.1f}%{100 * np.mean(ew < 0.1):7.0f}%{beats}')


def main():
    p = argparse.ArgumentParser(description='Test 3 of Ramirez et al. (2024) on the observed SNe.')
    p.add_argument('--reg', type=float, default=1e-3, help='regularisation of the converged entropic solvers')
    p.add_argument('--solvers', nargs='+', default=['paper', 'exact'], choices=SOLVERS)
    p.add_argument('--max-gap', type=float, default=TEST3_MAX_GAP, help='pairs must span less than this [d]')
    p.add_argument('--sn', nargs='*', default=['SN2006cp', 'SN2006be'], help='supernovae drawn as Fig. 6')
    p.add_argument('--limit', type=int, default=None, help='only the first N supernovae (for quick tests)')
    p.add_argument('--backend', default='auto', choices=['auto', 'pot', 'numpy'])
    args = p.parse_args()
    sfx = suffix(args.reg)
    planes = sigma.load(plotting.OUTPUT_DIR / f'sigma_plane_{sfx}.json')['planes']
    keys = BASELINES + list(args.solvers)

    t0 = time.time()
    names = data.names('sn', min_ot_ready=3)[:args.limit]
    rows = []
    for name in names:
        rows += leave_one_out(name, data.info('sn', name)['type'], planes, args)
    print(f'leave-one-out: {len(rows)} withheld spectra from {len({r["sn"] for r in rows})} SNe '
          f'in {time.time() - t0:.0f} s')
    tag = '' if args.limit is None else f'_first{args.limit}'
    with open(plotting.OUTPUT_DIR / f'test3_loo_{sfx}{tag}.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows({k: (f'{v:.5f}' if isinstance(v, float) else v) for k, v in r.items()} for r in rows)
    types = np.array([r['type'] for r in rows])
    gap = np.array([r['gap_single'] for r in rows])
    ph = np.array([r['phase'] for r in rows])
    for way in ('single', 'weighted'):
        eps_by = {k: np.array([r[f'eps_{way}_{k}'] for r in rows]) for k in keys}
        for s in args.solvers:
            source = 'the tightest bracketing pair' if way == 'single' else 'all bracketing pairs < 40 d, weighted'
            title = f'Observed SNe, each withheld spectrum from {source}: {plotting.SERIES[s]["label"]}'
            path = plotting.gap_scatter(types, gap, ph, eps_by, ['linear', s], OBS_PHASES, title,
                                        'Gap of the tightest bracketing pair [d]', f'test3_loo_{way}_{s}_{sfx}{tag}.png',
                                        ylabel='ε (Eq. 4) over the SN window [%]',
                                        panel='SN {type} ({n} withheld spectra)')
            print('wrote', path)
    summarize(rows, keys)

    for name in args.sn:
        m, days, series = daily_series(name, data.info('sn', name)['type'], planes, args)
        print('wrote', plot_fig6(name, m, days, series, keys, sfx))


if __name__ == '__main__':
    main()
