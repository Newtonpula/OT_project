"""Replicate Fig. 5 of Ramirez et al. (2024), Test 2: weighted mean over all pairs of four model spectra.

    python replicate_paper_results/fig5_model_windows.py [--solvers paper exact] [--rebin 1] [--reg 1e-3]

The test runs over every model, every window of Fig. 4 (otrep.pairs.windows) and every withheld target:

1. every input pair that brackets the target is interpolated to the target's phase, with every OT solver in
   --solvers, linearly, and as a geometric mean, and rescaled with the true total of the target;
2. the interpolations are averaged with weights 1 / Sigma(eps)^2. This is Eq. 5 without MAD(K). Each method uses
   its own Test 1 plane, so linear interpolation goes through the same scheme as OT;
3. the mean is compared with the true target spectrum through Eq. 4.

Each (input pair, target) interpolation is computed once and shared between the windows that use it.

Outputs (replicate_paper_results/outputs/):
    fig5_models_<solver>_<suffix>.png   Fig. 5: eps against the largest phase gap among the pairs in the mean
    fig5_models_<suffix>.csv            one row per window and target

Needs sigma_plane_<suffix>.json, written by fig3_model_pairs.py with the same --reg and --rebin, and holding
every series used here.
"""
import argparse
import csv

import matplotlib
matplotlib.use('Agg')
import numpy as np  # noqa: E402

from fig3_model_pairs import run_series, summarize  # noqa: E402
from otrep import data, plotting, sigma  # noqa: E402
from otrep.barycenter import cost_matrix  # noqa: E402
from otrep.config import MAX_GAP, MODEL_PHASES, MODEL_WINDOW, SOLVERS, suffix  # noqa: E402
from otrep.interpolate import combine  # noqa: E402
from otrep.metrics import eps  # noqa: E402
from otrep.pairs import windows  # noqa: E402


def collect(rebin, max_gap):
    """Unique (pair, target) interpolation problems of every model, and the windows that combine them."""
    keys, left, right, alpha, total = {}, [], [], [], []
    wins, m = [], None
    for sn_type, rng in MODEL_PHASES.items():
        for name in data.names('model', sn_type=sn_type):
            m = data.load('model', name, rebin=rebin, window=MODEL_WINDOW, phase_range=rng)
            t = m['t']
            for cfg in windows(t, max_gap=max_gap):
                j = cfg['target']
                ids = []
                for i, k in cfg['pairs']:
                    key = (name, i, j, k)
                    if key not in keys:
                        keys[key] = len(left)
                        left.append(m['rho'][i])
                        right.append(m['rho'][k])
                        alpha.append((t[j] - t[i]) / (t[k] - t[i]))
                        total.append(m['total'][j])
                    ids.append(keys[key])
                wins.append(dict(type=sn_type, model=name, start=cfg['start'], W=cfg['W'], p=cfg['p'], ph=t[j],
                                 ids=np.array(ids), dph=np.array([t[k] - t[i] for i, k in cfg['pairs']]),
                                 truth=m['mass'][j]))
    return m['wave'], m['edges'], wins, np.array(left), np.array(right), np.array(alpha), np.array(total)


def main():
    p = argparse.ArgumentParser(description='Replicate Fig. 5 of Ramirez et al. (2024) (Test 2).')
    p.add_argument('--reg', type=float, default=1e-3, help='regularisation of the converged entropic solvers')
    p.add_argument('--solvers', nargs='+', default=['paper', 'exact'], choices=SOLVERS)
    p.add_argument('--max-gap', type=float, default=MAX_GAP, help='largest phase gap in a window [d]')
    p.add_argument('--rebin', type=int, default=1, help='merge this many 5 A bins (speed; 1 = native grid)')
    p.add_argument('--backend', default='auto', choices=['auto', 'pot', 'numpy'])
    args = p.parse_args()
    sfx = suffix(args.reg, args.rebin)
    planes = sigma.load(plotting.OUTPUT_DIR / f'sigma_plane_{sfx}.json')['planes']

    wave, edges, wins, left, right, alpha, total = collect(args.rebin, args.max_gap)
    M = cost_matrix(wave)
    print(f'{len(wins)} windows and targets, {len(left)} unique interpolations on {len(wave)} bins')
    interp = run_series(left, right, alpha, total, M, edges, args.reg, args.solvers, args.backend)

    eps_by = {k: np.empty(len(wins)) for k in interp}
    for n, w in enumerate(wins):
        for k in interp:
            f = combine(interp[k][w['ids']], sigma.sigma(planes[k][w['type']], w['dph'], w['ph']))
            eps_by[k][n] = eps(w['truth'], f)

    types = np.array([w['type'] for w in wins])
    gap = np.array([w['dph'].max() for w in wins])
    ph = np.array([w['ph'] for w in wins])
    with open(plotting.OUTPUT_DIR / f'fig5_models_{sfx}.csv', 'w', newline='') as fh:
        cw = csv.writer(fh)
        cw.writerow(['type', 'model', 'start', 'W', 'p', 'ph_target', 'n_pairs', 'dph_max'] + [f'eps_{k}' for k in eps_by])
        for n, w in enumerate(wins):
            cw.writerow([w['type'], w['model'], w['start'], w['W'], w['p'], f'{w["ph"]:.3f}', len(w['ids']),
                         f'{gap[n]:.3f}'] + [f'{eps_by[k][n]:.5f}' for k in eps_by])
    bins = f'{5 * args.rebin} Å bins'
    for s in args.solvers:
        reg_note = '' if s in ('paper', 'exact') else f', reg = {args.reg:g}'
        path = plotting.gap_scatter(types, gap, ph, eps_by, ['linear', s], MODEL_PHASES,
                                    f'Test 2, weighted mean over the pairs of four spectra: '
                                    f'{plotting.SERIES[s]["label"]}{reg_note}, {bins}',
                                    'Largest phase gap among the pairs in the mean [d]', f'fig5_models_{s}_{sfx}.png')
        print('wrote', path)
    summarize(types, gap, eps_by)


if __name__ == '__main__':
    main()
