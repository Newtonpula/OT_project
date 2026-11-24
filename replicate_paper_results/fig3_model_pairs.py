"""Replicate Fig. 3 of Ramirez et al. (2024), Test 1: single-pair interpolation of CMFGEN model spectra.

    python replicate_paper_results/fig3_model_pairs.py [--reg 1e-3] [--solvers paper exact sinkhorn_log] [--rebin 1]

The test runs over every model and every triplet of epochs ph1 < ph2 < ph3 with 0.45 <= alpha <= 0.55 and
ph3 - ph1 <= --max-gap. For each triplet:

1. the spectrum at ph2 is interpolated from those at ph1 and ph3, with every OT solver in --solvers
   (otrep/barycenter.py), linearly, and as a geometric mean;
2. it is rescaled with the true total at ph2 (methods.md, Sect. 6);
3. it is compared with the true spectrum at ph2 through Eq. 4.

A plane eps = a * dph + b * ph2 + c is then fitted for each SN type and method. It is the Sigma(eps) of Tests 2-3.
The grid and the phase ranges are in otrep/config.py.

Outputs (replicate_paper_results/outputs/):
    fig3_models_<solver>_<suffix>.png   Fig. 3: eps against the phase gap, OT and linear
    fig3_models_<suffix>.csv            one row per triplet
    sigma_plane_<suffix>.json           the fitted planes, read by Test 2
<suffix> is reg<reg>, plus _rebin<k> when --rebin k > 1.

Needs data/processed/model_spectra_ot.h5 (scripts/download_models.py, then scripts/build_models.py).
"""
import argparse
import csv
import time

import matplotlib
matplotlib.use('Agg')
import numpy as np  # noqa: E402

from otrep import data, plotting, sigma  # noqa: E402
from otrep.barycenter import cost_matrix  # noqa: E402
from otrep.config import BASELINES, MAX_GAP, MODEL_PHASES, MODEL_WINDOW, SOLVERS, TEST1_ALPHA, suffix  # noqa: E402
from otrep.interpolate import interpolate, rescale  # noqa: E402
from otrep.metrics import eps  # noqa: E402
from otrep.pairs import triplets  # noqa: E402


def collect(rebin, max_gap):
    """Triplets of every model, stacked: endpoint shapes, alpha, true masses at ph2, and one metadata row each."""
    rows, left, right, alpha, truth, m = [], [], [], [], [], None
    for sn_type, rng in MODEL_PHASES.items():
        for name in data.names('model', sn_type=sn_type):
            m = data.load('model', name, rebin=rebin, window=MODEL_WINDOW, phase_range=rng)
            idx, al = triplets(m['t'], TEST1_ALPHA, max_gap)
            t = m['t']
            rows += [dict(type=sn_type, model=name, ph1=t[i], ph2=t[j], ph3=t[k], dph=t[k] - t[i], alpha=a)
                     for (i, j, k), a in zip(idx, al)]
            left.append(m['rho'][idx[:, 0]])
            right.append(m['rho'][idx[:, 2]])
            alpha.append(al)
            truth.append(m['mass'][idx[:, 1]])
    return m['wave'], m['edges'], rows, *(np.concatenate(x) for x in (left, right, alpha, truth))


def run_series(left, right, alpha, total, M, edges, reg, solvers, backend):
    """Baselines and OT solvers on stacked pairs; each result is rescaled to `total`. Returns {key: masses}."""
    out = {b: rescale(interpolate(left, right, alpha, method=b)[0], total) for b in BASELINES}
    for s in solvers:
        t0 = time.time()
        shapes, info = interpolate(left, right, alpha, method='ot', M=M, reg=reg, solver=s, edges=edges,
                                   backend=backend)
        out[s] = rescale(shapes, total)
        extra = (f', reg {np.nanmin(info["reg"]):.1e}-{np.nanmax(info["reg"]):.1e}' if 'reg' in info else
                 f', iterations {info["n_iter"].min()}-{info["n_iter"].max()}' if s.startswith('sinkhorn') else '')
        print(f'{s}: backend {info["backend"]}, {time.time() - t0:.0f} s{extra}')
    return out


def summarize(types, gap, eps_by, planes=None, limits=None):
    """Print, per SN type and method: median, maximum, maximum up to a gap limit, and the share below 10%."""
    limits = limits or {'Ia': 25.0, 'II': 40.0}
    for sn_type in MODEL_PHASES:
        sel = types == sn_type
        lim = limits[sn_type]
        print(f'\nSN {sn_type}: {sel.sum()} spectra, phase gaps {gap[sel].min():.1f}-{gap[sel].max():.1f} d')
        print(f'  {"method":14s}{"median":>8s}{"max":>8s}{f"max (gap<={lim:.0f} d)":>18s}{"< 10%":>8s}')
        for k, e in eps_by.items():
            e = e[sel]
            line = (f'  {k:14s}{100 * np.median(e):7.1f}%{100 * e.max():7.1f}%'
                    f'{100 * e[gap[sel] <= lim].max():17.1f}%{100 * np.mean(e < 0.1):7.0f}%')
            if planes:
                q = planes[k][sn_type]
                line += f'   plane eps = {q["a"]:.4f} dph {q["b"]:+.4f} ph {q["c"]:+.3f} (R^2 {q["r2"]:.2f})'
            print(line)


def main():
    p = argparse.ArgumentParser(description='Replicate Fig. 3 of Ramirez et al. (2024) (Test 1).')
    p.add_argument('--reg', type=float, default=1e-3)
    p.add_argument('--solvers', nargs='+', default=['paper', 'exact', 'sinkhorn_log'], choices=SOLVERS)
    p.add_argument('--max-gap', type=float, default=MAX_GAP, help='largest phase gap ph3 - ph1 [d]')
    p.add_argument('--rebin', type=int, default=1, help='merge this many 5 A bins (speed; 1 = native grid)')
    p.add_argument('--backend', default='auto', choices=['auto', 'pot', 'numpy'])
    args = p.parse_args()
    sfx = suffix(args.reg, args.rebin)

    wave, edges, rows, left, right, alpha, truth = collect(args.rebin, args.max_gap)
    M = cost_matrix(wave)
    print(f'{len(rows)} triplets on {len(wave)} bins')
    interp = run_series(left, right, alpha, truth.sum(axis=1), M, edges, args.reg, args.solvers, args.backend)
    eps_by = {k: eps(truth, f) for k, f in interp.items()}
    if 'paper' in eps_by:
        d = np.abs(eps_by['paper'] - eps_by['geometric'])
        print(f'paper vs geometric mean: |eps difference| median {100 * np.median(d):.2f}, max {100 * d.max():.2f} '
              'percentage points')

    types = np.array([r['type'] for r in rows])
    gap = np.array([r['dph'] for r in rows])
    ph = np.array([r['ph2'] for r in rows])
    planes = {k: {t: sigma.fit_plane(gap[types == t], ph[types == t], e[types == t]) for t in MODEL_PHASES}
              for k, e in eps_by.items()}
    out = plotting.OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    sigma.save(out / f'sigma_plane_{sfx}.json', planes, reg=args.reg, rebin=args.rebin, max_gap=args.max_gap,
               window=MODEL_WINDOW, phases=MODEL_PHASES, alpha_range=TEST1_ALPHA)
    with open(out / f'fig3_models_{sfx}.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['type', 'model', 'ph1', 'ph2', 'ph3', 'dph', 'alpha'] + [f'eps_{k}' for k in eps_by])
        for i, r in enumerate(rows):
            w.writerow([r['type'], r['model']] + [f'{r[c]:.3f}' for c in ('ph1', 'ph2', 'ph3', 'dph', 'alpha')]
                       + [f'{eps_by[k][i]:.5f}' for k in eps_by])
    bins = f'{5 * args.rebin} Å bins'
    for s in args.solvers:
        reg_note = '' if s in ('paper', 'exact') else f', reg = {args.reg:g}'
        path = plotting.gap_scatter(types, gap, ph, eps_by, ['linear', s], MODEL_PHASES,
                                    f'Test 1, single pairs: {plotting.SERIES[s]["label"]}{reg_note}, {bins}',
                                    'Phase gap of the pair, ph3 − ph1 [d]', f'fig3_models_{s}_{sfx}.png')
        print('wrote', path)
    summarize(types, gap, eps_by, planes)


if __name__ == '__main__':
    main()
