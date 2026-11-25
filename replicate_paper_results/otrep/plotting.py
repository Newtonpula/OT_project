"""Figure style shared by all replication figures.

Colours follow the entity in every figure: the true spectrum in ink, OT in blue, linear interpolation in red.
The paper itself swaps blue and red between its Fig. 2 and Fig. 3; we keep one mapping throughout. The two OT
solvers share the blue hue at two lightness steps, and line style and direct labels tell them apart. The
values come from the reference palette of the data-viz guidelines (categorical blue and red, the blue ramp, ink
and chrome). Unbalanced OT will take the next categorical slot (orange).
"""
import matplotlib
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb

from .paths import OUTPUT_DIR

INK = '#0b0b0b'
SECONDARY = '#52514e'
MUTED = '#898781'
GRID = '#e1e0d9'
AXIS = '#c3c2b7'

# series key -> colour, line style (second channel, so identity is never colour alone), label
SERIES = {
    'truth': dict(color=INK, ls='-', label='True spectrum'),
    'linear': dict(color='#e34948', ls='-', label='Linear (L2 barycenter)'),
    'geometric': dict(color='#1baf7a', ls='-.', label='Geometric mean'),
    'paper': dict(color='#2a78d6', ls='-', label="OT, authors' settings (10 iterations)"),
    'exact': dict(color='#6250d6', ls='--', label='OT, exact W2'),
    'sinkhorn': dict(color='#86b6ef', ls='--', label='OT, POT default, converged'),
    'sinkhorn_log': dict(color='#184f95', ls='-', label='OT, sinkhorn_log, converged'),
}

STYLE = {
    'figure.dpi': 110, 'savefig.dpi': 200, 'savefig.bbox': 'tight',
    'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
    'font.size': 10, 'axes.titlesize': 10.5, 'axes.labelsize': 10, 'legend.fontsize': 9,
    'axes.edgecolor': AXIS, 'axes.labelcolor': SECONDARY, 'axes.titlecolor': INK,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
    'xtick.color': AXIS, 'ytick.color': AXIS, 'xtick.labelcolor': SECONDARY, 'ytick.labelcolor': SECONDARY,
    'lines.linewidth': 1.5, 'legend.frameon': False,
}


def use_style():
    matplotlib.rcParams.update(STYLE)


def ramp(color, light=0.75, dark=0.45):
    """One-hue colormap for a magnitude: from a light tint of `color`, through `color`, to a dark shade of it."""
    c = np.array(to_rgb(color))
    return LinearSegmentedColormap.from_list('ramp', [c + (1 - c) * light, c, c * (1 - dark)])


def binned_median(x, y, edges):
    """Median of y in bins of x; NaN for empty bins. Returns the bin centres and the medians."""
    mid = 0.5 * (edges[1:] + edges[:-1])
    med = [np.median(y[(x >= a) & (x < b)]) if np.any((x >= a) & (x < b)) else np.nan
           for a, b in zip(edges[:-1], edges[1:])]
    return mid, np.array(med)


def gap_scatter(types, gap, phase, eps_by, keys, phase_ranges, title, xlabel, name, bin_width=5.0,
                ylabel='ε (Eq. 4) over 3000–10000 Å [%]', panel='SN {type} models ({n} spectra)'):
    """Figs. 3 and 5: eps against the phase gap, one panel per SN type.

    Marker size and darkness grow with the phase of the interpolated spectrum, as in the paper. A line gives the
    median per bin_width days.

    types, gap, phase  (n,) arrays, one entry per interpolated spectrum
    eps_by             {series key: (n,) eps}; `keys` picks and orders the series drawn
    phase_ranges       {type: (lo, hi)} colour range of each panel
    """
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D
    use_style()
    fig, axes = plt.subplots(1, len(phase_ranges), figsize=(11, 4.3), sharey=True)
    edges = np.arange(0.0, gap.max() + bin_width, bin_width)
    for ax, sn_type in zip(np.atleast_1d(axes), phase_ranges):
        sel = types == sn_type
        norm = Normalize(*phase_ranges[sn_type])
        tn = np.clip(norm(phase[sel]), 0, 1)
        for key in keys:
            y = 100 * eps_by[key][sel]
            ax.scatter(gap[sel], y, s=6 + 26 * tn, c=tn, cmap=ramp(SERIES[key]['color']), vmin=0, vmax=1,
                       linewidths=0, alpha=0.85, zorder=2)
            mid, med = binned_median(gap[sel], y, edges)
            ax.plot(mid, med, color=SERIES[key]['color'], ls=SERIES[key]['ls'], lw=1.6, zorder=3)
        cb = fig.colorbar(ScalarMappable(norm=norm, cmap=ramp('#6e6d69')), ax=ax, pad=0.02, fraction=0.05)
        cb.set_label('Phase of the interpolated spectrum [d]', color=SECONDARY)
        cb.outline.set_visible(False)
        ax.set_title(panel.format(type=sn_type, n=sel.sum()), loc='left')
        ax.set_xlabel(xlabel)
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel(ylabel)
    handles = [Line2D([], [], marker='o', ls=SERIES[k]['ls'], color=SERIES[k]['color'], ms=5, lw=1.6,
                      label=SERIES[k]['label']) for k in keys]
    leg = axes[0].legend(handles=handles, loc='upper left', title=f'lines: median per {bin_width:g} d',
                         alignment='left')
    leg.get_title().set_color(SECONDARY)
    fig.suptitle(title, x=0.01, ha='left', fontsize=10, color=SECONDARY)
    return save(fig, name)


def save(fig, name):
    """Save a figure to replicate_paper_results/outputs/<name>, close it and return the path."""
    import matplotlib.pyplot as plt
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    fig.savefig(path)
    plt.close(fig)
    return path
