"""Which spectra to interpolate: triplets for Test 1, Fig. 4 windows for Test 2.

All functions take the sorted epochs t of one series and return indices into it.
"""
import numpy as np


def triplets(t, alpha_range=(0.45, 0.55), max_gap=None):
    """Test 1 (Sect. 4.1): all (i, j, k) with t[i] < t[j] < t[k] whose weight alpha = (t[j] - t[i]) / (t[k] - t[i])
    lies in alpha_range, and whose gap t[k] - t[i] is at most max_gap.

    Returns an (n, 3) int array and the (n,) alphas.
    """
    t = np.asarray(t, float)
    idx, alpha = [], []
    for i in range(len(t)):
        for k in range(i + 2, len(t)):
            gap = t[k] - t[i]
            if max_gap is not None and gap > max_gap:
                break
            for j in range(i + 1, k):
                a = (t[j] - t[i]) / gap
                if alpha_range[0] <= a <= alpha_range[1]:
                    idx.append((i, j, k))
                    alpha.append(a)
    return np.array(idx, int).reshape(-1, 3), np.array(alpha)


def windows(t, max_gap=None, n_inputs=4):
    """Test 2 (Fig. 4): input spectra, withheld targets, and the input pairs that interpolate each target.

    A window holds W consecutive spectra: n_inputs inputs, and a contiguous block of W - n_inputs withheld targets
    after the first p inputs, for p = 1 .. n_inputs - 1. Each target is interpolated from every pair of a left
    input and a right input. Fig. 4 shows W = 5 for p = 1, 2, 3 (cycles 1-3: 3, 4 and 3 pairs) and W = 6 for
    p = 1 (cycle 4). Windows slide along the series, and W grows while the widest pair, from the first spectrum of
    the window to the last, spans at most max_gap.

    Returns a list of dicts with keys target (int), pairs (list of (i, k)), start, W and p.
    """
    t = np.asarray(t, float)
    out = []
    for W in range(n_inputs + 1, len(t) + 1):
        grew = False
        for w in range(len(t) - W + 1):
            if max_gap is not None and t[w + W - 1] - t[w] > max_gap:
                continue
            grew = True
            for p in range(1, n_inputs):
                left = list(range(w, w + p))
                block = list(range(w + p, w + p + W - n_inputs))
                right = list(range(w + p + W - n_inputs, w + W))
                for j in block:
                    out.append(dict(target=j, pairs=[(i, k) for i in left for k in right], start=w, W=W, p=p))
        if not grew:
            break
    return out
