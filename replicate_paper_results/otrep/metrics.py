"""Error measures for interpolated spectra."""
import numpy as np


def eps(f_true, f_interp, mask=None):
    """Mean relative spectral residual of the paper (Eq. 4): the mean over bins of |f_true - f_interp| / f_true.

    f_true, f_interp  (L,) or (m, L), in the same units (mass per bin or flux density)
    mask              optional (L,) boolean selection of bins, e.g. a wavelength window

    Bins where f_true <= 0 are skipped. Returns a float, or an (m,) array for (m, L) input.
    """
    f_true, f_interp = np.asarray(f_true, float), np.asarray(f_interp, float)
    sel = np.ones(f_true.shape[-1], bool) if mask is None else np.asarray(mask, bool)
    t, i = f_true[..., sel], f_interp[..., sel]
    good = t > 0
    r = np.where(good, np.abs(t - i) / np.where(good, t, 1.0), 0.0)
    return r.sum(axis=-1) / good.sum(axis=-1)
