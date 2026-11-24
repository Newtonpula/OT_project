"""The paper's uncertainty model Sigma(eps): a plane eps = a * dph + b * ph + c fitted to the Test 1 residuals
(Sect. 4.1), separately for SNe Ia and SNe II. Tests 2-3 use it to weight interpolations (Eqs. 5-6).
"""
import json

import numpy as np

# Lower bound on Sigma(eps), as in the authors' code (STS_OT, '4 - Weighted Mean OT .ipynb':
# "if epsilon < 0.01: epsilon = 0.01"). It also keeps 1 / Sigma^2 finite where the plane drops to zero or below.
FLOOR = 0.01


def fit_plane(dph, ph, eps):
    """Least-squares plane eps = a * dph + b * ph + c. Returns dict(a, b, c, r2, rms, n)."""
    dph, ph, eps = (np.asarray(x, float) for x in (dph, ph, eps))
    X = np.c_[dph, ph, np.ones_like(dph)]
    coef, *_ = np.linalg.lstsq(X, eps, rcond=None)
    resid = eps - X @ coef
    r2 = 1 - resid.var() / eps.var() if eps.var() > 0 else np.nan
    return dict(a=float(coef[0]), b=float(coef[1]), c=float(coef[2]), r2=float(r2),
                rms=float(np.sqrt(np.mean(resid ** 2))), n=int(len(eps)))


def sigma(plane, dph, ph, floor=FLOOR):
    """Sigma(eps) of interpolations with gap dph at phase ph, clipped below at floor."""
    val = plane['a'] * np.asarray(dph, float) + plane['b'] * np.asarray(ph, float) + plane['c']
    return np.maximum(val, floor)


def save(path, planes, **meta):
    with open(path, 'w') as fh:
        json.dump(dict(meta, planes=planes), fh, indent=2)


def load(path):
    with open(path) as fh:
        return json.load(fh)
