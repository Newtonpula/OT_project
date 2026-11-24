"""Load the processed datasets as balanced-OT inputs.

Only `mass` and `total` from `snprep.load` are used (methods.md, Sect. 6). The OT input is the unit-sum shape
rho = mass / total, and `total` is the brightness used to rescale interpolated shapes.
"""
import numpy as np

from . import paths  # noqa: F401  (puts the repository root on sys.path for snprep)
from snprep.load import BBDataset, ModelDataset, SNDataset  # noqa: E402

READERS = {'bb': BBDataset, 'model': ModelDataset, 'sn': SNDataset}


def names(kind, **kwargs):
    """Series names of a dataset, e.g. names('model', sn_type='Ia')."""
    return READERS[kind]().names(**kwargs)


def load(kind, name, rebin=1, **kwargs):
    """`ot_measures()` of one series, as balanced-OT input.

    kind     'bb' (black bodies), 'model' (CMFGEN models) or 'sn' (observed supernovae)
    name     series, model or SN name, e.g. 'paper', 'DDC15', 'SN2011fe'
    rebin    merge this many adjacent bins (mass is summed, so totals are unchanged; any leftover bins at the red
             end are dropped)
    kwargs   passed on to `ot_measures()` (window=..., phase_range=..., ...)

    Returns the `ot_measures()` dict (wave, mass, total, t, ...) with three extra entries:
        rho     (N, L) unit-sum shapes, mass / total
        dlam    (L,) bin widths [A]; the flux density is mass / dlam
        edges   (L + 1,) bin edges [A]
    """
    m = READERS[kind]().ot_measures(name, **kwargs)
    if rebin > 1:
        L = m['mass'].shape[1] // rebin * rebin
        m['mass'] = m['mass'][:, :L].reshape(len(m['mass']), -1, rebin).sum(axis=2)
        m['wave'] = m['wave'][:L].reshape(-1, rebin).mean(axis=1)
        m['total'] = m['mass'].sum(axis=1)
    m['rho'] = m['mass'] / m['total'][:, None]
    m['dlam'] = np.gradient(m['wave'])
    m['edges'] = np.r_[m['wave'] - m['dlam'] / 2, m['wave'][-1] + m['dlam'][-1] / 2]
    return m
