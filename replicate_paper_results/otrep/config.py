"""Settings shared by the model-spectra tests (Sect. 4.1) and their output names."""

MODEL_WINDOW = (3000.0, 10000.0)                   # the SN grid: OT input and Eq. 4 bins
MODEL_PHASES = {'Ia': (-5.0, 30.0), 'II': (-120.0, 0.0)}   # colour-bar ranges of Fig. 3
TEST1_ALPHA = (0.45, 0.55)                         # Test 1 uses only alpha in this range
MAX_GAP = 45.0                                     # largest phase gap [d]; the x axes of Figs. 3 and 5 end near 45 d
TEST3_MAX_GAP = 40.0                               # Sect. 4.2: pairs of spectra less than 40 d apart
OBS_PHASES = {'Ia': (-20.0, 80.0), 'II': (-130.0, 120.0)}  # colour ranges for the observed SNe
SOLVERS = ['paper', 'exact', 'sinkhorn', 'sinkhorn_log']   # OT solvers (otrep/barycenter.py)
BASELINES = ['linear', 'geometric']                # non-OT interpolations, always computed


def suffix(reg, rebin=1):
    """Name part shared by the outputs of one run, e.g. 'reg1e-03' or 'reg1e-03_rebin4'."""
    return f'reg{reg:.0e}' + (f'_rebin{rebin}' if rebin > 1 else '')
