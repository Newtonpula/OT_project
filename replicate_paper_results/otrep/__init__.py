"""Shared code for replicating Ramirez et al. (2024) and comparing balanced and unbalanced OT.

Every test script (black bodies, model spectra, observed SNe) builds on these modules:

    paths        repository paths; makes the repository's `snprep` package importable
    data         load any of the three datasets as unit-sum shapes rho = mass / total, plus the totals
    barycenter   cost matrix and entropic Wasserstein barycenters (POT, or a NumPy port of the same algorithm)
    interpolate  alpha from times, OT or linear interpolation of shapes, rescaling with a total
    metrics      the paper's mean relative spectral residual (Eq. 4)
    plotting     figure style and colours shared by all replication figures
"""
