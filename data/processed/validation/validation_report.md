# Validation report: sn_spectra_ot.h5

## 1. Zero points (flux of a 0-mag source, photon-weighted)

| band | system | F0 this work | F0 paper (Table B.1) | ratio |
|---|---|---|---|---|
| B | Vega | 6.2164e-09 | 6.460e-09 | 0.962 |
| V | Vega | 3.7037e-09 | 3.675e-09 | 1.008 |
| R | Vega | 2.2287e-09 | 2.232e-09 | 0.999 |
| I | Vega | 1.1607e-09 | 1.177e-09 | 0.986 |
| g | AB | 4.9249e-09 | 4.791e-09 | 1.028 |
| r | AB | 2.8527e-09 | 2.818e-09 | 1.012 |
| i | AB | 1.9370e-09 | 1.907e-09 | 1.016 |
| b_uvot | Vega | 6.3854e-09 | - | - |
| v_uvot | Vega | 3.6991e-09 | - | - |

## 2. Calibration against photometry

- Ia: 2273 calibrated spectra, MAD(K)/K~ median 0.029; 92.8% pass the 10% golden cut
- II: 515 calibrated spectra, MAD(K)/K~ median 0.037; 85.6% pass the 10% golden cut

Per-band K_x / K~ for golden spectra (a band-dependent offset = zero-point or system mismatch):

| band | N | median | robust sigma | frac within 10% |
|---|---|---|---|---|
| B | 2164 | 1.011 | 0.042 | 88.6% |
| V | 2350 | 0.994 | 0.031 | 97.1% |
| R | 1170 | 0.997 | 0.031 | 94.7% |
| I | 1098 | 1.003 | 0.066 | 79.0% |
| g | 874 | 0.966 | 0.065 | 72.5% |
| r | 1127 | 1.000 | 0.034 | 93.4% |
| i | 692 | 1.000 | 0.048 | 86.4% |
| b_uvot | 618 | 1.004 | 0.051 | 85.4% |
| v_uvot | 740 | 1.017 | 0.037 | 90.7% |

Leave-one-band-out relative residual phi = F_syn/F_obs - 1 (golden spectra with >= 4 bands).
Paper Fig. 8 values are for OT-*interpolated* spectra, so they are a looser reference:

| band | N (Ia) | |phi|<0.1 Ia | paper Ia | N (II) | |phi|<0.1 II | paper II |
|---|---|---|---|---|---|---|
| B | 1165 | 77.9% | 67.6 | 233 | 76.0% | 68.2 |
| V | 1247 | 90.7% | 85.1 | 248 | 90.3% | 83.0 |
| R | 853 | 91.3% | 84.0 | 181 | 82.9% | 75.4 |
| I | 811 | 68.4% | 60.6 | 169 | 78.7% | 80.6 |
| g | 709 | 66.1% | 63.5 | 103 | 62.1% | 54.5 |
| r | 811 | 86.7% | 76.6 | 155 | 91.0% | 84.5 |
| i | 511 | 79.6% | 56.2 | 122 | 81.1% | 84.7 |
| b_uvot | 429 | 80.7% | - | 105 | 77.1% | - |
| v_uvot | 515 | 82.9% | - | 122 | 86.9% | - |

## 3. Sample vs paper Table C.1

- Ia: 110 SNe; golden spectra 2109, OT-ready 1853 (paper golden: 1714); SNe with >= 3 OT-ready spectra: 107
- II: 31 SNe; golden spectra 441, OT-ready 361 (paper golden: 298); SNe with >= 3 OT-ready spectra: 28

SNe with fewer than half of the paper's spectra (spectra absent from both OSC and the CSP release, or photometry too sparse to calibrate):

| SN | type | input spectra (OSC+CSP) | calibrated | golden | paper | phot bands |
|---|---|---|---|---|---|---|
| SN2007il | II | 23 | 3 | 2 | 5 | BVRI |

Phase of the earliest golden spectrum minus the paper's: median -1.0 d, |diff| <= 2 d for 78/132 SNe (differences come from different spectrum subsets and from t_ref; SNe Ia: V max, SNe II: t_PT).
- t_ref could not be measured for 9 SNe: SN2005ay, SN2013am, SN2013ai, SN1990E, SN2002gw, SN2007il, SN2007aa, SN2002bx, SN2008bj (phase = NaN, use MJD).

## 4. Total flux ("mass") per epoch

- Rebinning conserves flux: grid integral / native integral = 1.00023 (1st-99th pct 1.00001-1.00157).
- Ia: photometry-only SED total / spectral total = 1.015 (robust sigma 0.035, N=1853).
- II: photometry-only SED total / spectral total = 0.986 (robust sigma 0.040, N=361).
- Leave-one-out test of the daily photometric total-flux curve (mass_template) at 1929 spectral epochs, predicting each OT-ready spectrum's total from its neighbours + photometry: median |rel. error| 1.5%, 90th pct 5.7%; for neighbour gaps <= 10 d: 1.4%.

## 5. Usability: unbalanced-OT interpolation smoke test

For each of 133 SNe, the OT-ready spectrum with the widest bracketing gap (<= 40 d, target near the middle) is predicted from its two neighbours. 20 A bins; cost = squared wavelength distance normalised to 1; reg=0.0003 (entropic blur ~ sqrt(reg) x window width, ~69 A for a 4000 A window), reg_m=1. eps = mean |relative residual| of mass-normalised spectra (paper Eq. 4); mass error = predicted/true total - 1.

| bracket gap | N | eps UOT | eps linear | |mass err| UOT | |mass err| linear |
|---|---|---|---|---|---|
| 0-10 d | 27 | 0.109 | 0.064 | 3.4% | 3.5% |
| 10-20 d | 37 | 0.140 | 0.084 | 7.0% | 10.2% |
| 20-40 d | 69 | 0.221 | 0.146 | 10.0% | 12.4% |

This only shows that the arrays work as UOT input (non-negative, finite, common support, physical masses that can be rescaled to O(1)); it is an untuned baseline, not a result about the method.

## 6. Figures

- `calibration_loo_residuals.png`
- `cleaning_examples.png`
- `gallery_SN1999em.png`
- `gallery_SN2006be.png`
- `gallery_SN2006cp.png`
- `gallery_SN2011fe.png`
- `sample_vs_paper.png`
- `total_flux_checks.png`
