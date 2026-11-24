# Datasets for unbalanced-OT interpolation of spectra

Three datasets for reproducing Ramirez et al. 2024 (A&A, arXiv:2409.10701) with (unbalanced) optimal transport:

1. **Black-body test series** (paper §3, Fig. 2): Planck spectra from 5000 K to 10000 K, the paper's first
   test of OT interpolation.
2. **Model spectra** (paper §4.1, Figs. 3–5): noiseless CMFGEN spectral time series of 16 SN Ia and 6 SN II
   explosion models (Dessart et al. 2014, 2013), used by the paper to test the interpolation.
3. **SN spectral time series**: the paper's observational sample of 141 supernovae, flux-calibrated against
   photometry.

All three use the same form: 5 Å wavelength bins, `flux` as a density per Å, mass per bin = flux × 5 Å, and a
reader in `snprep.load` whose `ot_measures()` returns `wave`, `mass`, `total` and `t`.

## Building the .h5 files locally

The HDF5 files are **not in the git repository** (`data/processed/*.h5` is listed in `.gitignore`; the SN
file is 131 MB, above GitHub's 100 MB file limit), and neither are the raw downloads in `data/raw/`.
After cloning, generate them on your own machine. Run every command from the repository root
(the `OT_project/` folder).

**1. Python environment** (tested with Python 3.14). The black-body series needs only numpy, h5py and
matplotlib; the model spectra also need scipy and requests; the SN dataset needs all of them.

```bash
python3 -m venv .venv
```
```bash
source .venv/bin/activate
```
```bash
pip install numpy scipy h5py matplotlib requests astropy extinction
```

**2. Black-body series.** No download; takes about a second.

```bash
python scripts/build_blackbody.py
```

This writes `data/processed/blackbody_spectra_ot.h5` (about 2.5 MB), `data/processed/summary_blackbody.csv`
and `data/processed/validation/blackbody_series.png`. It prints its numerical checks and stops with an
error if one fails.

**3. Model spectra.** Download the CMFGEN models from Zenodo (about 200 MB, a few minutes; details in
[section 2](#2-model-spectra-paper-41-figs-35)), then build the dataset (a few seconds):

```bash
python scripts/download_models.py
```
```bash
python scripts/build_models.py
```

This writes `data/processed/model_spectra_ot.h5` (about 14 MB), `summary_models.csv`,
`summary_model_spectra.csv`, `build_models_log.txt` and `data/processed/validation/models_overview.png`.

**4. SN dataset.** First download the raw data. This needs network access and fetches about 720 MB:
the OSC JSON files from the `astrocatalogs/sne-*` GitHub repositories into `data/raw/osc/`, and the CSP
SN II tarball into `data/raw/csp/`. Files that already exist are skipped, so the command can simply be
run again after an interruption.

```bash
python scripts/download_data.py
```

Then build the dataset (about 20 s for all 141 SNe). This writes `data/processed/sn_spectra_ot.h5`
(about 131 MB), `summary_sne.csv`, `summary_spectra.csv` and `build_log.txt`.

```bash
python scripts/build_dataset.py
```

Optionally, regenerate the validation report and figures in `data/processed/validation/` (about 1 minute):

```bash
python scripts/validate_dataset.py
```

**5. Check** that all three files load; this should print `['dense', 'paper'] 22 141`:

```bash
python -c "from snprep.load import BBDataset, ModelDataset, SNDataset; print(BBDataset().names(), len(ModelDataset().names()), len(SNDataset().names()))"
```

## 1. Black-body test series (paper §3, Fig. 2)

Before using real SNe, the paper tests OT on Planck spectra: it interpolates between black bodies at 5000 K
and 10000 K and compares the Wasserstein and L2 paths with the true intermediate black bodies.
`processed/blackbody_spectra_ot.h5` holds those spectra in the same form as the SN data.

### Quick start

```python
from snprep.load import BBDataset

bb = BBDataset()                                    # reads data/processed/blackbody_spectra_ot.h5
m = bb.ot_measures('paper')                         # 10 spectra, full 1000–15000 Å grid
m['wave'], m['mass'], m['total']                    # (2800,), (10, 2800), (10,)
m['T'], m['alpha']                                  # temperature [K]; alpha = (T − 5000)/5000 (also in m['t'])
g = bb.ot_measures('dense', window=(3000, 10000))   # 101 spectra on exactly the SN grid (wave_rest of sn_spectra_ot.h5)
```

### Files

| File | Contents |
|---|---|
| `processed/blackbody_spectra_ot.h5` | B_λ(T) for 5000–10000 K (about 2.5 MB; not in git, build it locally) |
| `processed/summary_blackbody.csv` | One row per spectrum: T, α, Wien peak, totals |
| `processed/validation/blackbody_series.png` | The curves of the paper's Fig. 2, drawn from the file |

Code: `scripts/build_blackbody.py` (build and checks), `snprep/blackbody.py` (Planck functions, usable for
any temperature).

### HDF5 layout

```
/wave_rest (2800,)  /wave_edges (2801,)     1000–15000 Å, 5 Å bins
attrs: flux_units, mass_units, constants
/series/<paper|dense>/
    attrs: description, window_rest = (1000, 15000)
    spectra/  (N spectra, sorted by temperature)
        temperature (N,)    [K]
        alpha (N,)          (T − 5000)/5000, the barycenter weight between the two ends
        flux (N,2800)       B_λ averaged over each bin [erg s⁻¹ cm⁻² Å⁻¹ sr⁻¹]
        mass_window (N,)    ∫B_λ dλ over the grid [erg s⁻¹ cm⁻² sr⁻¹]
        mass_full (N,)      ∫B_λ dλ over all wavelengths = σT⁴/π
        peak_wave, peak_flux  Wien peak [Å] and B_λ there
```

### Notes

- **Quantity.** `flux` is the spectral radiance B_λ in erg s⁻¹ cm⁻² Å⁻¹ sr⁻¹, the unit of the paper's
  Fig. 2, averaged over each bin (Gauss–Legendre quadrature, so `mass` is the exact integral over the bin).
  It is computed with exact SI-2019 constants. Mass is in erg s⁻¹ cm⁻² sr⁻¹; the 10000 K peak is 4.10×10⁷,
  as on the paper's axis.
- **Grid.** 1000–15000 Å with 5 Å bins (2800 bins), the range plotted in Fig. 2. The edges line up with
  the SN grid, so `window=(3000, 10000)` selects exactly the 1400 SN bins. The 10000 K peak (2898 Å)
  lies outside that window.
- **Series.**
  - `paper`: `np.linspace(5000, 10000, 10)`, the ten black curves of Fig. 2. The paper does not state
    them; they were read off the peak heights (4.1, 3.1, 2.3, 1.6, 1.2, … ×10⁷). Round 500 K steps would
    give 3.2, 2.4, 1.8, … instead.
  - `dense`: 50 K steps (101 spectra), ground truth for any intermediate α.
  - The endpoints of both series are the same two spectra.
- **Unbalanced.** The total mass grows about 19× from 5000 K to 10000 K over the grid; over all
  wavelengths it grows 16×, since it is ∝ T⁴. The grid holds 83% (5000 K) to 97% (10000 K) of the
  total `mass_full` = σT⁴/π, because the red tail of the cool spectra extends beyond 15000 Å.
- **Checks** (run on every build). The window integral agrees with the analytic band integral of the
  Planck function to 1e-15, and every bin agrees to 2e-11. The brightest bin always contains the Wien
  peak.
- **Paper setup of the OT test.** The paper calls `ot.bregman.barycenter` with the weights [1−α, α];
  it works on normalised histograms, so divide each spectrum by its total first. For unbalanced OT, use
  the raw masses.

## 2. Model spectra (paper §4.1, Figs. 3–5)

For its first two sets of tests (Figs. 3–5), the paper uses noiseless model spectra instead of observations:
the public CMFGEN radiative-transfer time series of Dessart et al. (2014) for SNe Ia and Dessart et al. (2013)
for SNe II. A spectrum between two others is interpolated, rescaled against the synthetic photometry and
compared with the true model spectrum (Eq. 4). `processed/model_spectra_ot.h5` holds these models in the
same form as the other two datasets.

### Download and build

Both sets are on Zenodo under CC-BY-4.0. The author's own pages (www-n.oca.eu) were unreachable when this was
written, so the script uses Zenodo only:

| Set | Paper | Zenodo | Models | Download |
|---|---|---|---|---|
| SN Ia | Dessart, Blondin, Hillier & Khokhlov 2014, MNRAS 441, 532 | [10.5281/zenodo.5524943](https://doi.org/10.5281/zenodo.5524943) | 16 delayed detonations: DDC0, 6, 10, 15, 17, 20, 22, 25, DDC10_M4 (DDC10 with mixing), PDDEL1, 3, 4, 7, 9, 11, 12 | 11 MB (43 MB extracted) |
| SN II | Dessart, Hillier, Waldman & Livne 2013, MNRAS 433, 1745 | [10.5281/zenodo.5524916](https://doi.org/10.5281/zenodo.5524916) | 6 explosions of 15 M☉ red supergiants: m15mlt1, m15mlt3, m15z2m2, m15z2m3, m15z4m2, m15z8m3 | 190 MB (0.9 GB extracted) |

From the repository root:

```bash
python scripts/download_models.py
```
```bash
python scripts/build_models.py
```

The first command reads the file list of both Zenodo records, downloads the archives into
`data/raw/models/dessart14_snia/` and `data/raw/models/dessart13_snii/`, checks their MD5 sums and extracts
them; archives that are already present and intact are skipped. The second command reads the raw files,
puts every spectrum on the common grid and writes the files listed below. Both scripts are needed only once.

Raw files, in case you want to read them directly:

- **SN Ia:** `SNIa_DDC_DDC10M4_PDDEL_D14b/<model>_spec_1500_10000_interp5_<t>d.dat`, where t is days since
  explosion. Two columns: wavelength [Å] and f_λ [erg s⁻¹ cm⁻² Å⁻¹ at 10 pc], every 5 Å over 1500–10000 Å.
  The archive also has a `README` and the hydrodynamical inputs (`*_SN_HYDRO_DATA_*`).
- **SN II:** `<model>/<prefix>_<k>.fl` and `<prefix>_<k>nt.fl`, at native resolution over 17 Å–76 µm. Two
  columns: wavelength [Å] and f_λ at 1 kpc. `<prefix>_list` maps each file to days since explosion. Eight
  `.fl` files have no entry there and are skipped (listed in `build_models_log.txt`).

### Quick start

```python
import numpy as np
from snprep.load import ModelDataset
from snprep.synphot import Filters

md = ModelDataset()                                  # reads data/processed/model_spectra_ot.h5
md.names('Ia'), md.names('II')                       # 16 SN Ia models, 6 SN II models
m = md.ot_measures('DDC15', window=(3000, 10000), phase_range=(-5, 30))
m['wave'], m['mass'], m['total']                     # (1400,), (12, 1400), (12,): exactly the SN grid
m['t'], m['t_exp']                                   # days from V maximum (t_PT for SNe II); days since explosion
m['F_band'], m['mag']                                # (12, 7) synthetic photometry in md.bands = B,V,R,I,g,r,i

# the paper's single-pair test (Sect. 4.1, Fig. 3) for spectra at ph1 < ph2 < ph3
filt = Filters('data/meta/filters')
f = m['mass'] / 5.0                                  # f_lambda [erg s^-1 cm^-2 A^-1 at 10 pc]
k1, k2, k3 = 4, 5, 6
alpha = (m['t'][k2] - m['t'][k1]) / (m['t'][k3] - m['t'][k1])
f_interp = (1 - alpha) * f[k1] + alpha * f[k3]       # linear interpolation; replace with the OT barycenter
F_syn = np.array([filt.synflux(m['wave'], f_interp, b)[0] for b in md.bands])
K = F_syn / m['F_band'][k2]                          # paper Eqs. 1-2, against the photometry at ph2
f_resc = f_interp / np.median(K)                     # MAD(K)/median(K) is the calibration error
eps = np.mean(np.abs(f[k2] - f_resc) / f[k2])        # paper Eq. 4; 0.025 here (alpha = 0.48, gap 5.2 d)
```

### Files

| File | Contents |
|---|---|
| `processed/model_spectra_ot.h5` | **Model dataset** (about 14 MB; not in git, build it locally) |
| `processed/summary_models.csv` | One row per model: epochs, t_ref and how it was found, phase range, peak M_V, M_V at 50 d, M(⁵⁶Ni) for SNe II |
| `processed/summary_model_spectra.csv` | One row per spectrum: t_exp, phase, totals, synthetic magnitudes |
| `processed/build_models_log.txt` | Per-model checks, Olivares fits of the SNe II, skipped raw files |
| `processed/validation/models_overview.png` | V light curves with the phase zero points, and two spectral sequences |
| `raw/models/` | Zenodo archives and extracted files (about 1 GB; not in git) |

Code: `scripts/download_models.py`, `scripts/build_models.py`, `snprep/models.py` (raw readers, phase
references).

### HDF5 layout

```
/wave_rest (1700,)  /wave_edges (1701,)     1500–10000 Å, 5 Å bins (edges line up with the SN grid)
attrs: bands = "B,V,R,I,g,r,i"   ← column order of mag and F_band;  flux_units, mass_units, mag_system
/models/<name>/
    attrs: type (Ia/II), reference, doi, raw_dir, raw_prefix, t_ref, t_ref_method, window_rest = (1500, 10000)
           SNe II also: Z, R_star_Rsun, E_kin_1e51erg, M_Ni56_Msun   (Dessart et al. 2013, Table 1)
    spectra/  (N spectra, sorted by time)
        t_exp (N,)          days since explosion
        phase (N,)          t_exp − t_ref: days from V maximum (Ia) or from t_PT (II)
        flux (N,1700)       f_λ at 10 pc [erg s⁻¹ cm⁻² Å⁻¹], averaged over each bin
        mass_window (N,)    ∫f dλ over the grid [erg s⁻¹ cm⁻²]
        mass_full (N,)      ∫f dλ over the whole model spectrum (SNe II: 17 Å–76 µm, nearly bolometric)
        F_band (N,7)        synthetic photometry: photon-weighted mean f_λ in each band
        mag (N,7)           the same as absolute magnitudes (Vega for BVRI, AB for gri)
        filename, rebin_int_ratio
```

### Notes

- **Units.** The SN Ia files are f_λ at 10 pc (stated in the archive README). The SN II files have no README.
  They are f_λ at 1 kpc, the CMFGEN convention, and are multiplied by 10⁴ to bring them to 10 pc as well.
  As a check, the nebular-phase luminosity of every SN II model is 0.86–1.27× the ⁵⁶Co decay power for its
  ⁵⁶Ni mass in Dessart et al. 2013 (Table 1). Peak M_V of the SN Ia models runs from −17.8 (DDC25) to −19.7
  (DDC0); M_V at 50 d of the SN II models runs from −16.9 to −17.7. Total masses range from about 0.05 to
  1500 erg s⁻¹ cm⁻², so divide by a constant before running Sinkhorn, as for the SN data.
- **Phases** follow the paper (§4). SNe Ia are referred to V maximum, from a second-order polynomial within
  ±5 d of the brightest epoch (19.7–23.4 d after explosion). SNe II are referred to t_PT, the midpoint of the
  plateau-to-tail transition, from an Olivares et al. (2010) fit to the V light curve (134–156 d after
  explosion; Dessart et al. 2013 quote plateaus of 120–150 d).
- **Which models.** The paper does not list the models it used. The colour scales of its Fig. 3 cover phases
  −5 to 30 d (SNe Ia) and −120 to 0 d (SNe II); `phase_range` selects such a range. Zenodo holds 6 of the
  SN II models in Table 1 of Dessart et al. 2013: all 15 M☉ progenitors, differing in radius (mixing
  length) and metallicity.
- **Photometry.** The paper's Johnson–Cousins and Sloan bands are BVRI and gri here, with the same filter
  curves and zero points as the SN dataset (`snprep.synphot`). The photometry is computed on the gridded
  spectra, so a stored spectrum has K̃ = 1 exactly against its own `F_band`.
- **Grid.** 1500–10000 Å is the full range of the SN Ia files. The SN II spectra extend further and are cut
  to the same range (`mass_full` keeps their total). The SN II flux below 3000 Å falls to as little as 10⁻⁴
  of the optical peak, and such bins dominate the relative residuals of Eq. 4, so compute ε in a window
  such as `window=(3000, 10000)`, which selects exactly the SN grid.
- **Checks** (run on every build). All fluxes are finite and non-negative, the integral over the grid
  matches the native spectrum to within 1e-4, and every phase reference lies in the expected range.

## 3. SN spectral time-series dataset

This is a rebuild of the observational sample used by Ramirez et al. 2024 (A&A, arXiv:2409.10701):
**141 supernovae (110 Ia + 31 II, i.e. the paper's Table C.1)**. The data come from the Open Supernova
Catalog (OSC); 14 SNe II are supplemented with the CSP SN II spectra (Gutiérrez et al. 2017). The spectra
are flux-calibrated in absolute terms against photometry, resampled onto a common rest-frame wavelength
grid, and come with the **total flux (mass) at every epoch**.
The spectra are not normalised, so the integral of the spectrum can differ from epoch to epoch, which
is exactly the setting of unbalanced OT, where the measures have unequal masses.

### Quick start

```python
import sys; sys.path.insert(0, '/path/to/OT_project')  # the folder you cloned; can be skipped if you start Python in that folder
from snprep.load import SNDataset

ds = SNDataset()                              # reads data/processed/sn_spectra_ot.h5
names = ds.names(sn_type='Ia', min_ot_ready=3)  # 107 SNe Ia; 28 SNe II

m = ds.ot_measures('SN2011fe')                # UOT input
m['wave']    # (L,)   rest-frame wavelength bin centres [Å] (5 Å bins, restricted to this SN's common window)
m['mass']    # (N, L) mass per bin = f_λ × 5 Å [erg s⁻¹ cm⁻²], non-negative
m['total']   # (N,)   total flux of each spectrum = mass.sum(1)
m['t']       # (N,)   rest-frame days (from t_ref; from the first spectrum when t_ref is unknown)
m['mjd']     # (N,)   observed MJD

lc = ds.total_flux_curve('SN2011fe')          # daily total flux (from photometry): mass target / ground truth at interpolated epochs
lc['mjd'], lc['mass_template'], lc['mass_template_err'], lc['mass_sed']
```

Numerical advice: masses are of order 1e-13 to 1e-9, so divide by a constant before running Sinkhorn
(for example the mean total mass of the two neighbouring spectra) and multiply back afterwards. A cost
matrix of `(x_i - x_j)^2` normalised by `M.max()` works. Entropic regularisation broadens spectral
features by roughly `sqrt(reg) × window width`; the median window width is 3755 Å, so `reg=1e-3`
corresponds to about 120 Å of broadening. For small reg use log-domain iterations
(`uot_barycenter` in `scripts/validate_dataset.py` is a ready-to-run implementation).

### Files

| File | Contents |
|---|---|
| `processed/sn_spectra_ot.h5` | **Main dataset** (about 131 MB; not in git, build it locally) |
| `processed/summary_sne.csv` | One row per SN: z, t_ref, spectrum counts, comparison with the paper, common window |
| `processed/summary_spectra.csv` | One row per input spectrum (`origin` = OSC or CSP), including rejected ones, with the metrics behind each rejection |
| `processed/build_log.txt` | Rejected photometry sources, how t_ref was determined, etc. |
| `processed/validation/` | Validation report `validation_report.md` and figures |
| `raw/osc/` | Raw Open Supernova Catalog JSON (about 716 MB; can be re-downloaded with the script) |
| `raw/csp/Gutierrez_et_al_2017/` | CSP SN II spectra (FITS; 8.7 MB tarball; CC-BY) |
| `meta/paper_table_C1.csv` | The paper's Table C.1: sample list, number of spectra, phase range, photometric bands |
| `meta/filters/` | Filter curves (Bessell BVRI, SDSS gri, Swift UVOT b/v) and the CALSPEC Vega spectrum |

### HDF5 layout

```
/wave_rest (1400,)  /wave_edges (1401,)     3000–10000 Å rest frame, 5 Å bins
/filters/<band>                             (λ, T); attrs: F0, ab_minus_vega, wave_eff, system
attrs: bands = "B,V,R,I,g,r,i,b_uvot,v_uvot"   ← column order of every per-band array
/sne/<SN>/
    attrs: type, redshift, ebv_mw, ra, dec, t_ref, t_ref_method, window_rest(lo,hi),
           paper_n_spec, paper_phase_range, paper_phot_bands, osc_sources,
           n_input_spectra (OSC+CSP), n_csp_spectra ...
    a_lambda_mw (1400,)                     Milky Way extinction A_λ (F99, R_V=3.1); not applied to the data
    spectra/  (N = calibrated, non-duplicate spectra; sorted by MJD)
        flux (N,1400)       f_λ [erg s⁻¹ cm⁻² Å⁻¹], rest frame; NaN outside the coverage; may contain a few negative values
        flux_err (N,1400)   only when the original data have errors, otherwise NaN
        flags (N,1400)      bit flags: 1=spike replaced, 2=telluric band bridged, 4=narrow host line removed
        mjd, phase          phase = (MJD - t_ref)/(1+z)
        golden              MAD(K)/K̃ < 10% with at least 2 bands (the paper's golden-sample criterion)
        ot_ready            golden, and fully covers this SN's common window window_rest
        mass_window         ∫f dλ over the common window [erg s⁻¹ cm⁻²]  ← total flux at each epoch
        mass_full           ∫f dλ over the covered range (coverage differs between spectra, so not directly comparable)
        mass_sed            total flux in the same window estimated from photometry alone (cross-check)
        K_tilde, K_mad_rel, n_bands, K_band_over_K_tilde (N,9)
        F_obs_band, F_obs_err_band, F_syn_band (N,9)   observed flux in each band at that epoch / synthetic flux of the calibrated spectrum
        origin (OSC/CSP), snr, instrument, filename, source, u_fluxes, osc_index (-1 for CSP),
        cov_lo, cov_hi, rebin_int_ratio
        native/<i>/         cleaned golden spectra at their original resolution (observed frame, calibrated):
                            wave_obs, flux_obs, err_obs, flags
    photometry/   mjd, band, mag, mag_err, err_assumed, system, source, used_in_fit
    lightcurve/   daily: mjd, phase, mag (T,9), mag_err, n_bands,
                  mass_template, mass_template_err, mass_sed, dt_nearest_spec
```

### Processing pipeline

1. **Download.** The OSC website is offline, so the backup in the `astrocatalogs/sne-*` repositories
   on GitHub is used. All 141 SNe of the paper were found.
   The OSC splits ASASSN-14lp into two entries that differ only in letter case; they are merged on
   loading (15 extra spectra).
   The OSC lacks the spectra of several SNe II (SN1986L, SN2003T, SN2003bn, SN2002hx, SN2002gw, ...),
   and WISeREP does not have them either: three of them are not in its database, and for the others it
   only holds copies imported from the OSC. These spectra were originally released on the CSP website
   (`Gutierrez_et_al_2017.tar.gz`), which covers 14 SNe II of this sample with 150 spectra, 75 of whose
   epochs are not in the OSC. They are read by `snprep/csp.py`:
   - Old FITS headers contain illegal keywords and non-ASCII characters, so the header cards are parsed by hand.
   - Dates come from MJD-OBS/JD first, then DATE-OBS+UT. Old files are ambiguous between dd/mm/yy and
     mm/dd/yy; the reading that agrees with the date in the file name is used.
   - The blue and red arms of the same night are joined into one spectrum, averaging the overlap.
   - Cross-correlating with the same spectrum in the OSC gives a 0 Å shift, confirming the observed frame.
     Copies of OSC spectra are removed by the deduplication step.
2. **Spectrum cleaning** (paper Appendix A)
   - Sort, drop non-finite and duplicate pixels, apply the OSC `exclude` ranges, and drop pixels with
     absurd values (|f| > 1000 × median).
   - Spectra that were already de-redshifted are first transformed back to the observed frame.
   - Trim noisy or junk ends: 25 consecutive pixels must have S/N > 1, positive flux, and a level below
     20 times that of the central part of the spectrum.
   - Cosmic rays: the Whitaker–Hayes modified z-score of the first difference used by the paper's CRD
     (threshold 3.5, m=3). A pixel is replaced only if it deviates from a 7-pixel running median by more
     than 5σ, by more than 2% of the local flux, and belongs to a run of at most 3 pixels, which avoids
     damaging real steep features. Replacement values are neighbourhood means. The median number of
     replaced pixels per golden spectrum is 9.
   - The atmospheric O₂ B band (6860–6950 Å) and A band (7590–7700 Å, observed frame) are bridged linearly.
   - Narrow host-galaxy emission lines (Hα, [N II], [S II], [O III], Hβ, [O II], positioned at the SN
     redshift) are removed only if narrower than 15 Å and more than 5σ above the continuum. The broad Hα
     of SNe II is not affected.
   - Deduplication: the OSC often lists the same spectrum from several sources, with the date rounded.
     The criterion is |Δt| < 0.75 d and a flux correlation coefficient > 0.9995.
3. **Photometry.** Only BVRI (Vega), gri (AB) and Swift UVOT b/v (Vega) are kept. Four SNe Ia have
   UVOT photometry only, as in the paper; UVOT data use the UVOT filter curves and are not treated as
   Johnson B/V. Upper limits, host-contaminated photometry and model points are removed, as are isolated
   jumps.
   **Per-source consistency check:** each round drops the source with the largest offset from the other
   sources, if that offset exceeds 0.25 mag, until no such source is left (28 groups in total; see
   build_log). For example, the Faran+2014 data of SN2004et are about 1 mag too bright.
   BVRI data labelled "AB" in the OSC (4 SNe, from PESSTO and LCO) were shown by synthetic photometry
   to be Vega magnitudes, so they are not converted.
4. **Light-curve interpolation.** Following the paper's ALR, an error-weighted local quadratic LOESS is
   used. The smoothing window is chosen by leave-one-out cross-validation; every window holds at least 5
   distinct epochs; outliers are rejected iteratively, but points next to large gaps are protected.
   Interpolated values are only used where sampling is adequate: data within 10 d (Ia) or 20 d (II) on
   both sides.
5. **Calibration** (paper Eqs. 1–2). For every band with ≥ 95% of its response covered,
   K_x = F_syn/F_obs is computed. K̃ = median(K), `flux = f / K̃`; a spectrum is golden when
   MAD(K)/K̃ < 10%. Zero points are computed self-consistently from CALSPEC Vega (0.03 mag in BVRI) and
   the AB definition; they agree with the paper's Table B.1 to within 1–4% (see §1 of the validation report).
6. **Phase reference.** SNe Ia: V-band maximum from a quadratic fit around the peak. SNe II: t_PT from a
   fit of the Olivares+2010 model. If the drop at the end of the plateau was not sampled (for example
   because it falls in an observing gap), t_PT is left empty.
7. **Resampling.** Spectra are first moved to the rest frame λ/(1+z) with f_rest = f_obs·(1+z), so that
   the integral equals the observed total flux, and then resampled onto 5 Å bins with exact flux
   conservation (integrals conserved to 0.02%).
8. **Common window.** For each SN, the rest-frame window that maximises (number of golden spectra fully
   covering it) × (window width) is chosen (width at least 2000 Å; median 3755 Å). Golden spectra that
   fully cover the window are `ot_ready`.
9. **Daily total flux** (only given within ±5 d of the time span covered by the OT-ready spectra):
   - `mass_template`: the nearest OT-ready spectra on either side of t are rescaled to the brightness at
     t by the median of the per-band photometric ratios (the paper's K̃ logic) and averaged with weights
     based on time distance; `mass_template_err` is half the difference between the two estimates.
   - `mass_sed`: photometry only, a trapezoidal integral of the band fluxes placed at their effective
     wavelengths.

### Validation results (details in `processed/validation/validation_report.md`)

- Sample: SNe Ia have 2109 golden / 1853 OT-ready spectra (paper golden: 1714); SNe II have 441 / 361
  (paper: 298). With the CSP spectra added, the golden phase range of SN1986L, [−106.7, −74.0], matches
  the paper's [−107, −74]; for SN2003bn it is [−105.5, −16.7] versus the paper's [−103, −16].
  The median number of OT-ready spectra per SN is 11. The 9 golden spectra of SN2006cp include all
  8 observed epochs of the paper's Fig. 6.
- Calibration: in the leave-one-band-out test, the fraction with |F_syn/F_obs − 1| < 10% is at least the
  paper's Fig. 8 value in every band (e.g. 91% for V in SNe Ia versus the paper's 85%), except the i and I
  bands of SNe II, which are slightly lower (81% versus 85%, and 79% versus 81%). The g band has a −3.4%
  systematic offset, comparable to the difference between the SDSS and USNO g′ filter curves.
- Total flux:
  - The ratio of integrals after/before resampling is 1.0002.
  - Photometry-only SED integral / spectral integral: 1.015±0.035 for SNe Ia and 0.986±0.040 for SNe II.
  - Leave-one-out test of `mass_template` (predicting a spectrum's total flux from its neighbouring
    spectra plus photometry): median error 1.5%, 90th percentile 5.7%.
- Usability: a smoke test with a log-domain unbalanced Sinkhorn barycenter on 133 SNe runs without NaNs
  or negative values. Its parameters were not tuned, so the results say nothing about how good the
  method is.

### Known limitations

- SN2007il has only 2 golden spectra (5 in the paper). It has 23 input spectra, but the photometry is
  too sparse to calibrate most epochs.
- SN2002gw has 6 golden spectra (9 in the paper) and SN2003T has 2 (3 in the paper).
- t_PT could not be measured reliably for 9 SNe II; their `phase` is NaN, so use MJD for interpolation.
- The I and g bands have the largest calibration scatter (natural systems do not match the standard
  filter curves; the paper sees the same). The golden criterion uses the median over all bands, so the
  impact is limited.
- Narrow interstellar absorption lines such as Na D are smoothed out as spikes when they meet the spike
  criterion (flag bit 1).
