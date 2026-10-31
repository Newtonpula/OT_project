# SN spectral time-series dataset (for unbalanced-OT interpolation)

This is a rebuild of the observational sample used by Ramirez et al. 2024 (A&A, arXiv:2409.10701):
**141 supernovae (110 Ia + 31 II, i.e. the paper's Table C.1)**. The data come from the Open Supernova
Catalog (OSC); 14 SNe II are supplemented with the CSP SN II spectra (Gutiérrez et al. 2017). The spectra
are flux-calibrated in absolute terms against photometry, resampled onto a common rest-frame wavelength
grid, and come with the **total flux (mass) at every epoch**.
The spectra are not normalised, so the integral of the spectrum can differ from epoch to epoch, which
is exactly the setting of unbalanced OT, where the measures have unequal masses.

## Quick start

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

## Files

| File | Contents |
|---|---|
| `processed/sn_spectra_ot.h5` | **Main dataset** (about 128 MB) |
| `processed/summary_sne.csv` | One row per SN: z, t_ref, spectrum counts, comparison with the paper, common window |
| `processed/summary_spectra.csv` | One row per input spectrum (`origin` = OSC or CSP), including rejected ones, with the metrics behind each rejection |
| `processed/build_log.txt` | Rejected photometry sources, how t_ref was determined, etc. |
| `processed/validation/` | Validation report `validation_report.md` and figures |
| `raw/osc/` | Raw Open Supernova Catalog JSON (about 716 MB; can be re-downloaded with the script) |
| `raw/csp/Gutierrez_et_al_2017/` | CSP SN II spectra (FITS; 8.7 MB tarball; CC-BY) |
| `meta/paper_table_C1.csv` | The paper's Table C.1: sample list, number of spectra, phase range, photometric bands |
| `meta/filters/` | Filter curves (Bessell BVRI, SDSS gri, Swift UVOT b/v) and the CALSPEC Vega spectrum |

To regenerate (about 20 s for all 141 SNe; validation takes about 1 minute):

```bash
.venv/bin/python scripts/download_data.py
```
```bash
.venv/bin/python scripts/build_dataset.py
```
```bash
.venv/bin/python scripts/validate_dataset.py
```

## HDF5 layout

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

## Processing pipeline

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

## Validation results (details in `processed/validation/validation_report.md`)

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

## Known limitations

- SN2007il has only 2 golden spectra (5 in the paper). It has 23 input spectra, but the photometry is
  too sparse to calibrate most epochs.
- SN2002gw has 6 golden spectra (9 in the paper) and SN2003T has 2 (3 in the paper).
- t_PT could not be measured reliably for 9 SNe II; their `phase` is NaN, so use MJD for interpolation.
- The Dessart+2013/2014 model spectra of the paper's §4.1 (used for method tests) are not included:
  neither the Zenodo API nor the author's website was reachable from the network used here.
- The I and g bands have the largest calibration scatter (natural systems do not match the standard
  filter curves; the paper sees the same). The golden criterion uses the median over all bands, so the
  impact is limited.
- Narrow interstellar absorption lines such as Na D are smoothed out as spikes when they meet the spike
  criterion (flag bit 1).
