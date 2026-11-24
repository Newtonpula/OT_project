# Methods of Ramirez et al. (2024)

A short summary of the methods in *A novel optimal transport-based approach for interpolating spectral time
series* (M. Ramirez et al., A&A 2024, arXiv:2409.10701v2; the PDF is in the repository root). It keeps only what
is needed to replicate the paper's OT interpolation tests. These serve as the balanced-OT baseline for our
comparison of balanced and unbalanced OT. The paper's photometric calibration and photometric validation are
left out.

Section, equation and figure numbers refer to the paper. §6 lists where we depart from the paper, and §7 the
details the paper does not state, with the choices we made. Code, how to run it, and results are in
[replicate_results.md](replicate_results.md).

## 1. Overview

The paper interpolates supernova spectra in time with an entropic Wasserstein barycenter (OT):

1. Every pair of spectra that brackets a target epoch is interpolated with OT, with a weight α set by the phases.
2. Each interpolated spectrum is rescaled to an absolute flux.
3. All interpolations for the epoch are averaged, giving less weight to pairs with a large phase gap.

The OT step is tested on black bodies and on noiseless model spectra, with linear (L2) interpolation as the
baseline. It is then used to build daily spectral time series of observed supernovae.

## 2. Interpolating two spectra with OT (Sect. 3)

- **Solver:** the POT library (Flamary et al. 2021), `ot.bregman.barycenter(A, M, reg, weights)`.
  - `A` holds the two spectra as columns, as distributions over the $n$ wavelength bins.
  - `M` is the normalised $n \times n$ loss matrix, built with a POT `ot.utils` function.
  - `reg` is the entropic regularisation.
  - `weights` is $[1-\alpha,\ \alpha]$.
  - The authors' code also sets `method='sinkhorn', numItermax=10` and scans `reg` (§7). The result is then
    close to a pixel-wise geometric mean of the two spectra rather than a transport of mass.
- **Weight α:** for spectra at phases $ph_a < ph_b$ and a target phase $ph$,
  $\alpha = (ph - ph_a)/(ph_b - ph_a)$. For example, $ph_a = 2$ and $ph_b = 6$ give α = 0.25, 0.5 and 0.75 at
  $ph$ = 3, 4 and 5.
- **Phases:** SNe Ia are counted in days from the V-band maximum, found with a second-order polynomial fitted
  around the peak. SNe II are counted in days from $t_\mathrm{PT}$, the midpoint of the transition from plateau
  to tail (Olivares et al. 2010). In our datasets these phases are the `t` returned by `ot_measures()`.
- **Baseline:** linear interpolation $(1-\alpha)\,f_a + \alpha\,f_b$, which the paper calls the "L2" or
  Euclidean barycenter.
- **Flux scale:** the barycenter gives only the shape of the spectrum. The paper sets the absolute flux from
  photometry, with $\tilde K$, the median ratio of synthetic to observed band fluxes. We use `total` instead (§6).
- **Fig. 1 (1D toy example):** the L2 barycenter of two distributions is bimodal, while the Wasserstein
  barycenter is a single shape shifted between them. The lower panel shows the Wasserstein path as α goes from
  0 to 1.
- **Fig. 2 (black bodies):**
  - The interpolation runs between Planck spectra at 5000 K and 10000 K, with $B_\lambda$ in
    erg cm⁻² s⁻¹ Å⁻¹ sr⁻¹ over about 0–15000 Å.
  - The OT path (top panel) and the L2 path (bottom panel) are drawn over the true black bodies at
    intermediate temperatures.
  - The OT path follows the thermal evolution; the L2 path does not.

## 3. Test 1: single pairs of model spectra (Sect. 4.1, Fig. 3)

- **Data:** the noiseless CMFGEN spectral time series of Dessart et al. (2014) for SNe Ia and of Dessart et
  al. (2013) for SNe II.
- **Which triplets:** all triplets of model epochs $ph_1 < ph_2 < ph_3$ with $0.45 \le \alpha \le 0.55$, since
  α ≈ 0.5 is the hardest case. The paper describes covering all pairs on a moving grid, with the gap between
  the pair growing step by step.
- **Steps for each triplet:**
  1. Interpolate the spectrum at $ph_2$ from the spectra at $ph_1$ and $ph_3$, with OT and, for comparison,
     linearly.
  2. Rescale it to the brightness at $ph_2$. The paper uses $\tilde K$ against the synthetic photometry of the
     true model at $ph_2$; we use the `total` of that spectrum.
  3. Compare it with the true spectrum at $ph_2$ using the mean relative spectral residual (Eq. 4), given below.

$$\epsilon = \frac{1}{n}\sum_{\lambda=\lambda_0}^{n}\frac{\lvert f^{M}_{\lambda} - f^{I}_{\lambda}\rvert}{f^{M}_{\lambda}}$$

Here $f^M$ is the model flux, $f^I$ the interpolated flux and $n$ the number of wavelength bins.

- **Output (Fig. 3):**
  - ε is plotted against $\Delta ph = ph_3 - ph_1$, in one panel per SN type, with linear in blue and OT in red.
  - Marker size and darkness increase with the phase of the interpolated spectrum.
  - The colour bars run from −5 to 30 d (Ia) and from −120 to 0 d (II).
- **Uncertainty model:**
  - A plane is fitted in $(\Delta ph,\ ph,\ \epsilon)$ space, separately for SNe Ia and SNe II.
  - The plane gives $\Sigma(\epsilon)$, the expected error of an interpolation with gap $\Delta ph$ at phase $ph$.
  - Later tests use $\Sigma(\epsilon)$ to down-weight pairs that are far apart.
- **Reported results:**
  - The OT ε stays below 10% up to Δph = 40 d for SNe II and 25 d for SNe Ia. The Conclusions instead say below
    20% (Ia) and 10% (II) at 40 d.
  - The linear ε grows faster and exceeds 40% at large gaps.

## 4. Test 2: weighted mean over all pairs of four spectra (Sect. 4.1, Figs. 4–5)

- **Scheme (Fig. 4):**
  - Four model spectra are inputs (black boxes); the spectra between them are withheld as targets (white boxes).
  - Each target is interpolated from every input pair that brackets it.
  - The four cycles drawn in Fig. 4 use spectra A–E, plus F in cycle 4:

    | Cycle | Target | Inputs | Pairs used | Interpolations |
    |---|---|---|---|---|
    | 1 | B | A, C, D, E | AC, AD, AE | 3 |
    | 2 | C | A, B, D, E | AD, AE, BD, BE | 4 |
    | 3 | D | A, B, C, E | AE, BE, CE | 3 |
    | 4 | B and C | A, D, E, F | AD, AE, AF (for each target) | 3 per target |

  - In cycle 4 the grid is extended by one spectrum (F), and two adjacent spectra (B and C) are withheld
    together.
- **Combination:**
  - Each interpolation $i$ is rescaled to the brightness of the target, and the final spectrum is the weighted
    mean of Eq. 5.
  - The paper weights each interpolation by $1/[\mathrm{MAD}(K)_i^2 + \Sigma(\epsilon)_i^2]$, where MAD(K) is its
    photometric calibration error. Without photometry we keep only the $\Sigma(\epsilon)$ term, as shown below.
  - Linear interpolation goes through the same scheme, so both methods use the same information.

$$f(\lambda) = \frac{\sum_{i=1}^{n} f_i(\lambda)\,/\,\Sigma(\epsilon)_i^2}{\sum_{i=1}^{n} 1\,/\,\Sigma(\epsilon)_i^2}$$

- **Output (Fig. 5):** ε of the final spectrum against Δph, here the largest gap among the pairs that enter the mean.
- **Reported results:**
  - At the same Δph, ε is larger than in Test 1, because pairs with wider gaps enter the mean. It still stays
    below 20% (Ia) and 10% (II).
  - The authors argue that averaging matters for real spectra, because it reduces the effect of noise and
    miscalibration.

## 5. Daily series of observed supernovae (Sect. 4.2, Fig. 6)

For each SN, on a daily grid of epochs:

1. **Pairs:** take every pair of golden spectra that brackets the epoch and is less than 40 d apart. There is no
   limit of four spectra here. In our data the golden spectra are the `ot_ready` spectra returned by
   `ot_measures()`.
2. **Each interpolation:** OT-interpolate the pair to the epoch, and take Σ(ε) from the plane of Test 1.
3. **Edge taper:** the edges of spectra are usually poorly calibrated, so a weight $C_i(\lambda)$ falls
   linearly from 1 to 0 within 50 Å of each end of interpolated spectrum $i$. In our data every `ot_ready`
   spectrum covers the SN's whole common window. All $C_i$ are then equal and cancel in Eq. 6, so the taper can
   be skipped.
4. **Combination:** Eq. 6, again without the MAD(K) term:

$$f(\lambda) = \frac{\sum_{i} f_i(\lambda)\,C_i(\lambda)\,/\,\Sigma(\epsilon)_i^2}{\sum_{i} C_i(\lambda)\,/\,\Sigma(\epsilon)_i^2}$$

**Examples (Fig. 6).** The OT series is drawn at integer days as normalised flux with vertical offsets, so only
the shapes are needed. This matters because most days have no spectrum and therefore no `total`.

- SN 2006cp (Ia): 8 observed spectra, from −11.75 to +8.19 d.
- SN 2006be (II): 4 observed spectra, from −72.06 to −39.06 d.

## 6. Where we depart from the paper

Our goal is to compare balanced and unbalanced OT, so the photometric steps of the paper are dropped:

- **OT input:** each spectrum is normalised to unit sum, `rho = mass / total[:, None]`. The paper does not state
  its normalisation.
- **Flux scale:** interpolated spectra are rescaled with `total` instead of $\tilde K$. In Tests 1–2 this is
  the total of the true model spectrum at the target epoch.
- **Weights:** Eqs. 5–6 use $1/\Sigma(\epsilon)_i^2$ only; the paper's MAD(K) term needs photometry.
- **Σ(ε):** the plane is refitted to the ε values obtained with `total` rescaling.
- **Not reproduced:** the photometric validation of Sect. 4.2. This covers the residuals φ and Φ (Eqs. 7–8),
  the leave-one-out test, and Figs. 7, 8, C.1 and C.2.

## 7. Not stated in the paper, and our choices

- **OT settings.** The paper gives none, but the authors' public code does
  (github.com/mramirezz/STS_OT, `Codes/Funciones_colab.ipynb`, functions `OT` and `best_reg`):
  - `M = ot.utils.dist0(n)`, then `M /= M.max()`: the squared distance between pixel indices. On our uniform
    grid this is the same as our `cost_matrix`.
  - `ot.bregman.barycenter(A, M, reg, [1 - α, α], method='sinkhorn', numItermax=10)`: at most **10**
    iterations.
  - `reg` is scanned upward over `5 * 10**np.arange(-7, -1, 0.1)`, and the first value whose result has no NaN
    or inf is kept. For every black body and model triplet, the first value, 5e-7, already works.
  - With such a narrow kernel and so few iterations, almost no mass moves. The result is within a few tenths of
    a percent of the pixel-wise weighted geometric mean $\rho_a^{1-\alpha}\rho_b^{\alpha}$ (replicate_results.md, Results).
  - Our solver `paper` reproduces these settings (code: `otrep/barycenter.py`). For comparison we also run:
    - converged OT: the exact W2 barycenter (`exact`: quantile interpolation, no regularisation);
    - POT's `sinkhorn_log` at reg = 1e-3;
    - POT's default `sinkhorn` run to convergence. Its unweighted starting scalings pull barycenters with
      α ≠ 0.5 toward the α = 0.5 one.
  - The authors' code was found after this summary was first written. The paper text itself states none of
    these settings.
- **Weighted mean.** The authors' notebook `4 - Weighted Mean OT .ipynb` clips Σ(ε) below at 0.01 and tapers
  50 pixels at each spectrum edge (the paper says 50 Å). We use the same floor.
- **Linear baseline:** the paper does not say whether it mixes raw or normalised spectra. We mix the unit-sum
  shapes, $(1-\alpha)\rho_a + \alpha\rho_b$, and rescale with `total`, so that both methods get the same inputs.
- **Black-body test:**
  - The temperatures of the intermediate curves are not given; only 5000 K and 10000 K are labelled. Our
    dataset uses 10 temperatures equally spaced in T.
  - How α maps to temperature is not given. We evaluate α linear in T (temperature as time) and α linear in
    1/T.
  - The W2 geodesic between two black bodies consists of black bodies whose 1/T is linear in α. Normalised
    black bodies are rescaled copies of one distribution in λT, so their quantile functions scale as 1/T.
- **Models:** the paper does not list them. We use all 16 SN Ia and 6 SN II models on Zenodo, with phases −5 to
  30 d and −120 to 0 d (the colour bars of Fig. 3), and pair gaps of at most 45 d (the x axis of Fig. 3).
- **ε:** the wavelength range over which Eq. 4 is evaluated is not given. We use 3000–10000 Å (the SN grid) in
  every test.
- **The plane Σ(ε):**
  - Its exact form is not given. We fit $\epsilon = a\,\Delta ph + b\,ph + c$ by least squares, with ph the
    target's phase.
  - It is not said which points are fitted. We fit each method and SN type separately, and in Test 2 each method
    uses its own plane.
  - How it is evaluated for observed SNe outside the model phase range is still open, for example for SNe II long
    after $t_\mathrm{PT}$.
- **Test 2:** Fig. 4 draws only four cycles. We slide windows of W = 5, 6, … consecutive spectra:
  - each window has 4 inputs and a contiguous block of W − 4 withheld targets after p = 1, 2 or 3 inputs;
  - Fig. 4 shows W = 5 with p = 1–3, and W = 6 with p = 1;
  - W grows while the widest pair spans at most 45 d.

## 8. Paper parts and local data

| Paper | Content | Local data (see `data/data_description.md`) |
|---|---|---|
| Sect. 3, Fig. 1 | 1D toy example, L2 vs. Wasserstein | none (synthetic distributions) |
| Sect. 3, Fig. 2 | Black-body interpolation | `data/processed/blackbody_spectra_ot.h5` |
| Sect. 4.1, Figs. 3–5 | Tests 1–2 on CMFGEN models | `data/processed/model_spectra_ot.h5` |
| Sect. 4.2, Fig. 6 | Daily series of observed SNe | `data/processed/sn_spectra_ot.h5` |
