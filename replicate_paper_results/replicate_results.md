# Replicating Ramirez et al. (2024)

This folder replicates the paper's interpolation tests as the balanced-OT baseline for our comparison of balanced
and unbalanced OT. [methods.md](methods.md) summarises the paper's methods, where we depart from them, and the
settings the paper leaves open. The authors' public code fixes several of those settings
(github.com/mramirezz/STS_OT).

## Layout

| Path | Content |
|---|---|
| `otrep/` | code shared by every test, described below |
| `fig2_blackbody.py` | Fig. 2: black bodies |
| `fig3_model_pairs.py` | Test 1, Fig. 3: single pairs of model spectra; fits the Σ(ε) planes |
| `fig5_model_windows.py` | Test 2, Figs. 4–5: weighted mean over the pairs of four model spectra |
| `fig6_observed.py` | Test 3 on the observed SNe: leave-one-out check, and the daily series of Fig. 6 |
| `tests/test_otrep.py` | checks of `otrep` |
| `outputs/` | figures, tables and Σ(ε) planes written by the scripts |

The modules of `otrep/`:

| Module | Content |
|---|---|
| `data.py` | loads any dataset as unit-sum shapes `rho = mass / total`, plus `total` and the bin edges |
| `barycenter.py` | cost matrix and barycenter solvers |
| `interpolate.py` | α; OT, linear or geometric interpolation; rescaling with `total`; Eqs. 5–6 |
| `pairs.py` | Test 1 triplets, Fig. 4 windows, Test 3 bracketing pairs |
| `sigma.py` | the Σ(ε) plane |
| `metrics.py` | Eq. 4 |
| `config.py` | grid, phase ranges, output names |
| `plotting.py` | shared figure style |

The script for the observed SNe uses the same modules: `data.load('sn', ...)` returns the same keys as
`data.load('model', ...)`.

### Interpolation methods

Every test runs the same set of methods. Each result is a unit-sum shape, rescaled with the true `total`.

| Key | Method |
|---|---|
| `linear` | (1 − α) ρ_a + α ρ_b, the paper's L2 barycenter |
| `geometric` | ρ_a^(1−α) ρ_b^α, normalised: a pixel-wise weighted geometric mean |
| `paper` | the authors' code: POT `ot.bregman.barycenter`, `method='sinkhorn'`, `numItermax=10`, first finite reg of 5 × 10^(−7 … −1.1) |
| `exact` | converged balanced OT: the exact W2 barycenter of the 1D histograms (quantile interpolation, no regularisation) |
| `sinkhorn_log` | converged entropic OT, POT's `sinkhorn_log`, reg = 1e-3 |
| `sinkhorn` | POT's default solver run to convergence (not run by default) |

## Running

Run the commands from the repository root.

**Python environment:** the scripts need numpy, h5py and matplotlib. POT is optional:

- If POT is installed, `otrep` calls `ot.bregman.barycenter` directly.
- Otherwise it uses a NumPy port of the same code. When POT is available, `tests/test_otrep.py` checks the port
  against it.

On this machine the project's `.venv` has everything: Python 3.13, NumPy 2, and astropy and extinction for the SN
data, but no POT. It was set up as in [data/data_description.md](../data/data_description.md#building-the-h5-files-locally).
Activate it first:

```bash
source .venv/bin/activate
```

Build the data once, as described in `data/data_description.md`:

```bash
python scripts/build_blackbody.py
```

```bash
python scripts/download_models.py
```

```bash
python scripts/build_models.py
```

```bash
python scripts/download_data.py
```

```bash
python scripts/build_dataset.py
```

The data pipeline needs NumPy 2 (`np.trapezoid`). The replication scripts also run with NumPy 1.26.

Check the shared code:

```bash
python replicate_paper_results/tests/test_otrep.py
```

Fig. 2 (about 1 minute per command on 16 cores):

```bash
python replicate_paper_results/fig2_blackbody.py
```

```bash
python replicate_paper_results/fig2_blackbody.py --alpha invT
```

Test 1 (Fig. 3; about 1.5 minutes, mostly `sinkhorn_log`). Run it before Test 2, which reads its Σ(ε) planes:

```bash
python replicate_paper_results/fig3_model_pairs.py
```

Test 2 (Fig. 5; under a minute):

```bash
python replicate_paper_results/fig5_model_windows.py
```

Test 3 on the observed SNe (Fig. 6 and the leave-one-out check; a few minutes). It also reads the Test 1 planes:

```bash
python replicate_paper_results/fig6_observed.py
```

Common options:

- `--solvers` picks the OT methods.
- `--reg` sets the regularisation of the converged entropic solvers (default 1e-3).
- `--rebin k` merges k bins, for speed.
- `--backend auto|pot|numpy`.

## Results

### Fig. 2: black bodies

The 5000 K and 10000 K Planck spectra are interpolated as unit-sum shapes. Each result is rescaled with the true
total of the black body it is compared with.

| Output | What it shows |
|---|---|
| `fig2_blackbody_<method>_alpha-<map>_reg1e-03.png` | the two panels of Fig. 2 |
| `fig2_checks_reg1e-03.png` / `.csv` | ε (Eq. 4) against temperature in 50 K steps |

**The paper's qualitative result is reproduced by every OT variant.** The OT curves are single-peaked,
black-body-like shapes whose peak moves smoothly from 5800 Å to 2900 Å. The linear curves are mixtures: they keep
the 10000 K peak and gain a red shoulder.

**ε (Eq. 4) over 3000–10000 Å.** Each cell gives the maximum over the intermediate temperatures, and in brackets the
values at 5000 K and 10000 K:

| Method | α linear in T | α linear in 1/T |
|---|---|---|
| Linear | 14.4% (0, 0) | 10.2% (0, 0) |
| Geometric mean | 15.2% (0, 0) | 1.1% (0, 0) |
| Authors' settings (`paper`) | 15.2% (0, 0) | 1.1% (0, 0) |
| Exact W2 | 13.9% (0, 0) | 2.8% (0, 0) |
| `sinkhorn_log`, reg = 1e-3 | 14.4% (0.6%, 0.7%) | 2.3% (0.6%, 0.7%) |

**What the numbers show:**

1. **The temperature parametrisation decides the comparison.**
   - The W2 geodesic between two black bodies consists of black bodies whose 1/T is linear in α (methods.md, §7).
   - With α linear in T, every method is compared at the wrong temperature and ε reaches 14–15%. At α = 0.5 the
     OT curve is a ~6667 K black body, compared with the 7500 K one.
   - In the later tests α is linear in time, so the time parametrisation can matter as much as the method.
2. **With α linear in 1/T, all OT variants nearly reproduce black bodies, and the authors' settings do best.**
   - Their result is the geometric mean, which in the Wien regime is exactly a Wien spectrum at the harmonic
     temperature. A point-wise mean is also unaffected by the edges of the grid.
   - Exact W2 is exact only on an infinite domain. The 1000–15000 Å grid misses 17% of the 5000 K spectrum's mass.
   - `sinkhorn_log` adds entropic blur. Its end points are off by 0.6–0.7%, the blur of a single spectrum.
3. **Smaller reg does not converge.** At reg = 1e-4, `sinkhorn_log` does not converge within POT's default 10,000
   iterations, and the unconverged result is worse: max ε 17.9%.
4. **POT's default solver run to convergence (`sinkhorn`, not in the default run) is biased for α ≠ 0.5.** At the
   end points it returns spectra 1.2% and 2.0% away from the inputs.

### Test 1 (Fig. 3): single pairs of model spectra

**Setup**
- Models: all 16 SN Ia models (phases −5 to 30 d) and all 6 SN II models (−120 to 0 d).
- Grid: 3000–10000 Å in 5 Å bins.
- Triplets: every triplet with 0.45 ≤ α ≤ 0.55 and a gap of at most 45 d, which gives 644 SN Ia and 458 SN II
  triplets.
- Each interpolation is rescaled with the true total at ph2.

| Output | What it shows |
|---|---|
| `fig3_models_<method>_reg1e-03.png` | Fig. 3: linear against one OT method |
| `fig3_models_reg1e-03.csv` | ε of every method for every triplet |
| `sigma_plane_reg1e-03.json` | the Σ(ε) planes, read by Test 2 |

**Median ε per 5-day bin of the gap:**

| Gap | Ia linear | Ia authors' | Ia exact W2 | II linear | II authors' | II exact W2 |
|---|---|---|---|---|---|---|
| 10 d | 6.2% | 5.8% | 29.2% | 1.4% | 1.3% | 16.8% |
| 20 d | 19.1% | 17.0% | 45.1% | 4.4% | 2.9% | 22.7% |
| 25 d | 21.8% | 20.5% | 48.8% | | | |
| 30 d | 27.2% | 24.3% | 48.7% | 9.3% | 5.9% | 28.7% |
| 40 d | | | | 11.2% | 7.4% | 31.7% |

The maxima are 49.7%, 41.5% and 67.7% for SN Ia, and 60.5%, 23.4% and 46.5% for SN II (linear, authors' settings,
exact W2). `sinkhorn_log` at reg = 1e-3 has median ε 35.5% for SN Ia and 26.4% for SN II.

**What the numbers show:**

1. **With the authors' settings, the paper's comparison holds qualitatively.** Their "OT" error grows more slowly with
   the gap than the linear error:
   - clearly for SNe II: 7.4% against 11.2% at 40 d, with a maximum of 23% against 60%;
   - only slightly for SNe Ia: 20.5% against 21.8% at 25 d.
2. **Quantitatively, SN II matches the paper and SN Ia does not.**
   - SN II: the median stays below 10% up to 40 d, as the paper states.
   - SN Ia: the median at 25 d is 20.5%, twice the paper's "below 10% up to 25 d".
   - Linear interpolation exceeds 40% at the largest gaps for both types, as in the paper.
3. **The authors' "OT" is the geometric mean.**
   - Over all 1102 triplets, its ε differs from the geometric mean's by a median of 0.03 and at most 0.58
     percentage points.
   - Their reg scan keeps its first value, 5e-7, for every triplet.
4. **Converged balanced OT is much worse than linear interpolation at every gap.**
   - Exact W2 already reaches a median of 29% at 10 d for SN Ia, against 6% for linear.
   - The reason: 1D transport follows the quantiles of the whole spectrum. When the continuum changes, line features
     are carried to the wrong wavelengths.
   - Entropic OT adds blur on top.
5. **Σ(ε) planes for the authors' settings** (ph is the target's phase):
   - SN Ia: ε = 0.0101 Δph − 0.0013 ph − 0.021 (R² 0.81).
   - SN II: ε = 0.0027 Δph − 0.0002 ph − 0.032 (R² 0.50).

### Test 2 (Figs. 4–5): weighted mean over the pairs of four spectra

**Setup**
- The Fig. 4 windows slide along each model and grow until the widest pair spans 45 d. This gives 6735 SN Ia and
  4557 SN II targets, computed from 6801 unique pair interpolations.
- The interpolations of each method are averaged with weights 1/Σ(ε)², from that method's own Test 1 plane.
  Σ(ε) is floored at 0.01, as in the authors' code.

**Median ε per 5-day bin of the largest gap in the mean:**

| Largest gap | Ia linear | Ia authors' | Ia exact W2 | II linear | II authors' | II exact W2 |
|---|---|---|---|---|---|---|
| 10 d | 3.3% | 3.0% | 20.3% | 1.6% | 1.4% | 9.8% |
| 20 d | 11.5% | 10.3% | 35.9% | 4.7% | 2.7% | 17.5% |
| 30 d | 18.6% | 16.4% | 41.4% | 6.4% | 4.3% | 22.4% |
| 40 d | | | | 7.7% | 5.4% | 25.3% |

**What the numbers show:**

1. **The paper's statement is reproduced with the authors' settings.** The medians stay below 20% for SNe Ia
   (16.4% at 30 d) and below 10% for SNe II (5.4% at 40 d).
2. **Averaging helps.**
   - At the same largest gap, ε is lower than in Test 1 (SN Ia: 16.4% against 24.3% at 30 d), because narrower
     pairs also enter the mean.
   - The paper says the opposite: residuals larger than in Fig. 3 at a given Δph. That does not fit its own
     definition of Δph as the largest gap in the mean.
3. **Exact W2 stays far worse than linear interpolation:** 41% against 19% for SN Ia at 30 d.

### Test 3 (Sect. 4.2, Fig. 6): observed supernovae

Only `mass` and `total` are used, so the paper's photometric validation (Figs. 7, 8, C.1, C.2) is replaced by a
leave-one-out check (methods.md, §5).

**Setup**
- Sample: 132 SNe with interior OT-ready spectra. This gives 1864 withheld spectra: 1602 SN Ia spectra from 105
  SNe and 262 SN II spectra from 27 SNe.
- Each withheld spectrum is interpolated from the others in two ways:
  - **single:** from the tightest bracketing pair;
  - **weighted:** from all bracketing pairs spanning less than 40 d, with weights 1/Σ(ε)² from each method's model
    plane.
- In total this is 226,788 pair interpolations, which take about 2 minutes.

| Output | What it shows |
|---|---|
| `test3_loo_reg1e-03.csv` | ε of every method, single and weighted, for every withheld spectrum |
| `test3_loo_<single\|weighted>_<paper\|exact>_reg1e-03.png` | ε against the gap of the tightest bracketing pair |
| `fig6_SN2006cp_reg1e-03.png`, `fig6_SN2006be_reg1e-03.png` | Fig. 6: daily series over the observed spectra, one column per method |

**Median ε by the gap of the tightest bracketing pair.** Each cell gives the single value / the weighted value:

| Gap | Ia linear | Ia authors' | Ia exact W2 | II linear | II authors' | II exact W2 |
|---|---|---|---|---|---|---|
| 0–5 d | 4.5 / 4.3% | 4.5 / 4.3% | 6.2 / 8.9% | 4.8 / 3.9% | 4.8 / 3.9% | 5.5 / 6.5% |
| 5–10 d | 5.6 / 6.9% | 5.4 / 7.0% | 9.9 / 14.2% | 4.9 / 4.7% | 4.6 / 4.5% | 8.4 / 11.0% |
| 10–20 d | 8.2 / 11.1% | 7.9 / 10.8% | 13.7 / 18.4% | 6.3 / 6.5% | 5.9 / 5.8% | 11.5 / 14.2% |
| 20–40 d | 12.3 / 14.8% | 12.0 / 14.4% | 24.5 / 26.1% | 9.9 / 10.0% | 9.4 / 9.6% | 20.0 / 20.0% |

The bins hold 1017, 230, 222 and 133 SN Ia spectra, and 108, 53, 50 and 51 SN II spectra. The geometric mean
behaves like the authors' settings.

**What the numbers show:**

1. **The ranking is the same as on the models.** Authors' settings ≈ geometric mean ≳ linear ≫ exact W2.
   - The authors' "OT" beats linear interpolation for 66–76% of the withheld spectra, but by only about 0.3
     percentage points in median.
   - Exact W2, the converged balanced OT, has about twice the error of linear interpolation at gaps above 5 d. It
     beats linear for only 13–15% of the withheld spectra.
2. **Most withheld spectra have close neighbours.** The median gap is 3.1 d for SNe Ia and 7.0 d for SNe II, so the
   medians sit near the noise floor of the observations, about 4–5%.
3. **The weighted mean helps only at gaps below 5 d**, where it averages out noise.
   - At larger gaps it is worse than the tightest pair: the Σ(ε) weights from the models do not down-weight wide
     pairs enough.
   - Fig. 6 shows this for SN2006be: just before −47.5 d, the daily spectra still look like the early ones.
4. **Fig. 6 is reproduced** for SN2006cp and SN2006be in the paper's format. All methods give smooth sequences;
   exact W2 visibly reshapes features, for example near 6100 Å in SN2006cp.

### What this means for the balanced-vs-unbalanced comparison

- **The paper's advantage of OT comes from its settings.** With them, the "barycenter" is a pixel-wise geometric
  mean. Converged balanced OT, the textbook W2 barycenter, is worse than linear interpolation on SN spectra.
- **The balanced baseline must therefore be named explicitly.** Report both the authors' procedure, which
  reproduces the paper's numbers, and converged balanced OT (exact W2). Keep the geometric mean as a reference
  that involves no transport.
- **Unbalanced OT** can create and destroy mass locally instead of moving it. The next step is to test whether
  that keeps line features in place better than balanced OT does.
