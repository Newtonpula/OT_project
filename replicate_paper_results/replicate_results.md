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
| `tests/test_otrep.py` | checks of `otrep` |
| `outputs/` | figures, tables and Σ(ε) planes written by the scripts |

The modules of `otrep/`:

| Module | Content |
|---|---|
| `data.py` | loads any dataset as unit-sum shapes `rho = mass / total`, plus `total` and the bin edges |
| `barycenter.py` | cost matrix and barycenter solvers |
| `interpolate.py` | α; OT, linear or geometric interpolation; rescaling with `total`; Eqs. 5–6 |
| `pairs.py` | Test 1 triplets and Fig. 4 windows |
| `sigma.py` | the Σ(ε) plane |
| `metrics.py` | Eq. 4 |
| `config.py` | grid, phase ranges, output names |
| `plotting.py` | shared figure style |

The scripts for the observed SNe will reuse `otrep` unchanged: `data.load('sn', ...)` returns the same keys as
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

### What this means for the balanced-vs-unbalanced comparison

- **The paper's advantage of OT comes from its settings.** With them, the "barycenter" is a pixel-wise geometric
  mean. Converged balanced OT, the textbook W2 barycenter, is worse than linear interpolation on SN spectra.
- **The balanced baseline must therefore be named explicitly.** Report both the authors' procedure, which
  reproduces the paper's numbers, and converged balanced OT (exact W2). Keep the geometric mean as a reference
  that involves no transport.
- **Unbalanced OT** can create and destroy mass locally instead of moving it. The next step is to test whether
  that keeps line features in place better than balanced OT does.
