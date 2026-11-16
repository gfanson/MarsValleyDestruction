# Reconstruction of Pre-Impact Valley Depths

## Overview

This folder contains the reconstruction of Fanson et al. (2026). It estimates the frequency distribution of Martian valley depths before 3.5 billion years of impact cratering, given the present-day (post-impact) distribution measured on Mars and the transition tables produced by the cratering simulation in `../Cratering_Sim`. From the reconstructed distribution it computes the pre-impact eroded volume of the valley networks, the percent decrease in eroded volume, and the volume of water required to carve the valleys.

## Requirements

Python 3.9 or later with `numpy`, `pandas`, `scipy` and `matplotlib`:

```bash
pip install numpy pandas scipy matplotlib
```

The optional functions that prepare inputs from a Mars depth raster (see *Functions*) also need `rasterio`, and `geopandas` if a polygon mask is used.

## Quick start

From this folder:

```bash
python run_reconstruction.py
```

This reconstructs the global distribution from the example transition tables in `inputs/transition_tables/global` and writes the results to `outputs/global/`. With the default 1000 bootstrap resamples the example takes about 15 minutes on one processor core; `N_PROCESSES` spreads the resamples over several cores. For the paper's 1000 runs, the convergence test adds a further 1000 reconstructions.

The example tables are the output of the cratering example: running `python run_cratering.py` in `../Cratering_Sim` with its default settings writes the same tables to `../Cratering_Sim/outputs/global` and `outputs/goudge` (5 runs, seeds 1144–1148, on the 4.3 km × 4.8 km example depth map). To reconstruct from tables you have generated, set `transition_dir` in `CALCULATIONS` to that folder.

The example demonstrates the workflow and does not reproduce the results of the paper, which pools 1000 runs on the 3 m depth map of the whole Kohala peninsula (973.8 km²). The example depth map covers only about 100 MOLA-scale pixels, with depths up to 394 m, so most rows of the transition matrix are taken from nearby depths. With `CALCULATION = 'goudge'`, some anchors reach the bounds of the optimizer and the interval of the percent decrease includes zero; both are consequences of the small example area.

## How it works

**Transition matrix.** A transition table counts, for each pre-impact depth *i*, the pixels that end at each post-impact depth *j*. The tables of all simulation runs are summed and each row is divided by the number of pixels that started at depth *i*, giving the probability *T*[*i*, *j*]. The shortfall of each row below 1 is the fraction of pixels destroyed. Aggregation to MOLA resolution leaves few pixels at each depth, so each row is built from all pixels within ±20% of its depth (`WINDOW_FRAC`), weighted by distance and shifted so that each pixel loses the same number of meters. Depths with fewer than 3 pixels per run (`MIN_PIXELS_PER_SEED`) take the row of the nearest adequately populated depth, shifted in the same way.

**Inverse model.** For a pre-impact distribution **p** (pixel counts per 1 m depth bin), the modeled post-impact distribution is **p**·*T*. The pre-impact distribution is described by 14 anchor depths spaced cubically between 0 and 773 m, so that shallow depths are resolved more finely, and interpolated with a cubic spline in log space. Starting from the measured distribution, the anchor values are adjusted with L-BFGS-B (Byrd et al., 1995) to minimize the sum of squared differences between the modeled and measured post-impact distributions.

**Volumes.** The eroded volume is *V* = Σ *A* *d*ᵢ *f*ᵢ, where *f*ᵢ is the number of pixels at depth *d*ᵢ and *A* is the depth-weighted mean area of a valley pixel: 193,325 m² for all valley networks and 207,172 m² for the Goudge et al. (2021) subset. The water volume follows Rosenberg et al. (2019): *V*f = *V* (1 − λ) *Q*f/*Q*s, evaluated for porosities λ of 0.2–0.4 and fluid-to-sediment flux ratios of 1,500–5,400, and is also reported as a global equivalent layer (GEL).

**Uncertainty.** The point estimate pools all runs. Its 95% interval comes from 1000 bootstrap resamples, in which runs are drawn with replacement and the reconstruction is repeated. The convergence test repeats the bootstrap with subsets of 50 to 1000 runs.

## Settings

All settings are at the top of `run_reconstruction.py`; the values shown are those of the paper.

| Setting | Value | Description |
| --- | --- | --- |
| `CALCULATION` | `global` | `global` for all valley networks; `goudge` for valleys within ±30° latitude on Hesperian or older terrain (Goudge et al., 2021). Selects the measured distribution, the transition tables and the pixel area. |
| `TRANSITION_PATTERN` | `transition_matrix_*.csv` | File names of the transition tables. |
| `MAX_DEPTH` | 773 | Deepest depth bin (m), the deepest valley pixel on Kohala. |
| `WINDOW_FRAC`, `WINDOW_MIN_M` | 0.2, 1 | Half-width of the depth window used to build each row (fraction of depth, minimum in m). 0 turns the window off. |
| `MIN_PIXELS_PER_SEED` | 3 | Depths with fewer pixels per run take the row of a nearby depth. |
| `NUM_ANCHORS`, `ANCHOR_EXPONENT` | 16, 3 | Requested anchors and spacing exponent. Rounding to whole meters and removing depth 0 leaves 14 anchors. |
| `INTERP_KIND` | `cubic` | Interpolation between anchors. |
| `TOLERANCE`, `MAX_ITER` | 1e-8, 2000 | Optimizer settings. |
| `LOG_FLOOR`, `LOG_HEADROOM` | 0, 3 | Bounds on the anchor values in log10 counts. The upper bound only keeps the optimizer finite; the summary reports any anchor that ends at a bound. |
| `N_BOOT`, `BOOT_RANDOM_SEED` | 1000, 0 | Bootstrap resamples and their random seed. |
| `CONVERGENCE_SIZES`, `N_BOOT_CONVERGENCE` | 50–1000, 200 | Numbers of runs, and resamples per number, for the convergence test. Sizes larger than the number of tables are skipped. |
| `N_PROCESSES` | 1 | Bootstrap replicates computed in parallel. |
| `POROSITY`, `FLUX_RATIO` | 0.2–0.4, 1,500–5,400 | Ranges used for the water volume. |

## Inputs

| File | Description |
| --- | --- |
| `inputs/mars_depth_distribution.csv` | Measured valley depths on Mars (Goudge, 2021, Texas Data Repository): pixel counts per 1 m depth bin, columns `Depth` and `Count`. |
| `inputs/mars_valley_depth_distribution_30_hesperian.csv` | The same for valleys within ±30° latitude on Hesperian or older terrain. |
| `inputs/transition_tables/global/`, `…/goudge/` | Transition tables at the two MOLA footprints, as written by the cratering example to `../Cratering_Sim/outputs/global` and `outputs/goudge`. |

## Outputs

Written to `outputs/<CALCULATION>/`:

| File | Description |
| --- | --- |
| `summary.txt` | Pre- and post-impact eroded volume, percent decrease with its 95% interval, water volume and GEL, and the convergence test. |
| `depth_distribution.csv` | Per depth: measured post-impact count, reconstructed pre-impact count with its 95% interval, modeled post-impact count, residual, and destroyed fraction. |
| `bootstrap_replicates.csv` | Volumes and percent decrease of each bootstrap replicate. |
| `bootstrap_convergence.csv` | 95% interval of the percent decrease and lost volume against the number of runs. |
| `reconstruction.png` | Measured, reconstructed pre-impact and modeled post-impact distributions. |
| `residuals.png` | Modeled minus measured post-impact counts. |

## Functions

`reconstruction_functions.py`:

* **load_observed_data**: reads a measured depth distribution.
* **load_seed_tables**: reads the transition tables of all runs.
* **smooth_counts**, **shift_row**, **normalize_transition_counts**, **pooled_transition_matrix**: build the transition matrix.
* **build_anchors**, **interpolate_distribution_log**, **calculate_error**, **reconstruct**: the inverse model and volumes.
* **run_pooled_bootstrap**, **convergence_summary**: bootstrap and convergence test.
* **fluid_volume**, **global_equivalent_layer**: water required to carve the valleys.
* **plot_bootstrap_envelope**, **plot_residuals**: figures.
* **depth_distribution_from_raster**, **depth_weighted_pixel_area**: prepare the measured distribution and the pixel area *A* from a Mars valley depth raster in an equidistant cylindrical projection, optionally limited to a latitude band and a polygon mask (e.g. the Hesperian-or-older terrain of Goudge et al., 2021).

## References

Byrd, R. H., Lu, P., Nocedal, J. & Zhu, C. A limited memory algorithm for bound constrained optimization. *SIAM J. Sci. Comput.* **16**, 1190–1208 (1995).

Goudge, T. A. The importance of lake breach floods for valley incision on early Mars. Texas Data Repository, https://doi.org/10.18738/T8/STRFZH (2021).

Goudge, T. A., Morgan, A. M., Stucky de Quay, G. & Fassett, C. I. The importance of lake breach floods for valley incision on early Mars. *Nature* **597**, 645–649 (2021).

Rosenberg, E. N., Palumbo, A. M., Cassanelli, J. P., Head, J. W. & Weiss, D. K. The volume of water required to carve the Martian valley networks: Improved constraints using updated methods. *Icarus* **317**, 379–387 (2019).

## Citation

If you use this code, please cite:

> Fanson, G. O., Morgan, A. M. & Stucky de Quay, G. Quantifying the effects of impact cratering on the preservation of Martian valley networks. *npj Space Exploration* (2026). DOI: to be added.

## Contact

Grace Fanson, Massachusetts Institute of Technology (gfanson@mit.edu). The Gaia Lab: thegaialab.com.
