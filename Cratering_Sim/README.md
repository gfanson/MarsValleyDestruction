# Impact Cratering Simulation

## Overview

This folder contains the impact cratering simulation of Fanson et al. (2026). The simulation takes a valley depth map of a fluvially dissected surface, emplaces the crater population expected over a chosen time window, and records which valley pixels are partially filled by ejecta and which are destroyed. Its main output is a **transition table** for each simulation run: the number of pixels that move from each pre-impact depth to each post-impact depth. The tables are written both at the resolution of the depth map and on the MOLA grid, where they are used by the reconstruction in `../Depth_Reconstruction`.

## Requirements

Python 3.9 or later with `numpy`, `pandas` and `rasterio`:

```bash
pip install numpy pandas rasterio
```

or

```bash
conda install -c conda-forge numpy pandas rasterio
```

## Quick start

From this folder:

```bash
python run_cratering.py
```

This runs five simulations (seeds 1144–1148) on `inputs/sample_depth_map.tif` (a 4.3 km × 4.8 km section of Kohala at 10 m/pixel) with the settings of the paper and writes the results to `outputs/`. The tables in `outputs/global` and `outputs/goudge` are the example inputs of the reconstruction in `../Depth_Reconstruction`, which ships with a copy of them. Before the first run, the crater models are built and cached; for the largest craters this takes a few minutes. The example then takes several minutes in total on a laptop.

## Settings

All settings are at the top of `run_cratering.py`.

| Setting | Paper value | Description |
| --- | --- | --- |
| `DEPTH_MAP` | 3 m Kohala depth map | Valley depth map (GeoTIFF). Depths are positive; non-valley pixels are 0 or nodata. Pixels must be square and in meters. |
| `OUTPUT_DIR` | | Output folder. |
| `FIRST_SEED`, `N_RUNS` | 1144, 1000 | Run *i* uses seed `FIRST_SEED + i`. |
| `N_PROCESSES` | | Number of runs computed in parallel. 1 runs them one after another. |
| `T_START`, `T_END` | 3.5, 0.0 | Cratering window (Ga before present). |
| `D_MIN_KM` | `None` | Smallest crater simulated (km). `None` selects the smallest bin that is at least one pixel wide (3.9 m on a 3 m map). |
| `D_MAX_KM` | 222 | Largest crater simulated (km). Bins with a minimum diameter above this are skipped, so the largest bin kept starts at 181 km. `None` simulates every bin of Michael (2013). |
| `PAD_EXTENT_R` | 10 | Craters are placed up to this many crater radii outside the study area. |
| `EJECTA_EXTENT_R` | 3 | Ejecta are truncated at this many crater radii. |
| `FOOTPRINTS` | `global`, `goudge` | MOLA footprints at which transition tables are written (see below). |
| `SAVE_TIF` | | Write the post-impact depth map of each run. |
| `SAVE_COORDINATES`, `COORDINATES_MIN_KM` | | Write the centers of all craters of at least this diameter. |

## Outputs

| File | Description |
| --- | --- |
| `run_info.json` | Settings used. |
| `crater_models.csv` | Morphology of each simulated bin: diameter, model, h1, h2, m, n, the fraction of the excavated volume deposited within the ejecta extent, and the grid correction k. |
| `mean_crater_counts.csv` | Mean number of craters per bin in the study area and in the padded placement region, and the area of that region. |
| `run_summary.csv` | For each run: number of craters, pre- and post-impact valley volume, and percent decrease. |
| `crater_counts_<seed>.csv` | Number of craters placed per bin in this run, with the mean it was drawn from. |
| `transition_matrix_<seed>.csv` | Transition table at the resolution of the depth map. |
| `global/transition_matrix_<seed>.csv`, `goudge/…` | Transition tables on the MOLA grid. |
| `post_impact_<seed>.tif` | Post-impact depth map (if `SAVE_TIF`). Destroyed and non-valley pixels are nodata. |
| `crater_coordinates_<seed>.csv` | Crater centers in pixel coordinates of the depth map, with diameters (if `SAVE_COORDINATES`). |

Each transition table has four columns: `Pre-Impact Depth`, `Original Pixel Count` (all pixels at that pre-impact depth), `Post-Impact Depth` and `Pixel Count`. Depths are in 1 m bins centered on whole meters. Destroyed pixels are included in `Original Pixel Count` but in no `Pixel Count`, so the shortfall is the number destroyed.

## Method

**Crater population.** Crater production follows Michael (2013): diameter bins with minimum diameters of 2^(i/2) km, incremental crater numbers from Table 1, and the Hartmann chronology function (Eq. 3) scaled to the cratering window. Every crater in a bin has the bin's minimum diameter. The number of craters in each bin is drawn from a Poisson distribution, and their centers are distributed uniformly over the study area padded by `PAD_EXTENT_R` crater radii on each side, so that craters centered outside the study area can reach it. Each bin has its own random stream, seeded from the run seed and the bin index, so runs that differ only in `D_MAX_KM` or `EJECTA_EXTENT_R` share identical craters in their common bins. Craters are emplaced in order of increasing diameter.

**Crater morphology.** The radial profile follows Howard (2007, Eqs. 3–4): a bowl Δ*H* = (*h*₂ − *h*₁) + *h*₁ (*r*/*R*)^*m* for *r* ≤ *R*, and ejecta Δ*H* = *h*₂ (*r*/*R*)^−*n* for *r* > *R*. Craters smaller than 4 km use the depth, rim height and bowl-shape exponent of Watters et al. (2015), extrapolated below 50 m following Millot et al. (2025). Craters of 4 km and larger use Howard (2007), Table 1. The ejecta exponent *n* is chosen so that the ejecta volume, integrated to infinity, equals the excavated volume, and the ejecta are then truncated at `EJECTA_EXTENT_R` radii. Pixel values are averages of the profile over each pixel, sampled more finely near the rim. Because the pixelated profile does not conserve volume exactly, the ejecta pixels are scaled by a factor k so that the ejecta volume on the grid equals the deposited fraction of the bowl volume on the grid.

**Superposition.** Valley depths are treated as negative values and the crater profile is added to them. A valley pixel is destroyed if the crater bowl is deeper than the valley or if the ejecta fill it completely. Ejecta thinner than the valley depth fill it partially, so the valley becomes shallower. A bowl shallower than the valley leaves it unchanged.

**MOLA grid.** Transition tables depend on pixel size, so a table applied to Mars must describe MOLA-scale pixels. For each footprint, the pre- and post-impact depth maps are averaged onto a coarse grid with weights equal to the exact fractional overlap of fine and coarse cells, with destroyed pixels counted as depth 0. The footprints have the north–south size of a MOLA cell (463.08 m) and an east–west size that gives the depth-weighted mean cell area of the Mars valley pixels: 463.08 m × 417.47 m (`global`, all valley networks) and 463.08 m × 447.37 m (`goudge`, valleys within ±30° latitude on Hesperian or older terrain).

## Functions

`cratering_functions.py`:

* **tif_to_array**, **array_to_tif**: read and write GeoTIFFs.
* **prep_data**: reads the depth map and builds the crater production rates.
* **get_michael_crater_bins**, **age_scaling**, **calculate_crater_rates**: crater production (Michael, 2013).
* **smallest_resolved_diameter**, **simulated_bins**: select the bins to simulate.
* **mean_crater_counts**: mean number of craters of each size over the study area.
* **placement_region**: padded region over which craters of one size are placed, and its area relative to the study area.
* **draw_crater_count**, **locgenerate**: number of craters in one run (Poisson draw) and their centers.
* **bowl_coefficients**, **ejecta_exponent**: crater morphology parameters.
* **build_crater**, **crater_table**: pixel-averaged crater models, cached per diameter.
* **boom**, **boom_clipped**: emplace craters on the depth map.
* **impact**: one simulation run.
* **transition_table**, **table_volumes**: transition table of a run and the valley volumes it implies.
* **overlap_weights**, **aggregate**, **aggregation_weights**: resampling to the MOLA grid.
* **load_setup**, **run_trial**, **run_trials**: run simulations and write outputs.

## Reproducing the paper

Set `DEPTH_MAP` to the 3 m Kohala depth map and `N_RUNS` to 1000. The 3 m map is 8,950 × 12,090 pixels and each run places about 4.6 × 10⁷ craters, so each run needs several gigabytes of memory; the runs in the paper were spread over a computing cluster. Seeds can be split freely between processes or machines, because each run depends only on its own seed.

Sensitivity tests use the same seeds with one setting changed: `EJECTA_EXTENT_R` = 2, 5 or 10 (100 runs each), and `D_MAX_KM` = 13.4 (half of the shorter side of the 3 m map, as in Howard, 2007; the largest bin kept starts at 11.3 km), 64, 128 or `None` (1000 runs each).

## References

Howard, A. D. Simulating the development of Martian highland landscapes through the interaction of impact cratering, fluvial erosion, and variable hydrologic forcing. *Geomorphology* **91**, 332–363 (2007).

Luo, W., Pingel, T., Heo, J., Howard, A. & Jung, J. A progressive black top hat transformation algorithm for estimating valley volumes on Mars. *Comput. Geosci.* **75**, 17–23 (2015).

Michael, G. G. Planetary surface dating from crater size–frequency distribution measurements: Multiple resurfacing episodes and differential isochron fitting. *Icarus* **226**, 885–890 (2013).

Millot, C., Quantin-Nataf, C., Dehouck, E., Torres, I. & Volat, M. Depth to diameter relationships for <50 m diameter Martian craters. *J. Geophys. Res. Planets* **130**, e2024JE008844 (2025).

Watters, W. A., Geiger, L. M., Fendrock, M. & Gibson, R. Morphometry of small recent impact craters on Mars: Size and terrain dependence, short-term modification. *J. Geophys. Res. Planets* **120**, 226–254 (2015).

## Citation

If you use this code, please cite:

> Fanson, G. O., Morgan, A. M. & Stucky de Quay, G. Quantifying the effects of impact cratering on the preservation of Martian valley networks. *npj Space Exploration* (2026). DOI: to be added.

## Contact

Grace Fanson, Massachusetts Institute of Technology (gfanson@mit.edu). The Gaia Lab: thegaialab.com.
