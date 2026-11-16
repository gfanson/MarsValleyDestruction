"""
Reconstruction of the pre-impact valley depth distribution on Mars.

Run from this folder:
    python run_reconstruction.py

The settings below are those of the paper. The example transition tables in
inputs/transition_tables are the tables that ../Cratering_Sim/run_cratering.py
writes with its default settings (5 runs on the example depth map). To use
tables from your own cratering runs, set 'transition_dir' below to the
cratering output folder, e.g. "../Cratering_Sim/outputs/global".
"""

import glob
import os
import re

import numpy as np
import pandas as pd

from reconstruction_functions import (load_observed_data, load_seed_tables,
                                      pooled_transition_matrix, run_pooled_bootstrap,
                                      convergence_summary, fluid_volume,
                                      global_equivalent_layer, plot_bootstrap_envelope,
                                      plot_residuals)

# ==========================================
# INPUTS AND OUTPUTS
# ==========================================
# 'global' : all valley networks (main result)
# 'goudge' : valleys within +/-30 deg latitude on Hesperian or older terrain,
#            used for the flood-canyon fraction (Goudge et al. 2021)
CALCULATION = 'global'

CALCULATIONS = {
    'global': {'observed_file': "inputs/mars_depth_distribution.csv",
               'transition_dir': "inputs/transition_tables/global",
               'pixel_area_m2': 193325.0},      # 463 m x 417 m
    'goudge': {'observed_file': "inputs/mars_valley_depth_distribution_30_hesperian.csv",
               'transition_dir': "/Users/gracefanson/Documents/GitHub/MarsValleyDestruction/Cratering_Sim/outputs/goudge",
               'pixel_area_m2': 207172.0},      # 463 m x 447 m
}
TRANSITION_PATTERN = "transition_matrix_*.csv"
OUTPUT_DIR = os.path.join("outputs", CALCULATION)

# ==========================================
# TRANSITION TABLE
# ==========================================
MAX_DEPTH = 773            # deepest valley pixel on Kohala (m)
WINDOW_FRAC = 0.2          # each row built from depths within +/-20%
WINDOW_MIN_M = 1.0         # smallest window half-width (m)
MIN_PIXELS_PER_SEED = 3    # sparser depths take the nearest adequately populated row

# ==========================================
# INVERSE MODEL
# ==========================================
NUM_ANCHORS = 16           # requested; rounding to whole meters leaves 14
ANCHOR_EXPONENT = 3        # cubic spacing, denser at shallow depths
INTERP_KIND = 'cubic'
TOLERANCE = 1e-8
MAX_ITER = 2000
LOG_FLOOR = 0              # lower bound on anchor counts: 10**0 = 1 pixel
LOG_HEADROOM = 3.0         # upper bound: largest measured count x 10**3

# ==========================================
# UNCERTAINTY
# ==========================================
N_BOOT = 1000                                  # bootstrap resamples of all runs
CONVERGENCE_SIZES = [50, 100, 200, 500, 1000]  # numbers of runs for the convergence test
N_BOOT_CONVERGENCE = 200                       # resamples per convergence size
BOOT_RANDOM_SEED = 0
N_PROCESSES = 1                                # 1 runs the replicates one after another

# ==========================================
# VALLEY-CARVING WATER
# ==========================================
POROSITY = (0.2, 0.4)
FLUX_RATIO = (1500.0, 5400.0)  # fluid-to-sediment flux ratio, 25th-75th percentile


def seed_number(path):
    m = re.search(r"_(\d+)\.csv$", path)
    return int(m.group(1)) if m else path


def main():
    calc = CALCULATIONS[CALCULATION]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    config = dict(num_anchors=NUM_ANCHORS, anchor_exponent=ANCHOR_EXPONENT,
                  interp_kind=INTERP_KIND, tolerance=TOLERANCE, max_iter=MAX_ITER,
                  log_floor=LOG_FLOOR, log_headroom=LOG_HEADROOM,
                  pixel_area_m2=calc['pixel_area_m2'], min_pixels_per_seed=MIN_PIXELS_PER_SEED,
                  window_frac=WINDOW_FRAC, window_min_m=WINDOW_MIN_M)

    # 1. Measured distribution and transition tables
    observed_counts, min_fit_depth = load_observed_data(calc['observed_file'], MAX_DEPTH)
    files = sorted(glob.glob(os.path.join(calc['transition_dir'], TRANSITION_PATTERN)),
                   key=seed_number)
    if not files:
        raise SystemExit(f"No transition tables found in {calc['transition_dir']}")
    n = len(files)
    tables = load_seed_tables(files, MAX_DEPTH)
    _, filled = pooled_transition_matrix(tables, min_pixels_per_seed=MIN_PIXELS_PER_SEED,
                                         window_frac=WINDOW_FRAC, window_min_m=WINDOW_MIN_M)
    print(f"{n} transition tables from {calc['transition_dir']}; "
          f"{len(filled)} depths take the row of a nearby depth.")

    # 2. Reconstruction and bootstrap
    sizes = [m for m in CONVERGENCE_SIZES if m <= n]
    print(f"Running {N_BOOT} bootstrap and {len(sizes) * N_BOOT_CONVERGENCE} "
          f"convergence reconstructions...", flush=True)
    point, boot, boot_originals, conv = run_pooled_bootstrap(
        tables, observed_counts, min_fit_depth, config, n_boot=N_BOOT,
        convergence_sizes=sizes, n_boot_convergence=N_BOOT_CONVERGENCE,
        n_processes=N_PROCESSES, random_seed=BOOT_RANDOM_SEED)
    conv_table = convergence_summary(boot, conv, n, point['volumes']['lost_km3'])

    # 3. Outputs
    lo, hi = np.percentile(boot_originals, [2.5, 97.5], axis=0)
    pd.DataFrame({
        'Depth_m': point['depths'],
        'Measured_Post_Impact': observed_counts,
        'Pre_Impact': point['original'],
        'Pre_Impact_CI95_Low': lo,
        'Pre_Impact_CI95_High': hi,
        'Modeled_Post_Impact': point['present'],
        'Residual': point['residuals'],
        'Destroyed_Fraction': point['destroyed_fraction'],
    }).to_csv(os.path.join(OUTPUT_DIR, "depth_distribution.csv"), index=False)
    boot.to_csv(os.path.join(OUTPUT_DIR, "bootstrap_replicates.csv"), index=False)
    conv_table.to_csv(os.path.join(OUTPUT_DIR, "bootstrap_convergence.csv"), index=False)
    plot_bootstrap_envelope(point, boot_originals, observed_counts, min_fit_depth, OUTPUT_DIR)
    plot_residuals(point, min_fit_depth, OUTPUT_DIR)

    # 4. Summary
    v = point['volumes']
    o_lo, o_hi = np.percentile(boot['original_km3'], [2.5, 97.5]) * 1e9
    p_lo, p_hi = np.percentile(boot['percent_decrease'], [2.5, 97.5])
    pre_min = fluid_volume(o_lo, POROSITY[1], FLUX_RATIO[0])
    pre_max = fluid_volume(o_hi, POROSITY[0], FLUX_RATIO[1])
    post_min = fluid_volume(v['present_km3'] * 1e9, POROSITY[1], FLUX_RATIO[0])
    post_max = fluid_volume(v['present_km3'] * 1e9, POROSITY[0], FLUX_RATIO[1])
    converged = int(point['success']) + int(boot['success'].sum())
    pinned = point['n_pinned'] + int(boot['n_pinned'].sum())

    summary = f"""
Calculation:            {CALCULATION}
Simulation runs pooled: {n}
Bootstrap resamples:    {N_BOOT}
Anchors used:           {point['num_anchors']}
Optimizer converged:    {converged} / {N_BOOT + 1}
Anchors at a bound:     {pinned}

Eroded volume (95% bootstrap interval)
  Pre-impact:           {v['original_km3'] * 1e9:.3e} m^3  ({o_lo:.3e} - {o_hi:.3e})
  Post-impact:          {v['present_km3'] * 1e9:.3e} m^3  (measured)
  Decrease:             {v['percent_decrease']:.2f}%  ({p_lo:.2f} - {p_hi:.2f}%)

Water required to carve the valleys
  Pre-impact volume:    {pre_min:.2e} - {pre_max:.2e} m^3  ({global_equivalent_layer(pre_min):,.0f} - {global_equivalent_layer(pre_max):,.0f} m GEL)
  Post-impact volume:   {post_min:.2e} - {post_max:.2e} m^3  ({global_equivalent_layer(post_min):,.0f} - {global_equivalent_layer(post_max):,.0f} m GEL)

Convergence: 95% interval of the percent decrease against the number of runs
""" + "\n".join(f"  {int(r.n_seeds_used):5d} runs: {r.percent_decrease_ci_low:.2f} - "
                f"{r.percent_decrease_ci_high:.2f}%" for r in conv_table.itertuples())

    print(summary)
    with open(os.path.join(OUTPUT_DIR, "summary.txt"), "w") as f:
        f.write(summary.lstrip() + "\n")
    print(f"\nOutputs written to '{OUTPUT_DIR}/'.")


if __name__ == "__main__":
    main()
