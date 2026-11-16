"""
Impact cratering simulation on a valley depth map.

Run from this folder:
    python run_cratering.py

The settings below are those of the paper, applied to a small example depth
map. To reproduce the paper, set DEPTH_MAP to the 3 m Kohala depth map and
N_RUNS to 1000.
"""

from cratering_functions import run_trials

# ==========================================
# INPUTS AND OUTPUTS
# ==========================================
DEPTH_MAP = "inputs/sample_depth_map.tif"   # valley depths (positive), square pixels in meters
OUTPUT_DIR = "outputs"

# ==========================================
# SIMULATION RUNS
# ==========================================
FIRST_SEED = 1144          # seeds are FIRST_SEED, FIRST_SEED + 1, ...
N_RUNS = 5                 # paper: 1000 (seeds 1144-2143)
N_PROCESSES = 1            # runs in parallel; 1 runs them one after another

# ==========================================
# CRATER POPULATION
# ==========================================
T_START = 3.5              # start of the cratering window (Ga): the age of the valleys
T_END = 0.0                # end of the cratering window (Ga); 0 = present
D_MIN_KM = None            # smallest crater (km); None = smallest bin at least one pixel wide
D_MAX_KM = 222.0           # largest crater (km): the largest crater formed since 3.5 Ga
                           # (Lyot; Robbins et al. 2013). Each bin is represented by its
                           # minimum diameter, so the largest crater simulated is 181 km.
                           # None simulates every bin of Michael (2013).
PAD_EXTENT_R = 10.0        # craters are placed up to this many radii outside the study area
                           # (at least EJECTA_EXTENT_R). 10 is the largest ejecta extent
                           # tested in the paper, so runs with different EJECTA_EXTENT_R
                           # place identical craters.

# ==========================================
# CRATER MORPHOLOGY
# ==========================================
EJECTA_EXTENT_R = 3.0      # outer edge of the ejecta deposit (crater radii); the excavated
                           # volume that would land beyond it is not deposited

# ==========================================
# OUTPUT OPTIONS
# ==========================================
FOOTPRINTS = ['global', 'goudge']   # MOLA footprints for the transition tables:
                                    # 'global' = all valley networks;
                                    # 'goudge' = valleys within +/-30 deg latitude on
                                    # Hesperian or older terrain (Goudge et al. 2021)
SAVE_TIF = True                     # post-impact depth map for each run (large at 3 m;
                                    # the paper saved them only for the drainage-density runs)
SAVE_COORDINATES = False            # crater centers for each run
COORDINATES_MIN_KM = 0.1            # smallest crater whose center is saved (km)


if __name__ == "__main__":
    summary = run_trials(
        DEPTH_MAP, range(FIRST_SEED, FIRST_SEED + N_RUNS), OUTPUT_DIR,
        t_start=T_START, t_end=T_END, d_min_km=D_MIN_KM, d_max_km=D_MAX_KM,
        ejecta_extent_r=EJECTA_EXTENT_R, pad_extent_r=PAD_EXTENT_R,
        footprints=FOOTPRINTS, save_tif=SAVE_TIF,
        save_coordinates=SAVE_COORDINATES, coordinates_min_km=COORDINATES_MIN_KM,
        n_processes=N_PROCESSES)

    d = summary['percent_decrease']
    print(f"\nValley volume decrease over {len(summary)} run(s): "
          f"mean {d.mean():.2f}%, median {d.median():.2f}%")
    print(f"Outputs written to '{OUTPUT_DIR}/'.")