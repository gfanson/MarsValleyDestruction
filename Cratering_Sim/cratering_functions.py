"""
Impact cratering simulation on a valley depth map.

Sign conventions
----------------
Input raster : valley depth as positive values; non-valley pixels are 0 or nodata.
Working map  : valley depth as negative values below a datum of 0, so that a
               crater profile (negative bowl, positive rim and ejecta) can be
               superposed by addition.
Output       : positive depths, as in the input. NaN marks a destroyed pixel.

Workflow
--------
1. prep_data reads the depth map and builds the crater production rates.
2. impact draws the crater population for one simulation run and emplaces it.
3. transition_table records how each pre-impact depth changed.
4. aggregate resamples the depth maps to the MOLA grid before the table is
   built, so that the table can be applied to Mars depth distributions.
run_trial and run_trials combine these steps and write the outputs.
"""

import json
import os

import numpy as np
import pandas as pd
import rasterio


# =============================================================================
# Raster input and output
# =============================================================================

def tif_to_array(file):
    """
    Read the first band of a GeoTIFF.

    Parameters
    ----------
    file : str
        Path to the GeoTIFF.

    Returns
    -------
    array : numpy.ndarray
        Raster values as float32, with nodata and NaN set to 0.
    res : tuple
        Pixel size (x, y) in map units.
    """
    with rasterio.open(file) as ds:
        band = ds.read(1, masked=False)
        nodata = ds.nodata
        res = ds.res

    array = band.copy().astype(np.float32)
    if nodata is not None:
        nd_mask = np.isnan(band) if np.isnan(nodata) else (band == nodata)
        array[nd_mask] = 0.0
    array[np.isnan(array)] = 0.0
    return array, res


def array_to_tif(source, array, outputfile):
    """
    Write a depth array to a GeoTIFF with the georeferencing of a source raster.

    Destroyed (NaN) pixels are written as NaN nodata, as are non-valley (0)
    pixels.

    Parameters
    ----------
    source : str
        Path to the raster whose transform and CRS are copied.
    array : numpy.ndarray
        Depth array with positive depths, same shape as the source.
    outputfile : str
        Path of the GeoTIFF to write.
    """
    with rasterio.open(source) as src:
        transform, shape, crs = src.transform, src.shape, src.crs

    array = array.copy().astype(np.float32)
    array[array == 0] = np.nan

    with rasterio.open(outputfile, 'w', driver='GTiff', height=shape[0],
                       width=shape[1], count=1, transform=transform,
                       dtype=rasterio.float32, crs=crs, nodata=np.nan) as dst:
        dst.write(array.reshape(shape), 1)


def prep_data(file, t_start, extent=None):
    """
    Load a depth map and build the crater production rates for its study area.

    Parameters
    ----------
    file : str
        Path to the depth map (positive depths, pixels in meters).
    t_start : float
        Start of the cratering window [Ga]; rates are tabulated from 0 to
        t_start.
    extent : tuple or None
        Study area (row_start, row_end, col_start, col_end) in pixels.
        None uses the full raster.

    Returns
    -------
    depth_map : numpy.ndarray
        Depth map with valleys negative (working convention).
    rates : pandas.DataFrame
        Craters formed per time step and diameter bin over the study area.
    sizes : dict
        Minimum diameter of each bin [km], keyed by bin index.
    res_m : float
        Pixel size [m].
    extent : tuple
        Study area actually used.
    sample_area : float
        Area of the study area [km^2].
    """
    raw, res = tif_to_array(file)

    if not np.isclose(res[0], abs(res[1]), rtol=1e-6):
        raise ValueError(f"Pixels are not square: {res}.")
    res_m = float(res[0])

    if np.nanmin(raw) < 0:
        raise ValueError("The depth map contains negative values. Depths must be "
                         "positive, with non-valley pixels set to 0 or nodata.")

    depth_map = -1.0 * raw

    if extent is None:
        extent = (0, raw.shape[0], 0, raw.shape[1])
    rows = extent[1] - extent[0]
    cols = extent[3] - extent[2]
    sample_area = (cols * res_m / 1000.0) * (rows * res_m / 1000.0)

    bin_idx, sizes, crat_num = get_michael_crater_bins()
    rates = calculate_crater_rates(t_start, sample_area, bin_idx, crat_num)

    return depth_map, rates, sizes, res_m, tuple(int(v) for v in extent), sample_area


# =============================================================================
# Crater population
# =============================================================================

def get_michael_crater_bins():
    """
    Crater size bins of Michael (2013), Table 1.

    Bin minima are D_min = 2**(i/2) km for i = -16 ... 19. The crater numbers
    are the incremental counts per km^2 per Gyr (column H of the table), based
    on the Hartmann (2004) production function. Their sum for D >= 1 km is
    5.84e-4, the normalization used in age_scaling.

    Returns
    -------
    bin_idx : numpy.ndarray
        Bin indices i.
    sizes : dict
        Minimum diameter of each bin [km].
    crat_num : dict
        Incremental crater number of each bin [km^-2 Gyr^-1].
    """
    bin_idx = np.arange(-16, 19 + 1)
    d_min = 2.0 ** (bin_idx / 2.0)

    N = [4.04e3, 2.33e3, 1.14e3, 4.58e2, 1.91e2, 6.66e1, 2.40e1, 9.44e0, 3.30e0,
         1.22e0, 4.37e-1, 1.47e-1, 4.70e-2, 1.38e-2, 4.02e-3, 1.15e-3, 3.08e-4,
         1.28e-4, 6.85e-5, 3.67e-5, 1.98e-5, 1.06e-5, 5.68e-6, 3.04e-6, 1.62e-6,
         8.71e-7, 4.67e-7, 2.40e-7, 1.12e-7, 5.21e-8, 2.43e-8, 1.13e-8, 5.28e-9,
         2.47e-9, 1.15e-9, 5.37e-10]

    return bin_idx, dict(zip(bin_idx, d_min)), dict(zip(bin_idx, N))


def age_scaling(gya):
    """
    Cumulative crater density at age gya relative to that at 1 Ga.

    Uses the Hartmann chronology function of Michael (2013), Eq. 3:
    N(t) = 3.79e-14 (exp(6.93 t) - 1) + 5.84e-4 t.

    Parameters
    ----------
    gya : float
        Age [Ga].

    Returns
    -------
    float
        N(gya) / N(1 Ga).
    """
    n_t = 3.79e-14 * (np.exp(6.93 * gya) - 1) + 5.84e-4 * gya
    n_1 = 3.79e-14 * (np.exp(6.93 * 1.0) - 1) + 5.84e-4 * 1.0
    return n_t / n_1


def calculate_crater_rates(max_age, sample_area, bin_idx, crat_num, t_step=0.1):
    """
    Number of craters formed in each time step and diameter bin.

    Each row label is the older edge of an age interval, and each value is the
    number of craters formed within that interval over sample_area. The count
    for a window t_end < t <= t_start is the sum of the rows in that window.

    Parameters
    ----------
    max_age : float
        Oldest age tabulated [Ga].
    sample_area : float
        Study area [km^2].
    bin_idx : numpy.ndarray
        Bin indices.
    crat_num : dict
        Incremental crater number per bin [km^-2 Gyr^-1].
    t_step : float
        Time step [Gyr].

    Returns
    -------
    pandas.DataFrame
        Index: age [Ga]; columns: bin index.
    """
    T = np.arange(t_step, max_age + t_step, t_step)
    rates = pd.DataFrame({'time': T}).round({'time': 4}).set_index('time')
    scal = np.array([age_scaling(t) for t in T])

    for b in bin_idx:
        cumulative = scal * crat_num[b] * sample_area
        rates[b] = np.diff(cumulative, prepend=0.0)
    return rates


def smallest_resolved_diameter(sizes, res_m):
    """
    Smallest bin minimum diameter [km] that is at least one pixel wide.

    For a 3 m depth map this is 3.9 m (bin i = -16); for a 10 m map, 11.0 m.
    """
    return min(d for d in sizes.values() if d * 1000.0 >= res_m * (1 - 1e-9))


def simulated_bins(sizes, d_min_km, d_max_km=None):
    """
    Bins whose minimum diameter lies within [d_min_km, d_max_km].

    Parameters
    ----------
    sizes : dict
        Minimum diameter of each bin [km].
    d_min_km : float
        Smallest crater simulated [km].
    d_max_km : float or None
        Largest crater simulated [km]; None keeps every larger bin.

    Returns
    -------
    list
        Bin indices in ascending order of diameter.
    """
    keep = []
    for b in sorted(sizes):
        d = sizes[b]
        if d < d_min_km * (1 - 1e-9):
            continue
        if d_max_km is not None and d > d_max_km:
            continue
        keep.append(b)
    return keep


def footprint_halfwidth(d_m, res_m, extent_r):
    """Half-width [px] of a square that holds a crater out to extent_r radii."""
    return int(np.ceil(extent_r * d_m / 2.0 / res_m + 0.5))


def mean_crater_counts(rates, bins, t_start, t_end):
    """
    Mean number of craters of each size formed over the study area.

    The mean is the sum of the production rates of calculate_crater_rates over
    the time window t_end < t <= t_start. It is the expected value of the
    number of craters; the number in a given simulation run is drawn from a
    Poisson distribution with this mean (see draw_crater_count).

    Parameters
    ----------
    rates : pandas.DataFrame
        Output of calculate_crater_rates.
    bins : list
        Bins to simulate (see simulated_bins).
    t_start, t_end : float
        Time window [Ga]; t_start is the older bound.

    Returns
    -------
    dict
        Mean number of craters over the study area, keyed by bin index.
    """
    eps = 1e-9
    in_window = (rates.index > t_end + eps) & (rates.index <= t_start + eps)
    return {b: float(rates.loc[in_window, b].sum()) for b in bins}


def placement_region(extent, d_km, res_m, pad_extent_r):
    """
    Region over which craters of one size are placed.

    Craters are placed over the study area padded on every side by
    pad_extent_r crater radii, so that craters centered outside the study area
    whose bowl or ejecta reach it are included. The padding grows with crater
    diameter. Because the areal crater density is the same everywhere, the
    mean number of craters in the padded region is the mean over the study
    area multiplied by area_ratio.

    Parameters
    ----------
    extent : tuple
        Study area (row_start, row_end, col_start, col_end) in pixels.
    d_km : float
        Crater diameter [km].
    res_m : float
        Pixel size [m].
    pad_extent_r : float
        Padding in crater radii.

    Returns
    -------
    region : tuple
        Padded region (row_start, row_end, col_start, col_end) in pixels; it
        extends beyond the raster.
    area_ratio : float
        Area of the padded region divided by the area of the study area.
    """
    r0, r1, c0, c1 = extent
    rows, cols = r1 - r0, c1 - c0
    W = footprint_halfwidth(d_km * 1000.0, res_m, pad_extent_r)
    area_ratio = (rows + 2 * W) * (cols + 2 * W) / (rows * cols)
    return (r0 - W, r1 + W, c0 - W, c1 + W), area_ratio


def draw_crater_count(mean, rng):
    """
    Number of craters in one simulation run.

    Crater formation is treated as a Poisson process, so the number of craters
    of a given size is drawn from a Poisson distribution with the given mean.
    Run-to-run differences in crater numbers come from this draw.

    Parameters
    ----------
    mean : float
        Mean number of craters in the placement region.
    rng : numpy.random.Generator

    Returns
    -------
    int
    """
    return int(rng.poisson(mean))


def locgenerate(region, n, rng):
    """
    Uniformly distributed crater centers.

    Parameters
    ----------
    region : tuple
        (row_start, row_end, col_start, col_end) in pixels; may extend beyond
        the raster.
    n : int
        Number of centers.
    rng : numpy.random.Generator

    Returns
    -------
    numpy.ndarray
        (n, 2) array of (row, col).
    """
    row_start, row_end, col_start, col_end = region
    rows = rng.integers(row_start, row_end, size=int(n))
    cols = rng.integers(col_start, col_end, size=int(n))
    return np.array([rows, cols]).T


# =============================================================================
# Crater morphology
# =============================================================================
#
# Radial profile relative to the far-field datum (Howard 2007, Eqs. 3-4):
#
#   bowl   (r <= R):                      dH = (h2 - h1) + h1 (r / R)**m
#   ejecta (R < r <= ejecta_extent_r R):  dH = h2 (r / R)**(-n)
#
# h1 is the rim-to-floor depth, h2 the rim height above the far field, m the
# bowl-shape exponent and n the ejecta exponent. ejecta_extent_r is the outer
# edge of the ejecta deposit, in crater radii.
#
#   D <  4 km : Watters et al. (2015). h1 from Eq. 1 (D > 50 m), h2 from Eq. 9,
#               and m from Table 3 (a4) for moderately modified craters. Below
#               50 m the depth law is extrapolated, following Millot et al. (2025).
#   D >= 4 km : Howard (2007), Table 1; simple craters below 7 km, complex
#               craters above.
#
# n is chosen so that the ejecta volume, integrated from R to infinity, equals
# the net volume excavated by the bowl. Ejecta are deposited only out to
# ejecta_extent_r radii, so part of the excavated volume is not deposited
# (see build_crater).
# =============================================================================

# Scaling-law parameters
WATTERS_HOWARD_SWITCH_M = 4000.0     # Watters below, Howard at and above [m]
HOWARD_SIMPLE_COMPLEX_M = 7000.0     # simple / complex transition [m]
WATTERS_M_SPLIT_M = 500.0            # boundary between Watters' m values [m]
WATTERS_M_SMALL = 1.65               # m for D < 500 m
WATTERS_M_LARGE = 1.88               # m for 500 m <= D < 4 km

# Numerical parameters (affect run time and memory, not results; above
# MAX_K_PIXELS, setting k = 1 is a negligible approximation)
N_SUB_MAX = 32                       # maximum sub-samples per pixel side
N_SUB_MIN = 4                        # minimum sub-samples per pixel side
SUBS_PER_RADIUS = 16.0               # target sub-samples per crater radius
MAX_K_PIXELS = 1.2e9                 # above this footprint size, k is set to 1
CHUNK_ELEMENTS = 4_000_000           # array size per chunk when evaluating profiles
PREBUILT_MAX_BYTES = 1.0e9           # larger stencils are evaluated window by window

_CRATER_CACHE = {}                   # crater models built in this process


def bowl_coefficients(d):
    """
    Morphology parameters of a crater of diameter d [m].

    Returns
    -------
    h1 : float
        Rim-to-floor depth [m].
    h2 : float
        Rim height above the far field [m].
    m : float
        Bowl-shape exponent.
    n : float
        Ejecta exponent (see ejecta_exponent).
    source : str
        Morphology model used.
    """
    d = float(d)
    if d < WATTERS_HOWARD_SWITCH_M - 1e-6:
        h1 = 0.205 * d ** 1.012
        h2 = 0.0354 * d ** 1.017
        m = WATTERS_M_SMALL if d < WATTERS_M_SPLIT_M else WATTERS_M_LARGE
        source = 'Watters 2015'
    elif d < HOWARD_SIMPLE_COMPLEX_M:
        h1 = 2.54 * d ** 0.67
        h2 = 1.93 * d ** 0.52
        m = 0.73 * d ** 0.11
        source = 'Howard 2007 simple'
    else:
        h1 = 12.20 * d ** 0.49
        h2 = 0.79 * d ** 0.60
        m = 0.64 * d ** 0.13
        source = 'Howard 2007 complex'

    return h1, h2, m, ejecta_exponent(h1, h2, m), source


def ejecta_exponent(h1, h2, m):
    """
    Ejecta exponent n for which ejecta volume equals excavated volume.

    The net volume of the bowl (r <= R) is pi R^2 [h1 m / (m + 2) - h2], and
    the volume of the ejecta from R to infinity is 2 pi R^2 h2 / (n - 2).
    Equating the two gives

        n = 2 + 2 h2 / [h1 m / (m + 2) - h2].
    """
    deficit = h1 * m / (m + 2.0) - h2
    if deficit <= 0:
        raise ValueError(f"No net excavation (h1={h1:.3g}, h2={h2:.3g}, m={m:.3g}).")
    target = deficit / (2.0 * h2)
    return 2.0 + 1.0 / target


def _profile_parts(r_m, d, h1, h2, m, n, r_max_m):
    """Bowl and ejecta components of the profile at distances r_m [m]."""
    R = d / 2.0
    x = r_m / R
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        bowl = np.where(x <= 1.0, (h2 - h1) + h1 * np.power(x, m), 0.0)
        ej = np.where((x > 1.0) & (r_m <= r_max_m), h2 * np.power(x, -n), 0.0)
    return bowl, ej


def _pixel_average_parts(dy, dx, res_m, c):
    """
    Pixel-averaged bowl and ejecta components at integer pixel offsets.

    Pixels near the rim, where the profile changes rapidly, are averaged over
    c['n_sub']**2 sub-samples. Farther pixels use the value at the pixel center.
    """
    dy = np.asarray(dy, dtype=float)
    dx = np.asarray(dx, dtype=float)
    r_px = np.hypot(dy, dx)
    bowl, ej = _profile_parts(r_px * res_m, c['d_m'], c['h1'], c['h2'], c['m'],
                              c['n'], c['r_max_m'])

    near = np.nonzero(r_px.ravel() < c['r_near_px'])[0]
    if near.size:
        ns = c['n_sub']
        sub = (np.arange(ns) + 0.5) / ns - 0.5
        sy, sx = np.meshgrid(sub, sub, indexing='ij')
        sy, sx = sy.ravel(), sx.ravel()
        dyf, dxf = dy.ravel(), dx.ravel()
        bowl_f, ej_f = bowl.ravel(), ej.ravel()
        step = max(1, CHUNK_ELEMENTS // sy.size)
        for s in range(0, near.size, step):
            idx = near[s:s + step]
            yy = dyf[idx, None] + sy[None, :]
            xx = dxf[idx, None] + sx[None, :]
            b, e = _profile_parts(np.hypot(yy, xx) * res_m, c['d_m'], c['h1'],
                                  c['h2'], c['m'], c['n'], c['r_max_m'])
            bowl_f[idx] = b.mean(axis=1)
            ej_f[idx] = e.mean(axis=1)
    return bowl, ej


def build_crater(d_m, res_m, domain_shape, ejecta_extent_r):
    """
    Crater model for one diameter on a raster of pixel size res_m.

    Every crater in a bin uses the bin's minimum diameter, so one model is
    built per bin and cached.

    Pixel values are averages of the profile over each pixel. Because n is
    derived from the continuous profile, the pixelated crater does not conserve
    volume exactly. The ejecta pixels are therefore multiplied by

        k = f * (bowl volume on the grid) / (ejecta volume on the grid),

    where f = 1 - ejecta_extent_r**(2 - n) is the fraction of the excavated
    volume that lies within the ejecta extent. The grid volumes are computed
    over the whole footprint, so ejecta that fall beyond the raster edge are
    lost rather than redistributed. For footprints larger than MAX_K_PIXELS,
    k is set to 1.

    Small craters are stored as a prebuilt stencil. Craters whose stencil
    would exceed the raster or PREBUILT_MAX_BYTES are evaluated only on the
    part of the footprint that overlaps the raster (see boom_clipped).

    Parameters
    ----------
    d_m : float
        Crater diameter [m].
    res_m : float
        Pixel size [m].
    domain_shape : tuple
        Shape of the depth map.
    ejecta_extent_r : float
        Outer edge of the ejecta deposit, in crater radii.

    Returns
    -------
    dict
        Morphology parameters, volumes, k, and either a stencil ('prebuilt'
        mode) or None ('clipped' mode).
    """
    key = (round(float(d_m), 6), float(res_m), tuple(domain_shape), float(ejecta_extent_r))
    if key in _CRATER_CACHE:
        return _CRATER_CACHE[key]

    h1, h2, m, n, source = bowl_coefficients(d_m)
    R_px = d_m / 2.0 / res_m
    W = footprint_halfwidth(d_m, res_m, ejecta_extent_r)
    side = 2 * W + 1

    c = dict(d_m=float(d_m), h1=h1, h2=h2, m=m, n=n, source=source,
             R_px=R_px, W=W, r_max_m=ejecta_extent_r * d_m / 2.0,
             r_near_px=max(1.5 * R_px, 6.0) + 1.0,
             n_sub=int(np.clip(np.ceil(SUBS_PER_RADIUS / R_px), N_SUB_MIN, N_SUB_MAX)))

    area = np.pi * (d_m / 2.0) ** 2
    c['V_bowl_analytic'] = area * (h1 * m / (m + 2.0) - h2)
    c['V_ejecta_analytic'] = (2.0 * area * h2 *
                              (1.0 - ejecta_extent_r ** (2.0 - n)) / (n - 2.0))
    c['retained_fraction'] = c['V_ejecta_analytic'] / c['V_bowl_analytic']

    clipped = (side > max(domain_shape)) or (float(side) ** 2 * 9 > PREBUILT_MAX_BYTES)
    c['mode'] = 'clipped' if clipped else 'prebuilt'
    px_area = res_m ** 2

    if not clipped:
        off = np.arange(-W, W + 1)
        dy, dx = np.meshgrid(off, off, indexing='ij')
        bowl, ej = _pixel_average_parts(dy, dx, res_m, c)
        vb, ve = -bowl.sum() * px_area, ej.sum() * px_area
        k = c['retained_fraction'] * vb / ve
        c['stencil'] = bowl + k * ej
        c['stencil_pos'] = c['stencil'] > 0
    else:
        c['stencil'] = None
        if float(side) ** 2 <= MAX_K_PIXELS:
            vb = ve = 0.0
            off = np.arange(-W, W + 1)
            rows_per = max(1, CHUNK_ELEMENTS // side)
            for r0 in range(0, side, rows_per):
                dy, dx = np.meshgrid(off[r0:r0 + rows_per], off, indexing='ij')
                bowl, ej = _pixel_average_parts(dy, dx, res_m, c)
                vb -= bowl.sum() * px_area
                ve += ej.sum() * px_area
            k = c['retained_fraction'] * vb / ve
        else:
            vb, ve, k = np.nan, np.nan, 1.0

    c['V_bowl_grid'], c['V_ejecta_grid'], c['k'] = vb, ve, k
    _CRATER_CACHE[key] = c
    return c


def crater_table(sizes, bins, res_m, domain_shape, ejecta_extent_r):
    """
    Morphology parameters of every simulated bin.

    Columns: bin, D_m, source, h1_m, h2_m, m, n, retained_fraction (share of
    the excavated volume deposited within the ejecta extent), k (grid volume
    correction; see build_crater) and mode ('prebuilt' or 'clipped').
    """
    rows = []
    for b in bins:
        c = build_crater(sizes[b] * 1000.0, res_m, domain_shape, ejecta_extent_r)
        rows.append({'bin': int(b), 'D_m': c['d_m'], 'source': c['source'],
                     'h1_m': c['h1'], 'h2_m': c['h2'], 'm': c['m'], 'n': c['n'],
                     'retained_fraction': c['retained_fraction'], 'k': c['k'],
                     'mode': c['mode']})
    return pd.DataFrame(rows)


# =============================================================================
# Crater emplacement
# =============================================================================

def _apply_rules(view, s_win, s_pos=None):
    """
    Superpose a crater profile on a window of the working depth map, in place.

    view  : valley depths (negative), NaN where already destroyed
    s_win : crater profile (negative bowl, positive rim and ejecta)

    1. view + s_win > 0                  -> destroyed (buried by rim or ejecta)
    2. s_win <= 0 and s_win < view       -> destroyed (bowl deeper than valley)
    3. s_win > 0 and view + s_win < 0    -> partially filled: view += s_win

    A bowl shallower than the valley meets none of the rules and leaves the
    pixel unchanged. NaN pixels fail every comparison and stay destroyed.
    """
    if s_pos is None:
        s_pos = s_win > 0
    summed = view + s_win
    fill = s_pos & (summed < 0)
    destroy = (summed > 0) | (~s_pos & (s_win < view))
    view[fill] = summed[fill]
    view[destroy] = np.nan


def boom(d_map, stencil, W, coordinates, stencil_pos=None):
    """
    Emplace craters with a prebuilt stencil.

    Parameters
    ----------
    d_map : numpy.ndarray
        Working depth map (valleys negative); modified in place.
    stencil : numpy.ndarray
        (2W + 1, 2W + 1) crater profile [m].
    W : int
        Stencil half-width [px].
    coordinates : array-like
        (n, 2) crater centers (row, col); centers may lie outside the raster.
    stencil_pos : numpy.ndarray or None
        Precomputed stencil > 0.

    Returns
    -------
    numpy.ndarray
        d_map after emplacement.
    """
    nrows, ncols = d_map.shape
    if stencil_pos is None:
        stencil_pos = stencil > 0

    for i, j in coordinates:
        r0, r1 = i - W, i + W + 1
        c0, c1 = j - W, j + W + 1
        ar0, ar1 = max(0, r0), min(nrows, r1)
        ac0, ac1 = max(0, c0), min(ncols, c1)
        if ar0 >= ar1 or ac0 >= ac1:
            continue
        sr0, sc0 = ar0 - r0, ac0 - c0
        sl = (slice(sr0, sr0 + (ar1 - ar0)), slice(sc0, sc0 + (ac1 - ac0)))
        _apply_rules(d_map[ar0:ar1, ac0:ac1], stencil[sl], stencil_pos[sl])

    return d_map


def boom_clipped(d_map, crater, coordinates, res_m):
    """
    Emplace large craters by evaluating the profile only where it overlaps the
    raster, in row chunks. Uses the same rules and k as boom.
    """
    nrows, ncols = d_map.shape
    W, k = crater['W'], crater['k']

    for i, j in coordinates:
        ar0, ar1 = max(0, i - W), min(nrows, i + W + 1)
        ac0, ac1 = max(0, j - W), min(ncols, j + W + 1)
        if ar0 >= ar1 or ac0 >= ac1:
            continue
        dx_row = np.arange(ac0, ac1) - j
        rows_per = max(1, CHUNK_ELEMENTS // dx_row.size)
        for r in range(ar0, ar1, rows_per):
            r_end = min(ar1, r + rows_per)
            dy, dx = np.meshgrid(np.arange(r, r_end) - i, dx_row, indexing='ij')
            bowl, ej = _pixel_average_parts(dy, dx, res_m, crater)
            _apply_rules(d_map[r:r_end, ac0:ac1], bowl + k * ej)

    return d_map


def impact(depth_map, rates, sizes, extent, res_m, seed, t_start=3.5, t_end=0.0,
           d_min_km=None, d_max_km=None, ejecta_extent_r=3.0, pad_extent_r=10.0,
           verbose=False):
    """
    One simulation run: draw a crater population and emplace it.

    For each bin, the mean number of craters (mean_crater_counts) is scaled to
    the padded placement region (placement_region), the number of craters is
    drawn from a Poisson distribution with that mean (draw_crater_count), and
    the centers are drawn uniformly over the region (locgenerate). Each bin
    has its own random stream, seeded from (seed, bin), so a bin's craters do
    not depend on which other bins are simulated. Runs with the same
    pad_extent_r that differ only in d_min_km, d_max_km or ejecta_extent_r
    therefore share identical craters in their common bins.
    Craters are emplaced in order of increasing diameter.

    Parameters
    ----------
    depth_map : numpy.ndarray
        Pre-impact depth map, valleys negative (from prep_data).
    rates : pandas.DataFrame
        Output of calculate_crater_rates.
    sizes : dict
        Minimum diameter of each bin [km].
    extent : tuple
        Study area (row_start, row_end, col_start, col_end).
    res_m : float
        Pixel size [m].
    seed : int
        Seed of this run.
    t_start, t_end : float
        Cratering window [Ga].
    d_min_km : float or None
        Smallest crater simulated [km]. None uses the smallest bin at least
        one pixel wide.
    d_max_km : float or None
        Largest crater simulated [km]. None simulates every bin.
    ejecta_extent_r : float
        Ejecta truncation radius in crater radii.
    pad_extent_r : float
        Placement padding in crater radii (at least ejecta_extent_r).
    verbose : bool
        Print the number of craters per bin.

    Returns
    -------
    post : numpy.ndarray
        Post-impact depth map, positive depths, NaN where destroyed.
    counts : pandas.DataFrame
        Number of craters placed per bin.
    events : list
        (bin, coordinates) for every bin with at least one crater.
    """
    if d_min_km is None:
        d_min_km = smallest_resolved_diameter(sizes, res_m)
    pad_extent_r = max(pad_extent_r, ejecta_extent_r)
    bins = simulated_bins(sizes, d_min_km, d_max_km)
    means = mean_crater_counts(rates, bins, t_start, t_end)

    counts, events = [], []
    for b in bins:
        region, area_ratio = placement_region(extent, sizes[b], res_m, pad_extent_r)
        mean = means[b] * area_ratio
        rng = np.random.default_rng([seed, int(b) + 1000])
        num = draw_crater_count(mean, rng)
        if num > 0:
            events.append((b, locgenerate(region, num, rng)))
            counts.append({'diameter_km': sizes[b], 'number': num, 'mean': mean,
                           'placement_area_km2': (region[1] - region[0]) *
                           (region[3] - region[2]) * res_m ** 2 / 1e6})

    d_map = np.copy(depth_map).astype(float)
    for b, coords in events:
        if verbose:
            print(f"  D = {sizes[b]:.5f} km : {len(coords)} craters", flush=True)
        c = build_crater(sizes[b] * 1000.0, res_m, depth_map.shape, ejecta_extent_r)
        if c['mode'] == 'prebuilt':
            boom(d_map, c['stencil'], c['W'], coords, c['stencil_pos'])
        else:
            boom_clipped(d_map, c, coords, res_m)

    return d_map * -1.0, pd.DataFrame(counts), events


# =============================================================================
# Transition tables
# =============================================================================

def transition_table(pre, post):
    """
    Counts of pixels moving from each pre-impact depth to each post-impact depth.

    Depths are binned in 1 m bins centered on whole meters. Each row of the
    output is one non-empty (pre, post) cell, with the total number of pixels
    at that pre-impact depth repeated in 'Original Pixel Count'.

    A destroyed pixel (NaN) is assigned a post-impact depth of 0: it is no
    longer valley. Depth 0 is excluded from the reconstruction fit, so these
    pixels carry no valley volume. Each pixel therefore appears in exactly one
    row, and the Pixel Counts of a pre-impact depth sum to its Original Pixel
    Count.

    Parameters
    ----------
    pre : numpy.ndarray
        Pre-impact depths, positive.
    post : numpy.ndarray
        Post-impact depths, positive, NaN where destroyed.

    Returns
    -------
    pandas.DataFrame
        Columns 'Pre-Impact Depth', 'Original Pixel Count',
        'Post-Impact Depth', 'Pixel Count'.
    """
    pre = np.asarray(pre, dtype=float)
    post = np.asarray(post, dtype=float)

    valid = ~np.isnan(pre) & (pre > 0)
    pre_flat = pre[valid]
    post_flat = post[valid]
    post_flat = np.where(np.isfinite(post_flat) & (post_flat > 0), post_flat, 0.0)

    max_center = int(np.ceil(pre_flat.max()))
    bins = np.arange(-0.5, max_center + 1.5, 1.0)
    centers = (bins[:-1] + bins[1:]) / 2

    H, _, _ = np.histogram2d(pre_flat, post_flat, bins=[bins, bins])
    total_pre, _ = np.histogram(pre_flat, bins=bins)

    x_idx, y_idx = np.nonzero(H)
    return pd.DataFrame({'Pre-Impact Depth': centers[x_idx],
                         'Original Pixel Count': total_pre[x_idx],
                         'Post-Impact Depth': centers[y_idx],
                         'Pixel Count': H[x_idx, y_idx]})


def table_volumes(table, pixel_area_m2):
    """
    Pre- and post-impact valley volume [m^3] from a transition table.

    Volumes are the sum of bin depth x pixel count x pixel area.
    """
    pre = table.drop_duplicates('Pre-Impact Depth')
    v_pre = (pre['Pre-Impact Depth'] * pre['Original Pixel Count']).sum() * pixel_area_m2
    v_post = (table['Post-Impact Depth'] * table['Pixel Count']).sum() * pixel_area_m2
    return float(v_pre), float(v_post)


# =============================================================================
# Aggregation to the MOLA grid
# =============================================================================
#
# The Mars valley depth map is equidistant cylindrical with 463.08 m cells:
# 463.08 m north-south and 463.08 m x cos(latitude) east-west. Each footprint
# below has the north-south size of a MOLA cell and the east-west size that
# gives the depth-weighted mean cell area of the valley pixels:
#
#   'global' : all valley networks; 193,325 m^2 (463.08 m x 417.47 m)
#   'goudge' : valleys within +/-30 deg latitude on Hesperian or older
#              terrain (Goudge et al. 2021); 207,172 m^2 (463.08 m x 447.37 m)
# =============================================================================

MOLA_NS_M = 463.083552699397501
FOOTPRINTS = {
    'global': (MOLA_NS_M, 193325.0 / MOLA_NS_M),
    'goudge': (MOLA_NS_M, 207172.0 / MOLA_NS_M),
}


def overlap_weights(n_fine, fine_res, coarse_res):
    """
    Fractional overlap of fine cells with coarse cells along one axis.

    Returns an (n_coarse, n_fine) matrix whose rows sum to 1. Only coarse
    cells that lie entirely within the fine raster are kept.
    """
    n_coarse = int(np.floor(n_fine * fine_res / coarse_res + 1e-9))
    edges = np.arange(n_fine + 1) * fine_res
    W = np.zeros((n_coarse, n_fine))
    for k in range(n_coarse):
        a, b = k * coarse_res, (k + 1) * coarse_res
        lo = np.clip(edges[:-1], a, b)
        hi = np.clip(edges[1:], a, b)
        W[k] = (hi - lo) / coarse_res
    return W


def aggregate(fine, Wy, Wx):
    """Area-weighted mean of a 2-D array on the coarse grid defined by Wy, Wx."""
    return Wy @ fine @ Wx.T


def aggregation_weights(shape, res_m, footprint):
    """
    Row and column weights for resampling a depth map to a MOLA footprint.

    Rows of the depth map run north-south, so the north-south cell size sets
    the row weights and the east-west size the column weights.

    Parameters
    ----------
    shape : tuple
        Shape of the study area.
    res_m : float
        Pixel size of the depth map [m].
    footprint : str or float
        A key of FOOTPRINTS, or the side of a square cell [m].
    """
    ns, ew = FOOTPRINTS[footprint] if isinstance(footprint, str) else (footprint, footprint)
    return overlap_weights(shape[0], res_m, ns), overlap_weights(shape[1], res_m, ew)


# =============================================================================
# Running simulations
# =============================================================================

def run_trial(seed, setup, settings, output_dir):
    """
    Run one simulation and write its outputs.

    Written to output_dir:
      crater_counts_<seed>.csv          craters placed per bin
      transition_matrix_<seed>.csv      transition table at the depth-map resolution
      <footprint>/transition_matrix_<seed>.csv
                                        transition table at each MOLA footprint
      post_impact_<seed>.tif            post-impact depth map (if save_tif)
      crater_coordinates_<seed>.csv     centers of craters of at least
                                        coordinates_min_km (if save_coordinates)

    Destroyed pixels count as depth 0 in every transition table, so the
    aggregation to the MOLA grid conserves valley volume.

    Parameters
    ----------
    seed : int
        Seed of this run.
    setup : dict
        Output of prep_data, as keys depth_map, rates, sizes, res_m, extent,
        plus depth_map_file and agg (cached aggregation weights).
    settings : dict
        Simulation settings (see run_trials).
    output_dir : str
        Folder for the outputs.

    Returns
    -------
    dict
        Seed, number of craters, and pre- and post-impact valley volume and
        percent decrease at the depth-map resolution.
    """
    post, counts, events = impact(
        setup['depth_map'], setup['rates'], setup['sizes'], setup['extent'],
        setup['res_m'], seed, t_start=settings['t_start'], t_end=settings['t_end'],
        d_min_km=settings['d_min_km'], d_max_km=settings['d_max_km'],
        ejecta_extent_r=settings['ejecta_extent_r'],
        pad_extent_r=settings['pad_extent_r'], verbose=settings['verbose'])

    r0, r1, c0, c1 = setup['extent']
    pre = -1.0 * setup['depth_map'][r0:r1, c0:c1]
    post_ext = post[r0:r1, c0:c1]

    counts.to_csv(os.path.join(output_dir, f"crater_counts_{seed}.csv"), index=False)
    table = transition_table(pre, post_ext)
    table.to_csv(os.path.join(output_dir, f"transition_matrix_{seed}.csv"), index=False)

    post_filled = np.where(np.isfinite(post_ext) & (post_ext > 0), post_ext, 0.0)
    for name, (Wy, Wx, pre_c) in setup['agg'].items():
        transition_table(pre_c, aggregate(post_filled, Wy, Wx)).to_csv(
            os.path.join(output_dir, name, f"transition_matrix_{seed}.csv"), index=False)

    if settings['save_tif']:
        array_to_tif(setup['depth_map_file'], post,
                     os.path.join(output_dir, f"post_impact_{seed}.tif"))

    if settings['save_coordinates']:
        parts = [pd.DataFrame({'row': c[:, 0], 'col': c[:, 1],
                               'diameter_km': setup['sizes'][b]})
                 for b, c in events if setup['sizes'][b] >= settings['coordinates_min_km']]
        coords = (pd.concat(parts, ignore_index=True) if parts
                  else pd.DataFrame(columns=['row', 'col', 'diameter_km']))
        coords.to_csv(os.path.join(output_dir, f"crater_coordinates_{seed}.csv"), index=False)

    v_pre, v_post = table_volumes(table, setup['res_m'] ** 2)
    return {'seed': seed, 'n_craters': int(counts['number'].sum()) if len(counts) else 0,
            'pre_volume_m3': v_pre, 'post_volume_m3': v_post,
            'percent_decrease': 100.0 * (v_pre - v_post) / v_pre}


def load_setup(depth_map_file, t_start, extent=None, footprints=()):
    """
    Read the depth map once and precompute everything shared by all runs.

    Returns a dict with the outputs of prep_data, the depth map path, and the
    aggregation weights and aggregated pre-impact map for each footprint.
    """
    depth_map, rates, sizes, res_m, extent, area = prep_data(depth_map_file, t_start, extent)
    r0, r1, c0, c1 = extent
    pre = np.clip(-depth_map[r0:r1, c0:c1].astype(np.float64), 0, None)
    agg = {}
    for name in footprints:
        Wy, Wx = aggregation_weights(pre.shape, res_m, name)
        agg[name] = (Wy, Wx, aggregate(pre, Wy, Wx))
    return dict(depth_map=depth_map, rates=rates, sizes=sizes, res_m=res_m,
                extent=extent, sample_area_km2=area, depth_map_file=depth_map_file,
                agg=agg)


_WORKER = {}


def _init_worker(depth_map_file, t_start, extent, footprints, settings, output_dir):
    """Load the depth map and settings once in each worker process."""
    _WORKER['setup'] = load_setup(depth_map_file, t_start, extent, footprints)
    _WORKER['settings'] = settings
    _WORKER['output_dir'] = output_dir


def _worker_trial(seed):
    """Run one simulation in a worker process."""
    return run_trial(seed, _WORKER['setup'], _WORKER['settings'], _WORKER['output_dir'])


def run_trials(depth_map_file, seeds, output_dir, t_start=3.5, t_end=0.0,
               d_min_km=None, d_max_km=None, ejecta_extent_r=3.0, pad_extent_r=10.0,
               footprints=('global',), extent=None, save_tif=False,
               save_coordinates=False, coordinates_min_km=0.1, n_processes=1,
               verbose=False):
    """
    Run the simulation for a list of seeds.

    Parameters
    ----------
    depth_map_file : str
        Pre-impact depth map (positive depths).
    seeds : iterable of int
        One simulation run per seed.
    output_dir : str
        Folder for the outputs.
    t_start, t_end : float
        Cratering window [Ga].
    d_min_km, d_max_km : float or None
        Smallest and largest crater simulated [km] (see impact).
    ejecta_extent_r : float
        Ejecta truncation radius in crater radii.
    pad_extent_r : float
        Placement padding in crater radii.
    footprints : iterable
        MOLA footprints (keys of FOOTPRINTS) at which to write transition tables.
    extent : tuple or None
        Study area in pixels; None uses the full raster.
    save_tif : bool
        Write each post-impact depth map.
    save_coordinates : bool
        Write the centers of craters of at least coordinates_min_km.
    coordinates_min_km : float
        Smallest crater whose center is written [km].
    n_processes : int
        Number of worker processes; 1 runs the seeds one after another.
    verbose : bool
        Print the number of craters per bin.

    Returns
    -------
    pandas.DataFrame
        One row per seed (see run_trial); also written to run_summary.csv.
    """
    seeds = [int(s) for s in seeds]
    footprints = tuple(footprints)
    os.makedirs(output_dir, exist_ok=True)
    for name in footprints:
        os.makedirs(os.path.join(output_dir, name), exist_ok=True)

    setup = load_setup(depth_map_file, t_start, extent, footprints)
    if d_min_km is None:
        d_min_km = smallest_resolved_diameter(setup['sizes'], setup['res_m'])
    settings = dict(t_start=t_start, t_end=t_end, d_min_km=d_min_km, d_max_km=d_max_km,
                    ejecta_extent_r=ejecta_extent_r,
                    pad_extent_r=max(pad_extent_r, ejecta_extent_r),
                    save_tif=save_tif, save_coordinates=save_coordinates,
                    coordinates_min_km=coordinates_min_km, verbose=verbose)

    bins = simulated_bins(setup['sizes'], d_min_km, d_max_km)
    crater_table(setup['sizes'], bins, setup['res_m'], setup['depth_map'].shape,
                 ejecta_extent_r).to_csv(os.path.join(output_dir, "crater_models.csv"),
                                         index=False)
    means = mean_crater_counts(setup['rates'], bins, t_start, t_end)
    mean_rows = []
    for b in bins:
        region, area_ratio = placement_region(setup['extent'], setup['sizes'][b],
                                              setup['res_m'], settings['pad_extent_r'])
        mean_rows.append({'bin': int(b), 'D_km': setup['sizes'][b],
                          'mean_in_study_area': means[b],
                          'mean_in_placement_region': means[b] * area_ratio,
                          'placement_area_km2': (region[1] - region[0]) *
                                                (region[3] - region[2]) *
                                                setup['res_m'] ** 2 / 1e6})
    pd.DataFrame(mean_rows).to_csv(os.path.join(output_dir, "mean_crater_counts.csv"),
                                   index=False)

    info = dict(settings, depth_map_file=os.path.abspath(depth_map_file),
                res_m=setup['res_m'], extent=list(setup['extent']),
                sample_area_km2=setup['sample_area_km2'], seeds=seeds,
                footprints={k: FOOTPRINTS[k] for k in footprints})
    with open(os.path.join(output_dir, "run_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    print(f"Depth map: {depth_map_file} ({setup['res_m']:g} m, "
          f"{setup['sample_area_km2']:.1f} km^2)")
    print(f"Craters {1000 * d_min_km:.1f} m to "
          f"{'all sizes' if d_max_km is None else f'{d_max_km:g} km'}, "
          f"ejecta to {ejecta_extent_r:g}R, padding {settings['pad_extent_r']:g}R; "
          f"{len(seeds)} run(s)", flush=True)

    if n_processes > 1:
        from multiprocessing import Pool
        init = (depth_map_file, t_start, extent, footprints, settings, output_dir)
        with Pool(n_processes, initializer=_init_worker, initargs=init) as pool:
            rows = []
            for r in pool.imap(_worker_trial, seeds):
                rows.append(r)
                print(f"  seed {r['seed']}: {r['percent_decrease']:.2f}% decrease", flush=True)
    else:
        rows = []
        for s in seeds:
            r = run_trial(s, setup, settings, output_dir)
            rows.append(r)
            print(f"  seed {s}: {r['percent_decrease']:.2f}% decrease", flush=True)

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(output_dir, "run_summary.csv"), index=False)
    return summary