"""
Reconstruction of the pre-impact valley depth distribution on Mars.

A transition table T gives, for each pre-impact depth i, the fraction of
pixels that end at each post-impact depth j; 1 - sum_j T[i, j] is the
fraction destroyed. For a pre-impact distribution p (pixel counts per 1 m
depth bin), the post-impact distribution is p @ T. Given the measured
(post-impact) distribution on Mars, p is found by least squares: p is
described by a small number of anchor points, interpolated in log space, and
the anchor values are adjusted with L-BFGS-B until p @ T matches the
measurements.

Workflow
--------
1. load_observed_data reads the measured Mars distribution.
2. load_seed_tables reads the per-run transition tables from the cratering
   simulation.
3. pooled_transition_matrix sums the runs, smooths the rows over a depth
   window and normalizes them.
4. reconstruct fits the pre-impact distribution and computes volumes.
5. run_pooled_bootstrap repeats steps 3 and 4 on runs resampled with
   replacement.
"""

import os

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import minimize

MARS_AREA_M2 = 1.4437e14       # surface area of Mars, for global equivalent layer


# =============================================================================
# Inputs
# =============================================================================

def load_observed_data(observed_file, max_depth):
    """
    Read a measured depth distribution.

    Parameters
    ----------
    observed_file : str
        CSV with columns 'Depth' (whole meters) and 'Count' (pixels).
    max_depth : int
        Deepest depth bin used [m]; deeper pixels are discarded.

    Returns
    -------
    observed_counts : numpy.ndarray
        Pixel count for each depth 0 ... max_depth.
    min_fit_depth : int
        Shallowest depth in the file. Shallower bins are excluded from the fit.
    """
    obs = pd.read_csv(observed_file)

    total_px = obs['Count'].sum()
    total_vol = (obs['Depth'] * obs['Count']).sum()
    above = obs[obs['Depth'] > max_depth]
    pct_px = 100.0 * above['Count'].sum() / total_px if total_px else 0.0
    pct_vol = 100.0 * (above['Depth'] * above['Count']).sum() / total_vol if total_vol else 0.0
    print(f"Depth cutoff of {max_depth} m discards {pct_px:.4f}% of measured pixels "
          f"and {pct_vol:.4f}% of measured volume.")

    obs = obs[obs['Depth'] <= max_depth]
    depths = obs['Depth'].astype(int).values
    observed_counts = np.zeros(max_depth + 1)
    observed_counts[depths] = obs['Count'].values
    min_fit_depth = int(depths.min()) if depths.size else 0
    return observed_counts, min_fit_depth


def load_seed_tables(file_list, max_depth):
    """
    Read the transition tables of individual simulation runs.

    Each table is read once and stored in sparse form, so that any
    combination of runs (for example a bootstrap resample) can be pooled
    without reading the files again.

    Parameters
    ----------
    file_list : list of str
        Transition table CSVs written by the cratering simulation.
    max_depth : int
        Deepest depth bin used [m].

    Returns
    -------
    dict
        'seed'  : run index of each non-empty cell
        'cell'  : flattened (pre, post) index of each cell
        'count' : pixel count of each cell
        'orig'  : (n_runs, n_depths) original pixel count per pre-impact depth
        'n_depths', 'files'
    """
    n = max_depth + 1
    seed_idx, cells, counts = [], [], []
    orig = np.zeros((len(file_list), n))

    for s, path in enumerate(file_list):
        t = pd.read_csv(path)
        t = t[(t['Pre-Impact Depth'] <= max_depth) & (t['Post-Impact Depth'] <= max_depth)]
        starts = t['Pre-Impact Depth'].astype(int).values
        ends = t['Post-Impact Depth'].astype(int).values
        seed_idx.append(np.full(starts.size, s, dtype=np.int32))
        cells.append(starts.astype(np.int64) * n + ends)
        counts.append(t['Pixel Count'].values.astype(float))

        first = t.drop_duplicates('Pre-Impact Depth')
        np.add.at(orig[s], first['Pre-Impact Depth'].astype(int).values,
                  first['Original Pixel Count'].values)

    return {'seed': np.concatenate(seed_idx), 'cell': np.concatenate(cells),
            'count': np.concatenate(counts), 'orig': orig, 'n_depths': n,
            'files': list(file_list)}


# =============================================================================
# Transition matrix
# =============================================================================

def shift_row(row_s, s, d):
    """
    Transfer a transition row from pre-impact depth s to depth d.

    Surviving pixels (post-impact depth >= 1 m) are moved by d - s meters, so
    that they lose the same number of meters as at depth s. Mass that lands
    between bins is split linearly between the two nearest bins; mass moved
    below 1 m joins the 0 m bin. The 0 m bin and the destroyed fraction (the
    shortfall of the row sum below 1) are unchanged.
    """
    n = row_s.size
    out = np.zeros(n)
    out[0] = row_s[0]
    j = np.nonzero(row_s[1:])[0] + 1
    if j.size == 0:
        return out
    tgt = np.clip(j + (d - s), 0, n - 1)
    lo = np.floor(tgt).astype(int)
    frac = tgt - lo
    hi = np.minimum(lo + 1, n - 1)
    np.add.at(out, lo, row_s[j] * (1 - frac))
    np.add.at(out, hi, row_s[j] * frac)
    return out


def smooth_counts(raw_counts, original_counts, window_frac=0.2, window_min_m=1.0):
    """
    Build each row of the transition table from a window of nearby depths.

    The row for pre-impact depth d combines the rows of every depth p within
    d +/- h, where h = max(window_min_m, window_frac * d), with triangular
    weights w = 1 - |p - d| / (h + 1). Each source row is shifted by d - p
    meters before it is added, so a pixel that lost x meters from depth p
    counts as a pixel that lost x meters from depth d. Shifted outcomes below
    1 m join the 0 m bin. The original counts are combined with the same
    weights, so destroyed pixels are carried over in proportion.

    Parameters
    ----------
    raw_counts : numpy.ndarray
        (n, n) pooled pixel counts, pre-impact x post-impact.
    original_counts : numpy.ndarray
        (n,) pooled original pixel count per pre-impact depth.
    window_frac : float
        Half-width of the window as a fraction of depth; 0 returns the
        counts unchanged.
    window_min_m : float
        Smallest half-width [m].

    Returns
    -------
    raw_smoothed, original_smoothed : numpy.ndarray
    """
    if window_frac <= 0:
        return raw_counts, original_counts
    n = raw_counts.shape[0]
    raw_s = np.zeros_like(raw_counts)
    orig_s = np.zeros_like(original_counts)
    present = np.nonzero(original_counts > 0)[0]
    present = present[present > 0]
    for d in range(1, n):
        h = max(window_min_m, window_frac * d)
        lo = np.searchsorted(present, d - h, side='left')
        hi = np.searchsorted(present, d + h, side='right')
        for p in present[lo:hi]:
            w = 1.0 - abs(p - d) / (h + 1.0)
            if w <= 0:
                continue
            row = raw_counts[p]
            delta = d - p
            out = raw_s[d]
            out[0] += w * row[0]
            if delta >= 0:
                out[1 + delta:] += w * row[1:n - delta]
            else:
                k = -delta
                out[1:n - k] += w * row[1 + k:]
                out[0] += w * row[1:1 + k].sum()
            orig_s[d] += w * original_counts[p]
    raw_s[0], orig_s[0] = raw_counts[0], original_counts[0]
    return raw_s, orig_s


def normalize_transition_counts(raw_counts, original_counts, min_original=0.0):
    """
    Convert pooled counts to transition probabilities.

    Row i is raw_counts[i] / original_counts[i]. Depths with original counts
    at or below min_original take the row of the nearest adequately populated
    depth, transferred with shift_row.

    Returns
    -------
    transition_matrix : numpy.ndarray
        (n, n) transition probabilities.
    filled : list of int
        Depths whose rows were transferred from another depth.
    """
    transition_matrix = np.zeros_like(raw_counts)
    good = original_counts > max(min_original, 0.0)
    populated = np.nonzero(good)[0]
    if (populated > 0).any():
        populated = populated[populated > 0]
    filled = []

    for i in range(raw_counts.shape[0]):
        if good[i]:
            transition_matrix[i, :] = raw_counts[i, :] / original_counts[i]
        elif populated.size:
            src = populated[np.argmin(np.abs(populated - i))]
            row_s = raw_counts[src, :] / original_counts[src]
            transition_matrix[i, :] = row_s if (i == 0 or src == 0) else shift_row(row_s, src, i)
            filled.append(i)
        else:
            transition_matrix[i, i] = 1.0

    return transition_matrix, filled


def pooled_transition_matrix(tables, weights=None, min_pixels_per_seed=3,
                             window_frac=0.2, window_min_m=1.0):
    """
    Transition matrix from the pooled counts of many simulation runs.

    Parameters
    ----------
    tables : dict
        Output of load_seed_tables.
    weights : numpy.ndarray or None
        Number of times each run is included; None includes every run once.
        A bootstrap resample is a set of integer weights.
    min_pixels_per_seed : float
        Depths with fewer pixels than this per run, on average, take the row
        of the nearest adequately populated depth.
    window_frac, window_min_m : float
        Depth window for smooth_counts.

    Returns
    -------
    transition_matrix : numpy.ndarray
    filled : list of int
        Depths whose rows were transferred from another depth.
    """
    n = tables['n_depths']
    w = np.ones(tables['orig'].shape[0]) if weights is None else np.asarray(weights, float)
    raw = np.bincount(tables['cell'], weights=tables['count'] * w[tables['seed']],
                      minlength=n * n).reshape(n, n)
    orig = w @ tables['orig']
    raw, orig = smooth_counts(raw, orig, window_frac, window_min_m)
    return normalize_transition_counts(raw, orig, min_original=min_pixels_per_seed * w.sum())


# =============================================================================
# Inverse model
# =============================================================================

def build_anchors(num_anchors, anchor_exponent, max_depth):
    """
    Anchor depths, concentrated at shallow depths.

    Depths are max_depth * linspace(0, 1, num_anchors)**anchor_exponent,
    rounded to whole meters. Duplicates and depth 0 are removed and max_depth
    is always kept, so fewer anchors than requested may be returned (16
    requested with exponent 3 and max_depth 773 gives 14).

    Depth 0 is excluded because it is not part of a valley and is not
    constrained by the fit.
    """
    curve = np.linspace(0, 1, num_anchors) ** anchor_exponent
    anchors = sorted(set(np.round(curve * max_depth).astype(int)))
    if max_depth not in anchors:
        anchors.append(max_depth)
    return np.array(sorted(a for a in anchors if a > 0))


def interpolate_distribution_log(x_anchors, log_y_anchors, full_x, interp_kind):
    """Interpolate log10 counts between anchors and return linear counts."""
    f = interp1d(x_anchors, log_y_anchors, kind=interp_kind, bounds_error=False,
                 fill_value="extrapolate")
    return 10 ** f(full_x)


def calculate_error(log_y_anchors, x_anchors, depths, interp_kind, transition_matrix,
                    min_fit_depth, observed_counts):
    """
    Sum of squared differences between modeled and measured post-impact counts.

    The pre-impact distribution is interpolated from the anchors, passed
    through the transition matrix, and compared with the measurements at
    depths of at least min_fit_depth.
    """
    pre = interpolate_distribution_log(x_anchors, log_y_anchors, depths, interp_kind)
    post = pre @ transition_matrix
    mask = depths >= min_fit_depth
    return np.sum((post[mask] - observed_counts[mask]) ** 2)


def reconstruct(transition_matrix, observed_counts, min_fit_depth, config):
    """
    Fit the pre-impact depth distribution and compute valley volumes.

    The initial guess is the measured distribution at the anchor depths. The
    anchors are optimized in log10 space with L-BFGS-B. The lower bound is
    config['log_floor']; the upper bound lies config['log_headroom'] orders of
    magnitude above the largest measured count and only keeps the optimizer
    finite. Anchors that end at a bound are counted in 'n_pinned'.

    Parameters
    ----------
    transition_matrix : numpy.ndarray
        (n, n) transition probabilities.
    observed_counts : numpy.ndarray
        Measured pixel count per depth.
    min_fit_depth : int
        Shallowest depth included in the fit.
    config : dict
        num_anchors, anchor_exponent, interp_kind, tolerance, max_iter,
        log_floor, log_headroom, pixel_area_m2.

    Returns
    -------
    dict
        Depths, pre-impact ('original'), modeled post-impact ('present') and
        residual counts, anchors, optimizer status, and volumes [km^3].
    """
    max_depth = transition_matrix.shape[0] - 1
    depths = np.arange(0, max_depth + 1)
    mask = depths >= min_fit_depth

    row_sums = transition_matrix.sum(axis=1)
    if row_sums.max() > 1.0 + 1e-9 or row_sums.min() < -1e-12:
        raise ValueError(f"Transition matrix row sums out of range "
                         f"[{row_sums.min():.6f}, {row_sums.max():.6f}].")

    x_anchors = build_anchors(config['num_anchors'], config['anchor_exponent'], max_depth)

    guess = interp1d(depths, observed_counts, kind='linear', fill_value="extrapolate")(x_anchors)
    guess[guess < 1] = 1.0
    log_guess = np.log10(guess)

    log_ceiling = np.log10(max(observed_counts.max(), 1.0)) + config['log_headroom']
    bounds = [(config['log_floor'], log_ceiling)] * len(x_anchors)

    result = minimize(calculate_error, log_guess,
                      args=(x_anchors, depths, config['interp_kind'], transition_matrix,
                            min_fit_depth, observed_counts),
                      method='L-BFGS-B', bounds=bounds, tol=config['tolerance'],
                      options={'maxiter': config['max_iter']})

    best = result.x
    n_pinned = int(np.sum((np.abs(best - config['log_floor']) < 1e-6) |
                          (np.abs(best - log_ceiling) < 1e-6)))

    original = interpolate_distribution_log(x_anchors, best, depths, config['interp_kind'])
    original[~mask] = 0.0
    present = original @ transition_matrix
    residuals = present - observed_counts

    area = config['pixel_area_m2']
    present_km3 = np.sum(depths * observed_counts * area) / 1e9
    original_km3 = np.sum(depths * original * area) / 1e9

    return {
        'depths': depths,
        'original': original,
        'present': present,
        'residuals': residuals,
        'destroyed_fraction': 1.0 - row_sums,
        'x_anchors': x_anchors,
        'y_anchors': 10 ** best,
        'num_anchors': len(x_anchors),
        'n_pinned': n_pinned,
        'success': bool(result.success),
        'nit': int(result.nit),
        'volumes': {
            'original_km3': original_km3,
            'present_km3': present_km3,
            'lost_km3': original_km3 - present_km3,
            'percent_decrease': 100.0 * (original_km3 - present_km3) / original_km3,
            'abs_residual_counts': float(np.sum(np.abs(residuals[mask]))),
        },
    }


# =============================================================================
# Pooled bootstrap
# =============================================================================
#
# Each simulation run is one realization of the cratering history. The point
# estimate pools every run. Its uncertainty is estimated by resampling runs
# with replacement and repeating the reconstruction on each resample. Runs,
# not pixels, are resampled, because pixels within a run are not independent.
# =============================================================================

_BOOT = {}


def _init_bootstrap(state):
    _BOOT.update(state)


def _bootstrap_task(args):
    rep_id, weights = args
    g = _BOOT
    cfg = g['config']
    tm, _ = pooled_transition_matrix(g['tables'], weights,
                                     min_pixels_per_seed=cfg['min_pixels_per_seed'],
                                     window_frac=cfg['window_frac'],
                                     window_min_m=cfg['window_min_m'])
    res = reconstruct(tm, g['observed_counts'], g['min_fit_depth'], cfg)
    v = res['volumes']
    return {'rep': rep_id, 'n_seeds_used': int(weights.sum()),
            'original_km3': v['original_km3'], 'present_km3': v['present_km3'],
            'lost_km3': v['lost_km3'], 'percent_decrease': v['percent_decrease'],
            'success': res['success'], 'nit': res['nit'], 'n_pinned': res['n_pinned'],
            'original': res['original']}


def run_pooled_bootstrap(tables, observed_counts, min_fit_depth, config, n_boot=1000,
                         convergence_sizes=(), n_boot_convergence=200,
                         n_processes=1, random_seed=0):
    """
    Point estimate from all runs, with bootstrap and convergence replicates.

    Parameters
    ----------
    tables : dict
        Output of load_seed_tables.
    observed_counts : numpy.ndarray
        Measured pixel count per depth.
    min_fit_depth : int
        Shallowest depth included in the fit.
    config : dict
        Settings for pooled_transition_matrix and reconstruct.
    n_boot : int
        Resamples of all runs, with replacement.
    convergence_sizes : iterable of int
        Numbers of runs for the convergence test; sizes larger than the
        number of runs are skipped.
    n_boot_convergence : int
        Resamples per convergence size.
    n_processes : int
        Worker processes; 1 runs the replicates one after another.
    random_seed : int
        Seed for the resampling.

    Returns
    -------
    point : dict
        reconstruct output using every run once.
    boot : pandas.DataFrame
        One row per bootstrap replicate.
    boot_originals : numpy.ndarray
        (n_boot, n_depths) pre-impact distributions of the replicates.
    conv : pandas.DataFrame
        One row per convergence replicate.
    """
    n = tables['orig'].shape[0]
    rng = np.random.default_rng(random_seed)

    tm, _ = pooled_transition_matrix(tables, min_pixels_per_seed=config['min_pixels_per_seed'],
                                     window_frac=config['window_frac'],
                                     window_min_m=config['window_min_m'])
    point = reconstruct(tm, observed_counts, min_fit_depth, config)

    def weights_for(m):
        return np.bincount(rng.integers(0, n, size=m), minlength=n)

    tasks = [(b + 1, weights_for(n)) for b in range(n_boot)]
    conv_tasks = [(len(tasks) + k + 1, weights_for(m))
                  for m in convergence_sizes if m <= n
                  for k in range(n_boot_convergence)]
    all_tasks = tasks + conv_tasks

    state = dict(tables=tables, observed_counts=observed_counts,
                 min_fit_depth=min_fit_depth, config=config)
    out = []
    if n_processes > 1:
        from multiprocessing import Pool
        with Pool(n_processes, initializer=_init_bootstrap, initargs=(state,)) as pool:
            for k, r in enumerate(pool.imap(_bootstrap_task, all_tasks, chunksize=4), 1):
                out.append(r)
                if k % 50 == 0 or k == len(all_tasks):
                    print(f"  {k}/{len(all_tasks)} replicates done", flush=True)
    else:
        _init_bootstrap(state)
        for k, t in enumerate(all_tasks, 1):
            out.append(_bootstrap_task(t))
            if k % 50 == 0 or k == len(all_tasks):
                print(f"  {k}/{len(all_tasks)} replicates done", flush=True)

    boot_rows, conv_rows = out[:len(tasks)], out[len(tasks):]
    boot_originals = np.array([r.pop('original') for r in boot_rows])
    for r in conv_rows:
        r.pop('original')
    return point, pd.DataFrame(boot_rows), boot_originals, pd.DataFrame(conv_rows)


def convergence_summary(boot, conv, n_seeds, point_lost_km3):
    """
    Width of the 95% interval of lost volume against the number of runs.

    Returns one row per number of runs with the 2.5th and 97.5th percentiles
    of lost volume and the interval half-width as a percentage of the point
    estimate.
    """
    groups = list(conv.groupby('n_seeds_used')) if len(conv) else []
    rows = []
    for m, g in groups + [(n_seeds, boot)]:
        lo, hi = np.percentile(g['lost_km3'], [2.5, 97.5])
        pd_lo, pd_hi = np.percentile(g['percent_decrease'], [2.5, 97.5])
        rows.append({'n_seeds_used': int(m), 'lost_km3_ci_low': lo, 'lost_km3_ci_high': hi,
                     'percent_decrease_median': float(np.median(g['percent_decrease'])),
                     'percent_decrease_ci_low': pd_lo, 'percent_decrease_ci_high': pd_hi,
                     'half_width_pct_of_estimate': 100 * (hi - lo) / 2 / abs(point_lost_km3)})
    return (pd.DataFrame(rows).drop_duplicates('n_seeds_used')
            .sort_values('n_seeds_used').reset_index(drop=True))


# =============================================================================
# Valley-carving water
# =============================================================================

def fluid_volume(eroded_volume_m3, porosity, flux_ratio):
    """
    Volume of water required to carve an eroded volume (Rosenberg et al. 2019).

    V_f = V_VN (1 - porosity) Q_f / Q_s, where V_VN is the eroded volume
    (rock and pore space) and Q_f / Q_s the fluid-to-sediment flux ratio.
    """
    return eroded_volume_m3 * (1.0 - porosity) * flux_ratio


def global_equivalent_layer(volume_m3):
    """Depth [m] of a volume spread evenly over the surface of Mars."""
    return volume_m3 / MARS_AREA_M2


# =============================================================================
# Plots
# =============================================================================

def plot_bootstrap_envelope(point, boot_originals, observed_counts, min_fit_depth, output_dir):
    """
    Pre-impact distribution with its 95% bootstrap interval, the modeled
    post-impact distribution, and the measured distribution.

    Writes reconstruction.png to output_dir.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    i0 = min_fit_depth
    depths = point['depths'][i0:]
    lo, hi = np.percentile(boot_originals, [2.5, 97.5], axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(depths, lo[i0:], hi[i0:], alpha=0.4, color='#E8743B',
                    label='95% bootstrap interval')
    ax.plot(depths, point['original'][i0:], color='#E8743B', linewidth=2.5,
            label='Pre-impact (reconstructed)')
    ax.plot(depths, point['present'][i0:], color='#E8743B', linestyle='--', linewidth=1.5,
            label='Post-impact (modeled)')
    ax.scatter(depths, observed_counts[i0:], color='black', s=4, label='Measured')
    ax.scatter(point['x_anchors'], [0] * len(point['x_anchors']), marker='x', color='black',
               s=40, zorder=6, clip_on=False, label='Anchor depths')
    ax.set_xlim(0, depths.max())
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Valley depth (m)")
    ax.set_ylabel("Pixel count")
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False)
    fig.savefig(os.path.join(output_dir, "reconstruction.png"), dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_residuals(point, min_fit_depth, output_dir):
    """Modeled minus measured post-impact counts. Writes residuals.png."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    i0 = min_fit_depth
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(point['depths'][i0:], point['residuals'][i0:], color='black', linewidth=1.5)
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel("Valley depth (m)")
    ax.set_ylabel("Modeled - measured count")
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    fig.savefig(os.path.join(output_dir, "residuals.png"), dpi=200, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
# Preparing Mars inputs
# =============================================================================
#
# These functions read a Mars valley depth raster in an equidistant
# cylindrical projection (e.g. the dataset of the Texas Data Repository,
# https://doi.org/10.18738/T8/STRFZH). A cell spans its nominal size
# north-south and the nominal size x cos(latitude) east-west.
# =============================================================================

def _valley_strips(depth_raster, max_depth=None, lat_range=None, mask=None, strip=256):
    """
    Yield (depths, cos(latitude)) of the valley pixels, one block of rows at a time.

    Depths may be stored as positive or negative values; the sign is taken from
    the majority of non-zero pixels. NoData, NaN and 0 are not valley pixels.
    mask is an optional polygon file (read with geopandas); only pixels whose
    centers fall inside a polygon are kept.
    """
    import rasterio
    from rasterio.windows import Window

    with rasterio.open(depth_raster) as src:
        t, nodata = src.transform, src.nodata
        try:
            from pyproj import CRS
            radius = CRS.from_wkt(src.crs.to_wkt()).ellipsoid.semi_major_metre
        except Exception:
            radius = 3396190.0

        shapes = None
        if mask is not None:
            import geopandas as gpd
            shapes = [g for g in gpd.read_file(mask).to_crs(src.crs.to_wkt()).geometry
                      if g is not None and not g.is_empty]

        def read(r0, h):
            d = src.read(1, window=Window(0, r0, src.width, h)).astype(np.float64)
            ok = np.isfinite(d) & (d != 0)
            if nodata is not None:
                ok &= d != nodata
            return d, ok

        n_pos = n_neg = 0
        for r0 in range(0, src.height, strip):
            d, ok = read(r0, min(strip, src.height - r0))
            n_pos += int((d[ok] > 0).sum())
            n_neg += int((d[ok] < 0).sum())
        sign = -1.0 if n_neg > n_pos else 1.0

        for r0 in range(0, src.height, strip):
            h = min(strip, src.height - r0)
            d, ok = read(r0, h)
            d = sign * d
            ok &= d > 0
            if max_depth is not None:
                ok &= d <= max_depth
            lat = np.degrees((t.f + t.e * (np.arange(r0, r0 + h) + 0.5)) / radius)
            if lat_range is not None:
                ok &= ((lat >= lat_range[0]) & (lat <= lat_range[1]))[:, None]
            if shapes is not None and ok.any():
                from rasterio.features import geometry_mask
                tt = t * t.translation(0, r0)
                ok &= geometry_mask(shapes, out_shape=(h, src.width), transform=tt, invert=True)
            if ok.any():
                cos_lat = np.broadcast_to(np.cos(np.radians(lat))[:, None], d.shape)
                yield d[ok], cos_lat[ok]


def depth_distribution_from_raster(depth_raster, output_csv, lat_range=None, mask=None):
    """
    Count valley pixels in 1 m depth bins and write a 'Depth', 'Count' CSV.

    Depths are rounded to the nearest meter. lat_range (south, north) in
    degrees and mask (polygon file) restrict the pixels used.
    """
    counts = np.zeros(1, dtype=np.int64)
    for d, _ in _valley_strips(depth_raster, lat_range=lat_range, mask=mask):
        b = np.floor(d + 0.5).astype(np.int64)
        if b.max() >= counts.size:
            counts = np.pad(counts, (0, b.max() + 1 - counts.size))
        counts += np.bincount(b, minlength=counts.size)
    df = pd.DataFrame({'Depth': np.arange(counts.size), 'Count': counts})
    df = df[df.Count > 0]
    df.to_csv(output_csv, index=False)
    return df


def depth_weighted_pixel_area(depth_raster, max_depth=None, lat_range=None, mask=None):
    """
    Depth-weighted mean cell area [m^2] of the valley pixels.

    With A = sum(d_k a_k) / sum(d_k), where d_k and a_k are the depth and area
    of valley pixel k, the volume sum_i A d_i f_i computed from a depth
    distribution equals the sum of the volumes of the individual pixels.
    """
    import rasterio
    with rasterio.open(depth_raster) as src:
        cell = abs(src.transform.a * src.transform.e)
    sum_d = sum_dcos = 0.0
    for d, c in _valley_strips(depth_raster, max_depth=max_depth, lat_range=lat_range, mask=mask):
        sum_d += d.sum()
        sum_dcos += (d * c).sum()
    if sum_d == 0:
        raise ValueError("No valley pixels within the given limits.")
    return cell * sum_dcos / sum_d
